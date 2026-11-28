"""Integration tests for the pooled-signal kNN — requires Postgres + pgvector.

Run via:
    uv run pytest -m integration

These need a real database: they insert `interactions` rows and assert on the
SQL aggregation result.
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import insert, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ingovo_search.db.models import Interaction, Property
from ingovo_search.db.session import asyncpg_connect_args
from ingovo_search.lib.env import get_settings, sqlalchemy_asyncpg_url
from ingovo_search.lib.vector_db import VECTOR_DIM, vector_bind_value
from ingovo_search.recommender.pooled_signal import POOLED_SIGNAL_SQL, pooled_signal

pytestmark = pytest.mark.integration

# Query embedding and three listing-side embeddings positioned relative to it.
QUERY_VEC: list[float] = [1.0, *([0.0] * (VECTOR_DIM - 1))]
SIMILAR_VEC: list[float] = [1.0, 0.05, *([0.0] * (VECTOR_DIM - 2))]  # cosine ~0.999
DISSIMILAR_VEC: list[float] = [0.0, 1.0, *([0.0] * (VECTOR_DIM - 2))]  # cosine 0.0


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Fresh DB session that rolls back after each test."""
    settings = get_settings()
    engine = create_async_engine(
        sqlalchemy_asyncpg_url(settings.database_url),
        echo=False,
        future=True,
        connect_args=asyncpg_connect_args(),
    )
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as sess:
        try:
            yield sess
        finally:
            await sess.rollback()
    await engine.dispose()


async def _insert_property(sess: AsyncSession) -> str:
    pid = str(uuid4())
    sess.add(Property(id=pid, title="Test property", city="Budapest"))
    await sess.flush()
    return pid


async def _insert_interaction(
    sess: AsyncSession,
    property_id: str,
    signal_type: str,
    query_embedding: Sequence[float],
    *,
    created_at: datetime | None = None,
) -> None:
    sess.add(
        Interaction(
            id=str(uuid4()),
            property_id=property_id,
            signal_type=signal_type,
            signal_weight=1.0,
            query_embedding=list(query_embedding),
            created_at=created_at or datetime.now(tz=UTC),
        )
    )
    await sess.flush()


async def test_no_matching_interactions_returns_zero(session: AsyncSession) -> None:
    pid = await _insert_property(session)
    assert await pooled_signal(session, pid, QUERY_VEC) == 0.0


async def test_five_thumb_up_similar_queries_returns_positive(session: AsyncSession) -> None:
    pid = await _insert_property(session)
    for _ in range(5):
        await _insert_interaction(session, pid, "thumb_up", SIMILAR_VEC)

    result = await pooled_signal(session, pid, QUERY_VEC)
    # All neighbours carry signal value 1.0, so the weighted average is 1.0.
    assert result > 0.0
    assert abs(result - 1.0) < 1e-6


async def test_five_thumb_down_returns_negative(session: AsyncSession) -> None:
    pid = await _insert_property(session)
    for _ in range(5):
        await _insert_interaction(session, pid, "thumb_down", SIMILAR_VEC)

    result = await pooled_signal(session, pid, QUERY_VEC)
    assert result < 0.0
    assert abs(result - (-1.0)) < 1e-6


async def test_interaction_older_than_window_is_excluded(session: AsyncSession) -> None:
    pid = await _insert_property(session)
    await _insert_interaction(
        session,
        pid,
        "thumb_up",
        SIMILAR_VEC,
        created_at=datetime.now(tz=UTC) - timedelta(days=400),
    )
    # The only interaction falls outside the 365-day window → no neighbours.
    assert await pooled_signal(session, pid, QUERY_VEC) == 0.0


async def test_dissimilar_query_interaction_is_excluded(session: AsyncSession) -> None:
    pid = await _insert_property(session)
    await _insert_interaction(session, pid, "thumb_up", DISSIMILAR_VEC)
    # Cosine similarity 0.0 < 0.85 threshold → excluded.
    assert await pooled_signal(session, pid, QUERY_VEC) == 0.0


async def test_pooled_signal_scale_10k_rows(session: AsyncSession) -> None:
    """10K interactions across many properties; the per-property query stays fast.

    The query filters `property_id = :pid` first, so the btree index
    `ix_interactions_property_id_created_at` is what keeps this sub-100ms — a
    sequential scan of the whole table is the failure mode this guards against.
    (The HNSW index on `query_embedding` serves ORDER BY <=> LIMIT ANN queries;
    this aggregate is a WHERE-filter, so btree on property_id is the hot path.)
    """
    target_pid = await _insert_property(session)
    other_pids = [await _insert_property(session) for _ in range(20)]

    for start_idx in range(0, 10_000, 1000):
        await session.execute(
            insert(Interaction),
            [
                {
                    "id": str(uuid4()),
                    "property_id": target_pid if i % 100 == 0 else other_pids[i % len(other_pids)],
                    "signal_type": "thumb_up" if i % 2 == 0 else "view_long",
                    "signal_weight": 1.0,
                    "query_embedding": SIMILAR_VEC if i % 100 == 0 else None,
                    "created_at": datetime.now(tz=UTC) - timedelta(days=10 if i % 3 != 0 else 500),
                }
                for i in range(start_idx, min(start_idx + 1000, 10_000))
            ],
        )
    await session.flush()

    start = time.perf_counter()
    result = await pooled_signal(session, target_pid, QUERY_VEC)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert isinstance(result, float)
    assert elapsed_ms < 1000.0, f"pooled_signal client roundtrip took {elapsed_ms:.1f}ms"

    plan_rows = await session.execute(
        text(f"EXPLAIN (ANALYZE, FORMAT TEXT) {POOLED_SIGNAL_SQL}"),
        {"pid": target_pid, "q": await vector_bind_value(session, QUERY_VEC)},
    )
    plan = "\n".join(str(row[0]) for row in plan_rows)
    assert "Seq Scan on interactions" not in plan, f"pooled-signal query seq-scanned interactions:\n{plan}"
    execution_ms = _explain_execution_time_ms(plan)
    assert execution_ms < 100.0, f"pooled_signal DB execution took {execution_ms:.1f}ms (budget 100ms)"


def _explain_execution_time_ms(plan: str) -> float:
    match = re.search(r"Execution Time: ([0-9.]+) ms", plan)
    assert match is not None, f"EXPLAIN output did not include execution time:\n{plan}"
    return float(match.group(1))
