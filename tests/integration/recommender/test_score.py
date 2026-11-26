"""Integration tests for §8 recommender scoring — DB-backed acceptance tests.

Requires Postgres + pgvector + postgis. Run via:
    uv run pytest -m integration

Covers §8 acceptance 1 (empty filter returns all searchable), 2 (price hard
filter excludes), and 8 (1K-candidate scale + EXPLAIN), plus an end-to-end
`recommend` smoke that the full pipeline returns a `ScoreBreakdown`.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import insert, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ingovo_search.db.models import Property
from ingovo_search.db.session import asyncpg_connect_args
from ingovo_search.lib.env import get_settings, sqlalchemy_asyncpg_url
from ingovo_search.lib.llm import MockLlmClient
from ingovo_search.lib.vector_db import VECTOR_DIM, insert_embedding, insert_embeddings_batch
from ingovo_search.recommender.parser_schema import HardFilters, ParsedQuery
from ingovo_search.recommender.recommend import recommend
from ingovo_search.recommender.score import ScoreBreakdown
from ingovo_search.recommender.settlements import nearby_settlements
from ingovo_search.recommender.sql_filter import build_hard_filter_query, run_hard_filter

pytestmark = pytest.mark.integration


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
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


def _query_vec() -> list[float]:
    return [1.0, *([0.0] * (VECTOR_DIM - 1))]


def _listing_vec() -> list[float]:
    return [1.0, *([0.0] * (VECTOR_DIM - 1))]


async def _insert_property(
    sess: AsyncSession,
    *,
    status: str = "active",
    price_huf: int | None = None,
    embed: bool = False,
    city: str = "Budapest",
) -> str:
    pid = str(uuid4())
    sess.add(
        Property(
            id=pid,
            title="Test property",
            city=city,
            status=status,
            price_huf=price_huf,
            created_at=datetime.now(tz=UTC),
        )
    )
    await sess.flush()
    if embed:
        await insert_embedding(sess, pid, "factual", _listing_vec())
        await insert_embedding(sess, pid, "descriptive", _listing_vec())
        await sess.flush()
    return pid


# --- Acceptance 1 ------------------------------------------------------------


async def test_empty_hard_filter_returns_all_searchable_listings(session: AsyncSession) -> None:
    searchable = [await _insert_property(session, status="active") for _ in range(3)]
    inactive = await _insert_property(session, status="draft")  # not eligible -> excluded

    ids = await run_hard_filter(session, HardFilters())
    assert set(searchable).issubset(set(ids))
    assert inactive not in ids


# --- Acceptance 2 ------------------------------------------------------------


async def test_price_huf_max_excludes_listings_above_threshold(session: AsyncSession) -> None:
    cheap = await _insert_property(session, price_huf=30_000_000)
    pricey = await _insert_property(session, price_huf=70_000_000)
    no_price = await _insert_property(session, price_huf=None)
    # Pin the SQL NULL explicitly so this assertion measures the numeric
    # hard-filter contract rather than any insert-default behaviour (the legacy
    # `price` column that a trigger once normalised to 0 is gone since 0061).
    await session.execute(text("UPDATE properties SET price_huf = NULL WHERE id = :id"), {"id": no_price})

    ids = await run_hard_filter(session, HardFilters(price_huf_max=50_000_000))
    assert cheap in ids
    assert pricey not in ids
    # Numeric filter excludes NULL prices per the §8 missing-data policy.
    assert no_price not in ids


# --- R3 verified nearby candidate pool --------------------------------------


def _budapest_nearby_city_scope() -> list[str]:
    return [item.settlement.name for item in nearby_settlements("ksh:13578", 30.0)]


async def test_nearby_city_inside_30km_can_enter_candidate_pool(session: AsyncSession) -> None:
    budaors = await _insert_property(session, city="Budaörs")
    ids = await run_hard_filter(
        session,
        HardFilters(city="Budapest", city_id="ksh:13578"),
        city_scope=_budapest_nearby_city_scope(),
    )
    assert budaors in ids


async def test_distant_city_outside_30km_cannot_enter_candidate_pool(session: AsyncSession) -> None:
    gyor = await _insert_property(session, city="Győr")
    ids = await run_hard_filter(
        session,
        HardFilters(city="Budapest", city_id="ksh:13578"),
        city_scope=_budapest_nearby_city_scope(),
    )
    assert gyor not in ids


async def test_unverified_city_cannot_enter_nearby_candidate_pool(session: AsyncSession) -> None:
    unknown = await _insert_property(session, city="Ismeretlen település")
    ids = await run_hard_filter(
        session,
        HardFilters(city="Budapest", city_id="ksh:13578"),
        city_scope=_budapest_nearby_city_scope(),
    )
    assert unknown not in ids


# --- End-to-end recommend ---------------------------------------------------


async def test_recommend_returns_score_breakdowns(session: AsyncSession) -> None:
    """`recommend` returns at least one (Property, ScoreBreakdown) for a simple query.

    Scoped to a sentinel city: the dev DB also holds the ~2K-listing eval
    corpus, and an unfiltered pool flips on the vector prefilter whose
    similarity threshold starves these mock-vector rows.
    """
    city = f"BreakdownTest-{uuid4().hex[:8]}"
    for _ in range(3):
        await _insert_property(session, embed=True, city=city)

    result = await recommend(
        session,
        ParsedQuery(hard_filters=HardFilters(city=city)),
        query_emb=_query_vec(),
        client=MockLlmClient({}),
        top_n=10,
    )
    assert result, "recommend should return at least one candidate"
    for prop, breakdown in result:
        assert isinstance(prop, Property)
        assert isinstance(breakdown, ScoreBreakdown)
        assert breakdown.total >= 0.0


# --- Acceptance 8 (scale) ---------------------------------------------------


async def test_recommend_1000_candidates_under_5_seconds(session: AsyncSession) -> None:
    """§8 acceptance 8: scoring 1K candidates under 5 s on the dev DB.

    The per-candidate path runs sequential geo + pooled-signal lookups (v0
    N+1); v1 will batch them. This test is the wall-clock budget guard.
    """
    listing_emb = _listing_vec()
    city = f"ScaleTest-{uuid4().hex[:8]}"
    property_ids = [str(uuid4()) for _ in range(1_000)]
    await _bulk_insert_properties(
        session,
        property_ids,
        _SharedColumns(title="Scale property", status="active", price_huf=50_000_000, city=city),
    )
    await insert_embeddings_batch(session, "factual", [(pid, listing_emb) for pid in property_ids])
    await session.flush()

    start = time.perf_counter()
    result = await recommend(
        session,
        ParsedQuery(hard_filters=HardFilters(city=city)),
        query_emb=_query_vec(),
        client=MockLlmClient({}),
        top_n=30,
    )
    elapsed_s = time.perf_counter() - start

    assert len(result) == 30
    assert elapsed_s < 5.0, f"recommend over 1000 candidates took {elapsed_s:.2f}s (budget 5s)"


async def test_factual_embedding_index_used_in_candidate_retrieval(session: AsyncSession) -> None:
    """§8 acceptance 8 EXPLAIN clause: candidate retrieval uses the HNSW index."""
    index_exists = await session.scalar(
        text(
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_indexes "
            "WHERE schemaname = 'public' AND tablename = 'properties' "
            "AND indexdef ILIKE '%factual_embedding%' AND indexdef ILIKE '%hnsw%'"
            ")"
        )
    )
    if not index_exists:
        pytest.skip("0010_reassert_search_schema is not applied: factual HNSW index is missing")

    for _ in range(100):
        pid = str(uuid4())
        session.add(
            Property(
                id=pid,
                title="Indexed",
                city="Budapest",
                status="active",
                created_at=datetime.now(tz=UTC),
            )
        )
    await session.flush()

    # On a small integration fixture PostgreSQL reasonably prefers a sequence
    # scan. Disable it locally so this test verifies that the ANN index is a
    # viable execution path, not the planner's small-table cost preference.
    await session.execute(text("SET LOCAL enable_seqscan = off"))

    plan_rows = await session.execute(
        text(
            "EXPLAIN (ANALYZE, FORMAT TEXT) "
            "SELECT id, 1 - (factual_embedding <=> CAST(:q AS vector)) AS sim "
            "FROM properties WHERE factual_embedding IS NOT NULL "
            "ORDER BY factual_embedding <=> CAST(:q AS vector) LIMIT 500"
        ),
        {"q": "[" + ",".join(str(float(v)) for v in _query_vec()) + "]"},
    )
    plan = "\n".join(str(row[0]) for row in plan_rows)
    # The hot path must not seq-scan; either the HNSW index or a fallback index is acceptable.
    assert "Seq Scan on properties" not in plan, f"factual_embedding ANN query seq-scanned:\n{plan}"


# --- builder smoke against a real engine ------------------------------------


async def test_build_hard_filter_query_executes_against_real_db(session: AsyncSession) -> None:
    """Sanity check that the builder's SQL is syntactically accepted by Postgres."""
    stmt, params = build_hard_filter_query(HardFilters(city="Budapest", district_in=["12"], price_huf_max=80_000_000))
    result = await session.execute(stmt, params)
    rows = list(result)
    assert isinstance(rows, list)


@dataclass(frozen=True)
class _SharedColumns:
    """The column values every bulk-inserted property row shares."""

    title: str
    status: str
    price_huf: int | None
    city: str = "Budapest"


async def _bulk_insert_properties(session: AsyncSession, property_ids: Sequence[str], columns: _SharedColumns) -> None:
    for start in range(0, len(property_ids), 1000):
        chunk = property_ids[start : start + 1000]
        await session.execute(
            insert(Property),
            [
                {
                    "id": pid,
                    "title": columns.title,
                    "city": columns.city,
                    "status": columns.status,
                    "price_huf": columns.price_huf,
                    "created_at": datetime.now(tz=UTC),
                }
                for pid in chunk
            ],
        )
    await session.flush()
