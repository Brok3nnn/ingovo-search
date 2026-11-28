"""Integration tests for vector_db — requires local Postgres + pgvector + postgis.

Run via:
    uv run pytest -m integration

These cover acceptance tests that need a real database.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ingovo_search.db.models import Property
from ingovo_search.db.session import asyncpg_connect_args
from ingovo_search.lib.env import get_settings, sqlalchemy_asyncpg_url
from ingovo_search.lib.vector_db import (
    VECTOR_DIM,
    geo_set_point,
    geo_st_dwithin,
    insert_embedding,
    knn_by_query_embedding,
)

pytestmark = pytest.mark.integration


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
    prop = Property(id=pid, title="Test property", city="Budapest")
    sess.add(prop)
    await sess.flush()
    return pid


def _random_vec(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(VECTOR_DIM)]


async def test_extensions_present(session: AsyncSession) -> None:
    """After the schema is applied, both vector and postgis extensions are installed."""
    result = await session.execute(text("SELECT extname FROM pg_extension"))
    extensions = {row.extname for row in result}
    assert "vector" in extensions, "pgvector extension missing — run `uv run ingovo-search-seed`"
    assert "postgis" in extensions, "postgis extension missing — run `uv run ingovo-search-seed`"


async def test_insert_embedding_round_trip(session: AsyncSession) -> None:
    pid = await _insert_property(session)
    vec = _random_vec(1)

    await insert_embedding(session, pid, "factual", vec)
    await session.flush()

    result = await session.execute(
        text("SELECT factual_embedding IS NOT NULL AS has_emb FROM properties WHERE id = :pid"),
        {"pid": pid},
    )
    row = result.one()
    assert row.has_emb is True


async def test_knn_returns_ranked_by_similarity(session: AsyncSession) -> None:
    """Property kNN returns listings ranked by cosine similarity."""
    query_vec = [1.0, *([0.0] * (VECTOR_DIM - 1))]
    embeddings = [
        [1.0, *([0.0] * (VECTOR_DIM - 1))],
        [0.9, 0.1, *([0.0] * (VECTOR_DIM - 2))],
        [0.8, 0.2, *([0.0] * (VECTOR_DIM - 2))],
        [-1.0, *([0.0] * (VECTOR_DIM - 1))],
        [0.0, 1.0, *([0.0] * (VECTOR_DIM - 2))],
    ]
    expected_top_ids: list[str] = []

    for emb in embeddings:
        pid = await _insert_property(session)
        await insert_embedding(session, pid, "factual", emb)
        expected_top_ids.append(pid)

    await session.flush()

    # candidate_ids scopes the search to this test's rows — the dev DB also
    # holds the ~2K-listing eval corpus, whose real embeddings would otherwise
    # compete in a global kNN.
    rows = await knn_by_query_embedding(
        session, query_vec, kind="factual", sim_threshold=0.5, limit=5, candidate_ids=expected_top_ids
    )

    assert [row.property_id for row in rows] == expected_top_ids[:3]
    similarities = [r.similarity for r in rows]
    assert similarities == sorted(similarities, reverse=True), "results must be sorted by similarity desc"
    assert similarities[0] > 0.99


async def test_knn_filters_below_threshold(session: AsyncSession) -> None:
    query_vec = _random_vec(seed=0)

    # All inserted listing embeddings are unrelated — they should not pass a strict threshold.
    for i in range(3):
        pid = await _insert_property(session)
        emb = _random_vec(seed=i + 200)
        await insert_embedding(session, pid, "factual", emb)

    await session.flush()

    rows = await knn_by_query_embedding(session, query_vec, kind="factual", sim_threshold=0.99)
    # Threshold is very strict — random embeddings should not pass.
    assert rows == []


async def test_knn_filters_candidate_ids(session: AsyncSession) -> None:
    query_vec = [1.0, *([0.0] * (VECTOR_DIM - 1))]
    matching_id = await _insert_property(session)
    excluded_id = await _insert_property(session)
    await insert_embedding(session, matching_id, "factual", query_vec)
    await insert_embedding(session, excluded_id, "factual", query_vec)
    await session.flush()

    rows = await knn_by_query_embedding(
        session,
        query_vec,
        kind="factual",
        sim_threshold=0.5,
        candidate_ids=[matching_id],
    )

    assert [row.property_id for row in rows] == [matching_id]


async def test_geo_round_trip(session: AsyncSession) -> None:
    """`geo_set_point` + `geo_st_dwithin` round-trip returns the property ID."""
    pid = await _insert_property(session)

    # Budapest city centre
    await geo_set_point(session, pid, lat=47.4979, lng=19.0402)
    await session.flush()

    nearby = await geo_st_dwithin(session, center_lat=47.4979, center_lng=19.0402, radius_km=1.0)
    assert pid in nearby

    # Far away (Berlin)
    far = await geo_st_dwithin(session, center_lat=52.52, center_lng=13.405, radius_km=1.0)
    assert pid not in far


async def test_extension_creation_is_idempotent(session: AsyncSession) -> None:
    """Re-running CREATE EXTENSION IF NOT EXISTS does not raise — the schema script relies on it."""
    await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    await session.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
