"""Integration tests for the observability writers — requires the seeded local Postgres.

Run via:
    uv run pytest -m integration

Covers the DB-touching paths: `log_search` writes a row and `daily_llm_cost`
aggregates a real `EnrichmentLog` insert.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ingovo_search.db.models import EnrichmentLog, SearchLog
from ingovo_search.db.session import asyncpg_connect_args
from ingovo_search.lib.cost_tracker import daily_llm_cost, daily_llm_cost_total
from ingovo_search.lib.env import get_settings, sqlalchemy_asyncpg_url
from ingovo_search.lib.observability import log_search

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


async def test_log_search_persists_row(session: AsyncSession) -> None:
    row = await log_search(
        session,
        query_hash=str(uuid4()),
        hard_filter_count=2,
        sql_candidate_count=120,
        returned_count=30,
        latency_ms=87,
        score_version="v0",
        top_result_ids=["a", "b"],
        parsed_query={"semantic_query": "lakás"},
        vector_candidate_count=120,
    )
    await session.flush()
    result = await session.execute(text("SELECT id, query_hash FROM search_log WHERE id = :id"), {"id": row.id})
    fetched = result.one_or_none()
    assert fetched is not None
    assert fetched.query_hash == row.query_hash


async def test_daily_llm_cost_aggregates_inserted_rows(session: AsyncSession) -> None:
    today = datetime.now(tz=UTC).date()
    for cost in (1.0, 0.5, 2.0):
        session.add(
            EnrichmentLog(
                id=str(uuid4()),
                property_id=None,
                model_used="gemini-2.5-flash-lite",
                prompt_version="cost-test/v0",
                input_tokens=10,
                output_tokens=20,
                cost_usd=cost,
                latency_ms=100,
                outcome="success",
            )
        )
    await session.flush()

    total = await daily_llm_cost_total(session, today)
    assert total >= 3.5  # at least the rows we inserted


async def test_daily_llm_cost_breakdown_groups_by_model(session: AsyncSession) -> None:
    today = datetime.now(tz=UTC).date()
    for model, cost in (("gemini-2.5-flash-lite", 1.0), ("cohere-embed", 0.4)):
        session.add(
            EnrichmentLog(
                id=str(uuid4()),
                property_id=None,
                model_used=model,
                prompt_version="cost-test/v0",
                input_tokens=10,
                output_tokens=20,
                cost_usd=cost,
                latency_ms=100,
                outcome="success",
            )
        )
    await session.flush()

    breakdown = await daily_llm_cost(session, today)
    assert breakdown.get("gemini-2.5-flash-lite", 0.0) >= 1.0
    assert breakdown.get("cohere-embed", 0.0) >= 0.4


async def test_search_log_persists_through_search_log_table(session: AsyncSession) -> None:
    """SearchLog round-trip via direct ORM insert keeps the model behavior locked."""
    row = SearchLog(
        id=str(uuid4()),
        query_hash="hash-2",
        parsed_query=None,
        hard_filter_count=0,
        sql_candidate_count=0,
        returned_count=0,
        latency_ms=10,
        score_version="v0",
        top_result_ids=[],
    )
    session.add(row)
    await session.flush()
    fetched = await session.execute(text("SELECT id FROM search_log WHERE id = :id"), {"id": row.id})
    assert fetched.one_or_none() is not None
