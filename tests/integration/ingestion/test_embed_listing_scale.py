"""Embedding integration + scale tests.

`embed_listing` writes both vectors to the DB (verified by a raw SQL count), and a
500-listing scale run checks the batched call count and a bounded wall time.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ingovo_search.db.models import Property
from ingovo_search.db.session import asyncpg_connect_args
from ingovo_search.ingestion.embed_listing import embed_listing, embed_listings_batch
from ingovo_search.lib.embedding_cache import PostgresEmbeddingCache
from ingovo_search.lib.embeddings import COHERE_BATCH_LIMIT, MockEmbeddingClient
from ingovo_search.lib.env import get_settings, sqlalchemy_asyncpg_url

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


async def test_embed_listing_writes_both_vectors_to_db(session: AsyncSession) -> None:
    """Verified by raw SQL on the embedding columns."""
    source_tag = f"mock-embed-single-{uuid4()}"
    prop_id = await _insert_property(session, source_tag, index=0)
    await session.flush()

    client = MockEmbeddingClient()
    factual_cache = PostgresEmbeddingCache(session)
    descriptive_cache = PostgresEmbeddingCache(session)

    result = await embed_listing(
        prop_id,
        session=session,
        client=client,
        factual_cache=factual_cache,
        descriptive_cache=descriptive_cache,
    )
    await session.flush()

    assert result.factual_written is True
    assert result.descriptive_written is True

    row = await session.execute(
        text(
            """
            SELECT
              factual_embedding IS NOT NULL AS has_factual,
              descriptive_embedding IS NOT NULL AS has_descriptive
            FROM properties WHERE id = :pid
            """
        ),
        {"pid": prop_id},
    )
    one = row.one()
    assert one.has_factual is True
    assert one.has_descriptive is True


async def test_embed_listings_batch_500_rows_uses_batched_api(session: AsyncSession) -> None:
    """500 listings, all populated, batched call count, bounded wall time."""
    source_tag = f"mock-embed-stress-{uuid4()}"
    ids = [await _insert_property(session, source_tag, index=index) for index in range(500)]
    await session.flush()

    client = MockEmbeddingClient()

    started = time.monotonic()
    results = await embed_listings_batch(ids, session=session, client=client)
    elapsed = time.monotonic() - started
    await session.flush()

    assert elapsed < 60, f"Scale run took {elapsed:.2f}s, expected under 60s"
    assert len(results) == 500
    assert all(result.factual_written and result.descriptive_written for result in results)

    # Cohere's batch limit caps each call at COHERE_BATCH_LIMIT texts. For 500
    # listings across 2 input streams (factual + descriptive) that's at most
    # 2 * ceil(500 / 96) = 12 batch calls -- well below the unbatched 1000.
    expected_batch_calls = 2 * ((500 + COHERE_BATCH_LIMIT - 1) // COHERE_BATCH_LIMIT)
    assert client.batch_calls == expected_batch_calls
    assert client.batch_calls < 1000, "batched path must beat the per-text fallback"

    factual_count = await session.scalar(
        select(func.count())
        .select_from(Property)
        .where(
            Property.source == source_tag,
            Property.factual_embedding.is_not(None),
        )
    )
    descriptive_count = await session.scalar(
        select(func.count())
        .select_from(Property)
        .where(
            Property.source == source_tag,
            Property.descriptive_embedding.is_not(None),
        )
    )
    assert factual_count == 500
    assert descriptive_count == 500


async def _insert_property(session: AsyncSession, source_tag: str, *, index: int) -> str:
    prop_id = str(uuid4())
    prop = Property(
        id=prop_id,
        title=f"Stress listing {index}",
        description=f"Synthetic description for listing {index}, with a few HU words: szépen felújított panel.",
        cleaned_description=f"Cleaned: stress listing {index} — panel, déli tájolás.",
        one_paragraph_summary=f"Summary: stress listing {index} csendes, jó közlekedéssel.",
        price_huf=30_000_000 + index * 10_000,
        transaction_type="sale",
        city="Budapest",
        district=str((index % 22) + 1),
        rooms=str((index % 4) + 1),
        property_type="panel" if index % 2 == 0 else "tégla",
        area_m2=40.0 + (index % 50),
        status="active",
        images=[],
        view_count=0,
        save_count=0,
        contact_count=0,
        photo_count=0,
        source=source_tag,
        source_id=f"{source_tag}-{index}",
        has_balcony=bool(index % 3 == 0),
        elevator=bool(index % 5 == 0),
    )
    session.add(prop)
    return prop_id
