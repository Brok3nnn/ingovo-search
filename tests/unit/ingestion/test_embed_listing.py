"""Unit tests for `embed_listing` cache behavior.

The DB-write side of acceptance 3 + the scale path live in the integration
suite (`tests/integration/ingestion/test_embed_listing_scale.py`). This file
covers the cache-hit short-circuit (acceptance 4) with a lightweight stub
session that records `execute` calls but doesn't touch a real database.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.db.models import Property
from ingovo_search.ingestion.embed_listing import embed_listing
from ingovo_search.lib.embedding_cache import InMemoryEmbeddingCache
from ingovo_search.lib.embeddings import MockEmbeddingClient


class _StubResult:
    """Minimal stand-in for SQLAlchemy `Result` — `insert_embedding` ignores it."""


class _StubSession:
    """Records `session.get` / `session.execute` without a database."""

    def __init__(self, prop: Property) -> None:
        self._prop = prop
        self.execute_calls: int = 0

    async def get(self, _model: Any, pk: str) -> Property | None:
        return self._prop if pk == self._prop.id else None

    async def execute(self, _stmt: Any, _params: Any = None) -> _StubResult:
        self.execute_calls += 1
        return _StubResult()


def _make_property() -> Property:
    return Property(
        id=str(uuid4()),
        title="Test listing",
        city="Budapest",
        district="13",
        rooms="2",
        property_type="panel",
        area_m2=55.0,
        status="active",
        images=[],
        view_count=0,
        save_count=0,
        contact_count=0,
        photo_count=0,
        transaction_type="sale",
        has_balcony=True,
        elevator=True,
        cleaned_description="Felújított panel lakás.",
        one_paragraph_summary="Szépen felújított, csendes panel.",
    )


async def test_embed_listing_cache_hit_skips_embed_calls() -> None:
    """Acceptance 4 from §5 — second pass over the same property does not call `embed`."""
    prop = _make_property()
    session = cast(AsyncSession, _StubSession(prop))
    client = MockEmbeddingClient()
    factual_cache = InMemoryEmbeddingCache()
    descriptive_cache = InMemoryEmbeddingCache()

    first = await embed_listing(
        prop.id,
        session=session,
        client=client,
        factual_cache=factual_cache,
        descriptive_cache=descriptive_cache,
    )
    second = await embed_listing(
        prop.id,
        session=session,
        client=client,
        factual_cache=factual_cache,
        descriptive_cache=descriptive_cache,
    )

    assert first.factual_written is True
    assert first.descriptive_written is True
    assert first.factual_cache_hit is False
    assert first.descriptive_cache_hit is False
    assert second.factual_cache_hit is True
    assert second.descriptive_cache_hit is True
    assert client.calls == 2  # one factual + one descriptive on first pass; cache hits on second
    assert cast(_StubSession, session).execute_calls == 4  # two inserts per pass


async def test_embed_listing_returns_empty_result_when_property_missing() -> None:
    prop = _make_property()
    session = cast(AsyncSession, _StubSession(prop))
    client = MockEmbeddingClient()

    result = await embed_listing("nonexistent-id", session=session, client=client)

    assert not result.factual_written
    assert not result.descriptive_written
    assert client.calls == 0


async def test_embed_listing_skips_descriptive_when_text_empty() -> None:
    prop = _make_property()
    prop.cleaned_description = None
    prop.one_paragraph_summary = None
    prop.description = None
    session = cast(AsyncSession, _StubSession(prop))
    client = MockEmbeddingClient()

    result = await embed_listing(prop.id, session=session, client=client)

    assert result.factual_written is True
    assert result.descriptive_written is False
    assert client.calls == 1
