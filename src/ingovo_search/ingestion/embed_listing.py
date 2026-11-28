"""Embedding orchestrator — fetch property, build inputs, embed, persist.

`embed_listing` handles one property and uses the cache (cheap re-embeds for
re-runs). `embed_listings_batch` fans out across many properties using the
batched embedding API (≤ `COHERE_BATCH_LIMIT` texts per call) — the scale-test
path. Both call sites land vectors via `insert_embedding` so the
recommender's kNN helpers see them immediately.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.db.models import Property
from ingovo_search.ingestion.build_factual_text import build_descriptive_text, build_factual_text
from ingovo_search.lib.embedding_cache import EmbeddingCacheBackend, cache_key_for
from ingovo_search.lib.embedding_models import EMBED_MULTILINGUAL
from ingovo_search.lib.embeddings import COHERE_BATCH_LIMIT, EmbeddingClient, EmbeddingInputType
from ingovo_search.lib.vector_db import EmbeddingKind, insert_embedding, insert_embeddings_batch


@dataclass(slots=True)
class EmbedListingResult:
    property_id: str
    factual_written: bool = False
    descriptive_written: bool = False
    factual_cache_hit: bool = False
    descriptive_cache_hit: bool = False


async def embed_listing(
    property_id: str,
    *,
    session: AsyncSession,
    client: EmbeddingClient,
    factual_cache: EmbeddingCacheBackend | None = None,
    descriptive_cache: EmbeddingCacheBackend | None = None,
    model: str = EMBED_MULTILINGUAL,
) -> EmbedListingResult:
    """Embed one property's factual + descriptive texts and persist the vectors.

    Empty texts are skipped (no embedding call, no DB write). When a cache is
    provided, an unchanged text short-circuits the LLM call. Caller commits.
    """
    prop = await session.get(Property, property_id)
    if prop is None:
        return EmbedListingResult(property_id=property_id)

    factual_text = build_factual_text(prop)
    descriptive_text = build_descriptive_text(prop)
    result = EmbedListingResult(property_id=property_id)

    if factual_text:
        vec, hit = await _embed_with_cache(client, factual_cache, factual_text, "search_document", model)
        if factual_cache is not None and not hit:
            await factual_cache.set(
                cache_key_for(factual_text, model, "search_document"),
                text_value=factual_text,
                model=model,
                input_type="search_document",
                embedding=vec,
            )
        await insert_embedding(session, prop.id, "factual", vec)
        result.factual_written = True
        result.factual_cache_hit = hit

    if descriptive_text:
        vec, hit = await _embed_with_cache(client, descriptive_cache, descriptive_text, "search_document", model)
        if descriptive_cache is not None and not hit:
            await descriptive_cache.set(
                cache_key_for(descriptive_text, model, "search_document"),
                text_value=descriptive_text,
                model=model,
                input_type="search_document",
                embedding=vec,
            )
        await insert_embedding(session, prop.id, "descriptive", vec)
        result.descriptive_written = True
        result.descriptive_cache_hit = hit

    return result


async def embed_listings_batch(
    property_ids: Sequence[str],
    *,
    session: AsyncSession,
    client: EmbeddingClient,
    model: str = EMBED_MULTILINGUAL,
    chunk_size: int = COHERE_BATCH_LIMIT,
    db_lock: asyncio.Lock | None = None,
) -> list[EmbedListingResult]:
    """Embed many properties' factual + descriptive vectors using the batched API.

    The batched path skips the per-text cache — it's the bulk-rebuild path. For
    incremental re-embeds use `embed_listing`. `db_lock` serializes session
    access when the caller fans out concurrently.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    async with _maybe_lock(db_lock):
        props = await _load_properties(session, property_ids)

    by_id: dict[str, Property] = {prop.id: prop for prop in props}
    factual_texts: list[tuple[str, str]] = []
    descriptive_texts: list[tuple[str, str]] = []
    results: dict[str, EmbedListingResult] = {pid: EmbedListingResult(property_id=pid) for pid in property_ids}

    for prop_id in property_ids:
        prop = by_id.get(prop_id)
        if prop is None:
            continue
        factual = build_factual_text(prop)
        descriptive = build_descriptive_text(prop)
        if factual:
            factual_texts.append((prop.id, factual))
        if descriptive:
            descriptive_texts.append((prop.id, descriptive))

    await _embed_and_persist(
        session, client, factual_texts, "factual", "search_document", model, chunk_size, results, db_lock
    )
    await _embed_and_persist(
        session, client, descriptive_texts, "descriptive", "search_document", model, chunk_size, results, db_lock
    )

    return [results[pid] for pid in property_ids]


async def _embed_with_cache(
    client: EmbeddingClient,
    cache: EmbeddingCacheBackend | None,
    text_value: str,
    input_type: EmbeddingInputType,
    model: str,
) -> tuple[list[float], bool]:
    if cache is not None:
        cached = await cache.get(cache_key_for(text_value, model, input_type))
        if cached is not None:
            return cached, True
    vec = await client.embed(text_value, input_type)
    return vec, False


async def _embed_and_persist(
    session: AsyncSession,
    client: EmbeddingClient,
    items: list[tuple[str, str]],
    kind: EmbeddingKind,
    input_type: EmbeddingInputType,
    model: str,
    chunk_size: int,
    results: dict[str, EmbedListingResult],
    db_lock: asyncio.Lock | None,
) -> None:
    if not items:
        return
    for start in range(0, len(items), chunk_size):
        chunk = items[start : start + chunk_size]
        texts = [text for _, text in chunk]
        vectors = await client.embed_batch(texts, input_type)
        rows = [(prop_id, vec) for (prop_id, _text_value), vec in zip(chunk, vectors, strict=True)]
        async with _maybe_lock(db_lock):
            await insert_embeddings_batch(session, kind, rows)
        for (prop_id, _text_value), vec in zip(chunk, vectors, strict=True):
            _ = vec
            result = results.setdefault(prop_id, EmbedListingResult(property_id=prop_id))
            if kind == "factual":
                result.factual_written = True
            else:
                result.descriptive_written = True


async def _load_properties(session: AsyncSession, property_ids: Sequence[str]) -> list[Property]:
    if not property_ids:
        return []
    result = await session.execute(select(Property).where(Property.id.in_(list(property_ids))))
    return list(result.scalars())


@contextlib.asynccontextmanager
async def _maybe_lock(lock: asyncio.Lock | None) -> AsyncGenerator[None]:
    if lock is None:
        yield
        return
    async with lock:
        yield
