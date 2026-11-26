"""Query-embedding cache reads and the raw-response cache gate."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass

from pydantic import BaseModel

from ingovo_search.lib.cache import (
    Cache,
    embedding_cache_key,
    get_json,
    normalise_query,
)
from ingovo_search.lib.embedding_models import EMBED_MULTILINGUAL
from ingovo_search.lib.embeddings import EmbeddingInputType
from ingovo_search.search.models import SearchRequest


class CachedEmbedding(BaseModel):
    """JSON payload stored in the request cache for runtime query embeddings."""

    model_config = {"frozen": True}

    embedding: list[float]


@dataclass(frozen=True)
class SpeculativeCachedEmbedding:
    """In-flight cache read for the raw query embedding while the parser runs."""

    normalised_text: str
    task: asyncio.Task[CachedEmbedding | None]


def raw_response_cache_allowed(payload: SearchRequest, *, has_images: bool = False) -> bool:
    """Only cache final responses for clean, session-independent searches.

    Raw-query cache hits happen before parsing, so they cannot safely depend on
    clarification answers, refinement overrides, session signals, or narratives.
    Requiring `reset_session` mirrors the homepage/new-search path and prevents
    prior browsing state from changing the correct ranking behind the same text.

    Attached photos disable it for the same reason: the key is built from the
    raw text only, so two searches with the same sentence and different pictures
    would otherwise collide. The post-parse response cache still applies — its
    key is a hash of the *merged* parsed query.
    """
    return (
        bool((payload.query or "").strip())
        and not has_images
        and payload.reset_session
        and payload.include_rerank
        and not payload.include_narratives
        and payload.rerank_candidate_count is None
        and payload.refinement_overrides is None
        and not payload.prior_clarifications
        and not payload.session_signals
        and not payload.w_overrides
    )


def speculative_embedding_allowed(payload: SearchRequest) -> bool:
    """Start only a cache read, never a network embed, before parsing.

    Raw text speculation is useful when the same natural-language query was
    searched recently and the parsed `semantic_query` is identical. It is kept
    cache-only to avoid paying Cohere for fast-path/filter-only queries whose
    parser result would intentionally skip embeddings.
    """
    query = (payload.query or "").strip()
    return bool(query and len(query.split()) >= 3)


async def read_cached_embedding(
    *,
    text_value: str,
    input_type: EmbeddingInputType,
    cache: Cache,
) -> CachedEmbedding | None:
    normalised = normalise_query(text_value)
    key = embedding_cache_key(normalised, model=EMBED_MULTILINGUAL, input_type=input_type)
    return await get_json(cache, key, CachedEmbedding)


async def discard_speculative_embedding(speculative: SpeculativeCachedEmbedding | None) -> None:
    if speculative is None:
        return
    if not speculative.task.done():
        speculative.task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await speculative.task
