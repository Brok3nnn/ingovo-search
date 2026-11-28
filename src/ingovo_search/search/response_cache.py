"""The search response cache: its key, the visibility recheck, the expiry-capped TTL."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.lib.cache import Cache, search_cache_key
from ingovo_search.recommender.score import ScoredCandidate
from ingovo_search.recommender.sql_filter import cached_listing_ids_are_publicly_visible
from ingovo_search.search.constants import SEARCH_CACHE_TTL_SECONDS, SEARCH_RESPONSE_CACHE_VERSION, search_logger
from ingovo_search.search.models import SearchResponse
from ingovo_search.search.run import RerankBackend, SearchOptions


def rerank_cache_suffix(backend: RerankBackend) -> str:
    """Response-cache key suffix per reranker, so one backend never serves another's order.

    Gemini keeps the historical keys unchanged, and with them its warm cache.
    """
    return "" if backend == "gemini" else f":rb-{backend}"


def response_cache_key(parsed_hash: str, options: SearchOptions) -> str:
    """One entry per parsed query, sign-in state, response shape, reranker and language."""
    cache_key = (
        f"{search_cache_key(parsed_hash, user_signed_in=options.user_signed_in)}:{SEARCH_RESPONSE_CACHE_VERSION}"
    )
    if not options.include_narratives or not options.include_rerank:
        cache_key = f"{cache_key}:fast:narr{int(options.include_narratives)}:rerank{int(options.include_rerank)}"
    if options.include_rerank:
        cache_key = f"{cache_key}:rc{options.rerank_candidate_count}"
    cache_key = f"{cache_key}{rerank_cache_suffix(options.rerank_backend)}"
    # Narratives are part of the cached response, so a per-language response needs
    # its own entry; Hungarian keeps the existing key so its cache stays warm.
    if options.narrative_locale != "hu":
        cache_key = f"{cache_key}:nl{options.narrative_locale}"
    return cache_key


async def cached_response_is_publicly_visible(session: AsyncSession, response: SearchResponse) -> bool:
    """Check cached public result IDs at the same boundary as live search."""

    return await cached_listing_ids_are_publicly_visible(session, [result.listing_id for result in response.results])


async def discard_stale_response_cache_entry(cache: Cache, key: str) -> None:
    """Best-effort invalidation after a visibility miss; never re-serve it."""

    try:
        await cache.delete(key)
    except (OSError, ValueError, TypeError) as exc:
        search_logger.warning("stale search response cache delete failed", cache_key=key, error=str(exc))


def response_cache_ttl_seconds(
    results: list[ScoredCandidate],
    scored: Sequence[tuple[object, object]],
    *,
    now: datetime | None = None,
) -> int:
    """Cap a newly written response cache at its earliest known listing expiry.

    The mandatory hit-time DB recheck is the authorization boundary. This cap
    avoids avoidable stale entries on the usual write path, including the
    raw-query cache that reuses the computed TTL. `0` means do not cache a row
    expiring within the current second.
    """

    result_ids = {result.listing_id for result in results}
    if not result_ids:
        return SEARCH_CACHE_TTL_SECONDS

    reference = now or datetime.now(UTC)
    earliest_seconds: int | None = None
    for property_row, _breakdown in scored:
        if str(getattr(property_row, "id", "")) not in result_ids:
            continue
        expires_at = getattr(property_row, "expires_at", None)
        if not isinstance(expires_at, datetime):
            continue
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        remaining = int((expires_at - reference).total_seconds())
        earliest_seconds = remaining if earliest_seconds is None else min(earliest_seconds, remaining)

    if earliest_seconds is None:
        return SEARCH_CACHE_TTL_SECONDS
    return max(0, min(SEARCH_CACHE_TTL_SECONDS, earliest_seconds))
