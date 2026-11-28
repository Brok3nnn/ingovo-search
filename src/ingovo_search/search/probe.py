"""The homepage index probe: a grounded live preview of the listing pool.

Companion to ``/api/search/preview``: same two-step parse (LLM parser with a
deterministic fallback + gap-fill merge), but instead of only interpreting the
sentence it counts and previews the real index via ``recommender.probe``.
"""

from __future__ import annotations

import asyncio
from typing import Literal
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.lib.cache import Cache
from ingovo_search.lib.env import get_settings
from ingovo_search.lib.llm import LlmClient
from ingovo_search.recommender.parser import CacheParserCache, fallback_parse_query, parse_query
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.probe import SearchProbe, build_probe
from ingovo_search.recommender.search_preview import merge_missing_preview_fields
from ingovo_search.types.errors import LlmError

# structlog, not stdlib: the request id bound by `RequestIdMiddleware` lands on every line.
# The logger keeps its name from `ingovo_search.api.search_probe`, so the log lines read as before.
_logger = structlog.get_logger("ingovo_search.api.search_probe")


class SearchProbeRequest(BaseModel):
    model_config = {"frozen": True}

    query: str = Field(min_length=2, max_length=600)
    locale: Literal["hu", "en"] = "hu"
    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=100)


async def _parse_probe_query(
    payload: SearchProbeRequest, db_session: AsyncSession, llm_client: LlmClient, cache: Cache
) -> ParsedQuery:
    """The LLM reading gap-filled by the deterministic one, or the deterministic reading alone."""
    settings = get_settings()
    parser_call = parse_query(payload.query, client=llm_client, cache=CacheParserCache(cache), session=db_session)
    try:
        parsed = (
            await asyncio.wait_for(parser_call, timeout=settings.ai_search_parser_timeout_ms / 1000)
            if settings.ai_search_parser_timeout_ms > 0
            else await parser_call
        )
    except (TimeoutError, LlmError, RedisError) as exc:
        # Expected noise — a slow or unavailable model or cache (a schema miss is a `ParseError`).
        _logger.warning("search probe parser fallback", request_id=payload.request_id, error=type(exc).__name__)
        parsed = fallback_parse_query(payload.query)
    except Exception:
        # Deliberately broad: a model outage arrives as `LlmError` above, so what
        # reaches here is a bug in the parser path. The probe runs while the
        # visitor types, and the deterministic reading beats an error; logged at
        # error with the traceback so a real bug is not mistaken for an outage.
        _logger.exception("search probe parser fallback", request_id=payload.request_id)
        parsed = fallback_parse_query(payload.query)
    else:
        parsed = merge_missing_preview_fields(parsed, fallback_parse_query(payload.query))
    return parsed


async def probe_listing_pool(
    payload: SearchProbeRequest, db_session: AsyncSession, llm_client: LlmClient, cache: Cache
) -> SearchProbe:
    """Count and preview the live listing pool for the current sentence."""
    parsed = await _parse_probe_query(payload, db_session, llm_client, cache)
    return await build_probe(db_session, parsed, request_id=payload.request_id, locale=payload.locale)
