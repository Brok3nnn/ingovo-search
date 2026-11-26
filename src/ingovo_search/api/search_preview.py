"""Homepage semantic search preview route."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from redis.exceptions import RedisError

from ingovo_search.api.deps import get_llm_client, get_session_cache
from ingovo_search.api.envelope import Envelope, ok
from ingovo_search.lib.cache import Cache
from ingovo_search.lib.env import get_settings
from ingovo_search.lib.llm import LlmClient
from ingovo_search.recommender.parser import CacheParserCache, fallback_parse_query, parse_query
from ingovo_search.recommender.search_preview import SearchPreview, build_search_preview, merge_missing_preview_fields
from ingovo_search.types.errors import LlmError

router = APIRouter(prefix="/api/search", tags=["search-preview"])
# structlog, not stdlib: the request id bound by `RequestIdMiddleware` lands on every line.
_logger = structlog.get_logger("ingovo_search.api.search_preview")


class SearchPreviewRequest(BaseModel):
    model_config = {"frozen": True}

    query: str = Field(min_length=2, max_length=600)
    locale: Literal["hu", "en"] = "hu"
    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=100)
    previous_intent_hash: str | None = Field(default=None, max_length=100)


@router.post("/preview", response_model=Envelope[SearchPreview])
async def search_preview(
    payload: SearchPreviewRequest,
    llm_client: Annotated[LlmClient, Depends(get_llm_client)],
    cache: Annotated[Cache, Depends(get_session_cache)],
) -> Envelope[SearchPreview]:
    """Interpret the current sentence without running the full listing search."""

    settings = get_settings()
    parser_call = parse_query(payload.query, client=llm_client, cache=CacheParserCache(cache))
    try:
        parsed = (
            await asyncio.wait_for(parser_call, timeout=settings.ai_search_parser_timeout_ms / 1000)
            if settings.ai_search_parser_timeout_ms > 0
            else await parser_call
        )
    except (TimeoutError, LlmError, RedisError) as exc:
        # Expected noise — a slow or unavailable model or cache (a schema miss is a `ParseError`).
        _logger.warning("search preview parser fallback", request_id=payload.request_id, error=type(exc).__name__)
        parsed = fallback_parse_query(payload.query)
    except Exception:
        # Deliberately broad: `GoogleAiClient.complete` re-raises the raw SDK
        # exception, so the set is not enumerable here. The preview runs while
        # the visitor types, and the deterministic reading beats an error; logged
        # at error with the traceback so a real bug is not mistaken for an outage.
        _logger.exception("search preview parser fallback", request_id=payload.request_id)
        parsed = fallback_parse_query(payload.query)
    else:
        parsed = merge_missing_preview_fields(parsed, fallback_parse_query(payload.query))
    return ok(build_search_preview(payload.query, payload.locale, parsed, payload.request_id))


__all__ = ["SearchPreviewRequest", "router", "search_preview"]
