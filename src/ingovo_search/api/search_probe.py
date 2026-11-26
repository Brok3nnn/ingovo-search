"""Homepage index-probe route — grounded live preview of the listing pool.

Companion to ``/api/search/preview``: same two-step parse (LLM parser with a
deterministic fallback + gap-fill merge), but instead of only interpreting the
sentence it counts and previews the real index via ``recommender.probe``. The
parse and the probe live in :mod:`ingovo_search.search.probe`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from ingovo_search.api.deps import get_llm_client, get_session_cache
from ingovo_search.api.envelope import Envelope, ok
from ingovo_search.db.deps import DbSession
from ingovo_search.lib.cache import Cache
from ingovo_search.lib.llm import LlmClient
from ingovo_search.recommender.probe import SearchProbe
from ingovo_search.search.probe import SearchProbeRequest, probe_listing_pool

router = APIRouter(prefix="/api/search", tags=["search-probe"])


@router.post("/probe", response_model=Envelope[SearchProbe])
async def search_probe(
    payload: SearchProbeRequest,
    db_session: DbSession,
    llm_client: Annotated[LlmClient, Depends(get_llm_client)],
    cache: Annotated[Cache, Depends(get_session_cache)],
) -> Envelope[SearchProbe]:
    """Count and preview the live listing pool for the current sentence."""

    return ok(await probe_listing_pool(payload, db_session, llm_client, cache))


__all__ = ["SearchProbeRequest", "router", "search_probe"]
