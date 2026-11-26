"""Buyer search routes.

The orchestration lives in `ingovo_search.search`:
`request.execute_search_request` runs one request, `clarify.clarify_search` re-runs
the session's last parsed query with the buyer's answer, and
`pipeline.run_search_pipeline` is the shared §6.4 pipeline, one stage per function.
The DTOs and scoring builders live in `ingovo_search.search`, the session mutation
utilities in `search_session.py`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from ingovo_search.api.envelope import err
from ingovo_search.api.route_deps import SearchDependenciesDep, SessionContextDep
from ingovo_search.api.search_image import read_search_images
from ingovo_search.api.search_session import ClarifyRequest, get_or_create_session_state
from ingovo_search.search.clarify import clarify_search
from ingovo_search.search.constants import RERANK_DESCRIPTION_MAX_CHARS
from ingovo_search.search.models import (
    SCORE_VERSION,
    PipelineVersions,
    SearchEnvelope,
    SearchRequest,
    SearchResponse,
    search_request_from_json,
)
from ingovo_search.search.request import SearchImages, execute_search_request
from ingovo_search.search.scoring import listing_rerank_summary as _listing_rerank_summary
from ingovo_search.search.shadow import drain_rerank_shadow_tasks

__all__ = [
    "RERANK_DESCRIPTION_MAX_CHARS",
    "SCORE_VERSION",
    "PipelineVersions",
    "SearchRequest",
    "SearchResponse",
    "_listing_rerank_summary",
    "drain_rerank_shadow_tasks",
    "get_or_create_session_state",
    "router",
]


router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=SearchEnvelope[SearchResponse])
async def search(
    payload: SearchRequest,
    session_ctx: SessionContextDep,
    deps: SearchDependenciesDep,
) -> SearchEnvelope[SearchResponse] | JSONResponse:
    """Buyer-side search with full recommender pipeline. JSON, no attachments."""
    return await execute_search_request(payload, SearchImages(files=[], locale="hu"), session_ctx, deps)


@router.post("/search/with-images", response_model=SearchEnvelope[SearchResponse])
async def search_with_images(
    session_ctx: SessionContextDep,
    deps: SearchDependenciesDep,
    payload: Annotated[str, Form(description="`SearchRequest` as a JSON string")] = "{}",
    locale: Annotated[str, Form()] = "hu",
    files: Annotated[list[UploadFile], File()] = [],  # noqa: B006 — FastAPI reads the default, never mutates it
) -> SearchEnvelope[SearchResponse] | JSONResponse:
    """The same search, as one multipart request carrying the photos.

    Deliberately a single round trip: the pictures are analysed *inside* this
    request, concurrently with the parser, so attaching a photo does not make
    the user wait through an upload step before they can even hit enter.
    """
    parsed_payload = search_request_from_json(payload)
    return await execute_search_request(
        parsed_payload,
        SearchImages(files=await read_search_images(files), locale="hu" if locale == "hu" else "en"),
        session_ctx,
        deps,
    )


@router.post("/search/clarify", response_model=SearchEnvelope[SearchResponse])
async def clarify(
    payload: ClarifyRequest,
    session_ctx: SessionContextDep,
    deps: SearchDependenciesDep,
) -> SearchEnvelope[SearchResponse] | JSONResponse:
    """Append the answer to session state, re-run the search pipeline."""
    envelope = await clarify_search(payload, session_ctx, deps)
    if envelope is not None:
        return envelope
    # Clarify only makes sense after a prior /search call wrote a parsed query
    # into session state. Returning a 400 envelope is more useful than a 500.
    return JSONResponse(
        status_code=400,
        content=err(
            "no_prior_search",
            "No prior search to clarify. POST /api/search first.",
        ).model_dump(mode="json"),
    )
