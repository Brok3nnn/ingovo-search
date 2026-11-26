"""NDJSON transport for the real buyer-search execution trace.

The two routes read the request; :mod:`ingovo_search.search.stream` runs the search
and turns its public progress events into the streaming response. The events expose
observable pipeline state, never hidden model reasoning.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from ingovo_search.api.route_deps import SearchDependenciesDep, SessionContextDep
from ingovo_search.api.search_image import read_search_images
from ingovo_search.search.models import SearchRequest, search_request_from_json
from ingovo_search.search.request import SearchImages
from ingovo_search.search.stream import stream_search_response

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("/stream")
async def stream_search(
    payload: SearchRequest,
    session_ctx: SessionContextDep,
    deps: SearchDependenciesDep,
) -> StreamingResponse:
    return await stream_search_response(payload, SearchImages(files=[], locale="hu"), session_ctx, deps)


@router.post("/stream/with-images")
async def stream_search_with_images(
    session_ctx: SessionContextDep,
    deps: SearchDependenciesDep,
    payload: Annotated[str, Form(description="`SearchRequest` as a JSON string")] = "{}",
    locale: Annotated[str, Form()] = "hu",
    files: Annotated[list[UploadFile], File()] = [],  # noqa: B006
) -> StreamingResponse:
    parsed_payload = search_request_from_json(payload)
    return await stream_search_response(
        parsed_payload,
        SearchImages(files=await read_search_images(files), locale="hu" if locale == "hu" else "en"),
        session_ctx,
        deps,
    )


__all__ = ["router"]
