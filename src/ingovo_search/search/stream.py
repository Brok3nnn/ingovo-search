"""NDJSON transport for the real buyer-search execution trace.

The search itself is ``request.execute_search_request``; this module only turns its
public progress events into a streaming response for ``POST /api/search/stream``. The
events expose observable pipeline state, never hidden model reasoning.

The search runs as a producer task that puts its events on a queue, and the response
body drains that queue: ``search.started`` first, then the progress events, then one
terminal ``result`` or ``error`` event. A reader who goes away cancels the producer.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from typing import cast
from uuid import uuid4

import structlog
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from ingovo_search.api.search_events import (
    SearchErrorEvent,
    SearchProgressEvent,
    SearchResultEvent,
    SearchStartedEvent,
)
from ingovo_search.api.search_session import get_or_create_session_state
from ingovo_search.lib.session import SessionState
from ingovo_search.search.images import SearchImages
from ingovo_search.search.models import SearchEnvelope, SearchRequest, SearchResponse
from ingovo_search.search.request import SessionContext, StreamHandoff, execute_search_request
from ingovo_search.search.run import SearchDependencies
from ingovo_search.types.errors import ParseError

# Module-local on purpose (ruff's blind-except exemption), and it keeps its name from
# `ingovo_search.api.search_stream`, so the log lines read as before the move.
_logger = structlog.get_logger("ingovo_search.api.search_stream")

_STREAM_END = object()


@dataclass(frozen=True, slots=True)
class _StreamRun:
    """One streamed search: what the producer runs and the queue the response body drains."""

    payload: SearchRequest
    images: SearchImages
    session_ctx: SessionContext
    deps: SearchDependencies
    request_id: str
    initial_state: SessionState
    queue: asyncio.Queue[SearchProgressEvent | object]

    async def emit(self, event: SearchProgressEvent) -> None:
        await self.queue.put(event)


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) and value else str(uuid4())


def _search_failed(request_id: str) -> SearchErrorEvent:
    return SearchErrorEvent(
        request_id=request_id,
        code="search_failed",
        message="A keresést most nem sikerült befejezni. Kérlek, próbáld újra.",
    )


def _safe_error(exc: Exception, request_id: str) -> SearchErrorEvent:
    if isinstance(exc, ParseError):
        return SearchErrorEvent(
            request_id=request_id,
            code="invalid_input",
            message=str(exc),
            retryable=False,
        )
    return _search_failed(request_id)


def _terminal_event(envelope: SearchEnvelope[SearchResponse] | JSONResponse, request_id: str) -> SearchProgressEvent:
    """The finished search as the stream's last event: its result, or a safe error."""
    if isinstance(envelope, JSONResponse) or envelope.data is None:
        return _search_failed(request_id)
    return SearchResultEvent(
        request_id=request_id,
        data=envelope.data,
        timings_ms=envelope.meta.timings_ms,
    )


async def _run_search(run: _StreamRun) -> None:
    """The producer: run the search, emit its terminal event, always close the stream."""
    try:
        envelope = await execute_search_request(
            run.payload,
            run.images,
            run.session_ctx,
            run.deps,
            StreamHandoff(initial_state=run.initial_state, progress=run.emit, request_id=run.request_id),
        )
        await run.emit(_terminal_event(envelope, run.request_id))
    except ParseError as exc:
        # The reader's own bad input: no traceback, just the terminal event.
        _logger.warning("search stream rejected the request", request_id=run.request_id, error=str(exc))
        await run.emit(_safe_error(exc, run.request_id))
    except Exception as exc:
        # Deliberately broad: the NDJSON stream is already open with a 200, so
        # an exception escaping here could only truncate it — the reader needs
        # a terminal error event instead. The failure was previously invisible;
        # it is now logged at error with the traceback.
        _logger.exception("search stream failed", request_id=run.request_id)
        await run.emit(_safe_error(exc, run.request_id))
    finally:
        await run.queue.put(_STREAM_END)


async def _ndjson_lines(run: _StreamRun) -> AsyncIterator[str]:
    """The response body: ``search.started``, then every queued event until the producer ends."""
    yield (
        SearchStartedEvent(
            request_id=run.request_id,
            query=(run.payload.query or "").strip(),
        ).model_dump_json()
        + "\n"
    )
    producer = asyncio.create_task(_run_search(run))
    try:
        while True:
            item = await run.queue.get()
            if item is _STREAM_END:
                break
            event = cast(SearchProgressEvent, item)
            yield event.model_dump_json() + "\n"
    finally:
        if not producer.done():
            producer.cancel()
        with suppress(asyncio.CancelledError):
            await producer


async def stream_search_response(
    payload: SearchRequest,
    images: SearchImages,
    session_ctx: SessionContext,
    deps: SearchDependencies,
) -> StreamingResponse:
    """Open the session, then answer with the NDJSON stream; the search starts when the body is read."""
    request_id = _request_id(session_ctx.request)
    initial_state = await get_or_create_session_state(
        session_ctx.request, session_ctx.response, session_ctx.store, session_ctx.secret, session_ctx.cookie_secure
    )
    run = _StreamRun(
        payload=payload,
        images=images,
        session_ctx=session_ctx,
        deps=deps,
        request_id=request_id,
        initial_state=initial_state,
        queue=asyncio.Queue(),
    )
    stream_response = StreamingResponse(
        _ndjson_lines(run),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
    for cookie in session_ctx.response.headers.getlist("set-cookie"):
        stream_response.headers.append("set-cookie", cookie)
    return stream_response
