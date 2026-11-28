"""Contract tests for the buyer-search NDJSON transport."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from tests._fakes import fake_db_session

from ingovo_search.api.cookies import SESSION_COOKIE_NAME
from ingovo_search.api.deps import (
    get_db_session,
    get_embedding_client,
    get_llm_client,
    get_session_cache,
    get_session_cookie_secure,
    get_session_secret,
    get_session_store,
)
from ingovo_search.api.envelope import install_exception_handlers
from ingovo_search.api.search_events import CandidatesScoredEvent, IntentParsedEvent, SearchProgressEmitter
from ingovo_search.api.search_stream import router as stream_router
from ingovo_search.lib.cache import Cache, MemoryCache
from ingovo_search.lib.embeddings import MockEmbeddingClient
from ingovo_search.lib.llm import MockLlmClient
from ingovo_search.lib.session import SessionStore
from ingovo_search.search.models import SearchEnvelope, SearchMeta, SearchRequest, SearchResponse, ok_with_timings
from ingovo_search.search.request import SearchImages, SessionContext, StreamHandoff
from ingovo_search.search.run import SearchDependencies
from ingovo_search.search.stream import stream_search_response
from ingovo_search.types.errors import ParseError

SECRET = b"x" * 32
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"stream image" * 4


# FastAPI resolves an override's own signature, and these classes' constructors take parameters it
# would turn into request parameters, so each override is a parameterless factory (fresh per request).
def _mock_llm_client() -> MockLlmClient:
    return MockLlmClient()


def _mock_embedding_client() -> MockEmbeddingClient:
    return MockEmbeddingClient()


@pytest.fixture
def session_cache() -> Cache:
    return MemoryCache()


@pytest.fixture
def app(session_cache: Cache) -> FastAPI:
    application = FastAPI()
    install_exception_handlers(application)
    application.include_router(stream_router)
    application.dependency_overrides[get_session_store] = lambda: SessionStore(session_cache)
    application.dependency_overrides[get_session_cache] = lambda: session_cache
    application.dependency_overrides[get_session_secret] = lambda: SECRET
    application.dependency_overrides[get_session_cookie_secure] = lambda: False
    application.dependency_overrides[get_llm_client] = _mock_llm_client
    application.dependency_overrides[get_embedding_client] = _mock_embedding_client
    application.dependency_overrides[get_db_session] = fake_db_session
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def _events(response_text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in response_text.splitlines() if line.strip()]


def test_stream_keeps_event_order_request_id_and_cookie(client: TestClient) -> None:
    async def fake_execute(
        _payload: object, _images: object, _session: object, _deps: object, stream: StreamHandoff
    ) -> object:
        emit = cast(SearchProgressEmitter, stream.progress)
        request_id = str(stream.request_id)
        await emit(
            IntentParsedEvent(
                request_id=request_id,
                summary="Eladó lakás Budapesten",
                confidence=0.91,
            )
        )
        await emit(
            CandidatesScoredEvent(
                request_id=request_id,
                candidate_count=12,
                exact_count=18,
            )
        )
        return ok_with_timings(SearchResponse(total_candidates=12), {"total": 7}, request_id=request_id)

    with patch("ingovo_search.search.stream.execute_search_request", side_effect=fake_execute):
        response = client.post("/api/search/stream", json={"query": "budapesti lakás"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert SESSION_COOKIE_NAME in response.cookies
    events = _events(response.text)
    assert [event["type"] for event in events] == [
        "search.started",
        "intent.parsed",
        "candidates.scored",
        "result",
    ]
    request_ids = {event["request_id"] for event in events}
    assert len(request_ids) == 1
    assert events[2]["candidate_count"] == 12
    assert events[2]["exact_count"] == 18
    assert events[-1]["timings_ms"] == {"total": 7}


def test_cache_style_completion_emits_no_fake_pipeline_events(client: TestClient) -> None:
    async def fake_cache_hit(
        _payload: object, _images: object, _session: object, _deps: object, stream: StreamHandoff
    ) -> object:
        return ok_with_timings(
            SearchResponse(total_candidates=4),
            {"raw_response_cache_read": 1},
            request_id=str(stream.request_id),
        )

    with patch("ingovo_search.search.stream.execute_search_request", side_effect=fake_cache_hit):
        response = client.post("/api/search/stream", json={"query": "gyors cache találat"})

    events = _events(response.text)
    assert [event["type"] for event in events] == ["search.started", "result"]


def test_domain_error_is_a_safe_terminal_event(client: TestClient) -> None:
    async def fake_error(*_args: object) -> object:
        raise ParseError("Adj meg egy értelmezhető keresést.")

    with patch("ingovo_search.search.stream.execute_search_request", side_effect=fake_error):
        response = client.post("/api/search/stream", json={"query": "sa"})

    event = _events(response.text)[-1]
    assert event["type"] == "error"
    assert event["code"] == "invalid_input"
    assert event["retryable"] is False


def test_unexpected_failure_is_logged_and_ends_the_stream_with_a_safe_error(client: TestClient) -> None:
    async def fake_crash(*_args: object) -> object:
        raise RuntimeError("pipeline bug")

    logger = MagicMock()
    with (
        patch("ingovo_search.search.stream.execute_search_request", side_effect=fake_crash),
        patch("ingovo_search.search.stream._logger", logger),
    ):
        response = client.post("/api/search/stream", json={"query": "budapesti lakás"})

    events = _events(response.text)
    assert [event["type"] for event in events] == ["search.started", "error"]
    assert events[-1]["code"] == "search_failed"
    assert events[-1]["retryable"] is True
    assert events[-1]["request_id"] == events[0]["request_id"]
    logger.exception.assert_called_once_with("search stream failed", request_id=events[0]["request_id"])


@pytest.mark.parametrize(
    "envelope",
    [
        JSONResponse(status_code=500, content={"ok": False}),
        SearchEnvelope[SearchResponse](ok=False, meta=SearchMeta(timestamp="t", request_id="r", timings_ms={})),
    ],
    ids=["error-response", "envelope-without-data"],
)
def test_a_search_without_a_result_ends_with_a_safe_error(client: TestClient, envelope: object) -> None:
    async def fake_no_result(*_args: object) -> object:
        return envelope

    with patch("ingovo_search.search.stream.execute_search_request", side_effect=fake_no_result):
        response = client.post("/api/search/stream", json={"query": "budapesti lakás"})

    events = _events(response.text)
    assert [event["type"] for event in events] == ["search.started", "error"]
    assert events[-1]["code"] == "search_failed"
    assert events[-1]["message"] == "A keresést most nem sikerült befejezni. Kérlek, próbáld újra."
    assert events[-1]["retryable"] is True


def test_multipart_stream_reaches_the_same_terminal_contract(client: TestClient) -> None:
    async def fake_execute(
        _payload: object, images: SearchImages, _session: object, _deps: object, stream: StreamHandoff
    ) -> object:
        assert len(images.files) == 1
        return ok_with_timings(
            SearchResponse(total_candidates=1),
            {},
            request_id=str(stream.request_id),
        )

    with patch("ingovo_search.search.stream.execute_search_request", side_effect=fake_execute):
        response = client.post(
            "/api/search/stream/with-images",
            data={"payload": '{"query":"ilyen házat keresek"}', "locale": "hu"},
            files={"files": ("haz.jpg", JPEG_BYTES, "image/jpeg")},
        )

    assert [event["type"] for event in _events(response.text)] == ["search.started", "result"]


async def test_client_cancellation_stops_the_pipeline_task(session_cache: Cache) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def never_finishes(*_args: object) -> object:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/search/stream",
            "query_string": b"",
            "headers": [],
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "scheme": "http",
        }
    )
    request.state.request_id = "cancel-me"
    response = Response()

    with patch("ingovo_search.search.stream.execute_search_request", side_effect=never_finishes):
        stream = await stream_search_response(
            SearchRequest(query="lassú keresés"),
            SearchImages(files=[], locale="hu"),
            SessionContext(request, response, SessionStore(session_cache), SECRET, False),
            SearchDependencies(AsyncMock(), MockLlmClient(), MockEmbeddingClient(), session_cache),
        )
        iterator = cast(AsyncIterator[str | bytes], stream.body_iterator.__aiter__())
        first_chunk = await anext(iterator)
        assert '"type":"search.started"' in str(first_chunk)

        waiting_for_pipeline: asyncio.Future[str | bytes] = asyncio.ensure_future(anext(iterator))
        await asyncio.wait_for(entered.wait(), timeout=1)
        waiting_for_pipeline.cancel()
        with suppress(asyncio.CancelledError):
            await waiting_for_pipeline
        await asyncio.wait_for(cancelled.wait(), timeout=1)
