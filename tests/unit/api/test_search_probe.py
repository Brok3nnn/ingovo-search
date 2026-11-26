"""The homepage index-probe route and its two-step parse (`ingovo_search.search.probe`).

The probe's SQL (`recommender.probe.build_probe`) is stubbed here; what is pinned is
which reading of the sentence reaches it: the LLM reading gap-filled by the
deterministic one, or the deterministic reading alone when the parser is slow,
down or broken.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ingovo_search.api.deps import get_db_session, get_llm_client, get_session_cache
from ingovo_search.api.envelope import install_exception_handlers
from ingovo_search.api.search_probe import router as probe_router
from ingovo_search.lib.cache import Cache, MemoryCache
from ingovo_search.lib.llm import LlmClient, MockLlmClient
from ingovo_search.recommender.parser import fallback_parse_query
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.probe import PriceStats, SearchProbe
from ingovo_search.recommender.search_preview import merge_missing_preview_fields
from ingovo_search.search.probe import SearchProbeRequest, probe_listing_pool
from ingovo_search.types.errors import LlmError

QUERY = "kétszobás lakás Budapesten 60 millióig"


def _probe(request_id: str = "probe-1") -> SearchProbe:
    return SearchProbe(
        request_id=request_id,
        intent_hash="hash",
        parsed_query=ParsedQuery(),
        understood_summary="Kétszobás lakás Budapesten",
        exact_count=3,
        price_stats=PriceStats(segment_count=3),
    )


def _db_session() -> AsyncMock:
    return AsyncMock()


def _llm_client() -> LlmClient:
    return MockLlmClient()


def _cache() -> Cache:
    return MemoryCache()


@pytest.fixture
def client() -> Iterator[TestClient]:
    application = FastAPI()
    install_exception_handlers(application)
    application.include_router(probe_router)
    application.dependency_overrides[get_db_session] = _db_session
    application.dependency_overrides[get_llm_client] = _llm_client
    application.dependency_overrides[get_session_cache] = _cache
    with TestClient(application) as test_client:
        yield test_client


def _parsed_sent_to_probe(build_probe: AsyncMock) -> ParsedQuery:
    build_probe.assert_awaited_once()
    assert build_probe.await_args is not None
    parsed = build_probe.await_args.args[1]
    assert isinstance(parsed, ParsedQuery)
    return parsed


def test_route_probes_the_llm_reading_gap_filled_by_the_deterministic_one(client: TestClient) -> None:
    llm_parsed = ParsedQuery(semantic_query="kétszobás lakás")
    with (
        patch("ingovo_search.search.probe.parse_query", new_callable=AsyncMock, return_value=llm_parsed),
        patch("ingovo_search.search.probe.build_probe", new_callable=AsyncMock, return_value=_probe()) as build,
    ):
        response = client.post("/api/search/probe", json={"query": QUERY, "locale": "en", "request_id": "probe-1"})

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["data"]["request_id"] == "probe-1"
    assert body["data"]["exact_count"] == 3
    assert _parsed_sent_to_probe(build) == merge_missing_preview_fields(llm_parsed, fallback_parse_query(QUERY))
    assert build.await_args is not None
    assert build.await_args.kwargs == {"request_id": "probe-1", "locale": "en"}


def test_route_rejects_a_too_short_sentence(client: TestClient) -> None:
    with patch("ingovo_search.search.probe.build_probe", new_callable=AsyncMock) as build:
        response = client.post("/api/search/probe", json={"query": "a"})

    assert response.status_code == 422
    build.assert_not_awaited()


@pytest.mark.parametrize("failure", [LlmError("model down"), TimeoutError()])
async def test_expected_parser_outage_falls_back_to_the_deterministic_reading(failure: Exception) -> None:
    logger = MagicMock()
    with (
        patch("ingovo_search.search.probe.parse_query", new_callable=AsyncMock, side_effect=failure),
        patch("ingovo_search.search.probe.build_probe", new_callable=AsyncMock, return_value=_probe()) as build,
        patch("ingovo_search.search.probe._logger", logger),
    ):
        probe = await probe_listing_pool(
            SearchProbeRequest(query=QUERY, request_id="probe-1"), AsyncMock(), MockLlmClient(), MemoryCache()
        )

    assert probe.request_id == "probe-1"
    assert _parsed_sent_to_probe(build) == fallback_parse_query(QUERY)
    logger.warning.assert_called_once_with(
        "search probe parser fallback", request_id="probe-1", error=type(failure).__name__
    )
    logger.exception.assert_not_called()


async def test_unexpected_parser_failure_is_logged_with_traceback_and_falls_back() -> None:
    logger = MagicMock()
    with (
        patch("ingovo_search.search.probe.parse_query", new_callable=AsyncMock, side_effect=RuntimeError("bug")),
        patch("ingovo_search.search.probe.build_probe", new_callable=AsyncMock, return_value=_probe()) as build,
        patch("ingovo_search.search.probe._logger", logger),
    ):
        await probe_listing_pool(
            SearchProbeRequest(query=QUERY, request_id="probe-1"), AsyncMock(), MockLlmClient(), MemoryCache()
        )

    assert _parsed_sent_to_probe(build) == fallback_parse_query(QUERY)
    logger.exception.assert_called_once_with("search probe parser fallback", request_id="probe-1")
    logger.warning.assert_not_called()


async def test_a_zero_parser_timeout_awaits_the_parser_without_a_deadline() -> None:
    llm_parsed = ParsedQuery(semantic_query="kétszobás lakás")
    settings = SimpleNamespace(ai_search_parser_timeout_ms=0)
    with (
        patch("ingovo_search.search.probe.get_settings", return_value=settings),
        patch("ingovo_search.search.probe.parse_query", new_callable=AsyncMock, return_value=llm_parsed),
        patch("ingovo_search.search.probe.build_probe", new_callable=AsyncMock, return_value=_probe()) as build,
    ):
        await probe_listing_pool(SearchProbeRequest(query=QUERY), AsyncMock(), MockLlmClient(), MemoryCache())

    assert _parsed_sent_to_probe(build) == merge_missing_preview_fields(llm_parsed, fallback_parse_query(QUERY))
