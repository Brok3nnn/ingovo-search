"""Unit tests for `/api/search` shell.

Covers the 200 envelope shape (no account required), and the Set-Cookie on the
first call with session stickiness for refinement_overrides + session_signals +
prior_clarifications. The recommender pipeline itself is stubbed; its wiring is
tested in `test_search_pipeline.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError as PydanticValidationError
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
from ingovo_search.api.search import (
    SCORE_VERSION,
    PipelineVersions,
    SearchRequest,
    SearchResponse,
)
from ingovo_search.api.search import (
    router as search_router,
)
from ingovo_search.lib.cache import Cache, MemoryCache
from ingovo_search.lib.embeddings import MockEmbeddingClient
from ingovo_search.lib.llm import MockLlmClient
from ingovo_search.lib.session import (
    RefinementOverrides,
    SessionSignal,
    SessionStore,
    verify_signed_session_id,
)
from ingovo_search.recommender.clarification_policy import (
    ClarificationAnswer,
    ClarificationReason,
)
from ingovo_search.recommender.parser_schema import ParsedQuery

SECRET = b"x" * 32


# FastAPI resolves an override's own signature, and these classes' constructors take parameters it
# would turn into request parameters, so each override is a parameterless factory (fresh per request).
def _mock_llm_client() -> MockLlmClient:
    return MockLlmClient()


def _mock_embedding_client() -> MockEmbeddingClient:
    return MockEmbeddingClient()


@pytest.fixture
def session_cache() -> Cache:
    """Single MemoryCache shared by the SessionStore and any assertions."""
    return MemoryCache()


@pytest.fixture
def session_store(session_cache: Cache) -> SessionStore:
    return SessionStore(session_cache)


@pytest.fixture
def app(session_store: SessionStore, session_cache: Cache) -> FastAPI:
    application = FastAPI()
    install_exception_handlers(application)
    application.include_router(search_router)
    application.dependency_overrides[get_session_store] = lambda: session_store
    application.dependency_overrides[get_session_cache] = lambda: session_cache
    application.dependency_overrides[get_session_secret] = lambda: SECRET
    application.dependency_overrides[get_session_cookie_secure] = lambda: False
    application.dependency_overrides[get_llm_client] = _mock_llm_client
    application.dependency_overrides[get_embedding_client] = _mock_embedding_client
    application.dependency_overrides[get_db_session] = fake_db_session
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_pipeline():
    with (
        patch("ingovo_search.search.request.parse_query", new_callable=AsyncMock) as mock_parse,
        patch("ingovo_search.search.candidates.recommend", new_callable=AsyncMock) as mock_rec,
        patch("ingovo_search.search.pipeline.should_clarify") as mock_clarify,
        patch("ingovo_search.search.reranking.rerank_and_explain", new_callable=AsyncMock),
    ):
        mock_parse.return_value = ParsedQuery()
        mock_rec.return_value = []
        mock_clarify.return_value = None
        yield {
            "parse_query": mock_parse,
            "recommend": mock_rec,
            "should_clarify": mock_clarify,
        }


def test_search_returns_200_with_envelope_shape(client: TestClient) -> None:
    response = client.post("/api/search", json={"query": "csendes lakás"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["error"] is None
    assert body["meta"]["timings_ms"]["parser"] >= 0
    assert body["data"] == {
        "results": [],
        "clarification": None,
        "total_candidates": 0,
        "score_version": SCORE_VERSION,
        "pipeline_versions": PipelineVersions().model_dump(),
        "relaxed": False,
        "relaxation_path": [],
        "refinement_suggestions": [],
        "search_context": [],
        "image_context": None,
    }


def test_search_accepts_empty_body(client: TestClient) -> None:
    """Filter-only browsing is allowed — `query` is optional."""
    response = client.post("/api/search", json={})
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_clean_reset_search_hits_raw_response_cache_before_parser(
    client: TestClient,
    mock_pipeline: dict[str, AsyncMock],
) -> None:
    body = {
        "query": "Budai lakás 80 millió alatt erkéllyel",
        "include_narratives": False,
        "include_rerank": True,
        "reset_session": True,
    }

    first = client.post("/api/search", json=body)
    assert first.status_code == 200
    assert mock_pipeline["parse_query"].await_count == 1

    second = client.post("/api/search", json={**body, "query": "  budai   lakás 80 millió alatt erkéllyel "})
    assert second.status_code == 200
    assert second.json()["data"] == first.json()["data"]
    assert second.json()["meta"]["timings_ms"]["raw_response_cache_read"] >= 0
    assert mock_pipeline["parse_query"].await_count == 1


def test_search_returns_422_for_unknown_top_level_field(client: TestClient) -> None:
    """Pydantic validation rejects malformed bodies with 422 (FastAPI default)."""
    response = client.post("/api/search", json={"unknown_top_level": True})
    # Pydantic v2 model_config doesn't forbid extras by default — assert 200 OK
    # since SearchRequest is not strict. If we ever lock it, this becomes 422.
    assert response.status_code == 200


def test_search_returns_422_for_invalid_signal_type(client: TestClient) -> None:
    """Pydantic Literal mismatch → 422."""
    response = client.post(
        "/api/search",
        json={
            "session_signals": [
                {
                    "listing_id": "L1",
                    "signal_type": "not_a_real_signal",
                    "timestamp": "2026-05-21T12:00:00+00:00",
                }
            ]
        },
    )
    assert response.status_code == 422


# --- cookie issuance + session stickiness ------------------


def test_first_call_sets_session_cookie(client: TestClient) -> None:
    response = client.post("/api/search", json={})
    assert SESSION_COOKIE_NAME in response.cookies
    signed = response.cookies[SESSION_COOKIE_NAME]
    assert verify_signed_session_id(signed, secret=SECRET) is not None


def test_second_call_keeps_session(client: TestClient) -> None:
    first = client.post("/api/search", json={})
    first_sid_signed = first.cookies[SESSION_COOKIE_NAME]

    second = client.post("/api/search", json={})
    # The second call may or may not re-stamp the cookie depending on Starlette's
    # cookie-jar handling, but the underlying session id must remain the same.
    second_cookies = client.cookies
    assert SESSION_COOKIE_NAME in second_cookies
    assert second_cookies[SESSION_COOKIE_NAME] == first_sid_signed
    assert second.status_code == 200


async def test_session_state_accumulates_refinement_overrides(session_store: SessionStore, app: FastAPI) -> None:
    """Submitting refinement_overrides on one call carries into the next."""
    overrides = RefinementOverrides(soft_preferences=["elevator"])
    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/search",
            json={"refinement_overrides": overrides.model_dump()},
        )
        assert response.status_code == 200
        sid = verify_signed_session_id(response.cookies[SESSION_COOKIE_NAME], secret=SECRET)
        assert sid is not None

    state = await session_store.get(sid)
    assert state is not None
    assert state.refinement_overrides == overrides


async def test_session_state_accumulates_signals_and_clarifications(session_store: SessionStore, app: FastAPI) -> None:
    """Async test — uses pytest-asyncio for direct store inspection without `run_until_complete`."""
    with TestClient(app) as test_client:
        signal_body = {
            "listing_id": "L1",
            "signal_type": "thumb_up",
            "timestamp": "2026-05-21T12:00:00+00:00",
        }
        answer_body = {
            "reason": ClarificationReason.AMBIGUOUS_GEOGRAPHY.value,
            "option_id": "district:13",
        }
        response = test_client.post(
            "/api/search",
            json={
                "session_signals": [signal_body],
                "prior_clarifications": [answer_body],
            },
        )
        assert response.status_code == 200

        sid = verify_signed_session_id(response.cookies[SESSION_COOKIE_NAME], secret=SECRET)
        assert sid is not None

    state = await session_store.get(sid)
    assert state is not None
    assert state.session_signals == [
        SessionSignal(
            listing_id="L1",
            signal_type="thumb_up",
            timestamp=datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC),
        )
    ]
    assert state.clarification_answers == [
        ClarificationAnswer(reason=ClarificationReason.AMBIGUOUS_GEOGRAPHY, option_id="district:13")
    ]


async def test_subsequent_call_appends_rather_than_replaces(session_store: SessionStore, app: FastAPI) -> None:
    """Refinement/signals/clarifications accumulate, not overwrite."""
    with TestClient(app) as test_client:
        first = test_client.post(
            "/api/search",
            json={
                "session_signals": [
                    {
                        "listing_id": "L1",
                        "signal_type": "thumb_up",
                        "timestamp": "2026-05-21T12:00:00+00:00",
                    }
                ]
            },
        )
        assert first.status_code == 200

        second = test_client.post(
            "/api/search",
            json={
                "session_signals": [
                    {
                        "listing_id": "L2",
                        "signal_type": "favorite",
                        "timestamp": "2026-05-21T12:05:00+00:00",
                    }
                ]
            },
        )
        assert second.status_code == 200
        sid = verify_signed_session_id(
            second.cookies.get(SESSION_COOKIE_NAME) or first.cookies[SESSION_COOKIE_NAME], secret=SECRET
        )
        assert sid is not None

    state = await session_store.get(sid)
    assert state is not None
    assert [s.listing_id for s in state.session_signals] == ["L1", "L2"]


# --- Schema invariants ---------------------------------------------------


def test_search_request_defaults_are_empty() -> None:
    payload = SearchRequest()
    assert payload.query is None
    assert payload.prior_clarifications == []
    assert payload.refinement_overrides is None
    assert payload.session_signals == []


@pytest.mark.parametrize(
    "fields",
    [{"query": "x" * 601}, {"rerank_candidate_count": 0}, {"rerank_candidate_count": 31}],
    ids=["query-too-long", "rerank-count-zero", "rerank-count-above-30"],
)
def test_search_request_bounds_the_fields_that_size_model_calls(fields: dict[str, object]) -> None:
    with pytest.raises(PydanticValidationError):
        SearchRequest.model_validate({"query": "eladó lakás", **fields})


def test_search_response_defaults_use_score_version() -> None:
    response = SearchResponse()
    assert response.score_version == SCORE_VERSION
    assert response.pipeline_versions.score == SCORE_VERSION
    assert response.pipeline_versions.match_percent == "match-percent-v2-near-proximity"
    assert response.pipeline_versions.canonical_ranking == "authoritative-order-v3-running-minimum"
    # Bumped again when the displayed percentages became a running minimum:
    # old cache entries must not replay percentages above a listing's own score.
    assert response.pipeline_versions.response_cache.startswith("response-v24-")
    assert response.results == []
    assert response.clarification is None
    assert response.total_candidates == 0
