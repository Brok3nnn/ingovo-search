from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from tests._fakes import fake_db_session

from ingovo_search.api.deps import (
    get_db_session,
    get_embedding_client,
    get_llm_client,
    get_session_cache,
    get_session_store,
)
from ingovo_search.api.main import create_app
from ingovo_search.lib.cache import MemoryCache
from ingovo_search.lib.session import SessionStore
from ingovo_search.recommender.parser_schema import ParsedQuery


@pytest.fixture
def memory_cache() -> MemoryCache:
    return MemoryCache()


@pytest.fixture
def memory_session_store(memory_cache: MemoryCache) -> SessionStore:
    return SessionStore(memory_cache)


@pytest.fixture
def client(memory_session_store: SessionStore, memory_cache: MemoryCache) -> Any:
    app = create_app()
    app.dependency_overrides[get_session_store] = lambda: memory_session_store  # type: ignore
    app.dependency_overrides[get_session_cache] = lambda: memory_cache  # type: ignore
    app.dependency_overrides[get_db_session] = fake_db_session  # type: ignore
    app.dependency_overrides[get_llm_client] = lambda: AsyncMock()  # type: ignore  # noqa: PLW0108 -- FastAPI would read the class's own signature
    app.dependency_overrides[get_embedding_client] = lambda: AsyncMock()  # type: ignore  # noqa: PLW0108 -- FastAPI would read the class's own signature

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def mock_pipeline() -> Any:
    with (
        patch("ingovo_search.search.request.parse_query", new_callable=AsyncMock) as mock_parse,
        patch("ingovo_search.search.candidates.recommend", new_callable=AsyncMock) as mock_rec,
        patch("ingovo_search.search.pipeline.should_clarify") as mock_clarify,
    ):
        mock_parse.return_value = ParsedQuery()
        mock_rec.return_value = []
        mock_clarify.return_value = None
        yield (mock_parse, mock_rec, mock_clarify)


@pytest.mark.usefixtures("mock_pipeline")
def test_search_returns_200_and_sets_cookie(client: TestClient) -> None:
    """POST /api/search with valid body returns 200, sets cookie."""
    response = client.post("/api/search", json={"query": "eladó lakás budapest"})

    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "results" in data["data"]
    assert "set-cookie" in response.headers
    assert "ingovo_sid" in client.cookies


@pytest.mark.usefixtures("mock_pipeline")
def test_clarify_accumulates_in_session(client: TestClient) -> None:
    """Missing session cookie triggers Set-Cookie; subsequent calls accumulate session."""
    # Call 1: Start session
    resp1 = client.post("/api/search", json={"query": "eladó lakás"})
    assert resp1.status_code == 200
    assert "ingovo_sid" in client.cookies

    # Call 2: Clarify (uses the same client with the cookie)
    resp2 = client.post(
        "/api/search/clarify",
        json={"answer": {"reason": "conflicting_constraints", "option_id": "relax:price_huf_max"}},
    )
    assert resp2.status_code == 200

    # Call 3: Search again
    resp3 = client.post("/api/search", json={"query": "eladó lakás"})
    assert resp3.status_code == 200
