"""The search-pipeline golden harness.

Builds the FastAPI search app with the same dependency overrides the unit tests use and
patches the pipeline's four call sites (parser, recommender, clarification, rerank) with
deterministic doubles, so ``POST /api/search`` exercises the orchestration of
``execute_search_request`` and ``run_search_pipeline`` on a fixed, LLM-free path.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests._fakes import fake_db_session

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
from ingovo_search.api.search import router as search_router
from ingovo_search.lib.cache import MemoryCache
from ingovo_search.lib.session import SessionStore
from ingovo_search.recommender.clarification_policy import Clarification
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.rerank import RerankItem, RerankResult
from ingovo_search.recommender.score import ScoreBreakdown

SECRET = b"golden-session-secret-at-least-32-chars"


def build_client() -> TestClient:
    """A fresh app per scenario: the response cache must not leak between cases."""
    application = FastAPI()
    install_exception_handlers(application)
    application.include_router(search_router)
    cache = MemoryCache()
    store = SessionStore(cache)
    embedding_client = AsyncMock()
    embedding_client.embed.return_value = [0.1] * 1024
    application.dependency_overrides[get_session_store] = lambda: store
    application.dependency_overrides[get_session_cache] = lambda: cache
    application.dependency_overrides[get_session_secret] = lambda: SECRET
    application.dependency_overrides[get_session_cookie_secure] = lambda: False
    # FastAPI reads the override's signature; a bare `AsyncMock` class would become query params.
    application.dependency_overrides[get_llm_client] = lambda: AsyncMock()  # noqa: PLW0108
    application.dependency_overrides[get_embedding_client] = lambda: embedding_client
    application.dependency_overrides[get_db_session] = fake_db_session
    return TestClient(application)


def make_property(listing_id: str, **overrides: Any) -> SimpleNamespace:
    """The fields ``search.py`` reads off a ``Property`` row (canonical columns)."""
    defaults: dict[str, Any] = {
        "id": listing_id,
        "title": f"Listing {listing_id}",
        "description": "Csendes panel a XIII. kerületben.",
        "price_huf": 80_000_000,
        "currency": "HUF",
        "property_type": "panel",
        "transaction_type": "sale",
        "area_m2": 50.0,
        "rooms": "2",
        "city": "Budapest",
        "district": "13",
        "address": "Példa utca 1.",
        "images": [],
        "latitude": None,
        "longitude": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_breakdown(total: float, **parts: float) -> ScoreBreakdown:
    names = (
        "content",
        "factual",
        "softpref_core",
        "softpref_free",
        "anchor",
        "quality",
        "geo",
        "pooled",
        "recency",
        "explore",
    )
    values: dict[str, float] = {name: 0.0 for name in names}
    values.update(parts)
    return ScoreBreakdown(total=total, **values)


def make_rerank(ids_in_order: list[str], *, low_match: bool = False, fast_path: bool = False) -> RerankResult:
    items = [
        RerankItem(listing_id=lid, rank=idx + 1, rationale_hu=f"rationale-{lid}")
        for idx, lid in enumerate(ids_in_order)
    ]
    return RerankResult(items=items, low_match=low_match, fast_path=fast_path)


@contextmanager
def patched_pipeline(
    *,
    parsed: ParsedQuery,
    recommend_return: list[tuple[SimpleNamespace, ScoreBreakdown]],
    rerank_return: RerankResult | None,
    clarification: Clarification | None = None,
    should_rerank: bool = True,
) -> Generator[None, None, None]:
    with (
        patch("ingovo_search.search.request.parse_query", new_callable=AsyncMock) as mock_parse,
        patch("ingovo_search.search.candidates.recommend", new_callable=AsyncMock) as mock_rec,
        patch("ingovo_search.search.pipeline.should_clarify") as mock_clarify,
        patch("ingovo_search.search.reranking.rerank_and_explain", new_callable=AsyncMock) as mock_rerank,
        patch("ingovo_search.search.pipeline.apply_clarification") as mock_apply,
        patch(
            "ingovo_search.search.response_cache.cached_listing_ids_are_publicly_visible",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch("ingovo_search.search.reranking.should_rerank", return_value=should_rerank),
    ):
        mock_parse.return_value = parsed
        mock_rec.return_value = recommend_return
        mock_clarify.return_value = clarification
        if rerank_return is not None:
            mock_rerank.return_value = rerank_return

        def _identity(parsed_: ParsedQuery, _answer: Any) -> ParsedQuery:
            return parsed_

        mock_apply.side_effect = _identity
        yield
