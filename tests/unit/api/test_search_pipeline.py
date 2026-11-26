"""§14e-3 — pipeline wiring tests for `POST /api/search`.

The acceptance tests in `test_search.py` stub the entire pipeline and only
check envelope shape + cookie + session accumulation. These tests verify the
*wiring* between the stubs:

1. parse → apply_clarification (per session answer) → apply refinements
2. cache miss → embed → recommend → should_clarify → rerank (when applicable)
3. rerank guardrails (hard-filter preservation, bounded movement clamp)
4. response: ScoredCandidate carries `rationale` merged from rerank
5. cache hit short-circuits the pipeline
6. fast-path skips rerank when `semantic_query` is empty
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
from ingovo_search.api.search import router as search_router
from ingovo_search.lib.cache import Cache, MemoryCache, embedding_cache_key, normalise_query, search_cache_key, set_json
from ingovo_search.lib.embedding_models import EMBED_MULTILINGUAL
from ingovo_search.lib.session import SessionStore, verify_signed_session_id
from ingovo_search.recommender.clarification_policy import (
    Clarification,
    ClarificationOption,
    ClarificationReason,
)
from ingovo_search.recommender.parser_schema import GeoHints, HardFilters, NearFilter, ParsedQuery
from ingovo_search.recommender.rerank import RerankItem, RerankResult
from ingovo_search.recommender.score import WEIGHTS, ScoreBreakdown, ScoredCandidate
from ingovo_search.recommender.search_contract import ResultScope
from ingovo_search.search.constants import RERANK_CANDIDATE_COUNT, SEARCH_RESPONSE_CACHE_VERSION
from ingovo_search.search.embedding_cache import CachedEmbedding
from ingovo_search.search.models import SCORE_VERSION, SearchResponse
from ingovo_search.search.observability import SearchObservation, log_search_observability
from ingovo_search.search.scopes import administrative_geo_city_scope
from ingovo_search.search.scoring import (
    build_scored_candidates,
    compromise_labels,
    public_property_features,
    should_rerank,
)

SECRET = b"x" * 32


# FastAPI resolves an override's own signature, and AsyncMock's `*args, **kwargs` would become required
# query parameters, so the override is a parameterless factory (a fresh mock per request).
def _fresh_async_mock() -> AsyncMock:
    return AsyncMock()


def test_administrative_geo_scope_excludes_out_of_area_settlements() -> None:
    pest_scope = administrative_geo_city_scope(ParsedQuery(geo_hints=GeoHints(county="Pest")))
    south_scope = administrative_geo_city_scope(ParsedQuery(geo_hints=GeoHints(balaton_shore="Déli")))

    assert pest_scope is not None
    assert "budaörs" in pest_scope
    assert "budapest" not in pest_scope
    assert south_scope is not None
    assert "siófok" in south_scope
    assert "balatonfüred" not in south_scope


# --- builders -------------------------------------------------------------


def _make_property(listing_id: str, **overrides: Any) -> SimpleNamespace:
    """Stub matching the fields `search.py` reads off a `Property` row."""
    defaults: dict[str, Any] = {
        "id": listing_id,
        "title": f"Listing {listing_id}",
        "description": "Csendes panel a XIII. kerületben.",
        "price_huf": 80_000_000,
        "property_type": "panel",
        "transaction_type": "sale",
        "area_m2": 50.0,
        "rooms": "2",
        "city": "Budapest",
        "district": "13",
        "address": "Példa utca 1.",
        "images": [],
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_breakdown(total: float) -> ScoreBreakdown:
    return ScoreBreakdown(
        content=0.0,
        factual=0.0,
        softpref_core=0.0,
        softpref_free=0.0,
        anchor=0.0,
        quality=0.0,
        geo=0.0,
        pooled=0.0,
        recency=0.0,
        explore=0.0,
        total=total,
    )


def _make_rerank_result(ids_in_order: list[str], *, low_match: bool = False) -> RerankResult:
    items = [
        RerankItem(listing_id=lid, rank=idx + 1, rationale_hu=f"rationale-{lid}")
        for idx, lid in enumerate(ids_in_order)
    ]
    return RerankResult(items=items, low_match=low_match, fast_path=False)


def _embedding(value: float = 0.1) -> list[float]:
    return [value] * 1024


def run_sync[T](coro: Awaitable[T]) -> T:
    """Run an awaitable to completion from a sync test body."""
    return asyncio.new_event_loop().run_until_complete(coro)


def _search_response_cache_key(
    parsed: ParsedQuery,
    *,
    user_signed_in: bool = False,
    include_rerank: bool = True,
) -> str:
    parsed_hash = hashlib.sha256(parsed.model_dump_json().encode("utf-8")).hexdigest()
    suffix = f":{SEARCH_RESPONSE_CACHE_VERSION}:fast:narr0:rerank{int(include_rerank)}"
    if include_rerank:
        suffix += f":rc{RERANK_CANDIDATE_COUNT}"
    return f"{search_cache_key(parsed_hash, user_signed_in=user_signed_in)}{suffix}"


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def session_cache() -> Cache:
    return MemoryCache()


@pytest.fixture
def session_store(session_cache: Cache) -> SessionStore:
    return SessionStore(session_cache)


@pytest.fixture
def embedding_client() -> AsyncMock:
    client = AsyncMock()
    client.embed.return_value = _embedding()
    return client


@pytest.fixture
def llm_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def app(
    session_store: SessionStore,
    session_cache: Cache,
    embedding_client: AsyncMock,
    llm_client: AsyncMock,
) -> FastAPI:
    application = FastAPI()
    install_exception_handlers(application)
    application.include_router(search_router)
    application.dependency_overrides[get_session_store] = lambda: session_store
    application.dependency_overrides[get_session_cache] = lambda: session_cache
    application.dependency_overrides[get_session_secret] = lambda: SECRET
    application.dependency_overrides[get_session_cookie_secure] = lambda: False
    application.dependency_overrides[get_llm_client] = lambda: llm_client
    application.dependency_overrides[get_embedding_client] = lambda: embedding_client
    application.dependency_overrides[get_db_session] = _fresh_async_mock
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# --- pipeline patch helper ------------------------------------------------


@dataclass
class PipelineMocks:
    """Handles to each patched pipeline call site."""

    parse: AsyncMock
    recommend: AsyncMock
    clarify: MagicMock
    rerank: AsyncMock
    apply_clarification: MagicMock
    visibility: AsyncMock


@contextmanager
def patched_pipeline(
    *,
    parsed: ParsedQuery,
    recommend_return: list[tuple[SimpleNamespace, ScoreBreakdown]],
    rerank_return: RerankResult | None,
    clarification: Clarification | None = None,
) -> Generator[PipelineMocks, None, None]:
    """Patch every pipeline call site `search.py` reaches into."""
    with (
        patch("ingovo_search.search.request.parse_query", new_callable=AsyncMock) as mock_parse,
        patch("ingovo_search.search.candidates.recommend", new_callable=AsyncMock) as mock_rec,
        patch("ingovo_search.search.pipeline.should_clarify") as mock_clarify,
        patch("ingovo_search.search.reranking.rerank_and_explain", new_callable=AsyncMock) as mock_rerank,
        patch("ingovo_search.search.pipeline.apply_clarification") as mock_apply,
        # Pipeline tests use an AsyncMock session, not a PostgreSQL database.
        # The cache authorization boundary itself is covered by the dedicated
        # visibility-failure test below and by test_sql_filter; ordinary cache
        # wiring tests must declare their synthetic IDs currently visible.
        patch(
            "ingovo_search.search.response_cache.cached_listing_ids_are_publicly_visible",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_visibility,
        # Conditional rerank (§4) is exercised by its own dedicated tests; here we
        # keep every semantic search rerank-eligible so the pipeline-flow tests
        # always reach the rerank step regardless of fixture size.
        patch("ingovo_search.search.reranking.should_rerank", return_value=True),
    ):
        mock_parse.return_value = parsed
        mock_rec.return_value = recommend_return
        mock_clarify.return_value = clarification
        if rerank_return is not None:
            mock_rerank.return_value = rerank_return

        def _identity(parsed_: ParsedQuery, _answer: Any) -> ParsedQuery:
            return parsed_

        mock_apply.side_effect = _identity
        yield PipelineMocks(
            parse=mock_parse,
            recommend=mock_rec,
            clarify=mock_clarify,
            rerank=mock_rerank,
            apply_clarification=mock_apply,
            visibility=mock_visibility,
        )


# --- pipeline data flow ---------------------------------------------------


def test_pipeline_calls_parse_then_recommend_then_rerank(client: TestClient) -> None:
    parsed = ParsedQuery(semantic_query="kétszobás panel XIII")
    props = [
        (_make_property("L1"), _make_breakdown(0.9)),
        (_make_property("L2"), _make_breakdown(0.8)),
    ]
    rerank = _make_rerank_result(["L2", "L1"])

    with patched_pipeline(parsed=parsed, recommend_return=props, rerank_return=rerank) as mocks:
        response = client.post("/api/search", json={"query": "kétszobás panel XIII"})

    assert response.status_code == 200
    mocks.parse.assert_awaited_once()
    mocks.recommend.assert_awaited_once()
    mocks.clarify.assert_called_once()
    mocks.rerank.assert_awaited_once()


def test_search_response_and_cache_include_map_locations(client: TestClient) -> None:
    props = [
        (_make_property("LOCATED", latitude=47.4351902, longitude=19.075689), _make_breakdown(0.9)),
        (_make_property("UNLOCATED"), _make_breakdown(0.8)),
    ]
    with patched_pipeline(parsed=ParsedQuery(), recommend_return=props, rerank_return=None) as mocks:
        first = client.post("/api/search", json={"query": "térképes találatok koordinátákkal"})
        cached = client.post("/api/search", json={"query": "térképes találatok koordinátákkal"})

    assert first.status_code == 200
    assert cached.status_code == 200
    mocks.recommend.assert_awaited_once()
    for response in [first, cached]:
        properties = {result["listing_id"]: result["property"] for result in response.json()["data"]["results"]}
        assert properties["LOCATED"]["latitude"] == 47.4351902
        assert properties["LOCATED"]["longitude"] == 19.075689
        assert properties["UNLOCATED"]["latitude"] is None
        assert properties["UNLOCATED"]["longitude"] is None


def test_parser_timeout_uses_deterministic_fallback(client: TestClient) -> None:
    async def _slow_parse(*_args: Any, **_kwargs: Any) -> ParsedQuery:
        await asyncio.sleep(0.05)
        return ParsedQuery(semantic_query="should not arrive")

    props = [
        (_make_property("L1", city="Győr", price_huf=42_000_000), _make_breakdown(0.9)),
        (_make_property("L2", city="Győr", price_huf=44_000_000), _make_breakdown(0.8)),
    ]

    with (
        patched_pipeline(
            parsed=ParsedQuery(semantic_query="unused"),
            recommend_return=props,
            rerank_return=None,
        ) as mocks,
        patch("ingovo_search.search.request.get_settings") as mock_settings,
    ):
        mock_settings.return_value.llm_daily_budget_usd = 25.0
        mock_settings.return_value.ai_search_rerank_rank_only = True
        mock_settings.return_value.ai_search_conditional_rerank = True
        mock_settings.return_value.ai_search_parser_timeout_ms = 1
        mock_settings.return_value.ai_search_rerank_timeout_ms = 700
        mocks.parse.side_effect = _slow_parse
        response = client.post(
            "/api/search",
            json={"query": "eladó lakás Győrben 45 millió alatt", "include_rerank": False},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["timings_ms"]["parser_timeout"] == 1
    assert body["data"]["results"][0]["property"]["city"] == "Győr"


def test_rank_only_setting_is_passed_to_rerank(client: TestClient) -> None:
    parsed = ParsedQuery(semantic_query="csendes napos lakás budán")
    props = [(_make_property(f"L{i}"), _make_breakdown(1.0 - i * 0.01)) for i in range(4)]
    rerank = _make_rerank_result(["L0", "L1", "L2", "L3"])

    with (
        patched_pipeline(parsed=parsed, recommend_return=props, rerank_return=rerank) as mocks,
        patch("ingovo_search.search.request.get_settings") as mock_settings,
    ):
        mock_settings.return_value.llm_daily_budget_usd = 25.0
        mock_settings.return_value.ai_search_rerank_rank_only = True
        mock_settings.return_value.ai_search_conditional_rerank = True
        mock_settings.return_value.ai_search_parser_timeout_ms = 500
        mock_settings.return_value.ai_search_rerank_timeout_ms = 1800
        response = client.post("/api/search", json={"query": "csendes napos lakás budán"})

    assert response.status_code == 200
    mocks.rerank.assert_awaited_once()
    # The fourth positional argument is the call's `RerankOptions`.
    assert mocks.rerank.await_args.args[3].rank_only is True


def test_rerank_timeout_falls_back_to_deterministic_order(client: TestClient) -> None:
    parsed = ParsedQuery(semantic_query="csendes napos lakás budán")
    props = [
        (_make_property("L1"), _make_breakdown(0.9)),
        (_make_property("L2"), _make_breakdown(0.8)),
        (_make_property("L3"), _make_breakdown(0.7)),
        (_make_property("L4"), _make_breakdown(0.6)),
    ]

    async def _slow_rerank(*_args: Any, **_kwargs: Any) -> RerankResult:
        await asyncio.sleep(0.05)
        return _make_rerank_result(["L4", "L3", "L2", "L1"])

    with (
        patched_pipeline(parsed=parsed, recommend_return=props, rerank_return=None) as mocks,
        patch("ingovo_search.search.request.get_settings") as mock_settings,
    ):
        mock_settings.return_value.llm_daily_budget_usd = 25.0
        mock_settings.return_value.ai_search_rerank_rank_only = True
        mock_settings.return_value.ai_search_conditional_rerank = True
        mock_settings.return_value.ai_search_parser_timeout_ms = 500
        mock_settings.return_value.ai_search_rerank_timeout_ms = 1
        mocks.rerank.side_effect = _slow_rerank
        response = client.post("/api/search", json={"query": "csendes napos lakás budán"})

    assert response.status_code == 200
    mocks.rerank.assert_awaited_once()
    body = response.json()
    assert [r["listing_id"] for r in body["data"]["results"]] == ["L1", "L2", "L3", "L4"]
    assert body["meta"]["timings_ms"]["rerank_timeout"] == 1


def test_response_carries_rerank_rationale(client: TestClient) -> None:
    parsed = ParsedQuery(semantic_query="kétszobás panel")
    props = [
        (_make_property("L1"), _make_breakdown(0.9)),
        (_make_property("L2"), _make_breakdown(0.5)),
    ]
    rerank = _make_rerank_result(["L2", "L1"])

    with patched_pipeline(parsed=parsed, recommend_return=props, rerank_return=rerank):
        response = client.post("/api/search", json={"query": "kétszobás panel"})

    body = response.json()
    results = body["data"]["results"]
    # Equal percentages use the bounded reranker relevance signal; rationale
    # remains attached to the correct candidate.
    assert [r["listing_id"] for r in results] == ["L2", "L1"]
    assert results[0]["rationale"] == "rationale-L2"
    assert results[1]["rationale"] == "rationale-L1"
    assert body["data"]["total_candidates"] == 2


def test_rerank_skipped_when_semantic_query_empty(client: TestClient) -> None:
    """§6.4 filter-only fast path: empty semantic_query → no LLM rerank."""
    parsed = ParsedQuery(semantic_query="")
    props = [(_make_property("L1"), _make_breakdown(0.9))]

    with patched_pipeline(parsed=parsed, recommend_return=props, rerank_return=None) as mocks:
        response = client.post("/api/search", json={})

    assert response.status_code == 200
    mocks.rerank.assert_not_awaited()
    body = response.json()
    assert [r["listing_id"] for r in body["data"]["results"]] == ["L1"]


# --- conditional rerank ------------------


def _scored(*totals: float) -> list[tuple[SimpleNamespace, ScoreBreakdown]]:
    return [(_make_property(f"L{i}"), _make_breakdown(total)) for i, total in enumerate(totals)]


def test_should_rerank_skips_thin_candidate_set() -> None:
    # Fewer than RERANK_MIN_CANDIDATES → deterministic order is trivially fine.
    assert should_rerank(ParsedQuery(semantic_query="kétszobás lakás"), _scored(0.9, 0.5)) is False


def test_should_rerank_skips_simple_query_with_clear_leader() -> None:
    # 2-word query, no soft prefs, top leads #2 by >12% → trust deterministic order.
    assert should_rerank(ParsedQuery(semantic_query="kétszobás lakás"), _scored(0.90, 0.60, 0.55, 0.50)) is False


def test_should_rerank_runs_on_tight_top_cluster() -> None:
    # Top and #2 within 12% → the reranker disambiguates, so it must run.
    assert should_rerank(ParsedQuery(semantic_query="kétszobás lakás"), _scored(0.90, 0.88, 0.85, 0.80)) is True


def test_should_rerank_runs_on_long_natural_language_query() -> None:
    # ≥4 semantic words signals rich intent → rerank even with a clear leader.
    parsed = ParsedQuery(semantic_query="csendes napos erkélyes lakás budán")
    assert should_rerank(parsed, _scored(0.90, 0.50, 0.40, 0.30)) is True


def test_conditional_rerank_skips_simple_search_in_pipeline(client: TestClient) -> None:
    """With the flag on and a clear deterministic leader, the pipeline serves the
    deterministic order and never calls the LLM reranker."""
    parsed = ParsedQuery(semantic_query="lakás")
    props = _scored(0.9, 0.5, 0.4, 0.3, 0.2)

    with (
        patched_pipeline(parsed=parsed, recommend_return=props, rerank_return=None) as mocks,
        patch("ingovo_search.search.reranking.should_rerank", return_value=False),
    ):
        response = client.post("/api/search", json={"query": "lakás", "include_rerank": True})

    assert response.status_code == 200
    mocks.rerank.assert_not_awaited()
    body = response.json()
    assert [r["listing_id"] for r in body["data"]["results"]] == ["L0", "L1", "L2", "L3", "L4"]
    assert body["data"]["results"][0]["rationale"] is None


def test_embed_skipped_when_semantic_query_empty(client: TestClient, embedding_client: AsyncMock) -> None:
    """Filter-only path must not pay for an embedding call."""
    parsed = ParsedQuery(semantic_query="")

    with patched_pipeline(parsed=parsed, recommend_return=[], rerank_return=None):
        response = client.post("/api/search", json={})

    assert response.status_code == 200
    embedding_client.embed.assert_not_awaited()


def test_embed_called_for_soft_prefs_freetext_only(client: TestClient, embedding_client: AsyncMock) -> None:
    """No semantic query but freetext prefs → one embed for the freetext, none for the query."""
    parsed = ParsedQuery(semantic_query="", soft_prefs_freetext=["jó otthonról dolgozni"])

    with patched_pipeline(parsed=parsed, recommend_return=[], rerank_return=None):
        response = client.post("/api/search", json={})

    assert response.status_code == 200
    assert embedding_client.embed.await_count == 1
    args, _ = embedding_client.embed.await_args
    assert args[0] == "jó otthonról dolgozni"


def test_semantic_embedding_is_cached_across_distinct_searches(
    client: TestClient,
    embedding_client: AsyncMock,
) -> None:
    """Same semantic text across two parsed queries should reuse the cached query embedding."""
    parsed_a = ParsedQuery(semantic_query="balatoni ingatlan", hard_filters=HardFilters(price_huf_min=1))
    parsed_b = ParsedQuery(semantic_query="balatoni ingatlan", hard_filters=HardFilters(price_huf_min=2))
    props = [(_make_property(f"L{i}"), _make_breakdown(1.0 - i * 0.01)) for i in range(12)]

    with patched_pipeline(parsed=parsed_a, recommend_return=props, rerank_return=None) as mocks:
        mocks.parse.side_effect = [parsed_a, parsed_b]
        first = client.post(
            "/api/search",
            json={"query": "balatoni ingatlan", "include_rerank": False, "reset_session": True},
        )
        second = client.post(
            "/api/search",
            json={"query": "Balatoni   ingatlan", "include_rerank": False, "reset_session": True},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert embedding_client.embed.await_count == 1
    args, _ = embedding_client.embed.await_args
    assert args == ("balatoni ingatlan", "search_query")


def test_speculative_cached_raw_embedding_reused_after_parser(
    client: TestClient,
    session_cache: Cache,
    embedding_client: AsyncMock,
) -> None:
    """A raw-query embedding cache hit can overlap parser latency and skip embed()."""
    query = "csendes napos lakás"
    parsed = ParsedQuery(semantic_query=query)
    cached_vector = _embedding(0.7)
    key = embedding_cache_key(normalise_query(query), model=EMBED_MULTILINGUAL, input_type="search_query")
    run_sync(set_json(session_cache, key, CachedEmbedding(embedding=cached_vector)))

    with patched_pipeline(parsed=parsed, recommend_return=[], rerank_return=None) as mocks:
        response = client.post("/api/search", json={"query": query, "include_rerank": False, "reset_session": True})

    assert response.status_code == 200
    embedding_client.embed.assert_not_awaited()
    assert mocks.recommend.await_args is not None
    assert mocks.recommend.await_args.kwargs["query_emb"] == cached_vector


def test_speculative_raw_embedding_ignored_when_parser_rewrites_semantic_query(
    client: TestClient,
    session_cache: Cache,
    embedding_client: AsyncMock,
) -> None:
    """If parser changes semantic_query, the speculative raw-query cache is not reused."""
    raw_query = "csendes napos lakás"
    parsed = ParsedQuery(semantic_query="világos lakás")
    key = embedding_cache_key(normalise_query(raw_query), model=EMBED_MULTILINGUAL, input_type="search_query")
    run_sync(set_json(session_cache, key, CachedEmbedding(embedding=_embedding(0.7))))

    with patched_pipeline(parsed=parsed, recommend_return=[], rerank_return=None):
        response = client.post("/api/search", json={"query": raw_query, "include_rerank": False, "reset_session": True})

    assert response.status_code == 200
    embedding_client.embed.assert_awaited_once_with("világos lakás", "search_query")


def test_rerank_skipped_when_no_candidates(client: TestClient) -> None:
    parsed = ParsedQuery(semantic_query="lakás")
    with patched_pipeline(parsed=parsed, recommend_return=[], rerank_return=None) as mocks:
        response = client.post("/api/search", json={"query": "lakás"})

    assert response.status_code == 200
    mocks.rerank.assert_not_awaited()
    assert response.json()["data"]["total_candidates"] == 0


def test_search_relaxes_until_first_page_has_twelve_results(client: TestClient) -> None:
    """A thin strict result set is preserved, then broadened to fill page one."""
    parsed = ParsedQuery(
        semantic_query="lakás",
        hard_filters=HardFilters(price_huf_max=50_000_000),
    )
    strict_props = [(_make_property(f"S{i}", price_huf=45_000_000), _make_breakdown(1.0 - i * 0.01)) for i in range(6)]
    relaxed_props = [
        (_make_property(f"R{i}", price_huf=70_000_000), _make_breakdown(1.0 - i * 0.01)) for i in range(14)
    ]

    with patched_pipeline(parsed=parsed, recommend_return=strict_props, rerank_return=None) as mocks:
        mocks.recommend.side_effect = [strict_props, relaxed_props]
        response = client.post(
            "/api/search",
            json={"query": "lakás", "include_rerank": False},
        )

    assert response.status_code == 200
    body = response.json()["data"]
    assert len(body["results"]) == 20
    assert body["total_candidates"] == 20
    assert [item["listing_id"] for item in body["results"][:6]] == [f"S{i}" for i in range(6)]
    assert body["relaxed"] is True
    assert body["relaxation_path"] == ["max ár"]
    assert body["search_context"] == [{"key": "price", "label": "Max. 50 M Ft", "relaxed": True}]
    assert body["results"][0]["compromise"] is None
    assert body["results"][6]["compromise"] == "A keretnél 20 M Ft-tal drágább"


def test_exact_city_keeps_exact_results_when_no_other_inventory_exists(client: TestClient) -> None:
    parsed = ParsedQuery(
        semantic_query="tiszafüredi lakás",
        hard_filters=HardFilters(city="Tiszafüred", city_id="ksh:29726"),
    )
    exact = [(_make_property(f"E{i}", city="Tiszafüred"), _make_breakdown(0.9 - i * 0.1)) for i in range(3)]

    with (
        patched_pipeline(parsed=parsed, recommend_return=exact, rerank_return=None) as mocks,
        patch(
            "ingovo_search.search.candidates.run_hard_filter", new_callable=AsyncMock, return_value=[]
        ) as hard_filter,
    ):
        response = client.post(
            "/api/search",
            json={"query": "csak tiszafüred", "include_rerank": False, "reset_session": True},
        )

    body = response.json()["data"]
    assert response.status_code == 200
    assert [item["listing_id"] for item in body["results"]] == ["E0", "E1", "E2"]
    assert body["total_candidates"] == 3
    assert body["clarification"] is None
    assert body["relaxation_path"] == []
    assert mocks.recommend.await_count == 1
    assert hard_filter.await_args is not None
    assert hard_filter.await_args.kwargs["city_scope"]


def test_thin_city_search_automatically_adds_nearest_available_city(client: TestClient) -> None:
    parsed = ParsedQuery(
        semantic_query="tiszafüredi lakás",
        hard_filters=HardFilters(city="Tiszafüred", city_id="ksh:29726"),
    )
    exact = [(_make_property("E1", city="Tiszafüred"), _make_breakdown(0.9))]
    nearest = [(_make_property("N1", city="Eger"), _make_breakdown(0.7))]

    with (
        patched_pipeline(parsed=parsed, recommend_return=exact, rerank_return=None) as mocks,
        patch("ingovo_search.search.candidates.ordered_distant_city_scope", return_value=("Poroszló", "Eger", "Győr")),
        patch("ingovo_search.search.candidates.run_hard_filter", new_callable=AsyncMock) as hard_filter,
    ):
        # No <=30 km inventory; the nearest-city prefix search checks Poroszló,
        # then finds the first eligible listing when Eger enters the scope.
        hard_filter.side_effect = [[], [], ["N1"]]
        mocks.recommend.side_effect = [exact, nearest]
        response = client.post(
            "/api/search",
            json={"query": "tiszafüredi lakás", "include_rerank": False, "reset_session": True},
        )

    body = response.json()["data"]
    assert response.status_code == 200
    assert [item["listing_id"] for item in body["results"]] == ["E1", "N1"]
    assert body["clarification"] is None
    assert body["relaxation_path"] == ["legközelebbi elérhető település"]
    assert body["search_context"] == [{"key": "location", "label": "Tiszafüred", "relaxed": True}]
    assert hard_filter.await_args is not None
    assert hard_filter.await_args.kwargs["city_scope"] == ("Poroszló", "Eger")


def test_verified_nearby_candidate_can_enter_second_pool(client: TestClient) -> None:
    parsed = ParsedQuery(
        semantic_query="budapesti lakás",
        hard_filters=HardFilters(city="Budapest", city_id="ksh:13578"),
    )
    exact = [(_make_property("E1", city="Budapest"), _make_breakdown(0.9))]
    nearby = [(_make_property("N1", city="Budaörs"), _make_breakdown(0.8))]

    with (
        patched_pipeline(parsed=parsed, recommend_return=exact, rerank_return=None) as mocks,
        patch(
            "ingovo_search.search.candidates.run_hard_filter", new_callable=AsyncMock, return_value=["N1"]
        ) as hard_filter,
    ):
        mocks.recommend.side_effect = [exact, nearby]
        response = client.post(
            "/api/search",
            json={"query": "budapest plusz közel", "include_rerank": False, "reset_session": True},
        )

    body = response.json()["data"]
    assert response.status_code == 200
    assert [item["listing_id"] for item in body["results"]] == ["E1", "N1"]
    assert [item["rank"] for item in body["results"]] == [1, 2]
    assert body["results"][0]["result_scope"] == "exact"
    assert body["results"][0]["distance_km"] is None
    assert body["results"][1]["result_scope"] == "nearby"
    assert 0 < body["results"][1]["distance_km"] <= 30
    assert body["results"][0]["match_percent"] > body["results"][1]["match_percent"]
    nearby_filter_call = hard_filter.await_args_list[0]
    assert "Budaörs" in nearby_filter_call.kwargs["city_scope"]
    assert "Győr" not in nearby_filter_call.kwargs["city_scope"]
    assert mocks.recommend.await_args.kwargs["candidate_ids"] == ["N1"]
    assert body["relaxation_path"] == []


def test_public_results_preserve_ranker_order_and_align_displayed_percent(client: TestClient) -> None:
    parsed = ParsedQuery(semantic_query="világos lakás")
    weak = _make_breakdown(0.9).model_copy(
        update={"content": WEIGHTS["content"] * 0.2, "factual": WEIGHTS["factual"] * 0.2}
    )
    strong = _make_breakdown(0.4).model_copy(
        update={"content": WEIGHTS["content"] * 0.9, "factual": WEIGHTS["factual"] * 0.9}
    )
    candidates = [(_make_property("LOW"), weak), (_make_property("HIGH"), strong)]

    with patched_pipeline(parsed=parsed, recommend_return=candidates, rerank_return=None):
        response = client.post(
            "/api/search",
            json={"query": "százalék szerinti sorrend", "include_rerank": False, "reset_session": True},
        )

    body = response.json()["data"]
    percentages = [item["match_percent"] for item in body["results"]]
    assert [item["listing_id"] for item in body["results"]] == ["LOW", "HIGH"]
    assert [item["rank"] for item in body["results"]] == [1, 2]
    assert percentages == sorted(percentages, reverse=True)


def test_explicit_location_approval_enables_farther_pool(client: TestClient) -> None:
    parsed = ParsedQuery(
        semantic_query="tiszafüredi lakás",
        hard_filters=HardFilters(city="Tiszafüred", city_id="ksh:29726"),
        allow_distant_location_expansion=True,
    )
    exact = [(_make_property("E1", city="Tiszafüred"), _make_breakdown(0.9))]
    distant = [(_make_property("D1", city="Győr"), _make_breakdown(0.7))]

    with (
        patched_pipeline(parsed=parsed, recommend_return=exact, rerank_return=None) as mocks,
        patch("ingovo_search.search.candidates.run_hard_filter", new_callable=AsyncMock) as hard_filter,
    ):
        hard_filter.side_effect = [[], ["D1"]]
        mocks.recommend.side_effect = [exact, distant]
        response = client.post(
            "/api/search",
            json={"query": "tiszafüred távolabb is", "include_rerank": False, "reset_session": True},
        )

    body = response.json()["data"]
    assert response.status_code == 200
    assert [item["listing_id"] for item in body["results"]] == ["E1", "D1"]
    assert body["relaxation_path"] == ["távolabbi terület (jóváhagyva)"]
    assert all(step not in {"város", "kerület"} for step in body["relaxation_path"])
    expanded_filters = hard_filter.await_args_list[-1].args[1]
    assert expanded_filters.city is None
    assert expanded_filters.city_id is None


def test_clarification_returned_when_should_clarify_fires(client: TestClient) -> None:
    parsed = ParsedQuery()
    clarif = Clarification(
        reason=ClarificationReason.AMBIGUOUS_GEOGRAPHY,
        question="Pontosíts!",
        options=[ClarificationOption(id="district:13", label="13. kerület")],
    )
    with patched_pipeline(parsed=parsed, recommend_return=[], rerank_return=None, clarification=clarif):
        response = client.post("/api/search", json={})

    body = response.json()
    assert body["data"]["clarification"] is not None
    assert body["data"]["clarification"]["reason"] == "ambiguous_geography"


# --- guardrails -----------------------------------------------------------


def test_hard_filter_violation_returns_500_envelope(client: TestClient) -> None:
    """Rerank surfacing a listing id absent from `pre_order` must fail loud."""
    parsed = ParsedQuery(semantic_query="lakás")
    props = [
        (_make_property("L1"), _make_breakdown(0.9)),
        (_make_property("L2"), _make_breakdown(0.8)),
    ]
    rerank = _make_rerank_result(["L2", "LX"])  # LX never appeared in recommend()

    with patched_pipeline(parsed=parsed, recommend_return=props, rerank_return=rerank):
        response = client.post("/api/search", json={"query": "lakás"})

    assert response.status_code == 500
    body = response.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "ranking_guardrail"


def test_bounded_rerank_clamps_movement_beyond_max_delta(client: TestClient) -> None:
    """A listing moving > MAX_RERANK_DELTA positions is pulled back; output dedupes."""
    parsed = ParsedQuery(semantic_query="lakás")
    pre_ids = [f"L{i}" for i in range(12)]
    props = [(_make_property(lid), _make_breakdown(1.0 - i * 0.01)) for i, lid in enumerate(pre_ids)]
    # Move L11 from index 11 to index 0 (delta=-11, beyond MAX_RERANK_DELTA=10).
    reordered = ["L11", *pre_ids[:11]]
    rerank = _make_rerank_result(reordered)

    with patched_pipeline(parsed=parsed, recommend_return=props, rerank_return=rerank):
        response = client.post("/api/search", json={"query": "lakás"})

    assert response.status_code == 200
    result_ids = [r["listing_id"] for r in response.json()["data"]["results"]]
    # L11 should be clamped back from the top, not moved beyond the cap.
    assert result_ids[0] != "L11"
    assert "L11" in result_ids
    assert sorted(result_ids) == sorted(pre_ids)


def test_empty_rerank_order_falls_back_to_deterministic_results(client: TestClient) -> None:
    """An empty/invalid rerank order must not erase otherwise valid candidates."""
    parsed = ParsedQuery(semantic_query="lakás")
    pre_ids = [f"L{i}" for i in range(3)]
    props = [(_make_property(lid), _make_breakdown(1.0 - i * 0.01)) for i, lid in enumerate(pre_ids)]
    rerank = _make_rerank_result([])

    with patched_pipeline(parsed=parsed, recommend_return=props, rerank_return=rerank):
        response = client.post("/api/search", json={"query": "lakás"})

    assert response.status_code == 200
    result_ids = [r["listing_id"] for r in response.json()["data"]["results"]]
    assert result_ids == pre_ids


@pytest.mark.parametrize("candidate_count", [12, 20])
def test_partial_rerank_preserves_every_pipeline_candidate(
    client: TestClient,
    candidate_count: int,
) -> None:
    parsed = ParsedQuery(semantic_query="sok jelölt biztonságos rerank")
    pre_ids = [f"P{i:02d}" for i in range(candidate_count)]
    props = [(_make_property(listing_id), _make_breakdown(0.9)) for listing_id in pre_ids]
    rerank = _make_rerank_result(list(reversed(pre_ids[:10])))

    with patched_pipeline(parsed=parsed, recommend_return=props, rerank_return=rerank):
        response = client.post(
            "/api/search",
            json={"query": f"részleges rerank {candidate_count}", "reset_session": True},
        )

    assert response.status_code == 200
    results = response.json()["data"]["results"]
    assert len(results) == candidate_count
    assert {item["listing_id"] for item in results} == set(pre_ids)
    assert [item["rank"] for item in results] == list(range(1, candidate_count + 1))
    assert [item["match_percent"] for item in results] == sorted(
        [item["match_percent"] for item in results], reverse=True
    )


# --- cache behaviour ------------------------------------------------------


def test_cache_miss_writes_response(client: TestClient, session_cache: Cache) -> None:
    """First call populates the §13 search cache under the parsed-query hash."""
    parsed = ParsedQuery(semantic_query="kétszobás panel")
    props = [(_make_property("L1"), _make_breakdown(0.9))]
    rerank = _make_rerank_result(["L1"])

    with patched_pipeline(parsed=parsed, recommend_return=props, rerank_return=rerank):
        response = client.post("/api/search", json={"query": "kétszobás panel"})

    assert response.status_code == 200

    raw = run_sync(session_cache.get(_search_response_cache_key(parsed)))
    assert raw is not None  # cache write happened


def test_cache_hit_short_circuits_pipeline(client: TestClient, session_cache: Cache) -> None:
    """Pre-populating the cache makes the second call skip recommend/rerank."""
    parsed = ParsedQuery(semantic_query="kétszobás panel")
    cached = SearchResponse(results=[], clarification=None, total_candidates=0, score_version=SCORE_VERSION)
    run_sync(set_json(session_cache, _search_response_cache_key(parsed), cached))

    with patched_pipeline(parsed=parsed, recommend_return=[], rerank_return=None) as mocks:
        response = client.post("/api/search", json={"query": "kétszobás panel"})

    assert response.status_code == 200
    # parse_query still runs (cache key derives from its output), but recommend / rerank must not.
    mocks.recommend.assert_not_awaited()
    mocks.rerank.assert_not_awaited()


def test_nonempty_cache_hit_preserves_canonical_order_and_public_fields(client: TestClient) -> None:
    parsed = ParsedQuery(semantic_query="cache sorrend")
    weak = _make_breakdown(0.9).model_copy(update={"content": WEIGHTS["content"] * 0.2})
    strong = _make_breakdown(0.4).model_copy(update={"content": WEIGHTS["content"] * 0.9})
    props = [(_make_property("LOW"), weak), (_make_property("HIGH"), strong)]

    with patched_pipeline(parsed=parsed, recommend_return=props, rerank_return=None) as mocks:
        first = client.post("/api/search", json={"query": "cache sorrend", "include_rerank": False})
        second = client.post("/api/search", json={"query": "cache sorrend", "include_rerank": False})

    assert first.status_code == second.status_code == 200
    first_data = first.json()["data"]
    second_data = second.json()["data"]
    assert second_data == first_data
    assert [item["listing_id"] for item in second_data["results"]] == ["LOW", "HIGH"]
    assert [item["rank"] for item in second_data["results"]] == [1, 2]
    assert mocks.recommend.await_count == 1

    round_trip = SearchResponse.model_validate_json(SearchResponse.model_validate(second_data).model_dump_json())
    assert [item.rank for item in round_trip.results] == [1, 2]
    assert all(item.result_scope.value == "exact" for item in round_trip.results)


def test_cache_hit_visibility_recheck_fails_closed_and_rebuilds(client: TestClient, session_cache: Cache) -> None:
    """An expired/hidden cached listing is never replayed just because its TTL remains."""
    parsed = ParsedQuery(semantic_query="lejárt cache")
    stale = ScoredCandidate(
        listing_id="STALE",
        rank=1,
        score=0.9,
        match_percent=91,
        breakdown=_make_breakdown(0.9),
    )
    run_sync(
        set_json(
            session_cache,
            _search_response_cache_key(parsed, include_rerank=False),
            SearchResponse(results=[stale], clarification=None, total_candidates=1, score_version=SCORE_VERSION),
        )
    )
    fresh = [(_make_property("FRESH"), _make_breakdown(0.8))]

    with patched_pipeline(parsed=parsed, recommend_return=fresh, rerank_return=None) as mocks:
        mocks.visibility.return_value = False
        response = client.post("/api/search", json={"query": "lejárt cache", "include_rerank": False})

    assert response.status_code == 200
    assert [item["listing_id"] for item in response.json()["data"]["results"]] == ["FRESH"]
    mocks.visibility.assert_awaited_once()
    mocks.recommend.assert_awaited_once()


def test_old_response_cache_version_is_not_served(
    client: TestClient,
    session_cache: Cache,
) -> None:
    parsed = ParsedQuery(semantic_query="régi cache")
    parsed_hash = hashlib.sha256(parsed.model_dump_json().encode("utf-8")).hexdigest()
    old_key = (
        f"{search_cache_key(parsed_hash, user_signed_in=False)}"
        ":response-v13-search-context-compromise:fast:narr0:rerank0"
    )
    run_sync(set_json(session_cache, old_key, SearchResponse()))
    props = [(_make_property("NEW"), _make_breakdown(0.8))]

    with patched_pipeline(parsed=parsed, recommend_return=props, rerank_return=None) as mocks:
        response = client.post("/api/search", json={"query": "régi cache", "include_rerank": False})

    assert response.status_code == 200
    assert [item["listing_id"] for item in response.json()["data"]["results"]] == ["NEW"]
    mocks.recommend.assert_awaited_once()


def test_cache_hit_rejects_non_monotonic_match_order(client: TestClient, session_cache: Cache) -> None:
    parsed = ParsedQuery(semantic_query="hibás cache sorrend")
    low = ScoredCandidate(
        listing_id="LOW",
        rank=1,
        score=0.1,
        match_percent=75,
        breakdown=_make_breakdown(0.1),
    )
    high = ScoredCandidate(
        listing_id="HIGH",
        rank=2,
        score=0.9,
        match_percent=84,
        breakdown=_make_breakdown(0.9),
    )
    run_sync(
        set_json(
            session_cache,
            _search_response_cache_key(parsed),
            SearchResponse(results=[low, high], total_candidates=2),
        )
    )

    with patched_pipeline(parsed=parsed, recommend_return=[], rerank_return=None) as mocks:
        response = client.post("/api/search", json={"query": "hibás cache sorrend"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "ranking_guardrail"
    mocks.recommend.assert_not_awaited()


# --- clarification + refinement application -------------------------------


def test_prior_clarifications_applied_to_parsed_query(client: TestClient) -> None:
    """Every answer in session_state.clarification_answers is fed through apply_clarification."""
    parsed = ParsedQuery()
    answer_body = {
        "reason": ClarificationReason.AMBIGUOUS_GEOGRAPHY.value,
        "option_id": "district:13",
    }

    with patched_pipeline(parsed=parsed, recommend_return=[], rerank_return=None) as mocks:
        first = client.post("/api/search", json={"prior_clarifications": [answer_body]})
        assert first.status_code == 200
        mocks.apply_clarification.reset_mock()
        # Second call should re-apply the accumulated answer.
        second = client.post("/api/search", json={})
        assert second.status_code == 200
        assert mocks.apply_clarification.call_count == 1


def test_refinement_overrides_applied_after_session_clarifications(client: TestClient) -> None:
    """refinement_overrides supplied in payload persist and mutate the parsed query."""
    parsed = ParsedQuery(hard_filters=HardFilters(district_in=["12"]))

    with patched_pipeline(parsed=parsed, recommend_return=[], rerank_return=None) as mocks:
        response = client.post(
            "/api/search",
            json={"refinement_overrides": {"hard_filters": {"district_in": ["13"]}}},
        )

    assert response.status_code == 200
    # recommend was called with the refinement-merged parsed query on the FIRST
    # invocation. (M3 relaxation may re-call recommend with widened filters
    # when the first call returns []; those secondary calls would intentionally
    # have an empty district_in, so we pin the assertion to the initial call.)
    assert mocks.recommend.await_args_list, "recommend was never called"
    parsed_arg: ParsedQuery = mocks.recommend.await_args_list[0].args[1]  # pyright: ignore[reportAny]
    assert parsed_arg.hard_filters.district_in == ["13"]


# --- /api/search/clarify (§14e-4) ----------------------------------------


def test_clarify_returns_400_when_no_prior_search(client: TestClient) -> None:
    """Without a prior /search call there is no parsed query in session to re-run."""
    response = client.post(
        "/api/search/clarify",
        json={"answer": {"reason": "ambiguous_geography", "option_id": "district:13"}},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "no_prior_search"


def test_clarify_reruns_pipeline_with_appended_answer(client: TestClient) -> None:
    """After a /search call, /search/clarify re-runs the pipeline with the new answer.

    The new answer has to actually mutate the parsed query — otherwise the cache
    key matches the first call and `recommend` is correctly skipped on cache hit.
    Override `apply_clarification` to flip `hard_filters.district_in` so the
    second call produces a distinct cache key.
    """
    parsed = ParsedQuery(semantic_query="kétszobás panel")
    props = [(_make_property("L1"), _make_breakdown(0.9))]
    rerank = _make_rerank_result(["L1"])

    def _mutating_apply(p: ParsedQuery, _answer: Any) -> ParsedQuery:
        # Simulate "district:13" landing on hard_filters.district_in.
        return p.model_copy(update={"hard_filters": p.hard_filters.model_copy(update={"district_in": ["13"]})})

    with patched_pipeline(parsed=parsed, recommend_return=props, rerank_return=rerank) as mocks:
        first = client.post("/api/search", json={"query": "kétszobás panel"})
        assert first.status_code == 200

        # Switch apply_clarification to mutate, count only the post-/search calls.
        mocks.apply_clarification.side_effect = _mutating_apply
        mocks.recommend.reset_mock()
        mocks.parse.reset_mock()
        mocks.apply_clarification.reset_mock()

        clarify = client.post(
            "/api/search/clarify",
            json={"answer": {"reason": "ambiguous_geography", "option_id": "district:13"}},
        )

    assert clarify.status_code == 200
    body = clarify.json()
    assert body["ok"] is True
    assert body["data"]["score_version"] is not None
    mocks.parse.assert_not_awaited()  # clarify never re-parses; uses session-stored parsed query
    # The clarified search must re-run recommendation; if that result is thin,
    # the first-page top-up relaxation may call recommend again.
    assert mocks.recommend.await_count >= 1
    mocks.apply_clarification.assert_called_once()  # exactly the newly-appended answer


def test_clarify_persists_answer_in_session(client: TestClient, session_store: SessionStore) -> None:
    """The answer posted to /clarify accumulates in state.clarification_answers."""
    parsed = ParsedQuery(semantic_query="kétszobás panel")

    with patched_pipeline(parsed=parsed, recommend_return=[], rerank_return=None):
        first = client.post("/api/search", json={"query": "kétszobás panel"})
        assert first.status_code == 200

        clarify = client.post(
            "/api/search/clarify",
            json={"answer": {"reason": "ambiguous_geography", "option_id": "district:13"}},
        )
        assert clarify.status_code == 200

    sid = verify_signed_session_id(client.cookies[SESSION_COOKIE_NAME], secret=SECRET)
    assert sid is not None
    state = run_sync(session_store.get(sid))
    assert state is not None
    assert [a.option_id for a in state.clarification_answers] == ["district:13"]


def test_clarify_keeps_session_cookie_continuous(client: TestClient) -> None:
    """The same session cookie is used across /search and /search/clarify."""
    parsed = ParsedQuery(semantic_query="kétszobás panel")

    with patched_pipeline(parsed=parsed, recommend_return=[], rerank_return=None):
        first = client.post("/api/search", json={"query": "kétszobás panel"})
        first_sid = first.cookies[SESSION_COOKIE_NAME]
        assert first.status_code == 200

        clarify = client.post(
            "/api/search/clarify",
            json={"answer": {"reason": "ambiguous_geography", "option_id": "district:13"}},
        )
        assert clarify.status_code == 200

    # The clarify response may or may not restamp the cookie, but the persisted
    # session id on the client must be unchanged.
    assert client.cookies[SESSION_COOKIE_NAME] == first_sid


def test_search_observability_log_carries_scope_and_location() -> None:
    """R11.1-R11.5 — one structured line with location, scope counts, nearby distances.

    Asserts against the module logger directly (not structlog's global capture),
    because the heavily-used `_search_logger` proxy may already be cached by the
    time this runs in the full suite, which would defeat `capture_logs`.
    """
    parsed = ParsedQuery(
        hard_filters=HardFilters(
            city="Budapest",
            city_id="ksh:13578",
            near=NearFilter(lat=47.4979, lng=19.0545, radius_km=2.0),
        )
    )
    results = [
        ScoredCandidate(
            listing_id="A",
            rank=1,
            score=0.9,
            match_percent=96,
            breakdown=_make_breakdown(0.9),
            result_scope=ResultScope.EXACT,
        ),
        ScoredCandidate(
            listing_id="B",
            rank=2,
            score=0.5,
            match_percent=66,
            breakdown=_make_breakdown(0.5),
            result_scope=ResultScope.NEARBY,
            distance_km=1.8,
        ),
    ]

    mock_logger = MagicMock()
    log_search_observability(
        parsed,
        SearchObservation(
            results=results, total_candidates=5, relaxation_path=[], rerank_pre_count=12, rerank_post_count=12
        ),
        logger=mock_logger,
    )

    mock_logger.info.assert_called_once()
    assert mock_logger.info.call_args.args[0] == "search.observability"
    fields = mock_logger.info.call_args.kwargs
    assert fields["location"]["city"] == "Budapest"
    assert fields["location"]["city_id"] == "ksh:13578"
    assert fields["location"]["near"] is True
    assert fields["scope_counts"] == {"exact": 1, "nearby": 1, "out_of_scope": 0}
    assert fields["nearby_distances_km"] == [1.8]
    assert fields["excluded_count"] == 3
    assert fields["rerank_pre_count"] == 12
    assert fields["match_order_ok"] is True


def test_remote_settlement_inclusion_emits_alert_and_raises() -> None:
    """R11.7 — an out-of-scope settlement reaching public scoring alerts + raises."""
    parsed = ParsedQuery(hard_filters=HardFilters(city="Tiszafüred", city_id="ksh:29726"))
    far = _make_property("FAR", city="Győr", district=None)
    scored = [(far, _make_breakdown(0.5))]

    with (
        patch("ingovo_search.search.scoring.search_logger") as mock_logger,
        pytest.raises(ValueError, match="out-of-scope candidate"),
    ):
        build_scored_candidates(scored, parsed, parsed)

    mock_logger.error.assert_called_once()
    assert mock_logger.error.call_args.args[0] == "alert.remote_settlement_included"
    assert mock_logger.error.call_args.kwargs["listing_id"] == "FAR"


def test_public_search_card_projects_an_unsafe_canonical_price_without_float_rounding() -> None:
    prop = _make_property(
        "PRECISE",
        price_huf=9_007_199_254_740_993,
        currency="HUF",
    )

    candidate = build_scored_candidates([(prop, _make_breakdown(0.8))], ParsedQuery())[0]

    assert candidate.property is not None
    assert candidate.property["price_huf"] == "9007199254740993"
    assert candidate.property["price"] is None


def test_public_property_features_forwards_card_amenities() -> None:
    """Result cards receive the enriched flags that drive their feature chips."""
    prop = _make_property(
        "FEATURED",
        features={"parking": "saját beálló"},
        has_terrace=True,
        has_parking=True,
        has_garage=False,
    )

    assert public_property_features(prop) == {
        "parking": "saját beálló",
        "has_terrace": True,
        "has_parking": True,
        "has_garage": False,
    }


def test_compromise_labels_lists_every_unmet_requested_condition() -> None:
    """Card flags: one entry per requested condition the listing does not satisfy."""
    parsed = ParsedQuery(
        hard_filters=HardFilters(rooms=6, price_huf_max=80_000_000, property_type_in=["Lakás"]),
        soft_preferences=["elevator", "parking", "terrace", "renovated"],
    )
    prop = _make_property(
        "X",
        property_type="Lakás",  # requested type matches → no flag
        rooms="3",  # 6 requested → "3 szobával kevesebb"
        price_huf=95_000_000,  # over 80M → "…drágább"
        elevator=False,  # confirmed missing → "Nincs lift"
        has_parking=None,  # unknown → no flag (an unverified compromise is never asserted)
        has_terrace=True,  # present → no flag
        renovation_state="original",  # → "Felújítatlan"
    )

    labels = compromise_labels(prop, parsed)

    assert "3 szobával kevesebb" in labels
    assert any("drágább" in label for label in labels)
    assert "Nincs lift" in labels
    assert "Felújítatlan" in labels
    # Unknown amenity data is never surfaced as a compromise — only a definite
    # absence ("Nincs X") counts, so a null `has_parking` produces no flag.
    assert all("parkol" not in label.lower() for label in labels)
    # A satisfied requested amenity (terrace) and a matching type never produce a flag.
    assert all("terasz" not in label.lower() for label in labels)
    assert all("típus" not in label.lower() for label in labels)


def test_compromise_labels_only_flag_proven_directional_mismatches() -> None:
    """Regression: a matching transaction or unknown type must never be flagged."""
    rent_query = ParsedQuery(hard_filters=HardFilters(transaction_type="rent", property_type_in=["Lakás"]))

    # A rent listing for a rent search → no transaction flag at all.
    rent_match = _make_property("A", transaction_type="rent", property_type="Lakás")
    assert compromise_labels(rent_match, rent_query) == []

    # A sale listing for a rent search → directional, human-readable flag.
    sale_listing = _make_property("B", transaction_type="sale", property_type="Lakás")
    assert "Eladó (kiadó helyett)" in compromise_labels(sale_listing, rent_query)

    # Unknown property type → no "nincs igazolva" noise (we can't prove a mismatch).
    unknown_type = _make_property("C", transaction_type="rent", property_type=None)
    assert compromise_labels(unknown_type, rent_query) == []

    # A different, *known* type → "Ház (lakás helyett)".
    house = _make_property("D", transaction_type="rent", property_type="Ház")
    assert "Ház (lakás helyett)" in compromise_labels(house, rent_query)
