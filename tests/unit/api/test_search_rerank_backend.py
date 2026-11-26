"""`AI_SEARCH_RERANK_BACKEND` wiring in `POST /api/search`.

The Gemini reranker is patched out and counted; the Jev path runs the real
`jev_rerank` against a `FakeJevClient`, so the search guardrails see a genuine
Jev result. Since 2026-09-28 `jev` means Jev first with the Gemini rerank as the
fallback when Jev cannot serve; the last section covers that.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
from ingovo_search.lib.budget_guard import BudgetPolicy
from ingovo_search.lib.cache import Cache, MemoryCache
from ingovo_search.lib.jev import FakeJevClient
from ingovo_search.lib.session import SessionStore
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.ranking_guardrails import enforce_bounded_rerank
from ingovo_search.recommender.rerank import RerankItem, RerankResult
from ingovo_search.recommender.score import ScoreBreakdown
from ingovo_search.search import reranking as reranking_module
from ingovo_search.search.reranking import serving_rerank_backend
from ingovo_search.search.response_cache import rerank_cache_suffix
from ingovo_search.types.errors import JevError

SECRET = b"x" * 32
QUERY = "csendes, erkélyes lakás Budán"


# FastAPI resolves an override's own signature, and AsyncMock's `*args, **kwargs` would become required
# query parameters, so the override is a parameterless factory (a fresh mock per request).
def _fresh_async_mock() -> AsyncMock:
    return AsyncMock()


def _property(listing_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=listing_id,
        title=f"Listing {listing_id}",
        description="Csendes budai lakás erkéllyel.",
        price_huf=80_000_000,
        property_type="Lakás",
        transaction_type="sale",
        area_m2=50.0,
        rooms="2",
        city="Budapest",
        district="2",
        address="Példa utca 1.",
        images=[],
    )


def _breakdown(total: float) -> ScoreBreakdown:
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


def _scored(n: int) -> list[tuple[SimpleNamespace, ScoreBreakdown]]:
    return [(_property(f"L{index}"), _breakdown(0.9 - index * 0.01)) for index in range(n)]


def _gemini_result(ids: list[str]) -> RerankResult:
    return RerankResult(items=[RerankItem(listing_id=lid, rank=i + 1) for i, lid in enumerate(ids)])


@pytest.fixture
def session_cache() -> Cache:
    return MemoryCache()


@pytest.fixture(autouse=True)
def reset_jev_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cooldown is process state; every test starts with Jev available."""
    monkeypatch.setattr(reranking_module.jev_cooldown, "until", 0.0)


@pytest.fixture
def client(session_cache: Cache) -> TestClient:
    embedding_client = AsyncMock()
    embedding_client.embed.return_value = [0.1] * 1024
    application = FastAPI()
    install_exception_handlers(application)
    application.include_router(search_router)
    application.dependency_overrides[get_session_store] = lambda: SessionStore(session_cache)
    application.dependency_overrides[get_session_cache] = lambda: session_cache
    application.dependency_overrides[get_session_secret] = lambda: SECRET
    application.dependency_overrides[get_session_cookie_secure] = lambda: False
    application.dependency_overrides[get_llm_client] = _fresh_async_mock
    application.dependency_overrides[get_embedding_client] = lambda: embedding_client
    application.dependency_overrides[get_db_session] = _fresh_async_mock
    return TestClient(application)


@dataclass
class Handles:
    gemini: AsyncMock
    jev: FakeJevClient


@dataclass(frozen=True)
class Backend:
    """One request's rerank setup: the configured mode, the scored candidates and the Jev knobs."""

    mode: str
    scored: list[tuple[SimpleNamespace, ScoreBreakdown]]
    jev: FakeJevClient | None = None
    jev_percent: int = 0
    jev_timeout_ms: int = 1200
    budget: BudgetPolicy | None = None
    cooldown_s: float = 60.0
    no_key: bool = False

    @contextmanager
    def patched(self) -> Generator[Handles, None, None]:
        """Patch the pipeline call sites and the rerank settings for one request."""
        fake_jev = self.jev or FakeJevClient()
        settings = MagicMock()
        settings.llm_daily_budget_usd = 25.0
        settings.ai_search_rerank_rank_only = False
        settings.ai_search_conditional_rerank = True
        settings.ai_search_parser_timeout_ms = 500
        settings.ai_search_rerank_timeout_ms = 1800
        settings.ai_search_rerank_backend = self.mode
        settings.ai_search_rerank_jev_percent = self.jev_percent
        settings.ai_search_rerank_jev_timeout_ms = self.jev_timeout_ms
        settings.ai_search_rerank_jev_cooldown_s = self.cooldown_s
        policy = self.budget or BudgetPolicy(allow_narratives=True, allow_rerank=True, spend_fraction=0.0)
        with (
            patch("ingovo_search.search.request.get_settings", return_value=settings),
            patch("ingovo_search.search.request.get_jev_client", return_value=None if self.no_key else fake_jev),
            patch("ingovo_search.search.request.current_budget_policy", new_callable=AsyncMock, return_value=policy),
            patch(
                "ingovo_search.search.request.parse_query",
                new_callable=AsyncMock,
                return_value=ParsedQuery(semantic_query=QUERY),
            ),
            patch("ingovo_search.search.candidates.recommend", new_callable=AsyncMock, return_value=self.scored),
            patch("ingovo_search.search.pipeline.should_clarify", return_value=None),
            patch("ingovo_search.search.reranking.rerank_and_explain", new_callable=AsyncMock) as gemini,
            patch(
                "ingovo_search.search.response_cache.cached_listing_ids_are_publicly_visible",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("ingovo_search.search.reranking.should_rerank", return_value=True),
        ):
            gemini.return_value = _gemini_result([prop.id for prop, _ in reversed(self.scored)])
            yield Handles(gemini=gemini, jev=fake_jev)


def _search(client: TestClient) -> dict[str, Any]:
    response = client.post("/api/search", json={"query": QUERY})
    assert response.status_code == 200, response.text
    return response.json()


def _result_ids(body: dict[str, Any]) -> list[str]:
    return [result["listing_id"] for result in body["data"]["results"]]


# --- backend selection ------------------------------------------------------


def test_gemini_mode_calls_only_gemini(client: TestClient) -> None:
    with Backend("gemini", scored=_scored(4)).patched() as handles:
        body = _search(client)

    handles.gemini.assert_awaited_once()
    assert handles.jev.calls == []
    assert _result_ids(body) == ["L3", "L2", "L1", "L0"]


def test_jev_mode_calls_only_jev_and_serves_its_order(client: TestClient) -> None:
    jev = FakeJevClient(scores={"c0": 1.0, "c1": 3.0, "c2": 0.5, "c3": 2.0})

    with Backend("jev", scored=_scored(4), jev=jev).patched() as handles:
        body = _search(client)

    handles.gemini.assert_not_awaited()
    assert len(jev.calls) == 1
    assert _result_ids(body) == ["L1", "L3", "L0", "L2"]


def test_jev_mode_sends_the_listing_summaries_and_pinned_model(client: TestClient) -> None:
    jev = FakeJevClient(scores={"c0": 1.0, "c1": 2.0})

    with Backend("jev", scored=_scored(2), jev=jev, jev_timeout_ms=900).patched():
        _search(client)

    [call] = jev.calls
    assert call.model == "jev-1.13.0"
    assert call.timeout_s == 0.9
    candidates = call.state["jelöltek"]
    assert isinstance(candidates, dict)
    assert set(candidates) == {"c0", "c1"}


@pytest.mark.parametrize(("percent", "expect_jev"), [(0, False), (100, True)])
def test_canary_mode_follows_the_percentage(client: TestClient, percent: int, expect_jev: bool) -> None:
    jev = FakeJevClient(scores={"c0": 1.0, "c1": 3.0})

    with Backend("canary", scored=_scored(2), jev=jev, jev_percent=percent).patched() as handles:
        _search(client)

    assert (len(jev.calls) == 1) is expect_jev
    assert handles.gemini.await_count == (0 if expect_jev else 1)


def test_off_mode_calls_neither_and_keeps_the_deterministic_order(client: TestClient) -> None:
    with Backend("off", scored=_scored(3)).patched() as handles:
        body = _search(client)

    handles.gemini.assert_not_awaited()
    assert handles.jev.calls == []
    assert _result_ids(body) == ["L0", "L1", "L2"]
    assert body["meta"]["timings_ms"]["rerank"] == 0


def test_shadow_mode_still_serves_gemini(client: TestClient) -> None:
    with Backend("shadow", scored=_scored(2)).patched() as handles:
        _search(client)

    handles.gemini.assert_awaited_once()


def test_unrecognised_mode_serves_gemini() -> None:
    assert serving_rerank_backend("unknown", session_id="s", jev_percent=100) == "gemini"


def test_canary_bucket_is_sticky_per_session_and_roughly_proportional() -> None:
    sessions = [f"session-{index}" for index in range(2000)]

    first = [serving_rerank_backend("canary", session_id=s, jev_percent=50) for s in sessions]
    second = [serving_rerank_backend("canary", session_id=s, jev_percent=50) for s in sessions]

    assert first == second
    assert 900 < first.count("jev") < 1100


# --- guardrails, timeouts, failures -----------------------------------------


def test_jev_order_still_passes_the_bounded_rerank_guardrail(client: TestClient) -> None:
    # Jev wants the last of 12 on top: an 11-place jump, beyond MAX_RERANK_DELTA (10).
    jev = FakeJevClient(scores={f"c{index}": (3.0 if index == 11 else 1.0) for index in range(12)})
    head_ids = [f"L{index}" for index in range(12)]
    jev_order = ["L11", *head_ids[:11]]

    with Backend("jev", scored=_scored(12), jev=jev).patched():
        body = _search(client)

    assert _result_ids(body) == enforce_bounded_rerank(head_ids, jev_order)
    assert _result_ids(body)[0] != "L11"


def test_slow_jev_hits_the_pipeline_ceiling_and_falls_back(client: TestClient) -> None:
    # Client timeout 50 ms + 300 ms grace; the fake ignores its timeout and sleeps 1 s.
    jev = FakeJevClient(scores={"c0": 1.0, "c1": 3.0}, delay_s=1.0)

    with Backend("jev", scored=_scored(2), jev=jev, jev_timeout_ms=50).patched() as handles:
        body = _search(client)

    assert _result_ids(body) == ["L0", "L1"]
    assert body["meta"]["timings_ms"]["rerank_timeout"] == 1
    # The time budget is spent: no Gemini on top, and no cooldown.
    handles.gemini.assert_not_awaited()
    assert reranking_module.jev_cooldown.until == 0.0


def test_jev_client_timeout_keeps_the_deterministic_order_without_gemini(client: TestClient) -> None:
    jev = FakeJevClient(error=JevError("slow", reason="timeout"))

    with Backend("jev", scored=_scored(3), jev=jev).patched() as handles:
        body = _search(client)

    assert _result_ids(body) == ["L0", "L1", "L2"]
    handles.gemini.assert_not_awaited()
    assert body["meta"]["timings_ms"]["rerank_gemini_fallback"] == 0
    assert reranking_module.jev_cooldown.until == 0.0


def test_budget_hard_overage_skips_the_jev_rerank(client: TestClient) -> None:
    jev = FakeJevClient(scores={"c0": 1.0, "c1": 3.0})
    overage = BudgetPolicy(allow_narratives=False, allow_rerank=False, spend_fraction=1.3)

    with Backend("jev", scored=_scored(2), jev=jev, budget=overage).patched():
        body = _search(client)

    assert jev.calls == []
    assert _result_ids(body) == ["L0", "L1"]


# --- response caches ----------------------------------------------------------


def test_gemini_cache_keys_are_unchanged_and_other_backends_get_their_own() -> None:
    assert rerank_cache_suffix("gemini") == ""
    assert rerank_cache_suffix("jev") == ":rb-jev"
    assert rerank_cache_suffix("off") == ":rb-off"


def test_a_gemini_cached_response_is_not_served_in_jev_mode(client: TestClient) -> None:
    jev = FakeJevClient(scores={"c0": 3.0, "c1": 1.0})

    with Backend("gemini", scored=_scored(2)).patched():
        gemini_body = _search(client)
    with Backend("jev", scored=_scored(2), jev=jev).patched():
        jev_body = _search(client)

    assert _result_ids(gemini_body) == ["L1", "L0"]
    assert len(jev.calls) == 1
    assert _result_ids(jev_body) == ["L0", "L1"]


# --- Jev first, Gemini as the fallback (2026-09-28) -------------------------------------


@pytest.mark.parametrize("status", [429, 529])
def test_a_jev_limit_hands_the_rerank_to_gemini(client: TestClient, status: int) -> None:
    jev = FakeJevClient(error=JevError("no capacity", reason="http", status=status))

    with Backend("jev", scored=_scored(3), jev=jev).patched() as handles:
        body = _search(client)

    assert len(jev.calls) == 1
    handles.gemini.assert_awaited_once()
    assert _result_ids(body) == ["L2", "L1", "L0"]
    assert body["meta"]["timings_ms"]["rerank_gemini_fallback"] == 1
    assert body["meta"]["timings_ms"]["rerank_timeout"] == 0


def test_after_a_limit_gemini_serves_directly_during_the_cooldown(client: TestClient) -> None:
    jev = FakeJevClient(error=JevError("rate limited", reason="http", status=429))

    with Backend("jev", scored=_scored(3), jev=jev).patched() as handles:
        _search(client)
        body = _search(client)

    # The first search tried Jev; the second went straight to Gemini.
    assert len(jev.calls) == 1
    assert handles.gemini.await_count == 2
    assert _result_ids(body) == ["L2", "L1", "L0"]


def test_jev_is_tried_again_after_the_cooldown_and_the_fallback_was_not_cached(client: TestClient) -> None:
    failing = FakeJevClient(error=JevError("overloaded", reason="http", status=529))
    working = FakeJevClient(scores={"c0": 1.0, "c1": 3.0, "c2": 2.0})

    with Backend("jev", scored=_scored(3), jev=failing, cooldown_s=0.0).patched():
        first = _search(client)
    with Backend("jev", scored=_scored(3), jev=working).patched() as handles:
        second = _search(client)

    assert _result_ids(first) == ["L2", "L1", "L0"]
    # Not served from a cached Gemini response: Jev ran and its order came back.
    assert len(working.calls) == 1
    handles.gemini.assert_not_awaited()
    assert _result_ids(second) == ["L1", "L2", "L0"]


def test_a_rejected_key_falls_back_to_gemini_and_logs_an_error(client: TestClient) -> None:
    jev = FakeJevClient(error=JevError("unauthorized", reason="http", status=401))

    with (
        Backend("jev", scored=_scored(3), jev=jev).patched() as handles,
        patch("ingovo_search.search.reranking.search_logger") as logger,
    ):
        body = _search(client)

    handles.gemini.assert_awaited_once()
    assert _result_ids(body) == ["L2", "L1", "L0"]
    logger.error.assert_called_once()
    assert "rejected" in logger.error.call_args.args[0]


def test_an_outage_falls_back_to_gemini(client: TestClient) -> None:
    jev = FakeJevClient(error=JevError("connection refused", reason="connection"))

    with Backend("jev", scored=_scored(3), jev=jev).patched() as handles:
        body = _search(client)

    handles.gemini.assert_awaited_once()
    assert _result_ids(body) == ["L2", "L1", "L0"]


def test_without_a_key_gemini_serves(client: TestClient) -> None:
    with Backend("jev", scored=_scored(3), no_key=True).patched() as handles:
        body = _search(client)

    assert handles.jev.calls == []
    handles.gemini.assert_awaited_once()
    assert _result_ids(body) == ["L2", "L1", "L0"]
    assert body["meta"]["timings_ms"]["rerank_gemini_fallback"] == 1


def test_the_search_log_records_the_gemini_fallback_and_the_cache_hit_arm(client: TestClient) -> None:
    jev = FakeJevClient(error=JevError("rate limited", reason="http", status=429))
    working = FakeJevClient(scores={"c0": 1.0, "c1": 3.0, "c2": 2.0})

    with (
        Backend("jev", scored=_scored(3), jev=jev).patched(),
        patch("ingovo_search.search.pipeline.log_search", new_callable=AsyncMock) as log_search,
    ):
        _search(client)
    assert log_search.await_args is not None
    assert log_search.await_args.kwargs["rerank_backend"] == "jev"
    assert log_search.await_args.kwargs["rerank_outcome"] == "gemini_fallback"

    reranking_module.jev_cooldown.until = 0.0
    with (
        Backend("jev", scored=_scored(3), jev=working).patched(),
        patch("ingovo_search.search.pipeline.log_search", new_callable=AsyncMock) as log_search,
    ):
        _search(client)
        _search(client)
    # The second identical search is a response-cache hit; its row still names the arm.
    outcomes = [call.kwargs["rerank_outcome"] for call in log_search.await_args_list]
    arms = [call.kwargs["rerank_backend"] for call in log_search.await_args_list]
    assert outcomes[0] == "success"
    assert outcomes[-1] == "cache_hit"
    assert set(arms) == {"jev"}
