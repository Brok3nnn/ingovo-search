"""`AI_SEARCH_RERANK_BACKEND=shadow`.

Gemini serves the search; the Jev rerank runs behind it on the same head and
one `rerank_shadow_comparison` row records both orders. The harness mirrors
`test_search_rerank_backend.py`; the shadow task is awaited through the
TestClient's portal, which owns the app's event loop.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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
from ingovo_search.db.models import EnrichmentLog, RerankShadowComparison
from ingovo_search.lib.budget_guard import BudgetPolicy
from ingovo_search.lib.cache import MemoryCache
from ingovo_search.lib.jev import FakeJevClient
from ingovo_search.lib.session import SessionStore
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.rerank import RerankItem, RerankResult
from ingovo_search.recommender.rerank_jev_prompt import JEV_RERANK_PROMPT_VERSION
from ingovo_search.recommender.score import ScoreBreakdown
from ingovo_search.search.shadow import (
    _rerank_shadow_tasks,  # pyright: ignore[reportPrivateUsage]
    drain_rerank_shadow_tasks,
    shadow_sampled,
)
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


def _app() -> FastAPI:
    embedding_client = AsyncMock()
    embedding_client.embed.return_value = [0.1] * 1024
    session_cache = MemoryCache()
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
    return application


class _RecordingSession:
    """The shadow task's own session: records `add` and `commit`, works as an async context."""

    def __init__(self) -> None:
        self.added: list[Any] = []
        self.commits = 0

    def add(self, entity: Any) -> None:
        self.added.append(entity)

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        return None

    async def __aenter__(self) -> _RecordingSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    def shadow_rows(self) -> list[RerankShadowComparison]:
        return [entity for entity in self.added if isinstance(entity, RerankShadowComparison)]


@contextmanager
def shadow(
    mode: str = "shadow",
    *,
    scored: list[tuple[SimpleNamespace, ScoreBreakdown]],
    jev: FakeJevClient,
    sample: float = 1.0,
    jev_timeout_ms: int = 1200,
) -> Generator[tuple[AsyncMock, _RecordingSession], None, None]:
    settings = MagicMock()
    settings.llm_daily_budget_usd = 25.0
    settings.ai_search_rerank_rank_only = False
    settings.ai_search_conditional_rerank = True
    settings.ai_search_parser_timeout_ms = 500
    settings.ai_search_rerank_timeout_ms = 1800
    settings.ai_search_rerank_backend = mode
    settings.ai_search_rerank_jev_percent = 0
    settings.ai_search_rerank_jev_timeout_ms = jev_timeout_ms
    settings.ai_search_rerank_shadow_sample = sample
    policy = BudgetPolicy(allow_narratives=True, allow_rerank=True, spend_fraction=0.0)
    session = _RecordingSession()
    with (
        patch("ingovo_search.search.request.get_settings", return_value=settings),
        patch("ingovo_search.search.request.get_jev_client", return_value=jev),
        patch("ingovo_search.search.shadow.get_async_session_factory", return_value=lambda: session),
        patch("ingovo_search.search.request.current_budget_policy", new_callable=AsyncMock, return_value=policy),
        patch(
            "ingovo_search.search.request.parse_query",
            new_callable=AsyncMock,
            return_value=ParsedQuery(semantic_query=QUERY),
        ),
        patch("ingovo_search.search.candidates.recommend", new_callable=AsyncMock, return_value=scored),
        patch("ingovo_search.search.pipeline.should_clarify", return_value=None),
        patch("ingovo_search.search.reranking.rerank_and_explain", new_callable=AsyncMock) as gemini,
        patch(
            "ingovo_search.search.response_cache.cached_listing_ids_are_publicly_visible",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch("ingovo_search.search.reranking.should_rerank", return_value=True),
    ):
        gemini.return_value = _gemini_result([prop.id for prop, _ in reversed(scored)])
        yield gemini, session


def _drain(client: TestClient) -> None:
    """Wait for the shadow task inside the app's own event loop."""
    assert client.portal is not None
    client.portal.call(drain_rerank_shadow_tasks)


def _search(client: TestClient) -> list[str]:
    response = client.post("/api/search", json={"query": QUERY})
    assert response.status_code == 200, response.text
    return [result["listing_id"] for result in response.json()["data"]["results"]]


def test_shadow_serves_gemini_and_records_both_orders_without_waiting() -> None:
    jev = FakeJevClient(scores={"c0": 1.0, "c1": 3.0, "c2": 0.5, "c3": 2.0}, delay_s=0.6)

    with TestClient(_app()) as client, shadow(scored=_scored(4), jev=jev) as (gemini, session):
        started = time.perf_counter()
        ids = _search(client)
        elapsed = time.perf_counter() - started

        assert ids == ["L3", "L2", "L1", "L0"]
        assert elapsed < 0.5, "the response must not wait for the shadow Jev call"
        gemini.assert_awaited_once()
        assert session.shadow_rows == []
        _drain(client)

        assert len(jev.calls) == 1
        [row] = session.shadow_rows
        assert row.served_backend == "gemini"
        assert row.candidate_ids == ["L0", "L1", "L2", "L3"]
        assert row.gemini_order == ["L3", "L2", "L1", "L0"]
        assert row.jev_order == ["L1", "L3", "L0", "L2"]
        assert row.jev_scores == {"L0": 0.3333, "L1": 1.0, "L2": 0.1667, "L3": 0.6667}
        assert (row.gemini_outcome, row.jev_outcome) == ("success", "success")
        assert row.jev_input_tokens == 1000
        assert row.jev_model == "jev-1.13.0"
        assert row.prompt_version == JEV_RERANK_PROMPT_VERSION
        assert row.jev_latency_ms is not None
        assert row.jev_latency_ms >= 500
        assert row.semantic_query == QUERY
        assert any(isinstance(entity, EnrichmentLog) for entity in session.added)
        assert session.commits == 1
        assert not _rerank_shadow_tasks


def test_jev_error_in_shadow_keeps_the_search_and_records_the_outcome() -> None:
    jev = FakeJevClient(error=JevError("overloaded", reason="http", status=529))

    with TestClient(_app()) as client, shadow(scored=_scored(3), jev=jev) as (_gemini, session):
        assert _search(client) == ["L2", "L1", "L0"]
        _drain(client)

        [row] = session.shadow_rows
        assert row.jev_outcome == "rate_limited"
        assert row.jev_order is None
        assert row.jev_scores is None
        assert row.gemini_order == ["L2", "L1", "L0"]
        assert session.commits == 1


def test_slow_shadow_jev_is_recorded_as_a_timeout() -> None:
    # Client timeout 50 ms + 300 ms grace; the fake ignores its timeout and sleeps 1 s.
    jev = FakeJevClient(scores={"c0": 1.0, "c1": 3.0}, delay_s=1.0)

    with TestClient(_app()) as client, shadow(scored=_scored(2), jev=jev, jev_timeout_ms=50) as (_gemini, session):
        assert _search(client) == ["L1", "L0"]
        _drain(client)

        [row] = session.shadow_rows
        assert row.jev_outcome == "timeout"
        assert row.jev_order is None
        assert row.jev_latency_ms is not None
        assert row.jev_latency_ms < 1000


def test_zero_sample_makes_no_jev_call_and_writes_nothing() -> None:
    jev = FakeJevClient(scores={"c0": 1.0, "c1": 3.0})

    with TestClient(_app()) as client, shadow(scored=_scored(2), jev=jev, sample=0.0) as (_gemini, session):
        assert _search(client) == ["L1", "L0"]
        _drain(client)

    assert jev.calls == []
    assert session.added == []
    assert not _rerank_shadow_tasks


def test_gemini_mode_never_shadows() -> None:
    jev = FakeJevClient(scores={"c0": 1.0, "c1": 3.0})

    with TestClient(_app()) as client, shadow("gemini", scored=_scored(2), jev=jev) as (_gemini, session):
        _search(client)
        _drain(client)

    assert jev.calls == []
    assert session.added == []


def test_shadow_sampling_is_sticky_per_session_and_roughly_proportional() -> None:
    sessions = [f"session-{index}" for index in range(2000)]

    first = [shadow_sampled(s, 0.25) for s in sessions]
    second = [shadow_sampled(s, 0.25) for s in sessions]

    assert first == second
    assert 400 < first.count(True) < 600
    assert not any(shadow_sampled(s, 0.0) for s in sessions)
    assert all(shadow_sampled(s, 1.0) for s in sessions)


def test_search_log_carries_the_serving_arm_and_the_rerank_outcome() -> None:
    jev = FakeJevClient(scores={"c0": 1.0, "c1": 3.0})

    with (
        TestClient(_app()) as client,
        shadow("gemini", scored=_scored(2), jev=jev),
        patch("ingovo_search.search.pipeline.log_search", new_callable=AsyncMock) as log_search,
    ):
        _search(client)

    log_search.assert_awaited_once()
    assert log_search.await_args is not None
    assert log_search.await_args.kwargs["rerank_backend"] == "gemini"
    assert log_search.await_args.kwargs["rerank_outcome"] == "success"


def test_search_log_records_a_skipped_rerank_with_its_reason() -> None:
    jev = FakeJevClient(scores={"c0": 1.0, "c1": 3.0})

    with (
        TestClient(_app()) as client,
        shadow("off", scored=_scored(2), jev=jev),
        patch("ingovo_search.search.pipeline.log_search", new_callable=AsyncMock) as log_search,
    ):
        _search(client)

    assert log_search.await_args is not None
    assert log_search.await_args.kwargs["rerank_backend"] == "off"
    assert log_search.await_args.kwargs["rerank_outcome"] == "skipped:disabled"
