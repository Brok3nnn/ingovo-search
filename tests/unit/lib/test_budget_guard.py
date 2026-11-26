"""Unit tests for §16 LLM budget guardrails."""

from __future__ import annotations

from datetime import date
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.lib.budget_guard import (
    BudgetPolicyCache,
    budget_policy_for,
    current_budget_policy,
)
from ingovo_search.types.errors import DbError

_BUDGET = 25.0


# --- budget_policy_for (pure) ------------------------------------------------


def test_under_budget_allows_everything() -> None:
    policy = budget_policy_for(10.0, _BUDGET)  # 40%
    assert policy.allow_narratives is True
    assert policy.allow_rerank is True
    assert policy.throttled is False


def test_at_budget_drops_narratives_keeps_rerank() -> None:
    policy = budget_policy_for(25.0, _BUDGET)  # 100%
    assert policy.allow_narratives is False
    assert policy.allow_rerank is True
    assert policy.throttled is True


def test_hard_overage_drops_rerank_too() -> None:
    policy = budget_policy_for(32.0, _BUDGET)  # 128% > 125%
    assert policy.allow_narratives is False
    assert policy.allow_rerank is False


def test_non_positive_budget_disables_guard() -> None:
    policy = budget_policy_for(1000.0, 0.0)
    assert policy.allow_narratives is True
    assert policy.allow_rerank is True


# --- current_budget_policy (cache + fail-open) -------------------------------


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


async def test_policy_is_cached_within_ttl() -> None:
    clock = _Clock()
    cache = BudgetPolicyCache(clock=clock)
    calls = {"n": 0}

    async def cost_fn(_session: AsyncSession, _day: date) -> float:
        calls["n"] += 1
        return 40.0  # 160% of budget → over the hard (rerank-off) overage

    session = cast(AsyncSession, object())
    first = await current_budget_policy(session, budget_usd=_BUDGET, cache=cache, cost_fn=cost_fn)
    second = await current_budget_policy(session, budget_usd=_BUDGET, cache=cache, cost_fn=cost_fn)

    assert first.allow_rerank is False
    assert second == first
    assert calls["n"] == 1  # second call served from cache

    clock.t += 31.0  # advance past the 30s TTL
    await current_budget_policy(session, budget_usd=_BUDGET, cache=cache, cost_fn=cost_fn)
    assert calls["n"] == 2  # cache expired → re-read


async def test_fails_open_on_db_error() -> None:
    cache = BudgetPolicyCache()

    async def cost_fn(_session: AsyncSession, _day: date) -> float:
        raise DbError("boom")

    session = cast(AsyncSession, object())
    policy = await current_budget_policy(session, budget_usd=_BUDGET, cache=cache, cost_fn=cost_fn)

    assert policy.allow_narratives is True
    assert policy.allow_rerank is True
    assert cache.get() is None  # a failed read is not cached
