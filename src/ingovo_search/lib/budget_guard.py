"""LLM budget guardrails — degrade optional AI when the day's spend is high.

The cost-alert job only *notifies* at 90% of `LLM_DAILY_BUDGET_USD`. This module
adds *enforcement* for the search path so spend cannot run away unbounded:

- below budget: everything on (parser, rerank, narratives).
- at/above 100% of budget: drop non-essential narratives, keep core rerank.
- at/above the hard overage: also drop the rerank (Gemini or Jev) — search degrades to the
  deterministic scoring pipeline, which still returns useful results.

The parser is never gated: it is cheap, cached, and often handled by the
deterministic fast path with no LLM call at all.

`budget_policy_for` is pure and unit-tested. `current_budget_policy` adds a short
in-process TTL cache so the day's spend is read at most once per `TTL` seconds,
not once per search, and fails *open* (everything allowed) if the read errors —
a transient DB problem must never break search.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.lib.cost_tracker import daily_llm_cost_total
from ingovo_search.types.errors import DbError

_logger = structlog.get_logger("ingovo_search.lib.budget_guard")

# Fractions of the daily budget at which optional LLM features switch off.
NARRATIVE_OFF_FRACTION = 1.0  # at/above 100% of budget: stop non-essential narratives
RERANK_OFF_FRACTION = 1.25  # at/above 125%: also drop the rerank, whichever backend (deterministic only)

POLICY_CACHE_TTL_SECONDS = 30.0


@dataclass(frozen=True)
class BudgetPolicy:
    """What optional LLM work the current spend level permits."""

    allow_narratives: bool
    allow_rerank: bool
    spend_fraction: float

    @property
    def throttled(self) -> bool:
        return not (self.allow_narratives and self.allow_rerank)


def budget_policy_for(daily_cost_usd: float, budget_usd: float) -> BudgetPolicy:
    """Pure policy decision from the day's spend and the configured budget.

    A non-positive budget disables the guard entirely (everything allowed).
    """
    if budget_usd <= 0:
        return BudgetPolicy(allow_narratives=True, allow_rerank=True, spend_fraction=0.0)
    fraction = daily_cost_usd / budget_usd
    return BudgetPolicy(
        allow_narratives=fraction < NARRATIVE_OFF_FRACTION,
        allow_rerank=fraction < RERANK_OFF_FRACTION,
        spend_fraction=fraction,
    )


class BudgetPolicyCache:
    """Tiny single-slot TTL cache for the global budget policy."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._entry: tuple[float, BudgetPolicy] | None = None

    def get(self) -> BudgetPolicy | None:
        if self._entry is not None and self._clock() < self._entry[0]:
            return self._entry[1]
        return None

    def set(self, policy: BudgetPolicy, ttl_seconds: float = POLICY_CACHE_TTL_SECONDS) -> None:
        self._entry = (self._clock() + ttl_seconds, policy)

    def clear(self) -> None:
        self._entry = None


_default_cache = BudgetPolicyCache()


async def current_budget_policy(
    session: AsyncSession,
    *,
    budget_usd: float,
    day: date | None = None,
    cache: BudgetPolicyCache = _default_cache,
    cost_fn: Callable[[AsyncSession, date], Awaitable[float]] = daily_llm_cost_total,
) -> BudgetPolicy:
    """Return the current `BudgetPolicy`, cached for `POLICY_CACHE_TTL_SECONDS`.

    Fails open: if the spend read raises, everything is allowed and a warning is
    logged — a budget read must never take search down.
    """
    cached = cache.get()
    if cached is not None:
        return cached
    target_day = day or datetime.now(tz=UTC).date()
    try:
        total = await cost_fn(session, target_day)
        policy = budget_policy_for(total, budget_usd)
    except DbError as exc:
        _logger.warning("budget policy read failed; allowing all LLM features", exc_info=exc)
        return BudgetPolicy(allow_narratives=True, allow_rerank=True, spend_fraction=0.0)
    cache.set(policy)
    return policy
