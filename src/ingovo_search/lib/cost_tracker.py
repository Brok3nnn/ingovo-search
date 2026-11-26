"""§16 LLM cost tracking — per-call pricing and the `EnrichmentLog` cost per UTC day.

Every LLM call writes an `EnrichmentLog` row with `cost_usd`. The budget guardrail
(`lib/budget_guard.py`) reads the day's total to switch optional LLM work off as
spend approaches `LLM_DAILY_BUDGET_USD`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.lib.llm_usage import LlmUsage
from ingovo_search.types.errors import DbError

# Per-model USD pricing (per 1M tokens). The Gemini rows were checked against
# ai.google.dev/gemini-api/docs/pricing (paid tier) on 2026-09-29.
PRICING_USD_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    # (input, output)
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.0-flash": (0.075, 0.30),
    # TypeSafe bills input tokens only; output is free (docs.typesafe.ai/models, 2026-09).
    "jev-1.13.0": (0.042, 0.0),
}

# Context caching (same page, same date). A cached input token — an implicit
# cache hit or an explicit `cachedContents` handle — is billed at this rate
# instead of the input rate; an explicit cache also pays storage per hour.
CACHED_INPUT_USD_PER_M_TOKENS: dict[str, float] = {
    "gemini-2.5-flash-lite": 0.01,
    "gemini-2.5-flash": 0.03,
    "gemini-3.1-flash-lite": 0.025,
}
CACHE_STORAGE_USD_PER_M_TOKEN_HOURS: dict[str, float] = {
    "gemini-2.5-flash-lite": 1.00,
    "gemini-2.5-flash": 1.00,
    "gemini-3.1-flash-lite": 1.00,
}

# Heuristic: Hungarian ~3.5-4 characters per LLM token. Off by ~10% from
# the real tokenizer; for cost-budget alerts that resolution is fine.
_CHARS_PER_TOKEN = 4.0


def estimate_token_count(text_value: str | None) -> int:
    """Rough character-based token estimate (~4 chars / token)."""
    if not text_value:
        return 0
    return max(1, round(len(text_value) / _CHARS_PER_TOKEN))


def estimate_cost_usd(model: str, *, prompt_text: str, completion_text: str, system_text: str = "") -> float:
    """Estimate one Gemini call's USD cost from input + completion text length.

    `system_text` is billed as input tokens exactly like `prompt_text`, so it
    must be included for the estimate to track provider billing — the system
    prompt is often larger than the user prompt for parser/rerank/enrich calls.

    Falls back to the cheapest tier when the model is unknown, so a typo
    never makes the cost row a NaN. Returns 0.0 only when all texts are
    empty (mock calls in tests).
    """
    if model not in PRICING_USD_PER_M_TOKENS:
        in_rate, out_rate = PRICING_USD_PER_M_TOKENS["gemini-2.5-flash-lite"]
    else:
        in_rate, out_rate = PRICING_USD_PER_M_TOKENS[model]
    in_tokens = estimate_token_count(system_text) + estimate_token_count(prompt_text)
    out_tokens = estimate_token_count(completion_text)
    return (in_tokens * in_rate + out_tokens * out_rate) / 1_000_000


def cost_usd_from_tokens(model: str, *, input_tokens: int, output_tokens: int) -> float:
    """USD cost of one call from the token counts the provider reported.

    Prefer this over `estimate_cost_usd` whenever the response carries real
    usage (Jev always does): the character heuristic is only an approximation.
    Unknown models fall back to the cheapest Gemini tier, as there.
    """
    if model not in PRICING_USD_PER_M_TOKENS:
        in_rate, out_rate = PRICING_USD_PER_M_TOKENS["gemini-2.5-flash-lite"]
    else:
        in_rate, out_rate = PRICING_USD_PER_M_TOKENS[model]
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def cost_usd_from_usage(usage: LlmUsage, *, fallback_model: str = "gemini-2.5-flash-lite") -> float:
    """USD cost of one call from what the provider reported, cache discount included.

    The cached share of the prompt is billed at the cached rate, the rest at the
    input rate, and thinking tokens at the output rate. `fallback_model` prices a
    usage whose model is unknown or empty, as `cost_usd_from_tokens` does.
    """
    model = usage.model if usage.model in PRICING_USD_PER_M_TOKENS else fallback_model
    in_rate, out_rate = PRICING_USD_PER_M_TOKENS.get(model, PRICING_USD_PER_M_TOKENS["gemini-2.5-flash-lite"])
    cached_rate = CACHED_INPUT_USD_PER_M_TOKENS.get(model, in_rate)
    return (
        usage.uncached_prompt_tokens * in_rate
        + usage.cached_tokens * cached_rate
        + usage.billed_output_tokens * out_rate
    ) / 1_000_000


def cache_storage_cost_usd(model: str, *, tokens: int, seconds: float) -> float:
    """What keeping an explicit cache of `tokens` alive for `seconds` costs.

    Zero for a model with no known storage price, so a missing row can only
    understate, never break, the bookkeeping.
    """
    rate = CACHE_STORAGE_USD_PER_M_TOKEN_HOURS.get(model, 0.0)
    return tokens * rate * (seconds / 3600) / 1_000_000


@dataclass(frozen=True, slots=True)
class CallCost:
    """The figures one `EnrichmentLog` row carries for a call."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    # None for an estimate: a text length says nothing about what a cache served.
    cached_input_tokens: int | None = None


def call_cost(
    model: str,
    *,
    usage: LlmUsage | None,
    prompt_text: str = "",
    completion_text: str = "",
    system_text: str = "",
) -> CallCost:
    """Real figures when the provider reported usage, the character estimate otherwise.

    With usage, the row names the model that actually answered — after a
    fallback that is not the one requested, and it is priced differently.
    """
    if usage is not None:
        answered_by = usage.model or model
        return CallCost(
            model=answered_by,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.billed_output_tokens,
            cost_usd=cost_usd_from_usage(usage.with_model(answered_by), fallback_model=model),
            cached_input_tokens=usage.cached_tokens,
        )
    return CallCost(
        model=model,
        input_tokens=estimate_token_count(system_text) + estimate_token_count(prompt_text),
        output_tokens=estimate_token_count(completion_text),
        cost_usd=estimate_cost_usd(
            model, prompt_text=prompt_text, completion_text=completion_text, system_text=system_text
        ),
    )


_DAILY_COST_BY_MODEL_SQL = """
SELECT model_used, COALESCE(SUM(cost_usd), 0.0) AS total
FROM enrichment_log
WHERE created_at >= :day_start AND created_at < :day_end
GROUP BY model_used
"""


def _day_bounds(day: date) -> dict[str, datetime]:
    """UTC-day bounds as datetime objects. `[day_start, day_end)` is exclusive on the right."""
    day_start = datetime.combine(day, time.min, tzinfo=UTC)
    return {"day_start": day_start, "day_end": day_start + timedelta(days=1)}


async def daily_llm_cost(session: AsyncSession, day: date) -> dict[str, float]:
    """Return `{model_used: total_cost_usd}` for one UTC day."""
    try:
        result = await session.execute(text(_DAILY_COST_BY_MODEL_SQL), _day_bounds(day))
    except SQLAlchemyError as exc:
        raise DbError(f"failed to aggregate daily LLM cost for {day.isoformat()}") from exc
    return {str(row.model_used): float(row.total) for row in result}


async def daily_llm_cost_total(session: AsyncSession, day: date) -> float:
    """Sum of all per-model costs for one UTC day."""
    breakdown = await daily_llm_cost(session, day)
    return sum(breakdown.values())


def feature_of(prompt_version: str) -> str:
    """Feature bucket for a cost row = the segment before the first '/'.

    Every prompt version is namespaced `feature/version`
    ("parser/v1.8-hungary-settlement-registry", "rerank/v1.1-factual-fit", "narrator/v2",
    "region_hint/v1-final", "text_enrich/v1-final"), so the prefix names the
    feature that spent the money. Empty/odd values fall back to "unknown".
    """
    if not prompt_version:
        return "unknown"
    return prompt_version.split("/", 1)[0]


_DAILY_COST_BY_PROMPT_VERSION_SQL = """
SELECT prompt_version, COALESCE(SUM(cost_usd), 0.0) AS total
FROM enrichment_log
WHERE created_at >= :day_start AND created_at < :day_end
GROUP BY prompt_version
"""


async def daily_llm_cost_by_feature(session: AsyncSession, day: date) -> dict[str, float]:
    """Return `{feature: total_cost_usd}` for one UTC day.

    The DB aggregates to one row per prompt version (cheap); we fold those into
    feature buckets here so it is clear which product feature — parser, rerank,
    narrator, region hint, text enrichment — generated the day's Gemini spend.
    """
    try:
        result = await session.execute(text(_DAILY_COST_BY_PROMPT_VERSION_SQL), _day_bounds(day))
    except SQLAlchemyError as exc:
        raise DbError(f"failed to aggregate daily LLM cost by feature for {day.isoformat()}") from exc
    buckets: dict[str, float] = {}
    for row in result:
        feature = feature_of(str(row.prompt_version) if row.prompt_version is not None else "")
        buckets[feature] = buckets.get(feature, 0.0) + float(row.total)
    return buckets


# ── M0.7 AI-agent cost baseline ────────────────────────────────────────────
