"""One `EnrichmentLog` row per LLM attempt — the parser, the rerank and the region hint.

The three wrote the same row through three private copies of one helper; this is
that helper. The row carries the provider-reported tokens when the call just made
reported them, and the character estimate otherwise (`lib/llm_usage.py`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.db.models import EnrichmentLog
from ingovo_search.lib.cost_tracker import call_cost
from ingovo_search.lib.llm import LlmRequest
from ingovo_search.lib.llm_usage import last_llm_usage


@dataclass(frozen=True, slots=True)
class LlmAttempt:
    """One LLM call to log: the request as sent, the prompt version it ran under, its monotonic start."""

    request: LlmRequest
    prompt_version: str
    started: float


def log_llm_attempt(session: AsyncSession, attempt: LlmAttempt, *, outcome: str, raw: str) -> None:
    """Add the attempt's `EnrichmentLog` row to `session`; `raw` is the model's answer, empty on a transport error."""
    request = attempt.request
    # Real tokens when the call just made reported them.
    cost = call_cost(
        request.model, usage=last_llm_usage(), prompt_text=request.user, completion_text=raw, system_text=request.system
    )
    session.add(
        EnrichmentLog(
            property_id=None,
            model_used=cost.model,
            prompt_version=attempt.prompt_version,
            input_tokens=cost.input_tokens,
            output_tokens=cost.output_tokens,
            cost_usd=cost.cost_usd,
            cached_input_tokens=cost.cached_input_tokens,
            latency_ms=int((time.monotonic() - attempt.started) * 1000),
            outcome=outcome,
        )
    )


__all__ = ["LlmAttempt", "log_llm_attempt"]
