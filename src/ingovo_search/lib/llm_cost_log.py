"""Cost rows for LLM calls whose numbers are known only after the answer is out.

The image analysis of a photo search runs as a background task, so its cost rows
cannot ride on the request's session the way the parser's do. They are written
here, with a short session of their own, the same pattern as the Jev shadow writer
in `search/shadow.py`.

Best-effort by design: bookkeeping must never cost the reader an answer, so a
failed or slow write is logged and dropped.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from typing import Final

import structlog

from ingovo_search.db.models import EnrichmentLog
from ingovo_search.lib.cost_tracker import call_cost
from ingovo_search.lib.llm_usage import LlmUsage

_logger = structlog.get_logger("ingovo_search.lib.llm_cost_log")

# The write runs after `done`, but the response stays open until it returns, so a
# wedged pool must not hold it for long.
RECORD_TIMEOUT_SECONDS: Final = 5.0


def usage_log_enabled() -> bool:
    """`AI_USAGE_LOG=off` stops these rows without a deploy. On by default.

    The unit-test conftest switches it off: the suite reads the developer's
    `.env`, and the database behind it may be a real one.
    """
    return os.environ.get("AI_USAGE_LOG", "on").strip().lower() not in {"off", "0", "false"}


def llm_call_row(
    *,
    usage: LlmUsage | None,
    requested_model: str,
    prompt_version: str,
    latency_ms: int,
    outcome: str,
    property_id: str | None = None,
    prompt_text: str = "",
    completion_text: str = "",
) -> EnrichmentLog:
    """One `EnrichmentLog` row: the provider's numbers when it reported them, the
    character estimate from the texts otherwise."""
    cost = call_cost(requested_model, usage=usage, prompt_text=prompt_text, completion_text=completion_text)
    return EnrichmentLog(
        property_id=property_id,
        model_used=cost.model,
        prompt_version=prompt_version,
        input_tokens=cost.input_tokens,
        output_tokens=cost.output_tokens,
        cost_usd=cost.cost_usd,
        cached_input_tokens=cost.cached_input_tokens,
        latency_ms=max(latency_ms, 0),
        outcome=outcome,
    )


async def record_llm_calls(rows: Sequence[EnrichmentLog]) -> None:
    """Persist `rows` with a session of their own. Never raises."""
    if not rows or not usage_log_enabled():
        return
    try:
        await asyncio.wait_for(_write(list(rows)), timeout=RECORD_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001
        # Fail-soft by contract (module docstring): bookkeeping that runs after the
        # answer is out, usually as a fire-and-forget task, so anything the write
        # raises — the timeout, a SQLAlchemy/asyncpg error, a socket OSError, a
        # settings error while the engine is built — is logged with its traceback
        # and dropped rather than becoming an orphaned task exception.
        await _logger.awarning(
            "llm_usage_log_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            rows=len(rows),
            prompt_versions=sorted({row.prompt_version for row in rows}),
            property_ids=sorted({row.property_id for row in rows if row.property_id}),
            exc_info=exc,
        )


async def _write(rows: list[EnrichmentLog]) -> None:
    # Imported here: building the engine reads the database settings, and code
    # that only builds rows (the tests, the offline chat tier) should not pay for it.
    from ingovo_search.db.session import get_async_session_factory

    async with get_async_session_factory()() as session:
        session.add_all(rows)
        await session.commit()


# Fire-and-forget writes stay referenced until they finish, or the loop may
# garbage-collect them mid-flight — the same rule as the rerank shadow tasks in
# `api/search.py`.
_pending_writes: set[asyncio.Task[None]] = set()


def record_llm_calls_later(rows: Sequence[EnrichmentLog]) -> None:
    """`record_llm_calls` without waiting, for a call on a request's critical path."""
    if not rows or not usage_log_enabled():
        return
    task = asyncio.create_task(record_llm_calls(rows))
    _pending_writes.add(task)
    task.add_done_callback(_pending_writes.discard)


async def drain_usage_writes() -> None:
    """Wait for the background writes still running (tests, shutdown)."""
    if _pending_writes:
        await asyncio.gather(*list(_pending_writes), return_exceptions=True)


__all__ = [
    "RECORD_TIMEOUT_SECONDS",
    "drain_usage_writes",
    "llm_call_row",
    "record_llm_calls",
    "record_llm_calls_later",
    "usage_log_enabled",
]
