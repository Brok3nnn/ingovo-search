"""Observability: structlog setup, product-event logging, and the SearchLog writer.

- `configure_logging` sets up structured logs once at boot (JSON in production,
  a readable console renderer elsewhere).
- `track_event` records one named product event (a search ran, a clarification was
  shown, a refinement chip was removed) as a structured log line. The original
  product forwarded these to an analytics backend; here they stay in the logs, and
  the event names are still validated so a typo at a call site fails a test.
- `log_search` + `record_latency_ms` write one `search_log` row per search and time
  it, which the eval and reporting scripts read.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.db.models import SearchLog

EVENT_SEARCH_RUN = "search_run"
EVENT_REFINEMENT_CHIP_REMOVE = "refinement_chip_remove"
EVENT_REFINEMENT_DISAMBIGUATE = "refinement_disambiguate"
EVENT_CLARIFICATION_SHOWN = "clarification_shown"
EVENT_CLARIFICATION_ANSWERED = "clarification_answered"
EVENT_CLARIFICATION_SKIPPED = "clarification_skipped"

TRACKED_EVENTS: frozenset[str] = frozenset(
    {
        EVENT_SEARCH_RUN,
        EVENT_REFINEMENT_CHIP_REMOVE,
        EVENT_REFINEMENT_DISAMBIGUATE,
        EVENT_CLARIFICATION_SHOWN,
        EVENT_CLARIFICATION_ANSWERED,
        EVENT_CLARIFICATION_SKIPPED,
    }
)


def configure_logging(env: str = "development") -> None:
    """Configure structlog once at app boot. JSON output in prod, pretty in dev."""
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if env == "production":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


def analytics_distinct_id(*, user_id: str | None = None, session_id: str | None = None) -> str:
    """The opaque identity an event is attributed to: never an email or a name."""
    return user_id or session_id or "anonymous"


def track_event(name: str, *, distinct_id: str, properties: dict[str, object] | None = None) -> None:
    """Log one product event. An unknown event name raises: that is a bug, not a runtime state.

    Callers pass only the shape of what happened (counts, flags), never the query text:
    the sentence a visitor types can name a street, a workplace or a school run.
    """
    if name not in TRACKED_EVENTS:
        raise ValueError(f"unknown event '{name}' (allowed: {sorted(TRACKED_EVENTS)})")
    logger = structlog.get_logger("ingovo_search.events")
    logger.info("product_event", event_name=name, distinct_id=distinct_id, **(properties or {}))


@contextmanager
def record_latency_ms() -> Generator[Callable[[], int], None, None]:
    """Context manager that yields a callable returning elapsed ms since enter."""
    start = time.monotonic()
    yield lambda: int((time.monotonic() - start) * 1000)


async def log_search(
    session: AsyncSession,
    *,
    query_hash: str,
    hard_filter_count: int,
    sql_candidate_count: int,
    returned_count: int,
    latency_ms: int,
    score_version: str,
    top_result_ids: list[str],
    parsed_query: dict[str, Any] | None = None,
    vector_candidate_count: int | None = None,
    cache_hit: bool = False,
    user_id: str | None = None,
    rerank_backend: str | None = None,
    rerank_outcome: str | None = None,
) -> SearchLog:
    """Add one `SearchLog` row describing a retrieval (the caller commits).

    `rerank_backend` names the arm that served the search (gemini | jev | off) and
    `rerank_outcome` how its rerank ended (success | fallback | timeout | fast_path |
    skipped:<reason>).
    """
    row = SearchLog(
        user_id=user_id,
        rerank_backend=rerank_backend,
        rerank_outcome=rerank_outcome,
        query_hash=query_hash,
        parsed_query=parsed_query,
        hard_filter_count=hard_filter_count,
        sql_candidate_count=sql_candidate_count,
        vector_candidate_count=vector_candidate_count,
        returned_count=returned_count,
        cache_hit=cache_hit,
        latency_ms=latency_ms,
        score_version=score_version,
        top_result_ids=list(top_result_ids),
    )
    session.add(row)
    return row
