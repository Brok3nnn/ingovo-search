"""Unit tests for the observability helpers: event logging, the latency timer, the SearchLog writer."""

from __future__ import annotations

import time
from typing import cast

import pytest
import structlog.testing
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.db.models import SearchLog
from ingovo_search.lib.observability import (
    EVENT_CLARIFICATION_ANSWERED,
    EVENT_CLARIFICATION_SHOWN,
    EVENT_CLARIFICATION_SKIPPED,
    EVENT_REFINEMENT_CHIP_REMOVE,
    EVENT_REFINEMENT_DISAMBIGUATE,
    EVENT_REFINEMENT_TRADEOFF,
    EVENT_SEARCH_RUN,
    TRACKED_EVENTS,
    analytics_distinct_id,
    log_search,
    record_latency_ms,
    track_event,
)

# --- Events -----------------------------------------------------------------


def test_tracked_events_are_the_search_funnel() -> None:
    assert (
        frozenset(
            {
                EVENT_SEARCH_RUN,
                EVENT_REFINEMENT_CHIP_REMOVE,
                EVENT_REFINEMENT_DISAMBIGUATE,
                EVENT_REFINEMENT_TRADEOFF,
                EVENT_CLARIFICATION_SHOWN,
                EVENT_CLARIFICATION_ANSWERED,
                EVENT_CLARIFICATION_SKIPPED,
            }
        )
        == TRACKED_EVENTS
    )


def test_track_event_logs_the_event_with_its_properties() -> None:
    with structlog.testing.capture_logs() as logs:
        track_event(EVENT_SEARCH_RUN, distinct_id="session-1", properties={"returned_count": 3})

    assert logs == [
        {
            "event": "product_event",
            "event_name": EVENT_SEARCH_RUN,
            "distinct_id": "session-1",
            "returned_count": 3,
            "log_level": "info",
        }
    ]


def test_track_event_rejects_unknown_event_name() -> None:
    with pytest.raises(ValueError, match="unknown event"):
        track_event("search_rnu", distinct_id="session-1")


# --- distinct id ------------------------------------------------------------


def test_analytics_distinct_id_prefers_the_signed_in_user() -> None:
    assert analytics_distinct_id(user_id="user-1", session_id="session-9") == "user-1"


def test_analytics_distinct_id_falls_back_to_the_session() -> None:
    assert analytics_distinct_id(user_id=None, session_id="session-9") == "session-9"


def test_analytics_distinct_id_has_a_last_resort() -> None:
    assert analytics_distinct_id() == "anonymous"


# --- Latency timer ----------------------------------------------------------


def test_record_latency_ms_returns_non_negative_int() -> None:
    with record_latency_ms() as elapsed:
        time.sleep(0.001)
        value = elapsed()
    assert isinstance(value, int)
    assert value >= 0


# --- SearchLog writer -------------------------------------------------------


class _RecordingSession:
    def __init__(self) -> None:
        self.added: list[object] = []

    def add(self, entity: object) -> None:
        self.added.append(entity)


async def test_log_search_adds_row_with_full_metadata() -> None:
    session = _RecordingSession()

    row = await log_search(
        cast(AsyncSession, session),
        query_hash="hash-1",
        hard_filter_count=2,
        sql_candidate_count=120,
        returned_count=30,
        latency_ms=87,
        score_version="v0",
        top_result_ids=["a", "b"],
        parsed_query={"semantic_query": "lakás"},
        vector_candidate_count=120,
        cache_hit=False,
        user_id="user-42",
    )

    assert isinstance(row, SearchLog)
    assert session.added == [row]
    assert row.query_hash == "hash-1"
    assert row.user_id == "user-42"
    assert row.hard_filter_count == 2
    assert row.sql_candidate_count == 120
    assert row.vector_candidate_count == 120
    assert row.returned_count == 30
    assert row.cache_hit is False
    assert row.latency_ms == 87
    assert row.score_version == "v0"
    assert row.top_result_ids == ["a", "b"]
    assert row.parsed_query == {"semantic_query": "lakás"}


async def test_log_search_records_the_serving_rerank_backend() -> None:
    session = _RecordingSession()

    row = await log_search(
        cast(AsyncSession, session),
        query_hash="hash-2",
        hard_filter_count=1,
        sql_candidate_count=5,
        returned_count=5,
        latency_ms=12,
        score_version="v3",
        top_result_ids=["a"],
        rerank_backend="jev",
        rerank_outcome="success",
    )

    assert row.rerank_backend == "jev"
    assert row.rerank_outcome == "success"
    assert session.added == [row]
