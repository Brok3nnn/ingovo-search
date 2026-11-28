"""Unit tests for the pooled-signal kNN — the DB-free layer.

`pooled_signal` uses an asyncpg scalar fast path in integration and a DB-free
Python fallback in unit tests, while `pooled_signals` keeps a raw-SQL batch
aggregate. End-to-end SQL behaviour is covered by the integration suite in
`tests/integration/recommender/test_pooled_signal.py`.

These unit tests cover what does not need a database: the signal-weight table
holds the v0 values, the rendered SQL embeds the tuning constants, parameter
binding is correct, and the Python-side result handling (NULL → 0.0, error
wrapping, dimension validation) behaves.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.lib.vector_db import VECTOR_DIM, vector_literal
from ingovo_search.recommender.pooled_signal import (
    DECAY_WINDOW_SECONDS,
    POOLED_SIGNAL_ROWS_SQL,
    POOLED_SIGNAL_SQL,
    POOLED_SIGNALS_SQL,
    SIGNAL_WEIGHTS,
    SIMILARITY_THRESHOLD,
    TIME_WINDOW_DAYS,
    pooled_signal,
    pooled_signals,
)
from ingovo_search.types.errors import DbError

# The signal-to-weight mapping — the v0 hand-set values.
APPENDIX_C_WEIGHTS = {
    "thumb_up": 1.0,
    "thumb_down": -1.0,
    "favorite": 0.7,
    "save": 0.7,
    "contact": 1.5,
    "viewing_booked": 2.0,
    "view_long": 0.2,
    "view_short": -0.1,
    "hide": -1.5,
}


class _FakeResult:
    """Stands in for a SQLAlchemy `Result` exposing scalar and iteration APIs."""

    def __init__(self, value: float | None, rows: list[Any] | None = None) -> None:
        self._value = value
        self._rows = rows or []

    def scalar_one_or_none(self) -> float | None:
        return self._value

    def __iter__(self):
        return iter(self._rows)


class _StubSession:
    """No-DB session double — records `execute` calls and returns a fixed scalar."""

    def __init__(
        self,
        value: float | None = None,
        *,
        rows: list[Any] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._value = value
        self._rows = rows or []
        self._raises = raises
        self.calls: list[tuple[Any, dict[str, Any] | None]] = []

    async def execute(self, statement: Any, params: dict[str, Any] | None = None) -> _FakeResult:
        self.calls.append((statement, params))
        if self._raises is not None:
            raise self._raises
        return _FakeResult(self._value, self._rows)


def _vec() -> list[float]:
    return [0.0] * VECTOR_DIM


def _query_vec() -> list[float]:
    return [1.0, *([0.0] * (VECTOR_DIM - 1))]


def _similar_vec() -> list[float]:
    return [1.0, 0.05, *([0.0] * (VECTOR_DIM - 2))]


def _session(
    value: float | None = None, *, rows: list[Any] | None = None, raises: Exception | None = None
) -> _StubSession:
    return _StubSession(value, rows=rows, raises=raises)


def _interaction_row(
    signal_type: str,
    *,
    signal_weight: float | int | None = 1.0,
    query_embedding: list[float] | str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        signal_type=signal_type,
        signal_weight=signal_weight,
        query_embedding=query_embedding if query_embedding is not None else _similar_vec(),
        created_at=datetime.now(tz=UTC),
    )


# --- Signal-weight table -----------------------------------------------------


def test_signal_weights_match_appendix_c() -> None:
    assert SIGNAL_WEIGHTS == APPENDIX_C_WEIGHTS


def test_refinement_signals_excluded_from_pool() -> None:
    """`refinement_*` signals feed analytics only — they must not be pooled."""
    assert not any(name.startswith("refinement") for name in SIGNAL_WEIGHTS)


# --- Rendered SQL ------------------------------------------------------------


def test_sql_embeds_tuning_constants() -> None:
    assert SIMILARITY_THRESHOLD == 0.85
    assert TIME_WINDOW_DAYS == 365
    assert DECAY_WINDOW_SECONDS == 7_776_000
    assert "> 0.85" in POOLED_SIGNAL_SQL
    assert "/ 7776000" in POOLED_SIGNAL_SQL
    assert "INTERVAL '365 days'" in POOLED_SIGNAL_SQL


def test_sql_has_one_case_arm_per_signal() -> None:
    for name in SIGNAL_WEIGHTS:
        assert f"WHEN '{name}' THEN" in POOLED_SIGNAL_SQL
        assert f"WHEN '{name}' THEN" in POOLED_SIGNALS_SQL
    assert POOLED_SIGNAL_SQL.count("WHEN '") == len(SIGNAL_WEIGHTS)
    assert POOLED_SIGNALS_SQL.count("WHEN '") == len(SIGNAL_WEIGHTS)


# --- Result handling ---------------------------------------------------------


async def test_returns_scalar_value() -> None:
    session = _session(rows=[_interaction_row("contact")])
    result = await pooled_signal(cast(AsyncSession, session), "prop-1", _query_vec())
    assert abs(result - 1.5) < 1e-12


async def test_returns_zero_when_no_qualifying_interactions() -> None:
    """An empty neighbour set yields 0.0."""
    session = _session(rows=[])
    result = await pooled_signal(cast(AsyncSession, session), "prop-1", _query_vec())
    assert result == 0.0


async def test_integer_signal_weight_coerced_to_float() -> None:
    session = _session(rows=[_interaction_row("contact", signal_weight=2)])
    result = await pooled_signal(cast(AsyncSession, session), "prop-1", _query_vec())
    assert isinstance(result, float)
    assert abs(result - 3.0) < 1e-12


async def test_binds_only_property_id_for_single_property_fast_path() -> None:
    session = _session(rows=[])
    vec = _query_vec()
    await pooled_signal(cast(AsyncSession, session), "prop-42", vec)

    assert len(session.calls) == 1
    statement, params = session.calls[0]
    assert str(statement) == POOLED_SIGNAL_ROWS_SQL
    assert params == {"pid": "prop-42"}


async def test_batch_returns_values_for_each_property_with_zero_default() -> None:
    rows = [
        SimpleNamespace(property_id="prop-2", pooled_signal=1.25),
        SimpleNamespace(property_id="prop-3", pooled_signal=None),
    ]
    session = _session(rows=rows)

    result = await pooled_signals(cast(AsyncSession, session), ["prop-1", "prop-2", "prop-3"], _vec())

    assert result == {"prop-1": 0.0, "prop-2": 1.25, "prop-3": 0.0}


async def test_batch_deduplicates_property_ids_and_binds_vector_literal() -> None:
    session = _session(rows=[])
    vec = _vec()

    result = await pooled_signals(cast(AsyncSession, session), ["prop-1", "prop-1", "prop-2"], vec)

    assert result == {"prop-1": 0.0, "prop-2": 0.0}
    assert len(session.calls) == 1
    _, params = session.calls[0]
    assert params == {"pids": ["prop-1", "prop-2"], "q": vector_literal(vec)}


async def test_batch_empty_property_ids_skips_query() -> None:
    session = _session(rows=[])

    result = await pooled_signals(cast(AsyncSession, session), [], _vec())

    assert result == {}
    assert session.calls == []


# --- Error handling ----------------------------------------------------------


async def test_wrong_dimension_raises_db_error_without_query() -> None:
    session = _session(1.0)
    with pytest.raises(DbError):
        await pooled_signal(cast(AsyncSession, session), "prop-1", [0.0, 1.0, 2.0])
    assert session.calls == []


async def test_sqlalchemy_error_wrapped_as_db_error() -> None:
    session = _session(raises=SQLAlchemyError("connection lost"))
    with pytest.raises(DbError):
        await pooled_signal(cast(AsyncSession, session), "prop-1", _vec())


async def test_batch_sqlalchemy_error_wrapped_as_db_error() -> None:
    session = _session(raises=SQLAlchemyError("connection lost"))
    with pytest.raises(DbError):
        await pooled_signals(cast(AsyncSession, session), ["prop-1"], _vec())
