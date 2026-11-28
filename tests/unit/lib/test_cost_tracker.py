"""Unit tests for the LLM cost aggregation.

`daily_llm_cost` runs one SQL aggregation; these tests stub the session so the
Python-side handling (row → dict conversion, total summing, error wrapping) is
verified without a database. End-to-end DB behaviour belongs to the
integration suite.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.lib.cost_tracker import (
    daily_llm_cost,
    daily_llm_cost_by_feature,
    daily_llm_cost_total,
    feature_of,
)
from ingovo_search.types.errors import DbError

DAY = datetime(2026, 5, 19, tzinfo=UTC).date()


class _Row:
    def __init__(self, *, model_used: str, total: float) -> None:
        self.model_used = model_used
        self.total = total


class _FeatureRow:
    def __init__(self, *, prompt_version: str, total: float) -> None:
        self.prompt_version = prompt_version
        self.total = total


class _ResultIter:
    def __init__(self, rows: list[_Row] | list[_FeatureRow]) -> None:
        self._rows = rows

    def __iter__(self) -> Any:
        return iter(self._rows)


class _StubSession:
    def __init__(
        self,
        rows: list[_Row] | list[_FeatureRow] | None = None,
        *,
        raises: Exception | None = None,
    ) -> None:
        self._rows = rows or []
        self._raises = raises
        self.executed: list[tuple[Any, dict[str, Any] | None]] = []

    async def execute(self, statement: Any, params: dict[str, Any] | None = None) -> _ResultIter:
        self.executed.append((statement, params))
        if self._raises is not None:
            raise self._raises
        return _ResultIter(self._rows)


async def test_daily_llm_cost_aggregates_rows_by_model() -> None:
    session = _StubSession(
        [
            _Row(model_used="gemini-2.5-flash-lite", total=1.23),
            _Row(model_used="embed-multilingual-v3.0", total=0.45),
        ]
    )

    breakdown = await daily_llm_cost(cast(AsyncSession, session), DAY)

    assert breakdown == {"gemini-2.5-flash-lite": 1.23, "embed-multilingual-v3.0": 0.45}


async def test_daily_llm_cost_returns_empty_dict_when_no_rows() -> None:
    session = _StubSession([])
    assert await daily_llm_cost(cast(AsyncSession, session), DAY) == {}


async def test_daily_llm_cost_total_sums_breakdown() -> None:
    session = _StubSession(
        [
            _Row(model_used="gemini-2.5-flash-lite", total=1.0),
            _Row(model_used="cohere-embed", total=2.5),
        ]
    )
    assert await daily_llm_cost_total(cast(AsyncSession, session), DAY) == 3.5


async def test_daily_llm_cost_total_zero_for_no_rows() -> None:
    session = _StubSession([])
    assert await daily_llm_cost_total(cast(AsyncSession, session), DAY) == 0.0


async def test_daily_llm_cost_binds_day_window_params() -> None:
    session = _StubSession([])
    await daily_llm_cost(cast(AsyncSession, session), DAY)
    assert len(session.executed) == 1
    _, params = session.executed[0]
    assert params is not None
    assert params["day_start"] == datetime(2026, 5, 19, tzinfo=UTC)
    assert params["day_end"] == datetime(2026, 5, 20, tzinfo=UTC)


async def test_daily_llm_cost_wraps_sqlalchemy_errors_as_db_error() -> None:
    session = _StubSession(raises=SQLAlchemyError("connection lost"))
    with pytest.raises(DbError):
        await daily_llm_cost(cast(AsyncSession, session), DAY)


# --- feature_of + daily_llm_cost_by_feature ----------------------------------


def test_feature_of_extracts_namespace_prefix() -> None:
    assert feature_of("parser/v1.7-city-adjectives") == "parser"
    assert feature_of("rerank/v1.1-factual-fit") == "rerank"
    assert feature_of("narrator/v2") == "narrator"
    assert feature_of("region_hint/v1-final") == "region_hint"
    assert feature_of("text_enrich/v1-final") == "text_enrich"


def test_feature_of_falls_back_to_unknown() -> None:
    assert feature_of("") == "unknown"
    assert feature_of("noslash") == "noslash"


async def test_daily_llm_cost_by_feature_folds_prompt_versions() -> None:
    session = _StubSession(
        [
            _FeatureRow(prompt_version="parser/v1.7-city-adjectives", total=1.0),
            _FeatureRow(prompt_version="parser/v1.5-old", total=0.5),  # same feature, older version
            _FeatureRow(prompt_version="rerank/v1.1-factual-fit", total=2.0),
            _FeatureRow(prompt_version="narrator/v2", total=0.25),
        ]
    )

    breakdown = await daily_llm_cost_by_feature(cast(AsyncSession, session), DAY)

    assert breakdown == {"parser": 1.5, "rerank": 2.0, "narrator": 0.25}


async def test_daily_llm_cost_by_feature_wraps_errors() -> None:
    session = _StubSession(raises=SQLAlchemyError("down"))
    with pytest.raises(DbError):
        await daily_llm_cost_by_feature(cast(AsyncSession, session), DAY)
