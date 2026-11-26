"""Unit tests for the per-listing POI distance refresh.

No Postgres — one UPDATE statement does all the work, so we verify the SQL
shape (jsonb_set target, GIST-friendly ST_DWithin radius, geocoded-listing
scope, RETURNING for the touched count) and the error contract with a stub session.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy.exc import SQLAlchemyError

from ingovo_search.ingestion.poi_distances import (
    POI_DISTANCE_RADIUS_M,
    refresh_poi_distances,
)
from ingovo_search.types.errors import DbError


class _ScalarsResult:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def all(self) -> list[str]:
        return list(self._ids)


class _ExecResult:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def scalars(self) -> _ScalarsResult:
        return _ScalarsResult(self._ids)


class _StubSession:
    def __init__(self, return_ids: list[str]) -> None:
        self._return_ids = return_ids
        self.executed: list[tuple[Any, dict[str, Any] | None]] = []

    async def execute(self, statement: Any, params: dict[str, Any] | None = None) -> _ExecResult:
        self.executed.append((statement, params))
        return _ExecResult(self._return_ids)


class _FailingSession:
    async def execute(self, _statement: Any, _params: dict[str, Any] | None = None) -> _ExecResult:
        raise SQLAlchemyError("boom")


@pytest.mark.asyncio
async def test_returns_touched_row_count() -> None:
    session = _StubSession(return_ids=["p1", "p2", "p3"])

    touched = await refresh_poi_distances(session)  # type: ignore[arg-type]

    assert touched == 3


@pytest.mark.asyncio
async def test_default_radius_is_bound_as_parameter() -> None:
    session = _StubSession(return_ids=[])

    await refresh_poi_distances(session)  # type: ignore[arg-type]

    _, params = session.executed[0]
    assert params == {"radius_m": POI_DISTANCE_RADIUS_M}


@pytest.mark.asyncio
async def test_custom_radius_overrides_default() -> None:
    session = _StubSession(return_ids=[])

    await refresh_poi_distances(session, radius_m=500)  # type: ignore[arg-type]

    _, params = session.executed[0]
    assert params == {"radius_m": 500}


@pytest.mark.asyncio
async def test_sql_shape_covers_the_contract() -> None:
    session = _StubSession(return_ids=[])

    await refresh_poi_distances(session)  # type: ignore[arg-type]

    statement, _ = session.executed[0]
    sql = str(statement)

    # Writes into features["poi_m"] without clobbering the rest of features.
    assert "jsonb_set(COALESCE(p.features, '{}'::jsonb), '{poi_m}', agg.poi_m)" in sql
    # Per-category nearest distance, radius-bounded so the GIST index applies.
    assert "MIN(ST_Distance(pr.geo, poi.geo))::int" in sql
    assert "ST_DWithin(pr.geo, poi.geo, :radius_m)" in sql
    assert "GROUP BY poi.category" in sql
    # Only geocoded listings, and the touched count comes from RETURNING.
    assert "pr.geo IS NOT NULL" in sql
    assert "RETURNING p.id" in sql


@pytest.mark.asyncio
async def test_sqlalchemy_error_wrapped_as_db_error() -> None:
    with pytest.raises(DbError):
        await refresh_poi_distances(_FailingSession())  # type: ignore[arg-type]
