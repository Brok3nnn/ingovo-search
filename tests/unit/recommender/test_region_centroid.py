"""Unit tests for the learned region centroids — the DB-free layer.

`weighted_centroid`, `hint_district_score`, the positive-signal table, and the
LLM hint call are pure / mockable and fully covered here. `matches_named_region`
is covered against a stub session for its branch logic; its real DB behaviour
(centroid lookup, hint caching, `recompute_region_centroid`) lives in
`tests/integration/recommender/test_geo_centroid.py`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.lib.llm import LlmClient, LlmRequest, MockLlmClient, fixture_key
from ingovo_search.lib.llm_models import TIER_S_PARSER
from ingovo_search.recommender.geo_score import ListingGeo
from ingovo_search.recommender.region_centroid import (
    POSITIVE_SIGNAL_WEIGHTS,
    ResolvedRegionMatcher,
    matches_named_region,
    resolve_named_region,
    weighted_centroid,
)
from ingovo_search.recommender.region_llm_hint import (
    RegionHint,
    RegionHintDistrict,
    hint_district_score,
    region_llm_hint,
)
from ingovo_search.types.errors import LlmError, ParseError

SOURCE = Path(__file__).resolve().parents[3] / "src" / "ingovo_search" / "recommender" / "region_centroid.py"
VALID_HINT = json.dumps({"districts": [{"district": "12", "weight": 0.6}, {"district": "2", "weight": 0.25}]})


def _hint_key(region_name: str) -> str:
    return fixture_key(LlmRequest(model=TIER_S_PARSER, system="", user=region_name))


def _correction_key(region_name: str) -> str:
    return fixture_key(LlmRequest(model=TIER_S_PARSER, system="", user=f"[CORRECT_SCHEMA]\n{region_name}"))


class _Row:
    def __init__(self, **fields: object) -> None:
        for key, value in fields.items():
            setattr(self, key, value)


class _Result:
    def __init__(self, row: object | None = None) -> None:
        self._row = row

    def one_or_none(self) -> object | None:
        return self._row


class _StubSession:
    """No-DB session double — returns a fixed centroid row for the first query."""

    def __init__(self, centroid_row: object | None = None) -> None:
        self._centroid_row = centroid_row
        self.execute_count = 0
        self.added: list[object] = []

    async def execute(self, _statement: object, _params: object | None = None) -> _Result:
        self.execute_count += 1
        return _Result(self._centroid_row if self.execute_count == 1 else None)

    def add(self, obj: object) -> None:
        self.added.append(obj)


# --- weighted_centroid -------------------------------------------------------


def test_weighted_centroid_single_point() -> None:
    lat, lng, spread = weighted_centroid([(47.5, 19.0, 1.0)])
    assert (lat, lng) == (47.5, 19.0)
    assert spread == 0.0


def test_weighted_centroid_midpoint_of_equal_weights() -> None:
    lat, lng, _ = weighted_centroid([(47.5, 19.0, 1.0), (47.6, 19.0, 1.0)])
    assert math.isclose(lat, 47.55)
    assert math.isclose(lng, 19.0)


def test_weighted_centroid_biases_toward_heavier_point() -> None:
    lat, _, _ = weighted_centroid([(47.5, 19.0, 1.0), (47.6, 19.0, 3.0)])
    # Weighted mean: (47.5*1 + 47.6*3) / 4 = 47.575
    assert math.isclose(lat, 47.575)


def test_weighted_centroid_spread_is_positive_for_spread_points() -> None:
    _, _, spread = weighted_centroid([(47.5, 19.0, 1.0), (47.6, 19.0, 1.0)])
    assert spread > 0.0


def test_weighted_centroid_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one point"):
        weighted_centroid([])


def test_weighted_centroid_zero_total_weight_raises() -> None:
    with pytest.raises(ValueError, match="positive total weight"):
        weighted_centroid([(47.5, 19.0, 0.0)])


# --- positive-signal table ---------------------------------------------------


def test_positive_signal_weights_contains_only_positive_signals() -> None:
    assert set(POSITIVE_SIGNAL_WEIGHTS) == {
        "thumb_up",
        "favorite",
        "save",
        "contact",
        "viewing_booked",
        "view_long",
    }
    assert all(weight > 0 for weight in POSITIVE_SIGNAL_WEIGHTS.values())


def test_upsert_centroid_sql_casts_asyncpg_parameters() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "CAST(:lat AS double precision)" in text
    assert "CAST(:lng AS double precision)" in text
    assert "CAST(:spread_km AS double precision)" in text
    assert "CAST(:n_signals AS integer)" in text


# --- hint_district_score -----------------------------------------------------


def test_hint_district_score_returns_matched_weight() -> None:
    hints = [RegionHintDistrict(district="12", weight=0.6), RegionHintDistrict(district="2", weight=0.25)]
    assert hint_district_score("12", hints) == 0.6
    assert hint_district_score("2", hints) == 0.25


def test_hint_district_score_zero_for_unmatched_or_missing() -> None:
    hints = [RegionHintDistrict(district="12", weight=0.6)]
    assert hint_district_score("7", hints) == 0.0
    assert hint_district_score(None, hints) == 0.0
    assert hint_district_score("12", []) == 0.0


# --- region_llm_hint ---------------------------------------------------------


async def test_region_llm_hint_returns_validated_hint() -> None:
    region = "Budai-hegyvidék"
    client = MockLlmClient({_hint_key(region): VALID_HINT})

    hint = await region_llm_hint(region, client=cast(LlmClient, client))

    assert isinstance(hint, RegionHint)
    assert [(d.district, d.weight) for d in hint.districts] == [("12", 0.6), ("2", 0.25)]


async def test_region_llm_hint_persistent_failure_raises_parse_error() -> None:
    region = "Ismeretlen"
    client = MockLlmClient({_hint_key(region): "not json", _correction_key(region): "still not json"})

    with pytest.raises(ParseError):
        await region_llm_hint(region, client=cast(LlmClient, client))


async def test_region_llm_hint_recovers_on_correction_retry() -> None:
    region = "Hibás majd jó"
    client = MockLlmClient({_hint_key(region): "broken", _correction_key(region): VALID_HINT})

    hint = await region_llm_hint(region, client=cast(LlmClient, client))
    assert len(hint.districts) == 2


async def test_region_llm_hint_transport_error_propagates() -> None:
    client = MockLlmClient({})  # no fixture → LlmError
    with pytest.raises(LlmError):
        await region_llm_hint("Bármi", client=cast(LlmClient, client))


# --- matches_named_region (branch logic against a stub session) --------------


async def test_matches_named_region_uses_learned_centroid_when_enough_signal() -> None:
    row = _Row(region_name="X", lat=47.5, lng=19.0, spread_km=2.0, n_signals=100, llm_hint_districts=None)
    session = _StubSession(centroid_row=row)
    client = MockLlmClient({})  # must not be called on the learned path

    score = await matches_named_region(
        cast(AsyncSession, session), "X", ListingGeo(lat=47.5, lng=19.0), client=cast(LlmClient, client)
    )
    assert score == 1.0  # property sits exactly on the learned centroid


async def test_matches_named_region_cold_start_generates_and_scores_hint() -> None:
    region = "Budai-hegyvidék"
    session = _StubSession(centroid_row=None)
    client = MockLlmClient({_hint_key(region): VALID_HINT})

    score = await matches_named_region(
        cast(AsyncSession, session), region, ListingGeo(district="12"), client=cast(LlmClient, client)
    )
    assert score == 0.6  # district 12 carries hint weight 0.6


async def test_matches_named_region_uses_cached_hint_without_llm() -> None:
    row = _Row(
        region_name="X",
        lat=None,
        lng=None,
        spread_km=None,
        n_signals=10,  # below the learned threshold
        llm_hint_districts=[{"district": "5", "weight": 0.9}],
    )
    session = _StubSession(centroid_row=row)
    client = MockLlmClient({})  # no fixtures → would raise LlmError if called

    score = await matches_named_region(
        cast(AsyncSession, session), "X", ListingGeo(district="5"), client=cast(LlmClient, client)
    )
    assert score == 0.9


# --- resolve_named_region + ResolvedRegionMatcher (batch path) ---------------


async def test_resolve_learned_region_scores_many_listings_db_free() -> None:
    """Resolve once, then score every candidate in memory.

    The per-listing `get_region_centroid` round-trip is the N+1 this removes:
    after one resolve, scoring more listings must not touch the session or LLM.
    """
    row = _Row(region_name="X", lat=47.5, lng=19.0, spread_km=2.0, n_signals=100, llm_hint_districts=None)
    session = _StubSession(centroid_row=row)
    client = MockLlmClient({})  # learned path must not call the LLM

    matcher = await resolve_named_region(cast(AsyncSession, session), "X", client=cast(LlmClient, client))
    assert isinstance(matcher, ResolvedRegionMatcher)
    assert session.execute_count == 1  # exactly one centroid fetch

    on_centroid = matcher.score(property_lat=47.5, property_lng=19.0)
    offset = matcher.score(property_lat=47.5, property_lng=19.02)
    missing_coords = matcher.score(property_district="12")

    assert on_centroid == 1.0
    assert 0.0 < offset < 1.0
    assert missing_coords == 0.0  # learned path needs coordinates
    assert session.execute_count == 1  # scoring stayed in memory — no extra DB work


async def test_resolve_cold_start_region_scores_many_districts_with_one_llm_call() -> None:
    """Cold start: one LLM hint per search, then in-memory district scoring."""
    region = "Budai-hegyvidék"
    session = _StubSession(centroid_row=None)
    client = MockLlmClient({_hint_key(region): VALID_HINT})

    matcher = await resolve_named_region(cast(AsyncSession, session), region, client=cast(LlmClient, client))
    executes_after_resolve = session.execute_count

    assert matcher.score(property_district="12") == 0.6
    assert matcher.score(property_district="2") == 0.25
    assert matcher.score(property_district="7") == 0.0  # unmatched district
    # No further DB writes/reads once resolved — the hint is reused from memory.
    assert session.execute_count == executes_after_resolve


async def test_resolve_cold_start_region_without_a_model_scores_without_the_hint() -> None:
    """A model outage must not fail the search: the region adds no signal, and nothing is cached."""
    session = _StubSession(centroid_row=None)
    client = MockLlmClient({})  # no fixture -> LlmError, as in an outage

    matcher = await resolve_named_region(cast(AsyncSession, session), "Budai-hegyvidék", client=cast(LlmClient, client))

    assert matcher.hints == []
    assert matcher.score(property_district="12") == 0.0
    assert session.execute_count == 1  # the centroid lookup only; no hint row was written
