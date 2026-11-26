"""POI proximity scoring  — pure math, no DB.

Covers the walk-time conversion, transit mode aggregation, the graded
proximity score, the missing-data neutrality guarantee (step 5 guardrail:
derived POI data must never exclude or crater a listing), the `near_metro`
soft-tag fix, and the `geo_score` blend integration.
"""

from __future__ import annotations

import math
from typing import Any

from ingovo_search.recommender.geo_score import (
    DEFAULT_POI_TARGET_M,
    NEUTRAL_POI_SCORE,
    ListingGeo,
    geo_score,
    poi_distance_m,
    poi_distances_from_features,
    poi_proximity_score,
    walk_minutes_to_metres,
)
from ingovo_search.recommender.parser_schema import GeoHints, ParsedQuery, PoiNear
from ingovo_search.recommender.softpref_rules import NEAR_METRO_MAX_M, SOFTPREF_RULES


class _Listing:
    def __init__(self, features: dict[str, Any] | None) -> None:
        self.features = features


# --- walk-time conversion ---


def test_five_walk_minutes_is_about_308_metres() -> None:
    assert walk_minutes_to_metres(5) == 5 * 80.0 / 1.3


# --- poi_distances_from_features ---


def test_extracts_numeric_distances_only() -> None:
    features = {"poi_m": {"school": 320, "metro": 850.5, "bogus": "n/a"}, "other": True}

    assert poi_distances_from_features(features) == {"school": 320.0, "metro": 850.5}


def test_missing_or_malformed_features_yield_none() -> None:
    assert poi_distances_from_features(None) is None
    assert poi_distances_from_features({"poi_m": "corrupt"}) is None
    assert poi_distances_from_features({"poi_m": {}}) is None
    assert poi_distances_from_features("not-a-dict") is None


# --- poi_distance_m / transit aggregation ---


def test_transit_takes_the_nearest_of_any_mode() -> None:
    distances = {"metro": 900.0, "tram": 250.0, "bus": 140.0}

    assert poi_distance_m(distances, "transit") == 140.0


def test_transit_without_any_mode_is_none() -> None:
    assert poi_distance_m({"school": 300.0}, "transit") is None


def test_plain_category_reads_directly() -> None:
    assert poi_distance_m({"school": 320.0}, "school") == 320.0
    assert poi_distance_m({"school": 320.0}, "pharmacy") is None


# --- poi_proximity_score ---


def test_no_requests_means_no_component() -> None:
    assert poi_proximity_score([], {"school": 100.0}) is None


def test_on_top_of_the_poi_scores_one() -> None:
    score = poi_proximity_score([PoiNear(category="school")], {"school": 0.0})

    assert score == 1.0


def test_at_default_target_distance_scores_about_037() -> None:
    score = poi_proximity_score([PoiNear(category="school")], {"school": DEFAULT_POI_TARGET_M})

    assert score is not None
    assert math.isclose(score, math.exp(-1), rel_tol=1e-9)


def test_quantified_walk_time_sets_the_target() -> None:
    # 5 minutes ~ 307.7 m straight-line; a listing exactly there scores exp(-1).
    target = walk_minutes_to_metres(5)
    score = poi_proximity_score(
        [PoiNear(category="transit", max_walk_minutes=5)],
        {"bus": target},
    )

    assert score is not None
    assert math.isclose(score, math.exp(-1), rel_tol=1e-9)


def test_missing_category_contributes_neutral_not_zero() -> None:
    """Step 5 guardrail: absent POI data must not crater the listing."""
    score = poi_proximity_score([PoiNear(category="hospital")], {"school": 100.0})

    assert score == NEUTRAL_POI_SCORE


def test_no_poi_data_at_all_is_neutral_for_every_request() -> None:
    score = poi_proximity_score(
        [PoiNear(category="school"), PoiNear(category="transit", max_walk_minutes=5)],
        None,
    )

    assert score == NEUTRAL_POI_SCORE


def test_multiple_requests_average() -> None:
    score = poi_proximity_score(
        [PoiNear(category="school"), PoiNear(category="pharmacy")],
        {"school": 0.0},  # school perfect, pharmacy missing -> neutral
    )

    assert score is not None
    assert math.isclose(score, (1.0 + NEUTRAL_POI_SCORE) / 2, rel_tol=1e-9)


# --- geo_score blend integration ---


def test_geo_score_includes_poi_component() -> None:
    query = ParsedQuery(poi_near=[PoiNear(category="school")])

    value = geo_score(query, ListingGeo(poi_distances_m={"school": 0.0}))

    # Only the POI component is present, so the blend equals it exactly.
    assert value == 1.0


def test_geo_score_poi_requests_without_data_stay_neutral() -> None:
    query = ParsedQuery(poi_near=[PoiNear(category="school")])

    assert geo_score(query, ListingGeo(poi_distances_m=None)) == NEUTRAL_POI_SCORE


def test_geo_score_blends_poi_with_river_side() -> None:
    query = ParsedQuery(
        poi_near=[PoiNear(category="school")],
        geo_hints=GeoHints(side_of_river="Buda"),
    )

    value = geo_score(query, ListingGeo(district="12", poi_distances_m={"school": 0.0}))

    # Both components are perfect matches, so the renormalised blend is 1.0.
    assert value == 1.0


def test_geo_score_without_any_cue_is_zero() -> None:
    assert geo_score(ParsedQuery()) == 0.0


# --- near_metro soft tag (stub fix) ---


def test_near_metro_matches_within_800m() -> None:
    listing = _Listing({"poi_m": {"metro": NEAR_METRO_MAX_M}})

    assert SOFTPREF_RULES["near_metro"](listing) is True


def test_near_metro_rejects_beyond_800m() -> None:
    listing = _Listing({"poi_m": {"metro": NEAR_METRO_MAX_M + 1}})

    assert SOFTPREF_RULES["near_metro"](listing) is False


def test_near_metro_missing_data_is_false_not_error() -> None:
    assert SOFTPREF_RULES["near_metro"](_Listing(None)) is False
    assert SOFTPREF_RULES["near_metro"](_Listing({"poi_m": {"bus": 100}})) is False
