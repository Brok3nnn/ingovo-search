"""Unit tests for the geo scoring — the pure-math layer.

`geo_score` and its component functions are pure, so they are fully covered
here. The DB/LLM-backed `matches_named_region` is exercised in
`test_region_centroid.py` (stubs) and `tests/integration/recommender/`.
"""

from __future__ import annotations

import math

from ingovo_search.recommender.geo_score import (
    BUDA_DISTRICTS,
    ListingGeo,
    gaussian_region_score,
    geo_score,
    haversine_km,
    matches_river_side,
    proximity_score,
)
from ingovo_search.recommender.parser_schema import GeoHints, HardFilters, NearFilter, ParsedQuery, SideOfRiver


def _query(
    *, near: NearFilter | None = None, region_name: str | None = None, side: SideOfRiver | None = None
) -> ParsedQuery:
    return ParsedQuery(
        hard_filters=HardFilters(near=near),
        geo_hints=GeoHints(region_name=region_name, side_of_river=side),
    )


# --- haversine_km ------------------------------------------------------------


def test_haversine_zero_for_identical_points() -> None:
    assert haversine_km(47.5, 19.0, 47.5, 19.0) == 0.0


def test_haversine_one_degree_latitude() -> None:
    # One degree of latitude is ~111.19 km on a sphere of radius 6371.0088 km.
    assert math.isclose(haversine_km(47.0, 19.0, 48.0, 19.0), 111.19, abs_tol=0.2)


# --- matches_river_side -----------------------------------


def test_river_side_buda_district_matches_buda() -> None:
    """District 12 + side 'Buda' → 1.0."""
    assert matches_river_side("12", "Buda") == 1.0


def test_river_side_pest_district_does_not_match_buda() -> None:
    """District 7 + side 'Buda' → 0.0."""
    assert matches_river_side("7", "Buda") == 0.0


def test_river_side_pest_request() -> None:
    assert matches_river_side("7", "Pest") == 1.0
    assert matches_river_side("12", "Pest") == 0.0


def test_river_side_missing_inputs_score_zero() -> None:
    assert matches_river_side(None, "Buda") == 0.0
    assert matches_river_side("12", None) == 0.0


def test_buda_districts_set() -> None:
    expected = {"1", "2", "3", "11", "12", "22"}
    assert set(BUDA_DISTRICTS) == expected


# --- proximity_score ---------------------------------------------------------


def test_proximity_score_peaks_at_anchor() -> None:
    assert proximity_score(0.0, 2.0) == 1.0


def test_proximity_score_at_one_radius() -> None:
    assert math.isclose(proximity_score(2.0, 2.0), math.exp(-1.0))


def test_proximity_score_zero_radius_is_zero() -> None:
    assert proximity_score(1.0, 0.0) == 0.0


# --- gaussian_region_score ---------------------------------------------------


def test_gaussian_region_score_peaks_at_centroid() -> None:
    assert gaussian_region_score(0.0, 2.0) == 1.0


def test_gaussian_region_score_at_one_spread() -> None:
    assert math.isclose(gaussian_region_score(2.0, 2.0), math.exp(-1.0))


def test_gaussian_region_score_no_spread_is_zero() -> None:
    assert gaussian_region_score(1.0, None) == 0.0
    assert gaussian_region_score(1.0, 0.0) == 0.0


def test_gaussian_at_the_centroid_and_near_it() -> None:
    """A centroid at (47.5, 19.0) with spread_km=2.

    Property at the centroid scores 1.0. The original design note annotated a
    property at (47.5, 19.02) as "≈0.95" — but that offset is ~1.50 km, and the
    formula exp(-(d/spread)^2) yields ~0.57 there, not 0.95. Reaching 0.95
    needs an offset of ~0.006° (~0.45 km). This test asserts what the formula
    actually produces; the "≈0.95" annotation is inconsistent with its stated
    coordinates + spread.
    """
    distance = haversine_km(47.5, 19.0, 47.5, 19.02)
    assert math.isclose(distance, 1.50, abs_tol=0.05)
    assert gaussian_region_score(0.0, 2.0) == 1.0
    assert math.isclose(gaussian_region_score(distance, 2.0), 0.569, abs_tol=0.01)
    # A 0.45 km offset is what actually scores ≈0.95 under the spec formula.
    assert math.isclose(gaussian_region_score(0.45, 2.0), 0.95, abs_tol=0.005)


# --- geo_score (component blend) ---------------------------------------------


def test_geo_score_zero_without_any_geographic_cue() -> None:
    assert geo_score(_query()) == 0.0


def test_geo_score_river_side_only() -> None:
    query = _query(side="Buda")
    assert geo_score(query, ListingGeo(district="12")) == 1.0
    assert geo_score(query, ListingGeo(district="7")) == 0.0


def test_geo_score_proximity_only() -> None:
    query = _query(near=NearFilter(lat=47.5, lng=19.0, radius_km=2.0))
    assert geo_score(query, ListingGeo(lat=47.5, lng=19.0)) == 1.0


def test_geo_score_named_region_only() -> None:
    query = _query(region_name="Budai-hegyvidék")
    assert math.isclose(geo_score(query, named_region_score=0.8), 0.8)


def test_geo_score_blends_present_components_by_weight() -> None:
    query = _query(near=NearFilter(lat=47.5, lng=19.0, radius_km=2.0), region_name="X", side="Buda")
    score = geo_score(query, ListingGeo(lat=47.5, lng=19.0, district="12"), named_region_score=0.8)
    # Weighted mean: (0.5*1.0 + 0.35*0.8 + 0.15*1.0) / (0.5 + 0.35 + 0.15) = 0.93
    assert math.isclose(score, 0.93)


def test_geo_score_drops_region_component_when_score_missing() -> None:
    """region_name present but no named_region_score → region weight excluded."""
    query = _query(region_name="X", side="Buda")
    # Only the river-side component remains, so its value is returned as-is.
    assert geo_score(query, ListingGeo(district="12"), named_region_score=None) == 1.0
