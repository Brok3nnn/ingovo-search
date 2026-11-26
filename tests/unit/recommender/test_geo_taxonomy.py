"""Unit tests for §10 static geo taxonomy (counties + Balaton shores)."""

from __future__ import annotations

from ingovo_search.recommender.geo_score import ListingGeo, geo_score
from ingovo_search.recommender.geo_taxonomy import (
    BALATON_AREA_CITIES,
    BALATON_NORTH_SHORE,
    BALATON_SOUTH_SHORE,
    COUNTY_CITIES,
    is_balaton_region_name,
    matches_balaton_area,
    matches_balaton_shore,
    matches_county,
)
from ingovo_search.recommender.parser_schema import GeoHints, HardFilters, ParsedQuery

# --- dataset sanity ----------------------------------------------------------


def test_all_19_counties_plus_budapest_present() -> None:
    """Hungary has 19 vármegye + Budapest as a separate capital region."""
    assert len(COUNTY_CITIES) == 20


def test_pest_megye_includes_budapest_agglomeration() -> None:
    """Pest megye is the most-searched megye; the agglomeration must be covered."""
    cities = COUNTY_CITIES["Pest"]
    for must_have in ("érd", "gödöllő", "vác", "szentendre", "dunakeszi", "budaörs"):
        assert must_have in cities, f"{must_have!r} missing from Pest megye"


def test_balaton_shores_are_disjoint() -> None:
    """North and south shores must not overlap — same town cannot sit on both."""
    assert BALATON_NORTH_SHORE.isdisjoint(BALATON_SOUTH_SHORE)
    assert BALATON_AREA_CITIES == BALATON_NORTH_SHORE | BALATON_SOUTH_SHORE


def test_balaton_north_includes_known_north_settlements() -> None:
    for must_have in ("balatonfüred", "tihany", "keszthely", "balatonalmádi"):
        assert must_have in BALATON_NORTH_SHORE


def test_balaton_south_includes_known_south_settlements() -> None:
    for must_have in ("siófok", "fonyód", "balatonboglár", "balatonlelle"):
        assert must_have in BALATON_SOUTH_SHORE


# --- matches_county ----------------------------------------------------------


def test_matches_county_hit() -> None:
    assert matches_county("Nagykovácsi", "Pest") == 1.0


def test_matches_county_miss() -> None:
    assert matches_county("Debrecen", "Pest") == 0.0


def test_matches_county_case_insensitive() -> None:
    assert matches_county("nagykovácsi", "Pest") == 1.0
    assert matches_county("NAGYKOVÁCSI", "Pest") == 1.0


def test_matches_county_unknown_county_returns_zero() -> None:
    assert matches_county("Budapest", "Nyugat-Európa") == 0.0


def test_matches_county_missing_inputs() -> None:
    assert matches_county(None, "Pest") == 0.0
    assert matches_county("Budapest", None) == 0.0


# --- matches_balaton_shore ---------------------------------------------------


def test_matches_balaton_shore_north_hit() -> None:
    assert matches_balaton_shore("Balatonfüred", "Északi") == 1.0


def test_matches_balaton_shore_south_hit() -> None:
    assert matches_balaton_shore("Siófok", "Déli") == 1.0


def test_matches_balaton_shore_wrong_side() -> None:
    """A north-shore town queried as south must score 0, not partial credit."""
    assert matches_balaton_shore("Balatonfüred", "Déli") == 0.0
    assert matches_balaton_shore("Siófok", "Északi") == 0.0


def test_matches_balaton_shore_unknown_town() -> None:
    assert matches_balaton_shore("Debrecen", "Északi") == 0.0


def test_matches_balaton_area_accepts_both_shores() -> None:
    assert matches_balaton_area("Tihany") == 1.0
    assert matches_balaton_area("Siófok") == 1.0
    assert matches_balaton_area("Budapest") == 0.0


def test_is_balaton_region_name_recognises_generic_region_phrases() -> None:
    assert is_balaton_region_name("Balaton")
    assert is_balaton_region_name("Balaton környéke")
    assert is_balaton_region_name("balatoni nyaralóövezet")
    assert not is_balaton_region_name("Budapest")


# --- geo_score integration ---------------------------------------------------


def test_geo_score_picks_up_county_alone() -> None:
    """A county-only query: hit → 1.0 (single component, weights renormalise)."""
    query = ParsedQuery(hard_filters=HardFilters(), geo_hints=GeoHints(county="Pest"))
    assert geo_score(query, ListingGeo(city="Nagykovácsi")) == 1.0
    assert geo_score(query, ListingGeo(city="Debrecen")) == 0.0


def test_geo_score_picks_up_balaton_shore_alone() -> None:
    query = ParsedQuery(hard_filters=HardFilters(), geo_hints=GeoHints(balaton_shore="Északi"))
    assert geo_score(query, ListingGeo(city="Balatonfüred")) == 1.0
    assert geo_score(query, ListingGeo(city="Siófok")) == 0.0


def test_geo_score_picks_up_generic_balaton_region() -> None:
    query = ParsedQuery(hard_filters=HardFilters(), geo_hints=GeoHints(region_name="Balaton környéke"))
    assert geo_score(query, ListingGeo(city="Tihany"), named_region_score=0.0) == 1.0
    assert geo_score(query, ListingGeo(city="Budapest"), named_region_score=1.0) == 0.0


def test_geo_score_blends_county_and_shore() -> None:
    """Both cues present → weighted average (Veszprém county + north shore).

    Balatonfüred is in Veszprém megye AND on the north shore: both 1.0 → 1.0.
    Siófok is in Somogy megye AND on the south shore: both 0.0 vs query → 0.0.
    """
    query = ParsedQuery(
        hard_filters=HardFilters(),
        geo_hints=GeoHints(county="Veszprém", balaton_shore="Északi"),
    )
    assert geo_score(query, ListingGeo(city="Balatonfüred")) == 1.0
    assert geo_score(query, ListingGeo(city="Siófok")) == 0.0


def test_geo_score_returns_zero_without_any_geo_cue() -> None:
    query = ParsedQuery(hard_filters=HardFilters(), geo_hints=GeoHints())
    assert geo_score(query, ListingGeo(city="Budapest")) == 0.0
