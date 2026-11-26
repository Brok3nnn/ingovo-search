"""R4 absolute buyer-query match percentage tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ingovo_search.recommender.match_percent import MatchBreakdown, compute_match_breakdown
from ingovo_search.recommender.parser_schema import Anchors, HardFilters, NearFilter, ParsedQuery
from ingovo_search.recommender.score import WEIGHTS, ScoreBreakdown, ScoredCandidate
from ingovo_search.recommender.search_contract import ResultScope


def _listing(**updates: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "city": "Tiszafüred",
        "district": None,
        "price": 80_000_000.0,
        "area_m2": 60.0,
        "rooms": "2",
        "property_type": "Lakás",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _ranking(**updates: float) -> ScoreBreakdown:
    values = {name: 0.0 for name in WEIGHTS}
    values.update(updates)
    return ScoreBreakdown(total=sum(values.values()), **values)


def _tiszafured_query(**updates: object) -> ParsedQuery:
    values: dict[str, object] = {
        "hard_filters": HardFilters(city="Tiszafüred", city_id="ksh:29726"),
    }
    values.update(updates)
    return ParsedQuery(**values)  # type: ignore[arg-type]


def test_exact_location_scores_higher_than_verified_nearby() -> None:
    query = _tiszafured_query()
    exact = compute_match_breakdown(_listing(), _ranking(), query)
    nearby = compute_match_breakdown(_listing(city="Tiszaszőlős"), _ranking(), query)
    assert exact.result_scope is ResultScope.EXACT
    assert exact.location == 1.0
    assert nearby.result_scope is ResultScope.NEARBY
    assert nearby.distance_km is not None
    assert exact.match_percent > nearby.match_percent


def test_nearer_verified_location_never_scores_below_farther_one() -> None:
    query = _tiszafured_query()
    nearer = compute_match_breakdown(_listing(city="Tiszaszőlős"), _ranking(), query)
    farther = compute_match_breakdown(_listing(city="Tiszaörs"), _ranking(), query)
    assert nearer.distance_km is not None
    assert farther.distance_km is not None
    assert nearer.distance_km < farther.distance_km
    assert nearer.location is not None
    assert farther.location is not None
    assert nearer.location >= farther.location
    assert nearer.match_percent >= farther.match_percent


def _deak_ter_radius_query() -> ParsedQuery:
    # "lakás a Deák Ferenc tér 2 kilométeres körzetében" — the fast path sets
    # city="Budapest" *and* a near filter; the near filter must win.
    return ParsedQuery(
        hard_filters=HardFilters(
            city="Budapest",
            city_id="ksh:13578",
            near=NearFilter(lat=47.4979, lng=19.0545, radius_km=2.0),
            property_type_in=["Lakás"],
        ),
    )


def _bp_listing(district: str, **updates: object) -> SimpleNamespace:
    return _listing(city="Budapest", district=district, **updates)


def test_poi_radius_scores_by_proximity_not_city_equality() -> None:
    # geo ranking term = WEIGHTS["geo"] * proximity. A listing next to the POI
    # (proximity ~1.0) must beat one at the radius edge (proximity ~0.37); the
    # far one must NOT inherit a flat location = 1.0 just for being in Budapest.
    query = _deak_ter_radius_query()
    at_poi = compute_match_breakdown(_bp_listing("5"), _ranking(geo=WEIGHTS["geo"] * 0.99), query)
    near_edge = compute_match_breakdown(_bp_listing("13"), _ranking(geo=WEIGHTS["geo"] * 0.37), query)

    assert at_poi.location is not None
    assert at_poi.location > 0.9
    assert near_edge.location is not None
    assert near_edge.location < 0.5
    assert at_poi.match_percent > near_edge.match_percent
    assert near_edge.match_percent < 100


def test_poi_radius_listing_without_coordinates_is_capped() -> None:
    # No geo term (listing without coordinates) → location 0 → hard-filter cap 69.
    result = compute_match_breakdown(_bp_listing("13"), _ranking(geo=0.0), _deak_ter_radius_query())
    assert result.location == 0.0
    assert result.match_percent <= 69


def test_distant_location_is_out_of_scope_and_capped() -> None:
    result = compute_match_breakdown(_listing(city="Győr"), _ranking(), _tiszafured_query())
    assert result.result_scope is ResultScope.OUT_OF_SCOPE
    assert result.location == 0.0
    assert result.match_percent <= 49


def test_quality_recency_pooled_and_explore_do_not_change_percent() -> None:
    query = _tiszafured_query(anchors=Anchors(price_huf=80_000_000))
    clean = _ranking()
    noisy = _ranking(
        quality=WEIGHTS["quality"],
        recency=WEIGHTS["recency"],
        pooled=WEIGHTS["pooled"],
        explore=WEIGHTS["explore"],
    )
    assert (
        compute_match_breakdown(_listing(), clean, query).match_percent
        == compute_match_breakdown(_listing(), noisy, query).match_percent
    )


def test_missing_active_value_gets_zero_component_not_full_credit() -> None:
    query = ParsedQuery(hard_filters=HardFilters(price_huf_max=80_000_000))
    result = compute_match_breakdown(_listing(price=None), _ranking(), query)
    assert result.price == 0.0
    assert result.match_percent == 1
    assert result.hard_filter_cap == 69


def test_no_active_match_dimension_is_not_automatic_one_hundred() -> None:
    result = compute_match_breakdown(_listing(), _ranking(), ParsedQuery())
    assert result.active_weight == 0.0
    assert result.match_percent == 1


def test_unverified_requested_amenity_cannot_get_full_component_score() -> None:
    query = ParsedQuery(semantic_query="úszómedencés villa", soft_prefs_freetext=["úszómedence"])
    ranking = _ranking(
        content=WEIGHTS["content"],
        factual=WEIGHTS["factual"],
        softpref_free=WEIGHTS["softpref_free"],
    ).model_copy(update={"amenity_match": 0.0})
    result = compute_match_breakdown(_listing(), ranking, query)
    assert result.amenity_cap == 45
    assert result.match_percent <= 45


def test_percent_is_always_integer_in_closed_contract_range() -> None:
    queries = [
        ParsedQuery(),
        ParsedQuery(hard_filters=HardFilters(price_huf_max=1)),
        _tiszafured_query(),
        ParsedQuery(semantic_query="lakás"),
    ]
    for query in queries:
        value = compute_match_breakdown(_listing(), _ranking(), query).match_percent
        assert isinstance(value, int)
        assert 1 <= value <= 100


@pytest.mark.parametrize(
    ("component", "query", "matching_updates", "missing_updates", "ranking_updates"),
    [
        (
            "location",
            _tiszafured_query(),
            {"city": "Tiszafüred"},
            {"city": ""},
            {},
        ),
        (
            "price",
            ParsedQuery(hard_filters=HardFilters(price_huf_min=70_000_000, price_huf_max=90_000_000)),
            {"price": 80_000_000.0},
            {"price": None},
            {},
        ),
        (
            "area",
            ParsedQuery(hard_filters=HardFilters(area_m2_min=50.0, area_m2_max=70.0)),
            {"area_m2": 60.0},
            {"area_m2": None},
            {},
        ),
        (
            "rooms",
            ParsedQuery(hard_filters=HardFilters(rooms=2)),
            {"rooms": "2"},
            {"rooms": None},
            {},
        ),
        (
            "property_type",
            ParsedQuery(hard_filters=HardFilters(property_type_in=["Lakás"])),
            {"property_type": "Lakás"},
            {"property_type": None},
            {},
        ),
        (
            "preferences",
            ParsedQuery(soft_preferences=["parking"]),
            {},
            {},
            {"softpref_core": WEIGHTS["softpref_core"]},
        ),
        (
            "semantic",
            ParsedQuery(semantic_query="világos lakás"),
            {},
            {},
            {"content": WEIGHTS["content"], "factual": WEIGHTS["factual"]},
        ),
    ],
)
def test_each_component_has_explicit_maximum_and_missing_data_minimum(
    component: str,
    query: ParsedQuery,
    matching_updates: dict[str, object],
    missing_updates: dict[str, object],
    ranking_updates: dict[str, float],
) -> None:
    maximum = compute_match_breakdown(_listing(**matching_updates), _ranking(**ranking_updates), query)
    missing = compute_match_breakdown(_listing(**missing_updates), _ranking(), query)
    assert getattr(maximum, component) == 1.0
    assert getattr(missing, component) == 0.0


class TestPublicMatchComponents:
    """The public projection of a match breakdown (audit F11).

    The full ``MatchBreakdown`` is excluded from ``ScoredCandidate`` because it
    carries the RAW ``result_scope``, which the API layer sanitises on the
    candidate itself. These tests pin the boundary: the projection must carry the
    criterion scores and nothing that would route around that sanitisation.
    """

    def _breakdown(self, **updates: object) -> MatchBreakdown:
        values: dict[str, object] = {
            "location": 0.9,
            "price": 0.2,
            "area": 1.0,
            "rooms": 0.5,
            "property_type": 1.0,
            "preferences": None,
            "semantic": 0.7,
            "active_weight": 0.8,
            "weighted_score": 0.64,
            "hard_filter_cap": 100,
            "amenity_cap": 100,
            "match_percent": 64,
            "result_scope": ResultScope.EXACT,
            "distance_km": None,
        }
        values.update(updates)
        return MatchBreakdown(**values)  # type: ignore[arg-type]

    def test_carries_every_criterion_score(self) -> None:
        components = self._breakdown().public_components()

        assert components.location == 0.9
        assert components.price == 0.2
        assert components.area == 1.0
        assert components.rooms == 0.5
        assert components.property_type == 1.0
        assert components.semantic == 0.7

    def test_preserves_none_for_uncollected_criteria(self) -> None:
        """``None`` means the query never asked — not that the listing scored 0."""

        components = self._breakdown(preferences=None, rooms=None).public_components()

        assert components.preferences is None
        assert components.rooms is None

    def test_does_not_expose_scope_distance_or_weighting_internals(self) -> None:
        """The whole point of the subset: none of this may reach a consumer."""

        components = self._breakdown(result_scope=ResultScope.OUT_OF_SCOPE, distance_km=412.5).public_components()
        payload = components.model_dump()

        for leaked in (
            "result_scope",
            "distance_km",
            "active_weight",
            "weighted_score",
            "hard_filter_cap",
            "amenity_cap",
            "match_percent",
        ):
            assert leaked not in payload, f"{leaked} must stay internal"

    def test_out_of_scope_state_cannot_travel_through_the_projection(self) -> None:
        """Serialising the projection must never mention an internal scope."""

        components = self._breakdown(result_scope=ResultScope.OUT_OF_SCOPE).public_components()

        assert "out_of_scope" not in components.model_dump_json()

    def test_scored_candidate_serialises_components_but_not_the_breakdown(self) -> None:
        breakdown = self._breakdown(result_scope=ResultScope.OUT_OF_SCOPE, distance_km=412.5)
        candidate = ScoredCandidate(
            listing_id="abc",
            score=0.64,
            match_percent=64,
            result_scope=ResultScope.EXACT,
            distance_km=None,
            match_breakdown=breakdown,
            match_components=breakdown.public_components(),
            breakdown=_ranking(),
        )

        payload = candidate.model_dump()

        assert "match_breakdown" not in payload
        assert payload["match_components"]["location"] == 0.9
        # The sanitised scope on the parent survives; the raw one does not leak.
        assert payload["result_scope"] is ResultScope.EXACT
        assert "out_of_scope" not in candidate.model_dump_json()
