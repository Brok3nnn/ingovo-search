"""Absolute, set-independent buyer-query match percentage (R4)."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from ingovo_search.lib.property_price import resolve_property_price_huf
from ingovo_search.recommender.anchor_score import anchor_score
from ingovo_search.recommender.parser_schema import Anchors, HardFilters, ParsedQuery
from ingovo_search.recommender.search_contract import ResultScope, location_match_score
from ingovo_search.recommender.settlements import (
    budapest_district,
    resolve_settlement_name,
    settlement_by_id,
    settlement_distance_km,
)
from ingovo_search.recommender.sql_filter import DISTRICT_ROMAN, district_filter_values

MATCH_PERCENT_VERSION = "match-percent-v2-near-proximity"

MATCH_COMPONENT_WEIGHTS: dict[str, float] = {
    "location": 0.30,
    "price": 0.16,
    "area": 0.12,
    "rooms": 0.12,
    "property_type": 0.10,
    "preferences": 0.10,
    "semantic": 0.10,
}

_RANKING_TERM_WEIGHTS: dict[str, float] = {
    "content": 0.17,
    "factual": 0.08,
    "softpref_core": 0.15,
    "softpref_free": 0.07,
    "geo": 0.25,
}


class RankingBreakdown(Protocol):
    content: float
    factual: float
    softpref_core: float
    softpref_free: float
    geo: float
    amenity_match: float


class MatchBreakdown(BaseModel):
    """Traceable unweighted components and the final applied caps."""

    model_config = {"frozen": True}

    location: float | None = Field(default=None, ge=0.0, le=1.0)
    price: float | None = Field(default=None, ge=0.0, le=1.0)
    area: float | None = Field(default=None, ge=0.0, le=1.0)
    rooms: float | None = Field(default=None, ge=0.0, le=1.0)
    property_type: float | None = Field(default=None, ge=0.0, le=1.0)
    preferences: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic: float | None = Field(default=None, ge=0.0, le=1.0)
    active_weight: float = Field(ge=0.0, le=1.0)
    weighted_score: float = Field(ge=0.0, le=1.0)
    hard_filter_cap: int = Field(ge=1, le=100)
    amenity_cap: int = Field(ge=1, le=100)
    match_percent: int = Field(ge=1, le=100)
    result_scope: ResultScope | None = None
    distance_km: float | None = Field(default=None, ge=0.0)

    def public_components(self) -> MatchComponents:
        """Project the criterion scores that are safe to publish."""

        return MatchComponents(
            location=self.location,
            price=self.price,
            area=self.area,
            rooms=self.rooms,
            property_type=self.property_type,
            preferences=self.preferences,
            semantic=self.semantic,
        )


class MatchComponents(BaseModel):
    """The per-criterion view of a match percentage, safe for API consumers.

    Deliberately a strict SUBSET of :class:`MatchBreakdown` rather than the
    whole thing. The full breakdown stays internal (``exclude=True`` on
    ``ScoredCandidate``) because it carries the RAW ``result_scope`` — including
    ``OUT_OF_SCOPE`` — while ``_build_scored_candidates`` sanitises that state on
    the candidate itself (it either rejects the row or rewrites the scope to
    ``EXACT`` and drops the distance). Serialising the nested breakdown would
    route straight around that sanitisation, which is exactly what the exclusion
    exists to prevent. The sanitised scope and distance are already public on the
    parent, so nothing is lost by leaving them out here.

    The weighting internals (``active_weight``, ``weighted_score``, the two
    caps) stay internal too: they describe how the number was assembled, not
    what it means, and publishing them would freeze implementation detail into
    the API contract.

    ``None`` means the query never asked about that criterion — which is a
    different statement from a low score, and the UI must not collapse the two.
    A reader can only be shown what the ranker actually evaluated.
    """

    model_config = {"frozen": True}

    location: float | None = Field(default=None, ge=0.0, le=1.0)
    price: float | None = Field(default=None, ge=0.0, le=1.0)
    area: float | None = Field(default=None, ge=0.0, le=1.0)
    rooms: float | None = Field(default=None, ge=0.0, le=1.0)
    property_type: float | None = Field(default=None, ge=0.0, le=1.0)
    preferences: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic: float | None = Field(default=None, ge=0.0, le=1.0)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _term_raw(breakdown: RankingBreakdown, term: str) -> float:
    return _clamp01(float(getattr(breakdown, term)) / _RANKING_TERM_WEIGHTS[term])


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _range_score(value: float | None, minimum: float | None, maximum: float | None) -> float:
    if value is None:
        return 0.0
    if minimum is not None and value < minimum:
        return _clamp01(value / minimum) if minimum > 0 else 0.0
    if maximum is not None and value > maximum:
        return _clamp01(maximum / value) if value > 0 else 0.0
    return 1.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _district_number(value: Any) -> int | None:
    cleaned = str(value or "").casefold().replace("kerület", "").strip().strip(".")
    if cleaned.isdigit() and 1 <= int(cleaned) <= 23:
        return int(cleaned)
    reverse = {roman.casefold(): int(number) for number, roman in DISTRICT_ROMAN.items()}
    return reverse.get(cleaned)


def _location_component(
    listing: Any, query: ParsedQuery, breakdown: RankingBreakdown
) -> tuple[float | None, ResultScope | None, float | None]:
    hard = query.hard_filters
    geo = query.geo_hints
    has_location = bool(
        hard.city
        or hard.district_in
        or hard.near
        or geo.region_name
        or geo.side_of_river
        or geo.county
        or geo.balaton_shore
    )
    if not has_location:
        return None, None, None

    # A POI-radius query ("... 2 km-es körzetében", "within 2 km of ...") must score by *actual*
    # proximity to the anchor, not by city/district equality. The fast path sets
    # `hard.city = "Budapest"` for such queries, so without this branch every
    # Budapest listing took the exact-city path below and scored location = 1.0 —
    # a listing at the far edge of the radius looked identical to one next door
    # to the POI (a calibration bug: for "within 2 km of Deák tér" a listing in the
    # XIII. district scored 100% against a reference judgement of 10%). The geo ranking term already carries the
    # Gaussian proximity falloff (1.0 at the anchor, ~0.37 at the radius edge),
    # so reuse it. A listing with no coordinates yields geo = 0 here, which the
    # hard-filter cap then limits to 69% — we cannot confirm its proximity.
    if hard.near is not None:
        return _term_raw(breakdown, "geo"), None, None

    if hard.city:
        return _city_location(listing, hard, hard.city)

    return _term_raw(breakdown, "geo"), None, None


def _city_location(listing: Any, hard: HardFilters, city: str) -> tuple[float | None, ResultScope | None, float | None]:
    """The city/district query: an exact hit, a nearby settlement or district, or out of scope."""
    listing_city = str(getattr(listing, "city", "") or "")
    exact_city = listing_city.casefold() == city.casefold()
    accepted_districts = set(district_filter_values(hard.district_in))
    listing_district = str(getattr(listing, "district", "") or "").casefold()
    exact_district = not accepted_districts or listing_district in accepted_districts
    if exact_city and exact_district:
        return 1.0, ResultScope.EXACT, None

    origin = settlement_by_id(hard.city_id) if hard.city_id else None
    candidate = resolve_settlement_name(listing_city).settlement
    if hard.district_in and origin is not None and origin.name == "Budapest":
        origin_number = _district_number(hard.district_in[0])
        candidate_number = _district_number(getattr(listing, "district", None))
        origin = budapest_district(origin_number) if origin_number is not None else None
        candidate = budapest_district(candidate_number) if candidate_number is not None else None
    if origin is None or candidate is None:
        return 0.0, ResultScope.OUT_OF_SCOPE, None
    distance = settlement_distance_km(origin, candidate)
    if distance is None or distance > 30.0:
        return 0.0, ResultScope.OUT_OF_SCOPE, distance
    return location_match_score(ResultScope.NEARBY, distance_km=distance), ResultScope.NEARBY, distance


def _price_component(listing: Any, hard: HardFilters, anchors: Anchors) -> float | None:
    price_scores: list[float] = []
    price = resolve_property_price_huf(listing)
    if hard.price_huf_min is not None or hard.price_huf_max is not None:
        price_scores.append(_range_score(price, hard.price_huf_min, hard.price_huf_max))
    if anchors.price_huf is not None:
        price_scores.append(anchor_score(price, anchors.price_huf, "price_huf") if price is not None else 0.0)
    return _mean(price_scores) if price_scores else None


def _area_component(listing: Any, hard: HardFilters, anchors: Anchors) -> float | None:
    area_scores: list[float] = []
    area = _number(getattr(listing, "area_m2", None))
    if hard.area_m2_min is not None or hard.area_m2_max is not None:
        area_scores.append(_range_score(area, hard.area_m2_min, hard.area_m2_max))
    if anchors.area_m2 is not None:
        area_scores.append(anchor_score(area, anchors.area_m2, "area_m2") if area is not None else 0.0)
    return _mean(area_scores) if area_scores else None


def _rooms_component(listing: Any, hard: HardFilters, anchors: Anchors) -> float | None:
    room_scores: list[float] = []
    rooms = _number(getattr(listing, "rooms", None))
    if hard.rooms is not None:
        room_scores.append(1.0 if rooms == float(hard.rooms) else 0.0)
    if hard.rooms_min is not None:
        room_scores.append(_range_score(rooms, float(hard.rooms_min), None))
    if anchors.rooms is not None:
        room_scores.append(anchor_score(rooms, float(anchors.rooms), "rooms") if rooms is not None else 0.0)
    return _mean(room_scores) if room_scores else None


def _property_type_component(listing: Any, hard: HardFilters) -> float | None:
    property_type: float | None = None
    listing_type = str(getattr(listing, "property_type", "") or "").casefold()
    if hard.property_type_in or hard.property_type_exclude:
        included = not hard.property_type_in or listing_type in {item.casefold() for item in hard.property_type_in}
        excluded = listing_type in {item.casefold() for item in hard.property_type_exclude}
        property_type = 1.0 if included and not excluded else 0.0
    return property_type


def _preferences_component(query: ParsedQuery, breakdown: RankingBreakdown) -> float | None:
    preference_scores: list[float] = []
    if query.soft_preferences:
        preference_scores.append(_term_raw(breakdown, "softpref_core"))
    if query.soft_prefs_freetext:
        preference_scores.append(_term_raw(breakdown, "softpref_free"))
    return _mean(preference_scores) if preference_scores else None


def _semantic_component(query: ParsedQuery, breakdown: RankingBreakdown) -> float | None:
    semantic: float | None = None
    if query.semantic_query:
        semantic = _mean([_term_raw(breakdown, "content"), _term_raw(breakdown, "factual")])
    return semantic


def _hard_filter_cap(components: dict[str, float | None], result_scope: ResultScope | None) -> int:
    hard_values = [
        value
        for name, value in components.items()
        if name in {"location", "price", "area", "rooms", "property_type"} and value is not None
    ]
    hard_filter_cap = 69 if any(value <= 0.0 for value in hard_values) else 100
    if result_scope is ResultScope.OUT_OF_SCOPE:
        hard_filter_cap = min(hard_filter_cap, 49)
    return hard_filter_cap


def compute_match_breakdown(listing: Any, breakdown: RankingBreakdown, query: ParsedQuery) -> MatchBreakdown:
    """Compute one absolute percentage without inspecting any other listing."""

    hard = query.hard_filters
    anchors = query.anchors
    location, result_scope, distance = _location_component(listing, query, breakdown)

    components: dict[str, float | None] = {
        "location": location,
        "price": _price_component(listing, hard, anchors),
        "area": _area_component(listing, hard, anchors),
        "rooms": _rooms_component(listing, hard, anchors),
        "property_type": _property_type_component(listing, hard),
        "preferences": _preferences_component(query, breakdown),
        "semantic": _semantic_component(query, breakdown),
    }
    active_weight = sum(MATCH_COMPONENT_WEIGHTS[name] for name, value in components.items() if value is not None)
    weighted_score = (
        sum(MATCH_COMPONENT_WEIGHTS[name] * value for name, value in components.items() if value is not None)
        / active_weight
        if active_weight > 0
        else 0.0
    )

    hard_filter_cap = _hard_filter_cap(components, result_scope)
    amenity_cap = round(45 + 20 * _clamp01(breakdown.amenity_match)) if breakdown.amenity_match < 1.0 else 100
    percent = max(1, min(100, round(weighted_score * 100), hard_filter_cap, amenity_cap))
    return MatchBreakdown(
        **components,
        active_weight=active_weight,
        weighted_score=weighted_score,
        hard_filter_cap=hard_filter_cap,
        amenity_cap=amenity_cap,
        match_percent=percent,
        result_scope=result_scope,
        distance_km=distance,
    )
