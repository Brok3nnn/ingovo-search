"""Administrative, nearby and distant settlement scopes for the SQL candidate pool."""

from __future__ import annotations

from dataclasses import dataclass

from ingovo_search.recommender.geo_taxonomy import (
    BALATON_AREA_CITIES,
    BALATON_NORTH_SHORE,
    BALATON_SOUTH_SHORE,
    COUNTY_CITIES,
    is_balaton_region_name,
)
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.search_contract import (
    NEARBY_MAX_DISTANCE_KM,
)
from ingovo_search.recommender.settlements import (
    SettlementKind,
    budapest_district,
    nearby_settlements,
    settlement_by_id,
)
from ingovo_search.recommender.sql_filter import DISTRICT_ROMAN


def administrative_geo_city_scope(parsed: ParsedQuery) -> tuple[str, ...] | None:
    """Resolve county/Balaton cues to an eligible settlement set.

    Concrete city, district and radius filters already have stricter SQL paths.
    For administrative/curated multi-settlement geography, intersect every
    active cue before scoring so an out-of-scope city cannot survive on a weak
    geo score alone.
    """
    hard = parsed.hard_filters
    if hard.city is not None or hard.district_in or hard.near is not None:
        return None

    scopes: list[set[str]] = []
    geo = parsed.geo_hints
    if geo.county is not None:
        scopes.append(set(COUNTY_CITIES.get(geo.county, frozenset())))
    if geo.balaton_shore == "Északi":
        scopes.append(set(BALATON_NORTH_SHORE))
    elif geo.balaton_shore == "Déli":
        scopes.append(set(BALATON_SOUTH_SHORE))
    elif is_balaton_region_name(geo.region_name):
        scopes.append(set(BALATON_AREA_CITIES))

    if not scopes:
        return None
    eligible = scopes[0]
    for scope in scopes[1:]:
        eligible &= scope
    return tuple(sorted(eligible))


@dataclass(frozen=True, slots=True)
class NearbyFilterScope:
    """SQL administrative scope derived from verified settlement distances."""

    cities: tuple[str, ...]
    districts: tuple[str, ...] | None = None


def district_number(value: str) -> int | None:
    cleaned = value.casefold().replace("kerület", "").strip().strip(".")
    if cleaned.isdigit() and 1 <= int(cleaned) <= 23:
        return int(cleaned)
    reverse_roman = {roman.casefold(): int(number) for number, roman in DISTRICT_ROMAN.items()}
    return reverse_roman.get(cleaned)


def nearby_filter_scope(parsed: ParsedQuery) -> NearbyFilterScope | None:
    """Build an exact, verified <=30 km administrative candidate scope."""

    hard = parsed.hard_filters
    if hard.city_id is None or hard.city is None:
        return None

    origin = settlement_by_id(hard.city_id)
    if origin is None:
        return None
    if origin.name == "Budapest" and len(hard.district_in) == 1:
        district_no = district_number(hard.district_in[0])
        if district_no is None:
            return None
        origin = budapest_district(district_no)
        if origin is None:
            return None

    neighbours = nearby_settlements(origin.id, NEARBY_MAX_DISTANCE_KM)
    if origin.kind is SettlementKind.BUDAPEST_DISTRICT:
        districts = tuple(
            str(item.settlement.budapest_district)
            for item in neighbours
            if item.settlement.budapest_district is not None
        )
        return NearbyFilterScope(cities=("Budapest",), districts=districts) if districts else None

    cities = tuple(item.settlement.name for item in neighbours)
    return NearbyFilterScope(cities=cities) if cities else None


def ordered_distant_city_scope(parsed: ParsedQuery) -> tuple[str, ...]:
    """Return verified settlements beyond the automatic 30 km pool, nearest first.

    The caller probes prefixes of this tuple. Because every prefix contains all
    settlements closer than its last item, the first non-empty prefix identifies
    the geographically closest settlement that has matching inventory.

    A Budapest district search intentionally uses Budapest itself as the origin
    here: the <=30 km phase already probes the other districts, while this phase
    needs municipality-level alternatives such as Budaörs or Szentendre.
    """

    hard = parsed.hard_filters
    if hard.city_id is None or hard.city is None:
        return ()
    origin = settlement_by_id(hard.city_id)
    if origin is None:
        return ()
    return tuple(
        item.settlement.name
        for item in nearby_settlements(origin.id, float("inf"))
        if item.distance_km > NEARBY_MAX_DISTANCE_KM
    )
