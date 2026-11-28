"""Geo scoring — pure soft-scoring math for fuzzy geographic terms.

`geo_score` blends three components:
proximity to an explicit POI anchor, a learned/cold-start named-region match,
and which side of the Danube the listing sits on. The named-region component
is DB- and LLM-backed (`region_centroid.matches_named_region`); it is computed
upstream and passed in here as `named_region_score`, keeping this module pure
and fully unit-testable.

When a query supplies only some of the three geographic cues, the score is the
weighted average over the *present* components (weights renormalised), so an
absent cue neither helps nor penalises a listing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast

from ingovo_search.recommender.geo_taxonomy import (
    is_balaton_region_name,
    matches_balaton_area,
    matches_balaton_shore,
    matches_county,
)
from ingovo_search.recommender.parser_schema import ParsedQuery, PoiNear

EARTH_RADIUS_KM = 6371.0088

# Budapest districts on the Buda (west) bank of the Danube; the rest are Pest.
# Hard-coded, not learned — administrative fact, not engagement signal.
BUDA_DISTRICTS: frozenset[str] = frozenset({"1", "2", "3", "11", "12", "22"})

# Component weights. Proximity and named-region remain the strongest
# cues; administrative-boundary cues (river side, county, Balaton shore) are
# precise but coarse, so they each contribute a moderate, comparable weight.
ALPHA_PROXIMITY = 0.5
BETA_NAMED_REGION = 0.35
GAMMA_RIVER_SIDE = 0.15
DELTA_COUNTY = 0.2
EPSILON_BALATON_SHORE = 0.15
ZETA_BALATON_AREA = 0.55
# POI proximity is an explicit user request ("iskola legyen a közelben"), so
# it weighs close to the explicit near-anchor cue.
ETA_POI_NEAR = 0.45

# poi_near scoring constants. Distances come from the nightly job's
# `features["poi_m"]` map (metres to the nearest POI per category, 2 km cap).
DEFAULT_POI_TARGET_M = 500.0
WALK_METRES_PER_MINUTE = 80.0
WALK_DETOUR_FACTOR = 1.3  # straight-line target = walked metres / detour
# Missing POI data contributes neutrally — derived data must never exclude or
# crater a listing (missing-data policy).
NEUTRAL_POI_SCORE = 0.5
TRANSIT_MODES: tuple[str, ...] = ("metro", "tram", "bus")


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two WGS84 points, in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def proximity_score(distance_km: float, radius_km: float) -> float:
    """Gaussian falloff for an explicit POI anchor: `exp(-(distance/radius)^2)`.

    Equals 1.0 at the anchor and ~0.37 at exactly `radius_km` away.
    """
    if radius_km <= 0:
        return 0.0
    return math.exp(-((distance_km / radius_km) ** 2))


def gaussian_region_score(distance_km: float, spread_km: float | None) -> float:
    """Gaussian on distance from a learned region centroid: `exp(-(distance/spread)^2)`."""
    if spread_km is None or spread_km <= 0:
        return 0.0
    return math.exp(-((distance_km / spread_km) ** 2))


def matches_river_side(district: str | None, side_of_river: str | None) -> float:
    """Return 1.0 when the listing's district is on the requested bank, else 0.0."""
    if district is None or side_of_river is None:
        return 0.0
    is_buda = district in BUDA_DISTRICTS
    if side_of_river == "Buda":
        return 1.0 if is_buda else 0.0
    if side_of_river == "Pest":
        return 0.0 if is_buda else 1.0
    return 0.0


def walk_minutes_to_metres(minutes: float) -> float:
    """'5 perc gyalog' -> straight-line metres (~80 m/min walked, 1.3 detour)."""
    return minutes * WALK_METRES_PER_MINUTE / WALK_DETOUR_FACTOR


def poi_distances_from_features(features: object) -> dict[str, float] | None:
    """Extract the nightly job's `poi_m` map from a listing's `features` JSONB."""
    if not isinstance(features, dict):
        return None
    features_map = cast("dict[str, object]", features)
    poi_m = features_map.get("poi_m")
    if not isinstance(poi_m, dict):
        return None
    poi_map = cast("dict[object, object]", poi_m)
    distances: dict[str, float] = {}
    for category, value in poi_map.items():
        if isinstance(category, str) and isinstance(value, (int, float)):
            distances[category] = float(value)
    return distances or None


def poi_distance_m(distances: dict[str, float] | None, category: str) -> float | None:
    """Nearest-POI metres for one requested category; 'transit' = best of any mode."""
    if not distances:
        return None
    if category == "transit":
        mode_distances = [distances[mode] for mode in TRANSIT_MODES if mode in distances]
        return min(mode_distances) if mode_distances else None
    return distances.get(category)


def poi_proximity_score(
    poi_near: Sequence[PoiNear],
    distances: dict[str, float] | None,
) -> float | None:
    """Graded 0..1 score over the query's POI requests; `None` when none requested.

    Each request scores `exp(-(distance/target)^2)` — 1.0 on top of the POI,
    ~0.37 at exactly the target distance. The target is the quantified walking
    time when given ("5 perc alatt gyalog" -> ~308 m) or `DEFAULT_POI_TARGET_M`
    for plain "közelben". A request with no distance data (category absent
    within the nightly job's radius, or no POI data at all) contributes
    `NEUTRAL_POI_SCORE` so missing derived data never zeroes a listing.
    """
    if not poi_near:
        return None
    total = 0.0
    for request in poi_near:
        distance = poi_distance_m(distances, request.category)
        if distance is None:
            total += NEUTRAL_POI_SCORE
            continue
        target = (
            walk_minutes_to_metres(request.max_walk_minutes)
            if request.max_walk_minutes is not None and request.max_walk_minutes > 0
            else DEFAULT_POI_TARGET_M
        )
        total += math.exp(-((distance / target) ** 2))
    return total / len(poi_near)


@dataclass(frozen=True, slots=True)
class ListingGeo:
    """Where one listing is, as the geographic scoring reads it; any part may be unknown.

    `poi_distances_m` is the nightly POI job's `features["poi_m"]` map
    (`poi_distances_from_features`).
    """

    lat: float | None = None
    lng: float | None = None
    district: str | None = None
    city: str | None = None
    poi_distances_m: dict[str, float] | None = None


NO_LISTING_GEO: Final = ListingGeo()


def geo_score(
    query: ParsedQuery,
    listing: ListingGeo = NO_LISTING_GEO,
    *,
    named_region_score: float | None = None,
) -> float:
    """Blend the geographic soft-score for one listing under one parsed query.

    `named_region_score` is the precomputed output of
    `region_centroid.matches_named_region` (DB/LLM-backed); pass `None` when the
    query has no `geo_hints.region_name`. Components whose query cue is absent
    are dropped and the remaining weights renormalised. Returns 0.0 when the
    query carries no geographic cue at all.
    """
    components: list[tuple[float, float]] = []  # (weight, value)

    near = query.hard_filters.near
    if near is not None and listing.lat is not None and listing.lng is not None:
        distance = haversine_km(near.lat, near.lng, listing.lat, listing.lng)
        components.append((ALPHA_PROXIMITY, proximity_score(distance, near.radius_km)))

    if is_balaton_region_name(query.geo_hints.region_name):
        components.append((ZETA_BALATON_AREA, matches_balaton_area(listing.city)))
    elif query.geo_hints.region_name is not None and named_region_score is not None:
        components.append((BETA_NAMED_REGION, named_region_score))

    if query.geo_hints.side_of_river is not None:
        components.append((GAMMA_RIVER_SIDE, matches_river_side(listing.district, query.geo_hints.side_of_river)))

    if query.geo_hints.county is not None:
        components.append((DELTA_COUNTY, matches_county(listing.city, query.geo_hints.county)))

    if query.geo_hints.balaton_shore is not None:
        components.append((EPSILON_BALATON_SHORE, matches_balaton_shore(listing.city, query.geo_hints.balaton_shore)))

    poi_value = poi_proximity_score(query.poi_near, listing.poi_distances_m)
    if poi_value is not None:
        components.append((ETA_POI_NEAR, poi_value))

    if not components:
        return 0.0
    total_weight = sum(weight for weight, _ in components)
    return sum(weight * value for weight, value in components) / total_weight
