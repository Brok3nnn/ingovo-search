"""§7 parser output schema — `ParsedQuery`.

Single source of truth for the Hungarian-NL → JSON query parser output. The
shape mirrors the design spec §6.4 "Parser output schema" exactly. Both
`parser.py` and the §17.1 eval runner import `ParsedQuery` from here so they
validate against identical types.

The parser splits a query into three buckets by linguistic cue:
- `hard_filters` — strict constraints applied as a SQL WHERE clause.
- `anchors` — Gaussian-distance scoring targets ("80 millió körül"), not cutoffs.
- `soft_preferences` / `soft_prefs_freetext` — fixed-vocab tags + long-tail phrases.

Every model is frozen and every field carries a default, so a structurally
incomplete LLM response still validates leniently at the trust boundary.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ingovo_search.lib.property_price import json_safe_huf

type TransactionType = Literal["sale", "rent"]
type SideOfRiver = Literal["Buda", "Pest"]
type BalatonShore = Literal["Északi", "Déli"]
type LocationResolutionStatus = Literal["resolved", "ambiguous", "not_found"]
type PropertyType = Literal[
    "Lakás", "Ház", "Iroda", "Üzlethelyiség", "Telek", "Garázs", "Ipari", "Nyaraló", "Vendéglátás", "Mezőgazdasági"
]

# Locked v0 soft-preference vocabulary — see the design spec Appendix A.
# Tags outside this set go into `ParsedQuery.soft_prefs_freetext` verbatim.
type SoftPrefTag = Literal[
    "renovated",
    "partially_renovated",
    "new_build",
    "fixer_upper",
    "parking",
    "garage",
    "garden",
    "terrace",
    "balcony",
    "basement",
    "pool",
    "elevator",
    "air_conditioning",
    "natural_light",
    "sunny",
    "view",
    "central",
    "quiet",
    "near_metro",
    "walkable",
    "large",
    "small",
]

SOFT_PREF_TAGS: tuple[SoftPrefTag, ...] = (
    "renovated",
    "partially_renovated",
    "new_build",
    "fixer_upper",
    "parking",
    "garage",
    "garden",
    "terrace",
    "balcony",
    "basement",
    "pool",
    "elevator",
    "air_conditioning",
    "natural_light",
    "sunny",
    "view",
    "central",
    "quiet",
    "near_metro",
    "walkable",
    "large",
    "small",
)


def _empty_softpref_list() -> list[SoftPrefTag]:
    """Typed empty-list factory for the `soft_preferences` field default."""
    return []


def _empty_property_type_list() -> list[PropertyType]:
    """Typed empty-list factory for property type filter defaults."""
    return []


class NearFilter(BaseModel):
    """Explicit POI + radius — the only geographic constraint treated as strict."""

    model_config = {"frozen": True}

    lat: float
    lng: float
    radius_km: float


class HardFilters(BaseModel):
    """Strict constraints, applied verbatim as a SQL WHERE clause.

    Only formal administrative geography is strict here: `city`, `district_in`
    (Budapest districts 1-23 and 1-to-1 aliases such as Újpest -> "4"), and
    `near` radii. Fuzzy named regions belong in `GeoHints`, never here.
    """

    model_config = {"frozen": True}

    city: str | None = None
    city_id: str | None = None
    district_in: list[str] = Field(default_factory=list)
    transaction_type: TransactionType | None = None
    property_type_in: list[PropertyType] = Field(default_factory=_empty_property_type_list)
    property_type_exclude: list[PropertyType] = Field(default_factory=_empty_property_type_list)
    rooms: int | None = None
    rooms_min: int | None = None
    area_m2_min: float | None = None
    area_m2_max: float | None = None
    price_huf_min: int | None = None
    price_huf_max: int | None = None
    floor_min: int | None = None
    near: NearFilter | None = None


class Anchors(BaseModel):
    """Gaussian-distance scoring targets — "körül" / "kb." / "nagyjából" land here."""

    model_config = {"frozen": True}

    price_huf: int | None = None
    area_m2: float | None = None
    rooms: int | None = None


class GeoHints(BaseModel):
    """Fuzzy geography — soft-scored against learned centroids, never filtered."""

    model_config = {"frozen": True}

    region_name: str | None = None
    side_of_river: SideOfRiver | None = None
    # Hungarian county (vármegye) — canonical base name without the " megye"
    # suffix, e.g. "Pest", "Veszprém", "Bács-Kiskun". Resolved against
    # `geo_taxonomy.COUNTY_CITIES` at scoring time, not as a hard filter.
    county: str | None = None
    # Balaton lakeside — "Északi" / "Déli". Resolved against
    # `geo_taxonomy.BALATON_NORTH_SHORE` / `BALATON_SOUTH_SHORE`.
    balaton_shore: BalatonShore | None = None


# POI proximity vocabulary. The first ten mirror `ingestion.poi_osm.POI_CATEGORIES`
# (the `poi` table's category values); "transit" is a parser-level aggregate the
# scorer resolves as min(metro, tram, bus) — users say "tömegközlekedés" far more
# often than a specific mode.
type PoiCategory = Literal[
    "school",
    "kindergarten",
    "metro",
    "tram",
    "bus",
    "supermarket",
    "pharmacy",
    "park",
    "hospital",
    "playground",
    "transit",
]


class PoiNear(BaseModel):
    """One POI-proximity request — "iskola legyen a közelben".

    `max_walk_minutes` is set only when the query quantifies it ("5 perc alatt
    gyalog elérhető"); scoring converts minutes to metres via ~80 m/min ÷ 1.3
    detour factor. Always a soft scoring signal, never a hard filter — POI
    distances are derived data and missing values must not exclude listings.
    """

    model_config = {"frozen": True}

    category: PoiCategory
    max_walk_minutes: int | None = None


def _empty_poi_near_list() -> list[PoiNear]:
    return []


class ParseConfidence(BaseModel):
    """Confidence scores for parsed fields."""

    model_config = {"frozen": True}

    hard_filters: float | None = None
    anchors: float | None = None
    soft_preferences: float | None = None
    geo_hints: float | None = None
    rooms: float | None = None
    district_in: float | None = None
    price_huf_max: float | None = None
    x: float | None = None


class LocationCandidate(BaseModel):
    """One deterministic settlement candidate exposed for clarification."""

    model_config = {"frozen": True}

    settlement_id: str
    name: str
    county: str


def _empty_location_candidates() -> list[LocationCandidate]:
    return []


class LocationResolution(BaseModel):
    """Deterministic location state; ambiguous input must not be selected silently."""

    model_config = {"frozen": True}

    status: LocationResolutionStatus
    candidates: list[LocationCandidate] = Field(default_factory=_empty_location_candidates)
    matched_text: str | None = None


class ParsedQuery(BaseModel):
    """Structured form of a Hungarian natural-language property query."""

    model_config = {"frozen": True}

    hard_filters: HardFilters = Field(default_factory=HardFilters)
    anchors: Anchors = Field(default_factory=Anchors)
    soft_preferences: list[SoftPrefTag] = Field(default_factory=_empty_softpref_list)
    soft_prefs_freetext: list[str] = Field(default_factory=list)
    poi_near: list[PoiNear] = Field(default_factory=_empty_poi_near_list)
    geo_hints: GeoHints = Field(default_factory=GeoHints)
    semantic_query: str = ""
    conflict_flag: bool = False
    parse_confidence: ParseConfidence | None = None
    location_resolution: LocationResolution | None = None
    allow_distant_location_expansion: bool = False


def public_hard_filters_json(hard_filters: HardFilters) -> dict[str, Any]:
    """Return one hard-filter object with unsafe HUF values as strings."""

    payload = hard_filters.model_dump()
    for field in ("price_huf_min", "price_huf_max"):
        value = payload.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            payload[field] = json_safe_huf(value)
    return payload


def public_anchors_json(anchors: Anchors) -> dict[str, Any]:
    """Return one anchor object with an unsafe HUF amount as a string."""

    payload = anchors.model_dump()
    anchor_price = payload.get("price_huf")
    if isinstance(anchor_price, int) and not isinstance(anchor_price, bool):
        payload["price_huf"] = json_safe_huf(anchor_price)
    return payload


def public_parsed_query_json(query: ParsedQuery) -> dict[str, Any]:
    """Return the browser contract for a parsed query without rounding HUF.

    ``ParsedQuery`` stays an integer-only domain object so SQL filters and
    ranking continue to receive ordinary Python integers.  The two public
    homepage responses expose it verbatim, though, and JSON numbers above
    JavaScript's safe-integer boundary would be rounded by the browser before
    the HUF renderer ever sees them.  Only the monetary query fields cross that
    boundary as decimal strings; all non-monetary intent fields retain their
    existing response shape.
    """

    payload = query.model_dump()
    payload["hard_filters"] = public_hard_filters_json(query.hard_filters)
    payload["anchors"] = public_anchors_json(query.anchors)
    return payload
