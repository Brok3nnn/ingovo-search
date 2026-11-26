"""The search-context chips: the parsed query as stable, user-facing labels."""

from __future__ import annotations

from dataclasses import dataclass, field

from ingovo_search.lib.property_price import MAX_LEGACY_SAFE_INTEGER
from ingovo_search.recommender.parser_schema import Anchors, GeoHints, HardFilters, ParsedQuery
from ingovo_search.search.constants import (
    AUTOMATIC_LOCATION_FALLBACK_LABEL,
    EXPLICIT_LOCATION_EXPANSION_LABEL,
    SOFT_PREFERENCE_LABELS,
)
from ingovo_search.search.models import SearchContextItem


def format_huf(value: int) -> str:
    # Public search labels must not round canonical BigInts through float
    # million formatting. The compact form remains for values JavaScript can
    # represent exactly; otherwise show the full exact HUF integer.
    if value > MAX_LEGACY_SAFE_INTEGER:
        return f"{value:,}".replace(",", " ") + " Ft"
    if value >= 1_000_000:
        millions = value / 1_000_000
        formatted = str(int(millions)) if millions.is_integer() else f"{millions:.1f}".replace(".", ",")
        return f"{formatted} M Ft"
    return f"{value:,}".replace(",", " ") + " Ft"


def format_number(value: float) -> str:
    return f"{value:g}".replace(".", ",")


@dataclass(slots=True)
class _ContextItems:
    """The labels collected so far, and which relaxation names mark an item as relaxed."""

    relaxed: set[str]
    items: list[SearchContextItem] = field(default_factory=list[SearchContextItem])

    def add(self, key: str, label: str, *, relaxed_by: tuple[str, ...] = ()) -> None:
        self.items.append(
            SearchContextItem(key=key, label=label, relaxed=any(name in self.relaxed for name in relaxed_by))
        )


def _add_type_and_location(ctx: _ContextItems, hard: HardFilters) -> None:
    if hard.transaction_type:
        ctx.add("transaction", "Eladó" if hard.transaction_type == "sale" else "Kiadó")
    if hard.property_type_in:
        ctx.add("property_type", ", ".join(hard.property_type_in))

    location_parts: list[str] = []
    if hard.city:
        location_parts.append(hard.city)
    if hard.district_in:
        location_parts.append(", ".join(f"{district}. kerület" for district in hard.district_in))
    if location_parts:
        ctx.add(
            "location",
            " · ".join(location_parts),
            relaxed_by=(EXPLICIT_LOCATION_EXPANSION_LABEL, AUTOMATIC_LOCATION_FALLBACK_LABEL),
        )


def _add_price(ctx: _ContextItems, hard: HardFilters, anchors: Anchors) -> None:
    if hard.price_huf_min is not None and hard.price_huf_max is not None:
        ctx.add(
            "price",
            f"{format_huf(hard.price_huf_min)} - {format_huf(hard.price_huf_max)}",
            relaxed_by=("max ár",),
        )
    elif hard.price_huf_max is not None:
        ctx.add("price", f"Max. {format_huf(hard.price_huf_max)}", relaxed_by=("max ár",))
    elif hard.price_huf_min is not None:
        ctx.add("price", f"Min. {format_huf(hard.price_huf_min)}")
    elif anchors.price_huf is not None:
        ctx.add("price", f"Kb. {format_huf(anchors.price_huf)}")


def _add_area(ctx: _ContextItems, hard: HardFilters, anchors: Anchors) -> None:
    if hard.area_m2_min is not None and hard.area_m2_max is not None:
        ctx.add(
            "area",
            f"{format_number(hard.area_m2_min)}-{format_number(hard.area_m2_max)} m²",
            relaxed_by=("méret",),
        )
    elif hard.area_m2_min is not None:
        ctx.add("area", f"Min. {format_number(hard.area_m2_min)} m²", relaxed_by=("méret",))
    elif hard.area_m2_max is not None:
        ctx.add("area", f"Max. {format_number(hard.area_m2_max)} m²", relaxed_by=("méret",))
    elif anchors.area_m2 is not None:
        ctx.add("area", f"Kb. {format_number(anchors.area_m2)} m²")


def _add_rooms_floor_and_exclusions(ctx: _ContextItems, hard: HardFilters, anchors: Anchors) -> None:
    if hard.rooms is not None:
        ctx.add("rooms", f"{hard.rooms} szoba", relaxed_by=("pontos szobaszám",))
    elif hard.rooms_min is not None:
        ctx.add("rooms", f"Min. {hard.rooms_min} szoba", relaxed_by=("min. szobaszám",))
    elif anchors.rooms is not None:
        ctx.add("rooms", f"Kb. {anchors.rooms} szoba")

    if hard.floor_min is not None:
        ctx.add("floor", f"Min. {hard.floor_min}. emelet", relaxed_by=("min. emelet",))
    if hard.property_type_exclude:
        ctx.add(
            "excluded_types",
            f"Nem: {', '.join(hard.property_type_exclude)}",
            relaxed_by=("kizárt típusok",),
        )
    if hard.near is not None:
        ctx.add("distance", f"{format_number(hard.near.radius_km)} km-en belül", relaxed_by=("közelség-sugár",))


def _add_geo(ctx: _ContextItems, geo: GeoHints) -> None:
    if geo.region_name:
        ctx.add("region", geo.region_name)
    if geo.side_of_river:
        ctx.add("river_side", geo.side_of_river)
    if geo.county:
        ctx.add("county", f"{geo.county} vármegye")
    if geo.balaton_shore:
        ctx.add("balaton_shore", f"Balaton {geo.balaton_shore.lower()} part")


def _add_preferences(ctx: _ContextItems, parsed: ParsedQuery) -> None:
    seen_preferences: set[str] = set()
    for preference in [*parsed.soft_preferences, *parsed.soft_prefs_freetext]:
        normalized = preference.strip()
        if not normalized or normalized.casefold() in seen_preferences:
            continue
        seen_preferences.add(normalized.casefold())
        ctx.add(f"preference_{len(seen_preferences)}", SOFT_PREFERENCE_LABELS.get(normalized, normalized))


def build_search_context(parsed: ParsedQuery, relaxation_path: list[str]) -> list[SearchContextItem]:
    """Turn parser output into stable UI labels without exposing parser internals."""
    ctx = _ContextItems(relaxed=set(relaxation_path))
    hard = parsed.hard_filters
    anchors = parsed.anchors
    _add_type_and_location(ctx, hard)
    _add_price(ctx, hard, anchors)
    _add_area(ctx, hard, anchors)
    _add_rooms_floor_and_exclusions(ctx, hard, anchors)
    _add_geo(ctx, parsed.geo_hints)
    _add_preferences(ctx, parsed)
    return ctx.items
