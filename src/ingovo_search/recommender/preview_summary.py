"""The preview's one-line "understood" summary of the strict constraints.

The hard filters — transaction, property type, city, district or river side or region, rooms,
area, price ceiling or anchor, excluded types — read in the locale's words, joined with " · ",
with the retained-constraint keys beside them. The exact direction, every preference direction,
the progress event and the homepage probe all show this one phrasing. Each section is a small
function yielding at most one (phrase, key) pair, in the order the sentence reads.
"""

from __future__ import annotations

from ingovo_search.lib.property_price import MAX_LEGACY_SAFE_INTEGER
from ingovo_search.recommender.parser_schema import Anchors, HardFilters, ParsedQuery
from ingovo_search.recommender.preview_models import PreviewLocale


def _format_huf(value: int, locale: PreviewLocale) -> str:
    if value > MAX_LEGACY_SAFE_INTEGER:
        formatted = f"{value:,}".replace(",", " ")
        return f"{formatted} Ft" if locale == "hu" else f"HUF {formatted}"
    if value >= 1_000_000:
        amount = value / 1_000_000
        number = f"{amount:g}".replace(".", "," if locale == "hu" else ".")
        return f"{number} M Ft" if locale == "hu" else f"HUF {number}M"
    formatted = f"{value:,}".replace(",", " ")
    return f"{formatted} Ft" if locale == "hu" else f"HUF {formatted}"


def _format_number(value: float) -> str:
    return f"{value:g}".replace(".", ",")


# One summary section: the phrase shown and the retained-constraint key, or nothing.
type _Entry = tuple[str, str] | None


def _transaction_entry(hard: HardFilters, locale: PreviewLocale) -> _Entry:
    if not hard.transaction_type:
        return None
    part = (
        ("Eladó" if hard.transaction_type == "sale" else "Kiadó") if locale == "hu" else hard.transaction_type.title()
    )
    return part, f"transaction:{hard.transaction_type}"


def _property_type_entry(hard: HardFilters, locale: PreviewLocale) -> _Entry:
    if not hard.property_type_in:
        return None
    type_labels = ", ".join(hard.property_type_in)
    if locale == "en":
        type_labels = type_labels.replace("Lakás", "Apartment").replace("Ház", "House").replace("Iroda", "Office")
    return type_labels, "property_type:" + ",".join(hard.property_type_in)


def _city_entry(hard: HardFilters) -> _Entry:
    if not hard.city:
        return None
    return hard.city, f"city:{hard.city}"


def _location_entry(parsed: ParsedQuery, locale: PreviewLocale) -> _Entry:
    """The districts, else the river side, else the named region."""
    hard = parsed.hard_filters
    if hard.district_in:
        district = ", ".join(
            f"{value}. kerület" if locale == "hu" else f"District {value}" for value in hard.district_in
        )
        return district, "district:" + ",".join(hard.district_in)
    if parsed.geo_hints.side_of_river:
        return parsed.geo_hints.side_of_river, f"river_side:{parsed.geo_hints.side_of_river}"
    if parsed.geo_hints.region_name:
        return parsed.geo_hints.region_name, f"region:{parsed.geo_hints.region_name}"
    return None


def _rooms_entry(hard: HardFilters, anchors: Anchors, locale: PreviewLocale) -> _Entry:
    """The exact room count, else the minimum, else the anchor."""
    if hard.rooms is not None:
        part = f"{hard.rooms} szoba" if locale == "hu" else f"{hard.rooms} bedrooms"
        return part, f"rooms:{hard.rooms}"
    if hard.rooms_min is not None:
        part = f"min. {hard.rooms_min} szoba" if locale == "hu" else f"min. {hard.rooms_min} bedrooms"
        return part, f"rooms_min:{hard.rooms_min}"
    if anchors.rooms is not None:
        part = f"kb. {anchors.rooms} szoba" if locale == "hu" else f"around {anchors.rooms} bedrooms"
        return part, f"rooms_anchor:{anchors.rooms}"
    return None


def _area_entry(hard: HardFilters) -> _Entry:
    if hard.area_m2_min is None and hard.area_m2_max is None:
        return None
    if hard.area_m2_min is not None and hard.area_m2_max is not None:
        area = f"{_format_number(hard.area_m2_min)}-{_format_number(hard.area_m2_max)} m²"
    elif hard.area_m2_min is not None:
        area = f"min. {_format_number(hard.area_m2_min)} m²"
    else:
        area = f"max. {_format_number(hard.area_m2_max or 0)} m²"
    return area, f"area:{hard.area_m2_min or ''}:{hard.area_m2_max or ''}"


def _price_entry(hard: HardFilters, anchors: Anchors, locale: PreviewLocale) -> _Entry:
    """The price ceiling, else the price anchor."""
    price = hard.price_huf_max or anchors.price_huf
    if price is None:
        return None
    prefix = "max." if hard.price_huf_max is not None else ("kb." if locale == "hu" else "around")
    part = f"{prefix} {_format_huf(price, locale)}"
    return part, f"price:{'max' if hard.price_huf_max is not None else 'anchor'}:{price}"


def _excluded_type_entry(hard: HardFilters, locale: PreviewLocale) -> _Entry:
    if not hard.property_type_exclude:
        return None
    excluded = ", ".join(hard.property_type_exclude)
    part = f"nem {excluded}" if locale == "hu" else f"excluding {excluded}"
    return part, "excluded_type:" + ",".join(hard.property_type_exclude)


def constraint_summary_with_keys(parsed: ParsedQuery, locale: PreviewLocale) -> tuple[str, list[str]]:
    """The one-line summary of the strict constraints and the retained-constraint keys behind it."""
    hard = parsed.hard_filters
    anchors = parsed.anchors
    entries = (
        _transaction_entry(hard, locale),
        _property_type_entry(hard, locale),
        _city_entry(hard),
        _location_entry(parsed, locale),
        _rooms_entry(hard, anchors, locale),
        _area_entry(hard),
        _price_entry(hard, anchors, locale),
        _excluded_type_entry(hard, locale),
    )
    present = [entry for entry in entries if entry is not None]
    parts = [part for part, _ in present]
    keys = [key for _, key in present]

    fallback = "A megadott keresési feltételek" if locale == "hu" else "Your stated search criteria"
    return " · ".join(parts) if parts else fallback, keys


def constraint_summary(parsed: ParsedQuery, locale: PreviewLocale) -> str:
    """Public one-line "understood" summary of the strict constraints.

    Thin wrapper over :func:`constraint_summary_with_keys` for callers that show
    only the sentence (the homepage probe), so they reuse the same canonical
    phrasing.
    """
    return constraint_summary_with_keys(parsed, locale)[0]
