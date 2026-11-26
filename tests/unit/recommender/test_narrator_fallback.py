"""The heuristic narrative: compact prices and the four hard-filter chips."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ingovo_search.recommender.narrator_fallback import fallback_narrative, format_price_short
from ingovo_search.recommender.parser_schema import HardFilters, ParsedQuery


def _listing(**fields: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "id": "p",
        "title": "Világos lakás",
        "property_type": "Lakás",
        "transaction_type": "sale",
        "city": "Budapest",
        "district": "XIII",
        "area_m2": 60.0,
        "rooms": "2",
        "price_huf": 50_000_000,
        "currency": "HUF",
        "year_built": None,
        "renovation_state": "felújított",
        "description": None,
    }
    return SimpleNamespace(**{**base, **fields})


def _chips(listing: SimpleNamespace, **filters: object) -> dict[str, str]:
    query = ParsedQuery(hard_filters=HardFilters.model_validate(filters))
    return {chip.group: chip.status for chip in fallback_narrative(listing, query).rationale_chips}


@pytest.mark.parametrize(
    ("price", "is_rent", "expected"),
    [
        (None, False, "ár nélkül"),
        (45_000_000, False, "45 M Ft"),
        (1_500_000, True, "1,5 M Ft / hó"),
        (950_000, False, "950 000 Ft"),
        (2_000_000.0, False, "2 M Ft"),
        (9_007_199_254_740_993, True, "9 007 199 254 740 993 Ft / hó"),
    ],
)
def test_format_price_short(price: int | float | None, is_rent: bool, expected: str) -> None:
    assert format_price_short(price, is_rent=is_rent) == expected


def test_location_chip_misses_another_city_and_warns_on_another_district() -> None:
    assert _chips(_listing(), city="Debrecen")["location"] == "miss"
    assert _chips(_listing(), city=" budapest ", district_in=["V"])["location"] == "warning"
    assert _chips(_listing(district=None), city="Budapest", district_in=["XIII"])["location"] == "warning"
    assert _chips(_listing(), city="Budapest", district_in=["XIII"])["location"] == "match"


def test_size_chip_follows_the_area_bounds_and_the_room_count() -> None:
    assert _chips(_listing(), area_m2_min=70.0)["size"] == "miss"
    assert _chips(_listing(), area_m2_max=50.0)["size"] == "miss"
    assert _chips(_listing(), rooms=3)["size"] == "warning"
    assert _chips(_listing(rooms="sok"), rooms=2)["size"] == "warning"
    assert _chips(_listing(rooms="2 + 1 fél"), rooms=2, area_m2_min=55.0, area_m2_max=65.0)["size"] == "match"


def test_price_chip_allows_ten_percent_over_budget_as_a_warning() -> None:
    assert _chips(_listing(), price_huf_max=50_000_000)["price"] == "match"
    assert _chips(_listing(), price_huf_max=46_000_000)["price"] == "warning"
    assert _chips(_listing(), price_huf_max=40_000_000)["price"] == "miss"
    assert _chips(_listing(price_huf=None))["price"] == "match"


@pytest.mark.parametrize(
    ("state", "expected"),
    [("Új építésű ", "match"), ("felujitott", "match"), ("felújítandó", "miss"), ("jó", "warning"), (None, "warning")],
)
def test_renovation_chip_reads_the_listing_state(state: str | None, expected: str) -> None:
    assert _chips(_listing(renovation_state=state))["renovation"] == expected


def test_fallback_narrative_text_names_every_missed_group() -> None:
    query = ParsedQuery(hard_filters=HardFilters(city="Debrecen", price_huf_max=40_000_000))

    narrative = fallback_narrative(_listing(transaction_type="rent", price_huf=250_000), query)

    assert narrative.headline == "Lakás — Budapest, XIII. kerület"
    assert narrative.body == "Világos lakás, Budapest, XIII. kerület, 250 000 Ft / hó. 60 m² · 2 szoba."
    assert narrative.compromise == "Kompromisszum a következő szempont(ok)ban: lokáció."
    assert narrative.vs_next is None


def test_fallback_narrative_without_size_facts_has_a_one_sentence_body() -> None:
    narrative = fallback_narrative(_listing(area_m2=None, rooms=None, title=None), ParsedQuery())

    assert narrative.body == "Lakás, Budapest, XIII. kerület, 50 M Ft."
    assert narrative.compromise == ""
