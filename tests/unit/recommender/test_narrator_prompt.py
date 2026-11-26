"""The narrator's prompt briefs: every line, in the order the model reads them."""

from __future__ import annotations

from types import SimpleNamespace

from ingovo_search.recommender.narrator_prompt import (
    _render_listing_brief,  # pyright: ignore[reportPrivateUsage]
    _render_query_brief,  # pyright: ignore[reportPrivateUsage]
)
from ingovo_search.recommender.parser_schema import GeoHints, HardFilters, ParsedQuery


def _bare_listing(**fields: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "id": "p",
        "title": None,
        "property_type": None,
        "transaction_type": None,
        "city": None,
        "district": None,
        "area_m2": None,
        "rooms": None,
        "price_huf": None,
        "currency": "HUF",
        "year_built": None,
        "renovation_state": None,
        "description": None,
    }
    return SimpleNamespace(**{**base, **fields})


def test_query_brief_renders_every_criterion_in_prompt_order() -> None:
    query = ParsedQuery(
        hard_filters=HardFilters(
            transaction_type="rent",
            property_type_in=["Lakás", "Ház"],
            city="Budapest",
            district_in=["XIII", "V"],
            rooms=3,
            rooms_min=2,
            area_m2_min=50.4,
            area_m2_max=120.0,
            price_huf_max=90_000_000,
            price_huf_min=20_000_000,
        ),
        geo_hints=GeoHints(county="Pest", balaton_shore="Déli", side_of_river="Pest", region_name="Újlipótváros"),
        soft_preferences=["new_build", "quiet"],
        soft_prefs_freetext=["nagy erkély", "lift"],
    )

    assert _render_query_brief(query).split("\n") == [
        "Tranzakció: Kiadó",
        "Típus: Lakás, Ház",
        "Város: Budapest",
        "Kerület: XIII, V",
        "Vármegye: Pest",
        "Balaton-part: Déli",
        "Duna-oldal: Pest",
        "Régió/környék: Újlipótváros",
        "Szobaszám: 3 (egzakt)",
        "Szobaszám: min. 2",
        "Méret: min. 50 m²",
        "Méret: max. 120 m²",
        "Max ár: 90 M Ft",
        "Min ár: 20 M Ft",
        "Puha preferenciák (címkék): new_build, quiet",
        "Puha preferenciák (szabad szöveg): nagy erkély | lift",
    ]


def test_query_brief_without_criteria_says_free_search() -> None:
    assert _render_query_brief(ParsedQuery()) == "(nincs explicit kritérium — szabad keresés)"


def test_query_brief_keeps_a_bound_beyond_the_js_safe_range_exact() -> None:
    query = ParsedQuery(hard_filters=HardFilters(transaction_type="sale", price_huf_max=9_007_199_254_740_993))

    assert _render_query_brief(query) == "Tranzakció: Eladó\nMax ár: 9 007 199 254 740 993 Ft"


def test_listing_brief_renders_every_fact_in_prompt_order() -> None:
    listing = _bare_listing(
        title="Világos lakás",
        property_type="Lakás",
        transaction_type="sale",
        city="Budapest",
        district="XIII",
        area_m2=64.6,
        rooms="2 + 1 fél",
        price_huf=58_500_000,
        year_built=1938,
        renovation_state="felújított",
        description="Csendes belső udvar.",
    )

    assert _render_listing_brief(listing).split("\n") == [
        "Cím: Világos lakás",
        "Típus: Lakás",
        "Tranzakció: Eladó",
        "Helyszín: Budapest, XIII. kerület",
        "Méret: 65 m²",
        "Szobaszám: 2 + 1 fél",
        "Ár: 58,5 M Ft",
        "Építés éve: 1938",
        "Állapot: felújított",
        "Leírás: Csendes belső udvar.",
    ]


def test_listing_brief_flattens_and_truncates_a_long_description() -> None:
    listing = _bare_listing(transaction_type="rent", city="Szeged", description="  Első sor\nmásodik sor " + "x" * 300)

    lines = _render_listing_brief(listing).split("\n")

    assert lines[:2] == ["Tranzakció: Kiadó", "Helyszín: Szeged"]
    assert lines[2].startswith("Leírás: Első sor második sor xxx")
    assert lines[2].endswith("x…")
    assert len(lines[2]) == len("Leírás: ") + 240 + 1


def test_listing_brief_skips_empty_facts() -> None:
    listing = _bare_listing(
        title="", property_type="", city="", area_m2=0, rooms="", renovation_state="", description=""
    )

    assert _render_listing_brief(listing) == ""
