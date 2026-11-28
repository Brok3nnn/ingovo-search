"""Unit tests for the deterministic parser fast path.

The fast path must (a) parse simple explicit queries correctly without an LLM
call, and (b) decline — return `None` — on anything carrying soft preferences,
free text, ambiguity, or an unsupported construct, so `parse_query` falls back
to the Gemini parser. The decline behaviour is the safety contract: the fast
path may never *misread* a query.
"""

from __future__ import annotations

from typing import cast

import pytest

from ingovo_search.lib.llm import LlmClient, MockLlmClient
from ingovo_search.recommender.parser import fallback_parse_query, parse_query
from ingovo_search.recommender.parser_fastpath import deterministic_parse, deterministic_parse_loose
from ingovo_search.types.errors import LlmError

# --- Example queries that must take the fast path ---------------------------


def test_sale_flat_city_price_under() -> None:
    parsed = deterministic_parse("elado lakas Budapest 50 millio alatt")
    assert parsed is not None
    hf = parsed.hard_filters
    assert hf.transaction_type == "sale"
    assert hf.property_type_in == ["Lakás"]
    assert hf.city == "Budapest"
    assert hf.price_huf_max == 50_000_000
    assert hf.price_huf_min is None
    assert parsed.semantic_query == ""  # filter-only path → no embedding/rerank


def test_rent_flat_district_price_under() -> None:
    parsed = deterministic_parse("kiado lakas XI kerulet 300 ezer alatt")
    assert parsed is not None
    hf = parsed.hard_filters
    assert hf.transaction_type == "rent"
    assert hf.property_type_in == ["Lakás"]
    assert hf.city == "Budapest"
    assert hf.district_in == ["11"]
    assert hf.price_huf_max == 300_000


def test_rooms_house_city_price_ig_suffix() -> None:
    parsed = deterministic_parse("3 szobas haz Debrecen 80 millioig")
    assert parsed is not None
    hf = parsed.hard_filters
    assert hf.rooms == 3
    assert hf.property_type_in == ["Ház"]
    assert hf.city == "Debrecen"
    assert hf.price_huf_max == 80_000_000
    # Transaction left unset here; parse_query applies the sale default downstream.
    assert hf.transaction_type is None


def test_garzon_district_rent() -> None:
    parsed = deterministic_parse("garzon XIII kerulet kiado")
    assert parsed is not None
    hf = parsed.hard_filters
    assert hf.transaction_type == "rent"
    assert hf.property_type_in == ["Lakás"]
    assert hf.rooms == 1
    assert hf.city == "Budapest"
    assert hf.district_in == ["13"]


# --- More positive cases -----------------------------------------------------


def test_price_above_sets_min() -> None:
    parsed = deterministic_parse("haz Budapest 50 millio felett")
    assert parsed is not None
    assert parsed.hard_filters.price_huf_min == 50_000_000
    assert parsed.hard_filters.price_huf_max is None


def test_price_around_sets_anchor_not_filter() -> None:
    parsed = deterministic_parse("lakas 80 millio korul")
    assert parsed is not None
    assert parsed.anchors.price_huf == 80_000_000
    assert parsed.hard_filters.price_huf_max is None
    assert parsed.hard_filters.price_huf_min is None


def test_rooms_min_with_legalabb() -> None:
    parsed = deterministic_parse("legalább 3 szoba haz")
    assert parsed is not None
    assert parsed.hard_filters.rooms_min == 3
    assert parsed.hard_filters.rooms is None


def test_roman_district_first() -> None:
    parsed = deterministic_parse("XI. kerület eladó lakás")
    assert parsed is not None
    assert parsed.hard_filters.district_in == ["11"]
    assert parsed.hard_filters.city == "Budapest"
    assert parsed.hard_filters.transaction_type == "sale"


def test_bare_city_and_type() -> None:
    parsed = deterministic_parse("eladó ház Szeged")
    assert parsed is not None
    assert parsed.hard_filters.city == "Szeged"
    assert parsed.hard_filters.property_type_in == ["Ház"]


def test_city_adjective_and_search_filler() -> None:
    parsed = deterministic_parse("Tihanyi házat keresek")
    assert parsed is not None
    assert parsed.hard_filters.city == "Tihany"
    assert parsed.hard_filters.property_type_in == ["Ház"]


def test_city_adjective_with_generic_property_filler() -> None:
    parsed = deterministic_parse("budapesti ingatlant keresek")
    assert parsed is not None
    assert parsed.hard_filters.city == "Budapest"
    assert parsed.hard_filters.property_type_in == []


def test_million_tol_suffix_sets_min() -> None:
    parsed = deterministic_parse("lakás 30 milliótól")
    assert parsed is not None
    assert parsed.hard_filters.price_huf_min == 30_000_000


def test_price_range_between_sets_min_and_max() -> None:
    parsed = deterministic_parse("eladó lakás Szegeden 25 és 45 millió között")
    assert parsed is not None
    assert parsed.hard_filters.city == "Szeged"
    assert parsed.hard_filters.price_huf_min == 25_000_000
    assert parsed.hard_filters.price_huf_max == 45_000_000


def test_rent_monthly_price_forint_sets_max() -> None:
    parsed = deterministic_parse_loose("kiadó garzon a belvárosban havi 200 ezer forintért")
    assert parsed is not None
    assert parsed.hard_filters.transaction_type == "rent"
    assert parsed.hard_filters.rooms == 1
    assert parsed.hard_filters.price_huf_max == 200_000


def test_loose_fallback_keeps_filters_with_currency_token_before_bound() -> None:
    parsed = fallback_parse_query("Kiadó 2 szobás lakás Budán 300 ezer Ft alatt, erkéllyel")

    assert parsed.hard_filters.transaction_type == "rent"
    assert parsed.hard_filters.property_type_in == ["Lakás"]
    assert parsed.hard_filters.city == "Budapest"
    assert parsed.hard_filters.rooms == 2
    assert parsed.hard_filters.price_huf_max == 300_000


def test_price_range_accepts_currency_token_before_between() -> None:
    parsed = deterministic_parse("eladó lakás 50 és 80 millió Ft között")

    assert parsed is not None
    assert parsed.hard_filters.price_huf_min == 50_000_000
    assert parsed.hard_filters.price_huf_max == 80_000_000


# --- Inflected city / colloquial capital forms ------------------------------


def test_locative_city_debrecenben() -> None:
    parsed = deterministic_parse("3 szobás ház Debrecenben")
    assert parsed is not None
    hf = parsed.hard_filters
    assert hf.city == "Debrecen"
    assert hf.property_type_in == ["Ház"]
    assert hf.rooms == 3


def test_locative_city_szegeden_with_transaction() -> None:
    parsed = deterministic_parse("kiadó lakás Szegeden")
    assert parsed is not None
    hf = parsed.hard_filters
    assert hf.city == "Szeged"
    assert hf.property_type_in == ["Lakás"]
    assert hf.transaction_type == "rent"


def test_locative_city_miskolcon() -> None:
    parsed = deterministic_parse("ház Miskolcon")
    assert parsed is not None
    assert parsed.hard_filters.city == "Miskolc"


def test_locative_city_budapesten_with_price() -> None:
    parsed = deterministic_parse("eladó lakás Budapesten 50 millió alatt")
    assert parsed is not None
    hf = parsed.hard_filters
    assert hf.city == "Budapest"
    assert hf.transaction_type == "sale"
    assert hf.price_huf_max == 50_000_000


@pytest.mark.parametrize("query", ["budai lakás", "pesti lakás"])
def test_colloquial_capital_halves_map_to_budapest(query: str) -> None:
    parsed = deterministic_parse(query)
    assert parsed is not None
    assert parsed.hard_filters.city == "Budapest"
    assert parsed.hard_filters.property_type_in == ["Lakás"]


def test_loose_fastpath_keeps_filters_when_soft_text_remains() -> None:
    parsed = deterministic_parse_loose("eladó kétszobás lakás Győrben 45 millió alatt jó közlekedéssel")

    assert parsed is not None
    assert parsed.hard_filters.transaction_type == "sale"
    assert parsed.hard_filters.property_type_in == ["Lakás"]
    assert parsed.hard_filters.rooms == 2  # spelled-out "kétszobás" → 2
    assert parsed.hard_filters.city == "Győr"
    assert parsed.hard_filters.price_huf_max == 45_000_000
    assert parsed.semantic_query


@pytest.mark.parametrize(
    ("query", "rooms", "property_type"),
    [
        ("kétszobás téglalakás", 2, "Lakás"),  # spelled-out rooms + compound type
        ("háromszobás panellakás", 3, "Lakás"),
        ("egyszobás garzonlakás", 1, "Lakás"),
        ("öt szobás sorház", 5, "Ház"),  # split word-number + compound house type
        ("hatszobás ikerház", 6, "Ház"),
    ],
)
def test_word_number_rooms_and_compound_types(query: str, rooms: int, property_type: str) -> None:
    # When the LLM parser is slow, the deterministic fallback must still read the
    # inflected/compound Hungarian forms a buyer actually types.
    parsed = deterministic_parse_loose(query)
    assert parsed is not None
    assert parsed.hard_filters.rooms == rooms
    assert parsed.hard_filters.property_type_in == [property_type]


def test_loose_fastpath_salvages_hard_filters_from_long_query() -> None:
    # Regression: the >80-char gate used to make the loose fallback return None
    # for complex queries, so a parser timeout produced results with no
    # district / type / price / room filter. The loose path must salvage them.
    query = (
        "felújított 3 szobás lakás Budapest XI. kerületében, legalább 60 négyzetméter, legfeljebb 90 millió, parkolóval"
    )
    assert deterministic_parse(query) is None  # strict path still defers to LLM
    parsed = deterministic_parse_loose(query)

    assert parsed is not None
    assert parsed.hard_filters.city == "Budapest"
    assert parsed.hard_filters.district_in == ["11"]
    assert parsed.hard_filters.property_type_in == ["Lakás"]
    assert parsed.hard_filters.rooms == 3
    assert parsed.hard_filters.price_huf_max == 90_000_000
    assert parsed.semantic_query


def test_budapest_neighborhood_alias_sets_district() -> None:
    parsed = deterministic_parse_loose("kiadó modern garzon Újlipótvárosban 230 ezer alatt")
    assert parsed is not None
    assert parsed.hard_filters.city == "Budapest"
    assert parsed.hard_filters.district_in == ["13"]
    assert parsed.hard_filters.price_huf_max == 230_000


@pytest.mark.parametrize(
    ("query", "city", "ptype"),
    [
        ("balatonfüredi lakás", "Balatonfüred", "Lakás"),
        ("siófoki nyaraló", "Siófok", "Nyaraló"),
        ("tihanyi ház", "Tihany", "Ház"),
    ],
)
def test_adjectival_city_forms(query: str, city: str, ptype: str) -> None:
    parsed = deterministic_parse(query)
    assert parsed is not None
    assert parsed.hard_filters.city == city
    assert parsed.hard_filters.property_type_in == [ptype]


# --- Balaton region (generic, not a specific lakeside town) ------------------


@pytest.mark.parametrize(
    ("query", "ptype"),
    [
        ("balatoni ingatlan", None),
        ("balatoni ház", "Ház"),
        ("balatoni nyaraló", "Nyaraló"),
        ("Balaton környéki ingatlan", None),
        ("Balaton menti nyaraló", "Nyaraló"),
    ],
)
def test_balaton_region_sets_geo_hint(query: str, ptype: str | None) -> None:
    parsed = deterministic_parse(query)
    assert parsed is not None
    assert parsed.geo_hints.region_name == "Balaton"
    assert parsed.hard_filters.city is None  # region, not a single city
    assert parsed.hard_filters.property_type_in == ([ptype] if ptype else [])
    assert parsed.semantic_query == ""  # still filter-only → no embedding/rerank


@pytest.mark.parametrize(
    ("query", "city"),
    [
        ("siofoki nyaralo", "Siófok"),
        ("eladó ház Gyorben", "Győr"),
        ("kiadó lakás Pecsen", "Pécs"),
        ("ház Debrecenben", "Debrecen"),
    ],
)
def test_accent_insensitive_city_match(query: str, city: str) -> None:
    # Users routinely type without diacritics — these must still fast-path.
    parsed = deterministic_parse(query)
    assert parsed is not None
    assert parsed.hard_filters.city == city


def test_specific_lakeside_town_stays_a_city_not_region() -> None:
    # A named lakeside town must resolve to that exact city, not the generic region.
    parsed = deterministic_parse("balatonfüredi lakás")
    assert parsed is not None
    assert parsed.hard_filters.city == "Balatonfüred"
    assert parsed.geo_hints.region_name is None


def test_balaton_south_shore_is_preserved_by_fast_path() -> None:
    parsed = deterministic_parse("nyaraló a Balaton déli partján")

    assert parsed is not None
    assert parsed.geo_hints.balaton_shore == "Déli"
    assert parsed.hard_filters.city is None


def test_county_fallback_does_not_reinterpret_pest_as_budapest() -> None:
    parsed = fallback_parse_query("családi ház Pest vármegyében")

    assert parsed.geo_hints.county == "Pest"
    assert parsed.hard_filters.city is None
    assert parsed.hard_filters.property_type_in == ["Ház"]


def test_verified_deak_square_radius_is_a_strict_near_filter() -> None:
    parsed = deterministic_parse("lakás a Deák Ferenc tér 2 kilométeres körzetében")

    assert parsed is not None
    assert parsed.hard_filters.city == "Budapest"
    assert parsed.hard_filters.near is not None
    assert parsed.hard_filters.near.radius_km == 2.0
    assert parsed.hard_filters.near.lat == 47.4983
    assert parsed.hard_filters.near.lng == 19.0541


# --- Negative cases (must decline → LLM fallback) ----------------------------


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "napos csendes erkélyes lakás Budapest",  # soft preferences
        "lakás panel Budapest",  # 'panel' is a construction type, not a PropertyType
        "eladó lakás Budapest 50 millió",  # bare price, no direction
        "lakás 65 m2 Budapest",  # explicit area not supported
        "2 milliárd ház",  # milliárd must not read as millió
        "eladó kiadó lakás",  # conflicting transaction intent
        "lakás Szeged vagy Debrecen",  # two cities
        "valami teljesen más szöveg",  # nothing recognised
        "lakás közel a metróhoz",  # free-text phrase
    ],
)
def test_declines_to_fastpath(query: str) -> None:
    assert deterministic_parse(query) is None


def test_declines_when_only_transaction() -> None:
    # A bare transaction word carries no real constraint → defer.
    assert deterministic_parse("eladó") is None


def test_declines_overly_long_query() -> None:
    long_query = "eladó lakás " + "Budapest " * 20
    assert deterministic_parse(long_query) is None


# --- Integration with parse_query: the gate decides LLM vs no-LLM ------------


async def test_parse_query_fast_path_skips_llm() -> None:
    # MockLlmClient with no fixtures raises if .complete is ever called.
    client = MockLlmClient({})
    parsed = await parse_query("eladó lakás Budapest 50 millió alatt", client=cast(LlmClient, client))
    assert parsed.hard_filters.city == "Budapest"
    assert parsed.hard_filters.price_huf_max == 50_000_000


async def test_parse_query_falls_back_to_llm_when_not_simple() -> None:
    client = MockLlmClient({})  # no fixture → LlmError proves the LLM path was taken
    with pytest.raises(LlmError):
        await parse_query("napos csendes erkélyes lakás a Duna-parton", client=cast(LlmClient, client))


def test_loose_salvage_skips_negated_district() -> None:
    """Exclusions must not be salvaged as inclusive district filters."""
    parsed = deterministic_parse_loose("kiadó lakás maximum 250 ezerért, de nem a 8. kerületben")

    assert parsed is not None
    assert parsed.hard_filters.district_in == []


def test_loose_salvage_keeps_included_district_next_to_negated_one() -> None:
    parsed = deterministic_parse_loose("eladó lakás a 9. kerületben, de nem a 8. kerületben")

    assert parsed is not None
    assert parsed.hard_filters.district_in == ["9"]


def test_loose_salvage_skips_kivetel_district() -> None:
    parsed = deterministic_parse_loose("kiadó lakás Budapesten, a 8. kerület kivételével")

    assert parsed is not None
    assert parsed.hard_filters.district_in == []
