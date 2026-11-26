"""Tests for the structured homepage search preview."""

import pytest

from ingovo_search.recommender.parser_schema import Anchors, GeoHints, HardFilters, ParsedQuery, PoiNear, PropertyType
from ingovo_search.recommender.search_preview import PreviewLocale, build_search_preview, merge_missing_preview_fields


def test_complex_hungarian_preview_keeps_constraints_and_builds_eight_plus_five() -> None:
    parsed = ParsedQuery(
        hard_filters=HardFilters(
            transaction_type="rent",
            property_type_in=["Lakás"],
            rooms_min=2,
            price_huf_max=420_000,
        ),
        anchors=Anchors(rooms=3),
        geo_hints=GeoHints(side_of_river="Buda"),
        soft_preferences=["quiet", "elevator", "garden"],
        soft_prefs_freetext=["kutyabarát", "ne legyen földszinti"],
        poi_near=[PoiNear(category="park")],
        semantic_query="csendes kutyabarát budai lakás lift vagy kertkapcsolat",
    )

    preview = build_search_preview(
        "Kiadó, legalább 2, de inkább 3 szobás lakást keresek Budán maximum 420 ezerért, kutyával",
        "hu",
        parsed,
        "req-1",
    )

    assert len(preview.image_concepts) == 8
    assert len({item.id for item in preview.image_concepts}) == 8
    assert len(preview.search_directions) == 5
    assert preview.search_directions[0].title == "A keresés pontosan így"
    assert all("transaction:rent" in item.retained_constraints for item in preview.search_directions)
    assert all("price:max:420000" in item.retained_constraints for item in preview.search_directions)
    assert any(item.id == "pet_friendly" for item in preview.image_concepts)
    assert any(item.id == "quiet" for item in preview.image_concepts)


def test_english_preview_is_native_and_non_repetitive() -> None:
    parsed = ParsedQuery(
        hard_filters=HardFilters(
            city="Budapest",
            transaction_type="sale",
            property_type_in=["Ház"],
            price_huf_max=160_000_000,
        ),
        soft_preferences=["garden", "natural_light"],
        semantic_query="bright family house with a garden in Budapest",
    )

    preview = build_search_preview(
        "Bright family house with a garden in Budapest under HUF 160M", "en", parsed, "req-2"
    )

    assert len(preview.search_directions) == 5
    assert preview.search_directions[0].title == "Your search as stated"
    assert "House" in preview.understood_summary
    assert len({item.title for item in preview.search_directions}) == 5
    assert all(item.canonical_query for item in preview.search_directions)


def test_deterministic_guardrail_fills_only_missing_explicit_fields() -> None:
    semantic = ParsedQuery(
        hard_filters=HardFilters(transaction_type="rent", rooms_min=2, price_huf_max=420_000),
        anchors=Anchors(rooms=3),
        geo_hints=GeoHints(side_of_river="Buda"),
    )
    deterministic = ParsedQuery(
        hard_filters=HardFilters(
            city="Budapest",
            transaction_type="sale",
            property_type_in=["Lakás"],
            rooms=3,
            price_huf_max=500_000,
        )
    )

    merged = merge_missing_preview_fields(semantic, deterministic)

    assert merged.hard_filters.city == "Budapest"
    assert merged.hard_filters.property_type_in == ["Lakás"]
    assert merged.hard_filters.transaction_type == "rent"
    assert merged.hard_filters.rooms is None
    assert merged.hard_filters.rooms_min == 2
    assert merged.hard_filters.price_huf_max == 420_000
    assert merged.anchors.rooms == 3


def test_preview_json_keeps_bigint_huf_intent_fields_as_decimal_strings() -> None:
    parsed = ParsedQuery(
        hard_filters=HardFilters(price_huf_min=9_007_199_254_740_992, price_huf_max=9_007_199_254_740_993),
        anchors=Anchors(price_huf=9_007_199_254_740_994),
    )

    encoded = build_search_preview("drága lakás", "hu", parsed, "req-precise").model_dump(mode="json")
    hard = encoded["parsed_query"]["hard_filters"]
    anchors = encoded["parsed_query"]["anchors"]

    assert hard["price_huf_min"] == "9007199254740992"
    assert hard["price_huf_max"] == "9007199254740993"
    assert anchors["price_huf"] == "9007199254740994"


def test_deterministic_guardrail_takes_rooms_and_ranges_only_when_wholly_missing() -> None:
    semantic = ParsedQuery(hard_filters=HardFilters(price_huf_max=50_000_000))
    deterministic = ParsedQuery(
        hard_filters=HardFilters(
            rooms_min=2, area_m2_min=40, area_m2_max=70, price_huf_min=30_000_000, price_huf_max=60_000_000
        )
    )

    merged = merge_missing_preview_fields(semantic, deterministic).hard_filters

    assert merged.rooms_min == 2
    assert (merged.area_m2_min, merged.area_m2_max) == (40, 70)
    assert (merged.price_huf_min, merged.price_huf_max) == (None, 50_000_000)
    assert merged.model_fields_set == {"rooms_min", "area_m2_min", "area_m2_max", "price_huf_max"}

    exact_rooms_win = merge_missing_preview_fields(
        ParsedQuery(), ParsedQuery(hard_filters=HardFilters(rooms=2, rooms_min=1))
    ).hard_filters
    assert (exact_rooms_win.rooms, exact_rooms_win.rooms_min) == (2, None)


@pytest.mark.parametrize(
    ("property_type", "directions", "first_concept"),
    [
        ("Iroda", ["exact", "1-home_office", "2-near_metro", "3-natural_light", "4-parking"], "office_interior"),
        ("Nyaraló", ["exact", "1-waterfront", "2-terrace", "3-view", "4-quiet"], "holiday_home"),
        ("Ház", ["exact", "1-garden", "2-quiet", "3-garage", "4-natural_light"], "house_exterior"),
        ("Lakás", ["exact", "1-natural_light", "2-balcony", "3-quiet", "4-near_metro"], "apartment_interior"),
    ],
)
def test_preview_without_signals_falls_back_to_the_property_type(
    property_type: PropertyType, directions: list[str], first_concept: str
) -> None:
    parsed = ParsedQuery(hard_filters=HardFilters(property_type_in=[property_type]))

    preview = build_search_preview("ingatlan", "en", parsed, "req-fallback")

    assert [item.id for item in preview.search_directions] == directions
    assert preview.image_concepts[0].id == first_concept
    assert len(preview.image_concepts) == 8
    assert preview.confidence == 0.42
    assert preview.clarification is None


@pytest.mark.parametrize(
    ("parsed", "query", "locale", "clarification"),
    [
        (
            ParsedQuery(conflict_flag=True),
            "x",
            "en",
            "I found conflicting requirements. Which one matters more?",
        ),
        (ParsedQuery(), "valami szépet keresek", "hu", "Pontosítanád a helyszínt vagy az ingatlantípust?"),
        (ParsedQuery(), "valami szépet keresek", "en", "Could you clarify the location or property type?"),
        (ParsedQuery(), "  rövid   ", "hu", None),
    ],
)
def test_preview_asks_for_clarification_on_conflict_or_low_confidence(
    parsed: ParsedQuery, query: str, locale: PreviewLocale, clarification: str | None
) -> None:
    assert build_search_preview(query, locale, parsed, "req-clarify").clarification == clarification
