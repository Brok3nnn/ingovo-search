"""Tests for the complete Hungarian settlement registry and resolver."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from ingovo_search.recommender.clarification_policy import (
    ClarificationAnswer,
    ClarificationReason,
    apply_clarification,
    should_clarify,
)
from ingovo_search.recommender.parser import fallback_parse_query
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.settlements import (
    ResolutionStatus,
    SettlementKind,
    all_settlements,
    budapest_district,
    county_city_index,
    dataset_metadata,
    distance_km,
    is_within_distance,
    nearby_settlements,
    resolve_location_text,
    resolve_settlement_name,
    settlement_by_id,
    settlement_distance_km,
)


def test_registry_contains_every_ksh_row_and_all_budapest_districts() -> None:
    settlements = all_settlements()
    assert len(settlements) == 3178
    assert len({item.id for item in settlements}) == 3178
    assert sum(item.kind is SettlementKind.MUNICIPALITY for item in settlements) == 3155
    assert sum(item.kind is SettlementKind.BUDAPEST_DISTRICT for item in settlements) == 23


def test_registry_metadata_pins_sources_and_hashes() -> None:
    metadata = dataset_metadata()
    assert metadata["schema_version"] == "hungary-settlements-v1"
    assert metadata["ksh_as_of"] == "2025-01-01"
    assert metadata["geonames_as_of"] == "2026-06-21"
    assert metadata["license"] == "CC BY 4.0"
    assert len(metadata["ksh_sha256"]) == len(metadata["geonames_sha256"]) == 64


def test_small_settlement_has_stable_id_administration_and_coordinates() -> None:
    resolution = resolve_settlement_name("Tiszafüred")
    settlement = resolution.settlement
    assert settlement is not None
    assert settlement.id == "ksh:29726"
    assert settlement.county == "Jász-Nagykun-Szolnok"
    assert settlement.district_name == "Tiszafüredi"
    assert settlement.latitude is not None
    assert settlement.longitude is not None


def test_complete_county_index_includes_small_villages() -> None:
    counties = county_city_index()
    assert len(counties) == 20
    assert "tiszafüred" in counties["Jász-Nagykun-Szolnok"]
    assert "iborfia" in counties["Zala"]
    assert "budapest" in counties["Budapest"]


@pytest.mark.parametrize(
    ("query_form", "canonical"),
    [
        ("Tiszafüreden", "Tiszafüred"),
        ("tiszafureden", "Tiszafüred"),
        ("Győrben", "Győr"),
        ("Gyorben", "Győr"),
        ("Debrecenben", "Debrecen"),
        ("Szegeden", "Szeged"),
        ("Pécsen", "Pécs"),
        ("Pécsett", "Pécs"),
        ("Miskolcon", "Miskolc"),
        ("Sopronban", "Sopron"),
        ("Kecskeméten", "Kecskemét"),
        ("Egerben", "Eger"),
        ("Vácon", "Vác"),
        ("Tihanyi", "Tihany"),
        ("siofoki", "Siófok"),
        ("budapesti", "Budapest"),
        ("Nyíregyházán", "Nyíregyháza"),
        ("Szombathelyen", "Szombathely"),
        ("Balatonfüreden", "Balatonfüred"),
        ("Gödöllőn", "Gödöllő"),
        ("Budaörsön", "Budaörs"),
        ("Érden", "Érd"),
        ("Szentendrén", "Szentendre"),
        ("Zalaegerszegen", "Zalaegerszeg"),
        ("Budapestről", "Budapest"),
    ],
)
def test_canonical_accent_free_and_inflected_forms(query_form: str, canonical: str) -> None:
    resolution = resolve_settlement_name(query_form)
    assert resolution.status is ResolutionStatus.RESOLVED
    assert resolution.settlement is not None
    assert resolution.settlement.name == canonical


@pytest.mark.parametrize(
    ("typo", "canonical"),
    [
        ("Budapset", "Budapest"),
        ("Deberecen", "Debrecen"),
        ("Miskloc", "Miskolc"),
        ("Szged", "Szeged"),
        ("Nyiregyhza", "Nyíregyháza"),
    ],
)
def test_known_common_typos_are_deterministic(typo: str, canonical: str) -> None:
    assert resolve_settlement_name(typo).settlement is not None
    assert resolve_settlement_name(typo).settlement.name == canonical  # type: ignore[union-attr]


def test_two_explicit_settlements_produce_ambiguity_not_silent_choice() -> None:
    resolution = resolve_location_text("lakás Szeged vagy Debrecen")
    assert resolution.status is ResolutionStatus.AMBIGUOUS
    assert {item.name for item in resolution.candidates} == {"Szeged", "Debrecen"}


def test_parser_attaches_stable_city_id_for_inflected_small_town() -> None:
    parsed = fallback_parse_query("eladó ház Tiszafüreden")
    assert parsed.hard_filters.city == "Tiszafüred"
    assert parsed.hard_filters.city_id == "ksh:29726"
    assert parsed.location_resolution is not None
    assert parsed.location_resolution.status == "resolved"


def test_parser_exposes_ambiguous_location_for_clarification() -> None:
    parsed = fallback_parse_query("lakás Szeged vagy Debrecen")
    assert parsed.hard_filters.city is None
    assert parsed.hard_filters.city_id is None
    assert parsed.location_resolution is not None
    assert parsed.location_resolution.status == "ambiguous"

    clarification = should_clarify(parsed, [])
    assert clarification is not None
    assert clarification.reason is ClarificationReason.AMBIGUOUS_SETTLEMENT
    assert {option.label for option in clarification.options} == {"Szeged, Csongrád-Csanád", "Debrecen, Hajdú-Bihar"}


def test_city_clarification_sets_name_and_stable_id() -> None:
    parsed = fallback_parse_query("lakás Szeged vagy Debrecen")
    selected = apply_clarification(
        parsed,
        ClarificationAnswer(
            reason=ClarificationReason.AMBIGUOUS_SETTLEMENT,
            option_id="city:ksh:15130",
        ),
    )
    assert selected.hard_filters.city == "Debrecen"
    assert selected.hard_filters.city_id == "ksh:15130"
    assert selected.location_resolution is not None
    assert selected.location_resolution.status == "resolved"


def test_budapest_districts_have_parent_city_and_verified_points() -> None:
    district = budapest_district(13)
    assert district is not None
    assert district.id == "ksh:24299"
    assert district.parent_id == "ksh:13578"
    assert district.county == "Budapest"
    assert district.latitude is not None
    assert district.longitude is not None
    assert budapest_district(24) is None


def test_distance_is_zero_for_same_point_and_realistic_between_cities() -> None:
    assert distance_km(47.5, 19.0, 47.5, 19.0) == 0.0
    budapest = cast("object", settlement_by_id("ksh:13578"))
    gyor = cast("object", settlement_by_id("ksh:25584"))
    measured = settlement_distance_km(budapest, gyor)  # type: ignore[arg-type]
    assert measured is not None
    assert 100.0 < measured < 130.0


def test_missing_coordinate_never_becomes_false_nearby() -> None:
    budapest = settlement_by_id("ksh:13578")
    gyor = settlement_by_id("ksh:25584")
    assert budapest is not None
    assert gyor is not None
    missing = replace(gyor, latitude=None, longitude=None)
    assert settlement_distance_km(budapest, missing) is None
    assert not is_within_distance(budapest, missing, maximum_km=30.0)


def test_nearby_registry_includes_in_radius_and_excludes_distant_city() -> None:
    nearby = nearby_settlements("ksh:13578", 30.0)
    by_name = {item.settlement.name: item.distance_km for item in nearby}
    assert 0 < by_name["Budaörs"] < 30.0
    assert "Győr" not in by_name
    assert all(item.distance_km <= 30.0 for item in nearby)


def test_budapest_district_neighbours_never_mix_with_municipalities() -> None:
    nearby = nearby_settlements("ksh:24299", 30.0)
    assert nearby
    assert all(item.settlement.kind is SettlementKind.BUDAPEST_DISTRICT for item in nearby)
    assert {item.settlement.budapest_district for item in nearby} >= {3, 5, 14}


def test_unknown_place_is_not_assigned_coordinates() -> None:
    resolution = resolve_settlement_name("Nemlétezőfalva")
    assert resolution.status is ResolutionStatus.NOT_FOUND
    assert resolution.candidates == ()


def test_parsed_query_location_contract_round_trips() -> None:
    parsed = fallback_parse_query("ház Gyorben")
    assert ParsedQuery.model_validate_json(parsed.model_dump_json()) == parsed


@pytest.mark.parametrize(
    "text",
    [
        # Real Budapest streets built from settlement adjectives — regression
        # for the 2026-07-03 eval catch where "Váci úton" resolved city=Vác.
        "A munkahelyem a Váci úton van, ezért a 13. kerületben keresek kiadó lakást",
        "eladó lakás a Szentendrei úton",
        "iroda a Budaörsi útnál",
        "üzlethelyiség a Váci utcában",
        "lakás a Csömöri úton",
    ],
)
def test_street_names_built_from_settlement_words_are_not_settlements(text: str) -> None:
    assert resolve_location_text(text).status is ResolutionStatus.NOT_FOUND


def test_settlement_token_without_street_word_still_resolves() -> None:
    resolution = resolve_location_text("eladó ház Vácon")
    assert resolution.status is ResolutionStatus.RESOLVED
    assert resolution.settlement is not None
    assert resolution.settlement.name == "Vác"


def test_adjectival_settlement_form_without_street_word_still_resolves() -> None:
    # "váci lakás" legitimately means a flat in Vác.
    resolution = resolve_location_text("váci eladó lakás kerttel")
    assert resolution.status is ResolutionStatus.RESOLVED
    assert resolution.settlement is not None
    assert resolution.settlement.name == "Vác"


def test_street_token_is_skipped_but_other_settlement_still_resolves() -> None:
    resolution = resolve_location_text("eladó lakás Budapest Váci út 12")
    assert resolution.status is ResolutionStatus.RESOLVED
    assert resolution.settlement is not None
    assert resolution.settlement.name == "Budapest"


def test_parser_does_not_set_city_from_street_name() -> None:
    parsed = fallback_parse_query("A munkahelyem a Váci úton van, ezért a 13. kerületben keresek kiadó lakást")
    assert parsed.hard_filters.city != "Vác"
