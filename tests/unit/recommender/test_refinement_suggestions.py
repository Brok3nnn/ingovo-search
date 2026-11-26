"""Unit tests for M3 refinement-suggestion generator."""

from __future__ import annotations

from typing import Any

from ingovo_search.recommender.parser_schema import GeoHints, HardFilters, ParsedQuery
from ingovo_search.recommender.refinement_suggestions import (
    LOOSEN_THRESHOLD,
    TIGHTEN_THRESHOLD,
    suggest_refinements,
)
from ingovo_search.recommender.score import ScoreBreakdown, ScoredCandidate
from ingovo_search.search.models import SearchResponse

# --- helpers ----------------------------------------------------------------


def _bd() -> ScoreBreakdown:
    return ScoreBreakdown(
        content=0.0,
        factual=0.0,
        softpref_core=0.0,
        softpref_free=0.0,
        anchor=0.0,
        quality=0.0,
        geo=0.0,
        pooled=0.0,
        recency=0.0,
        explore=0.0,
        total=0.0,
    )


def _candidate(id_: str, prop_type: str, **extra: Any) -> ScoredCandidate:
    return ScoredCandidate(
        listing_id=id_,
        score=0.5,
        breakdown=_bd(),
        district=extra.get("district"),
        city=extra.get("city"),
        property={"id": id_, "type": prop_type, **extra},
    )


def _make_results(n: int, prop_type: str = "Lakás") -> list[ScoredCandidate]:
    return [_candidate(f"p{i}", prop_type) for i in range(n)]


# --- sparse-result loosening ------------------------------------------------


def test_sparse_results_loosen_price_when_max_set() -> None:
    """Few results + price ceiling set → loosen-price chip appears."""
    parsed = ParsedQuery(hard_filters=HardFilters(price_huf_max=80_000_000))
    out = suggest_refinements(
        original_parsed=parsed,
        relaxed_parsed=parsed,
        results=_make_results(2),
        relaxation_path=[],
    )
    kinds = [s.kind for s in out]
    assert "loosen_price_20pct" in kinds
    # Label uses the +20% new ceiling (80 -> 96)
    price_chip = next(s for s in out if s.kind == "loosen_price_20pct")
    assert "96" in price_chip.label
    assert price_chip.overrides.hard_filters is not None
    assert price_chip.overrides.hard_filters.price_huf_max == 96_000_000


def test_refinement_response_keeps_a_bigint_huf_override_as_a_decimal_string() -> None:
    parsed = ParsedQuery(hard_filters=HardFilters(price_huf_max=9_007_199_254_740_993))
    suggestions = suggest_refinements(
        original_parsed=parsed,
        relaxed_parsed=parsed,
        results=_make_results(2),
        relaxation_path=[],
    )

    encoded = SearchResponse(refinement_suggestions=suggestions).model_dump(mode="json")
    price_override = encoded["refinement_suggestions"][0]["overrides"]["hard_filters"]["price_huf_max"]

    # ceil(6/5) is 10_808_639_105_689_192; it must not be rounded by JSON.
    assert price_override == "10808639105689192"


def test_sparse_results_no_filters_yields_no_loosen_chips() -> None:
    """No filters → nothing to loosen, no suggestions emitted."""
    parsed = ParsedQuery(hard_filters=HardFilters())
    out = suggest_refinements(
        original_parsed=parsed,
        relaxed_parsed=parsed,
        results=_make_results(2),
        relaxation_path=[],
    )
    assert out == []


def test_sparse_results_drop_district_chip_when_districts_set() -> None:
    parsed = ParsedQuery(hard_filters=HardFilters(district_in=["5", "13"]))
    out = suggest_refinements(
        original_parsed=parsed,
        relaxed_parsed=parsed,
        results=_make_results(1),
        relaxation_path=[],
    )
    chip = next(s for s in out if s.kind == "drop_district")
    assert chip.overrides.hard_filters is not None
    assert chip.overrides.hard_filters.district_in == []


def test_sparse_results_drop_county_chip_when_county_set() -> None:
    parsed = ParsedQuery(geo_hints=GeoHints(county="Pest"))
    out = suggest_refinements(
        original_parsed=parsed,
        relaxed_parsed=parsed,
        results=_make_results(2),
        relaxation_path=[],
    )
    chip = next(s for s in out if s.kind == "drop_county")
    assert chip.overrides.geo_hints is not None
    assert chip.overrides.geo_hints.county is None


# --- plentiful-result tightening --------------------------------------------


def test_plentiful_results_suggest_dominant_type_when_no_type_set() -> None:
    """No type filter + plentiful results that are mostly Lakás → 'Csak Lakás'."""
    parsed = ParsedQuery(hard_filters=HardFilters())
    # 12 Lakás + 1 Ház → clear 12:1 majority
    results = [_candidate(f"p{i}", "Lakás") for i in range(12)] + [_candidate("px", "Ház")]
    out = suggest_refinements(
        original_parsed=parsed,
        relaxed_parsed=parsed,
        results=results,
        relaxation_path=[],
    )
    chip = next(s for s in out if s.kind == "tighten_type")
    assert chip.label == "Csak Lakás"
    assert chip.overrides.hard_filters is not None
    assert chip.overrides.hard_filters.property_type_in == ["Lakás"]


def test_plentiful_results_no_dominant_type_no_tighten_chip() -> None:
    """Balanced mix of types → no clear majority → no tighten_type chip."""
    parsed = ParsedQuery(hard_filters=HardFilters())
    # 7 Lakás + 7 Ház → no majority by the 2x rule
    results = [_candidate(f"l{i}", "Lakás") for i in range(7)] + [_candidate(f"h{i}", "Ház") for i in range(7)]
    out = suggest_refinements(
        original_parsed=parsed,
        relaxed_parsed=parsed,
        results=results,
        relaxation_path=[],
    )
    assert all(s.kind != "tighten_type" for s in out)


def test_plentiful_results_suggest_soft_prefs_not_already_set() -> None:
    """Plentiful results + no soft prefs → suggest 'Csak új építésű' etc."""
    parsed = ParsedQuery(hard_filters=HardFilters())
    out = suggest_refinements(
        original_parsed=parsed,
        relaxed_parsed=parsed,
        results=_make_results(TIGHTEN_THRESHOLD + 1),
        relaxation_path=[],
    )
    kinds = [s.kind for s in out]
    assert "add_new_build" in kinds
    new_build_chip = next(s for s in out if s.kind == "add_new_build")
    assert new_build_chip.overrides.soft_preferences == ["new_build"]


def test_plentiful_results_skip_soft_pref_already_in_parsed() -> None:
    """If new_build is already a soft pref, don't suggest it."""
    parsed = ParsedQuery(soft_preferences=["new_build"])
    out = suggest_refinements(
        original_parsed=parsed,
        relaxed_parsed=parsed,
        results=_make_results(TIGHTEN_THRESHOLD + 1),
        relaxation_path=[],
    )
    assert all(s.kind != "add_new_build" for s in out)


# --- relaxation reversal ----------------------------------------------------


def test_relaxation_path_present_offers_reversal_chip_first() -> None:
    """If the pipeline relaxed something, the first suggestion offers to undo it."""
    original = ParsedQuery(hard_filters=HardFilters(price_huf_max=30_000_000))
    relaxed = ParsedQuery(hard_filters=HardFilters())  # price dropped by relaxation
    out = suggest_refinements(
        original_parsed=original,
        relaxed_parsed=relaxed,
        results=_make_results(15),
        relaxation_path=["max ár"],
    )
    assert out[0].kind == "revert_relaxation"
    assert "Visszateszem a max árt" in out[0].label
    assert out[0].overrides.hard_filters is not None
    assert out[0].overrides.hard_filters.price_huf_max == 30_000_000


def test_relaxation_path_empty_no_reversal_chip() -> None:
    parsed = ParsedQuery(hard_filters=HardFilters(price_huf_max=80_000_000))
    out = suggest_refinements(
        original_parsed=parsed,
        relaxed_parsed=parsed,
        results=_make_results(15),
        relaxation_path=[],
    )
    assert all(s.kind != "revert_relaxation" for s in out)


# --- limits + dedup ---------------------------------------------------------


def test_max_six_suggestions_returned() -> None:
    """Even with every possible chip eligible, output is capped at MAX_SUGGESTIONS=6."""
    parsed = ParsedQuery(
        hard_filters=HardFilters(price_huf_max=80_000_000, district_in=["5"]),
        geo_hints=GeoHints(county="Pest", balaton_shore="Északi", region_name="Belváros"),
    )
    out = suggest_refinements(
        original_parsed=parsed,
        relaxed_parsed=parsed,
        results=_make_results(LOOSEN_THRESHOLD - 1),
        relaxation_path=[],
    )
    assert len(out) <= 6


def test_suggestion_kinds_are_unique() -> None:
    """Same `kind` never appears twice."""
    parsed = ParsedQuery(
        hard_filters=HardFilters(price_huf_max=80_000_000, district_in=["5"]),
        geo_hints=GeoHints(county="Pest"),
    )
    out = suggest_refinements(
        original_parsed=parsed,
        relaxed_parsed=parsed,
        results=_make_results(1),
        relaxation_path=[],
    )
    kinds = [s.kind for s in out]
    assert len(kinds) == len(set(kinds))
