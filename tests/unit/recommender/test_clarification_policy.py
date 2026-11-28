"""Unit tests for the clarification policy.

`apply_clarification` returns a new `ParsedQuery` without introducing a
clarification-specific scoring weight; `should_clarify` returns at most one
clarification and never blocks results.
"""

from __future__ import annotations

import pytest

from ingovo_search.recommender import clarification_apply, clarification_schema
from ingovo_search.recommender.clarification_policy import (
    LOW_CONFIDENCE_THRESHOLD,
    MIN_RESULTS_BEFORE_CLARIFY,
    SPLIT_MIN_SIDE_FRACTION,
    TOP_N_FOR_SPLIT_DETECTION,
    WEAK_MAJORITY_FRACTION,
    WEAK_RESULT_THRESHOLD,
    Anchors,
    Clarification,
    ClarificationAnswer,
    ClarificationOption,
    ClarificationReason,
    GeoHints,
    HardFilters,
    ParsedQuery,
    apply_clarification,
    should_clarify,
)
from ingovo_search.recommender.parser_schema import ParseConfidence
from ingovo_search.recommender.score import ScoreBreakdown, ScoredCandidate
from ingovo_search.types.errors import ValidationError

# --- Helpers ---------------------------------------------------------------


def _empty_breakdown() -> ScoreBreakdown:
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


def _candidate(*, listing_id: str = "x", score: float = 0.9, district: str | None = None) -> ScoredCandidate:
    return ScoredCandidate(
        listing_id=listing_id,
        score=score,
        breakdown=_empty_breakdown(),
        district=district,
    )


def _strong_pool(size: int = 30, district: str = "13") -> list[ScoredCandidate]:
    return [_candidate(listing_id=f"l{i}", score=0.85, district=district) for i in range(size)]


def _populated_parsed() -> ParsedQuery:
    """A well-specified query — no trigger should fire on this."""
    return ParsedQuery(
        hard_filters=HardFilters(city="Budapest", district_in=["13"], price_huf_max=85_000_000),
        anchors=Anchors(price_huf=80_000_000),
        soft_preferences=["elevator"],
        semantic_query="napos, csendes, felújított",
        parse_confidence=ParseConfidence(hard_filters=0.9, anchors=0.9),
    )


# --- should_clarify: no trigger -------------------------------------------


def test_should_clarify_returns_none_when_results_are_good() -> None:
    """Well-specified query + strong result pool → no clarification."""
    parsed = _populated_parsed()
    scored = _strong_pool()
    assert should_clarify(parsed, scored) is None


# --- should_clarify: conflict trigger -------------------------------------


def test_should_clarify_fires_on_conflict_flag() -> None:
    parsed = ParsedQuery(
        hard_filters=HardFilters(price_huf_max=10_000_000, price_huf_min=50_000_000),
        conflict_flag=True,
        parse_confidence=ParseConfidence(hard_filters=0.9),
    )
    result = should_clarify(parsed, _strong_pool())
    assert result is not None
    assert result.reason == ClarificationReason.CONFLICTING_CONSTRAINTS


def test_conflict_clarification_offers_relax_options_for_set_fields() -> None:
    parsed = ParsedQuery(
        hard_filters=HardFilters(price_huf_max=10_000_000, area_m2_min=120.0),
        conflict_flag=True,
        parse_confidence=ParseConfidence(hard_filters=0.9),
    )
    result = should_clarify(parsed, _strong_pool())
    assert result is not None
    option_ids = {opt.id for opt in result.options}
    assert "relax:price_huf_max" in option_ids
    assert "relax:area_m2_min" in option_ids


# --- should_clarify: low confidence trigger -------------------------------


def test_should_clarify_fires_on_low_per_field_confidence() -> None:
    parsed = ParsedQuery(
        hard_filters=HardFilters(district_in=["13"]),
        parse_confidence=ParseConfidence(hard_filters=LOW_CONFIDENCE_THRESHOLD - 0.01),
    )
    result = should_clarify(parsed, _strong_pool())
    assert result is not None
    assert result.reason == ClarificationReason.LOW_PARSER_CONFIDENCE


def test_should_clarify_fires_on_empty_parse_with_no_telemetry() -> None:
    """Parser returned nothing AND no confidence telemetry → ask."""
    parsed = ParsedQuery()
    result = should_clarify(parsed, _strong_pool())
    assert result is not None
    assert result.reason == ClarificationReason.LOW_PARSER_CONFIDENCE


def test_should_clarify_skips_low_confidence_when_some_field_is_filled() -> None:
    """Empty `parse_confidence` plus any populated field → fall through to other triggers."""
    parsed = ParsedQuery(soft_preferences=["elevator"])
    result = should_clarify(parsed, _strong_pool())
    if result is not None:
        assert result.reason != ClarificationReason.LOW_PARSER_CONFIDENCE


# --- should_clarify: ambiguous geography trigger --------------------------


def test_should_clarify_fires_on_region_without_districts() -> None:
    parsed = ParsedQuery(
        geo_hints=GeoHints(region_name="rózsadomb"),
        parse_confidence=ParseConfidence(geo_hints=0.9),
    )
    result = should_clarify(parsed, _strong_pool())
    assert result is not None
    assert result.reason == ClarificationReason.AMBIGUOUS_GEOGRAPHY


def test_should_clarify_skips_geo_when_districts_set() -> None:
    parsed = ParsedQuery(
        hard_filters=HardFilters(district_in=["2"]),
        geo_hints=GeoHints(region_name="rózsadomb"),
        parse_confidence=ParseConfidence(geo_hints=0.9, hard_filters=0.9),
    )
    result = should_clarify(parsed, _strong_pool())
    if result is not None:
        assert result.reason != ClarificationReason.AMBIGUOUS_GEOGRAPHY


# --- should_clarify: thin result sets stay visible ------------------------


def test_should_clarify_does_not_interrupt_a_thin_result_set() -> None:
    parsed = _populated_parsed()
    scored = _strong_pool(size=MIN_RESULTS_BEFORE_CLARIFY - 1)
    assert should_clarify(parsed, scored) is None


def test_should_clarify_does_not_offer_manual_relaxation_for_thin_results() -> None:
    parsed = _populated_parsed()
    scored = _strong_pool(size=2)
    assert should_clarify(parsed, scored) is None


def test_should_clarify_does_not_ask_permission_for_farther_expansion() -> None:
    parsed = ParsedQuery(
        hard_filters=HardFilters(city="Tiszafüred", city_id="ksh:29726"),
        parse_confidence=ParseConfidence(hard_filters=0.9),
    )
    assert should_clarify(parsed, _strong_pool(size=2)) is None


# --- should_clarify: too many weak results trigger ------------------------


def test_should_clarify_fires_when_majority_of_results_are_weak() -> None:
    parsed = _populated_parsed()
    weak = [_candidate(listing_id=f"w{i}", score=WEAK_RESULT_THRESHOLD - 0.05, district="13") for i in range(8)]
    strong = [_candidate(listing_id=f"s{i}", score=0.9, district="13") for i in range(2)]
    result = should_clarify(parsed, weak + strong)
    assert result is not None
    assert result.reason == ClarificationReason.TOO_MANY_WEAK_RESULTS


def test_should_not_fire_weak_trigger_when_top_half_is_strong() -> None:
    parsed = _populated_parsed()
    strong = [_candidate(listing_id=f"s{i}", score=0.9, district="13") for i in range(8)]
    weak = [_candidate(listing_id=f"w{i}", score=0.2, district="13") for i in range(2)]
    result = should_clarify(parsed, strong + weak)
    if result is not None:
        assert result.reason != ClarificationReason.TOO_MANY_WEAK_RESULTS


# --- should_clarify: split top-result interpretations ---------------------


def test_should_clarify_fires_on_split_top_districts() -> None:
    parsed = _populated_parsed()
    half = TOP_N_FOR_SPLIT_DETECTION // 2
    pool = [_candidate(listing_id=f"a{i}", score=0.85, district="2") for i in range(half)] + [
        _candidate(listing_id=f"b{i}", score=0.85, district="13") for i in range(TOP_N_FOR_SPLIT_DETECTION - half)
    ]
    result = should_clarify(parsed, pool)
    assert result is not None
    assert result.reason == ClarificationReason.SPLIT_TOP_INTERPRETATIONS
    option_ids = {opt.id for opt in result.options}
    assert {"district:2", "district:13"} <= option_ids


def test_should_not_fire_split_when_pool_is_one_district() -> None:
    parsed = _populated_parsed()
    pool = _strong_pool(size=TOP_N_FOR_SPLIT_DETECTION, district="13")
    result = should_clarify(parsed, pool)
    if result is not None:
        assert result.reason != ClarificationReason.SPLIT_TOP_INTERPRETATIONS


def test_should_not_fire_split_when_minority_is_too_small() -> None:
    parsed = _populated_parsed()
    minority_size = max(1, int(SPLIT_MIN_SIDE_FRACTION * TOP_N_FOR_SPLIT_DETECTION) - 1)
    majority_size = TOP_N_FOR_SPLIT_DETECTION - minority_size
    pool = [_candidate(listing_id=f"a{i}", score=0.85, district="13") for i in range(majority_size)] + [
        _candidate(listing_id=f"b{i}", score=0.85, district="2") for i in range(minority_size)
    ]
    result = should_clarify(parsed, pool)
    if result is not None:
        assert result.reason != ClarificationReason.SPLIT_TOP_INTERPRETATIONS


# --- Priority ordering ----------------------------------------------------


def test_conflict_beats_low_confidence() -> None:
    parsed = ParsedQuery(
        hard_filters=HardFilters(price_huf_min=50_000_000, price_huf_max=10_000_000),
        conflict_flag=True,
        parse_confidence=ParseConfidence(hard_filters=0.1),
    )
    result = should_clarify(parsed, _strong_pool())
    assert result is not None
    assert result.reason == ClarificationReason.CONFLICTING_CONSTRAINTS


def test_low_confidence_beats_ambiguous_geo() -> None:
    parsed = ParsedQuery(
        geo_hints=GeoHints(region_name="rózsadomb"),
        parse_confidence=ParseConfidence(geo_hints=0.1),
    )
    result = should_clarify(parsed, _strong_pool())
    assert result is not None
    assert result.reason == ClarificationReason.LOW_PARSER_CONFIDENCE


def test_ambiguous_geo_still_clarifies_with_a_thin_pool() -> None:
    parsed = ParsedQuery(
        geo_hints=GeoHints(region_name="rózsadomb"),
        parse_confidence=ParseConfidence(geo_hints=0.9),
    )
    result = should_clarify(parsed, _strong_pool(size=2))
    assert result is not None
    assert result.reason == ClarificationReason.AMBIGUOUS_GEOGRAPHY


def test_should_clarify_signature_returns_one_or_none() -> None:
    """The API is `Clarification | None`, never a list."""
    parsed = ParsedQuery(conflict_flag=True, parse_confidence=ParseConfidence(x=0.1))
    result = should_clarify(parsed, _strong_pool(size=2))
    assert result is None or isinstance(result, Clarification)


# --- apply_clarification ---------------------------------------------------


def test_apply_skip_returns_unchanged() -> None:
    parsed = _populated_parsed()
    result = apply_clarification(
        parsed,
        ClarificationAnswer(reason=ClarificationReason.TOO_FEW_RESULTS, option_id="skip"),
    )
    assert result == parsed


def test_apply_district_appends_to_district_in() -> None:
    parsed = ParsedQuery(
        geo_hints=GeoHints(region_name="rózsadomb"),
        parse_confidence=ParseConfidence(geo_hints=0.9),
    )
    result = apply_clarification(
        parsed,
        ClarificationAnswer(reason=ClarificationReason.AMBIGUOUS_GEOGRAPHY, option_id="district:13"),
    )
    assert result.hard_filters.district_in == ["13"]
    # The original is untouched: ParsedQuery is frozen.
    assert parsed.hard_filters.district_in == []


def test_apply_districts_multi_appends_all() -> None:
    parsed = ParsedQuery(parse_confidence=ParseConfidence(hard_filters=0.9))
    result = apply_clarification(
        parsed,
        ClarificationAnswer(reason=ClarificationReason.AMBIGUOUS_GEOGRAPHY, option_id="districts:1,2,12"),
    )
    assert result.hard_filters.district_in == ["1", "2", "12"]


def test_apply_district_dedupes() -> None:
    parsed = ParsedQuery(
        hard_filters=HardFilters(district_in=["13"]),
        parse_confidence=ParseConfidence(hard_filters=0.9),
    )
    result = apply_clarification(
        parsed,
        ClarificationAnswer(reason=ClarificationReason.AMBIGUOUS_GEOGRAPHY, option_id="district:13"),
    )
    assert result.hard_filters.district_in == ["13"]


def test_apply_relax_clears_field() -> None:
    parsed = ParsedQuery(
        hard_filters=HardFilters(price_huf_max=10_000_000),
        parse_confidence=ParseConfidence(hard_filters=0.9),
    )
    result = apply_clarification(
        parsed,
        ClarificationAnswer(reason=ClarificationReason.TOO_FEW_RESULTS, option_id="relax:price_huf_max"),
    )
    assert result.hard_filters.price_huf_max is None
    # Other filter fields untouched
    assert parsed.hard_filters.price_huf_max == 10_000_000


def test_apply_relax_unknown_field_returns_unchanged() -> None:
    parsed = ParsedQuery(
        hard_filters=HardFilters(price_huf_max=10_000_000),
        parse_confidence=ParseConfidence(hard_filters=0.9),
    )
    result = apply_clarification(
        parsed,
        ClarificationAnswer(reason=ClarificationReason.TOO_FEW_RESULTS, option_id="relax:not_a_field"),
    )
    assert result == parsed


def test_apply_location_expansion_records_explicit_user_approval() -> None:
    parsed = ParsedQuery(
        hard_filters=HardFilters(city="Tiszafüred", city_id="ksh:29726"),
        parse_confidence=ParseConfidence(hard_filters=0.9),
    )
    result = apply_clarification(
        parsed,
        ClarificationAnswer(reason=ClarificationReason.TOO_FEW_RESULTS, option_id="expand:location"),
    )
    assert result.allow_distant_location_expansion is True
    assert result.hard_filters == parsed.hard_filters


def test_apply_softpref_appends_valid_tag() -> None:
    parsed = ParsedQuery(parse_confidence=ParseConfidence(soft_preferences=0.9))
    result = apply_clarification(
        parsed,
        ClarificationAnswer(reason=ClarificationReason.TOO_MANY_WEAK_RESULTS, option_id="softpref:elevator"),
    )
    assert "elevator" in result.soft_preferences


def test_apply_softpref_dedupes() -> None:
    parsed = ParsedQuery(soft_preferences=["elevator"], parse_confidence=ParseConfidence(x=0.9))
    result = apply_clarification(
        parsed,
        ClarificationAnswer(reason=ClarificationReason.TOO_MANY_WEAK_RESULTS, option_id="softpref:elevator"),
    )
    assert result.soft_preferences == ["elevator"]


def test_apply_softpref_rejects_unknown_tag() -> None:
    parsed = ParsedQuery(parse_confidence=ParseConfidence(x=0.9))
    with pytest.raises(ValidationError, match="unknown soft-preference tag"):
        apply_clarification(
            parsed,
            ClarificationAnswer(
                reason=ClarificationReason.TOO_MANY_WEAK_RESULTS,
                option_id="softpref:not_a_real_tag",
            ),
        )


def test_apply_anchor_sets_price() -> None:
    parsed = ParsedQuery(parse_confidence=ParseConfidence(anchors=0.9))
    result = apply_clarification(
        parsed,
        ClarificationAnswer(reason=ClarificationReason.TOO_FEW_RESULTS, option_id="anchor:price_huf=85000000"),
    )
    assert result.anchors.price_huf == 85_000_000


def test_apply_anchor_invalid_value_raises() -> None:
    parsed = ParsedQuery(parse_confidence=ParseConfidence(anchors=0.9))
    with pytest.raises(ValidationError, match="invalid anchor value"):
        apply_clarification(
            parsed,
            ClarificationAnswer(reason=ClarificationReason.TOO_FEW_RESULTS, option_id="anchor:price_huf=not_a_number"),
        )


def test_apply_unknown_prefix_returns_unchanged() -> None:
    parsed = _populated_parsed()
    result = apply_clarification(
        parsed,
        ClarificationAnswer(reason=ClarificationReason.TOO_FEW_RESULTS, option_id="not_a_prefix:13"),
    )
    assert result == parsed


def test_apply_empty_rhs_returns_unchanged() -> None:
    parsed = _populated_parsed()
    result = apply_clarification(
        parsed,
        ClarificationAnswer(reason=ClarificationReason.TOO_FEW_RESULTS, option_id="district:"),
    )
    assert result == parsed


def test_apply_expand_with_other_target_returns_unchanged() -> None:
    """`expand:` only knows `location`; any other target leaves the query untouched."""
    parsed = _populated_parsed()
    result = apply_clarification(
        parsed,
        ClarificationAnswer(reason=ClarificationReason.TOO_FEW_RESULTS, option_id="expand:radius"),
    )
    assert result == parsed
    assert result.allow_distant_location_expansion is False


def test_policy_reexports_the_answer_side_and_the_schema() -> None:
    """Callers import the whole policy from `clarification_policy`; the names are the real objects."""
    assert apply_clarification is clarification_apply.apply_clarification
    assert ClarificationAnswer is clarification_schema.ClarificationAnswer
    assert Clarification is clarification_schema.Clarification


# --- no clarification-specific scoring weight ---------------


def test_score_breakdown_shape_is_unchanged() -> None:
    """ScoreBreakdown carries the v0 ten-term + total schema, plus the
    `amenity_match` honesty signal (defaulted, so it adds no scoring weight)."""
    expected = {
        "content",
        "factual",
        "softpref_core",
        "softpref_free",
        "anchor",
        "quality",
        "geo",
        "pooled",
        "recency",
        "explore",
        "total",
        "amenity_match",
    }
    assert set(ScoreBreakdown.model_fields.keys()) == expected


def test_apply_clarification_does_not_emit_a_new_field() -> None:
    """The result is a ParsedQuery — same field set as before."""
    parsed = _populated_parsed()
    result = apply_clarification(
        parsed,
        ClarificationAnswer(reason=ClarificationReason.TOO_FEW_RESULTS, option_id="relax:price_huf_max"),
    )
    assert isinstance(result, ParsedQuery)
    assert set(type(result).model_fields.keys()) == set(ParsedQuery.model_fields.keys())


# --- Schema invariants ----------------------------------------------------


def test_clarification_models_are_frozen() -> None:
    """All four Pydantic models must be frozen so they can't drift downstream."""
    opt = ClarificationOption(id="x", label="y")
    clar = Clarification(reason=ClarificationReason.TOO_FEW_RESULTS, question="?", options=[opt])
    ans = ClarificationAnswer(reason=ClarificationReason.TOO_FEW_RESULTS, option_id="skip")
    with pytest.raises(ValueError, match="frozen"):
        opt.id = "z"  # type: ignore[misc]
    with pytest.raises(ValueError, match="frozen"):
        clar.question = "?"  # type: ignore[misc]
    with pytest.raises(ValueError, match="frozen"):
        ans.option_id = "x"  # type: ignore[misc]


def test_weak_majority_fraction_invariant() -> None:
    """If thresholds drift below sensible v0 values, fail fast."""
    assert 0.5 <= WEAK_MAJORITY_FRACTION <= 1.0
    assert 0.0 <= WEAK_RESULT_THRESHOLD <= 1.0
    assert MIN_RESULTS_BEFORE_CLARIFY >= 1
