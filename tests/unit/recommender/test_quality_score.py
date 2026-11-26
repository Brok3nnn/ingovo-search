"""Unit tests for the §6 listing quality score.

All functions under test are pure logic, so these run against lightweight
stubs rather than the real `Property` ORM model — no DB access.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from ingovo_search.recommender.quality_score import (
    COACHING_TEMPLATES,
    ENRICH_FIELDS,
    QualityBreakdown,
    agency_quality_score,
    compute_quality_score,
    description_completeness_score,
    has_room_coverage,
    photo_score,
    quality_score_breakdown,
    recency_score,
    structured_feature_fill_score,
    word_count,
)

NOW = datetime(2026, 5, 19, 12, 0, 0, tzinfo=UTC)

_ALL_ENRICH_VALUES: dict[str, Any] = {
    "renovation_state": "felújított",
    "orientation": "dél",
    "has_balcony": True,
    "has_terrace": False,
    "has_parking": True,
    "has_garage": False,
    "has_garden": True,
    "has_basement": False,
    "elevator": True,
    "air_conditioning": True,
    "noise_score": 0.3,
    "light_quality_score": 0.8,
}


class _Listing:
    """Mutable stand-in for `Property` exposing only the inspected attributes."""

    def __init__(self, **overrides: Any) -> None:
        defaults: dict[str, Any] = {
            "photo_count": 0,
            "features": None,
            "cleaned_description": None,
            "description": None,
            "templated_phrase_fraction": None,
            "created_at": NOW,
            "agency_id": None,
        }
        defaults.update({field: None for field in ENRICH_FIELDS})
        defaults.update(overrides)
        for key, value in defaults.items():
            setattr(self, key, value)


def _high_quality_listing() -> _Listing:
    return _Listing(
        photo_count=8,
        features={"photo_room_coverage": ["kitchen", "bathroom", "living_room"]},
        cleaned_description=" ".join(["szoba"] * 300),
        templated_phrase_fraction=0.0,
        created_at=NOW,
        agency_id="agency-new",
        **_ALL_ENRICH_VALUES,
    )


def _low_quality_listing() -> _Listing:
    return _Listing(
        photo_count=1,
        features=None,
        cleaned_description=" ".join(["lakás"] * 40),
        templated_phrase_fraction=0.0,
        created_at=NOW - timedelta(days=200),
        agency_id="agency-x",
        renovation_state="felújított",
        orientation="dél",
    )


# --- Acceptance test 1: high-quality listing ---------------------------------


def test_high_quality_listing_scores_near_095() -> None:
    score = compute_quality_score(_high_quality_listing(), now=NOW)
    assert math.isclose(score, 0.95, abs_tol=0.02)


# --- Acceptance test 2: low-quality listing ----------------------------------


def test_low_quality_listing_scores_near_015() -> None:
    score = compute_quality_score(_low_quality_listing(), now=NOW)
    assert math.isclose(score, 0.15, abs_tol=0.03)


# --- Acceptance test 3: coaching suggestions ---------------------------------


def test_breakdown_suggests_improvements_when_photo_score_low() -> None:
    listing = _low_quality_listing()
    assert photo_score(listing) < 0.5
    breakdown = quality_score_breakdown(listing, now=NOW)
    assert breakdown.suggested_improvements
    assert "add_more_photos" in breakdown.suggested_improvements
    assert all(t in COACHING_TEMPLATES for t in breakdown.suggested_improvements)


def test_breakdown_has_no_suggestions_for_high_quality_listing() -> None:
    breakdown = quality_score_breakdown(_high_quality_listing(), now=NOW)
    assert breakdown.suggested_improvements == []


# --- Acceptance test 4: determinism ------------------------------------------


def test_score_is_deterministic_for_same_input() -> None:
    listing = _low_quality_listing()
    assert compute_quality_score(listing, now=NOW) == compute_quality_score(listing, now=NOW)


def test_identical_listings_score_identically() -> None:
    assert compute_quality_score(_high_quality_listing(), now=NOW) == compute_quality_score(
        _high_quality_listing(), now=NOW
    )


# --- Sub-score behaviour -----------------------------------------------------


def test_word_count_handles_none_and_blank() -> None:
    assert word_count(None) == 0
    assert word_count("   ") == 0
    assert word_count("egy két három") == 3


def test_photo_score_caps_at_eight_photos_with_coverage() -> None:
    listing = _Listing(photo_count=20, features={"photo_room_coverage": ["kitchen", "bathroom"]})
    assert math.isclose(photo_score(listing), 1.0)


def test_photo_score_applies_coverage_penalty_without_room_coverage() -> None:
    listing = _Listing(photo_count=8, features=None)
    assert math.isclose(photo_score(listing), 0.7)


def test_has_room_coverage_requires_kitchen_and_bathroom() -> None:
    assert has_room_coverage(_Listing(features={"photo_room_coverage": ["kitchen", "bathroom"]})) is True
    assert has_room_coverage(_Listing(features={"photo_room_coverage": ["kitchen"]})) is False
    assert has_room_coverage(_Listing(features=None)) is False


def test_description_score_discounts_templated_phrases() -> None:
    text = " ".join(["szó"] * 300)
    full = description_completeness_score(_Listing(cleaned_description=text, templated_phrase_fraction=0.0))
    half = description_completeness_score(_Listing(cleaned_description=text, templated_phrase_fraction=0.5))
    assert math.isclose(full, 1.0)
    assert math.isclose(half, 0.5)


def test_description_score_falls_back_to_raw_description() -> None:
    text = " ".join(["szó"] * 150)
    listing = _Listing(cleaned_description=None, description=text, templated_phrase_fraction=None)
    assert math.isclose(description_completeness_score(listing), 0.5)


def test_structured_feature_fill_score_counts_non_null_fields() -> None:
    assert structured_feature_fill_score(_Listing()) == 0.0
    assert structured_feature_fill_score(_Listing(**_ALL_ENRICH_VALUES)) == 1.0
    half = dict(list(_ALL_ENRICH_VALUES.items())[:6])
    assert math.isclose(structured_feature_fill_score(_Listing(**half)), 0.5)


def test_recency_score_decays_with_age() -> None:
    assert math.isclose(recency_score(_Listing(created_at=NOW), now=NOW), 1.0)
    older = recency_score(_Listing(created_at=NOW - timedelta(days=60)), now=NOW)
    assert math.isclose(older, math.exp(-1.0), abs_tol=1e-9)


def test_recency_score_defaults_to_one_without_created_at() -> None:
    assert recency_score(_Listing(created_at=None), now=NOW) == 1.0


def test_agency_quality_score_defaults_to_half() -> None:
    assert agency_quality_score(None) == 0.5
    assert agency_quality_score("agency-x", {"other": 0.9}) == 0.5
    assert agency_quality_score("agency-x", {"agency-x": 0.8}) == 0.8


def test_breakdown_flags_each_weak_sub_score() -> None:
    listing = _low_quality_listing()
    breakdown = quality_score_breakdown(listing, now=NOW)
    assert isinstance(breakdown, QualityBreakdown)
    assert "expand_description" in breakdown.suggested_improvements
    assert "fill_structured_fields" in breakdown.suggested_improvements
    assert "refresh_listing" in breakdown.suggested_improvements


def test_breakdown_quality_score_matches_compute_quality_score() -> None:
    listing = _low_quality_listing()
    breakdown = quality_score_breakdown(listing, now=NOW)
    assert breakdown.quality_score == compute_quality_score(listing, now=NOW)


def test_breakdown_is_frozen() -> None:
    breakdown = quality_score_breakdown(_high_quality_listing(), now=NOW)
    with pytest.raises(ValidationError):
        breakdown.quality_score = 0.0  # type: ignore[misc]
