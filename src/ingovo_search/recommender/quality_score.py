"""Listing quality score — 0..1 score per listing.

Per the design spec §6: a weighted blend of five sub-scores (photos,
description completeness, structured-feature fill, recency, agency quality).
Used by the recommender and by the agent-dashboard coaching surface.

Every function here is pure logic and DB-free. The property argument is typed
`object` and read via `getattr` so lightweight test stubs and the real
`Property` ORM model both work — mirroring `ingestion.quality_gate`.

Re-computation hooks (the §3 upsert pipeline and the §12 nightly batch) live
in those components; this module only computes.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import cast

from pydantic import BaseModel

# §4 enrichment-output columns that are genuinely nullable pre-enrichment.
# `structured_feature_fill_score` measures how many of these are populated.
ENRICH_FIELDS: tuple[str, ...] = (
    "renovation_state",
    "orientation",
    "has_balcony",
    "has_terrace",
    "has_parking",
    "has_garage",
    "has_garden",
    "has_basement",
    "elevator",
    "air_conditioning",
    "noise_score",
    "light_quality_score",
)

# Rooms whose dedicated photos most influence buyer decisions (Appendix B).
REQUIRED_ROOM_COVERAGE: tuple[str, ...] = ("kitchen", "bathroom")

# v0 hand-set sub-score weights (sum to 1.0). v1 fits these on logged signals.
WEIGHT_PHOTO = 0.30
WEIGHT_DESCRIPTION = 0.25
WEIGHT_STRUCTURED = 0.20
WEIGHT_RECENCY = 0.15
WEIGHT_AGENCY = 0.10

# Sub-score thresholds below which a coaching template is suggested.
PHOTO_SCORE_THRESHOLD = 0.5
DESCRIPTION_SCORE_THRESHOLD = 0.5
STRUCTURED_SCORE_THRESHOLD = 0.6
RECENCY_SCORE_THRESHOLD = 0.3

# Default agency-quality score when no agency-level data is available.
DEFAULT_AGENCY_QUALITY = 0.5

# Coaching templates surfaced to agents (Appendix B). `quality_score_breakdown`
# returns the template ids; §14 maps ids to this Hungarian copy.
COACHING_TEMPLATES: dict[str, str] = {
    "add_more_photos": (
        "Tölts fel több fotót (legalább 8-at). A több fotóval rendelkező "
        "hirdetésekre lényegesen több érdeklődés érkezik."
    ),
    "add_kitchen_bathroom_photos": (
        "Adj fel külön fotót a konyháról és a fürdőszobáról. Ezek a fotók erősen befolyásolják az érdeklődők döntését."
    ),
    "expand_description": (
        "Bővítsd a leírást részletesebbre, és kerüld a sablonos "
        "megfogalmazásokat. A részletes, egyedi leírás növeli az érdeklődést."
    ),
    "fill_structured_fields": (
        "Töltsd ki a hiányzó adatokat (állapot, tájolás, fűtés, "
        "felszereltség). A teljes adatlap előrébb sorolja a hirdetést."
    ),
    "refresh_listing": (
        "Frissítsd a hirdetést. A régóta változatlan hirdetések kevésbé jelennek meg a találatok élén."
    ),
}


class QualityBreakdown(BaseModel):
    """Sub-score breakdown plus suggested coaching-template ids."""

    model_config = {"frozen": True}

    quality_score: float
    photo_score: float
    description_completeness_score: float
    structured_feature_fill_score: float
    recency_score: float
    agency_quality_score: float
    suggested_improvements: list[str]


def _int_attr(obj: object, name: str) -> int | None:
    value = getattr(obj, name, None)
    # bool is an int subclass; exclude it so flag columns never count as ints.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _string_attr(obj: object, name: str) -> str | None:
    value = getattr(obj, name, None)
    return value if isinstance(value, str) else None


def _float_attr(obj: object, name: str) -> float | None:
    value = getattr(obj, name, None)
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def word_count(text: str | None) -> int:
    """Whitespace-delimited word count; `None` and blank strings count as 0."""
    if not text:
        return 0
    return len(text.split())


def has_room_coverage(listing: object) -> bool:
    """True iff `features["photo_room_coverage"]` covers every required room.

    Photo-level room labelling is not produced in v0 (vision insights are
    deferred per the design spec §6.10), so most listings return False
    and take the reduced `photo_score` coverage multiplier.
    """
    features = getattr(listing, "features", None)
    if not isinstance(features, dict):
        return False
    covered = cast("dict[str, object]", features).get("photo_room_coverage")
    if not isinstance(covered, (list, tuple)):
        return False
    covered_rooms = {str(room).lower() for room in cast("list[object]", covered)}
    return all(room in covered_rooms for room in REQUIRED_ROOM_COVERAGE)


def photo_score(listing: object) -> float:
    """Photo count toward a target of 8, scaled by room-coverage completeness."""
    photo_count = _int_attr(listing, "photo_count") or 0
    base = min(photo_count / 8, 1.0)
    coverage_bonus = 1.0 if has_room_coverage(listing) else 0.7
    return base * coverage_bonus


def description_completeness_score(listing: object) -> float:
    """Description length toward 300 words, discounted by templated-phrase share."""
    text = _string_attr(listing, "cleaned_description") or _string_attr(listing, "description")
    length_factor = min(word_count(text) / 300, 1.0)
    templated_fraction = _float_attr(listing, "templated_phrase_fraction") or 0.0
    return length_factor * (1.0 - templated_fraction)


def structured_feature_fill_score(listing: object) -> float:
    """Fraction of §4 enrichment fields that are populated."""
    filled = sum(1 for field in ENRICH_FIELDS if getattr(listing, field, None) is not None)
    return filled / len(ENRICH_FIELDS)


def recency_score(listing: object, *, now: datetime | None = None) -> float:
    """Exponential decay on listing age with a 60-day characteristic time."""
    now = now or datetime.now(tz=UTC)
    created_at = getattr(listing, "created_at", None)
    if not isinstance(created_at, datetime):
        return 1.0
    age_days = max((now - created_at).total_seconds() / 86_400.0, 0.0)
    return math.exp(-age_days / 60.0)


def lookup_agency_quality(
    agency_id: str | None,
    quality_lookup: dict[str, float] | None = None,
) -> float | None:
    """Look up an agency-level quality score. v0 has no source, so this is a
    pure mapping lookup; callers pass `quality_lookup` once §16 supplies one."""
    if agency_id is None or quality_lookup is None:
        return None
    return quality_lookup.get(agency_id)


def agency_quality_score(
    agency_id: str | None,
    quality_lookup: dict[str, float] | None = None,
) -> float:
    """Agency quality score, defaulting to 0.5 when no data is available."""
    return lookup_agency_quality(agency_id, quality_lookup) or DEFAULT_AGENCY_QUALITY


def _combine(
    photo: float,
    description: float,
    structured: float,
    recency: float,
    agency: float,
) -> float:
    return (
        WEIGHT_PHOTO * photo
        + WEIGHT_DESCRIPTION * description
        + WEIGHT_STRUCTURED * structured
        + WEIGHT_RECENCY * recency
        + WEIGHT_AGENCY * agency
    )


def compute_quality_score(
    listing: object,
    *,
    agency_quality: dict[str, float] | None = None,
    now: datetime | None = None,
) -> float:
    """Weighted 0..1 quality score. Deterministic given the same listing + `now`."""
    return _combine(
        photo_score(listing),
        description_completeness_score(listing),
        structured_feature_fill_score(listing),
        recency_score(listing, now=now),
        agency_quality_score(getattr(listing, "agency_id", None), agency_quality),
    )


def quality_score_breakdown(
    listing: object,
    *,
    agency_quality: dict[str, float] | None = None,
    now: datetime | None = None,
) -> QualityBreakdown:
    """Per-sub-score breakdown plus coaching-template ids for weak sub-scores."""
    photo = photo_score(listing)
    description = description_completeness_score(listing)
    structured = structured_feature_fill_score(listing)
    recency = recency_score(listing, now=now)
    agency = agency_quality_score(getattr(listing, "agency_id", None), agency_quality)

    improvements: list[str] = []
    if photo < PHOTO_SCORE_THRESHOLD:
        if not has_room_coverage(listing):
            improvements.append("add_kitchen_bathroom_photos")
        improvements.append("add_more_photos")
    if description < DESCRIPTION_SCORE_THRESHOLD:
        improvements.append("expand_description")
    if structured < STRUCTURED_SCORE_THRESHOLD:
        improvements.append("fill_structured_fields")
    if recency < RECENCY_SCORE_THRESHOLD:
        improvements.append("refresh_listing")

    return QualityBreakdown(
        quality_score=_combine(photo, description, structured, recency, agency),
        photo_score=photo,
        description_completeness_score=description,
        structured_feature_fill_score=structured,
        recency_score=recency,
        agency_quality_score=agency,
        suggested_improvements=improvements,
    )
