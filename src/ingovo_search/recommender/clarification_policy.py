"""§14 step 7 — clarification policy.

`should_clarify(parsed, scored)` returns at most one skippable `Clarification`
based on five trigger categories, in priority order:

1. **Conflicting constraints** — `parsed.conflict_flag` set by the parser.
2. **Low parser confidence** — `parsed.parse_confidence` carries a value below
   `LOW_CONFIDENCE_THRESHOLD`, or no confidence telemetry at all AND every
   parsed field is empty.
3. **Ambiguous geography** — `geo_hints.region_name` is set but no `district_in`
   was extracted, so we can't disambiguate Buda's "rózsadomb" vs Pest's.
4. **Too many weak results** — at least `WEAK_MAJORITY_FRACTION` of the top-N
   carry `score < WEAK_RESULT_THRESHOLD`.
5. **Split top-result interpretations** — the top-N split across two distinct
   districts (each holding ≥ `SPLIT_MIN_SIDE_FRACTION` of the pool).

A thin result set is deliberately not a clarification trigger. The search
pipeline widens non-location constraints, checks verified nearby settlements,
then falls back to the nearest settlement with matching inventory. This keeps
results visible without asking the user to approve a broader location first.

The function returns the first match — never more than one — so it can never
block results: the frontend renders the result list AND the clarification
chip; the user picks the option or skips.

The answer side — `apply_clarification` and its option-id DSL — lives in
`clarification_apply`, and the schema both sides share in `clarification_schema`;
both are re-exported here, so callers keep importing the whole policy from this
module.

The Hungarian question and option labels below are the product's user-facing copy.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from ingovo_search.recommender.clarification_apply import apply_clarification
from ingovo_search.recommender.clarification_schema import (
    Clarification,
    ClarificationAnswer,
    ClarificationOption,
    ClarificationReason,
)
from ingovo_search.recommender.parser_schema import (
    Anchors,
    GeoHints,
    HardFilters,
    LocationCandidate,
    ParsedQuery,
    SideOfRiver,
)
from ingovo_search.recommender.score import ScoredCandidate
from ingovo_search.recommender.sql_filter import DISTRICT_ROMAN

# --- Thresholds (v0 hand-picked; v1 fits on logged session signals) --------

LOW_CONFIDENCE_THRESHOLD: float = 0.5
"""Per-field confidence below this fires the low-confidence trigger."""

MIN_RESULTS_BEFORE_CLARIFY: int = 5
"""Minimum pool size required for split-interpretation detection."""

WEAK_RESULT_THRESHOLD: float = 0.4
"""`score < this` is "weak" for the too-many-weak-results trigger."""

WEAK_MAJORITY_FRACTION: float = 0.6
"""Fraction of the top-N that must be weak to trip the weak-results trigger."""

TOP_N_FOR_SPLIT_DETECTION: int = 10
"""Inspect this many candidates for the split-top-interpretations trigger."""

SPLIT_MIN_SIDE_FRACTION: float = 0.3
"""Both sides of a split need at least this fraction of the inspected window."""


# --- Hungarian user-facing copy ---------------------------------------------

_CLARIFICATION_QUESTION_HU: dict[ClarificationReason, str] = {
    ClarificationReason.CONFLICTING_CONSTRAINTS: (
        "Egymásnak ellentmondó feltételeket találtam. Melyiket lazítsuk egy kicsit?"
    ),
    ClarificationReason.AMBIGUOUS_SETTLEMENT: "Melyik településre gondoltál?",
    ClarificationReason.LOW_PARSER_CONFIDENCE: (
        "Nem vagyok benne biztos, hogy pontosan értem. Tudnál kicsit pontosítani? (pl. ár, szobaszám, kerület)"
    ),
    ClarificationReason.AMBIGUOUS_GEOGRAPHY: ("Ez egy nagyobb terület. Melyik kerület érdekelne leginkább?"),
    ClarificationReason.TOO_MANY_WEAK_RESULTS: (
        "Sok kompromisszumos találatot találtam. Mi az, ami ezek közül a legfontosabb számodra?"
    ),
    ClarificationReason.SPLIT_TOP_INTERPRETATIONS: (
        "A találatok alapján két kerület is szóba jöhet. Melyiket preferálod?"
    ),
}


# --- should_clarify --------------------------------------------------------


def should_clarify(
    parsed: ParsedQuery,
    scored: Sequence[ScoredCandidate],
) -> Clarification | None:
    """Return the first applicable clarification, or `None` if results are fine.

    Triggers are evaluated in priority order; the first match wins. Returns
    `None` rather than blocking — the caller always returns results too.
    """
    if _has_conflict(parsed):
        return _conflict_clarification(parsed)

    if parsed.location_resolution is not None and parsed.location_resolution.status == "ambiguous":
        return _ambiguous_settlement_clarification(parsed.location_resolution.candidates)

    if _has_low_confidence(parsed):
        return _low_confidence_clarification()

    if _has_ambiguous_geography(parsed, scored):
        return _ambiguous_geography_clarification(scored)

    if _has_too_many_weak_results(scored):
        return _weak_results_clarification()

    split_districts = _detect_split_top(scored)
    return None if split_districts is None else _split_clarification(split_districts)


def _has_conflict(parsed: ParsedQuery) -> bool:
    return parsed.conflict_flag


def _has_low_confidence(parsed: ParsedQuery) -> bool:
    """Either a per-field confidence below threshold, or no telemetry + empty parse."""
    if parsed.parse_confidence:
        return any(
            score < LOW_CONFIDENCE_THRESHOLD for score in parsed.parse_confidence.model_dump(exclude_none=True).values()
        )
    return _parse_is_empty(parsed)


def _parse_is_empty(parsed: ParsedQuery) -> bool:
    hf = parsed.hard_filters
    has_filter = any(
        [
            hf.city,
            hf.district_in,
            hf.transaction_type,
            hf.property_type_in,
            hf.property_type_exclude,
            hf.rooms is not None,
            hf.rooms_min is not None,
            hf.area_m2_min is not None,
            hf.area_m2_max is not None,
            hf.price_huf_min is not None,
            hf.price_huf_max is not None,
            hf.floor_min is not None,
            hf.near is not None,
        ]
    )
    has_anchor = (
        parsed.anchors.price_huf is not None or parsed.anchors.area_m2 is not None or parsed.anchors.rooms is not None
    )
    has_softpref = bool(parsed.soft_preferences) or bool(parsed.soft_prefs_freetext)
    has_geo = parsed.geo_hints.region_name is not None or parsed.geo_hints.side_of_river is not None
    return not (has_filter or has_anchor or has_softpref or has_geo)


def _budapest_districts_in(scored: Sequence[ScoredCandidate]) -> list[str]:
    """Distinct Budapest districts (1-23) present in the result window, ordered by
    frequency. Empty for non-Budapest regions — Balaton towns, counties, etc. have
    no `kerület`, so this is how we tell "rózsadomb" (ask which district) apart
    from "Balaton" (districts make no sense, don't ask)."""
    counts: Counter[str] = Counter()
    for cand in scored[:TOP_N_FOR_SPLIT_DETECTION]:
        raw = (cand.district or "").strip().strip(".")
        if raw.isdigit() and 1 <= int(raw) <= 23:
            counts[raw] += 1
    return [district for district, _ in counts.most_common()]


def _has_ambiguous_geography(parsed: ParsedQuery, scored: Sequence[ScoredCandidate]) -> bool:
    """A named region without a chosen district → ask which district — but ONLY
    when the results actually fall in Budapest districts. This stops the nonsense
    of offering "I./II./XII. kerület" for a Balaton search."""
    if parsed.geo_hints.region_name is None or parsed.hard_filters.district_in:
        return False
    return len(_budapest_districts_in(scored)) >= 1


def _has_too_many_weak_results(scored: Sequence[ScoredCandidate]) -> bool:
    if not scored:
        return False
    weak = sum(1 for cand in scored if cand.score < WEAK_RESULT_THRESHOLD)
    return weak / len(scored) >= WEAK_MAJORITY_FRACTION


def _detect_split_top(scored: Sequence[ScoredCandidate]) -> tuple[str, str] | None:
    """Return the two top districts if both hold ≥ `SPLIT_MIN_SIDE_FRACTION`, else None."""
    window = scored[:TOP_N_FOR_SPLIT_DETECTION]
    if len(window) < MIN_RESULTS_BEFORE_CLARIFY:
        return None
    districts = [c.district for c in window if c.district]
    if len(districts) < MIN_RESULTS_BEFORE_CLARIFY:
        return None
    counts = Counter(districts)
    if len(counts) < 2:
        return None
    top_two = counts.most_common(2)
    (a_dist, a_count), (b_dist, b_count) = top_two
    threshold = SPLIT_MIN_SIDE_FRACTION * len(window)
    if a_count >= threshold and b_count >= threshold:
        return (a_dist, b_dist)
    return None


# --- Clarification builders ------------------------------------------------


def _conflict_clarification(parsed: ParsedQuery) -> Clarification:
    """Offer one relax option per constraint pair the parser flagged conflicting."""
    options = _relax_options(parsed.hard_filters)
    if not options:
        # Fallback if parser set the flag but we can't identify a specific field.
        options = [ClarificationOption(id="skip", label="Jó lesz így, mutasd az eredményeket")]
    return Clarification(
        reason=ClarificationReason.CONFLICTING_CONSTRAINTS,
        question=_CLARIFICATION_QUESTION_HU[ClarificationReason.CONFLICTING_CONSTRAINTS],
        options=options,
    )


def _low_confidence_clarification() -> Clarification:
    return Clarification(
        reason=ClarificationReason.LOW_PARSER_CONFIDENCE,
        question=_CLARIFICATION_QUESTION_HU[ClarificationReason.LOW_PARSER_CONFIDENCE],
        options=[],
    )


def _ambiguous_settlement_clarification(candidates: Sequence[LocationCandidate]) -> Clarification:
    return Clarification(
        reason=ClarificationReason.AMBIGUOUS_SETTLEMENT,
        question=_CLARIFICATION_QUESTION_HU[ClarificationReason.AMBIGUOUS_SETTLEMENT],
        options=[
            ClarificationOption(id=f"city:{candidate.settlement_id}", label=f"{candidate.name}, {candidate.county}")
            for candidate in candidates[:4]
        ],
    )


def _ambiguous_geography_clarification(scored: Sequence[ScoredCandidate]) -> Clarification:
    """Offer the districts that the actual results fall in (most common first),
    so the options are always real and relevant — never a hardcoded list."""
    options = [
        ClarificationOption(id=f"district:{d}", label=f"{DISTRICT_ROMAN.get(d, d)}. kerület")
        for d in _budapest_districts_in(scored)[:4]
    ]
    return Clarification(
        reason=ClarificationReason.AMBIGUOUS_GEOGRAPHY,
        question=_CLARIFICATION_QUESTION_HU[ClarificationReason.AMBIGUOUS_GEOGRAPHY],
        options=options,
    )


def _weak_results_clarification() -> Clarification:
    """Offer the most-common soft-pref upgrades — frontend may inline freetext later."""
    options = [
        ClarificationOption(id="softpref:renovated", label="Felújított állapot"),
        ClarificationOption(id="softpref:elevator", label="Van lift"),
        ClarificationOption(id="softpref:parking", label="Van parkoló"),
        ClarificationOption(id="softpref:terrace", label="Teraszos"),
    ]
    return Clarification(
        reason=ClarificationReason.TOO_MANY_WEAK_RESULTS,
        question=_CLARIFICATION_QUESTION_HU[ClarificationReason.TOO_MANY_WEAK_RESULTS],
        options=options,
    )


def _split_clarification(districts: tuple[str, str]) -> Clarification:
    a, b = districts
    options = [
        ClarificationOption(id=f"district:{a}", label=f"{a}. kerület"),
        ClarificationOption(id=f"district:{b}", label=f"{b}. kerület"),
    ]
    return Clarification(
        reason=ClarificationReason.SPLIT_TOP_INTERPRETATIONS,
        question=_CLARIFICATION_QUESTION_HU[ClarificationReason.SPLIT_TOP_INTERPRETATIONS],
        options=options,
    )


def _relax_options(hard_filters: HardFilters) -> list[ClarificationOption]:
    """Build a "which constraint to relax" option list from a `HardFilters`."""
    options: list[ClarificationOption] = []
    if hard_filters.price_huf_max is not None:
        options.append(
            ClarificationOption(
                id="relax:price_huf_max",
                label="Lehet kicsit drágább",
            )
        )
    if hard_filters.price_huf_min is not None:
        options.append(
            ClarificationOption(
                id="relax:price_huf_min",
                label="Lehet olcsóbb is",
            )
        )
    if hard_filters.area_m2_min is not None:
        options.append(
            ClarificationOption(
                id="relax:area_m2_min",
                label="Kisebb méret is jó",
            )
        )
    if hard_filters.area_m2_max is not None:
        options.append(
            ClarificationOption(
                id="relax:area_m2_max",
                label="Nagyobb méret is jó",
            )
        )
    if hard_filters.rooms_min is not None:
        options.append(
            ClarificationOption(
                id="relax:rooms_min",
                label="Kevesebb szoba is elég",
            )
        )
    if hard_filters.near is not None:
        options.append(
            ClarificationOption(
                id="relax:near",
                label="Nagyobb keresési sugár",
            )
        )
    if hard_filters.floor_min is not None:
        options.append(
            ClarificationOption(
                id="relax:floor_min",
                label="Alacsonyabb emelet is jó",
            )
        )
    return options


# --- Re-exports for convenience -------------------------------------------

__all__ = [
    "LOW_CONFIDENCE_THRESHOLD",
    "MIN_RESULTS_BEFORE_CLARIFY",
    "SPLIT_MIN_SIDE_FRACTION",
    "TOP_N_FOR_SPLIT_DETECTION",
    "WEAK_MAJORITY_FRACTION",
    "WEAK_RESULT_THRESHOLD",
    "Anchors",
    "Clarification",
    "ClarificationAnswer",
    "ClarificationOption",
    "ClarificationReason",
    "GeoHints",
    "HardFilters",
    "ParsedQuery",
    "ScoredCandidate",
    "SideOfRiver",
    "apply_clarification",
    "should_clarify",
]
