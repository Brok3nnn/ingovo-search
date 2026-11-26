"""§8 recommender scoring service — `compute_score` and the match percentages.

`compute_score` is pure: given a listing, a `ParsedQuery`, the two query
embeddings, and precomputed `pooled_signal` / `geo_score` values, it returns a
`ScoreBreakdown` carrying every weighted contribution (Acceptance test 6) plus
the total. The 10 weights match the design spec §8 step 5 and the formula
matches the design spec §6.4 "Runtime pipeline" step 6.

`recommend`, the DB-backed orchestrator (SQL hard filter → vector pre-filter →
per-candidate geo + pooled-signal lookups → score → sort top-N), lives in
`ingovo_search.recommender.recommend`. It is integration-tested only;
the pure scoring layer here carries the heavy unit-test coverage.

Two deviations from the spec are deliberate and documented:

- §8 step 5 names a single `freetext_emb` arg, but the §6.4 formula needs two
  distinct embeddings: `query_emb` (the `semantic_query` embedding) for the
  content + factual terms, and `freetext_pref_emb` (the combined
  `soft_prefs_freetext` embedding) for the `softpref_free` term. `compute_score`
  takes both.
- The `pooled_signal` raw value can sit anywhere in roughly `[-2, 2]` (signal
  values), so it is mapped to `[0, 1]` via `0.5 + p/4` (clamped) before being
  multiplied by `w_pooled`. The §6.4 formula does not specify normalisation.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from pydantic import BaseModel, Field, model_validator

from ingovo_search.lib.property_price import resolve_property_price_huf
from ingovo_search.recommender.amenity_verify import amenity_match_ratio
from ingovo_search.recommender.anchor_score import anchor_score
from ingovo_search.recommender.match_percent import MatchBreakdown, MatchComponents
from ingovo_search.recommender.parser_schema import Anchors, ParsedQuery
from ingovo_search.recommender.quality_score import recency_score
from ingovo_search.recommender.search_contract import NEARBY_MAX_DISTANCE_KM, ResultScope
from ingovo_search.recommender.softpref_rules import SOFTPREF_RULES

# §8 step 5 weights — v0 hand-set, then §17.2.6 calibrated.
# v0.1 (calibration #1, 179 ratings, geo x quality grid): `quality: 0.10 → 0.05`.
# v0.2 (calibration #2, 179 ratings, content x geo x anchor grid):
#   `content: 0.20 → 0.25`, `geo: 0.10 → 0.15`, `anchor: 0.10 → 0.075`.
# v0.3 (calibration #3, 354 ratings, content x factual x softpref_core grid):
#   `content: 0.25 → 0.20`, `factual: 0.15 → 0.10`, `softpref_core: 0.12 → 0.16`.
# v0.4 (calibration #4, 354 ratings, content x geo x quality grid):
#   NO WEIGHT CHANGE — the grid top-1 reproduced the current defaults. The
#   M1 §4 enrichment backfill (cleaned_description + summaries + populated
#   `quality_score`) alone moved avg nDCG@10 from 0.8601 (v0.3 measurement)
#   to 0.8798 on the same 354 ratings (+1.97 pp), with weights unchanged.
#   The hidden win is in the inputs: Cohere now embeds cleaner text, and the
#   `quality` term contributes a real signal instead of 0.
# v0.5: buyer-facing geo precision pass. Geographic intent must dominate over
# generic text/quality signals, so a Budapest listing cannot outrank a Balaton
# listing on a Balaton query just because its description is semantically rich.
WEIGHTS_VERSION = "v0.5"  # bump when any value below changes
WEIGHTS: dict[str, float] = {
    "content": 0.17,
    "factual": 0.08,
    "softpref_core": 0.15,
    "softpref_free": 0.07,
    "anchor": 0.08,
    "quality": 0.04,
    "geo": 0.25,
    "pooled": 0.04,
    "recency": 0.04,
    "explore": 0.02,
}


class ScoreBreakdown(BaseModel):
    """Per-listing weighted contribution per scoring term, plus the total.

    Every field is the term's *weighted* contribution (`w_term * term_value`),
    so summing the 10 term fields equals `total`. Acceptance 6 requires all
    ten term names to appear.
    """

    model_config = {"frozen": True}

    content: float
    factual: float
    softpref_core: float
    softpref_free: float
    anchor: float
    quality: float
    geo: float
    pooled: float
    recency: float
    explore: float
    total: float
    # Fraction of the buyer's explicitly-requested concrete amenities (pool,
    # fireplace, …) that the listing's text actually attests. 1.0 = nothing
    # requested or all verified (no penalty). Drives both the ranking penalty
    # (in `compute_score`) and the honest display cap (in `match_percentages`).
    amenity_match: float = 1.0


class ScoredCandidate(BaseModel):
    """One ranked listing returned to §14 API consumers.

    Carries the listing id, total score, full per-term breakdown, and the
    minimum listing metadata the clarification policy + frontend need
    without the full ORM row. `district` and `city` mirror `HardFilters` so
    the split-top-interpretations clarification trigger can spot geographic
    drift across the top-N. `ai_narrative` is the §17.3 narrator output that
    the search-results UX renders verbatim — present on hero rows (top 1-3)
    and `None` on secondary rows that only need the compact card treatment.
    """

    model_config = {"frozen": True}

    listing_id: str
    rank: int = Field(default=0, ge=0)
    score: float
    match_percent: int = Field(default=1, ge=1, le=100)
    result_scope: ResultScope = ResultScope.EXACT
    distance_km: float | None = Field(default=None, ge=0.0, le=NEARBY_MAX_DISTANCE_KM)
    # Internal R4 diagnostics. R7 promotes the allowed public subset; excluding
    # it here prevents an internal OUT_OF_SCOPE state from leaking through API.
    match_breakdown: MatchBreakdown | None = Field(default=None, exclude=True)
    # The allowed public subset the line above refers to: the per-criterion
    # scores only, with the raw scope/distance and the weighting internals left
    # behind. See MatchComponents for why the split is not cosmetic — the parent
    # sanitises result_scope, and a serialised nested breakdown would bypass it.
    # `None` on a component means the query never asked about that criterion.
    match_components: MatchComponents | None = None
    rerank_relevance: float | None = Field(default=None, ge=0.0, le=1.0, exclude=True)
    breakdown: ScoreBreakdown
    district: str | None = None
    city: str | None = None
    rationale: str | None = None
    compromise: str | None = None
    # Every requested condition this listing does not satisfy (hard mismatches +
    # unmet/unverified requested amenities), human-readable. The card surfaces all
    # of them with a warning marker; `compromise` keeps the first for back-compat.
    compromises: list[str] = Field(default_factory=list)
    property: dict[str, Any] | None = None
    ai_narrative: dict[str, Any] | None = None  # AINarrative.model_dump() when present

    @model_validator(mode="after")
    def validate_public_geography(self) -> ScoredCandidate:
        if self.result_scope is ResultScope.OUT_OF_SCOPE:
            raise ValueError("out_of_scope candidates cannot be public")
        if self.result_scope is ResultScope.EXACT and self.distance_km is not None:
            raise ValueError("exact candidates must not carry distance_km")
        if self.result_scope is ResultScope.NEARBY and self.distance_km is None:
            raise ValueError("nearby candidates require distance_km")
        return self


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


_DISPLAY_MATCH_WEIGHTS: dict[str, float] = {
    "content": 0.12,
    "factual": 0.08,
    "softpref_core": 0.16,
    "softpref_free": 0.10,
    "anchor": 0.24,
    "geo": 0.40,
}

# Hybrid display scaling.
#
# Structured terms (anchor / geo / soft-prefs) can legitimately reach their full
# weight — an exact price/area/location/pref fit really is a perfect match — so
# they stay absolute and an exact structured hit reads a true 100%.
#
# The fuzzy semantic terms (content/factual) never reach a literal 1.0 (cosine
# of two distinct texts tops out well under 1), so on a descriptive query they
# would pin even a strong match around 75%. For those we saturate: a normalized
# semantic score at/above `_SEMANTIC_FULL_MATCH` counts as a full textual match,
# scaled by a ceiling just under 100% so a descriptive search reads high but
# stays a notch below an exact structured fit. This is set-independent — a good
# listing reads high regardless of what else is in the result set.
_SEMANTIC_DISPLAY_TERMS: frozenset[str] = frozenset({"content", "factual"})
# Normalized semantic raw at/above this is treated as a full textual match.
_SEMANTIC_FULL_MATCH: float = 0.6
# Pure-semantic matches top out here rather than a literal 100%.
_SEMANTIC_DISPLAY_CEILING: float = 0.96

# Amenity verification (§ honest match). When the buyer explicitly requested a
# concrete amenity (pool, fireplace, …) that the listing's text does NOT attest,
# we (a) downrank its total and (b) cap its visible match %, so a pool-less villa
# can't read 83% for "úszómedencés villa". A fully-unverified request multiplies
# the total by `_AMENITY_VERIFY_FLOOR` and caps the % at `_AMENITY_PCT_BASE`;
# partial verification scales linearly up toward no penalty.
_AMENITY_VERIFY_FLOOR: float = 0.45
_AMENITY_PCT_BASE: float = 45.0
_AMENITY_PCT_SPAN: float = 20.0


def _query_has_geo_intent(query: ParsedQuery) -> bool:
    geo = query.geo_hints
    hard = query.hard_filters
    return any(
        (
            hard.near,
            geo.region_name,
            geo.side_of_river,
            geo.county,
            geo.balaton_shore,
        )
    )


def _query_has_anchor_intent(query: ParsedQuery) -> bool:
    anchors = query.anchors
    return any((anchors.price_huf is not None, anchors.area_m2 is not None, anchors.rooms is not None))


def _active_display_terms(query: ParsedQuery) -> tuple[str, ...]:
    terms: list[str] = []
    if query.semantic_query:
        terms.extend(("content", "factual"))
    if query.soft_preferences:
        terms.append("softpref_core")
    if query.soft_prefs_freetext:
        terms.append("softpref_free")
    if _query_has_anchor_intent(query):
        terms.append("anchor")
    if _query_has_geo_intent(query):
        terms.append("geo")
    return tuple(terms)


def _term_raw(breakdown: ScoreBreakdown, term: str) -> float:
    weight = WEIGHTS.get(term, 0.0)
    if weight <= 0.0:
        return 0.0
    return _clamp01(getattr(breakdown, term) / weight)


def _geo_cap(geo_raw: float) -> int:
    """Cap visible match when the listing misses a requested location.

    The UI percentage is buyer-facing fit, not a generic ranking score. When
    the buyer names a place/region, a wrong-location listing must never look
    like a strong match even if it has good text, recency, or listing quality.
    """
    if geo_raw < 0.05:
        return 49
    if geo_raw < 0.35:
        return 64
    if geo_raw < 0.70:
        return 82
    return 100


def match_percentages(breakdowns: Sequence[ScoreBreakdown], query: ParsedQuery) -> list[int]:
    """Return the former breakdown-only absolute percentage for internal compatibility.

    The percentage is computed only from terms that express
    the buyer's request: semantic/factual relevance, explicit soft preferences,
    numeric anchors, and geography. Listing quality, recency, pooled engagement
    and exploration are still useful for ranking, but they are not "egyezés" and
    must not inflate the visible percent.

    Product search uses `match_percent.compute_match_breakdown`, which also has
    access to listing hard-filter fields. This compatibility function is fully
    set-independent and cannot be called without the parsed query.
    """
    if not breakdowns:
        return []

    active_terms = _active_display_terms(query)
    if not active_terms:
        return [1 for _ in breakdowns]

    total_weight = sum(_DISPLAY_MATCH_WEIGHTS[term] for term in active_terms)
    if total_weight <= 0.0:
        return [1 for _ in breakdowns]

    has_geo = "geo" in active_terms

    def _display_term(breakdown: ScoreBreakdown, term: str) -> float:
        raw = _term_raw(breakdown, term)
        if term in _SEMANTIC_DISPLAY_TERMS:
            return _clamp01(raw / _SEMANTIC_FULL_MATCH) * _SEMANTIC_DISPLAY_CEILING
        return raw

    percentages: list[int] = []
    for breakdown in breakdowns:
        raw = sum(_DISPLAY_MATCH_WEIGHTS[term] * _display_term(breakdown, term) for term in active_terms)
        pct = round(_clamp01(raw / total_weight) * 100)
        if has_geo:
            pct = min(pct, _geo_cap(_term_raw(breakdown, "geo")))
        # Honest cap: a requested concrete amenity the listing can't attest must
        # not read as a strong match (a pool-less villa ≠ 83% for "úszómedencés").
        if breakdown.amenity_match < 1.0:
            pct = min(pct, round(_AMENITY_PCT_BASE + _AMENITY_PCT_SPAN * breakdown.amenity_match))
        percentages.append(max(1, min(100, pct)))
    return percentages


def cosine_similarity(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    """Cosine similarity; returns 0.0 if either vector is missing or zero-norm."""
    if a is None or b is None or len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def softpref_core_match(prefs: Sequence[str], listing: Any) -> float:
    """Fraction of `prefs` whose Appendix A rule evaluates `True` on `listing`."""
    if not prefs:
        return 0.0
    matches = 0
    for pref in prefs:
        rule = SOFTPREF_RULES.get(pref)  # type: ignore[arg-type]
        if rule is not None and rule(listing):
            matches += 1
    return matches / len(prefs)


def _rooms_as_float(rooms_text: str | None) -> float | None:
    """Best-effort numeric reading of Property.rooms (stored as text)."""
    if rooms_text is None:
        return None
    try:
        return float(rooms_text.strip())
    except ValueError:
        return None


def _mean_anchor_score(anchors: Anchors, listing: Any) -> float:
    """Average `anchor_score` over the anchors the query supplies + listing has."""
    scores: list[float] = []
    price_huf = resolve_property_price_huf(listing)
    if anchors.price_huf is not None and price_huf is not None:
        scores.append(anchor_score(price_huf, anchors.price_huf, "price_huf"))
    if anchors.area_m2 is not None and listing.area_m2 is not None:
        scores.append(anchor_score(listing.area_m2, anchors.area_m2, "area_m2"))
    if anchors.rooms is not None:
        rooms_val = _rooms_as_float(listing.rooms)
        if rooms_val is not None:
            scores.append(anchor_score(rooms_val, float(anchors.rooms), "rooms"))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _normalize_pooled(value: float) -> float:
    """Map a raw §9 pooled-signal value (roughly `[-2, 2]`) into `[0, 1]`."""
    return _clamp01(0.5 + value / 4.0)


def _raw_embedding_similarity(listing: Any, attr: str, query_vec: Sequence[float] | None) -> float:
    """Fallback similarity from a raw stored embedding, tolerating absence.

    The batch path pre-computes similarities in SQL and passes them in; this
    fallback only runs for listings missing from that dict — typically rows
    whose §5 embedding backfill hasn't happened yet. The scoring fetch
    (`candidate_rows.SCORING_PROPERTY_COLUMNS`) deliberately doesn't load the raw vectors,
    so a missing attribute (or NULL vector) must mean "no credit", never a
    crash: a freshly ingested, not-yet-embedded listing in the candidate
    pool would otherwise 500 the whole search.
    """
    if query_vec is None:
        return 0.0
    vec = getattr(listing, attr, None)
    if vec is None:
        return 0.0
    return _clamp01(cosine_similarity(query_vec, vec))


@dataclass(frozen=True, slots=True)
class ScoreInputs:
    """What `compute_score` reads besides the listing and the query; every field is optional.

    `query_emb` / `freetext_pref_emb` feed the fallback similarities for a listing
    the batch path has no precomputed value for. The similarities, the pooled
    signal and the geo score are the batch path's precomputed per-listing values.
    `explore`, `now` and `weights` pin the randomness, the clock and the per-term
    weights (tests, §17.2.6 calibration).
    """

    query_emb: Sequence[float] | None = None
    freetext_pref_emb: Sequence[float] | None = None
    content_similarity: float | None = None
    factual_similarity: float | None = None
    softpref_free_similarity: float | None = None
    pooled_signal_value: float = 0.0
    geo_score_value: float = 0.0
    explore: float | None = None
    now: datetime | None = None
    weights: dict[str, float] | None = None


DEFAULT_SCORE_INPUTS: Final = ScoreInputs()


def compute_score(listing: Any, query: ParsedQuery, inputs: ScoreInputs = DEFAULT_SCORE_INPUTS) -> ScoreBreakdown:
    """Weighted §6.4 score for one listing under one parsed query.

    `inputs.explore` defaults to `random.random()` per §6.4 "exploration
    injection"; tests pin it for determinism. `inputs.now` is forwarded to
    `recency_score` for the same reason. `inputs.weights` overrides the
    module-level `WEIGHTS` dict per term — used by §17.2.6 grid-search
    calibration. Missing keys fall back to the default value, so a partial
    override (e.g. only `{"geo": 0.18}`) leaves the other nine terms at their
    hand-set defaults.
    """
    w = WEIGHTS if inputs.weights is None else {**WEIGHTS, **inputs.weights}
    content_raw = (
        _clamp01(inputs.content_similarity)
        if inputs.content_similarity is not None
        else _raw_embedding_similarity(listing, "descriptive_embedding", inputs.query_emb)
    )
    factual_raw = (
        _clamp01(inputs.factual_similarity)
        if inputs.factual_similarity is not None
        else _raw_embedding_similarity(listing, "factual_embedding", inputs.query_emb)
    )
    softpref_core_raw = softpref_core_match(query.soft_preferences, listing)
    softpref_free_raw = (
        _clamp01(inputs.softpref_free_similarity)
        if inputs.softpref_free_similarity is not None
        else (
            _raw_embedding_similarity(listing, "descriptive_embedding", inputs.freetext_pref_emb)
            if query.soft_prefs_freetext
            else 0.0
        )
    )
    anchor_raw = _mean_anchor_score(query.anchors, listing)
    quality_raw = listing.quality_score or 0.0
    geo_raw = _clamp01(inputs.geo_score_value)
    pooled_raw = _normalize_pooled(inputs.pooled_signal_value)
    recency_raw = recency_score(listing, now=inputs.now)
    explore_raw = inputs.explore if inputs.explore is not None else random.random()

    contributions = {
        "content": w["content"] * content_raw,
        "factual": w["factual"] * factual_raw,
        "softpref_core": w["softpref_core"] * softpref_core_raw,
        "softpref_free": w["softpref_free"] * softpref_free_raw,
        "anchor": w["anchor"] * anchor_raw,
        "quality": w["quality"] * quality_raw,
        "geo": w["geo"] * geo_raw,
        "pooled": w["pooled"] * pooled_raw,
        "recency": w["recency"] * recency_raw,
        "explore": w["explore"] * explore_raw,
    }
    total = sum(contributions.values())
    # Honest amenity check: downrank listings that can't attest a concrete
    # amenity the buyer explicitly asked for (1.0 = nothing requested / all
    # verified → no change). A fully-unverified request scales the total down to
    # `_AMENITY_VERIFY_FLOOR`; partial verification interpolates toward 1.0.
    amenity_match = amenity_match_ratio(query, listing)
    if amenity_match < 1.0:
        total *= _AMENITY_VERIFY_FLOOR + (1.0 - _AMENITY_VERIFY_FLOOR) * amenity_match
    return ScoreBreakdown(**contributions, total=total, amenity_match=amenity_match)
