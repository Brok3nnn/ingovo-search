"""Unit tests for §8 scoring — `compute_score`, `softpref_core_match`, cosine.

`compute_score` is pure (given precomputed `pooled_signal_value` /
`geo_score_value`), so the §8 acceptance tests 3, 4 (in `test_anchor_score`),
5, 6, 7 land here. The DB-backed `recommend` orchestrator (acceptance 1, 2, 8)
lives in `tests/integration/recommender/test_score.py`.

Test fixtures build real `Property` instances — the `ListingFeatures` Protocol
is structural, so `Property` satisfies it without a DB.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from ingovo_search.db.models import Property
from ingovo_search.recommender.candidate_rows import fetch_searchable_knn_ids
from ingovo_search.recommender.parser_schema import Anchors, GeoHints, HardFilters, ParsedQuery
from ingovo_search.recommender.recommend import recommend
from ingovo_search.recommender.score import (
    WEIGHTS,
    ScoreBreakdown,
    ScoreInputs,
    compute_score,
    cosine_similarity,
    match_percentages,
    softpref_core_match,
)
from ingovo_search.search.scoring import build_scored_candidates

NOW = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)


async def test_recommend_carries_batched_coordinates_into_public_results() -> None:
    rows = [_make_property(id=property_id) for property_id in ["C", "B", "A"]]
    session = AsyncMock()
    coordinates = {"A": (47.4351902, 19.075689), "C": (47.51, 19.04)}
    query = ParsedQuery()
    with (
        patch(
            "ingovo_search.recommender.recommend.fetch_properties_in_order", new_callable=AsyncMock, return_value=rows
        ),
        patch(
            "ingovo_search.recommender.recommend.fetch_property_coords",
            new_callable=AsyncMock,
            return_value=coordinates,
        ) as fetch_coordinates,
        patch("ingovo_search.recommender.recommend.pooled_signals", new_callable=AsyncMock, return_value={}),
        patch(
            "ingovo_search.recommender.recommend.fetch_embedding_similarities", new_callable=AsyncMock, return_value={}
        ),
    ):
        ranked = await recommend(
            session, query, query_emb=[0.0] * 4, client=AsyncMock(), candidate_ids=[row.id for row in rows], now=NOW
        )

    fetch_coordinates.assert_awaited_once_with(session, ["C", "B", "A"])
    candidates = build_scored_candidates(ranked, query)
    public = {candidate.listing_id: candidate.model_dump(mode="json")["property"] for candidate in candidates}
    for property_id, (latitude, longitude) in coordinates.items():
        assert public[property_id]["latitude"] == latitude
        assert public[property_id]["longitude"] == longitude
    assert public["B"]["latitude"] is None
    assert public["B"]["longitude"] is None


def _make_property(**overrides: Any) -> Property:
    defaults: dict[str, Any] = {
        "id": "p-1",
        "title": "Test",
        "city": "Budapest",
        "status": "active",
        "images": [],
        "view_count": 0,
        "save_count": 0,
        "contact_count": 0,
        "photo_count": 0,
        "created_at": NOW,
    }
    defaults.update(overrides)
    return Property(**defaults)


def _query(
    *,
    soft_preferences: list[str] | None = None,
    soft_prefs_freetext: list[str] | None = None,
    anchors: Anchors | None = None,
    geo_hints: GeoHints | None = None,
) -> ParsedQuery:
    return ParsedQuery(
        hard_filters=HardFilters(),
        anchors=anchors or Anchors(),
        soft_preferences=list(soft_preferences) if soft_preferences else [],  # type: ignore[arg-type]
        soft_prefs_freetext=list(soft_prefs_freetext) if soft_prefs_freetext else [],
        geo_hints=geo_hints or GeoHints(),
    )


# --- cosine_similarity ------------------------------------------------------


def test_cosine_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]) == 1.0


def test_cosine_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]) == 0.0


def test_cosine_missing_or_empty_vectors_score_zero() -> None:
    assert cosine_similarity(None, [1.0, 0.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], None) == 0.0
    assert cosine_similarity([], [1.0, 0.0]) == 0.0


def test_cosine_mismatched_dimensions_score_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_cosine_zero_norm_vector_scores_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# --- softpref_core_match (Acceptance 3) -------------------------------------


def test_softpref_core_match_returns_half_for_one_of_two() -> None:
    """§8 acceptance 3: ["parking", "renovated"] with parking-yes / no-renovation → 0.5."""
    prop = _make_property(has_parking=True, renovation_state="original")
    assert softpref_core_match(["parking", "renovated"], prop) == 0.5


def test_softpref_core_match_full_overlap_returns_one() -> None:
    prop = _make_property(has_parking=True, renovation_state="fully_renovated")
    assert softpref_core_match(["parking", "renovated"], prop) == 1.0


def test_softpref_core_match_empty_prefs_returns_zero() -> None:
    assert softpref_core_match([], _make_property()) == 0.0


def test_softpref_core_match_handles_missing_features_safely() -> None:
    """Acceptance 7: missing categorical / boolean features stay eligible."""
    prop = _make_property(has_parking=None, has_garage=None, renovation_state=None)
    # No exception; nothing matches.
    assert softpref_core_match(["parking", "renovated"], prop) == 0.0


# --- compute_score: top match (Acceptance 5) --------------------------------


def test_compute_score_on_top_match_exceeds_07() -> None:
    """§8 acceptance 5: a deliberate top-match scores > 0.7."""
    emb = [1.0, 0.0, 0.0, 0.0]
    prop = _make_property(
        has_parking=True,
        renovation_state="fully_renovated",
        price_huf=80_000_000,
        area_m2=60.0,
        rooms="2",
        quality_score=0.9,
        descriptive_embedding=emb,
        factual_embedding=emb,
        created_at=NOW,
    )
    query = _query(
        soft_preferences=["parking", "renovated"],
        soft_prefs_freetext=["jó otthonról dolgozni"],
        anchors=Anchors(price_huf=80_000_000, area_m2=60.0, rooms=2),
    )

    breakdown = compute_score(
        prop,
        query,
        ScoreInputs(
            query_emb=emb, freetext_pref_emb=emb, pooled_signal_value=2.0, geo_score_value=1.0, explore=1.0, now=NOW
        ),
    )
    assert breakdown.total > 0.7


# --- compute_score: deliberate mismatch (Acceptance 5) ----------------------


def test_compute_score_on_mismatch_below_03() -> None:
    """§8 acceptance 5: a deliberate mismatch scores < 0.3."""
    prop = _make_property(
        has_parking=False,
        renovation_state="original",
        price_huf=200_000_000,
        area_m2=200.0,
        rooms="6",
        quality_score=0.0,
        descriptive_embedding=[0.0, 1.0, 0.0, 0.0],
        factual_embedding=[0.0, 1.0, 0.0, 0.0],
        created_at=NOW - timedelta(days=365),
    )
    query = _query(
        soft_preferences=["parking", "renovated"],
        anchors=Anchors(price_huf=50_000_000, area_m2=40.0, rooms=2),
    )

    breakdown = compute_score(
        prop,
        query,
        ScoreInputs(
            query_emb=[1.0, 0.0, 0.0, 0.0],
            freetext_pref_emb=None,
            pooled_signal_value=-2.0,
            geo_score_value=0.0,
            explore=0.0,
            now=NOW,
        ),
    )
    assert breakdown.total < 0.3


# --- ScoreBreakdown structure (Acceptance 6) --------------------------------


def test_score_breakdown_exposes_all_ten_term_contributions() -> None:
    """§8 acceptance 6: all ten weight terms appear in `ScoreBreakdown`."""
    breakdown = compute_score(_make_property(), _query(), ScoreInputs(explore=0.5, now=NOW))
    assert isinstance(breakdown, ScoreBreakdown)
    for term in (
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
    ):
        assert hasattr(breakdown, term)
    assert hasattr(breakdown, "total")


def test_weights_dict_matches_spec_step_5() -> None:
    # v0.5: buyer-facing geo precision pass; geographic intent is strong enough
    # to keep wrong-location listings below the requested area.
    assert WEIGHTS == {
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


def test_weights_sum_within_unit_band() -> None:
    """M4 calibration safety net: catch a fat-finger like `quality: 5.0` or
    a typo that drops every weight to 0. Allows ±10% off unit-sum so
    intentional partial calibrations (sum 0.95-1.05) are still legal."""
    total = sum(WEIGHTS.values())
    assert 0.90 <= total <= 1.10, f"WEIGHTS sum {total} out of [0.9, 1.1] band"


def test_weights_are_non_negative_and_bounded() -> None:
    """Every weight must be in [0, 1] — a negative weight would make the score
    incentivise the wrong direction, and >1 single-term weights overwhelm everything."""
    for term, value in WEIGHTS.items():
        assert 0.0 <= value <= 1.0, f"WEIGHTS[{term!r}] = {value} outside [0, 1]"


def test_score_breakdown_total_equals_sum_of_contributions() -> None:
    breakdown = compute_score(
        _make_property(quality_score=0.5, created_at=NOW),
        _query(),
        ScoreInputs(query_emb=[1.0, 0.0, 0.0, 0.0], pooled_signal_value=0.0, geo_score_value=0.5, explore=0.5, now=NOW),
    )
    total = (
        breakdown.content
        + breakdown.factual
        + breakdown.softpref_core
        + breakdown.softpref_free
        + breakdown.anchor
        + breakdown.quality
        + breakdown.geo
        + breakdown.pooled
        + breakdown.recency
        + breakdown.explore
    )
    assert abs(breakdown.total - total) < 1e-9


# --- compute_score: determinism / edge cases --------------------------------


def test_compute_score_is_deterministic_when_explore_pinned() -> None:
    args: dict[str, Any] = {
        "query_emb": [1.0, 0.0, 0.0, 0.0],
        "pooled_signal_value": 0.5,
        "geo_score_value": 0.3,
        "explore": 0.5,
        "now": NOW,
    }
    a = compute_score(_make_property(quality_score=0.4, created_at=NOW), _query(), ScoreInputs(**args))
    b = compute_score(_make_property(quality_score=0.4, created_at=NOW), _query(), ScoreInputs(**args))
    assert a.total == b.total


def test_compute_score_zero_when_query_emb_missing() -> None:
    breakdown = compute_score(
        _make_property(descriptive_embedding=[1.0, 0.0], factual_embedding=[1.0, 0.0], created_at=NOW),
        _query(),
        ScoreInputs(query_emb=None, pooled_signal_value=0.0, geo_score_value=0.0, explore=0.0, now=NOW),
    )
    assert breakdown.content == 0.0
    assert breakdown.factual == 0.0


def test_compute_score_pooled_term_normalisation_boundaries() -> None:
    """`pooled_signal_value` ∈ ~[-2, 2] maps into the pooled term via `0.5 + p/4`."""
    base = _make_property(created_at=NOW)
    q = _query()
    common: dict[str, Any] = {"explore": 0.0, "now": NOW}
    high = compute_score(base, q, ScoreInputs(pooled_signal_value=2.0, **common))
    mid = compute_score(base, q, ScoreInputs(pooled_signal_value=0.0, **common))
    low = compute_score(base, q, ScoreInputs(pooled_signal_value=-2.0, **common))
    assert abs(high.pooled - WEIGHTS["pooled"] * 1.0) < 1e-9
    assert abs(mid.pooled - WEIGHTS["pooled"] * 0.5) < 1e-9
    assert low.pooled == 0.0


def test_compute_score_anchor_term_zero_without_anchors_or_listing_values() -> None:
    breakdown = compute_score(
        _make_property(price_huf=None, area_m2=None, rooms=None),
        _query(anchors=Anchors()),
        ScoreInputs(query_emb=[1.0, 0.0, 0.0, 0.0], explore=0.0, now=NOW),
    )
    assert breakdown.anchor == 0.0


def test_compute_score_anchor_term_uses_listing_price_when_anchor_set() -> None:
    breakdown = compute_score(
        _make_property(price_huf=80_000_000),
        _query(anchors=Anchors(price_huf=80_000_000)),
        ScoreInputs(query_emb=[1.0, 0.0, 0.0, 0.0], explore=0.0, now=NOW),
    )
    # Single matched anchor: anchor_score == 1.0 → contribution = w_anchor * 1.0.
    assert abs(breakdown.anchor - WEIGHTS["anchor"]) < 1e-9


def test_compute_score_anchor_term_ignores_a_legacy_price_key_on_a_payload_shaped_listing() -> None:
    # A row has no `price` since Alembic 0061; a payload-shaped listing may still
    # carry the key, and the canonical `price_huf` must win over it.
    listing = _make_property(price_huf=80_000_000, currency="HUF")
    vars(listing)["price"] = 180_000_000.0
    breakdown = compute_score(
        listing,
        _query(anchors=Anchors(price_huf=80_000_000)),
        ScoreInputs(query_emb=[1.0, 0.0, 0.0, 0.0], explore=0.0, now=NOW),
    )

    assert abs(breakdown.anchor - WEIGHTS["anchor"]) < 1e-9


@pytest.mark.asyncio
async def test_searchable_knn_query_excludes_expired_rows_at_query_time() -> None:
    session = AsyncMock()
    session.execute.return_value = []
    with patch(
        "ingovo_search.recommender.candidate_rows.vector_bind_value", new_callable=AsyncMock, return_value="[0,0]"
    ):
        await fetch_searchable_knn_ids(session, [0.0] * 1024, kind="factual", limit=10)

    statement = str(session.execute.await_args.args[0])
    assert "expires_at IS NULL OR expires_at > now()" in statement


def test_compute_score_breakdown_is_frozen() -> None:
    breakdown = compute_score(_make_property(), _query(), ScoreInputs(explore=0.0, now=NOW))
    with pytest.raises(ValidationError):
        breakdown.total = 99.0  # type: ignore[misc]


# --- match_percentages (results-UI "egyezés %") ------------------------------


def _bd(total: float, **terms: float) -> ScoreBreakdown:
    """Build a ScoreBreakdown with the given term contributions (rest 0)."""
    fields: dict[str, float] = {name: 0.0 for name in WEIGHTS}
    fields.update(terms)
    return ScoreBreakdown(total=total, **fields)


def test_match_percentages_empty_input_is_query_aware() -> None:
    assert match_percentages([], ParsedQuery()) == []


def test_match_percentages_no_active_dimension_floors_to_one() -> None:
    assert match_percentages([_bd(0.0)], ParsedQuery()) == [1]


def test_query_aware_match_percent_can_reach_100_for_exact_structured_fit() -> None:
    query = _query(
        anchors=Anchors(price_huf=80_000_000, area_m2=60.0),
        geo_hints=GeoHints(region_name="Balaton"),
    )
    breakdown = _bd(
        WEIGHTS["anchor"] + WEIGHTS["geo"],
        anchor=WEIGHTS["anchor"],
        geo=WEIGHTS["geo"],
    )
    assert match_percentages([breakdown], query) == [100]


def test_query_aware_match_percent_caps_wrong_location() -> None:
    query = _query(
        anchors=Anchors(price_huf=80_000_000),
        geo_hints=GeoHints(region_name="Balaton"),
    )
    breakdown = _bd(
        WEIGHTS["content"] + WEIGHTS["factual"] + WEIGHTS["anchor"],
        content=WEIGHTS["content"],
        factual=WEIGHTS["factual"],
        anchor=WEIGHTS["anchor"],
        geo=0.0,
    )
    assert match_percentages([breakdown], query)[0] <= 49


def test_query_aware_match_percent_ignores_quality_recency_and_explore() -> None:
    query = _query(anchors=Anchors(price_huf=80_000_000))
    noisy_ranking_terms = WEIGHTS["quality"] + WEIGHTS["recency"] + WEIGHTS["explore"]
    assert match_percentages(
        [_bd(noisy_ranking_terms, quality=WEIGHTS["quality"], recency=WEIGHTS["recency"], explore=WEIGHTS["explore"])],
        query,
    ) == [1]


# --- amenity verification (honest match for "úszómedencés villa") ------------


def _amenity_query() -> ParsedQuery:
    return ParsedQuery(
        semantic_query="úszómedencés villa",
        soft_prefs_freetext=["úszómedence"],
    )


def test_compute_score_downranks_listing_without_requested_pool() -> None:
    emb = [1.0, 0.0, 0.0, 0.0]
    query = _amenity_query()
    with_pool = compute_score(
        _make_property(
            description="Luxusvilla saját úszómedencével.", descriptive_embedding=emb, factual_embedding=emb
        ),
        query,
        ScoreInputs(query_emb=emb, freetext_pref_emb=emb, geo_score_value=0.0, explore=0.0, now=NOW),
    )
    without_pool = compute_score(
        _make_property(
            description="Tágas villa nagy kerttel, kandallóval.", descriptive_embedding=emb, factual_embedding=emb
        ),
        query,
        ScoreInputs(query_emb=emb, freetext_pref_emb=emb, geo_score_value=0.0, explore=0.0, now=NOW),
    )
    # The pool-less villa is verified at 0.0 and penalised; the real one isn't.
    assert with_pool.amenity_match == 1.0
    assert without_pool.amenity_match == 0.0
    assert without_pool.total < with_pool.total


def test_compute_score_no_amenity_penalty_when_none_requested() -> None:
    emb = [1.0, 0.0, 0.0, 0.0]
    breakdown = compute_score(
        _make_property(description="bármi", descriptive_embedding=emb, factual_embedding=emb),
        _query(),
        ScoreInputs(query_emb=emb, explore=0.0, now=NOW),
    )
    assert breakdown.amenity_match == 1.0


def test_match_percentages_caps_unverified_amenity() -> None:
    query = _amenity_query()
    strong = WEIGHTS["content"] + WEIGHTS["factual"]
    # A semantically strong match (would read high) but the requested pool is
    # unverified (amenity_match=0.0) → capped to the honest base (~45%).
    capped = ScoreBreakdown(
        total=strong,
        content=WEIGHTS["content"],
        factual=WEIGHTS["factual"],
        softpref_core=0.0,
        softpref_free=0.0,
        anchor=0.0,
        quality=0.0,
        geo=0.0,
        pooled=0.0,
        recency=0.0,
        explore=0.0,
        amenity_match=0.0,
    )
    pct = match_percentages([capped], query)[0]
    assert pct <= 45


def test_match_percentages_unaffected_when_amenity_verified() -> None:
    query = _amenity_query()
    strong = WEIGHTS["content"] + WEIGHTS["factual"]
    verified = ScoreBreakdown(
        total=strong,
        content=WEIGHTS["content"],
        factual=WEIGHTS["factual"],
        softpref_core=0.0,
        softpref_free=0.0,
        anchor=0.0,
        quality=0.0,
        geo=0.0,
        pooled=0.0,
        recency=0.0,
        explore=0.0,
        amenity_match=1.0,
    )
    # Verified amenity → no cap, reads high.
    assert match_percentages([verified], query)[0] > 60
