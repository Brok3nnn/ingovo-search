"""Unit tests for §14 ranking guardrails (acceptance 9 + step 8)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from structlog.testing import capture_logs

from ingovo_search.recommender.ranking_guardrails import (
    MAX_RERANK_DELTA,
    PAID_PLACEMENT_FIELDS,
    RerankMovement,
    assert_hard_filter_preserved,
    assert_no_paid_placement_in_features,
    assert_no_paid_placement_in_weights,
    assert_rerank_candidate_set_preserved,
    enforce_bounded_rerank,
    measure_rerank_movements,
)
from ingovo_search.types.errors import RankingGuardrailViolation

# --- Paid placement --------------------------------------------------------


def test_paid_placement_fields_set_is_non_empty() -> None:
    """Empty set would defeat the guardrail — fail loudly if it ever shrinks."""
    assert len(PAID_PLACEMENT_FIELDS) > 0


def test_assert_no_paid_placement_in_weights_passes_clean_dict() -> None:
    weights = {"content": 0.5, "factual": 0.3, "quality": 0.2}
    assert_no_paid_placement_in_weights(weights)


def test_assert_no_paid_placement_in_weights_rejects_known_field() -> None:
    weights = {"content": 0.5, "boost_paid": 0.5}
    with pytest.raises(RankingGuardrailViolation, match="boost_paid"):
        assert_no_paid_placement_in_weights(weights)


def test_assert_no_paid_placement_in_weights_lists_all_violations() -> None:
    weights = {"is_promoted": 0.1, "ad_score": 0.1, "content": 0.5}
    with pytest.raises(RankingGuardrailViolation) as exc_info:
        assert_no_paid_placement_in_weights(weights)
    message = str(exc_info.value)
    assert "ad_score" in message
    assert "is_promoted" in message


@dataclass
class _FakeListing:
    """Stand-in for a SQLAlchemy `Property` — supports `getattr` lookups."""

    id: str = "x"


def test_assert_no_paid_placement_in_features_passes_clean_listing() -> None:
    assert_no_paid_placement_in_features(_FakeListing())


def test_assert_no_paid_placement_in_features_rejects_truthy_attribute() -> None:
    listing = _FakeListing()
    listing.is_promoted = True  # type: ignore[attr-defined]
    with pytest.raises(RankingGuardrailViolation, match="is_promoted"):
        assert_no_paid_placement_in_features(listing)


def test_assert_no_paid_placement_in_features_ignores_falsy_attribute() -> None:
    """A `False`/`0`/`None` value isn't paid placement — only truthy counts."""
    listing = _FakeListing()
    listing.is_promoted = False  # type: ignore[attr-defined]
    listing.boost_score = 0  # type: ignore[attr-defined]
    listing.placement_paid = None  # type: ignore[attr-defined]
    assert_no_paid_placement_in_features(listing)


# --- Hard-filter preservation ---------------------------------------------


def test_assert_hard_filter_preserved_accepts_subset() -> None:
    assert_hard_filter_preserved(
        hard_filter_ids=["a", "b", "c", "d"],
        post_rerank_ids=["b", "a", "c"],
    )


def test_assert_hard_filter_preserved_rejects_intruder() -> None:
    with pytest.raises(RankingGuardrailViolation, match="z"):
        assert_hard_filter_preserved(
            hard_filter_ids=["a", "b", "c"],
            post_rerank_ids=["a", "z"],
        )


def test_assert_hard_filter_preserved_lists_all_intruders_sorted() -> None:
    with pytest.raises(RankingGuardrailViolation) as exc_info:
        assert_hard_filter_preserved(
            hard_filter_ids=["a"],
            post_rerank_ids=["a", "z", "m", "b"],
        )
    message = str(exc_info.value)
    # Sorted intruders: b, m, z
    assert message.index("'b'") < message.index("'m'") < message.index("'z'")


def test_assert_hard_filter_preserved_accepts_empty() -> None:
    assert_hard_filter_preserved(hard_filter_ids=[], post_rerank_ids=[])


def test_rerank_candidate_set_guard_accepts_reorder() -> None:
    assert_rerank_candidate_set_preserved(pre_rerank_ids=["a", "b", "c"], post_rerank_ids=["c", "a", "b"])


@pytest.mark.parametrize("post", [["a", "b"], ["a", "b", "x"]])
def test_rerank_candidate_set_guard_rejects_missing_or_unknown(post: list[str]) -> None:
    with pytest.raises(RankingGuardrailViolation, match="candidate set changed"):
        assert_rerank_candidate_set_preserved(pre_rerank_ids=["a", "b", "c"], post_rerank_ids=post)


# --- Movement measurement --------------------------------------------------


def test_measure_rerank_movements_pairs_each_id() -> None:
    pre = ["a", "b", "c"]
    post = ["c", "a", "b"]
    movements = measure_rerank_movements(pre, post)

    by_id = {m.listing_id: m for m in movements}
    assert by_id["a"] == RerankMovement(listing_id="a", pre_index=0, post_index=1, delta=1)
    assert by_id["c"] == RerankMovement(listing_id="c", pre_index=2, post_index=0, delta=-2)


def test_measure_rerank_movements_skips_unknown_ids() -> None:
    """`measure_rerank_movements` is reachable only after `assert_hard_filter_preserved`,
    so an unknown id is treated as best-effort skip rather than a crash."""
    movements = measure_rerank_movements(["a", "b"], ["b", "z", "a"])
    listing_ids = {m.listing_id for m in movements}
    assert listing_ids == {"a", "b"}


# --- Bounded rerank --------------------------------------------------------


def test_enforce_bounded_rerank_within_bound_preserves_order() -> None:
    pre = ["a", "b", "c", "d", "e"]
    post = ["c", "a", "d", "b", "e"]
    result = enforce_bounded_rerank(pre, post, max_delta=4)
    assert result == post


def test_enforce_bounded_rerank_clamps_extreme_move() -> None:
    """An id moved by >max_delta is pulled back to `pre_index ± max_delta`."""
    pre = ["a", "b", "c", "d", "e", "f", "g", "h"]
    # "h" moved from index 7 to index 0 — delta = -7, exceeds max_delta=3
    post = ["h", "a", "b", "c", "d", "e", "f", "g"]
    result = enforce_bounded_rerank(pre, post, max_delta=3)

    # h should be clamped to index 4 (7 - 3); others should reflow around it.
    assert "h" in result
    h_position = result.index("h")
    assert h_position >= 4


def test_enforce_bounded_rerank_logs_when_clamping() -> None:
    """Clamped movements must surface in structlog at WARNING for review."""

    pre = ["a", "b", "c", "d", "e", "f", "g", "h"]
    post = ["h", "a", "b", "c", "d", "e", "f", "g"]
    with capture_logs() as logs:
        enforce_bounded_rerank(pre, post, max_delta=3)

    clamped = [entry for entry in logs if entry.get("event") == "rerank_movement_clamped"]
    assert len(clamped) == 1
    assert clamped[0]["log_level"] == "warning"
    assert clamped[0]["listing_id"] == "h"
    assert clamped[0]["delta"] == -7
    assert clamped[0]["clamped_index"] == 4


def test_enforce_bounded_rerank_max_delta_zero_freezes_order() -> None:
    """`max_delta=0` means no movement permitted — output must equal `pre_order`."""
    pre = ["a", "b", "c"]
    post = ["c", "b", "a"]
    result = enforce_bounded_rerank(pre, post, max_delta=0)
    assert result == pre


def test_enforce_bounded_rerank_rejects_negative_bound() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        enforce_bounded_rerank(["a"], ["a"], max_delta=-1)


def test_enforce_bounded_rerank_skips_unknown_post_ids() -> None:
    """Unknown ids are silently skipped (caller already ran the hard-filter check)."""
    result = enforce_bounded_rerank(["a", "b"], ["b", "z", "a"], max_delta=MAX_RERANK_DELTA)
    assert set(result) == {"a", "b"}
