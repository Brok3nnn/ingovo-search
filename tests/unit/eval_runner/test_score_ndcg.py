"""Unit tests for `eval/score_ndcg.py` (§17.2.6 graded NDCG@10 scorer).

Pure math — no DB. The async `main()` orchestrator that pulls
`SyntheticEvalRating` rows from Postgres is integration concern. We exercise
the discount-cumulative-gain formulas directly.

The implementation uses the graded formulation `(2^rel - 1) / log2(i + 2)`
which is standard for ratings in [0, 5] (vs. the simpler `rel / log2(i + 2)`
used for binary relevance). Tests pin that choice so a future refactor that
silently swaps formulas surfaces.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = REPO_ROOT / "eval" / "score_ndcg.py"

_spec = importlib.util.spec_from_file_location("score_ndcg", _MODULE_PATH)
assert _spec is not None, f"could not build spec for {_MODULE_PATH}"
assert _spec.loader is not None, f"spec has no loader for {_MODULE_PATH}"
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

compute_dcg = _module.compute_dcg
compute_ndcg = _module.compute_ndcg


def test_compute_dcg_single_rating_matches_formula() -> None:
    """DCG([5], k=1) = (2^5 - 1) / log2(2) = 31 / 1.0 = 31.0"""
    assert compute_dcg([5], k=1) == 31.0


def test_compute_dcg_two_ratings_matches_formula() -> None:
    """DCG([5, 3], k=2) = (2^5 - 1) / log2(2) + (2^3 - 1) / log2(3)
    = 31.0 + 7/log2(3)"""
    expected = 31.0 + 7.0 / math.log2(3)
    assert compute_dcg([5, 3], k=2) == pytest.approx(expected, rel=1e-9)  # pyright: ignore[reportUnknownMemberType]


def test_compute_dcg_caps_at_k() -> None:
    """DCG([5,5,5], k=2) ignores the third element."""
    assert compute_dcg([5, 5, 5], k=2) == compute_dcg([5, 5], k=2)


def test_compute_dcg_zero_relevance_contributes_nothing() -> None:
    """(2^0 - 1) = 0, so a 0-rated slot adds 0 regardless of position."""
    assert compute_dcg([0, 0, 0], k=3) == 0.0
    assert compute_dcg([5, 0, 5], k=3) == compute_dcg([5, 0, 5], k=3)


def test_compute_dcg_empty_list_returns_zero() -> None:
    assert compute_dcg([], k=10) == 0.0


def test_compute_dcg_k_larger_than_list_uses_full_list() -> None:
    """Slicing in Python tolerates over-long k; DCG([5], k=10) == DCG([5], k=1)."""
    assert compute_dcg([5], k=10) == compute_dcg([5], k=1)


def test_compute_ndcg_ideal_ordering_returns_one() -> None:
    """When relevances are already sorted desc, DCG == IDCG, NDCG = 1.0."""
    assert compute_ndcg([5, 4, 3, 2, 1], k=5) == 1.0


def test_compute_ndcg_reversed_ordering_below_one() -> None:
    """Worst ordering: NDCG < 1.0 but > 0."""
    score = compute_ndcg([1, 2, 3, 4, 5], k=5)
    assert 0.0 < score < 1.0


def test_compute_ndcg_swapped_top_two_below_ideal() -> None:
    """[4, 5] vs ideal [5, 4]: NDCG < 1.0 and exactly computable.

    DCG = 15/log2(2) + 31/log2(3) = 15 + 31/log2(3)
    IDCG = 31/log2(2) + 15/log2(3) = 31 + 15/log2(3)
    """
    dcg = 15.0 + 31.0 / math.log2(3)
    idcg = 31.0 + 15.0 / math.log2(3)
    expected = dcg / idcg
    assert compute_ndcg([4, 5], k=2) == pytest.approx(expected, rel=1e-9)  # pyright: ignore[reportUnknownMemberType]


def test_compute_ndcg_all_zero_relevances_returns_zero() -> None:
    """When IDCG == 0, the guard returns 0.0 (avoids division by zero)."""
    assert compute_ndcg([0, 0, 0], k=3) == 0.0


def test_compute_ndcg_single_zero_rating_returns_zero() -> None:
    """[0] alone: idcg=0 so NDCG=0."""
    assert compute_ndcg([0], k=1) == 0.0


def test_compute_ndcg_single_nonzero_rating_returns_one() -> None:
    """[5] alone: dcg=idcg=31, NDCG=1.0."""
    assert compute_ndcg([5], k=1) == 1.0


def test_compute_ndcg_at_k10_with_partial_window() -> None:
    """Real-world shape: 7 ratings, NDCG@10 still defined (k > len)."""
    rels = [5, 4, 3, 4, 5, 2, 1]
    ndcg = compute_ndcg(rels, k=10)
    assert 0.0 < ndcg <= 1.0


# --- group_ratings_by_recorded_rank: ratings must be scored in shown order ---

group_ratings_by_recorded_rank = _module.group_ratings_by_recorded_rank


class _Row:
    """Duck-typed stand-in for SyntheticEvalRating (query_id/rater_id/rank/rating)."""

    def __init__(self, query_id: str, rater_id: str, rank: int | None, rating: int) -> None:
        self.query_id = query_id
        self.rater_id = rater_id
        self.rank = rank
        self.rating = rating


def test_groups_are_keyed_by_query_and_rater() -> None:
    rows = [
        _Row("q1", "founder", 1, 5),
        _Row("q1", "gemini-judge-v1", 1, 4),
        _Row("q2", "founder", 1, 2),
    ]

    groups = group_ratings_by_recorded_rank(rows)  # type: ignore[arg-type]

    assert set(groups) == {("q1", "founder"), ("q1", "gemini-judge-v1"), ("q2", "founder")}


def test_ratings_ordered_by_recorded_rank_not_insertion_order() -> None:
    rows = [
        _Row("q1", "founder", 3, 1),
        _Row("q1", "founder", 1, 5),
        _Row("q1", "founder", 2, 3),
    ]

    groups = group_ratings_by_recorded_rank(rows)  # type: ignore[arg-type]

    assert groups[("q1", "founder")] == [5, 3, 1]


def test_unranked_rows_sort_after_ranked_rows_in_insertion_order() -> None:
    rows = [
        _Row("q1", "founder", None, 2),
        _Row("q1", "founder", 1, 5),
        _Row("q1", "founder", None, 4),
    ]

    groups = group_ratings_by_recorded_rank(rows)  # type: ignore[arg-type]

    assert groups[("q1", "founder")] == [5, 2, 4]
