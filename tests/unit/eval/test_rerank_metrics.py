"""Unit tests for the graded nDCG metrics used by the rerank sweep."""

from __future__ import annotations

from ingovo_search.eval.rerank_metrics import compute_ndcg, ndcg_at_k, top1_relevance

REL = {"a": 5.0, "b": 4.0, "c": 3.0}


def test_compute_ndcg_ideal_is_one() -> None:
    assert compute_ndcg([5, 4, 3], k=3) == 1.0


def test_ndcg_at_k_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["a", "b", "c"], REL, k=10) == 1.0


def test_ndcg_at_k_penalises_omitted_relevant_listing() -> None:
    full = ndcg_at_k(["a", "b", "c"], REL, k=10)
    omitted = ndcg_at_k(["a", "b"], REL, k=10)  # drops 'c' (rated 3)
    assert full == 1.0
    assert 0.0 < omitted < 1.0


def test_ndcg_at_k_suboptimal_order_below_one() -> None:
    assert ndcg_at_k(["b", "a", "c"], REL, k=10) < 1.0


def test_ndcg_at_k_unjudged_ids_score_zero() -> None:
    assert ndcg_at_k(["x", "y"], REL, k=10) == 0.0


def test_ndcg_at_k_empty_relevance_is_zero() -> None:
    assert ndcg_at_k(["a", "b"], {}, k=10) == 0.0


def test_top1_relevance_basic() -> None:
    assert top1_relevance(["a", "b"], REL) == 5.0
    assert top1_relevance(["x"], REL) == 0.0
    assert top1_relevance([], REL) == 0.0
