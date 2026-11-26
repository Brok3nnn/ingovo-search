"""Unit tests for the rerank candidate-count sweep (injected search_fn)."""

from __future__ import annotations

from ingovo_search.eval.rerank_candidate_sweep import (
    EvalQuery,
    SearchOutcome,
    VariantResult,
    format_comparison_table,
    recommend_candidate_count,
    run_comparison,
)

QUERIES = [EvalQuery(query_id="q1", query="eladó lakás")]
RELEVANCE = {"q1": {"a": 5.0, "b": 4.0, "c": 3.0}}


def _fake_search_fn_factory():
    # count=3 returns the full ideal order; count=2 drops the lowest-rated 'c'.
    rankings = {
        3: ["a", "b", "c"],
        2: ["a", "b"],
    }

    async def search_fn(_query: str, count: int) -> SearchOutcome:
        return SearchOutcome(
            ranked_ids=rankings[count],
            latency_ms=float(count) * 10.0,
            estimated_cost_usd=float(count) * 0.001,
        )

    return search_fn


async def test_run_comparison_aggregates_per_variant() -> None:
    results = await run_comparison(QUERIES, RELEVANCE, _fake_search_fn_factory(), [3, 2], k=10)

    assert [r.candidate_count for r in results] == [3, 2]
    top, smaller = results
    assert top.mean_ndcg_at_k == 1.0  # full ideal order
    assert 0.0 < smaller.mean_ndcg_at_k < 1.0  # dropped a rated listing
    assert top.mean_top1_relevance == 5.0
    assert top.mean_latency_ms == 30.0
    assert smaller.total_estimated_cost_usd == 0.002


async def test_recommend_picks_smallest_within_tolerance() -> None:
    results = await run_comparison(QUERIES, RELEVANCE, _fake_search_fn_factory(), [3, 2], k=10)

    # Strict tolerance → only the top-quality count qualifies.
    assert recommend_candidate_count(results, tolerance=0.001) == 3
    # Loose tolerance → the cheaper count is "good enough".
    assert recommend_candidate_count(results, tolerance=0.5) == 2


def test_recommend_requires_results() -> None:
    try:
        recommend_candidate_count([])
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty results")


def test_format_comparison_table_lists_each_variant() -> None:
    rows = [
        VariantResult(30, 5, 0.86, 4.2, 120.0, 0.05),
        VariantResult(20, 5, 0.859, 4.2, 95.0, 0.034),
    ]
    table = format_comparison_table(rows)
    assert "candidates" in table
    assert "30" in table
    assert "20" in table
