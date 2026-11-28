"""Rerank candidate-count sweep — compare top 30 / 20 / 15 offline.

Find the smallest number of
candidates sent to the Gemini reranker that preserves ranking quality, to cut
latency and token cost.

The orchestration here is pure aside from an injected `search_fn`, so the
metric aggregation and the "pick the smallest count that preserves quality"
decision are unit-tested with a fake search function. The live runner
(`rerank_candidate_count.py`) supplies a `search_fn` backed by the real pipeline
+ DB + Gemini and the ratings.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from ingovo_search.eval.rerank_metrics import ndcg_at_k, top1_relevance


@dataclass(frozen=True)
class EvalQuery:
    """One rated evaluation query."""

    query_id: str
    query: str


@dataclass(frozen=True)
class SearchOutcome:
    """What one search returned, for one candidate-count variant."""

    ranked_ids: list[str]
    latency_ms: float
    estimated_cost_usd: float = 0.0


# (query_text, rerank_candidate_count) -> SearchOutcome
SearchFn = Callable[[str, int], Awaitable[SearchOutcome]]


@dataclass(frozen=True)
class VariantResult:
    """Aggregate metrics for one candidate-count variant across all queries."""

    candidate_count: int
    query_count: int
    mean_ndcg_at_k: float
    mean_top1_relevance: float
    mean_latency_ms: float
    total_estimated_cost_usd: float


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


async def run_comparison(
    eval_queries: Sequence[EvalQuery],
    relevance_by_query: Mapping[str, Mapping[str, float]],
    search_fn: SearchFn,
    candidate_counts: Sequence[int],
    *,
    k: int = 10,
) -> list[VariantResult]:
    """Run each candidate-count variant over every query and aggregate metrics.

    `relevance_by_query[query_id]` maps property_id -> graded relevance (1-5).
    Results preserve the order of `candidate_counts`.
    """
    results: list[VariantResult] = []
    for count in candidate_counts:
        ndcgs: list[float] = []
        top1s: list[float] = []
        latencies: list[float] = []
        total_cost = 0.0
        for item in eval_queries:
            outcome = await search_fn(item.query, count)
            relevance = relevance_by_query.get(item.query_id, {})
            ndcgs.append(ndcg_at_k(outcome.ranked_ids, relevance, k))
            top1s.append(top1_relevance(outcome.ranked_ids, relevance))
            latencies.append(outcome.latency_ms)
            total_cost += outcome.estimated_cost_usd
        results.append(
            VariantResult(
                candidate_count=count,
                query_count=len(eval_queries),
                mean_ndcg_at_k=_mean(ndcgs),
                mean_top1_relevance=_mean(top1s),
                mean_latency_ms=_mean(latencies),
                total_estimated_cost_usd=total_cost,
            )
        )
    return results


def recommend_candidate_count(results: Sequence[VariantResult], *, tolerance: float = 0.005) -> int:
    """Smallest candidate count whose mean nDCG is within `tolerance` of the best.

    The decision rule: pick the smallest candidate set that still
    preserves quality. Ties on quality favour the cheaper (smaller) count.
    """
    if not results:
        raise ValueError("recommend_candidate_count requires at least one variant result")
    best_ndcg = max(r.mean_ndcg_at_k for r in results)
    eligible = [r.candidate_count for r in results if r.mean_ndcg_at_k >= best_ndcg - tolerance]
    return min(eligible)


def format_comparison_table(results: Sequence[VariantResult]) -> str:
    """Render the variant results as a fixed-width text table for the CLI."""
    header = f"{'candidates':>10} {'nDCG@k':>8} {'top1':>6} {'latency_ms':>11} {'cost_usd':>9}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.candidate_count:>10} {r.mean_ndcg_at_k:>8.4f} {r.mean_top1_relevance:>6.2f} "
            f"{r.mean_latency_ms:>11.1f} {r.total_estimated_cost_usd:>9.4f}"
        )
    return "\n".join(lines)
