"""Offline rerank comparison — deterministic vs Gemini vs Jev variants.

Pure orchestration and aggregation over injected per-arm rerank functions, so
the metrics, the adversarial check, the gate evaluation and the report are
unit-tested with fakes. The live runner (`eval/jev_rerank_offline.py` at the
repo root) supplies arms backed by the real pipeline, the database, Gemini and
Jev, and writes the report the G1/G2 gates read.

An *arm* is one way of ordering a query's rerank head: the deterministic order,
today's Gemini rerank, or one Jev variant at one alpha. Every arm sees the same
candidates, so the differences are the reranker's alone.

The adversarial set lives in `jev_rerank_adversarial`, the G1/G2 gates in
`jev_rerank_gates` and the Markdown report in `jev_rerank_report`; this module
holds the arms' data model, the per-query metrics, the per-arm summaries, the
run itself and the statistics over it.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from ingovo_search.eval.rerank_metrics import compute_dcg, ndcg_at_k, top1_relevance
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.rerank_prompt import RerankCandidate

DETERMINISTIC_ARM = "deterministic"
GEMINI_ARM = "gemini"
JEV_ARM_PREFIX = "jev:"


@dataclass(frozen=True)
class OfflineQuery:
    """One rated query with the rerank head the pipeline would send."""

    query_id: str
    query_text: str
    parsed: ParsedQuery
    # Deterministic order, i.e. the rerank head as `api/search.py` builds it.
    candidates: tuple[RerankCandidate, ...]
    # listing id -> `ScoreBreakdown.total`, for the alpha blend.
    deterministic_scores: Mapping[str, float]
    # What production's conditional-rerank gate would decide for this query.
    should_rerank: bool = True
    category: str | None = None

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.listing_id for candidate in self.candidates)


@dataclass(frozen=True)
class ArmRun:
    """One arm's result on one query."""

    arm: str
    query_id: str
    # What a user would see: the reranker's order after the bounded-rerank guardrail.
    ranked_ids: tuple[str, ...]
    # The reranker's own order, before any guardrail (the adversarial check reads this).
    raw_ranked_ids: tuple[str, ...]
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    # success | fallback | fast_path | deterministic
    outcome: str = "success"
    from_cache: bool = False


RerankArm = Callable[[OfflineQuery], Awaitable[ArmRun]]


@dataclass(frozen=True)
class QueryMetrics:
    arm: str
    query_id: str
    ndcg_at_k: float
    top1_relevance: float
    candidate_ndcg: float
    rated_candidates: int


@dataclass(frozen=True)
class ArmSummary:
    arm: str
    query_count: int
    mean_ndcg_at_k: float
    mean_top1_relevance: float
    mean_candidate_ndcg: float
    latency_p50_ms: float
    latency_p95_ms: float
    input_tokens_p50: int
    mean_cost_usd: float
    fallback_count: int
    cached_count: int


@dataclass(frozen=True)
class OfflineReport:
    k: int
    runs: tuple[ArmRun, ...]
    metrics: tuple[QueryMetrics, ...]
    summaries: tuple[ArmSummary, ...]


def candidate_ndcg(
    ranked_ids: Sequence[str], candidate_ids: Iterable[str], relevance: Mapping[str, float], k: int = 10
) -> float:
    """nDCG@k with the ideal taken over the candidates only.

    `ndcg_at_k` (the G1 metric) takes the ideal over every judged listing
    of the query, so a rerank that can only reorder its 12-item head never
    reaches 1.0. This variant asks what a perfect reorder of that head would
    score, which makes rerankers easier to tell apart. 0.0 when no candidate has
    a positive grade.
    """
    gains = [float(relevance.get(listing_id, 0.0)) for listing_id in ranked_ids]
    ideal = sorted((float(relevance.get(listing_id, 0.0)) for listing_id in candidate_ids), reverse=True)
    idcg = compute_dcg(ideal, k)
    return compute_dcg(gains, k) / idcg if idcg > 0 else 0.0


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile (`fraction` in 0..1); 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def query_metrics(run: ArmRun, query: OfflineQuery, relevance: Mapping[str, float], *, k: int = 10) -> QueryMetrics:
    return QueryMetrics(
        arm=run.arm,
        query_id=run.query_id,
        ndcg_at_k=ndcg_at_k(run.ranked_ids, relevance, k),
        top1_relevance=top1_relevance(run.ranked_ids, relevance),
        candidate_ndcg=candidate_ndcg(run.ranked_ids, query.candidate_ids, relevance, k),
        rated_candidates=sum(1 for listing_id in query.candidate_ids if listing_id in relevance),
    )


def summarise_arm(arm: str, runs: Sequence[ArmRun], metrics: Sequence[QueryMetrics]) -> ArmSummary:
    """Aggregate one arm over its queries; latency and tokens come from real calls only."""
    live = [run for run in runs if run.outcome not in ("deterministic", "fast_path")]
    latencies = [run.latency_ms for run in live if not run.from_cache or run.latency_ms > 0]
    tokens = [float(run.input_tokens) for run in live]
    return ArmSummary(
        arm=arm,
        query_count=len(runs),
        mean_ndcg_at_k=_mean([m.ndcg_at_k for m in metrics]),
        mean_top1_relevance=_mean([m.top1_relevance for m in metrics]),
        mean_candidate_ndcg=_mean([m.candidate_ndcg for m in metrics]),
        latency_p50_ms=statistics.median(latencies) if latencies else 0.0,
        latency_p95_ms=percentile(latencies, 0.95),
        input_tokens_p50=int(statistics.median(tokens)) if tokens else 0,
        mean_cost_usd=_mean([run.cost_usd for run in runs]),
        fallback_count=sum(1 for run in runs if run.outcome == "fallback"),
        cached_count=sum(1 for run in runs if run.from_cache),
    )


ProgressFn = Callable[[OfflineQuery, ArmRun], None]


async def run_offline(
    queries: Sequence[OfflineQuery],
    relevance_by_query: Mapping[str, Mapping[str, float]],
    arms: Mapping[str, RerankArm],
    *,
    k: int = 10,
    on_progress: ProgressFn | None = None,
) -> OfflineReport:
    """Run every arm on every query and aggregate per arm.

    Query-major, so a query's Jev variants run back to back (friendlier to the
    live runner's response cache and progress output). Summaries keep the order
    of `arms`.
    """
    runs: list[ArmRun] = []
    metrics: list[QueryMetrics] = []
    for query in queries:
        relevance = relevance_by_query.get(query.query_id, {})
        for arm_name, arm in arms.items():
            run = await arm(query)
            if run.arm != arm_name or run.query_id != query.query_id:
                raise ValueError(f"arm {arm_name!r} returned a run for {run.arm!r}/{run.query_id!r}")
            runs.append(run)
            metrics.append(query_metrics(run, query, relevance, k=k))
            if on_progress is not None:
                on_progress(query, run)
    summaries = [
        summarise_arm(
            arm_name,
            [run for run in runs if run.arm == arm_name],
            [m for m in metrics if m.arm == arm_name],
        )
        for arm_name in arms
    ]
    return OfflineReport(k=k, runs=tuple(runs), metrics=tuple(metrics), summaries=tuple(summaries))


def paired_bootstrap_diff(
    first: Sequence[float],
    second: Sequence[float],
    *,
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean of `first - second` over paired items, with a percentile bootstrap interval.

    Resamples the per-query differences with replacement; a fixed seed keeps
    the report reproducible. Returns (mean, low, high).
    """
    if len(first) != len(second):
        raise ValueError("paired samples must have the same length")
    if not first:
        return 0.0, 0.0, 0.0
    differences = [a - b for a, b in zip(first, second, strict=True)]
    count = len(differences)
    rng = random.Random(seed)
    means = sorted(sum(differences[rng.randrange(count)] for _ in range(count)) / count for _ in range(iterations))
    tail = (1.0 - confidence) / 2
    low = means[int(tail * iterations)]
    high = means[min(iterations - 1, int((1.0 - tail) * iterations))]
    return sum(differences) / count, low, high


@dataclass(frozen=True)
class Significance:
    label: str
    metric: str
    mean: float
    low: float
    high: float


def arm_significance(
    report: OfflineReport, first_arm: str, second_arm: str, *, metric: str = "ndcg_at_k", seed: int = 0
) -> Significance:
    """Paired bootstrap of one per-query metric between two arms over their shared queries."""
    first = {m.query_id: m for m in report.metrics if m.arm == first_arm}
    second = {m.query_id: m for m in report.metrics if m.arm == second_arm}
    shared = sorted(first.keys() & second.keys())
    mean, low, high = paired_bootstrap_diff(
        [float(getattr(first[q], metric)) for q in shared],
        [float(getattr(second[q], metric)) for q in shared],
        seed=seed,
    )
    return Significance(f"{first_arm} vs {second_arm}", metric, mean, low, high)


def best_jev_arm(summaries: Sequence[ArmSummary], *, prefix: str = JEV_ARM_PREFIX) -> ArmSummary | None:
    """The Jev arm with the highest nDCG@10; fewer input tokens break ties."""
    jev = [summary for summary in summaries if summary.arm.startswith(prefix)]
    if not jev:
        return None
    return max(jev, key=lambda summary: (round(summary.mean_ndcg_at_k, 6), -summary.input_tokens_p50))
