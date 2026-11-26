"""§17.2.6 weight-calibration grid search.

Pure-function calibrator: given a `(query_id, ground_truth_ratings)` eval set
and a callable `score_fn(query_id, weights) -> ranked_listing_ids`, sweep a
weight grid and return the `(weights, avg_ndcg)` log sorted by NDCG desc.

The spec asks for grid search over the §6.4 `w_*` weights against
`SyntheticEvalRating` ratings. The DB loader + the §8 `compute_score` plug
live behind v1 wiring — this module ships the pure search loop so the math
is locked behind a test suite before the heavier pipeline lands.

Usage (pure):

    eval_set = {
        "q1": {"L1": 5, "L2": 4, "L3": 2},
        "q2": {"L7": 5, "L9": 3},
    }
    grid = [{"content": 0.20}, {"content": 0.25}, ...]

    def my_score_fn(qid, weights):
        return run_§8_recommend(qid, weights)  # -> ["L1", "L2", ...]

    results = calibrate(eval_set, my_score_fn, grid, k=10)
    # results[0] is the best (weights, avg_ndcg)

CLI is a stub — the SyntheticEvalRating loader + the §8 recommend plug are
v1 work, and the printed pointer makes that explicit.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Reuse the NDCG math from score_ndcg.py rather than duplicating — both live
# in the same `eval/` package at repo root. Register in sys.modules so any
# future dataclass in score_ndcg.py doesn't trip dataclass's `KW_ONLY`
# detection (which reads sys.modules[cls.__module__]).
_NDCG_MODULE_NAME = "_score_ndcg_for_calibrate"
_NDCG_MODULE_PATH = Path(__file__).resolve().parent / "score_ndcg.py"
_ndcg_spec = importlib.util.spec_from_file_location(_NDCG_MODULE_NAME, _NDCG_MODULE_PATH)
assert _ndcg_spec is not None
assert _ndcg_spec.loader is not None
_ndcg_module = importlib.util.module_from_spec(_ndcg_spec)
sys.modules[_NDCG_MODULE_NAME] = _ndcg_module
_ndcg_spec.loader.exec_module(_ndcg_module)

compute_ndcg = _ndcg_module.compute_ndcg

# Type aliases for clarity in the public signatures.
QueryId = str
ListingId = str
Weights = dict[str, float]
GroundTruth = dict[QueryId, dict[ListingId, int]]
"""`{query_id: {listing_id: rating}}` — ratings are 1..5; missing listings count as relevance 0."""
ScoreFn = Callable[[QueryId, Weights], Sequence[ListingId]]
"""Pluggable scoring function — return the ranked listing ids for a (query, weights) pair."""


@dataclass(frozen=True)
class CalibrationResult:
    weights: Weights
    avg_ndcg: float
    per_query_ndcg: dict[QueryId, float]


def expand_grid(weight_grid: Sequence[dict[str, Sequence[float]]]) -> list[Weights]:
    """Cartesian-product expansion of a grid spec.

    Each grid entry maps weight name → candidate values. Returns one dict per
    combination across all entries' fields.

    Example:
        expand_grid([{"content": [0.2, 0.25], "quality": [0.1, 0.15]}])
        → [{"content": 0.2, "quality": 0.1},
           {"content": 0.2, "quality": 0.15},
           {"content": 0.25, "quality": 0.1},
           {"content": 0.25, "quality": 0.15}]

    Multiple grid entries are concatenated (NOT cross-producted) — useful when
    you want to sweep two independent regions of the weight space.
    """
    combos: list[Weights] = []
    for entry in weight_grid:
        if not entry:
            combos.append({})
            continue
        keys = list(entry.keys())
        # Build the product via repeated unfolding so the implementation has no scipy / sklearn dep.
        partial: list[Weights] = [{}]
        for k in keys:
            partial = [{**prev, k: v} for prev in partial for v in entry[k]]
        combos.extend(partial)
    return combos


def score_query(
    ranked_listing_ids: Sequence[ListingId],
    ratings: dict[ListingId, int],
    *,
    k: int = 10,
) -> float:
    """NDCG@k for one query — maps the ranking onto its ground-truth relevances."""
    relevances = [ratings.get(lid, 0) for lid in ranked_listing_ids[:k]]
    return compute_ndcg(relevances, k=k)


def calibrate(
    eval_set: GroundTruth,
    score_fn: ScoreFn,
    weight_grid: list[Weights],
    *,
    k: int = 10,
    query_weights: dict[QueryId, float] | None = None,
) -> list[CalibrationResult]:
    """Sweep `weight_grid`, return `(weights, avg_ndcg, per_query_ndcg)` sorted by NDCG desc.

    `score_fn` is the integration point: the caller plugs in the §8 `recommend`
    path (or a stub during calibration of pure-math invariants). Each grid
    point gets one call per query.

    `query_weights` makes the objective traffic-weighted: each query's NDCG
    contributes proportionally to its weight (the eval set's category counts
    are a coverage choice, not a traffic model — an AI-first search expects
    long-query-dominant traffic regardless of how many short queries the set
    keeps for coverage). Missing queries default to weight 1.0; None keeps
    the plain mean.

    Returns an empty list when `eval_set` is empty (no queries to score).
    """
    if not eval_set:
        return []

    results: list[CalibrationResult] = []
    for weights in weight_grid:
        per_query: dict[QueryId, float] = {}
        for qid, ratings in eval_set.items():
            ranked = score_fn(qid, weights)
            per_query[qid] = score_query(ranked, ratings, k=k)
        if query_weights is not None:
            total_w = sum(query_weights.get(qid, 1.0) for qid in per_query)
            avg = (
                sum(ndcg * query_weights.get(qid, 1.0) for qid, ndcg in per_query.items()) / total_w
                if total_w > 0
                else 0.0
            )
        else:
            avg = sum(per_query.values()) / len(per_query) if per_query else 0.0
        # Guard against NaN bleeding in via a buggy score_fn — surface as 0 in the log
        # so the sort still terminates.
        if math.isnan(avg):
            avg = 0.0
        results.append(CalibrationResult(weights=weights, avg_ndcg=avg, per_query_ndcg=per_query))

    results.sort(key=lambda r: r.avg_ndcg, reverse=True)
    return results


def main() -> None:
    """This module is the pure math; the DB-backed runner lives in run_calibration.py."""
    print(
        "calibrate_weights.py holds the pure grid-search math (calibrate / expand_grid /\n"
        "score_query). The DB-backed CLI that loads SyntheticEvalRating rows and plugs\n"
        "the §8 `recommend` path is `eval/run_calibration.py` — run that instead:\n\n"
        "    uv run python eval/run_calibration.py\n\n"
        "Pure math is exercised by tests/unit/eval_runner/test_calibrate_weights.py.",
        file=sys.stderr,
    )
    sys.exit(0)


_CALIBRATE_PUBLIC: list[Any] = [calibrate, expand_grid, score_query, CalibrationResult]


if __name__ == "__main__":
    main()
