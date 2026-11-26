"""Unit tests for `eval/calibrate_weights.py` (§17.2.6 grid-search calibrator).

Pure-function tests — the §8 `recommend` integration + the SyntheticEvalRating
loader are v1 work. The math here (grid expansion, per-query NDCG mapping,
sort by descending avg) is locked behind these tests so the v1 wiring slots
in cleanly.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = REPO_ROOT / "eval" / "calibrate_weights.py"

# Dataclass's `KW_ONLY` detection reads `sys.modules[cls.__module__]`, which is
# None for importlib-loaded modules unless we register them first.
_MODULE_NAME = "_test_calibrate_weights_under_test"
_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
assert _spec is not None, f"could not build spec for {_MODULE_PATH}"
assert _spec.loader is not None, f"spec has no loader for {_MODULE_PATH}"
_module = importlib.util.module_from_spec(_spec)
sys.modules[_MODULE_NAME] = _module
_spec.loader.exec_module(_module)

CalibrationResult = _module.CalibrationResult
calibrate = _module.calibrate
expand_grid = _module.expand_grid
score_query = _module.score_query


# --- expand_grid ----------------------------------------------------------


def test_expand_grid_empty_input_returns_empty() -> None:
    assert expand_grid([]) == []


def test_expand_grid_single_entry_single_field() -> None:
    grid = [{"content": [0.2, 0.25]}]
    combos = expand_grid(grid)

    assert combos == [{"content": 0.2}, {"content": 0.25}]


def test_expand_grid_cartesian_product_within_entry() -> None:
    grid = [{"content": [0.2, 0.25], "quality": [0.1, 0.15]}]
    combos = expand_grid(grid)

    assert len(combos) == 4
    assert {"content": 0.2, "quality": 0.1} in combos
    assert {"content": 0.2, "quality": 0.15} in combos
    assert {"content": 0.25, "quality": 0.1} in combos
    assert {"content": 0.25, "quality": 0.15} in combos


def test_expand_grid_concatenates_multiple_entries() -> None:
    """Multiple grid entries are NOT cross-producted — useful for sweeping disjoint regions."""
    grid = [{"content": [0.2, 0.25]}, {"quality": [0.1, 0.15]}]
    combos = expand_grid(grid)

    assert combos == [
        {"content": 0.2},
        {"content": 0.25},
        {"quality": 0.1},
        {"quality": 0.15},
    ]


def test_expand_grid_empty_dict_in_list_yields_empty_weights() -> None:
    """Bare `{}` means "no overrides" — useful as a baseline in the grid."""
    grid = [{"content": [0.2]}, {}]
    combos = expand_grid(grid)

    assert combos == [{"content": 0.2}, {}]


# --- score_query ----------------------------------------------------------


def test_score_query_perfect_ranking_returns_one() -> None:
    """Ground truth has highest-rated listing on top; matches ideal order."""
    ratings = {"L1": 5, "L2": 4, "L3": 3}
    ranked = ["L1", "L2", "L3"]

    assert score_query(ranked, ratings, k=3) == 1.0


def test_score_query_reverse_ranking_below_one() -> None:
    ratings = {"L1": 5, "L2": 4, "L3": 3}
    ranked = ["L3", "L2", "L1"]

    score = score_query(ranked, ratings, k=3)
    assert 0.0 < score < 1.0


def test_score_query_unrated_listing_contributes_zero_relevance() -> None:
    """A ranked listing missing from ground truth is treated as relevance 0."""
    ratings = {"L1": 5, "L2": 4}
    ranked = ["L1", "L99", "L2"]  # L99 is unrated

    # Compare to the same ranking but explicitly with rating 0 for L99 — they should match.
    ratings_with_zero = {"L1": 5, "L99": 0, "L2": 4}
    assert score_query(ranked, ratings, k=3) == score_query(ranked, ratings_with_zero, k=3)


def test_score_query_empty_ranking_returns_zero() -> None:
    ratings = {"L1": 5}
    assert score_query([], ratings, k=10) == 0.0


def test_score_query_truncates_at_k() -> None:
    """Only the first k of `ranked` count toward NDCG@k."""
    ratings = {"L1": 5, "L2": 4, "L3": 3}
    ranked_long = ["L1", "L2", "L3", "L99", "L100"]

    assert score_query(ranked_long, ratings, k=3) == score_query(ranked_long[:3], ratings, k=3)


# --- calibrate ------------------------------------------------------------


def _make_score_fn(per_weights_orderings: dict[tuple[tuple[str, float], ...], dict[str, list[str]]]):
    """Build a deterministic ScoreFn from a `{weights_key: {qid: [listings]}}` table.

    `weights_key` is the sorted tuple form of the weights dict so it's hashable.
    """

    def fn(qid: str, weights: dict[str, float]) -> Sequence[str]:
        key = tuple(sorted(weights.items()))
        if key not in per_weights_orderings:
            raise KeyError(f"no canned ordering for weights={weights}")
        return per_weights_orderings[key].get(qid, [])

    return fn


def test_calibrate_empty_eval_set_returns_empty() -> None:
    score_fn = _make_score_fn({})
    assert calibrate({}, score_fn, [{"content": 0.2}], k=10) == []


def test_calibrate_single_weights_single_query() -> None:
    weights = {"content": 0.2}
    key = tuple(sorted(weights.items()))
    score_fn = _make_score_fn({key: {"q1": ["L1", "L2", "L3"]}})

    eval_set: dict[str, dict[str, int]] = {"q1": {"L1": 5, "L2": 4, "L3": 3}}
    results = calibrate(eval_set, score_fn, [weights], k=3)

    assert len(results) == 1
    r = results[0]
    assert r.weights == weights
    assert r.avg_ndcg == 1.0
    assert r.per_query_ndcg == {"q1": 1.0}


def test_calibrate_sorts_results_by_avg_ndcg_desc() -> None:
    """Two weight points, one better than the other — result list is sorted desc."""
    weights_good = {"content": 0.2}
    weights_bad = {"content": 0.1}
    key_good = tuple(sorted(weights_good.items()))
    key_bad = tuple(sorted(weights_bad.items()))

    score_fn = _make_score_fn(
        {
            key_good: {"q1": ["L1", "L2", "L3"]},  # ideal order
            key_bad: {"q1": ["L3", "L2", "L1"]},  # reverse
        }
    )
    eval_set: dict[str, dict[str, int]] = {"q1": {"L1": 5, "L2": 4, "L3": 3}}

    results = calibrate(eval_set, score_fn, [weights_bad, weights_good], k=3)

    assert results[0].weights == weights_good
    assert results[1].weights == weights_bad
    assert results[0].avg_ndcg > results[1].avg_ndcg


def test_calibrate_avg_ndcg_is_mean_across_queries() -> None:
    """Two queries with different orderings under the same weights."""
    weights = {"content": 0.2}
    key = tuple(sorted(weights.items()))
    score_fn = _make_score_fn(
        {
            key: {
                "q1": ["L1", "L2"],  # ideal for q1
                "q2": ["L9", "L7"],  # reverse for q2
            }
        }
    )
    eval_set: dict[str, dict[str, int]] = {
        "q1": {"L1": 5, "L2": 4},
        "q2": {"L7": 5, "L9": 3},
    }

    results = calibrate(eval_set, score_fn, [weights], k=2)
    r = results[0]

    assert r.avg_ndcg == (r.per_query_ndcg["q1"] + r.per_query_ndcg["q2"]) / 2
    assert r.per_query_ndcg["q1"] == 1.0  # ideal
    assert 0.0 < r.per_query_ndcg["q2"] < 1.0  # reverse


def test_calibrate_result_is_frozen_dataclass() -> None:
    """`CalibrationResult` is frozen — no accidental mutation downstream."""
    weights = {"content": 0.2}
    key = tuple(sorted(weights.items()))
    score_fn = _make_score_fn({key: {"q1": ["L1"]}})
    results = calibrate({"q1": {"L1": 5}}, score_fn, [weights], k=1)

    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        results[0].avg_ndcg = 0.0  # type: ignore[misc]


def test_calibrate_handles_nan_score_fn_output() -> None:
    """If a buggy score_fn produces NaN (somehow), avg is forced to 0.0 so sort terminates.

    We trigger NaN via an empty-rating ground truth (idcg=0 → ndcg=0, which is *not*
    nan, so the NaN guard is exercised by explicitly stubbing). Use a custom score_fn
    that returns an empty ordering for a query whose ratings are also empty.
    """
    weights = {"content": 0.2}
    eval_set: dict[str, dict[str, int]] = {"q1": {}}  # no ratings → ndcg returns 0
    score_fn = _make_score_fn({tuple(sorted(weights.items())): {"q1": []}})

    results = calibrate(eval_set, score_fn, [weights], k=10)

    assert len(results) == 1
    assert results[0].avg_ndcg == 0.0
    assert not math.isnan(results[0].avg_ndcg)


def test_calibrate_query_weights_shift_the_average() -> None:
    """Traffic weighting: a heavily-weighted query dominates the objective.

    q_long ranks perfectly (NDCG 1.0), q_short ranks a zero-relevance listing
    first (NDCG 0.0). Plain mean = 0.5; weighting long 0.9 / short 0.1 pulls
    the average to 0.9 — the expected-traffic prior, not the eval
    set's category counts, decides each query's influence.
    """
    weights = {"content": 0.2}
    key = tuple(sorted(weights.items()))
    score_fn = _make_score_fn({key: {"q_long": ["L1"], "q_short": ["L2"]}})
    eval_set = {"q_long": {"L1": 5}, "q_short": {"L2": 0, "L3": 5}}

    plain = calibrate(eval_set, score_fn, [weights], k=1)
    weighted = calibrate(eval_set, score_fn, [weights], k=1, query_weights={"q_long": 0.9, "q_short": 0.1})

    assert plain[0].avg_ndcg == 0.5
    assert weighted[0].avg_ndcg == 0.9


def test_calibrate_query_weights_default_to_one_for_missing_queries() -> None:
    """Queries absent from the weight map contribute with weight 1.0."""
    weights = {"content": 0.2}
    key = tuple(sorted(weights.items()))
    score_fn = _make_score_fn({key: {"q1": ["L1"], "q2": ["L2"]}})
    eval_set = {"q1": {"L1": 5}, "q2": {"L2": 0, "L3": 5}}

    result = calibrate(eval_set, score_fn, [weights], k=1, query_weights={"q1": 1.0})

    assert result[0].avg_ndcg == 0.5
