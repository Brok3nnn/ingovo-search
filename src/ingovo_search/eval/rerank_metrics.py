"""Ranking-quality metrics — graded nDCG@k and top-1 relevance.

Canonical home for the discounted-cumulative-gain math (the top-level
`eval/score_ndcg.py` runner re-uses `compute_dcg` / `compute_ndcg` from here).
Graded formulation `(2^rel - 1) / log2(i + 2)`, standard for ratings in [0, 5].

Two nDCG flavours:
- `compute_ndcg(relevances, k)` — ideal taken over the *returned* grades, for
  scoring a single already-ranked list (the runner).
- `ndcg_at_k(ranked_ids, relevance, k)` — ideal taken over *all judged* items
  for the query, so a ranking that omits a highly-rated listing is penalised.
  This is the right metric for the rerank candidate-count sweep, where a smaller
  candidate set can drop relevant listings entirely.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def compute_dcg(relevances: Sequence[float], k: int) -> float:
    """Discounted cumulative gain at `k`, graded `(2^rel - 1) / log2(i + 2)`."""
    return sum((2.0**rel - 1.0) / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def compute_ndcg(relevances: Sequence[float], k: int) -> float:
    """nDCG@k of an already-ranked grade list; ideal over the same grades.

    Returns 0.0 when the ideal DCG is 0 (no positive grade).
    """
    dcg = compute_dcg(relevances, k)
    idcg = compute_dcg(sorted(relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int = 10) -> float:
    """nDCG@k of `ranked_ids` with the ideal over *all judged* items.

    Gains are the graded relevance of each ranked id (missing ids score 0); the
    ideal DCG ranks every judged item, so omitting a highly-rated listing lowers
    the score. Returns 0.0 when no judged item has positive relevance.
    """
    ranked_gains = [float(relevance.get(pid, 0.0)) for pid in ranked_ids]
    dcg = compute_dcg(ranked_gains, k)
    idcg = compute_dcg(sorted((float(v) for v in relevance.values()), reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def top1_relevance(ranked_ids: Sequence[str], relevance: Mapping[str, float]) -> float:
    """Graded relevance of the top-ranked id (0.0 for an empty ranking)."""
    if not ranked_ids:
        return 0.0
    return float(relevance.get(ranked_ids[0], 0.0))
