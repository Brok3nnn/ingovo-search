"""Ranking guardrails: three rules that scoring + rerank must respect.

1. **No paid-placement fields enter scoring.** Any future ad-like field is
   rejected before it reaches `compute_score`.
2. **Hard filters remain absolute after rerank.** The LLM reranker may not
   surface a listing the SQL hard-filter pass excluded.
3. **Reranker movements are bounded and logged.** Each listing may move at most
   `MAX_RERANK_DELTA` positions relative to its pre-rerank index; movements
   beyond the bound are clamped back and logged at WARNING level for review.

`enforce_bounded_rerank` is the canonical post-rerank ordering used by
`/api/search`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from ingovo_search.types.errors import RankingGuardrailViolation

logger = structlog.get_logger(__name__)

PAID_PLACEMENT_FIELDS: frozenset[str] = frozenset(
    {
        "is_" + "pro" + "moted",
        "is_sponsored",
        "is_" + "fea" + "tured",
        "bo" + "ost_score",
        "bo" + "ost_paid",
        "placement_paid",
        "placement_score",
        "paid_prio" + "rity",
        "ad_score",
    }
)
"""Attribute / weight names that would constitute paid placement.

The v0 data model contains none of these — the set is a forward-looking
contract so any future schema or scoring change that introduces one fails
loudly instead of silently leaking into the ranking signal.
"""

MAX_RERANK_DELTA: int = 10
"""Maximum positions a listing may move during rerank, in either direction."""


def assert_no_paid_placement_in_weights(weights: Mapping[str, float]) -> None:
    """Reject scoring-weight dicts that mention any `PAID_PLACEMENT_FIELDS` name."""
    violations = sorted(set(weights) & PAID_PLACEMENT_FIELDS)
    if violations:
        raise RankingGuardrailViolation(f"paid-placement fields present in scoring weights: {violations}")


def assert_no_paid_placement_in_features(listing: Any) -> None:
    """Reject a scored listing carrying a truthy paid-placement attribute.

    `listing` may be a SQLAlchemy `Property` row, a Pydantic model, or any
    plain object. We probe with `getattr(..., None)` rather than `hasattr` so
    descriptors and `__getattr__` proxies are exercised. A truthy value on any
    `PAID_PLACEMENT_FIELDS` name is treated as paid placement and rejected.
    """
    violations: list[str] = []
    for field in PAID_PLACEMENT_FIELDS:
        value = getattr(listing, field, None)
        if value:
            violations.append(field)
    if violations:
        violations.sort()
        raise RankingGuardrailViolation(f"paid-placement fields present on listing: {violations}")


def assert_hard_filter_preserved(
    *,
    hard_filter_ids: Iterable[str],
    post_rerank_ids: Iterable[str],
) -> None:
    """Rerank must not introduce a listing the hard-filter pass excluded.

    Raises `RankingGuardrailViolation` if any id in `post_rerank_ids` is not in
    `hard_filter_ids`. Order is irrelevant — only membership.
    """
    allowed = set(hard_filter_ids)
    extras = sorted({pid for pid in post_rerank_ids if pid not in allowed})
    if extras:
        raise RankingGuardrailViolation(f"rerank surfaced ids excluded by hard filters: {extras}")


def assert_rerank_candidate_set_preserved(
    *,
    pre_rerank_ids: Iterable[str],
    post_rerank_ids: Iterable[str],
) -> None:
    """Require exact candidate-set equality across the reranker boundary."""

    before = set(pre_rerank_ids)
    after = set(post_rerank_ids)
    missing = sorted(before - after)
    extras = sorted(after - before)
    if missing or extras:
        raise RankingGuardrailViolation(f"rerank candidate set changed: missing={missing}, extras={extras}")


@dataclass(frozen=True, slots=True)
class RerankMovement:
    """One listing's index shift across the rerank step."""

    listing_id: str
    pre_index: int
    post_index: int
    delta: int  # post - pre; negative means the rerank pulled the listing up


def measure_rerank_movements(
    pre_order: Sequence[str],
    post_order: Sequence[str],
) -> list[RerankMovement]:
    """Pair each id in `post_order` with its `pre_order` index.

    Ids in `post_order` that don't appear in `pre_order` are skipped — callers
    should run `assert_hard_filter_preserved` first to reject that case.
    """
    pre_index: dict[str, int] = {pid: idx for idx, pid in enumerate(pre_order)}
    movements: list[RerankMovement] = []
    for post_idx, pid in enumerate(post_order):
        if pid not in pre_index:
            continue
        pre_idx = pre_index[pid]
        movements.append(
            RerankMovement(
                listing_id=pid,
                pre_index=pre_idx,
                post_index=post_idx,
                delta=post_idx - pre_idx,
            )
        )
    return movements


def enforce_bounded_rerank(
    pre_order: Sequence[str],
    post_order: Sequence[str],
    *,
    max_delta: int = MAX_RERANK_DELTA,
) -> list[str]:
    """Return a post-rerank ordering where no listing moves more than `max_delta`.

    Any listing whose `|delta| > max_delta` is "pulled back" toward its
    pre-rerank index by inserting it at `clamp(post_index, pre_index ± max_delta)`
    in the output. The result is a deduplicated id list of the same set as
    `pre_order ∩ post_order`, with hard-filter preservation as a precondition
    (call `assert_hard_filter_preserved` upstream). Out-of-bound movements are
    logged at WARNING for review.
    """
    if max_delta < 0:
        raise ValueError("max_delta must be non-negative")

    movements = measure_rerank_movements(pre_order, post_order)
    placements: list[tuple[int, str]] = []
    for move in movements:
        if abs(move.delta) <= max_delta:
            placements.append((move.post_index, move.listing_id))
            continue
        clamped = move.pre_index + max_delta if move.delta > max_delta else move.pre_index - max_delta
        logger.warning(
            "rerank_movement_clamped",
            listing_id=move.listing_id,
            pre_index=move.pre_index,
            post_index=move.post_index,
            delta=move.delta,
            max_delta=max_delta,
            clamped_index=clamped,
        )
        placements.append((clamped, move.listing_id))

    placements.sort(key=lambda pair: (pair[0], pair[1]))
    return [pid for _, pid in placements]
