"""§17.2.6 NDCG@10 scorer over `SyntheticEvalRating` rows.

Groups ratings per (query_id, rater_id) and orders each group by the `rank`
recorded at rating time — i.e. the position the pipeline gave the listing when
the rating was collected — so the reported NDCG@10 measures the ranking that
was actually shown. Rows without a recorded rank sort after ranked rows in
insertion order.

To score a *candidate* weight configuration (re-running `recommend` instead of
replaying the recorded ranking), use `eval/run_calibration.py`, which sweeps a
weight grid against the same ratings.

Run from the repository root so the uv environment + DB env vars are available:
    uv run python eval/score_ndcg.py
"""

import asyncio
import sys

from sqlalchemy import select

from ingovo_search.db.models.synthetic_eval_rating import SyntheticEvalRating
from ingovo_search.db.session import get_session

# Canonical graded-nDCG math now lives in the importable eval package so the
# rerank candidate-count sweep and this runner share one implementation.
from ingovo_search.eval.rerank_metrics import compute_dcg, compute_ndcg

__all__ = ["compute_dcg", "compute_ndcg", "main"]


def group_ratings_by_recorded_rank(
    ratings: list[SyntheticEvalRating],
) -> dict[tuple[str, str], list[int]]:
    """`{(query_id, rater_id): [rating, ...]}` ordered by the rank recorded at rating time.

    Unranked rows (rank IS NULL) keep their relative order but sort after every
    ranked row, so a partially ranked group still yields a meaningful NDCG.
    """
    groups: dict[tuple[str, str], list[SyntheticEvalRating]] = {}
    for r in ratings:
        groups.setdefault((r.query_id, r.rater_id), []).append(r)

    ordered: dict[tuple[str, str], list[int]] = {}
    for key, rows in groups.items():
        rows.sort(key=lambda r: (r.rank is None, r.rank if r.rank is not None else 0))
        ordered[key] = [r.rating for r in rows]
    return ordered


async def main():
    print("Computing NDCG@10 from SyntheticEvalRatings (ordered by recorded rank)...")

    try:
        async for session in get_session():
            stmt = select(SyntheticEvalRating)
            result = await session.execute(stmt)
            ratings = list(result.scalars().all())

            if not ratings:
                print("No ratings found in the database. Run the rating flow and submit some ratings first.")
                sys.exit(0)

            groups = group_ratings_by_recorded_rank(ratings)

            total_ndcg = 0.0
            for (query_id, rater_id), rels in sorted(groups.items()):
                ndcg = compute_ndcg(rels, k=10)
                print(f"Query: {query_id[:50]:<50} | Rater: {rater_id:<18} | n={len(rels):<3} | NDCG@10: {ndcg:.4f}")
                total_ndcg += ndcg

            avg_ndcg = total_ndcg / len(groups)
            print("-" * 50)
            print(f"Average NDCG@10 across {len(groups)} (query, rater) groups: {avg_ndcg:.4f}")
            break  # only need one session

    except Exception as e:
        print(f"Failed to connect to DB or run query: {e}")
        print("Make sure the database is running and credentials are set.")


if __name__ == "__main__":
    asyncio.run(main())
