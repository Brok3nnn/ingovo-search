"""Weight-calibration runner — DB-backed grid search over SyntheticEvalRating.

Wires the three pieces the pure calibrator in `calibrate_weights.py` needs:
1. **Ground-truth loader** — pulls every (`query_id`, `property_id`, `rating`)
   for the configured rater from `synthetic_eval_rating` and shapes it into
   the `{query_id: {property_id: rating}}` dict `calibrate()` expects.
2. **Score function** — for each (query, weight-combo) it calls
   `recommend(..., weights=...)` and returns the ranked property-ID list. The
   per-query parser output and the Cohere query embedding are computed once
   per query and reused across the whole weight grid (LLM/Cohere round-trips
   are the dominant cost; the same parsed query + embedding is valid for
   every weight combo).
3. **Grid spec** — a compact 2-axis sweep around the hand-set
   defaults; widen via CLI flags once the quota expands.

CLI:
  uv run python eval/run_calibration.py             # default grid
  uv run python eval/run_calibration.py --top 10    # print top-10

Run from the repository root so `uv` picks up the right venv + env vars.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env", override=False)
sys.path.insert(0, str(REPO_ROOT / "eval"))

from calibrate_weights import CalibrationResult, calibrate, expand_grid
from ingovo_search.db.models.synthetic_eval_rating import SyntheticEvalRating
from ingovo_search.db.session import get_async_session_factory
from ingovo_search.lib.embeddings import EMBEDDING_DIM, CohereEmbeddingClient
from ingovo_search.lib.llm import GoogleAiClient, LlmRequest


class _EmptyRegionHintClient:
    """LlmClient stub for recommend's cold-start region hints (run_battery
    convention): deterministic, quota-free 'no districts known'."""

    async def complete(self, req: LlmRequest) -> str:
        return '{"districts": []}'


from ingovo_search.recommender.parser import parse_query
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.recommend import recommend
from ingovo_search.recommender.score import WEIGHTS

# Default rater is the human rating identity (B5/B6 flow); the
# synthetic Gemini-judge path stays reachable via --rater gemini-judge-v1.
DEFAULT_RATER_ID = "founder-admin"

# Traffic weighting (design decision 2026-07-16): the eval set's category
# counts are a coverage choice, but Ingovo is AI-first search, so the
# calibration objective weights queries by the *expected traffic mix* of
# their length class. Post-launch, replace the prior with the measured
# distribution from search_log via --traffic-mix.
DEFAULT_TRAFFIC_MIX = {"long": 0.6, "medium": 0.3, "short": 0.1}
LENGTH_CLASS_BY_CATEGORY = {
    "long_narrative": "long",
    "hard_constraint": "short",
    # everything else (multi_constraint, colloquial, fuzzy_geo, negation,
    # anchor, soft_pref, poi, adversarial) is medium.
}
EVAL_SET_PATH = REPO_ROOT / "eval" / "golden" / "eval_set.starter.jsonl"


def load_eval_records() -> dict[str, dict]:
    import json

    records: dict[str, dict] = {}
    with EVAL_SET_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[record["query"]] = record
    return records


def load_query_categories() -> dict[str, str]:
    return {q: (r.get("category") or "medium") for q, r in load_eval_records().items()}


def traffic_query_weights(
    eval_set: dict[str, dict[str, int]],
    traffic_mix: dict[str, float],
) -> dict[str, float]:
    """Per-query weights so each length class contributes its traffic share.

    Weight = class_share / n_rated_queries_in_class; classes absent from the
    rated set are dropped (their share implicitly renormalises away).
    """
    categories = load_query_categories()
    class_of = {qid: LENGTH_CLASS_BY_CATEGORY.get(categories.get(qid, ""), "medium") for qid in eval_set}
    counts: dict[str, int] = {}
    for cls in class_of.values():
        counts[cls] = counts.get(cls, 0) + 1
    return {qid: traffic_mix.get(cls, 0.0) / counts[cls] for qid, cls in class_of.items() if counts.get(cls)}


def parse_traffic_mix(raw: str) -> dict[str, float]:
    mix: dict[str, float] = {}
    for part in raw.split(","):
        name, _, value = part.partition("=")
        mix[name.strip()] = float(value)
    return mix


async def load_eval_set(rater_id: str) -> dict[str, dict[str, int]]:
    """Pull every rating for `rater_id` from the DB and reshape.

    Returns `{query_id: {property_id: rating}}`. Drops queries that ended up
    with fewer than 2 ratings — single-rating queries cannot meaningfully
    discriminate weight combos under nDCG.
    """
    Session = get_async_session_factory()
    async with Session() as session:
        rows = (
            (await session.execute(select(SyntheticEvalRating).where(SyntheticEvalRating.rater_id == rater_id)))
            .scalars()
            .all()
        )
    eval_set: dict[str, dict[str, int]] = {}
    for r in rows:
        eval_set.setdefault(r.query_id, {})[r.property_id] = r.rating
    return {q: ratings for q, ratings in eval_set.items() if len(ratings) >= 2}


async def precompute_rankings(
    eval_set: dict[str, dict[str, int]], weight_grid: list[dict[str, float]]
) -> dict[tuple[str, tuple[tuple[str, float], ...]], list[str]]:
    """Run every (query, weight-combo) through `recommend` and cache the ranking.

    Per-query parser output + Cohere embedding are computed once and shared
    across the grid, since they do not depend on the scoring weights. The
    keyed cache lets the synchronous `calibrate()` look rankings up by
    `(query_id, weight-tuple)` without re-entering an async loop.
    """
    llm_client = GoogleAiClient()
    cohere_key = os.environ.get("COHERE_API_KEY")
    if not cohere_key:
        raise SystemExit("COHERE_API_KEY missing — cannot embed queries")
    emb_client = CohereEmbeddingClient(api_key=cohere_key)
    Session = get_async_session_factory()

    rankings: dict[tuple[str, tuple[tuple[str, float], ...]], list[str]] = {}
    print(f"Precomputing rankings: {len(eval_set)} queries x {len(weight_grid)} weight combos", flush=True)

    eval_records = load_eval_records()
    for q_idx, query in enumerate(eval_set, start=1):
        record = eval_records.get(query)
        if record is not None and record.get("expected"):
            # The eval set's expected block IS the parse (same convention as
            # run_battery and the rating backend) — deterministic and free of
            # Gemini quota; the LLM path only covers ad-hoc queries.
            parsed = ParsedQuery.model_validate(record["expected"])
        else:
            parsed = await parse_query(query, client=llm_client)
        query_emb = (
            await emb_client.embed(parsed.semantic_query, "search_query")
            if parsed.semantic_query
            else [0.0] * EMBEDDING_DIM
        )
        freetext_emb = (
            await emb_client.embed(" ".join(parsed.soft_prefs_freetext), "search_query")
            if parsed.soft_prefs_freetext
            else None
        )
        print(f"  [{q_idx}/{len(eval_set)}] {query!r}", flush=True)

        for combo in weight_grid:
            async with Session() as session:
                scored = await recommend(
                    session,
                    parsed,
                    query_emb=query_emb,
                    freetext_pref_emb=freetext_emb,
                    # Stubbed LLM inside recommend (region cold-start hints):
                    # calibration must be deterministic and quota-free, same
                    # as run_battery. The real client only backs parse_query
                    # for ad-hoc queries missing from the eval file.
                    client=_EmptyRegionHintClient(),  # type: ignore[arg-type]
                    top_n=30,
                    weights=combo,
                )
            key = (query, tuple(sorted(combo.items())))
            rankings[key] = [str(prop.id) for prop, _ in scored]

    return rankings


def render_results(results: list[CalibrationResult], *, top: int, defaults: dict[str, float]) -> None:
    """Pretty-print the top-N calibration results with relative-to-default deltas."""
    print("\n=== TOP-N WEIGHT COMBOS BY AVG NDCG@10 ===")
    header = f"{'rank':>4}  {'avg_ndcg':>10}  weights (delta vs default)"
    print(header)
    print("-" * len(header))
    for rank, result in enumerate(results[:top], start=1):
        deltas = ", ".join(f"{k}={v:.3f} ({v - defaults.get(k, 0):+.3f})" for k, v in sorted(result.weights.items()))
        print(f"{rank:>4}  {result.avg_ndcg:>10.4f}  {deltas}")
    print(f"\nDefault baseline (vector hard-set in score.py): {defaults}")


async def main_async(args: argparse.Namespace) -> None:
    eval_set = await load_eval_set(args.rater)
    total_ratings = sum(len(v) for v in eval_set.values())
    print(f"Eval set: {len(eval_set)} queries (>=2 ratings), {total_ratings} ratings", flush=True)
    if not eval_set:
        raise SystemExit(
            f"No ratings for rater {args.rater!r} — rate via the internal rating tool or run build_synthetic_ratings.py."
        )

    # v0.4 grid: quality is now populated (enrichment backfill done),
    # so the weight is worth tuning again. Sweep quality x content x geo.
    # 4 x 3 x 2 = 24 combos.
    grid_spec = [
        {
            "quality": [0.05, 0.10, 0.15, 0.20],
            "content": [0.15, 0.20, 0.25],
            "geo": [0.10, 0.15],
        },
    ]
    weight_grid = expand_grid(grid_spec)

    rankings = await precompute_rankings(eval_set, weight_grid)

    def sync_score_fn(qid: str, weights: dict[str, float]) -> list[str]:
        return rankings.get((qid, tuple(sorted(weights.items()))), [])

    query_weights = None
    if args.traffic_mix != "off":
        mix = parse_traffic_mix(args.traffic_mix)
        query_weights = traffic_query_weights(eval_set, mix)
        by_class: dict[str, int] = {}
        categories = load_query_categories()
        for qid in eval_set:
            cls = LENGTH_CLASS_BY_CATEGORY.get(categories.get(qid, ""), "medium")
            by_class[cls] = by_class.get(cls, 0) + 1
        print(f"Traffic-weighted objective: mix={mix}, rated queries per class={by_class}", flush=True)

    results = calibrate(eval_set, sync_score_fn, weight_grid, k=10, query_weights=query_weights)
    render_results(results, top=args.top, defaults=dict(WEIGHTS))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weight calibration grid search.")
    parser.add_argument("--top", type=int, default=10, help="How many best combos to print (default 10).")
    parser.add_argument("--rater", default=DEFAULT_RATER_ID, help="rater_id whose ratings form the ground truth.")
    parser.add_argument(
        "--traffic-mix",
        default="long=0.6,medium=0.3,short=0.1",
        help="expected traffic share per length class, or 'off' for a plain mean.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
