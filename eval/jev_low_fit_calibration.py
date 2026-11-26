"""Calibrate `low_fit_threshold` from the cached Jev answers and the judge's grades.

`jev_rerank` flags a candidate `low_confidence` when its Jev fit (0..1) is
below `LOW_FIT_THRESHOLD`; `low_match` flips when most of the top 10 are
flagged. This script rebuilds the chosen
variant's request for every frozen head, reads the cached Jev answer
(`eval/out/jev_rerank_cache/`, written by `jev_rerank_offline.py`, so no new
calls), joins the fits with the rater's grades, and reports how well each
threshold separates weak matches (grade <= 2) from acceptable ones (grade >= 3):
balanced accuracy, precision and recall of the "weak" flag.

Run from the repository root:
    uv run python eval/jev_low_fit_calibration.py --variant score4-hu-short-r2 --lean --rater gemini-judge-v2-heads
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingovo_search.db.session import get_async_session_factory
from ingovo_search.lib.jev import JevResponse
from ingovo_search.lib.llm_models import TIER_L_RERANK_JEV
from ingovo_search.recommender.rerank_jev_prompt import build_jev_rerank_request
from jev_rerank_offline import (
    BASE_VARIANTS,
    CACHE_DIR,
    OUT_DIR,
    REPO_ROOT,
    DiskCache,
    _digest,  # pyright: ignore[reportPrivateUsage]
    _question_json,  # pyright: ignore[reportPrivateUsage]
    load_heads,
    load_relevance,
)

WEAK_GRADE_MAX = 2.0
THRESHOLDS = [round(0.1 * step, 2) for step in range(1, 10)]


def _fits_for(cache: DiskCache, query, variant) -> dict[str, float] | None:
    """listing id -> Jev fit (0..1) from the cache, or None when this request was never made."""
    request = build_jev_rerank_request(query.parsed, list(query.candidates), variant)
    key = _digest(
        {
            "model": TIER_L_RERANK_JEV,
            "state": request.state,
            "questions": {k: _question_json(q) for k, q in request.questions.items()},
        }
    )
    cached = cache.get(key)
    if cached is None:
        return None
    response = JevResponse.model_validate(cached["response"])
    if variant.primitive == "score":
        raw = {k: a.score / request.max_level for k, a in response.scores.items()}
    else:
        raw = dict(response.nouls)
    return {
        request.key_to_listing_id[k]: min(1.0, max(0.0, v)) for k, v in raw.items() if k in request.key_to_listing_id
    }


async def main_async(args: argparse.Namespace) -> None:
    variant = BASE_VARIANTS[args.variant]
    variant = replace(variant, lean_summary=args.lean, description_chars=args.description_chars)
    cache = DiskCache(CACHE_DIR)
    heads = load_heads(head=args.head)
    async with get_async_session_factory()() as session:
        relevance = await load_relevance(session, args.rater)

    pairs: list[tuple[float, float]] = []  # (fit, grade)
    missing = 0
    for query_id, query in heads.items():
        grades = relevance.get(query_id)
        if not grades or len(query.candidates) < 4:
            continue
        fits = _fits_for(cache, query, variant)
        if fits is None:
            missing += 1
            continue
        pairs.extend((fit, grades[lid]) for lid, fit in fits.items() if lid in grades)
    if not pairs:
        raise SystemExit(f"no cached answers for {variant.prompt_version}; run jev_rerank_offline.py first")

    weak = [fit for fit, grade in pairs if grade <= WEAK_GRADE_MAX]
    good = [fit for fit, grade in pairs if grade > WEAK_GRADE_MAX]
    rows: list[dict[str, float]] = []
    for threshold in THRESHOLDS:
        flagged_weak = sum(1 for fit in weak if fit < threshold)
        flagged_good = sum(1 for fit in good if fit < threshold)
        recall = flagged_weak / len(weak) if weak else 0.0
        specificity = 1 - (flagged_good / len(good) if good else 0.0)
        precision = flagged_weak / (flagged_weak + flagged_good) if flagged_weak + flagged_good else 0.0
        rows.append(
            {
                "threshold": threshold,
                "balanced_accuracy": (recall + specificity) / 2,
                "precision": precision,
                "recall": recall,
                "flagged_share": (flagged_weak + flagged_good) / len(pairs),
            }
        )
    best = max(rows, key=lambda row: row["balanced_accuracy"])
    label = args.label or "latest"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"jev_low_fit_calibration_{label}.json"
    out.write_text(
        json.dumps(
            {
                "variant": variant.prompt_version,
                "rater": args.rater,
                "pairs": len(pairs),
                "weak": len(weak),
                "good": len(good),
                "heads_without_cached_answer": missing,
                "mean_fit_weak": sum(weak) / len(weak) if weak else None,
                "mean_fit_good": sum(good) / len(good) if good else None,
                "rows": rows,
                "best": best,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        f"variant {variant.prompt_version}: {len(pairs)} graded candidates ({len(weak)} weak, {len(good)} good), {missing} heads without cached answer"
    )
    print(f"mean fit: weak {sum(weak) / len(weak) if weak else 0:.3f}, good {sum(good) / len(good) if good else 0:.3f}")
    print("threshold  bal.acc  precision  recall  flagged")
    for row in rows:
        marker = " <- best" if row is best else ""
        print(
            f"   {row['threshold']:.1f}     {row['balanced_accuracy']:.3f}    {row['precision']:.3f}    {row['recall']:.3f}   {row['flagged_share']:.1%}{marker}"
        )
    print(f"written: {out.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="score4-hu-short-r2", choices=sorted(BASE_VARIANTS))
    parser.add_argument("--lean", action="store_true")
    parser.add_argument("--description-chars", type=int, default=None)
    parser.add_argument("--rater", default="gemini-judge-v2-heads")
    parser.add_argument("--head", type=int, default=12)
    parser.add_argument("--label", default=None)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
