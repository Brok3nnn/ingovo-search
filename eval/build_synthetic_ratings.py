"""§17.2 synthetic eval rating builder — LLM-as-judge over recommended candidates.

For each query in `eval_set.starter.jsonl`, mirror the production search pipeline
(parse → embed → recommend top-K), then ask Gemini to rate each (query, listing)
pair on a 1-5 relevance scale. The resulting ratings populate
`SyntheticEvalRating`, the ground-truth dataset that `calibrate_weights.py`
consumes (§17.2.6).

The judge uses a tight rubric so the ratings stay calibrated across queries:
1 = irrelevant, 3 = mediocre match, 5 = excellent fit. Strict JSON output keeps
the parse cost negligible (one integer per call).

CLI:
  uv run python eval/build_synthetic_ratings.py --dry-run --max-queries 3 --top-k 5
  uv run python eval/build_synthetic_ratings.py --max-queries 27 --top-k 20

Run from the repository root so the uv environment + DB env vars are available.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy import text as sql_text

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = REPO_ROOT / "eval" / "golden" / "eval_set.starter.jsonl"

# Load the repository's .env so DATABASE_URL / GEMINI_API_KEY are present
# regardless of which CWD the script is launched from.
load_dotenv(REPO_ROOT / ".env", override=False)

from ingovo_search.db.models.synthetic_eval_rating import SyntheticEvalRating
from ingovo_search.db.session import get_async_session_factory
from ingovo_search.lib.embeddings import EMBEDDING_DIM, CohereEmbeddingClient
from ingovo_search.lib.llm import GoogleAiClient, LlmClient, LlmRequest
from ingovo_search.recommender.parser import parse_query
from ingovo_search.recommender.recommend import recommend

RATER_ID = "gemini-judge-v1"

JUDGE_SYSTEM_PROMPT = """You are a Hungarian real-estate relevance judge. Rate how well a listing matches a buyer's natural-language search query on a 1-5 scale.

Rubric:
- 5 = Excellent match: every hard requirement met (type, city/region, price, rooms, area) and most soft preferences satisfied.
- 4 = Good match: the main hard requirements met, minor mismatches on soft prefs.
- 3 = Mediocre: at least one hard requirement off (e.g. wrong city or wrong type) but adjacent enough to still be plausibly interesting.
- 2 = Weak: multiple hard requirements off; only loose semantic relation.
- 1 = Irrelevant: wrong type AND wrong location, or otherwise clearly off.

Output a single integer (1-5) and nothing else. No prose, no JSON wrapping, no quotes — just the digit."""


def load_queries(limit: int | None) -> list[str]:
    """Read query strings from `eval_set.starter.jsonl`, optionally truncated."""
    queries: list[str] = []
    with EVAL_SET_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            queries.append(json.loads(line)["query"])
    if limit is not None:
        queries = queries[:limit]
    return queries


def listing_brief(prop: Any) -> str:
    """One-line description fed to the judge alongside the query."""
    parts: list[str] = []
    if prop.title:
        parts.append(str(prop.title))
    bits: list[str] = []
    if prop.property_type:
        bits.append(str(prop.property_type))
    if prop.transaction_type:
        bits.append("Eladó" if prop.transaction_type == "sale" else "Kiadó")
    if prop.city:
        loc = str(prop.city)
        if prop.district:
            loc += f", {prop.district}. kerület"
        bits.append(loc)
    if prop.area_m2:
        bits.append(f"{prop.area_m2:.0f} m²")
    if prop.rooms:
        bits.append(f"{prop.rooms} szoba")
    if prop.price:
        if prop.price >= 1_000_000:
            bits.append(f"{prop.price / 1_000_000:.1f} M Ft")
        else:
            bits.append(f"{int(prop.price):,} Ft".replace(",", " "))
    if bits:
        parts.append(" | ".join(bits))
    if prop.description:
        desc = str(prop.description).strip().replace("\n", " ")
        if len(desc) > 280:
            desc = desc[:280] + "…"
        parts.append(f"Leírás: {desc}")
    return "\n".join(parts)


async def llm_judge(query: str, prop: Any, *, client: LlmClient) -> int:
    """Return a 1-5 integer rating for (query, prop). Defensive on parse errors."""
    user = f"Lekérdezés:\n{query}\n\nIngatlan:\n{listing_brief(prop)}\n\nÉrtékelés (1-5):"
    req = LlmRequest(model="gemini-2.5-flash-lite", system=JUDGE_SYSTEM_PROMPT, user=user, temperature=0.0)
    out = (await client.complete(req)).strip()
    # Extract first digit 1-5 — the prompt asks for bare integer but be defensive.
    for ch in out:
        if ch in "12345":
            return int(ch)
    raise ValueError(f"judge returned no rating in {out!r}")


async def candidates_for_query(
    session: Any, query: str, *, client: LlmClient, emb_client: Any, top_k: int
) -> list[Any]:
    """Mirror the prod /api/search wiring up to `recommend`."""
    parsed = await parse_query(query, client=client)
    if parsed.semantic_query:
        query_emb = await emb_client.embed(parsed.semantic_query, "search_query")
    else:
        query_emb = [0.0] * EMBEDDING_DIM
    freetext_emb = None
    if parsed.soft_prefs_freetext:
        freetext_emb = await emb_client.embed(" ".join(parsed.soft_prefs_freetext), "search_query")
    scored = await recommend(
        session, parsed, query_emb=query_emb, freetext_pref_emb=freetext_emb, client=client, top_n=top_k
    )
    return [prop for prop, _breakdown in scored]


async def upsert_rating(session: Any, *, query_id: str, property_id: str, rating: int) -> bool:
    """Insert a rating row, skipping silently if one already exists for this triple."""
    existing = (
        await session.execute(
            select(SyntheticEvalRating).where(
                SyntheticEvalRating.query_id == query_id,
                SyntheticEvalRating.property_id == property_id,
                SyntheticEvalRating.rater_id == RATER_ID,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        existing.rating = rating  # refresh in case the judge changes its mind
        return False
    session.add(
        SyntheticEvalRating(
            query_id=query_id,
            property_id=property_id,
            rater_id=RATER_ID,
            rating=rating,
            rating_source="synthetic",
            inventory_source="live",
        )
    )
    return True


async def main_async(args: argparse.Namespace) -> None:
    queries = load_queries(args.max_queries)
    print(f"Loaded {len(queries)} queries (top_k={args.top_k}, dry_run={args.dry_run})", flush=True)

    llm_client: LlmClient = GoogleAiClient()
    cohere_key = os.environ.get("COHERE_API_KEY")
    if not cohere_key:
        raise SystemExit("COHERE_API_KEY missing — cannot embed queries")
    emb_client = CohereEmbeddingClient(api_key=cohere_key)

    Session = get_async_session_factory()

    total_rated = 0
    distribution: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    async with Session() as session:
        for q_idx, query in enumerate(queries, start=1):
            print(f"\n[{q_idx}/{len(queries)}] {query!r}", flush=True)
            try:
                candidates = await candidates_for_query(
                    session, query, client=llm_client, emb_client=emb_client, top_k=args.top_k
                )
            except Exception as exc:
                print(f"  ⚠️  candidate fetch failed: {exc}", flush=True)
                continue
            if not candidates:
                print("  (no candidates)", flush=True)
                continue

            for prop in candidates:
                try:
                    rating = await llm_judge(query, prop, client=llm_client)
                except Exception as exc:
                    print(f"  ⚠️  judge failed for {prop.id}: {exc}", flush=True)
                    continue
                distribution[rating] = distribution.get(rating, 0) + 1
                total_rated += 1
                tag = "DRY" if args.dry_run else "WRITE"
                print(
                    f"  [{tag}] rating={rating}  {prop.city!s:>14} | {prop.property_type!s:>14} | {prop.title!s:.60}",
                    flush=True,
                )
                if not args.dry_run:
                    await upsert_rating(session, query_id=query, property_id=prop.id, rating=rating)
            if not args.dry_run:
                await session.commit()

    print("\n=== SUMMARY ===")
    print(f"Total ratings produced: {total_rated}")
    print(f"Distribution: {distribution}")
    if not args.dry_run:
        # Quick sanity: count rows persisted by this rater
        async with Session() as session:
            n = (
                await session.execute(
                    sql_text("SELECT count(*) FROM synthetic_eval_rating WHERE rater_id = :r"), {"r": RATER_ID}
                )
            ).scalar_one()
            print(f"DB rows for rater {RATER_ID}: {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SyntheticEvalRating via LLM-as-judge.")
    parser.add_argument("--max-queries", type=int, default=3, help="Truncate the eval-set head (default 3 for safety).")
    parser.add_argument("--top-k", type=int, default=5, help="Top candidates per query to judge (default 5).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run judge but DO NOT write to synthetic_eval_rating. Recommended for the first invocation.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main() or 0)
