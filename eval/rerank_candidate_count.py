"""Rerank candidate-count sweep runner (top 30 / 20 / 15).

Drives the live `POST /api/search` endpoint at several `rerank_candidate_count`
values and scores each against the human ratings, then prints an nDCG@10
/ top-1 / latency / cost comparison and the smallest count that preserves
quality (`recommend_candidate_count`).

The metric + aggregation logic lives in (and is unit-tested via)
`ingovo_search.eval.rerank_metrics` and `ingovo_search.eval.rerank_candidate_sweep`; this file
is the live wiring only.

Prerequisites (operator-supplied — cannot run in CI):
- the API server running locally (default http://127.0.0.1:8000),
- `synthetic_eval_rating` populated with graded ratings,
- the eval queries in `eval_set.starter.jsonl` (repo root).

Linkage assumption: a rating's `query_id` equals the query TEXT. If your rating
UI uses a different id, adjust `_query_id_for` below.

Usage:
    python -m eval.rerank_candidate_count            # 30,20,15 vs localhost
    API_BASE=http://127.0.0.1:8000 python -m eval.rerank_candidate_count
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from sqlalchemy import select

from ingovo_search.db.models.synthetic_eval_rating import SyntheticEvalRating
from ingovo_search.db.session import get_session
from ingovo_search.eval.rerank_candidate_sweep import (
    EvalQuery,
    SearchOutcome,
    format_comparison_table,
    recommend_candidate_count,
    run_comparison,
)

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
EVAL_SET_PATH = Path(__file__).resolve().parents[1] / "eval" / "golden" / "eval_set.starter.jsonl"
CANDIDATE_COUNTS = [30, 20, 15]


def _query_id_for(query_text: str) -> str:
    """How a query maps to its rating `query_id`. Adjust to match your rating UI."""
    return query_text


def _load_eval_queries() -> list[EvalQuery]:
    if not EVAL_SET_PATH.exists():
        raise SystemExit(f"eval set not found: {EVAL_SET_PATH}")
    queries: list[EvalQuery] = []
    for line in EVAL_SET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        text = json.loads(line)["query"]
        queries.append(EvalQuery(query_id=_query_id_for(text), query=text))
    return queries


async def _load_relevance() -> dict[str, dict[str, float]]:
    """`{query_id: {property_id: rating}}` from the ratings table."""
    relevance: dict[str, dict[str, float]] = {}
    async for session in get_session():
        rows = (await session.execute(select(SyntheticEvalRating))).scalars().all()
        for r in rows:
            relevance.setdefault(r.query_id, {})[r.property_id] = float(r.rating)
        break
    return relevance


def _make_search_fn(client: httpx.AsyncClient):
    async def search_fn(query: str, count: int) -> SearchOutcome:
        started = time.perf_counter()
        resp = await client.post(
            f"{API_BASE}/api/search",
            json={"query": query, "include_rerank": True, "rerank_candidate_count": count},
        )
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - started) * 1000.0
        data = resp.json().get("data", {})
        ranked_ids = [c["listing_id"] for c in data.get("results", [])]
        return SearchOutcome(ranked_ids=ranked_ids, latency_ms=latency_ms)

    return search_fn


async def main() -> None:
    eval_queries = _load_eval_queries()
    relevance = await _load_relevance()
    if not relevance:
        raise SystemExit("no ratings found — populate synthetic_eval_rating first.")

    rated = [q for q in eval_queries if relevance.get(q.query_id)]
    print(f"Sweeping {CANDIDATE_COUNTS} over {len(rated)}/{len(eval_queries)} rated queries via {API_BASE}\n")

    async with httpx.AsyncClient(timeout=60.0) as client:
        results = await run_comparison(rated, relevance, _make_search_fn(client), CANDIDATE_COUNTS, k=10)

    print(format_comparison_table(results))
    print("-" * 48)
    print(f"Recommended candidate count (smallest preserving quality): {recommend_candidate_count(results)}")


if __name__ == "__main__":
    asyncio.run(main())
