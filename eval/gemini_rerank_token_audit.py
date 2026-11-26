"""The Gemini rerank's real token count and cost.

`EnrichmentLog` estimates the rerank's tokens from characters (4 per token).
This script runs the production rerank (`rerank_and_explain`, same prompts,
same schema, same model) on the rated queries' real rerank heads through a
client that keeps Gemini's `usage_metadata`, and reports the billed input and
output tokens, the real cost per rerank and the ratio to the estimate.

The responses land in the same disk cache as `jev_rerank_offline.py`, so the
offline comparison's Gemini arm is free afterwards.

Run from the repository root (Gemini fees apply, ~20 calls):
    uv run python eval/gemini_rerank_token_audit.py
    uv run python eval/gemini_rerank_token_audit.py --queries 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingovo_search.db.session import get_async_session_factory
from ingovo_search.eval.gemini_usage import UsageRecordingGeminiClient
from ingovo_search.eval.jev_rerank_offline import percentile
from ingovo_search.lib.embeddings import CohereEmbeddingClient
from ingovo_search.lib.env import get_settings
from ingovo_search.lib.llm_models import TIER_L_RERANK
from jev_rerank_offline import (
    CACHE_DIR,
    OUT_DIR,
    REPO_ROOT,
    CachingGeminiClient,
    DiskCache,
    gemini_arm,
    load_eval_records,
    load_or_build_heads,
    load_relevance,
)


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


async def main_async(args: argparse.Namespace) -> None:
    settings = get_settings()
    records_by_text = load_eval_records()
    gemini = CachingGeminiClient(UsageRecordingGeminiClient.from_api_key(settings.gemini_api_key), DiskCache(CACHE_DIR))
    embedder = CohereEmbeddingClient(api_key=settings.cohere_api_key)

    async with get_async_session_factory()() as session:
        relevance = await load_relevance(session, args.rater)
        rated = [records_by_text[text] for text in records_by_text if text in relevance][: args.queries]
        queries = [q for q in await load_or_build_heads(session, rated, embedder) if len(q.candidates) >= 2]

    arm = gemini_arm(gemini, rank_only=settings.ai_search_rerank_rank_only)
    rows: list[dict[str, object]] = []
    for query in queries:
        first = len(gemini.usage)
        run = await arm(query)
        calls = gemini.usage[first:]
        for usage in calls:
            rows.append({"query": query.query_id, "head": len(query.candidates), "outcome": run.outcome, **usage})
        print(
            f"{query.query_id[:50]:<50} {run.outcome:<9} calls={len(calls)} "
            f"in={run.input_tokens} out={run.output_tokens} {'cache' if run.from_cache else f'{run.latency_ms:.0f} ms'}",
            flush=True,
        )

    real_in = [float(r["input_tokens"]) for r in rows]
    real_out = [float(r["output_tokens"]) for r in rows]
    est_in = [float(r["estimated_input_tokens"]) for r in rows]
    est_out = [float(r["estimated_output_tokens"]) for r in rows]
    real_cost = [float(r["cost_usd"]) for r in rows]
    est_cost = [float(r["estimated_cost_usd"]) for r in rows]
    latency = [float(r["latency_ms"]) for r in rows]
    thoughts = [float(r["thoughts_tokens"]) for r in rows]
    summary = {
        "model": TIER_L_RERANK,
        "rank_only": settings.ai_search_rerank_rank_only,
        "calls": len(rows),
        "queries": len(queries),
        "input_tokens_p50_real": _median(real_in),
        "input_tokens_mean_real": sum(real_in) / len(real_in) if real_in else 0.0,
        "input_tokens_p50_estimated": _median(est_in),
        "output_tokens_p50_real": _median(real_out),
        "output_tokens_p50_estimated": _median(est_out),
        "thoughts_tokens_p50": _median(thoughts),
        "cost_usd_p50_real": _median(real_cost),
        "cost_usd_mean_real": sum(real_cost) / len(real_cost) if real_cost else 0.0,
        "cost_usd_p50_estimated": _median(est_cost),
        "latency_ms_p50": _median(latency),
        "latency_ms_p95": percentile(latency, 0.95),
    }
    ratio_in = summary["input_tokens_p50_real"] / summary["input_tokens_p50_estimated"] if est_in else 0.0
    ratio_cost = summary["cost_usd_p50_real"] / summary["cost_usd_p50_estimated"] if est_cost else 0.0

    label = args.label or date.today().isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / f"gemini_rerank_token_audit_{label}.md"
    json_path = OUT_DIR / f"gemini_rerank_token_audit_{label}.json"
    lines = [
        f"# Gemini rerank — real token count and cost ({label})",
        "",
        f"- **Model:** `{TIER_L_RERANK}`, rank_only={settings.ai_search_rerank_rank_only}, through the production `rerank_and_explain` path",
        f"- **Calls:** {len(rows)} ({len(queries)} rated queries, each with its real rerank candidates)",
        "",
        "| | real (usage_metadata) | estimated (EnrichmentLog, 4 chars/token) | ratio |",
        "|---|---|---|---|",
        f"| input token, p50 | {summary['input_tokens_p50_real']:.0f} | {summary['input_tokens_p50_estimated']:.0f} | {ratio_in:.2f} |",
        f"| output token, p50 | {summary['output_tokens_p50_real']:.0f} | {summary['output_tokens_p50_estimated']:.0f} | - |",
        f"| cost / rerank, p50 | {summary['cost_usd_p50_real']:.6f} $ | {summary['cost_usd_p50_estimated']:.6f} $ | {ratio_cost:.2f} |",
        f"| cost / rerank, mean | {summary['cost_usd_mean_real']:.6f} $ | | |",
        f"| latency p50 / p95 | {summary['latency_ms_p50']:.0f} / {summary['latency_ms_p95']:.0f} ms | | |",
        f"| thinking tokens, p50 | {summary['thoughts_tokens_p50']:.0f} | | |",
        "",
        "## Per call",
        "",
        "| Query | candidates | outcome | input real | input estimated | output real | output estimated | cost real | latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {str(r['query'])[:50]} | {r['head']} | {r['outcome']} | {r['input_tokens']} | {r['estimated_input_tokens']} | "
            f"{r['output_tokens']} | {r['estimated_output_tokens']} | {float(r['cost_usd']):.6f} $ | {float(r['latency_ms']):.0f} ms |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps({"summary": summary, "calls": rows}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\nwritten: {md_path.relative_to(REPO_ROOT)} and {json_path.relative_to(REPO_ROOT)}")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=int, default=20, help="rated queries to run (default 20)")
    parser.add_argument("--rater", default=None, help="rater_id whose queries to use; default: any")
    parser.add_argument("--label", default=None, help="output file label (default: today's date)")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
