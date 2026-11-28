"""AI search latency + quality benchmark over 50 annotated queries.

Drives the live `POST /api/search` endpoint over a set of annotated queries
(`latency_benchmark_queries.jsonl`) and measures, per query:

- cold latency (first call) and cached latency (immediate identical repeat),
- returned result count and the top-10 listing-id order,
- whether the response respected the query's hard filters — *accounting for
  relaxation*: a constraint that the pipeline deliberately loosened (its label
  is in `relaxation_path`) is NOT counted as a violation,
- a heuristic cache-hit signal,
- per-phase timings (parser / recommend / rerank) when the API exposes them in
  `meta.timings_ms` (forward-compatible — see the note below).

It prints the required markdown table plus a p50/p95/max summary, and writes a
machine-readable JSON next to the report so a later run can diff `top-10` order
for the "quality vs baseline" column (`--baseline`).

Per-phase timings note
----------------------
The pipeline currently logs per-phase timings via structlog ("search pipeline
timing") but does not surface them in the HTTP response, so this harness reads
them from `meta.timings_ms` *if present*. Exposing that dict in the search
response envelope (a one-liner in the search endpoint, owned by the
cache/pipeline track) makes the parser/recommend/rerank columns populate
automatically — no change needed here.

Prerequisites (operator-supplied — cannot run in CI):
- the API server running locally (default http://127.0.0.1:8000).

Usage:
    uv run python -m eval.ai_search_latency_benchmark
    API_BASE=http://127.0.0.1:8000 uv run python -m eval.ai_search_latency_benchmark \
        --out eval/out/latency_report.md --json eval/out/latency_results.json
    uv run python -m eval.ai_search_latency_benchmark --baseline eval/out/baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
DEFAULT_QUERIES = Path(__file__).resolve().parent / "latency_benchmark_queries.jsonl"
TOP_N = 10
# A repeat call is treated as a cache hit when it returns the same top-10 and is
# meaningfully faster. Authoritative detection needs the cache_hit flag / the
# timings meta; this is a transparent heuristic for the report.
CACHE_HIT_SPEEDUP = 0.6


# --- relaxation-aware hard-filter checking ---------------------------------


def _norm(text: str | None) -> str:
    """Lowercased, accent-folded text for tolerant location matching."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in folded if not unicodedata.combining(ch)).lower().strip()


def _relaxed(path: list[str], *keywords: str) -> bool:
    """True if any `relaxation_path` label contains one of `keywords`."""
    norm_path = [_norm(label) for label in path]
    return any(any(kw in label for kw in keywords) for label in norm_path)


@dataclass
class FilterCheck:
    ok: bool = True
    violations: list[str] = field(default_factory=list)


def check_hard_filters(
    expect: dict[str, Any],
    results: list[dict[str, Any]],
    relaxation_path: list[str],
) -> FilterCheck:
    """Validate that returned listings respect the query's hard filters.

    A constraint that was deliberately relaxed (its label is in
    `relaxation_path`) is excluded — relaxation is intended behaviour, surfaced
    to the user as "Lazított találatok", not a guardrail violation.
    """
    check = FilterCheck()
    if not results:
        return check  # vacuously OK; the result-count column carries the signal

    def props() -> list[dict[str, Any]]:
        return [(r.get("property") or {}) for r in results]

    price_max = expect.get("price_huf_max")
    if price_max is not None and not _relaxed(relaxation_path, "max ar", "ar"):
        over = [int(p["price"]) for p in props() if p.get("price") and p["price"] > price_max]
        if over:
            check.ok = False
            check.violations.append(f"price>{price_max}: {len(over)} (e.g. {max(over)})")

    price_min = expect.get("price_huf_min")
    if price_min is not None and not _relaxed(relaxation_path, "min ar", "ar"):
        under = [int(p["price"]) for p in props() if p.get("price") and p["price"] < price_min]
        if under:
            check.ok = False
            check.violations.append(f"price<{price_min}: {len(under)} (e.g. {min(under)})")

    tx = expect.get("transaction_type")
    if tx is not None:  # transaction is never relaxed — always enforced
        bad = [p.get("transaction_type") for p in props() if p.get("transaction_type") not in (None, tx)]
        if bad:
            check.ok = False
            check.violations.append(f"transaction!={tx}: {len(bad)}")

    city_contains = expect.get("city_contains")
    if city_contains and not _relaxed(relaxation_path, "varos"):
        bad = [
            p.get("city")
            for p in props()
            if p.get("city") and not any(s in _norm(p.get("city")) for s in city_contains)
        ]
        if bad:
            check.ok = False
            check.violations.append(f"city not in {city_contains}: {len(bad)} (e.g. {bad[0]})")

    district_contains = expect.get("district_contains")
    if district_contains and not _relaxed(relaxation_path, "kerulet"):
        # Only meaningful for results that carry a district at all.
        bad = [
            p.get("district")
            for p in props()
            if p.get("district") and not any(s in _norm(p.get("district")) for s in district_contains)
        ]
        if bad:
            check.ok = False
            check.violations.append(f"district not in {district_contains}: {len(bad)} (e.g. {bad[0]})")

    return check


# --- per-query run ----------------------------------------------------------


@dataclass
class QueryResult:
    id: str
    category: str
    query: str
    cold_ms: float
    cached_ms: float
    parser_ms: float | None
    recommend_ms: float | None
    rerank_ms: float | None
    total_ms_server: float | None
    result_count: int
    top10: list[str]
    relaxed: bool
    relaxation_path: list[str]
    cache_hit: bool
    hard_filter_ok: bool
    violations: list[str]
    quality_vs_baseline: str = "n/a"
    error: str | None = None


def _timings(envelope: dict[str, Any]) -> dict[str, float]:
    meta = envelope.get("meta") or {}
    raw = meta.get("timings_ms") or {}
    return {k: float(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def _benchmark_ip(qid: str) -> str:
    digest = hashlib.sha256(qid.encode("utf-8")).digest()
    # RFC 2544 benchmarking range; used only as an X-Forwarded-For test key.
    return f"198.18.{digest[0]}.{max(1, digest[1])}"


async def _one_call(client: httpx.AsyncClient, query: str, *, qid: str) -> tuple[float, dict[str, Any]]:
    body = {"query": query, "include_rerank": True, "include_narratives": False, "reset_session": True}
    headers = {"X-Forwarded-For": _benchmark_ip(qid)}
    started = time.perf_counter()
    resp = await client.post(f"{API_BASE}/api/search", json=body, headers=headers)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    resp.raise_for_status()
    return elapsed_ms, resp.json()


async def run_query(client: httpx.AsyncClient, spec: dict[str, Any]) -> QueryResult:
    qid = spec.get("id", "?")
    category = spec.get("category", "")
    query = spec["query"]
    expect = spec.get("expect", {})

    try:
        cold_ms, cold_env = await _one_call(client, query, qid=qid)
        cached_ms, cached_env = await _one_call(client, query, qid=qid)
    except Exception as exc:  # network / timeout / 5xx — record and continue
        # Some transport errors (e.g. httpx timeouts) carry an empty str(exc);
        # always produce a non-empty, typed message so the row renders as ERROR.
        message = f"{type(exc).__name__}: {exc}".strip().rstrip(":").strip()
        return QueryResult(
            id=qid,
            category=category,
            query=query,
            cold_ms=0,
            cached_ms=0,
            parser_ms=None,
            recommend_ms=None,
            rerank_ms=None,
            total_ms_server=None,
            result_count=0,
            top10=[],
            relaxed=False,
            relaxation_path=[],
            cache_hit=False,
            hard_filter_ok=False,
            violations=[],
            error=message[:200],
        )

    data = cold_env.get("data") or {}
    results = data.get("results") or []
    relaxation_path = data.get("relaxation_path") or []
    top10 = [r.get("listing_id") for r in results[:TOP_N]]
    cached_data = cached_env.get("data") or {}
    cached_top10 = [r.get("listing_id") for r in (cached_data.get("results") or [])[:TOP_N]]

    fc = check_hard_filters(expect, results, relaxation_path)
    t = _timings(cold_env)
    cache_hit = bool(top10 and cached_top10 == top10 and cached_ms < cold_ms * CACHE_HIT_SPEEDUP)

    return QueryResult(
        id=qid,
        category=category,
        query=query,
        cold_ms=round(cold_ms, 1),
        cached_ms=round(cached_ms, 1),
        parser_ms=t.get("parser"),
        recommend_ms=t.get("recommend_initial", t.get("recommend")),
        rerank_ms=t.get("rerank"),
        total_ms_server=t.get("total"),
        result_count=len(results),
        top10=top10,
        relaxed=bool(data.get("relaxed")),
        relaxation_path=relaxation_path,
        cache_hit=cache_hit,
        hard_filter_ok=fc.ok,
        violations=fc.violations,
    )


# --- aggregation + reporting ------------------------------------------------


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


def _annotate_quality(results: list[QueryResult], baseline: dict[str, list[str]] | None) -> None:
    if not baseline:
        return
    for r in results:
        base_top = baseline.get(r.id)
        if base_top is None:
            r.quality_vs_baseline = "n/a"
        elif base_top[:TOP_N] == r.top10:
            r.quality_vs_baseline = "same"
        elif set(base_top[:TOP_N]) == set(r.top10):
            r.quality_vs_baseline = "reordered"
        else:
            r.quality_vs_baseline = "changed"


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.0f}"


def render_report(results: list[QueryResult]) -> str:
    lines: list[str] = []
    lines.append("# AI search latency + quality benchmark\n")
    lines.append(f"API base: `{API_BASE}` — {len(results)} queries\n")
    lines.append(
        "| # | Query | Cold ms | Cached ms | Parser ms | Recommend ms | Rerank ms "
        "| Results | Cache hit | Hard filter OK | Quality vs baseline |"
    )
    lines.append(
        "|---|-------|---------|-----------|-----------|--------------|-----------|---------|-----------|----------------|---------------------|"
    )
    for i, r in enumerate(results, 1):
        q = r.query if len(r.query) <= 48 else r.query[:45] + "..."
        if r.error:
            lines.append(f"| {i} | {q} | ERROR | | | | | | | | {r.error} |")
            continue
        hf = "yes" if r.hard_filter_ok else f"NO ({'; '.join(r.violations)})"
        lines.append(
            f"| {i} | {q} | {r.cold_ms:.0f} | {r.cached_ms:.0f} | {_fmt(r.parser_ms)} "
            f"| {_fmt(r.recommend_ms)} | {_fmt(r.rerank_ms)} | {r.result_count} "
            f"| {'yes' if r.cache_hit else 'no'} | {hf} | {r.quality_vs_baseline} |"
        )

    ok = [r for r in results if not r.error]
    cold = [r.cold_ms for r in ok]
    cached = [r.cached_ms for r in ok]
    violations = sum(1 for r in ok if not r.hard_filter_ok)
    regressions = sum(1 for r in ok if r.quality_vs_baseline == "changed")
    cache_hits = sum(1 for r in ok if r.cache_hit)
    relaxed = sum(1 for r in ok if r.relaxed)
    errors = len(results) - len(ok)

    lines.append("\n## Summary\n")
    lines.append(f"- queries run (no transport error): **{len(ok)}/{len(results)}**")
    lines.append(
        f"- cold latency ms — p50 **{_pct(cold, 0.5):.0f}**, p95 **{_pct(cold, 0.95):.0f}**, max **{max(cold) if cold else 0:.0f}**"
    )
    lines.append(
        f"- cached latency ms — p50 **{_pct(cached, 0.5):.0f}**, p95 **{_pct(cached, 0.95):.0f}**, max **{max(cached) if cached else 0:.0f}**"
    )
    lines.append(f"- hard-filter violation count (relaxation-aware): **{violations}**")
    lines.append(f"- quality regressions vs baseline (top-10 set changed): **{regressions}**")
    lines.append(f"- heuristic cache hits on repeat call: **{cache_hits}/{len(ok)}**")
    lines.append(f"- responses that relaxed at least one filter: **{relaxed}/{len(ok)}**")
    lines.append("\n### DoD gates")
    cold_max = max(cold) if cold else 0
    cached_p95 = _pct(cached, 0.95)
    lines.append(f"- [{'x' if errors == 0 else ' '}] all 50 queries completed (errors {errors})")
    lines.append(f"- [{'x' if cold_max <= 4000 else ' '}] cold max ≤ 4000 ms (actual {cold_max:.0f})")
    lines.append(f"- [{'x' if cached_p95 <= 1000 else ' '}] cached p95 ≤ 1000 ms (actual {cached_p95:.0f})")
    lines.append(f"- [{'x' if violations == 0 else ' '}] hard-filter violation count = 0 (actual {violations})")
    lines.append(f"- [{'x' if regressions == 0 else ' '}] quality regression count = 0 (actual {regressions})")
    note = (
        ""
        if any(r.parser_ms is not None for r in ok)
        else (
            "\n> Per-phase columns are blank because the API response does not expose "
            "`meta.timings_ms`. Have the cache/pipeline track add the per-phase `timings` "
            "dict to the search envelope meta to populate parser/recommend/rerank."
        )
    )
    lines.append(note)
    return "\n".join(lines)


def load_queries(path: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            specs.append(json.loads(line))
    return specs


async def main() -> None:
    parser = argparse.ArgumentParser(description="AI search latency + quality benchmark")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--out", type=Path, default=None, help="write the markdown report here")
    parser.add_argument("--json", type=Path, default=None, help="write machine-readable results here")
    parser.add_argument("--baseline", type=Path, default=None, help="prior --json run to diff top-10 against")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N queries (smoke)")
    args = parser.parse_args()

    specs = load_queries(args.queries)
    if args.limit:
        specs = specs[: args.limit]

    baseline: dict[str, list[str]] | None = None
    if args.baseline and args.baseline.exists():
        raw = json.loads(args.baseline.read_text(encoding="utf-8"))
        baseline = {r["id"]: r.get("top10", []) for r in raw.get("results", [])}

    results: list[QueryResult] = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for idx, spec in enumerate(specs, 1):
            r = await run_query(client, spec)
            results.append(r)
            status = (
                "ERR"
                if r.error
                else (f"{r.cold_ms:.0f}/{r.cached_ms:.0f}ms n={r.result_count} hf={'ok' if r.hard_filter_ok else 'NO'}")
            )
            print(f"[{idx}/{len(specs)}] {r.id} {status}")

    _annotate_quality(results, baseline)
    report = render_report(results)
    print("\n" + report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nReport written to {args.out}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"results": [asdict(r) for r in results]}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Results JSON written to {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
