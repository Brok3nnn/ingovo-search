"""Battery runner — end-to-end ranking regression over the eval set.

For every record in `eval_set.starter.jsonl`, build the ParsedQuery directly
from the record's `expected` block (deterministic — no LLM, no quota), run the
full retrieval + scoring pipeline (`recommend`) against the live database, and
check three things:

1. **Hard-filter integrity** — an independent verifier re-checks every
   returned listing against the query's hard filters; any violation fails the
   run (the guardrail invariant: hard filters are absolute).
2. **Golden top-10 snapshots** — result IDs per query are compared against
   `eval/golden/battery_top10.json`; ranking drift fails the run unless
   `--update` re-baselines. This is how scoring changes show their blast
   radius as a reviewable diff instead of a production surprise.
3. **Latency** — per-query wall time, reported p50/p95.

Embeddings use the deterministic `MockEmbeddingClient`: content/factual
similarity terms are stable but semantically meaningless, which is fine — the
battery guards filters, deterministic scoring terms (anchor, softpref, geo,
POI), and ranking stability; semantic quality is the rating loop's job.
The exploration injection (`explore` defaults to `random.random()`) is
zeroed via the weights override — production keeps its jitter, snapshots
must not.

Run from the repository root (DB env vars + venv):
    uv run python eval/run_battery.py            # check
    uv run python eval/run_battery.py --update   # re-baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = REPO_ROOT / "eval" / "golden" / "eval_set.starter.jsonl"
GOLDEN_PATH = REPO_ROOT / "eval" / "golden" / "battery_top10.json"
REPORT_PATH = REPO_ROOT / "eval" / "out" / "battery_report.md"


from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env", override=False)

from ingovo_search.db.session import get_async_session_factory
from ingovo_search.lib.embeddings import MockEmbeddingClient
from ingovo_search.lib.llm import LlmRequest
from ingovo_search.recommender.parser_schema import HardFilters, ParsedQuery
from ingovo_search.recommender.recommend import recommend
from ingovo_search.recommender.score import WEIGHTS
from ingovo_search.recommender.sql_filter import district_filter_values


class _EmptyRegionHintClient:
    """LlmClient stub: cold-start region hints resolve to 'no districts known'."""

    async def complete(self, req: LlmRequest) -> str:
        return '{"districts": []}'


def load_records(path: Path = EVAL_SET_PATH) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_query(expected: dict[str, Any]) -> ParsedQuery:
    """The eval record's `expected` block IS a partial ParsedQuery."""
    return ParsedQuery.model_validate(expected)


def _rooms_int(value: Any) -> int | None:
    text = str(value or "").strip()
    return int(text) if text.isdigit() else None


def hard_filter_violations(hard: HardFilters, listing: Any) -> list[str]:
    """Independent re-check of the hard-filter contract on one returned listing.

    Mirrors `sql_filter` semantics: categorical/boolean constraints tolerate
    NULL listing values (missing data stays eligible); numeric bounds do not.
    """
    violations: list[str] = []

    if hard.transaction_type is not None:
        actual = getattr(listing, "transaction_type", None)
        if actual is not None and str(actual).lower() != hard.transaction_type:
            violations.append(f"transaction_type={actual!r} != {hard.transaction_type!r}")

    if hard.property_type_in:
        actual = getattr(listing, "property_type", None)
        if actual is not None and actual not in hard.property_type_in:
            violations.append(f"property_type={actual!r} not in {hard.property_type_in}")

    if hard.property_type_exclude:
        actual = getattr(listing, "property_type", None)
        if actual is not None and actual in hard.property_type_exclude:
            violations.append(f"property_type={actual!r} is excluded")

    if hard.city is not None:
        actual = getattr(listing, "city", None)
        if actual is not None and str(actual).lower() != hard.city.lower():
            violations.append(f"city={actual!r} != {hard.city!r}")

    if hard.district_in:
        actual = getattr(listing, "district", None)
        if actual is not None and str(actual).lower() not in district_filter_values(hard.district_in):
            violations.append(f"district={actual!r} not in {hard.district_in}")

    price = getattr(listing, "price", None)
    if hard.price_huf_max is not None and (price is None or price > hard.price_huf_max):
        violations.append(f"price={price!r} > max {hard.price_huf_max}")
    if hard.price_huf_min is not None and (price is None or price < hard.price_huf_min):
        violations.append(f"price={price!r} < min {hard.price_huf_min}")

    area = getattr(listing, "area_m2", None)
    if hard.area_m2_min is not None and (area is None or area < hard.area_m2_min):
        violations.append(f"area_m2={area!r} < min {hard.area_m2_min}")
    if hard.area_m2_max is not None and (area is None or area > hard.area_m2_max):
        violations.append(f"area_m2={area!r} > max {hard.area_m2_max}")

    rooms = _rooms_int(getattr(listing, "rooms", None))
    if hard.rooms is not None and rooms is not None and rooms != hard.rooms:
        violations.append(f"rooms={rooms} != {hard.rooms}")
    if hard.rooms_min is not None and (rooms is None or rooms < hard.rooms_min):
        violations.append(f"rooms={rooms!r} < min {hard.rooms_min}")

    return violations


def diff_snapshots(golden: dict[str, list[str]], current: dict[str, list[str]]) -> list[str]:
    """Human-readable drift lines; empty when current matches golden."""
    drift: list[str] = []
    for query, ids in current.items():
        if query not in golden:
            drift.append(f"NEW query (no golden yet): {query!r}")
        elif golden[query] != ids:
            drift.append(f"DRIFT {query!r}: golden={golden[query]} current={ids}")
    for query in golden:
        if query not in current:
            drift.append(f"REMOVED query (golden is stale): {query!r}")
    return drift


async def run_battery(update_golden: bool) -> int:
    records = load_records()
    embedder = MockEmbeddingClient()
    client = _EmptyRegionHintClient()

    deterministic_weights = {**WEIGHTS, "explore": 0.0}
    snapshot: dict[str, list[str]] = {}
    all_violations: list[str] = []
    latencies_ms: list[float] = []
    result_counts: dict[str, int] = {}
    report_rows: list[str] = []

    async with get_async_session_factory()() as session:
        for record in records:
            query_text = record["query"]
            parsed = build_query(record["expected"])
            query_emb = await embedder.embed(query_text, "search_query")

            started = time.perf_counter()
            results = await recommend(
                session,
                parsed,
                query_emb=query_emb,
                client=client,  # type: ignore[arg-type]
                top_n=10,
                weights=deterministic_weights,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            latencies_ms.append(elapsed_ms)

            top_ids = [prop.id for prop, _breakdown in results]
            snapshot[query_text] = top_ids
            result_counts[query_text] = len(results)

            for prop, _breakdown in results:
                for violation in hard_filter_violations(parsed.hard_filters, prop):
                    all_violations.append(f"{query_text!r} -> listing {prop.id}: {violation}")

            report_rows.append(
                f"| {record.get('category', '—')} | {query_text[:60]} | {len(results)} | {elapsed_ms:.0f} ms |"
            )

    p50 = statistics.median(latencies_ms)
    p95 = sorted(latencies_ms)[max(0, int(len(latencies_ms) * 0.95) - 1)]
    empty = sum(1 for n in result_counts.values() if n == 0)

    golden: dict[str, list[str]] = {}
    if GOLDEN_PATH.exists():
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    drift = diff_snapshots(golden, snapshot)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Battery report\n\n"
        f"- queries: {len(records)}  |  empty result sets: {empty}\n"
        f"- latency p50: {p50:.0f} ms  |  p95: {p95:.0f} ms\n"
        f"- hard-filter violations: {len(all_violations)}\n"
        f"- snapshot drift entries: {len(drift)}\n\n"
        "| category | query | results | latency |\n|---|---|---|---|\n" + "\n".join(report_rows) + "\n",
        encoding="utf-8",
    )

    print(f"queries: {len(records)}  empty: {empty}  p50: {p50:.0f} ms  p95: {p95:.0f} ms")

    if all_violations:
        print(f"\nFAIL: {len(all_violations)} hard-filter violations:")
        for v in all_violations[:20]:
            print(f"  {v}")
        return 1

    if update_golden:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"golden snapshot written: {GOLDEN_PATH} ({len(snapshot)} queries)")
        return 0

    if drift:
        print(f"\nFAIL: snapshot drift in {len(drift)} entries; review, then re-baseline with --update:")
        for d in drift[:20]:
            print(f"  {d}")
        return 1

    print("OK: no violations, no drift")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="write the golden snapshot")
    args = parser.parse_args()
    sys.exit(asyncio.run(run_battery(update_golden=args.update)))


if __name__ == "__main__":
    main()
