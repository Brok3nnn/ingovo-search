"""Markdown report of the offline rerank comparison.

The Hungarian report the live runner writes to
`eval/out/jev_rerank_offline_<date>.md`: the per-arm table, the adversarial
set, the bootstrap intervals, the gate verdicts and the per-query nDCG table,
one section renderer each.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

from ingovo_search.eval.jev_rerank_adversarial import AdversarialSummary
from ingovo_search.eval.jev_rerank_gates import GateReport
from ingovo_search.eval.jev_rerank_offline import (
    DETERMINISTIC_ARM,
    GEMINI_ARM,
    OfflineQuery,
    OfflineReport,
    QueryMetrics,
    Significance,
)


def _status(passed: bool) -> str:
    return "pass" if passed else "FAIL"


def render_markdown(  # noqa: PLR0913 -- the run plus one keyword per optional section of the report
    report: OfflineReport,
    *,
    queries: Sequence[OfflineQuery],
    adversarial: Sequence[AdversarialSummary] = (),
    gates: GateReport | None = None,
    significance: Sequence[Significance] = (),
    meta: Mapping[str, object] | None = None,
    report_date: date | None = None,
) -> str:
    """The Markdown report (`eval/out/jev_rerank_offline_<date>.md`)."""
    lines = _title_lines(meta, report_date)
    lines += _arm_lines(report)
    if adversarial:
        lines += _adversarial_lines(adversarial)
    if significance:
        lines += _significance_lines(significance)
    if gates is not None:
        lines += _gate_lines(gates)
    lines += _query_lines(report, queries, gates)
    return "\n".join(lines)


def _title_lines(meta: Mapping[str, object] | None, report_date: date | None) -> list[str]:
    """The dated title and the run's facts as a bullet list."""
    lines = [f"# Offline rerank comparison — {(report_date or date.today()).isoformat()}", ""]  # noqa: DTZ011 -- the report is dated by the operator's calendar day, like its file name
    for key, value in (meta or {}).items():
        lines.append(f"- **{key}:** {value}")
    if meta:
        lines.append("")
    return lines


def _arm_lines(report: OfflineReport) -> list[str]:
    """One row per arm, best nDCG@10 first."""
    rows = [
        f"| `{summary.arm}` | {summary.mean_ndcg_at_k:.4f} | {summary.mean_top1_relevance:.2f} | "
        f"{summary.mean_candidate_ndcg:.4f} | {summary.latency_p50_ms:.0f} ms | {summary.latency_p95_ms:.0f} ms | "
        f"{summary.input_tokens_p50} | {summary.mean_cost_usd:.6f} | {summary.fallback_count} |"
        for summary in sorted(report.summaries, key=lambda s: s.mean_ndcg_at_k, reverse=True)
    ]
    return [
        "## By arm (nDCG@10, descending)",
        "",
        "| Arm | nDCG@10 | top-1 grade | candidate nDCG@10 | latency p50 | latency p95 | tokens p50 | $/rerank | fallbacks |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]


def _adversarial_lines(adversarial: Sequence[AdversarialSummary]) -> list[str]:
    """One row per arm on the injected heads, with every hit listed."""
    return [
        "## Adversarial set (G1 d)",
        "",
        "| Arm | queries | adversarial in top 3 | of which instruction / keyword | at rank 1 | cases |",
        "|---|---|---|---|---|---|",
        *(_adversarial_row(summary) for summary in adversarial),
        "",
    ]


def _adversarial_row(summary: AdversarialSummary) -> str:
    cases = "; ".join(f"{hit.query_id[:30]}… → {hit.adversarial_id} #{hit.position}" for hit in summary.hits)
    first = sum(1 for hit in summary.hits if hit.position == 1)
    return (
        f"| `{summary.arm}` | {summary.query_count} | {summary.hit_count} | "
        f"{summary.instruction_hit_count} / {summary.keyword_hit_count} | {first} | {cases or '-'} |"
    )


def _significance_lines(significance: Sequence[Significance]) -> list[str]:
    """The paired-bootstrap intervals, one row per comparison."""
    rows = [
        f"| `{item.label}` | {item.metric} | {item.mean:+.4f} | [{item.low:+.4f}, {item.high:+.4f}] |"
        for item in significance
    ]
    return [
        "## 95% intervals of the differences (paired bootstrap, 10,000 resamples)",
        "",
        "| Comparison | metric | mean | 95% interval |",
        "|---|---|---|---|",
        *rows,
        "",
    ]


def _gate_lines(gates: GateReport) -> list[str]:
    """The gate verdicts of the arm under test."""
    rows = [
        f"| {check.gate} | {check.description} | {check.measured:.4f} | {check.threshold:.4f} | "
        f"{_status(check.passed)} |"
        for check in gates.checks
    ]
    return [
        f"## Gates — `{gates.arm}` ({'all pass' if gates.passed else 'not met'})",
        "",
        "| Gate | Criterion | Measured | Threshold | Status |",
        "|---|---|---|---|---|",
        *rows,
        "",
    ]


def _query_lines(report: OfflineReport, queries: Sequence[OfflineQuery], gates: GateReport | None) -> list[str]:
    """Per-query nDCG@10 of the deterministic, Gemini and gated arms."""
    shown = [DETERMINISTIC_ARM, GEMINI_ARM]
    if gates is not None:
        shown.append(gates.arm)
    by_key = {(m.arm, m.query_id): m for m in report.metrics}
    return [
        "## By query (nDCG@10)",
        "",
        "| Query | category | rated candidates | " + " | ".join(f"`{arm}`" for arm in shown) + " |",
        "|---|---|---|" + "---|" * len(shown),
        *(_query_row(query, shown, by_key) for query in queries),
        "",
    ]


def _query_row(query: OfflineQuery, shown: Sequence[str], by_key: Mapping[tuple[str, str], QueryMetrics]) -> str:
    cells: list[str] = []
    rated = 0
    for arm in shown:
        metric = by_key.get((arm, query.query_id))
        cells.append(f"{metric.ndcg_at_k:.3f}" if metric is not None else "-")
        if metric is not None:
            rated = metric.rated_candidates
    return (
        f"| {query.query_text[:60]} | {query.category or '-'} | {rated}/{len(query.candidates)} | "
        + " | ".join(cells)
        + " |"
    )
