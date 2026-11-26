"""Offline rerank comparison — metrics, adversarial injection, gates, report (fake arms)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

import pytest

from ingovo_search.eval.jev_rerank_adversarial import (
    ADVERSARIAL_KINDS,
    ADVERSARIAL_SUMMARIES,
    AdversarialHit,
    AdversarialSummary,
    adversarial_kind,
    inject_adversarial,
    is_adversarial,
    run_adversarial,
)
from ingovo_search.eval.jev_rerank_gates import evaluate_gates
from ingovo_search.eval.jev_rerank_offline import (
    DETERMINISTIC_ARM,
    GEMINI_ARM,
    ArmRun,
    ArmSummary,
    OfflineQuery,
    arm_significance,
    best_jev_arm,
    candidate_ndcg,
    paired_bootstrap_diff,
    percentile,
    run_offline,
    summarise_arm,
)
from ingovo_search.eval.jev_rerank_report import render_markdown
from ingovo_search.recommender.parser_schema import HardFilters, ParsedQuery
from ingovo_search.recommender.rerank_prompt import RerankCandidate


def _query(query_id: str, ids: Sequence[str], *, category: str | None = "soft_pref") -> OfflineQuery:
    return OfflineQuery(
        query_id=query_id,
        query_text=query_id,
        parsed=ParsedQuery(hard_filters=HardFilters(), semantic_query=query_id),
        candidates=tuple(RerankCandidate(listing_id=i, short_summary=f"Lakás {i}") for i in ids),
        deterministic_scores={i: 1.0 - index * 0.05 for index, i in enumerate(ids)},
        category=category,
    )


def _arm(
    name: str, orders: dict[str, Sequence[str]], *, tokens: int = 0, latency: float = 0.0, outcome: str = "success"
):
    async def run(query: OfflineQuery) -> ArmRun:
        order = tuple(orders.get(query.query_id, query.candidate_ids))
        return ArmRun(
            arm=name,
            query_id=query.query_id,
            ranked_ids=order,
            raw_ranked_ids=order,
            latency_ms=latency,
            input_tokens=tokens,
            cost_usd=tokens * 0.042 / 1_000_000,
            outcome=outcome,
        )

    return run


def _summary(arm: str, *, ndcg: float, top1: float = 4.0, tokens: int = 3000, cost: float = 0.0001) -> ArmSummary:
    return ArmSummary(arm, 2, ndcg, top1, ndcg, 300.0, 400.0, tokens, cost, 0, 0)


QUERIES = [_query("q1", ["a", "b", "c", "d"]), _query("q2", ["e", "f", "g", "h"], category=None)]
RELEVANCE = {"q1": {"a": 3.0, "b": 5.0, "c": 1.0, "z": 4.0}, "q2": {"e": 2.0, "f": 2.0, "g": 5.0}}


# --- metrics ---------------------------------------------------------------------


def test_candidate_ndcg_is_one_for_the_best_reorder_of_the_head() -> None:
    assert candidate_ndcg(["b", "a", "c", "d"], ["a", "b", "c", "d"], RELEVANCE["q1"]) == 1.0
    assert candidate_ndcg(["a", "b", "c", "d"], ["a", "b", "c", "d"], RELEVANCE["q1"]) < 1.0
    assert candidate_ndcg(["d"], ["d"], RELEVANCE["q1"]) == 0.0


def test_percentile_is_nearest_rank() -> None:
    assert percentile([40.0, 10.0, 30.0, 20.0], 0.5) == 20.0
    assert percentile([40.0, 10.0, 30.0, 20.0], 0.95) == 40.0
    assert percentile([7.0], 0.95) == 7.0
    assert percentile([], 0.5) == 0.0


async def test_run_offline_aggregates_each_arm_over_every_query() -> None:
    arms = {
        DETERMINISTIC_ARM: _arm(DETERMINISTIC_ARM, {}, outcome="deterministic"),
        GEMINI_ARM: _arm(
            GEMINI_ARM, {"q1": ["b", "a", "c", "d"], "q2": ["g", "e", "f", "h"]}, tokens=1500, latency=2800
        ),
        "jev:score4-hu:a1.0": _arm("jev:score4-hu:a1.0", {"q1": ["b", "a", "d", "c"]}, tokens=4000, latency=350),
    }
    seen: list[tuple[str, str]] = []

    report = await run_offline(QUERIES, RELEVANCE, arms, on_progress=lambda q, r: seen.append((q.query_id, r.arm)))

    assert [s.arm for s in report.summaries] == list(arms)
    deterministic, gemini, jev = report.summaries
    assert deterministic.query_count == gemini.query_count == jev.query_count == 2
    assert gemini.mean_ndcg_at_k > jev.mean_ndcg_at_k > deterministic.mean_ndcg_at_k
    assert gemini.mean_top1_relevance == 5.0
    assert (gemini.input_tokens_p50, gemini.latency_p50_ms, gemini.latency_p95_ms) == (1500, 2800.0, 2800.0)
    assert (deterministic.input_tokens_p50, deterministic.latency_p50_ms) == (0, 0.0)
    assert math.isclose(jev.mean_cost_usd, 4000 * 0.042 / 1_000_000)
    assert seen[:3] == [("q1", DETERMINISTIC_ARM), ("q1", GEMINI_ARM), ("q1", "jev:score4-hu:a1.0")]
    q1_metrics = [m for m in report.metrics if m.query_id == "q1"]
    assert all(m.rated_candidates == 3 for m in q1_metrics)


async def test_run_offline_rejects_an_arm_that_answers_for_another() -> None:
    with pytest.raises(ValueError, match="returned a run"):
        await run_offline(QUERIES, RELEVANCE, {"x": _arm("y", {})})


def test_summarise_arm_counts_fallbacks_and_cache_hits_and_ignores_cached_latency() -> None:
    runs = [
        ArmRun("j", "q1", ("a",), ("a",), latency_ms=300.0, input_tokens=3000),
        ArmRun("j", "q2", ("a",), ("a",), latency_ms=0.0, input_tokens=3400, from_cache=True),
        ArmRun("j", "q3", ("a",), ("a",), latency_ms=900.0, input_tokens=0, outcome="fallback"),
    ]

    summary = summarise_arm("j", runs, [])

    assert (summary.fallback_count, summary.cached_count) == (1, 1)
    assert summary.latency_p50_ms == 600.0
    assert summary.input_tokens_p50 == 3000


# --- adversarial ---------------------------------------------------------------


def test_inject_adversarial_places_rotating_summaries_at_the_slots() -> None:
    query = _query("q", [f"L{i}" for i in range(12)])

    injected = inject_adversarial(query, rotation=9)

    ids = injected.candidate_ids
    assert len(ids) == 15
    assert [ids[3], ids[7], ids[11]] == ["adv-9", "adv-0", "adv-1"]
    assert [i for i in ids if not is_adversarial(i)] == list(query.candidate_ids)
    assert injected.candidates[3].short_summary == ADVERSARIAL_SUMMARIES[9]
    assert injected.deterministic_scores == query.deterministic_scores


def test_inject_adversarial_appends_when_the_head_is_short() -> None:
    injected = inject_adversarial(_query("q", ["a", "b"]), rotation=0)

    assert injected.candidate_ids == ("a", "b", "adv-0", "adv-1", "adv-2")


async def test_run_adversarial_reports_hits_in_the_raw_top_3_only() -> None:
    async def steered(query: OfflineQuery) -> ArmRun:
        ids = query.candidate_ids
        raw = (ids[3], *[i for i in ids if i != ids[3]])
        return ArmRun("jev:weak", query.query_id, raw_ranked_ids=raw, ranked_ids=ids)

    async def steady(query: OfflineQuery) -> ArmRun:
        return ArmRun("jev:strong", query.query_id, query.candidate_ids, query.candidate_ids)

    weak, strong = await run_adversarial(
        [_query("q1", [f"L{i}" for i in range(12)]), _query("tiny", ["a", "b"])],
        {"jev:weak": steered, "jev:strong": steady},
    )

    assert weak.hits == (AdversarialHit("jev:weak", "q1", "adv-0", 1),)
    assert weak.hit_count == 1
    assert strong.hits == ()
    # The two-candidate head is skipped: three injected listings would fill its top 3 by arithmetic.
    assert (strong.query_count, strong.top_n) == (1, 3)


# --- gates -----------------------------------------------------------------------


def test_gates_pass_when_every_threshold_is_met() -> None:
    summaries = [
        _summary(DETERMINISTIC_ARM, ndcg=0.80, top1=3.5),
        _summary(GEMINI_ARM, ndcg=0.85, top1=4.0, tokens=1500, cost=0.0003),
        _summary("jev:x:a1.0", ndcg=0.845, top1=3.95, tokens=3100, cost=0.00013),
    ]
    clean = AdversarialSummary("jev:x:a1.0", 20, 3, ())

    report = evaluate_gates(summaries, jev_arm="jev:x:a1.0", adversarial=clean)

    assert report.passed
    assert [check.gate for check in report.checks] == ["G1 (a)", "G1 (b)", "G1 (c)", "G1 (d)", "G2 token", "G2 cost"]
    assert math.isclose(report.checks[-1].threshold, 0.0003 * 0.6)


def test_gates_fail_on_tokens_deterministic_margin_and_adversarial_hits() -> None:
    summaries = [
        _summary(DETERMINISTIC_ARM, ndcg=0.84),
        _summary(GEMINI_ARM, ndcg=0.85, cost=0.0003),
        _summary("jev:x:a1.0", ndcg=0.85, tokens=4400, cost=0.00018),
    ]
    hit = AdversarialSummary("jev:x:a1.0", 20, 3, (AdversarialHit("jev:x:a1.0", "q", "adv-1", 2),))

    report = evaluate_gates(summaries, jev_arm="jev:x:a1.0", adversarial=hit, gemini_cost_per_rerank_usd=0.0004)

    failed = {check.gate for check in report.checks if not check.passed}
    assert failed == {"G1 (c)", "G1 (d)", "G2 token"}
    assert not report.passed
    cost = report.checks[-1]
    assert math.isclose(cost.threshold, 0.0004 * 0.6)


def test_gates_need_the_reference_arms() -> None:
    with pytest.raises(ValueError, match="no summary"):
        evaluate_gates([_summary("jev:x:a1.0", ndcg=0.8)], jev_arm="jev:x:a1.0")


def test_best_jev_arm_prefers_quality_then_fewer_tokens() -> None:
    summaries = [
        _summary(GEMINI_ARM, ndcg=0.99),
        _summary("jev:a", ndcg=0.85, tokens=4000),
        _summary("jev:b", ndcg=0.85, tokens=3000),
        _summary("jev:c", ndcg=0.84, tokens=2000),
    ]

    best = best_jev_arm(summaries)

    assert best is not None
    assert best.arm == "jev:b"
    assert best_jev_arm([_summary(GEMINI_ARM, ndcg=0.9)]) is None


# --- report ----------------------------------------------------------------------


async def test_markdown_report_lists_arms_gates_adversarial_and_queries() -> None:
    arms = {
        DETERMINISTIC_ARM: _arm(DETERMINISTIC_ARM, {}, outcome="deterministic"),
        GEMINI_ARM: _arm(GEMINI_ARM, {"q1": ["b", "a", "c", "d"]}, tokens=1500),
        "jev:score4-hu:a1.0": _arm("jev:score4-hu:a1.0", {"q1": ["b", "a", "d", "c"]}, tokens=4000),
    }
    report = await run_offline(QUERIES, RELEVANCE, arms)
    gates = evaluate_gates(report.summaries, jev_arm="jev:score4-hu:a1.0")
    adversarial = [
        AdversarialSummary("jev:score4-hu:a1.0", 2, 3, (AdversarialHit("jev:score4-hu:a1.0", "q1", "adv-4", 2),))
    ]

    text = render_markdown(
        report,
        queries=QUERIES,
        adversarial=adversarial,
        gates=gates,
        meta={"Rater": "gemini-judge-v1"},
        report_date=date(2026, 9, 28),
    )

    assert text.startswith("# Offline rerank comparison — 2026-09-28")
    assert "- **Rater:** gemini-judge-v1" in text
    assert "| `gemini` |" in text
    assert "| `jev:score4-hu:a1.0` |" in text
    assert "| G2 token |" in text
    assert "adv-4 #2" in text
    assert "| q1 | soft_pref | 3/4 |" in text
    assert "| q2 | - | 3/4 |" in text


# --- amended G1 (d), significance -------------------------------------------------


def test_adversarial_kinds_mark_the_keyword_list_only() -> None:
    assert len(ADVERSARIAL_KINDS) == len(ADVERSARIAL_SUMMARIES)
    assert adversarial_kind("adv-3") == "keyword"
    assert {adversarial_kind(f"adv-{i}") for i in range(10) if i != 3} == {"instruction"}
    hits = (
        AdversarialHit("j", "q", "adv-3", 1),
        AdversarialHit("j", "q", "adv-0", 2),
        AdversarialHit("j", "q", "adv-9", 3),
    )
    summary = AdversarialSummary("j", 1, 3, hits)
    assert (summary.instruction_hit_count, summary.keyword_hit_count) == (2, 1)


def test_amended_d_compares_with_gemini_on_the_same_injected_heads() -> None:
    summaries = [
        _summary(DETERMINISTIC_ARM, ndcg=0.80),
        _summary(GEMINI_ARM, ndcg=0.85, cost=0.0006),
        _summary("jev:x:a1.0", ndcg=0.85, tokens=3100, cost=0.00013),
    ]
    jev = AdversarialSummary(
        "jev:x:a1.0",
        59,
        3,
        (AdversarialHit("jev:x:a1.0", "q", "adv-3", 1), AdversarialHit("jev:x:a1.0", "q", "adv-1", 2)),
    )
    gemini_hits = tuple(AdversarialHit(GEMINI_ARM, "q", f"adv-{i}", 2) for i in (0, 1, 3))
    gemini = AdversarialSummary(GEMINI_ARM, 59, 3, gemini_hits)

    report = evaluate_gates(summaries, jev_arm="jev:x:a1.0", adversarial=jev, reference_adversarial=gemini)

    checks = {check.gate: check for check in report.checks}
    assert "G1 (d)" not in checks
    assert (checks["G1 (d) instruction"].measured, checks["G1 (d) instruction"].threshold) == (1.0, 2.0)
    assert (checks["G1 (d) all"].measured, checks["G1 (d) all"].threshold) == (2.0, 3.0)
    assert report.passed

    extra = (AdversarialHit("jev:x:a1.0", "q", "adv-5", 1), AdversarialHit("jev:x:a1.0", "q", "adv-6", 1))
    worse = AdversarialSummary("jev:x:a1.0", 59, 3, (*jev.hits, *extra))
    failed = evaluate_gates(summaries, jev_arm="jev:x:a1.0", adversarial=worse, reference_adversarial=gemini)
    assert {check.gate for check in failed.checks if not check.passed} == {"G1 (d) instruction", "G1 (d) all"}


def test_paired_bootstrap_brackets_the_mean_and_is_reproducible() -> None:
    first = [0.9, 0.8, 0.7, 0.95, 0.6, 0.85]
    second = [0.8, 0.8, 0.6, 0.9, 0.65, 0.8]

    mean, low, high = paired_bootstrap_diff(first, second, iterations=2000, seed=1)

    assert math.isclose(mean, sum(a - b for a, b in zip(first, second, strict=True)) / 6)
    assert low <= mean <= high
    assert paired_bootstrap_diff(first, second, iterations=2000, seed=1) == (mean, low, high)
    assert paired_bootstrap_diff([0.5] * 4, [0.5] * 4, iterations=100) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_diff([1.0], [])


async def test_arm_significance_pairs_queries_and_renders() -> None:
    arms = {
        DETERMINISTIC_ARM: _arm(DETERMINISTIC_ARM, {}, outcome="deterministic"),
        GEMINI_ARM: _arm(GEMINI_ARM, {"q1": ["b", "a", "c", "d"], "q2": ["g", "e", "f", "h"]}, tokens=1500),
    }
    report = await run_offline(QUERIES, RELEVANCE, arms)

    significance = arm_significance(report, GEMINI_ARM, DETERMINISTIC_ARM)
    text = render_markdown(report, queries=QUERIES, significance=[significance], report_date=date(2026, 9, 28))

    assert significance.mean > 0
    assert significance.low <= significance.mean <= significance.high
    assert "## 95% intervals of the differences" in text
    assert f"| `{GEMINI_ARM} vs {DETERMINISTIC_ARM}` | ndcg_at_k |" in text
