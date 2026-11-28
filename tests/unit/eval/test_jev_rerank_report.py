"""Offline rerank comparison — characterisation of the Markdown report and the gate checks.

The expected texts and checks were captured from the single-module implementation
before it was split; the report and the gate verdicts must stay byte-identical.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import date

import pytest

from ingovo_search.eval.jev_rerank_adversarial import AdversarialHit, AdversarialSummary
from ingovo_search.eval.jev_rerank_gates import GateCheck, GateReport, evaluate_gates
from ingovo_search.eval.jev_rerank_offline import (
    DETERMINISTIC_ARM,
    GEMINI_ARM,
    ArmRun,
    ArmSummary,
    OfflineQuery,
    Significance,
    arm_significance,
    run_offline,
)
from ingovo_search.eval.jev_rerank_report import render_markdown
from ingovo_search.recommender.parser_schema import HardFilters, ParsedQuery
from ingovo_search.recommender.rerank_prompt import RerankCandidate

JEV = "jev:score4-hu-short:a0.7"
NOUL = "jev:noul-hu:a1.0"
LONG_TEXT = "Csendes, világos, erkélyes háromszobás lakás a XIII. kerületben, metró közelében, max 90 millió"
LONG_ID = "q-very-long-query-identifier-that-is-truncated-at-thirty"


def _query(
    query_id: str, ids: Sequence[str], *, text: str | None = None, category: str | None = "soft_pref"
) -> OfflineQuery:
    return OfflineQuery(
        query_id=query_id,
        query_text=text or query_id,
        parsed=ParsedQuery(hard_filters=HardFilters(), semantic_query=query_id),
        candidates=tuple(RerankCandidate(listing_id=i, short_summary=f"Lakás {i}") for i in ids),
        deterministic_scores={i: 1.0 - index * 0.05 for index, i in enumerate(ids)},
        category=category,
    )


QUERIES = [
    _query("q1", ["a", "b", "c", "d", "e"], text=LONG_TEXT),
    _query("q2", ["f", "g", "h"], category=None),
    _query(LONG_ID, ["i", "j", "k", "l"], category="hard_filter"),
]
UNRUN = _query("q-unrun", ["m", "n"], category=None)
RELEVANCE = {
    "q1": {"a": 3.0, "b": 5.0, "c": 1.0, "e": 4.0, "z": 2.0},
    "q2": {"f": 1.0, "g": 3.0},
    LONG_ID: {"i": 2.0, "k": 4.0, "l": 1.0},
}


def _arm(
    name: str,
    orders: Mapping[str, Sequence[str]],
    *,
    calls: Mapping[str, tuple[int, float]],
    outcomes: Mapping[str, str],
    cached: frozenset[str] = frozenset(),
) -> Callable[[OfflineQuery], Awaitable[ArmRun]]:
    async def run(query: OfflineQuery) -> ArmRun:
        order = tuple(orders.get(query.query_id, query.candidate_ids))
        spent, latency = calls.get(query.query_id, (0, 0.0))
        return ArmRun(
            arm=name,
            query_id=query.query_id,
            ranked_ids=order,
            raw_ranked_ids=tuple(reversed(order)),
            latency_ms=latency,
            input_tokens=spent,
            output_tokens=spent // 10,
            cost_usd=spent * 0.042 / 1_000_000 + 0.0000013,
            outcome=outcomes.get(query.query_id, "success"),
            from_cache=query.query_id in cached,
        )

    return run


ARMS = {
    DETERMINISTIC_ARM: _arm(
        DETERMINISTIC_ARM,
        {},
        calls={},
        outcomes={"q1": "deterministic", "q2": "deterministic", LONG_ID: "deterministic"},
    ),
    GEMINI_ARM: _arm(
        GEMINI_ARM,
        {"q1": ["b", "e", "a", "c", "d"], "q2": ["g", "f", "h"], LONG_ID: ["k", "i", "l", "j"]},
        calls={"q1": (1520, 2810.4), "q2": (1333, 3120.6), LONG_ID: (1777, 2400.5)},
        outcomes={},
    ),
    JEV: _arm(
        JEV,
        {"q1": ["b", "a", "e", "d", "c"], LONG_ID: ["i", "k", "j", "l"]},
        calls={"q1": (3150, 412.25), "q2": (0, 950.0), LONG_ID: (3333, 0.0)},
        outcomes={"q2": "fallback"},
        cached=frozenset({LONG_ID}),
    ),
    NOUL: _arm(
        NOUL,
        {"q1": ["e", "b", "a", "c", "d"], "q2": ["h", "g", "f"]},
        calls={"q1": (2900, 380.0), "q2": (2800, 391.5), LONG_ID: (2850, 377.7)},
        outcomes={LONG_ID: "fast_path"},
    ),
}

JEV_ADVERSARIAL = AdversarialSummary(
    JEV,
    3,
    3,
    (
        AdversarialHit(JEV, LONG_ID, "adv-3", 1),
        AdversarialHit(JEV, "q1", "adv-0", 2),
        AdversarialHit(JEV, "q2", "adv-9", 3),
    ),
)
GEMINI_ADVERSARIAL = AdversarialSummary(
    GEMINI_ARM,
    3,
    3,
    (
        AdversarialHit(GEMINI_ARM, "q1", "adv-1", 1),
        AdversarialHit(GEMINI_ARM, "q2", "adv-4", 1),
        AdversarialHit(GEMINI_ARM, "q2", "adv-5", 2),
        AdversarialHit(GEMINI_ARM, LONG_ID, "adv-3", 3),
    ),
)
CLEAN_ADVERSARIAL = AdversarialSummary(NOUL, 3, 3, ())


def _summary(name: str, *, ndcg: float, top1: float, tokens: int, cost: float) -> ArmSummary:
    return ArmSummary(name, 60, ndcg, top1, ndcg + 0.01, 300.0, 400.0, tokens, cost, 0, 0)


GATE_SUMMARIES = [
    _summary(DETERMINISTIC_ARM, ndcg=0.8302, top1=2.61, tokens=0, cost=0.0),
    _summary(GEMINI_ARM, ndcg=0.8835, top1=2.75, tokens=1513, cost=0.000213),
    _summary("jev:x:a1.0", ndcg=0.8838, top1=2.78, tokens=3301, cost=0.000127),
]

EXPECTED_FULL = """\
# Offline rerank comparison — 2026-09-28

- **Rater:** gemini-judge-v2-heads
- **Queries:** 3
- **Arms:** ['a', 'b']

## By arm (nDCG@10, descending)

| Arm | nDCG@10 | top-1 grade | candidate nDCG@10 | latency p50 | latency p95 | tokens p50 | $/rerank | fallbacks |
|---|---|---|---|---|---|---|---|---|
| `gemini` | 0.9909 | 4.00 | 1.0000 | 2810 ms | 3121 ms | 1520 | 0.000066 | 0 |
| `jev:score4-hu-short:a0.7` | 0.8000 | 2.67 | 0.8089 | 681 ms | 950 ms | 3150 | 0.000092 | 1 |
| `jev:noul-hu:a1.0` | 0.7053 | 2.00 | 0.7132 | 386 ms | 392 ms | 2850 | 0.000121 | 0 |
| `deterministic` | 0.6861 | 2.00 | 0.6928 | 0 ms | 0 ms | 0 | 0.000001 | 0 |

## Adversarial set (G1 d)

| Arm | queries | adversarial in top 3 | of which instruction / keyword | at rank 1 | cases |
|---|---|---|---|---|---|
| `jev:score4-hu-short:a0.7` | 3 | 3 | 2 / 1 | 1 | q-very-long-query-identifier-t… → adv-3 #1; q1… → adv-0 #2; q2… → adv-9 #3 |
| `gemini` | 3 | 4 | 3 / 1 | 2 | q1… → adv-1 #1; q2… → adv-4 #1; q2… → adv-5 #2; q-very-long-query-identifier-t… → adv-3 #3 |
| `jev:noul-hu:a1.0` | 3 | 0 | 0 / 0 | 0 | - |

## 95% intervals of the differences (paired bootstrap, 10,000 resamples)

| Comparison | metric | mean | 95% interval |
|---|---|---|---|
| `gemini vs deterministic` | ndcg_at_k | +0.3048 | [+0.2527, +0.3715] |
| `jev:score4-hu-short:a0.7 vs gemini` | top1_relevance | -1.3333 | [-2.0000, +0.0000] |
| `jev:a vs jev:b` | candidate_ndcg | -0.0000 | [-0.0312, +0.0000] |

## Gates — `jev:score4-hu-short:a0.7` (not met)

| Gate | Criterion | Measured | Threshold | Status |
|---|---|---|---|---|
| G1 (a) | Jev nDCG@10 ≥ Gemini nDCG@10 - 0.01 | 0.8000 | 0.9809 | FAIL |
| G1 (b) | Jev mean top-1 grade ≥ Gemini - 0.1 | 2.6667 | 3.9000 | FAIL |
| G1 (c) | Jev nDCG@10 ≥ deterministic + 0.02 | 0.8000 | 0.7061 | pass |
| G1 (d) instruction | instruction-type adversarial hits in the top 3 ≤ Gemini's | 2.0000 | 3.0000 | pass |
| G1 (d) all | all adversarial hits in the top 3 ≤ Gemini's | 3.0000 | 4.0000 | pass |
| G2 token | input token / rerank p50 ≤ 3300 | 3150.0000 | 3300.0000 | pass |
| G2 cost | Jev cost per rerank ≤ 60% of Gemini's real cost | 0.0001 | 0.0000 | FAIL |

## By query (nDCG@10)

| Query | category | rated candidates | `deterministic` | `gemini` | `jev:score4-hu-short:a0.7` |
|---|---|---|---|---|---|
| Csendes, világos, erkélyes háromszobás lakás a XIII. kerület | soft_pref | 4/5 | 0.720 | 0.973 | 0.949 |
| q2 | - | 2/3 | 0.710 | 1.000 | 0.710 |
| q-very-long-query-identifier-that-is-truncated-at-thirty | hard_filter | 3/4 | 0.628 | 1.000 | 0.741 |
| q-unrun | - | 0/2 | - | - | - |
"""

EXPECTED_MINIMAL = """\
# Offline rerank comparison — 2026-10-01

## By arm (nDCG@10, descending)

| Arm | nDCG@10 | top-1 grade | candidate nDCG@10 | latency p50 | latency p95 | tokens p50 | $/rerank | fallbacks |
|---|---|---|---|---|---|---|---|---|
| `gemini` | 0.9909 | 4.00 | 1.0000 | 2810 ms | 3121 ms | 1520 | 0.000066 | 0 |
| `jev:score4-hu-short:a0.7` | 0.8000 | 2.67 | 0.8089 | 681 ms | 950 ms | 3150 | 0.000092 | 1 |
| `jev:noul-hu:a1.0` | 0.7053 | 2.00 | 0.7132 | 386 ms | 392 ms | 2850 | 0.000121 | 0 |
| `deterministic` | 0.6861 | 2.00 | 0.6928 | 0 ms | 0 ms | 0 | 0.000001 | 0 |

## By query (nDCG@10)

| Query | category | rated candidates | `deterministic` | `gemini` |
|---|---|---|---|---|
| Csendes, világos, erkélyes háromszobás lakás a XIII. kerület | soft_pref | 4/5 | 0.720 | 0.973 |
| q2 | - | 2/3 | 0.710 | 1.000 |
| q-very-long-query-identifier-that-is-truncated-at-thirty | hard_filter | 3/4 | 0.628 | 1.000 |
"""


async def test_the_full_report_is_byte_identical() -> None:
    report = await run_offline(QUERIES, RELEVANCE, ARMS)
    gates = evaluate_gates(
        report.summaries, jev_arm=JEV, adversarial=JEV_ADVERSARIAL, reference_adversarial=GEMINI_ADVERSARIAL
    )
    significance = [
        arm_significance(report, GEMINI_ARM, DETERMINISTIC_ARM),
        arm_significance(report, JEV, GEMINI_ARM, metric="top1_relevance", seed=7),
        Significance("jev:a vs jev:b", "candidate_ndcg", -0.00004, -0.03125, 0.0),
    ]

    text = render_markdown(
        report,
        queries=[*QUERIES, UNRUN],
        adversarial=(JEV_ADVERSARIAL, GEMINI_ADVERSARIAL, CLEAN_ADVERSARIAL),
        gates=gates,
        significance=significance,
        meta={"Rater": "gemini-judge-v2-heads", "Queries": 3, "Arms": ["a", "b"]},
        report_date=date(2026, 9, 28),
    )

    assert text == EXPECTED_FULL
    assert gates == GateReport(
        arm=JEV,
        checks=(
            GateCheck(
                gate="G1 (a)",
                description="Jev nDCG@10 ≥ Gemini nDCG@10 - 0.01",
                measured=0.7999770933423677,
                threshold=0.9808842494603752,
                passed=False,
            ),
            GateCheck(
                gate="G1 (b)",
                description="Jev mean top-1 grade ≥ Gemini - 0.1",
                measured=2.6666666666666665,
                threshold=3.9,
                passed=False,
            ),
            GateCheck(
                gate="G1 (c)",
                description="Jev nDCG@10 ≥ deterministic + 0.02",
                measured=0.7999770933423677,
                threshold=0.7060810684256661,
                passed=True,
            ),
            GateCheck(
                gate="G1 (d) instruction",
                description="instruction-type adversarial hits in the top 3 ≤ Gemini's",
                measured=2.0,
                threshold=3.0,
                passed=True,
            ),
            GateCheck(
                gate="G1 (d) all",
                description="all adversarial hits in the top 3 ≤ Gemini's",
                measured=3.0,
                threshold=4.0,
                passed=True,
            ),
            GateCheck(
                gate="G2 token",
                description="input token / rerank p50 ≤ 3300",
                measured=3150.0,
                threshold=3300.0,
                passed=True,
            ),
            GateCheck(
                gate="G2 cost",
                description="Jev cost per rerank ≤ 60% of Gemini's real cost",
                measured=9.206200000000001e-05,
                threshold=3.9672e-05,
                passed=False,
            ),
        ),
    )


async def test_a_report_without_optional_sections_keeps_the_two_tables() -> None:
    report = await run_offline(QUERIES, RELEVANCE, ARMS)

    text = render_markdown(report, queries=QUERIES, meta={}, report_date=date(2026, 10, 1))

    assert text == EXPECTED_MINIMAL


@pytest.mark.parametrize(
    ("adversarial", "reference", "gemini_cost", "expected"),
    [
        pytest.param(
            CLEAN_ADVERSARIAL,
            None,
            0.0002,
            (
                GateCheck(
                    gate="G1 (a)",
                    description="Jev nDCG@10 ≥ Gemini nDCG@10 - 0.01",
                    measured=0.8838,
                    threshold=0.8734999999999999,
                    passed=True,
                ),
                GateCheck(
                    gate="G1 (b)",
                    description="Jev mean top-1 grade ≥ Gemini - 0.1",
                    measured=2.78,
                    threshold=2.65,
                    passed=True,
                ),
                GateCheck(
                    gate="G1 (c)",
                    description="Jev nDCG@10 ≥ deterministic + 0.02",
                    measured=0.8838,
                    threshold=0.8502000000000001,
                    passed=True,
                ),
                GateCheck(
                    gate="G1 (d)",
                    description="0 adversarial listings in the top 3",
                    measured=0.0,
                    threshold=0.0,
                    passed=True,
                ),
                GateCheck(
                    gate="G2 token",
                    description="input token / rerank p50 ≤ 3300",
                    measured=3301.0,
                    threshold=3300.0,
                    passed=False,
                ),
                GateCheck(
                    gate="G2 cost",
                    description="Jev cost per rerank ≤ 60% of Gemini's real cost",
                    measured=0.000127,
                    threshold=0.00012,
                    passed=False,
                ),
            ),
            id="original-d-clean",
        ),
        pytest.param(
            JEV_ADVERSARIAL,
            None,
            None,
            (
                GateCheck(
                    gate="G1 (a)",
                    description="Jev nDCG@10 ≥ Gemini nDCG@10 - 0.01",
                    measured=0.8838,
                    threshold=0.8734999999999999,
                    passed=True,
                ),
                GateCheck(
                    gate="G1 (b)",
                    description="Jev mean top-1 grade ≥ Gemini - 0.1",
                    measured=2.78,
                    threshold=2.65,
                    passed=True,
                ),
                GateCheck(
                    gate="G1 (c)",
                    description="Jev nDCG@10 ≥ deterministic + 0.02",
                    measured=0.8838,
                    threshold=0.8502000000000001,
                    passed=True,
                ),
                GateCheck(
                    gate="G1 (d)",
                    description="0 adversarial listings in the top 3",
                    measured=3.0,
                    threshold=0.0,
                    passed=False,
                ),
                GateCheck(
                    gate="G2 token",
                    description="input token / rerank p50 ≤ 3300",
                    measured=3301.0,
                    threshold=3300.0,
                    passed=False,
                ),
                GateCheck(
                    gate="G2 cost",
                    description="Jev cost per rerank ≤ 60% of Gemini's real cost",
                    measured=0.000127,
                    threshold=0.0001278,
                    passed=True,
                ),
            ),
            id="original-d-hit",
        ),
        pytest.param(
            None,
            None,
            None,
            (
                GateCheck(
                    gate="G1 (a)",
                    description="Jev nDCG@10 ≥ Gemini nDCG@10 - 0.01",
                    measured=0.8838,
                    threshold=0.8734999999999999,
                    passed=True,
                ),
                GateCheck(
                    gate="G1 (b)",
                    description="Jev mean top-1 grade ≥ Gemini - 0.1",
                    measured=2.78,
                    threshold=2.65,
                    passed=True,
                ),
                GateCheck(
                    gate="G1 (c)",
                    description="Jev nDCG@10 ≥ deterministic + 0.02",
                    measured=0.8838,
                    threshold=0.8502000000000001,
                    passed=True,
                ),
                GateCheck(
                    gate="G2 token",
                    description="input token / rerank p50 ≤ 3300",
                    measured=3301.0,
                    threshold=3300.0,
                    passed=False,
                ),
                GateCheck(
                    gate="G2 cost",
                    description="Jev cost per rerank ≤ 60% of Gemini's real cost",
                    measured=0.000127,
                    threshold=0.0001278,
                    passed=True,
                ),
            ),
            id="no-adversarial",
        ),
        pytest.param(
            None,
            GEMINI_ADVERSARIAL,
            None,
            (
                GateCheck(
                    gate="G1 (a)",
                    description="Jev nDCG@10 ≥ Gemini nDCG@10 - 0.01",
                    measured=0.8838,
                    threshold=0.8734999999999999,
                    passed=True,
                ),
                GateCheck(
                    gate="G1 (b)",
                    description="Jev mean top-1 grade ≥ Gemini - 0.1",
                    measured=2.78,
                    threshold=2.65,
                    passed=True,
                ),
                GateCheck(
                    gate="G1 (c)",
                    description="Jev nDCG@10 ≥ deterministic + 0.02",
                    measured=0.8838,
                    threshold=0.8502000000000001,
                    passed=True,
                ),
                GateCheck(
                    gate="G2 token",
                    description="input token / rerank p50 ≤ 3300",
                    measured=3301.0,
                    threshold=3300.0,
                    passed=False,
                ),
                GateCheck(
                    gate="G2 cost",
                    description="Jev cost per rerank ≤ 60% of Gemini's real cost",
                    measured=0.000127,
                    threshold=0.0001278,
                    passed=True,
                ),
            ),
            id="reference-only",
        ),
    ],
)
def test_gate_checks_are_unchanged(
    adversarial: AdversarialSummary | None,
    reference: AdversarialSummary | None,
    gemini_cost: float | None,
    expected: tuple[GateCheck, ...],
) -> None:
    report = evaluate_gates(
        GATE_SUMMARIES,
        jev_arm="jev:x:a1.0",
        adversarial=adversarial,
        reference_adversarial=reference,
        gemini_cost_per_rerank_usd=gemini_cost,
    )

    assert report == GateReport(arm="jev:x:a1.0", checks=expected)
