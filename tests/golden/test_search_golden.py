"""Golden snapshot of the search orchestration over fixed scenarios.

The parser, recommender, clarification policy and reranker are replaced by deterministic
doubles (``_search_harness``), so what the snapshot pins is the work of
``run_search_pipeline`` and ``execute_search_request`` themselves: ordering, match
percentages, scope, compromises, clarification pass-through, cache behaviour, the
response contract. A refactor of the pipeline stages must leave these files unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.golden._search_harness import build_client, make_breakdown, make_property, make_rerank, patched_pipeline
from tests.golden._snapshot import check_snapshot

from ingovo_search.recommender.clarification_policy import Clarification, ClarificationOption, ClarificationReason
from ingovo_search.recommender.parser_schema import HardFilters, ParsedQuery

VOLATILE_META_KEYS = {"timestamp", "request_id", "timings_ms"}


def _normalised(body: dict[str, Any]) -> dict[str, Any]:
    """Drop what changes between runs; keep the contract shape of ``meta``."""
    meta = dict(body.get("meta") or {})
    body = dict(body)
    body["meta"] = {key: ("<volatile>" if key in VOLATILE_META_KEYS else value) for key, value in sorted(meta.items())}
    return body


def _three(**extra: Any) -> list[tuple[Any, Any]]:
    return [
        (make_property("L1", latitude=47.5, longitude=19.05, **extra), make_breakdown(0.9, factual=0.5)),
        (make_property("L2", price_huf=120_000_000, rooms="3", area_m2=85.0), make_breakdown(0.8, softpref_core=0.3)),
        # Price on request, another district; the recommender only returns in-scope rows.
        (make_property("L3", district="11", price_huf=None), make_breakdown(0.7)),
    ]


SCENARIOS: dict[str, dict[str, Any]] = {
    "plain_query_no_filters": {
        "parsed": ParsedQuery(),
        "props": _three(),
        "rerank": None,
        "request": {"query": "lakás"},
    },
    "semantic_city_rerank_reorder": {
        "parsed": ParsedQuery(semantic_query="kétszobás panel XIII", hard_filters=HardFilters(city="Budapest")),
        "props": _three(),
        "rerank": make_rerank(["L2", "L1", "L3"]),
        "request": {"query": "kétszobás panel XIII"},
    },
    "rerank_low_match": {
        "parsed": ParsedQuery(semantic_query="tetőtéri loft kilátással", hard_filters=HardFilters(city="Budapest")),
        "props": _three(),
        "rerank": make_rerank(["L1", "L2", "L3"], low_match=True),
        "request": {"query": "tetőtéri loft kilátással"},
    },
    "rerank_fast_path": {
        "parsed": ParsedQuery(semantic_query="panel", hard_filters=HardFilters(city="Budapest")),
        "props": _three(),
        "rerank": make_rerank(["L3", "L1", "L2"], fast_path=True),
        "request": {"query": "panel"},
    },
    "no_candidates": {
        "parsed": ParsedQuery(hard_filters=HardFilters(city="Sopron", price_huf_max=10_000_000)),
        "props": [],
        "rerank": None,
        "request": {"query": "Sopron 10 millió alatt"},
    },
    "price_and_rooms_filters": {
        "parsed": ParsedQuery(
            hard_filters=HardFilters(city="Budapest", price_huf_max=100_000_000, rooms_min=2, transaction_type="sale")
        ),
        "props": _three(),
        "rerank": None,
        "request": {"query": "budapest max 100 millió legalább 2 szoba eladó", "include_rerank": False},
    },
    "soft_preferences": {
        "parsed": ParsedQuery(
            semantic_query="felújított csendes",
            hard_filters=HardFilters(city="Budapest"),
            soft_preferences=["renovated", "quiet"],
        ),
        "props": _three(),
        "rerank": make_rerank(["L2", "L1", "L3"]),
        "request": {"query": "felújított csendes lakás Budapesten"},
    },
    "clarification_asked": {
        "parsed": ParsedQuery(hard_filters=HardFilters(city="Budapest")),
        "props": _three(),
        "rerank": None,
        "clarification": Clarification(
            reason=ClarificationReason.AMBIGUOUS_GEOGRAPHY,
            question="Melyik kerületre gondolsz?",
            options=[
                ClarificationOption(id="13", label="XIII. kerület"),
                ClarificationOption(id="11", label="XI. kerület"),
            ],
        ),
        "request": {"query": "budapest, a kerület mindegy"},
    },
    "conditional_rerank_skipped": {
        "parsed": ParsedQuery(semantic_query="panel", hard_filters=HardFilters(city="Budapest")),
        "props": _three(),
        "rerank": make_rerank(["L3", "L2", "L1"]),
        "request": {"query": "panel"},
        "should_rerank": False,
    },
    "non_huf_listing": {
        "parsed": ParsedQuery(hard_filters=HardFilters(city="Budapest")),
        "props": [(make_property("EUR", currency="EUR", price_huf=200_000), make_breakdown(0.6)), *_three()],
        "rerank": None,
        "request": {"query": "budapest"},
    },
    "rerank_requested_off": {
        "parsed": ParsedQuery(semantic_query="kertes ház", hard_filters=HardFilters(city="Budapest")),
        "props": _three(),
        "rerank": make_rerank(["L2", "L1", "L3"]),
        "request": {"query": "kertes ház", "include_rerank": False},
    },
    "district_filter": {
        "parsed": ParsedQuery(hard_filters=HardFilters(city="Budapest", district_in=["13"])),
        # Only the XIII. district rows are in scope for a district filter.
        "props": _three()[:2],
        "rerank": None,
        "request": {"query": "XIII. kerület"},
    },
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_search_pipeline_golden(request: pytest.FixtureRequest, name: str) -> None:
    scenario = SCENARIOS[name]
    client = build_client()
    with patched_pipeline(
        parsed=scenario["parsed"],
        recommend_return=scenario["props"],
        rerank_return=scenario["rerank"],
        clarification=scenario.get("clarification"),
        should_rerank=scenario.get("should_rerank", True),
    ):
        response = client.post("/api/search", json=scenario["request"])
        again = client.post("/api/search", json=scenario["request"])

    assert response.status_code == 200, response.text
    assert again.status_code == 200
    first = _normalised(response.json())
    second = _normalised(again.json())
    assert second["data"] == first["data"], "a repeated identical request is served from the response cache unchanged"
    check_snapshot(request, f"search_{name}", first)
