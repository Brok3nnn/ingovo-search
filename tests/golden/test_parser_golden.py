"""Golden snapshot of the deterministic query parser over the starter eval set (P5.1).

Every query of ``eval/golden/eval_set.starter.jsonl`` (141 real-looking Hungarian searches)
goes through both deterministic entry points without an LLM; the ``ParsedQuery`` each
returns — or ``None`` when the parser defers to the LLM — is pinned. A refactor of
``_deterministic_parse`` must leave this file untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.golden._snapshot import check_snapshot

from ingovo_search.recommender.parser_fastpath import deterministic_parse, deterministic_parse_loose

EVAL_SET = Path(__file__).resolve().parents[2] / "eval" / "golden" / "eval_set.starter.jsonl"


def _queries() -> list[str]:
    queries: list[str] = []
    for line in EVAL_SET.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record: dict[str, Any] = json.loads(line)
        queries.append(str(record["query"]))
    return queries


def _parsed(text: str, *, loose: bool) -> dict[str, Any] | None:
    parsed = deterministic_parse_loose(text) if loose else deterministic_parse(text)
    return None if parsed is None else parsed.model_dump(mode="json", exclude_none=True)


def test_the_eval_set_is_the_one_we_pin() -> None:
    assert len(_queries()) >= 100


@pytest.mark.parametrize("loose", [False, True], ids=["strict", "loose"])
def test_deterministic_parser_golden(request: pytest.FixtureRequest, loose: bool) -> None:
    snapshot = [{"query": query, "parsed": _parsed(query, loose=loose)} for query in _queries()]
    check_snapshot(request, f"parser_deterministic_{'loose' if loose else 'strict'}", snapshot)


def test_deterministic_parser_covers_a_meaningful_share() -> None:
    """The strict path answers simple queries itself; a regression to 'always defer' would hide behind the snapshot."""
    answered = sum(1 for query in _queries() if deterministic_parse(query) is not None)
    assert answered >= 10  # 13 of 141 on 2026-10-13; the loose path answers 120
