"""Ranking is never for sale: the ranking code carries no paid-placement field or term.

The ranking-sensitive modules are scanned for the vocabulary a paid placement would
need (a boost, a promoted or featured flag, a priority tier). A deliberate mention,
for example in guardrail code that rejects such a field, has to fragment the literal
so it does not match.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FORBIDDEN_KEYWORDS = ("boost", "promoted", "paid_rank", "featured", "priority")
FORBIDDEN_REGEX = re.compile("|".join(FORBIDDEN_KEYWORDS), re.IGNORECASE)

PACKAGE = Path(__file__).resolve().parents[3] / "src" / "ingovo_search"
SCOPED_FILES = (
    "recommender/score.py",
    "recommender/softpref_rules.py",
    "recommender/ranking_guardrails.py",
    "recommender/canonical_ranking.py",
    "recommender/match_percent.py",
    "recommender/rerank.py",
    "recommender/parser_schema.py",
    "db/models/property.py",
)


@pytest.mark.parametrize("rel_path", SCOPED_FILES)
def test_scoped_file_has_no_paid_placement_keywords(rel_path: str) -> None:
    target = PACKAGE / rel_path
    assert target.exists(), f"in-scope file {rel_path!r} not found at {target}"
    matches = FORBIDDEN_REGEX.findall(target.read_text(encoding="utf-8"))
    assert matches == [], f"{rel_path} contains paid-placement keyword(s): {sorted(set(matches))}"


def test_the_scan_catches_an_obvious_violation(tmp_path: Path) -> None:
    violation = tmp_path / "fake_property_model.py"
    violation.write_text("class FakeProperty:\n    boost: float = 0.0\n    promoted_until: int | None = None\n")
    matches = {m.lower() for m in FORBIDDEN_REGEX.findall(violation.read_text())}
    assert {"boost", "promoted"} <= matches
