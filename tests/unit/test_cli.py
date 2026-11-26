"""The `ingovo-search` CLI without a database or a model: offline parsing and the result table."""

from __future__ import annotations

import json

import pytest

from ingovo_search import cli
from ingovo_search.recommender.score import ScoreBreakdown, ScoredCandidate
from ingovo_search.search.models import SearchResponse

# "A 2-room flat in the 13th district up to 60 million HUF": covered by the deterministic parser.
FAST_PATH_QUERY = "2 szobás lakás a 13. kerületben 60 millióig"


def _breakdown() -> ScoreBreakdown:
    terms = dict.fromkeys(
        ("content", "factual", "softpref_core", "softpref_free", "anchor", "quality", "geo", "pooled", "recency"),
        0.05,
    )
    return ScoreBreakdown(**terms, explore=0.0, total=0.45)


def test_offline_mode_prints_the_deterministic_parse(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--offline", FAST_PATH_QUERY]) == 0

    out = capsys.readouterr().out
    parsed = json.loads(out.split("parsed query:\n", 1)[1])
    assert parsed["hard_filters"]["district_in"] == ["13"]
    assert parsed["hard_filters"]["rooms"] == 2
    assert parsed["hard_filters"]["price_huf_max"] == 60_000_000


def test_parse_only_uses_the_fast_path_without_a_model(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--parse-only", FAST_PATH_QUERY]) == 0
    assert '"district_in"' in capsys.readouterr().out


def test_a_missing_key_is_a_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    # Not on the fast path, so the parser needs Gemini; the conftest leaves the key unset.
    assert cli.main(["--parse-only", "csendes családi ház kerttel Debrecen környékén"]) == 1
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().err


def test_the_result_table_shows_match_price_and_rationale() -> None:
    response = SearchResponse(
        results=[
            ScoredCandidate(
                listing_id="p1",
                score=0.45,
                match_percent=87,
                breakdown=_breakdown(),
                rationale="Brick building in Buda with a balcony.",
                property={
                    "title": "Téglalakás erkéllyel",
                    "price_huf": 74_900_000,
                    "area": 58,
                    "rooms": "2",
                    "city": "Budapest",
                    "district": "11",
                },
            )
        ],
        total_candidates=41,
        relaxed=True,
        relaxation_path=["price +10%"],
    )

    table = cli.format_results(response, limit=10)

    assert "relaxed: price +10%" in table
    assert "1 results from 41 candidates" in table
    assert "87%" in table
    assert "74.9M Ft" in table
    assert "Budapest, 11 | Téglalakás erkéllyel" in table
    assert "Brick building in Buda with a balcony." in table
