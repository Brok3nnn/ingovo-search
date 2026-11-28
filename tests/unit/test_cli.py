"""The `ingovo-search` CLI without a database or a model: offline parsing and the result table."""

from __future__ import annotations

import json
from collections.abc import Iterator
from unittest.mock import AsyncMock

import pytest
import structlog

from ingovo_search import cli
from ingovo_search.lib.env import get_settings
from ingovo_search.recommender.score import ScoreBreakdown, ScoredCandidate
from ingovo_search.search.models import SearchResponse

# "A 2-room flat in the 13th district up to 60 million HUF": covered by the deterministic parser.
FAST_PATH_QUERY = "2 szobás lakás a 13. kerületben 60 millióig"


@pytest.fixture(autouse=True)
def restore_structlog_defaults() -> Iterator[None]:
    """`cli.main` configures logging for the process; give the rest of the suite the defaults back."""
    yield
    structlog.reset_defaults()


class _UnreachableSession:
    """An `AsyncSession` stand-in whose first query fails the way a stopped Postgres does."""

    async def __aenter__(self) -> _UnreachableSession:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def execute(self, *_args: object, **_kwargs: object) -> None:
        raise ConnectionRefusedError(61, "Connect call failed ('127.0.0.1', 5432)")


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


def test_json_mode_prints_only_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--offline", "--json", FAST_PATH_QUERY]) == 0
    assert json.loads(capsys.readouterr().out)["hard_filters"]["rooms"] == 2


def test_cli_logs_go_to_stderr_and_only_from_warning_up(capsys: pytest.CaptureFixture[str]) -> None:
    cli.configure_cli_logging(verbose=False)
    logger = structlog.get_logger("tests.cli.fresh_logger")

    logger.info("parser cache miss")
    logger.warning("rerank timed out")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "rerank timed out" in captured.err
    assert "parser cache miss" not in captured.err


def test_an_unreachable_database_is_a_clean_error_before_any_model_call(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://ingovo:s3cret@db.invalid:5432/ingovo")
    get_settings.cache_clear()
    parse = AsyncMock()
    monkeypatch.setattr(cli, "get_async_session_factory", lambda: _UnreachableSession)
    monkeypatch.setattr(cli, "parse_query", parse)

    assert cli.main([FAST_PATH_QUERY]) == 1

    err = capsys.readouterr().err
    assert "error: database unreachable at postgresql+asyncpg://ingovo:***@db.invalid:5432/ingovo" in err
    assert "docker compose up -d" in err
    assert "s3cret" not in err
    parse.assert_not_awaited()


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
