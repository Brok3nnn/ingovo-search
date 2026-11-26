"""`ingovo-search` — run one natural-language search from the terminal.

    ingovo-search "téglalakás Budán 80 millió alatt, erkéllyel"
    ingovo-search --parse-only "2 szobás lakás a 13. kerületben 60 millióig"
    ingovo-search --offline "csendes családi ház kerttel Debrecen környékén"

What each mode needs:

- `--offline`: nothing. The deterministic parser only (no model, no database);
  prints the filters it extracted.
- `--parse-only`: `GEMINI_API_KEY` for queries the deterministic fast path does not
  cover. Prints the parsed filters.
- default (full search): the seeded database (`DATABASE_URL`), `GEMINI_API_KEY`
  (parser and rerank) and `COHERE_API_KEY` (query embedding). Prints the parsed
  filters and the ranked listings with their match percentages.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from ingovo_search.db.session import get_async_session_factory
from ingovo_search.lib.cache import MemoryCache
from ingovo_search.lib.embeddings import CohereEmbeddingClient
from ingovo_search.lib.env import get_settings
from ingovo_search.lib.llm import GoogleAiClient
from ingovo_search.lib.session import SessionState
from ingovo_search.recommender.parser import CacheParserCache, ParserSettings, fallback_parse_query, parse_query
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.search.models import SearchResponse
from ingovo_search.search.pipeline import run_search_pipeline
from ingovo_search.search.run import SearchDependencies, SearchInput, SearchOptions
from ingovo_search.types.errors import IngovoError


def parsed_summary(parsed: ParsedQuery) -> dict[str, Any]:
    """The parts of a parsed query worth showing: everything that differs from the defaults."""
    return parsed.model_dump(mode="json", exclude_defaults=True, exclude_none=True)


def _money(value: object) -> str:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{value / 1_000_000:,.1f}M Ft"
    return "-"


def format_results(response: SearchResponse, *, limit: int) -> str:
    """A plain-text table of the ranked results, plus what the pipeline relaxed or asked."""
    lines: list[str] = []
    if response.relaxed:
        lines.append(f"relaxed: {' -> '.join(response.relaxation_path)}")
    if response.clarification is not None:
        lines.append(f"clarification suggested: {response.clarification.reason}")
    lines.append(f"{len(response.results)} results from {response.total_candidates} candidates")
    lines.append(f"{'#':>2}  {'match':>5}  {'price':>11}  {'m2':>5}  {'rooms':>5}  location / title")
    for index, result in enumerate(response.results[:limit], start=1):
        card = result.property or {}
        location = ", ".join(str(part) for part in (card.get("city"), card.get("district")) if part)
        area = card.get("area")
        lines.append(
            f"{index:>2}  {result.match_percent:>4}%  {_money(card.get('price_huf')):>11}  "
            f"{area if area is not None else '-':>5}  {card.get('rooms') or '-':>5}  "
            f"{location} | {card.get('title', result.listing_id)}"
        )
        if result.rationale:
            lines.append(f"    {result.rationale}")
    return "\n".join(lines)


async def parse_only(query: str, *, locale: str) -> ParsedQuery:
    cache = MemoryCache()
    return await parse_query(
        query, client=GoogleAiClient(), cache=CacheParserCache(cache), settings=ParserSettings(locale=locale)
    )


async def search(query: str, *, locale: str, rerank: bool, narratives: bool) -> tuple[ParsedQuery, SearchResponse]:
    """The full pipeline against the configured database, without the HTTP layer or a cookie session."""
    cache = MemoryCache()
    llm = GoogleAiClient()
    async with get_async_session_factory()() as db_session:
        parsed = await parse_query(
            query, client=llm, cache=CacheParserCache(cache), session=db_session, settings=ParserSettings(locale=locale)
        )
        now = datetime.now(UTC)
        state = SessionState(session_id="cli", created_at=now, updated_at=now)
        settings = get_settings()
        options = SearchOptions(
            user_signed_in=False,
            include_narratives=narratives,
            narrative_locale=locale,
            include_rerank=rerank,
            rerank_backend="gemini" if rerank else "off",
            rerank_rank_only=settings.ai_search_rerank_rank_only,
            rerank_timeout_ms=settings.ai_search_rerank_timeout_ms,
            conditional_rerank=settings.ai_search_conditional_rerank,
        )
        response = await run_search_pipeline(
            SearchInput(state, parsed), SearchDependencies(db_session, llm, CohereEmbeddingClient(), cache), options
        )
    return parsed, response


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingovo-search",
        description="Search the seeded listings with a natural-language query (Hungarian or English).",
    )
    parser.add_argument("query", help='e.g. "téglalakás Budán 80 millió alatt, erkéllyel"')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--parse-only", action="store_true", help="only parse the query (needs GEMINI_API_KEY)")
    mode.add_argument("--offline", action="store_true", help="deterministic parser only: no keys, no database")
    parser.add_argument("--locale", choices=("hu", "en"), default="hu", help="language of the query")
    parser.add_argument("--limit", type=int, default=10, help="how many results to print (default 10)")
    parser.add_argument("--no-rerank", action="store_true", help="skip the LLM rerank (deterministic order)")
    parser.add_argument("--narratives", action="store_true", help="also write the top results' narratives")
    parser.add_argument("--json", action="store_true", help="print the raw JSON response")
    return parser


def _print_parsed(parsed: ParsedQuery) -> None:
    print("parsed query:")  # noqa: T201
    print(json.dumps(parsed_summary(parsed), ensure_ascii=False, indent=2))  # noqa: T201


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.offline:
            _print_parsed(fallback_parse_query(args.query))
            return 0
        if args.parse_only:
            _print_parsed(asyncio.run(parse_only(args.query, locale=args.locale)))
            return 0
        parsed, response = asyncio.run(
            search(args.query, locale=args.locale, rerank=not args.no_rerank, narratives=args.narratives)
        )
    except IngovoError as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    if args.json:
        print(response.model_dump_json(indent=2))  # noqa: T201
        return 0
    _print_parsed(parsed)
    print()  # noqa: T201
    print(format_results(response, limit=args.limit))  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
