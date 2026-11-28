"""Unit tests for the query parser.

Covers:
1. Mock LLM returns the expected parser output for `hc01`; `parse_query`
   returns a validated `ParsedQuery` matching the fixture.
2. Mock LLM returns malformed JSON twice → raises `ParseError`.
3. Mock LLM returns wrong-schema JSON once, valid on retry → returns `ParsedQuery`.
4. Same query twice → second call is a cache hit, no LLM call.
5. `EnrichmentLog` row written for both success and failure attempts.

The live parser eval (`eval/run_parser_eval.py`) is not covered here: it needs
a Gemini key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.db.models import EnrichmentLog
from ingovo_search.lib.cache import MemoryCache
from ingovo_search.lib.llm import LlmClient, LlmRequest, MockLlmClient, fixture_key
from ingovo_search.lib.llm_models import TIER_S_PARSER
from ingovo_search.recommender.parser import (
    CacheParserCache,
    InMemoryParserCache,
    _with_budapest_district_default,  # pyright: ignore[reportPrivateUsage]
    parse_query,
    parser_cache_key,
)
from ingovo_search.recommender.parser_prompt import PARSER_PROMPT_VERSION
from ingovo_search.recommender.parser_schema import HardFilters, ParsedQuery, PoiNear
from ingovo_search.types.errors import LlmError, ParseError

FIXTURE_PATH = Path(__file__).parent.parent.parent / "fixtures" / "parser_responses.json"

HC01_QUERY = "Kétszobás panel a XII. kerületben, max 80 millió forint"
VIBE01_QUERY = "Csendes, világos otthon a Budai-hegyvidéken"


def _load_fixture(name: str) -> str:
    payload: dict[str, dict[str, Any]] = json.loads(FIXTURE_PATH.read_text())
    return json.dumps(payload[name])


class _RecordingSession:
    """Captures `session.add(...)` calls without touching a database."""

    def __init__(self) -> None:
        self.added: list[Any] = []

    def add(self, entity: Any) -> None:
        self.added.append(entity)


class _CountingLlmClient:
    """Wraps `MockLlmClient` to count how many `complete()` invocations occurred."""

    def __init__(self, fixtures: dict[str, str] | None = None) -> None:
        self._inner = MockLlmClient(fixtures)
        self.calls: int = 0

    def add(self, key: str, response: str) -> None:
        self._inner.add(key, response)

    async def complete(self, req: LlmRequest) -> str:
        self.calls += 1
        return await self._inner.complete(req)


class _Clock:
    """Manually advanced monotonic clock for TTL-expiry tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _initial_key(query: str) -> str:
    return fixture_key(LlmRequest(model=TIER_S_PARSER, system="", user=query))


def _correction_key(query: str) -> str:
    return fixture_key(LlmRequest(model=TIER_S_PARSER, system="", user=f"[CORRECT_SCHEMA]\n{query}"))


def _fake_session() -> AsyncSession:
    return cast(AsyncSession, _RecordingSession())


def _enrichment_logs(session: AsyncSession) -> list[EnrichmentLog]:
    return [entity for entity in cast(_RecordingSession, session).added if isinstance(entity, EnrichmentLog)]


# --- valid parse ------------------------------------------


async def test_parse_query_returns_validated_parsed_query() -> None:
    client = _CountingLlmClient({_initial_key(HC01_QUERY): _load_fixture("hc01")})
    session = _fake_session()

    result = await parse_query(HC01_QUERY, client=cast(LlmClient, client), session=session)

    assert isinstance(result, ParsedQuery)
    assert result.hard_filters.city == "Budapest"
    assert result.hard_filters.district_in == ["12"]
    assert result.hard_filters.property_type_in == ["Lakás"]
    assert result.hard_filters.rooms == 2
    assert result.hard_filters.price_huf_max == 80_000_000
    assert result.conflict_flag is False
    assert result.parse_confidence is not None
    assert result.parse_confidence.rooms == 0.95
    assert client.calls == 1

    logs = _enrichment_logs(session)
    assert len(logs) == 1
    assert logs[0].outcome == "success"
    assert logs[0].model_used == TIER_S_PARSER
    assert logs[0].prompt_version == PARSER_PROMPT_VERSION
    assert logs[0].property_id is None


async def test_parse_query_extracts_soft_prefs_and_geo_hints() -> None:
    client = _CountingLlmClient({_initial_key(VIBE01_QUERY): _load_fixture("vibe01")})

    result = await parse_query(VIBE01_QUERY, client=cast(LlmClient, client))

    assert result.soft_preferences == ["quiet", "natural_light"]
    assert result.soft_prefs_freetext == ["jó otthonról dolgozni"]
    assert result.geo_hints.region_name == "Budai-hegyvidék"
    assert result.geo_hints.side_of_river == "Buda"


# --- persistent parse failure -----------------------------


async def test_persistent_malformed_json_raises_parse_error() -> None:
    query = "Hibás lekérdezés egy"
    client = _CountingLlmClient(
        {
            _initial_key(query): "not json {",
            _correction_key(query): "still not json",
        }
    )
    session = _fake_session()

    with pytest.raises(ParseError):
        await parse_query(query, client=cast(LlmClient, client), session=session)

    assert client.calls == 2
    assert [log.outcome for log in _enrichment_logs(session)] == ["parse_error", "parse_error"]


# --- correction retry recovers ----------------------------


async def test_wrong_schema_then_valid_on_retry() -> None:
    query = "Rossz séma, majd jó"
    client = _CountingLlmClient(
        {
            _initial_key(query): json.dumps({"hard_filters": 123}),
            _correction_key(query): _load_fixture("hc01"),
        }
    )
    session = _fake_session()

    result = await parse_query(query, client=cast(LlmClient, client), session=session)

    assert isinstance(result, ParsedQuery)
    assert result.hard_filters.rooms == 2
    assert client.calls == 2
    assert [log.outcome for log in _enrichment_logs(session)] == ["parse_error", "success"]


# --- cache hit --------------------------------------------


async def test_same_query_twice_hits_cache() -> None:
    client = _CountingLlmClient({_initial_key(HC01_QUERY): _load_fixture("hc01")})
    cache = InMemoryParserCache()

    first = await parse_query(HC01_QUERY, client=cast(LlmClient, client), cache=cache)
    second = await parse_query(HC01_QUERY, client=cast(LlmClient, client), cache=cache)

    assert first == second
    assert client.calls == 1


async def test_shared_cache_adapter_survives_parser_cache_reinstantiation() -> None:
    client = _CountingLlmClient({_initial_key(HC01_QUERY): _load_fixture("hc01")})
    backend = MemoryCache()

    first = await parse_query(HC01_QUERY, client=cast(LlmClient, client), cache=CacheParserCache(backend))
    second = await parse_query(HC01_QUERY, client=cast(LlmClient, client), cache=CacheParserCache(backend))

    assert first == second
    assert client.calls == 1


async def test_cache_hit_still_applies_deterministic_defaults() -> None:
    # Carries a soft preference ("napos") so it bypasses the deterministic fast
    # path and genuinely exercises the LLM + cache path this test is about.
    query = "napos lakást vennék a XI. kerületben"
    client = _CountingLlmClient({_initial_key(query): _load_fixture("minimal")})
    cache = InMemoryParserCache()

    first = await parse_query(query, client=cast(LlmClient, client), cache=cache)
    second = await parse_query(query, client=cast(LlmClient, client), cache=cache)

    assert first.hard_filters.transaction_type == "sale"
    assert first.hard_filters.district_in == ["11"]
    assert second.hard_filters.transaction_type == "sale"
    assert second.hard_filters.district_in == ["11"]
    assert client.calls == 1


async def test_cache_key_normalises_whitespace_and_case() -> None:
    # A soft-preference query ("napos") can't fast-path, so it must reach the LLM
    # — exactly what this cache-normalisation test needs to exercise.
    client = _CountingLlmClient({_initial_key("napos lakás Budapesten"): _load_fixture("minimal")})
    cache = InMemoryParserCache()

    await parse_query("napos lakás Budapesten", client=cast(LlmClient, client), cache=cache)
    await parse_query("  napos   LAKÁS   BUDAPESTEN  ", client=cast(LlmClient, client), cache=cache)

    assert client.calls == 1


async def test_parser_defaults_ambiguous_property_search_to_sale() -> None:
    query = "budapesti lakás 80 millió alatt erkéllyel"
    client = _CountingLlmClient({_initial_key(query): _load_fixture("minimal")})

    result = await parse_query(query, client=cast(LlmClient, client))

    assert result.hard_filters.transaction_type == "sale"


async def test_parser_defaults_colloquial_buda_to_budapest_on_llm_path() -> None:
    query = "budai lakás 80 millió alatt erkéllyel"
    client = _CountingLlmClient({_initial_key(query): _load_fixture("minimal")})

    result = await parse_query(query, client=cast(LlmClient, client))

    assert result.hard_filters.city == "Budapest"
    assert result.hard_filters.transaction_type == "sale"


async def test_parser_defaults_explicit_rent_search_to_rent() -> None:
    query = "kiadó lakás Budapesten erkéllyel"
    client = _CountingLlmClient({_initial_key(query): _load_fixture("minimal")})

    result = await parse_query(query, client=cast(LlmClient, client))

    assert result.hard_filters.transaction_type == "rent"


@pytest.mark.parametrize(
    "query",
    [
        "albérletet keresek a 13. kerületben",
        "iroda bérlésre Budapesten",
        "irodát bérelnék a belvárosban",
    ],
)
async def test_parser_detects_inflected_rent_intent(query: str) -> None:
    client = _CountingLlmClient({_initial_key(query): _load_fixture("minimal")})

    result = await parse_query(query, client=cast(LlmClient, client))

    assert result.hard_filters.transaction_type == "rent"


@pytest.mark.parametrize(
    "query",
    [
        "lakást vennék a XI. kerületben",
        "lakást vásárolnék 70 millióig",
        "eladó családi házat keresek",
    ],
)
async def test_parser_detects_inflected_sale_intent(query: str) -> None:
    client = _CountingLlmClient({_initial_key(query): _load_fixture("minimal")})

    result = await parse_query(query, client=cast(LlmClient, client))

    assert result.hard_filters.transaction_type == "sale"


@pytest.mark.parametrize(
    ("query", "district"),
    [
        ("lakást vennék a XI. kerületben", "11"),
        ("eladó lakás a 13. kerületben", "13"),
        ("kiadó lakás a VII. kerületben", "7"),
        ("budapesti 2. kerületi lakás", "2"),
    ],
)
async def test_parser_extracts_budapest_district_mentions(query: str, district: str) -> None:
    client = _CountingLlmClient({_initial_key(query): _load_fixture("minimal")})

    result = await parse_query(query, client=cast(LlmClient, client))

    assert result.hard_filters.city == "Budapest"
    assert result.hard_filters.district_in == [district]


@pytest.mark.parametrize(
    "query",
    [
        # 2026-07-03 eval catch: exclusions must never become inclusive filters.
        "albérlet Pesten maximum 250 ezerért, de nem a 8. kerületben",
        "eladó lakás Budapesten, ne a 7. kerületben legyen",
        "kiadó lakás, a 8. kerület kivételével",
    ],
)
async def test_negated_district_mentions_are_not_merged(query: str) -> None:
    client = _CountingLlmClient({_initial_key(query): _load_fixture("minimal")})

    result = await parse_query(query, client=cast(LlmClient, client))

    assert result.hard_filters.district_in == []


async def test_negated_and_included_districts_mix() -> None:
    query = "eladó lakás a 9. kerületben, de nem a 8. kerületben"
    client = _CountingLlmClient({_initial_key(query): _load_fixture("minimal")})

    result = await parse_query(query, client=cast(LlmClient, client))

    assert result.hard_filters.district_in == ["9"]


async def test_llm_supplied_negated_district_is_stripped() -> None:
    """Even if the model fills district_in despite the prompt rule, the
    deterministic layer strips districts that are negated in the query text."""

    parsed = ParsedQuery(hard_filters=HardFilters(city="Budapest", district_in=["8"]))
    result = _with_budapest_district_default("albérlet Pesten maximum 250 ezerért, de nem a 8. kerületben", parsed)

    assert result.hard_filters.district_in == []
    # Stripping must not invent a city either way; the pre-set one survives.
    assert result.hard_filters.city == "Budapest"


async def test_stripping_negated_district_does_not_invent_city() -> None:
    parsed = ParsedQuery(hard_filters=HardFilters(district_in=["8"]))
    result = _with_budapest_district_default("kiadó lakás, de nem a 8. kerületben", parsed)

    assert result.hard_filters.district_in == []
    assert result.hard_filters.city is None


async def test_expired_cache_entry_triggers_new_llm_call() -> None:
    client = _CountingLlmClient({_initial_key(HC01_QUERY): _load_fixture("hc01")})
    clock = _Clock()
    cache = InMemoryParserCache(ttl_seconds=600.0, clock=clock)

    await parse_query(HC01_QUERY, client=cast(LlmClient, client), cache=cache)
    clock.now = 601.0
    await parse_query(HC01_QUERY, client=cast(LlmClient, client), cache=cache)

    assert client.calls == 2


# --- logging ----------------------------------------------


async def test_enrichment_log_written_for_success_and_failure() -> None:
    query = "Egy hibás, majd sikeres"
    client = _CountingLlmClient(
        {
            _initial_key(query): "broken",
            _correction_key(query): _load_fixture("minimal"),
        }
    )
    session = _fake_session()

    await parse_query(query, client=cast(LlmClient, client), session=session)

    logs = _enrichment_logs(session)
    assert len(logs) == 2
    for log in logs:
        assert log.model_used == TIER_S_PARSER
        assert log.prompt_version == PARSER_PROMPT_VERSION
        # cost-tracker: estimated from text length, non-zero on success.
        assert log.cost_usd >= 0.0
        assert log.latency_ms >= 0


async def test_transport_error_propagates_and_logs() -> None:
    client = _CountingLlmClient({})  # no fixture → MockLlmClient raises LlmError
    session = _fake_session()

    with pytest.raises(LlmError):
        await parse_query("Bármilyen lekérdezés", client=cast(LlmClient, client), session=session)

    logs = _enrichment_logs(session)
    assert len(logs) == 1
    assert logs[0].outcome == "transport_error"


async def test_transport_error_on_the_correction_attempt_propagates_after_both_logs() -> None:
    query = "Hibás válasz, utána kiesés"
    client = _CountingLlmClient({_initial_key(query): "broken"})  # no correction fixture → LlmError
    session = _fake_session()
    cache = InMemoryParserCache()

    with pytest.raises(LlmError) as exc_info:
        await parse_query(query, client=cast(LlmClient, client), session=session, cache=cache)

    assert client.calls == 2
    assert [log.outcome for log in _enrichment_logs(session)] == ["parse_error", "transport_error"]
    assert exc_info.value.__context__ is None
    assert await cache.get(parser_cache_key(query, TIER_S_PARSER, PARSER_PROMPT_VERSION)) is None


async def test_parse_error_is_chained_to_the_last_validation_error() -> None:
    query = "Hibás lekérdezés kettő"
    client = _CountingLlmClient({_initial_key(query): "not json {", _correction_key(query): "still not json"})

    with pytest.raises(ParseError, match="after 2 attempts") as exc_info:
        await parse_query(query, client=cast(LlmClient, client))

    assert isinstance(exc_info.value.__cause__, ValidationError)
    assert exc_info.value.__context__ is None
    assert "still not json" in str(exc_info.value.__cause__)


# --- Schema + cache-key behaviour --------------------------------------------


def test_minimal_json_validates_with_schema_defaults() -> None:
    parsed = ParsedQuery.model_validate_json(_load_fixture("minimal"))
    assert parsed.semantic_query == "lakás Budapesten"
    assert parsed.hard_filters.district_in == []
    assert parsed.anchors.price_huf is None
    assert parsed.soft_preferences == []
    assert parsed.conflict_flag is False


def test_cache_key_changes_with_prompt_version() -> None:
    key_a = parser_cache_key(HC01_QUERY, TIER_S_PARSER, "parser/v0-skeleton")
    key_b = parser_cache_key(HC01_QUERY, TIER_S_PARSER, "parser/v1")
    assert key_a != key_b


def test_parsed_query_is_frozen() -> None:
    parsed = ParsedQuery.model_validate_json(_load_fixture("minimal"))

    with pytest.raises(ValidationError):
        parsed.semantic_query = "más"  # type: ignore[misc]


# --- poi_near schema (POI step 3) ---


def test_poi_near_model_is_frozen_with_default_walk_minutes() -> None:
    item = PoiNear(category="school")

    assert item.max_walk_minutes is None
    with pytest.raises(ValidationError):
        item.category = "park"  # type: ignore[misc]


def test_parsed_query_defaults_to_empty_poi_near() -> None:
    assert ParsedQuery().poi_near == []


def test_poi_near_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        PoiNear(category="casino")  # type: ignore[arg-type]


def test_poi_near_walk_minutes_round_trips_through_dump() -> None:
    parsed = ParsedQuery(poi_near=[PoiNear(category="transit", max_walk_minutes=5)])

    assert parsed.model_dump()["poi_near"] == [{"category": "transit", "max_walk_minutes": 5}]
