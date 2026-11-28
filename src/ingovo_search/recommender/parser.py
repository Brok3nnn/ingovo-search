"""Query parser — Hungarian natural language → `ParsedQuery`.

`parse_query` is the single entry point: cache lookup, one Tier-S LLM call with
a single correction retry on schema failure, one `EnrichmentLog` row per
attempt, cache write on success.

The cache key folds in the model id and prompt version, so a
`PARSER_PROMPT_VERSION` bump invalidates every entry automatically. Production
uses `CacheParserCache` over the shared Redis-backed cache; tests can still
use `InMemoryParserCache`.

The prompt content lives in `parser_prompt.py`.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

import structlog
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.lib.cache import Cache, get_json, set_json
from ingovo_search.lib.llm import LlmClient, LlmRequest
from ingovo_search.lib.llm_models import TIER_S_PARSER
from ingovo_search.lib.llm_usage import reset_last_llm_usage
from ingovo_search.recommender.fastpath_lexicon import ROMAN_DISTRICTS
from ingovo_search.recommender.llm_attempt_log import LlmAttempt, log_llm_attempt
from ingovo_search.recommender.output_language import DEFAULT_OUTPUT_LOCALE
from ingovo_search.recommender.parser_fastpath import deterministic_parse, deterministic_parse_loose
from ingovo_search.recommender.parser_prompt import (
    PARSER_CORRECTION_SYSTEM_PROMPT,
    PARSER_PROMPT_VERSION,
    parser_system_prompt,
    render_correction_user_prompt,
    render_parser_user_prompt,
)
from ingovo_search.recommender.parser_schema import LocationCandidate, LocationResolution, ParsedQuery
from ingovo_search.recommender.settlements import ResolutionStatus, resolve_location_text
from ingovo_search.types.errors import LlmError, ParseError

DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60
_logger = structlog.get_logger("ingovo_search.recommender.parser")

RENT_INTENT_RE = re.compile(
    r"\b(kiad[oó]\w*|alb[eé]rlet\w*|b[eé]rl[eé]s\w*|b[eé]rel\w*)\b",
    re.IGNORECASE,
)
SALE_INTENT_RE = re.compile(
    r"\b(elad[oó]\w*|v[eé]tel\w*|v[aá]s[aá]rol\w*|v[aá]s[aá]rl[aá]s\w*|venni|venn[eé]\w*|megvenn[eé]\w*)\b",
    re.IGNORECASE,
)
# Stems are matched WITHOUT word boundaries so Hungarian inflections and
# compounds still register: "téglalakás", "kétszobás", "erkélyes",
# "kerületben", "házban" all count as a property search. (The rent/sale intent
# regexes already use `\w*` for the same reason — this one regressed to `\b`.)
PROPERTY_SEARCH_RE = re.compile(
    r"(ingatlan|lak[aá]s|h[aá]z|otthon|garzon|panel|telek|irod[aá]|gar[aá]zs|nyaral[oó]|"
    r"szob[aá]|ker[uü]let|budapest|erk[eé]ly|terasz|kert)\w*",
    re.IGNORECASE,
)
BUDAPEST_SIDE_RE = re.compile(
    r"\b(budai|bud[aá]n|bud[aá]ra|pesti|pesten|pestre)\b",
    re.IGNORECASE,
)
BUDAPEST_DISTRICT_RE = re.compile(
    r"\b(?P<district>[0-9]{1,2}|i{1,3}|iv|v|vi{0,3}|ix|x|xi{0,3}|xiv|xv|xvi{0,3}|xix|xx|xxi{0,3})"
    r"\.?\s*(?:ker(?:ü|u)let(?:ben|i|et|re|hez|e)?|ker\.?)\b",
    re.IGNORECASE,
)
# A district mention preceded by a negation word in the same clause ("de nem a
# 8. kerületben", "ne a 7. kerületben") or followed by "kivételével" is an
# EXCLUSION — it must never be merged into the inclusive `district_in` filter
# (2026-07-03 eval catch: the adversarial exclusion query gained district_in=["8"]).
# The window stops at clause punctuation so "olcsó, nem drága, a 8. kerületben"
# style clauses don't false-positive.
DISTRICT_NEGATION_BEFORE_RE = re.compile(r"\b(?:nem|ne|sem)\b[^,;.!?]{0,20}$")
DISTRICT_NEGATION_AFTER_RE = re.compile(r"^[^,;.!?]{0,26}\bkivétel")


@runtime_checkable
class ParserCache(Protocol):
    """Cache contract for `parse_query`."""

    async def get(self, key: str) -> ParsedQuery | None: ...
    async def set(self, key: str, value: ParsedQuery) -> None: ...


class InMemoryParserCache:
    """Process-local TTL cache. Entries expire `ttl_seconds` after they are set."""

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._store: dict[str, tuple[float, ParsedQuery]] = {}

    async def get(self, key: str) -> ParsedQuery | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if self._clock() >= expires_at:
            del self._store[key]
            return None
        return value

    async def set(self, key: str, value: ParsedQuery) -> None:
        self._store[key] = (self._clock() + self._ttl, value)


class CacheParserCache:
    """`ParserCache` adapter backed by the shared async cache."""

    def __init__(self, cache: Cache, *, ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS) -> None:
        self._cache = cache
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _key(key: str) -> str:
        return f"parser:{key}"

    async def get(self, key: str) -> ParsedQuery | None:
        return await get_json(self._cache, self._key(key), ParsedQuery)

    async def set(self, key: str, value: ParsedQuery) -> None:
        await set_json(self._cache, self._key(key), value, ttl_seconds=self._ttl_seconds)


def _normalise(text: str) -> str:
    """Collapse whitespace and case-fold so trivially different inputs share a key."""
    return " ".join(text.split()).casefold()


def parser_cache_key(text: str, model: str, prompt_version: str, locale: str = DEFAULT_OUTPUT_LOCALE) -> str:
    """Cache key: hash of model id + prompt version + locale + normalised query text.

    The locale is part of the key because it is part of the prompt: the same
    sentence read as Hungarian and read as Croatian can legitimately parse into
    different filters. Hungarian keeps the original three-part payload so entries
    written before the site went multilingual still hit.
    """
    payload = f"{model}\x00{prompt_version}\x00{_normalise(text)}"
    if locale != DEFAULT_OUTPUT_LOCALE:
        payload = f"{payload}\x00{locale}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _with_transaction_default(text: str, parsed: ParsedQuery) -> ParsedQuery:
    """Apply the product default for Hungarian property searches.

    The parser prompt asks the LLM to default ambiguous property searches to
    sale, but hard filters cannot rely on the model always doing that. This
    deterministic guard keeps rent listings out of ordinary buyer searches
    while preserving explicit rent intent.
    """
    if parsed.hard_filters.transaction_type is not None:
        return parsed

    normalized = _normalise(text)
    transaction_type = None
    if RENT_INTENT_RE.search(normalized):
        transaction_type = "rent"
    elif SALE_INTENT_RE.search(normalized) or PROPERTY_SEARCH_RE.search(normalized):
        transaction_type = "sale"

    if transaction_type is None:
        return parsed

    hard_filters = parsed.hard_filters.model_copy(update={"transaction_type": transaction_type})
    return parsed.model_copy(update={"hard_filters": hard_filters})


def _district_to_number(value: str) -> str | None:
    clean = value.strip().strip(".").casefold()
    if clean.isdigit():
        number = int(clean)
    else:
        number = ROMAN_DISTRICTS.get(clean)
        if number is None:
            return None
    if 1 <= number <= 23:
        return str(number)
    return None


def _with_budapest_district_default(text: str, parsed: ParsedQuery) -> ParsedQuery:
    """Extract explicit Budapest district mentions the LLM may miss.

    Negated mentions ("de nem a 8. kerületben", "a 8. kerület kivételével")
    are exclusions: they are never merged in, and they are stripped from the
    LLM's own `district_in` in case the model filled them despite the prompt.
    """
    normalized = _normalise(text)
    districts: list[str] = []
    negated: set[str] = set()
    for match in BUDAPEST_DISTRICT_RE.finditer(normalized):
        district = _district_to_number(match.group("district"))
        if not district:
            continue
        is_negated = bool(
            DISTRICT_NEGATION_BEFORE_RE.search(normalized[: match.start()])
            or DISTRICT_NEGATION_AFTER_RE.match(normalized[match.end() :])
        )
        if is_negated:
            negated.add(district)
        elif district not in districts:
            districts.append(district)

    hf = parsed.hard_filters
    if not districts and not (negated and hf.district_in):
        return parsed
    if hf.city is not None and hf.city.casefold() != "budapest":
        return parsed

    merged = [d for d in hf.district_in if d not in negated]
    for district in districts:
        if district not in merged:
            merged.append(district)

    update: dict[str, Any] = {"district_in": merged}
    if merged:
        # The Budapest city default only applies when an inclusive district
        # survives — stripping a negated district must not invent a city.
        update["city"] = hf.city or "Budapest"
    hard_filters = hf.model_copy(update=update)
    return parsed.model_copy(update={"hard_filters": hard_filters})


def _with_colloquial_budapest_default(text: str, parsed: ParsedQuery) -> ParsedQuery:
    """Treat colloquial Buda/Pest side mentions as Budapest-scoped searches.

    The side itself remains a geo hint for scoring; the hard city prevents an
    LLM parser miss from leaking non-Budapest inventory into "budai/pesti"
    searches.
    """
    if not BUDAPEST_SIDE_RE.search(_normalise(text)):
        return parsed

    hf = parsed.hard_filters
    if hf.city is not None:
        return parsed

    return parsed.model_copy(update={"hard_filters": hf.model_copy(update={"city": "Budapest"})})


def _with_location_resolution(text: str, parsed: ParsedQuery) -> ParsedQuery:
    """Attach stable KSH identity or an explicit ambiguity state to parsed geography."""

    # A county mention such as "Pest vármegyében" must not be reinterpreted as
    # the colloquial settlement alias Pest -> Budapest.
    if parsed.geo_hints.county is not None:
        return parsed

    resolved = resolve_location_text(text)
    if resolved.status is ResolutionStatus.NOT_FOUND:
        return parsed
    candidates = [
        LocationCandidate(settlement_id=item.id, name=item.name, county=item.county) for item in resolved.candidates
    ]
    state = LocationResolution(
        status=resolved.status.value,
        candidates=candidates,
        matched_text=resolved.matched_text,
    )
    hard_filters = parsed.hard_filters
    if resolved.status is ResolutionStatus.AMBIGUOUS:
        hard_filters = hard_filters.model_copy(update={"city": None, "city_id": None})
    else:
        settlement = resolved.settlement
        if settlement is not None:
            hard_filters = hard_filters.model_copy(update={"city": settlement.name, "city_id": settlement.id})
    return parsed.model_copy(update={"hard_filters": hard_filters, "location_resolution": state})


def _with_deterministic_defaults(text: str, parsed: ParsedQuery) -> ParsedQuery:
    return _with_location_resolution(
        text,
        _with_budapest_district_default(
            text,
            _with_colloquial_budapest_default(text, _with_transaction_default(text, parsed)),
        ),
    )


def fallback_parse_query(text: str) -> ParsedQuery:
    """Best-effort no-LLM parser used when the live parser exceeds latency budget."""
    parsed = deterministic_parse_loose(text) or ParsedQuery(semantic_query=text.strip())
    return _with_deterministic_defaults(text, parsed)


def _fast_path(text: str, locale: str) -> ParsedQuery | None:
    """The rule-based parse with the deterministic defaults, or None to ask the LLM."""
    # Deterministic fast path: simple, fully-explicit queries
    # ("eladó lakás Budapest 50 millió alatt") are parsed by rule, with no Gemini
    # call. It only fires when every token is accounted for; anything ambiguous
    # or carrying soft preferences falls through to the LLM unchanged.
    #
    # Hungarian only: every rule in it matches Hungarian word stems and suffixes, so
    # on a German or Polish sentence it can only be silent or wrong. Those go
    # straight to the model, which is the path that understands them.
    fast = deterministic_parse(text) if locale == DEFAULT_OUTPUT_LOCALE else None
    if fast is None:
        return None
    return _with_deterministic_defaults(text, fast)


@dataclass(frozen=True, slots=True)
class ParserSettings:
    """Which parser reads the query: the model, the prompt version, the reader's language.

    All three key the parser cache. `locale` is the site language the query arrives
    in; the parsed result is canonical Hungarian whatever it is.
    """

    model: str = TIER_S_PARSER
    prompt_version: str = PARSER_PROMPT_VERSION
    locale: str = DEFAULT_OUTPUT_LOCALE


DEFAULT_PARSER_SETTINGS: Final = ParserSettings()


@dataclass(frozen=True, slots=True)
class _LlmParse:
    """The LLM leg of one `parse_query` call: up to two attempts, one `EnrichmentLog` row each."""

    text: str
    client: LlmClient
    session: AsyncSession | None
    settings: ParserSettings

    def request(self, *, is_correction: bool) -> LlmRequest:
        return LlmRequest(
            model=self.settings.model,
            system=PARSER_CORRECTION_SYSTEM_PROMPT if is_correction else parser_system_prompt(self.settings.locale),
            user=(render_correction_user_prompt if is_correction else render_parser_user_prompt)(self.text),
            response_schema=ParsedQuery,
        )

    def log(self, req: LlmRequest, *, started: float, outcome: str, raw: str) -> None:
        if self.session is not None:
            log_llm_attempt(
                self.session, LlmAttempt(req, self.settings.prompt_version, started), outcome=outcome, raw=raw
            )

    async def attempt(self, *, is_correction: bool) -> ParsedQuery | ValidationError:
        """One LLM call: a schema failure comes back as the error, a transport failure propagates."""
        req = self.request(is_correction=is_correction)
        started = time.monotonic()
        try:
            reset_last_llm_usage()
            raw = await self.client.complete(req)
        except LlmError:
            self.log(req, started=started, outcome="transport_error", raw="")
            raise

        try:
            output = ParsedQuery.model_validate_json(raw)
        except ValidationError as exc:
            self.log(req, started=started, outcome="parse_error", raw=raw)
            return exc

        self.log(req, started=started, outcome="success", raw=raw)
        return output

    async def run(self) -> ParsedQuery:
        """The initial call plus one correction retry; `ParseError` when both fail the schema."""
        last_error: ValidationError | None = None
        for attempt in (1, 2):
            outcome = await self.attempt(is_correction=attempt > 1)
            if not isinstance(outcome, ValidationError):
                return outcome
            last_error = outcome
        raise ParseError(f"Query parse failed schema validation after 2 attempts: {last_error}") from last_error


async def parse_query(
    text: str,
    *,
    client: LlmClient,
    cache: ParserCache | None = None,
    session: AsyncSession | None = None,
    settings: ParserSettings = DEFAULT_PARSER_SETTINGS,
) -> ParsedQuery:
    """Parse a natural-language query into a validated `ParsedQuery`.

    `settings.locale` is the reader's site language: the query arrives in it, and the
    parser is told so. The result is canonical Hungarian in every language, so
    everything downstream (filters, scoring, the eval battery) is unaffected.

    On cache hit returns immediately without LLM contact. On miss makes up to
    two LLM calls (initial + correction) and writes one `EnrichmentLog` row per
    attempt. Persistent schema failure raises `ParseError`; transport failures
    propagate as `LlmError` after logging the failed attempt.
    """
    key = parser_cache_key(text, settings.model, settings.prompt_version, settings.locale)
    if cache is not None:
        cached = await cache.get(key)
        if cached is not None:
            _logger.info("parser cache hit", cache_key=key)
            return _with_deterministic_defaults(text, cached)
        _logger.info("parser cache miss", cache_key=key)

    fast = _fast_path(text, settings.locale)
    if fast is not None:
        _logger.info("parser fast path hit", cache_key=key)
        if cache is not None:
            await cache.set(key, fast)
        return fast

    llm = _LlmParse(text=text, client=client, session=session, settings=settings)
    output = _with_deterministic_defaults(text, await llm.run())

    if cache is not None:
        await cache.set(key, output)
        _logger.info("parser cache set", cache_key=key)
    return output
