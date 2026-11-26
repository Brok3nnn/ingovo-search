"""§17.3 AI narrator — natural-language explanation per result.

Generates the "AI ingatlanügynök" ("AI estate agent") voice the search-results UX
is built around. Each top-N candidate gets a small JSON
narrative blob attached to its envelope item:

- `headline`: one-sentence hook tied to the user's query phrasing
- `body`: 2-3 sentence personalised explanation
- `compromise`: 1-2 sentence honest trade-off (or empty if perfect match)
- `rationale_chips`: 4 high-level groups (location / size / price / renovation)
  with `match | warning | miss` status — already user-relative, never raw score
- `vs_next`: optional one-line comparison to the next-ranked candidate

The narrator is intentionally `LlmRequest`-driven and cache-keyed on
(property, query, breakdown) so warm-path latency stays sub-100 ms. The
prompt itself is locked behind `NARRATOR_PROMPT_VERSION`, so any tone-change
invalidates the cache automatically — same pattern as `parser_prompt.py`.

The narrator family mirrors the parser's: `narrator_schema.py`
holds the narrative contract, `narrator_prompt.py` what the model is told, and
`narrator_fallback.py` the deterministic narrative used when the model fails.
This module owns the call itself — the cache key, the two attempts, the
best-effort cache write-through and the backend-owned `vs_next`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

import structlog

from ingovo_search.lib.cache import Cache, get_json, set_json
from ingovo_search.lib.llm import LlmClient, LlmRequest
from ingovo_search.lib.llm_models import TIER_S_PARSER
from ingovo_search.recommender.narrator_fallback import fallback_narrative
from ingovo_search.recommender.narrator_prompt import (
    DEFAULT_NARRATOR_LOCALE,
    NARRATOR_PROMPT_VERSION,
    narrator_system_prompt,
    render_user_prompt,
)
from ingovo_search.recommender.narrator_schema import AINarrative, VsNext
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.score import ScoreBreakdown

_logger = structlog.get_logger("ingovo_search.recommender.narrator")

# Narrator outputs go through a cheap, fast model — the prompt is short and
# the structure tightly constrained, so the smallest Tier-S size suffices.
NARRATOR_MODEL: str = TIER_S_PARSER

# 30-day TTL: the narrative is keyed on the exact property + parsed-query +
# rounded breakdown, so it stays valid until any of those change. Listings
# rarely change after publication; queries cache-miss the first time and hit
# every subsequent identical search.
NARRATOR_CACHE_TTL_SECONDS = 60 * 60 * 24 * 30


@dataclass(frozen=True, slots=True)
class NarratorSettings:
    """Which narrator writes: the model, the prompt version and the language — all three key the cache."""

    model: str = NARRATOR_MODEL
    prompt_version: str = NARRATOR_PROMPT_VERSION
    locale: str = DEFAULT_NARRATOR_LOCALE


DEFAULT_NARRATOR_SETTINGS: Final = NarratorSettings()


@dataclass(frozen=True, slots=True)
class Narrator:
    """Who writes the narratives of one search: the model client, its cache and the settings."""

    client: LlmClient
    cache: Cache | None = None
    settings: NarratorSettings = DEFAULT_NARRATOR_SETTINGS


def narrator_cache_key(
    property_id: str,
    parsed_query_hash: str,
    breakdown: ScoreBreakdown,
    settings: NarratorSettings = DEFAULT_NARRATOR_SETTINGS,
) -> str:
    """Cache key keyed on the exact inputs that affect the narrative output.

    The breakdown is rounded to 3 decimals before hashing — sub-permille
    differences are noise and would needlessly invalidate the cache.
    """
    breakdown_compact = json.dumps(
        {k: round(v, 3) for k, v in breakdown.model_dump().items()},
        sort_keys=True,
        ensure_ascii=False,
    )
    payload = (
        f"{settings.model}\x00{settings.prompt_version}\x00{settings.locale}\x00"
        f"{property_id}\x00{parsed_query_hash}\x00{breakdown_compact}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"narrator:{digest}"


def _parsed_query_hash(query: ParsedQuery) -> str:
    """Stable hash over the parsed query — used inside the narrator cache key."""
    payload = query.model_dump_json()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coerce_narrative(raw: str) -> AINarrative:
    """Parse + validate the LLM's JSON output, raising ValidationError on schema miss."""
    # The prompt asks for raw JSON, but the model occasionally wraps in ```json fences.
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    payload: dict[str, Any] = json.loads(text)
    # Strip any `vs_next` the LLM might have hallucinated — the backend owns that field.
    payload.pop("vs_next", None)
    return AINarrative.model_validate(payload)


async def generate_narrative(
    listing: Any,
    query: ParsedQuery,
    breakdown: ScoreBreakdown,
    narrator: Narrator,
) -> AINarrative:
    """Return an `AINarrative` for one ranked listing.

    Cache-first: on hit the LLM is not contacted. On miss the prompt is sent
    once; a single retry handles transient JSON-schema misses. After both
    LLM attempts fail (transport error, persistent schema miss, anything),
    we return a deterministic heuristic narrative so the hero card NEVER
    falls back to a "narrative unavailable" placeholder. This keeps the
    product working under Gemini outages.
    """
    client, cache, settings = narrator.client, narrator.cache, narrator.settings
    model = settings.model
    pq_hash = _parsed_query_hash(query)
    key = narrator_cache_key(str(listing.id), pq_hash, breakdown, settings)

    if cache is not None:
        cached = await get_json(cache, key, AINarrative)
        if cached is not None:
            return cached

    user_prompt = render_user_prompt(listing, query, breakdown)
    for attempt in (1, 2):
        if attempt == 2:
            user_prompt += (
                "\n\nFIGYELEM: az előző válasz sémahibás volt. Csak érvényes JSON-t adj vissza,"
                " pontosan a megadott séma szerint, semmi extra szöveggel."
            )
        req = LlmRequest(model=model, system=narrator_system_prompt(settings.locale), user=user_prompt, temperature=0.2)
        narrative = await _attempt_narrative(client, req, attempt=attempt, model=model, listing=listing)
        if narrative is None:
            continue
        if cache is not None:
            # Outside the attempt: a cache outage used to count as a failed attempt,
            # which cost a second LLM call and then threw a valid narrative away for
            # the heuristic. The narrative is in hand; the write is best-effort.
            await _store_narrative(cache, key, narrative, model=model, listing=listing)
        return narrative

    _logger.warning(
        "narrator falling back to heuristic narrative",
        model=model,
        listing_id=getattr(listing, "id", None),
    )
    # Both LLM attempts failed (transport, schema, JSON, anything) — fall back
    # to a heuristic narrative so the UI never has to render a "missing" hero.
    # We intentionally DO NOT cache the fallback: a transient Gemini outage
    # should not poison the cache for 30 days.
    return fallback_narrative(listing, query)


async def _attempt_narrative(
    client: LlmClient,
    req: LlmRequest,
    *,
    attempt: int,
    model: str,
    listing: Any,
) -> AINarrative | None:
    """One narrator call: the validated narrative, or None once the failure is logged."""
    try:
        raw = await client.complete(req)
        narrative = _coerce_narrative(raw)
    except Exception as exc:
        # Fail-soft by contract (see `generate_narrative`): the hero card must render
        # under any LLM failure — an LlmError, the raw provider or transport
        # error `GoogleAiClient` re-raises once its fallback chain is spent, or
        # the JSONDecodeError/ValidationError of a schema miss. Previously
        # `continue` with no logging, which made this the only LLM phase whose
        # failures were completely invisible: the parser and the rerank both
        # log every attempt. A Gemini outage, an auth error or a run of
        # malformed JSON would quietly turn EVERY hero card into the generic
        # heuristic fallback with nothing in the logs to say so — the AI
        # narration silently stopped being AI narration.
        _logger.warning(
            "narrator attempt failed",
            attempt=attempt,
            model=model,
            listing_id=getattr(listing, "id", None),
            error_type=type(exc).__name__,
            error=str(exc)[:200],
            exc_info=exc,
        )
        return None
    return narrative


async def _store_narrative(cache: Cache, key: str, narrative: AINarrative, *, model: str, listing: Any) -> None:
    """Best-effort write-through of a generated narrative under its 30-day TTL."""
    try:
        await set_json(cache, key, narrative, ttl_seconds=NARRATOR_CACHE_TTL_SECONDS)
    except Exception as exc:
        # `Cache` is a Protocol; its backends raise their own families, and
        # none of them may cost the reader the narrative already generated.
        _logger.warning(
            "narrator cache write failed",
            model=model,
            listing_id=getattr(listing, "id", None),
            error_type=type(exc).__name__,
            error=str(exc)[:200],
            exc_info=exc,
        )


def attach_vs_next(narrative: AINarrative, *, compared_to: str, summary: str) -> AINarrative:
    """Return a new `AINarrative` with `vs_next` filled — owned by the backend, not the LLM."""
    return narrative.model_copy(update={"vs_next": VsNext(compared_to=compared_to, summary=summary)})
