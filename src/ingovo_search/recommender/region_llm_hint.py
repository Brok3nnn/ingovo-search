"""§10 cold-start region hint — Tier-S LLM fallback for fuzzy region names.

Before a named region (e.g. "Budai-hegyvidék") has ~50 positive engagements,
there is no learned centroid to score against. As a fallback the parser-tier
LLM suggests likely formal districts as a weighted hint
(`[{district, weight}, ...]` — the design spec §6.4 "Learned centroids").
The hint biases `geo_score`, never filters.

The hint is generated once per new region and cached in
`RegionCentroid.llm_hint_districts`; `region_centroid.matches_named_region`
owns that caching. This module owns the LLM call, its schema, and the pure
`hint_district_score` indicator.

Bump `REGION_HINT_PROMPT_VERSION` on any prompt change.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.lib.llm import LlmClient, LlmRequest
from ingovo_search.lib.llm_models import TIER_S_PARSER
from ingovo_search.lib.llm_usage import reset_last_llm_usage
from ingovo_search.recommender.llm_attempt_log import LlmAttempt, log_llm_attempt
from ingovo_search.types.errors import LlmError, ParseError

REGION_HINT_PROMPT_VERSION = "region_hint/v1-final"

REGION_HINT_SYSTEM_PROMPT = """A feladatod egy nem hivatalos, elmosódott határokkal rendelkező magyar földrajzi elnevezés (pl. régió, városrész, gyűjtőnév) leképezése azokra a hivatalos budapesti kerületekre vagy településekre, amelyekre a legvalószínűbben vonatkozik.

Egyetlen JSON objektumot kell visszaadnod, amely pontosan illeszkedik a RegionHint sémára. Ez egy `districts` (kerületek/települések) tömböt tartalmaz, benne `{district, weight}` párokkal, ahol a `weight` 0 és 1 közötti valószínűségi súly (pl. 0.4 = 40%), és az összes súly összege nagyjából 1-et ad ki. 

Példa: Ha a bemenet "Belváros", akkor az "V. kerület", "VI. kerület", "VII. kerület" stb. kapjanak magas súlyokat elosztva.

Kizárólag érvényes JSON formátumban válaszolj, ne fűzz hozzá semmilyen kísérőszöveget vagy Markdown formázást!
"""

REGION_HINT_CORRECTION_SYSTEM_PROMPT = """Az előző válaszod sémaellenőrzése (schema validation) sikertelen volt. Kérlek, generáld le újra a választ egyetlen JSON objektumban, amely pontosan illeszkedik a RegionHint sémára. Ne fűzz hozzá semmilyen magyarázatot, csak a nyers JSON-t add vissza."""


class RegionHintDistrict(BaseModel):
    """One weighted district guess for a cold-start region name."""

    model_config = {"frozen": True}

    district: str
    weight: float = Field(ge=0.0, le=1.0)


def _empty_district_list() -> list[RegionHintDistrict]:
    """Typed empty-list factory for the `districts` field default."""
    return []


class RegionHint(BaseModel):
    """LLM-suggested district distribution for a fuzzy region name."""

    model_config = {"frozen": True}

    districts: list[RegionHintDistrict] = Field(default_factory=_empty_district_list)


def render_region_hint_user_prompt(region_name: str) -> str:
    """User prompt for the initial hint call — region name verbatim."""
    return region_name


def render_correction_user_prompt(region_name: str) -> str:
    """User prompt for the retry-with-correction call."""
    return f"[CORRECT_SCHEMA]\n{region_name}"


def hint_district_score(property_district: str | None, hint_districts: list[RegionHintDistrict]) -> float:
    """Weighted indicator: the hint weight for the listing's district, else 0.0."""
    if property_district is None:
        return 0.0
    for hint in hint_districts:
        if hint.district == property_district:
            return hint.weight
    return 0.0


async def region_llm_hint(
    region_name: str,
    *,
    client: LlmClient,
    session: AsyncSession | None = None,
    model: str = TIER_S_PARSER,
) -> RegionHint:
    """Generate a weighted district hint for a cold-start region name.

    Makes up to two LLM calls (initial + correction on schema failure), writes
    one `EnrichmentLog` row per attempt. Raises `ParseError` on persistent
    schema failure; `LlmError` propagates on transport failure.
    """
    last_error: ValidationError | None = None
    hint: RegionHint | None = None

    for attempt in (1, 2):
        is_correction = attempt > 1
        req = LlmRequest(
            model=model,
            system=REGION_HINT_CORRECTION_SYSTEM_PROMPT if is_correction else REGION_HINT_SYSTEM_PROMPT,
            user=(render_correction_user_prompt if is_correction else render_region_hint_user_prompt)(region_name),
            response_schema=RegionHint,
        )
        attempt_log = LlmAttempt(req, REGION_HINT_PROMPT_VERSION, time.monotonic())
        try:
            reset_last_llm_usage()
            raw = await client.complete(req)
        except LlmError:
            if session is not None:
                log_llm_attempt(session, attempt_log, outcome="transport_error", raw="")
            raise

        try:
            hint = RegionHint.model_validate_json(raw)
        except ValidationError as exc:
            last_error = exc
            if session is not None:
                log_llm_attempt(session, attempt_log, outcome="parse_error", raw=raw)
            continue

        if session is not None:
            log_llm_attempt(session, attempt_log, outcome="success", raw=raw)
        break

    if hint is None:
        raise ParseError(f"region hint for '{region_name}' failed schema validation after 2 attempts: {last_error}")
    return hint
