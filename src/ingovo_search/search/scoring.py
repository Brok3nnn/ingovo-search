"""Scoring the candidates a search returns, and the inputs the rerankers read.

The card rows (`build_scored_candidates`, with the compromise labels and the public
amenity flags), the conditional-rerank decision (`should_rerank`), the factual packet
the rerankers read (`listing_rerank_summary`), the read-through query-embedding cache
(`embed_with_cache`) and the hero narratives (`attach_hero_narratives`). Moved here
from `ingovo_search.api.search_scoring`: none of it is HTTP.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import structlog
from redis.exceptions import RedisError

from ingovo_search.lib.cache import (
    Cache,
    embedding_cache_key,
    get_json,
    normalise_query,
    set_json,
)
from ingovo_search.lib.embedding_models import EMBED_MULTILINGUAL
from ingovo_search.lib.embeddings import EmbeddingClient, EmbeddingInputType
from ingovo_search.lib.env import get_settings
from ingovo_search.lib.llm import LlmClient
from ingovo_search.lib.property_price import public_price_fields, resolve_property_price_huf
from ingovo_search.recommender.match_percent import compute_match_breakdown
from ingovo_search.recommender.narrator import DEFAULT_NARRATOR_LOCALE, Narrator, NarratorSettings, generate_narrative
from ingovo_search.recommender.parser_schema import HardFilters, ParsedQuery
from ingovo_search.recommender.score import ScoredCandidate
from ingovo_search.recommender.search_contract import (
    ResultScope,
)
from ingovo_search.search.constants import (
    HERO_NARRATIVE_COUNT,
    QUERY_EMBEDDING_CACHE_TTL_SECONDS,
    RERANK_DESCRIPTION_MAX_CHARS,
    SOFT_PREF_FIELD,
    narrator_logger,
    search_logger,
)
from ingovo_search.search.context import format_huf, format_number
from ingovo_search.search.embedding_cache import CachedEmbedding
from ingovo_search.search.run import elapsed_ms

# Module-local on purpose: ruff's blind-except exemption only recognises a logger
# defined here, not one imported from another module, and the structlog
# contextvars put the request id on every line. The logger keeps its name from
# `ingovo_search.api.search_scoring`, so the log lines read as before the move.
_logger = structlog.get_logger("ingovo_search.api.search_scoring")

RERANK_MIN_CANDIDATES: int = 4
RERANK_CLEAR_LEAD_FRACTION: float = 0.12


def should_rerank(parsed: ParsedQuery, scored: list[tuple[Any, Any]]) -> bool:
    """Conditional rerank.

    The LLM rerank is the single biggest latency cost on a cold search (~3-5 s).
    It earns its keep on rich/ambiguous intent, but adds little when the request
    is simple and the deterministic top is already a clear winner. Return False
    to skip it in those safe cases; stay conservative — when in doubt, rerank.
    """
    if len(scored) < RERANK_MIN_CANDIDATES:
        return False
    # Rich or ambiguous intent → rerank genuinely reorders things.
    if len(parsed.soft_preferences) >= 2 or parsed.soft_prefs_freetext:
        return True
    if len((parsed.semantic_query or "").split()) >= 4:
        return True
    # Simple intent: trust the deterministic order only when #1 clearly leads #2;
    # a tight top cluster is exactly where the reranker helps, so rerank then.
    top = scored[0][1].total
    second = scored[1][1].total
    if top <= 0.0:
        return False
    return (top - second) / top < RERANK_CLEAR_LEAD_FRACTION


async def embed_with_cache(  # noqa: PLR0913 -- keyword-only; the search pipeline and the evals both call it
    *,
    text_value: str,
    input_type: EmbeddingInputType,
    embedding_client: EmbeddingClient,
    cache: Cache,
    timings: dict[str, int],
    timing_prefix: str,
) -> list[float]:
    """Read-through cache for runtime query embeddings.

    The cache key uses the embedding model, input type, and normalised text.
    This preserves quality because the vector is identical to what Cohere would
    return for the same input; it only avoids repeated network calls.
    """
    normalised = normalise_query(text_value)
    key = embedding_cache_key(normalised, model=EMBED_MULTILINGUAL, input_type=input_type)

    step_started = time.perf_counter()
    cached = await get_json(cache, key, CachedEmbedding)
    timings[f"{timing_prefix}_cache_read"] = elapsed_ms(step_started)
    if cached is not None:
        timings[f"{timing_prefix}_cache_hit"] = 1
        timings[timing_prefix] = 0
        return cached.embedding

    timings[f"{timing_prefix}_cache_hit"] = 0
    step_started = time.perf_counter()
    embedding = await embedding_client.embed(text_value, input_type)
    timings[timing_prefix] = elapsed_ms(step_started)

    step_started = time.perf_counter()
    await set_json(
        cache,
        key,
        CachedEmbedding(embedding=embedding),
        ttl_seconds=QUERY_EMBEDDING_CACHE_TTL_SECONDS,
    )
    timings[f"{timing_prefix}_cache_write"] = elapsed_ms(step_started)
    return embedding


def _rooms_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _transaction_label(transaction_type: str | None) -> str:
    """Human-readable Hungarian label for the `sale`/`rent` enum."""
    return "Kiadó" if str(transaction_type).strip().lower() == "rent" else "Eladó"


def _renovation_compromise(requested: set[str], prop: Any) -> str | None:
    """Renovation mismatch from the structured `renovation_state` enum.

    Vocabulary (text_enrich): original / partial / fully_renovated / new_build.
    Only a *proven* mismatch is surfaced — when the state is unknown we stay
    silent rather than printing an "unverified" badge that reads like a bug.
    """
    if not ({"renovated", "partially_renovated", "new_build"} & requested):
        return None
    state = (getattr(prop, "renovation_state", None) or "").strip().lower()
    if not state:
        return None
    if "new_build" in requested:
        return "Nem új építésű" if state != "new_build" else None
    # renovated / partially_renovated
    if state == "original":
        return "Felújítatlan"
    if state == "partial" and "renovated" in requested and "partially_renovated" not in requested:
        return "Csak részben felújított"
    return None


def _listing_kind_compromises(prop: Any, hard: HardFilters) -> list[str]:
    """The transaction and property-type mismatches, in `compromise_labels` order."""
    labels: list[str] = []
    property_type = getattr(prop, "property_type", None)
    transaction_type = getattr(prop, "transaction_type", None)
    if hard.transaction_type is not None and transaction_type is not None and transaction_type != hard.transaction_type:
        labels.append(
            f"{_transaction_label(transaction_type)} ({_transaction_label(hard.transaction_type).lower()} helyett)"
        )
    # Only flag a type mismatch we can actually prove: an unknown type is left
    # unlabelled rather than printed as "nincs igazolva".
    if hard.property_type_in and property_type and property_type not in hard.property_type_in:
        requested_types = " / ".join(t.lower() for t in hard.property_type_in)
        labels.append(f"{property_type} ({requested_types} helyett)")
    if hard.property_type_exclude and property_type and property_type in hard.property_type_exclude:
        labels.append(f"{property_type} - kizárt típus")
    return labels


def _measure_compromises(prop: Any, hard: HardFilters) -> list[str]:
    """The price, size, room-count and floor mismatches, in `compromise_labels` order."""
    labels: list[str] = []
    price = resolve_property_price_huf(prop)
    area = getattr(prop, "area_m2", None)
    rooms = _rooms_number(getattr(prop, "rooms", None))
    floor = getattr(prop, "floor", None)
    if hard.price_huf_max is not None and price is not None and price > hard.price_huf_max:
        labels.append(f"A keretnél {format_huf(round(price - hard.price_huf_max))}-tal drágább")
    if hard.price_huf_min is not None and price is not None and price < hard.price_huf_min:
        labels.append(f"A megadott minimumnál {format_huf(round(hard.price_huf_min - price))}-tal olcsóbb")
    if hard.area_m2_min is not None and area is not None and area < hard.area_m2_min:
        labels.append(f"A kértnél {format_number(hard.area_m2_min - area)} m²-rel kisebb")
    if hard.area_m2_max is not None and area is not None and area > hard.area_m2_max:
        labels.append(f"A kértnél {format_number(area - hard.area_m2_max)} m²-rel nagyobb")
    if hard.rooms is not None and rooms is not None and rooms != float(hard.rooms):
        difference = abs(rooms - hard.rooms)
        direction = "több" if rooms > hard.rooms else "kevesebb"
        labels.append(f"{format_number(difference)} szobával {direction}")
    elif hard.rooms_min is not None and rooms is not None and rooms < float(hard.rooms_min):
        labels.append(f"{format_number(hard.rooms_min - rooms)} szobával kevesebb")
    if hard.floor_min is not None and floor is not None and floor < hard.floor_min:
        labels.append(f"A kértnél alacsonyabb emelet: {floor}")
    return labels


def _soft_preference_compromises(prop: Any, parsed: ParsedQuery) -> list[str]:
    """The amenities asked for that the listing provably lacks, then the renovation mismatch."""
    labels: list[str] = []
    requested = set(parsed.soft_preferences or [])
    for tag in parsed.soft_preferences or []:
        mapping = SOFT_PREF_FIELD.get(tag)
        if mapping is None:
            continue
        field, noun = mapping
        value = getattr(prop, field, None)
        # Only a definite absence is a compromise; missing data stays silent.
        if value is False:
            labels.append(f"Nincs {noun}")
    renovation = _renovation_compromise(cast("set[str]", requested), prop)
    if renovation is not None:
        labels.append(renovation)
    return labels


def compromise_labels(prop: Any, parsed: ParsedQuery) -> list[str]:
    """Every requested condition this listing does NOT satisfy, human-readable.

    Location (city/district) is intentionally omitted: a different-location
    listing is surfaced as a NEARBY result and the card labels its actual
    location + distance separately, so repeating it here would be redundant.
    """
    hard = parsed.hard_filters
    return [
        *_listing_kind_compromises(prop, hard),
        *_measure_compromises(prop, hard),
        *_soft_preference_compromises(prop, parsed),
    ]


_PUBLIC_FEATURE_FIELDS: tuple[str, ...] = (
    "has_balcony",
    "has_terrace",
    "has_parking",
    "has_garage",
    "has_garden",
    "has_basement",
    "has_pool",
    "elevator",
    "air_conditioning",
)


def public_property_features(prop: Any) -> dict[str, Any]:
    """Merge enriched amenity flags into the property payload used by cards."""
    raw_features = getattr(prop, "features", None)
    features: dict[str, Any] = dict(cast("dict[str, Any]", raw_features)) if isinstance(raw_features, dict) else {}
    for field in _PUBLIC_FEATURE_FIELDS:
        value = getattr(prop, field, None)
        if value is not None:
            # The enriched, typed DB column is authoritative when both sources
            # carry the same amenity.
            features[field] = value
    return features


def build_scored_candidates(
    scored: list[tuple[Any, Any]],
    parsed: ParsedQuery,
    match_parsed: ParsedQuery | None = None,
) -> list[ScoredCandidate]:
    match_query = match_parsed or parsed
    results: list[ScoredCandidate] = []
    for prop, breakdown in scored:
        match = compute_match_breakdown(prop, breakdown, match_query)
        result_scope = match.result_scope or ResultScope.EXACT
        distance_km = match.distance_km
        if result_scope is ResultScope.OUT_OF_SCOPE:
            if not match_query.allow_distant_location_expansion:
                # R11.7 — alert on a remote settlement reaching public scoring
                # without an explicit distance expansion. Greppable event name
                # for a monitoring rule (`event="alert.remote_settlement_included"`).
                search_logger.error(
                    "alert.remote_settlement_included",
                    listing_id=prop.id,
                    listing_city=getattr(prop, "city", None),
                    distance_km=distance_km,
                    requested_city=match_query.hard_filters.city,
                )
                raise ValueError(f"out-of-scope candidate reached public scoring: {prop.id}")
            # The user explicitly replaced the original location boundary with
            # a farther scope; within that approved scope the candidate is exact.
            result_scope = ResultScope.EXACT
            distance_km = None
        price_huf, price = public_price_fields(prop)
        results.append(
            ScoredCandidate(
                listing_id=prop.id,
                score=breakdown.total,
                match_percent=match.match_percent,
                result_scope=result_scope,
                distance_km=distance_km,
                match_breakdown=match,
                # Projected from the same `match` object, but only the criterion
                # scores travel — the raw scope/distance above were just
                # sanitised, and the nested breakdown must not undo that.
                match_components=match.public_components(),
                breakdown=breakdown,
                district=prop.district,
                city=prop.city,
                compromise=(compromises[0] if (compromises := compromise_labels(prop, match_query)) else None),
                compromises=compromises,
                property={
                    "id": prop.id,
                    "title": prop.title,
                    # Stored machine translations. The card picks the
                    # reader's locale out of this and falls back to `title`, so a
                    # listing published before the translation job ran still renders.
                    "title_translations": getattr(prop, "title_translations", None),
                    "price": price,
                    "price_huf": price_huf,
                    "type": prop.property_type,
                    "transaction_type": prop.transaction_type,
                    "area": prop.area_m2,
                    "rooms": prop.rooms,
                    "city": prop.city,
                    "district": prop.district,
                    "address": prop.address,
                    # Explicit null means the location was checked and is absent;
                    # clients must not refetch the full listing to discover that.
                    "latitude": getattr(prop, "latitude", None),
                    "longitude": getattr(prop, "longitude", None),
                    "images": prop.images or [],
                    "features": public_property_features(prop),
                },
            )
        )
    return results


def listing_rerank_summary(prop: Any) -> str:
    """Compact factual packet for the LLM reranker.

    The reranker should choose from the already-filtered candidates, but it
    needs more than a raw description prefix to avoid irrelevant top picks.
    """
    facts: list[str] = []
    if title := getattr(prop, "title", None):
        facts.append(f"Cím: {title}")
    if transaction_type := getattr(prop, "transaction_type", None):
        facts.append("Tranzakció: eladó" if transaction_type == "sale" else "Tranzakció: kiadó")
    if property_type := getattr(prop, "property_type", None):
        facts.append(f"Típus: {property_type}")
    location = ", ".join(bit for bit in (getattr(prop, "city", None), getattr(prop, "district", None)) if bit)
    if location:
        facts.append(f"Hely: {location}")
    if (price := resolve_property_price_huf(prop)) is not None:
        facts.append(f"Ár: {int(price):,} Ft".replace(",", " "))
    if (area_m2 := getattr(prop, "area_m2", None)) is not None:
        facts.append(f"Méret: {area_m2:g} m2")
    if rooms := getattr(prop, "rooms", None):
        facts.append(f"Szobák: {rooms}")

    summary = (
        (getattr(prop, "short_summary", None) or "")
        or (getattr(prop, "one_paragraph_summary", None) or "")
        or (getattr(prop, "cleaned_description", None) or "")
        or (getattr(prop, "description", None) or "")
    )
    if summary:
        trimmed = str(summary).strip().replace("\n", " ")[:RERANK_DESCRIPTION_MAX_CHARS]
        facts.append("Leírás: " + trimmed)
    return " | ".join(facts)


async def attach_hero_narratives(  # noqa: PLR0913 -- the scored rows, the query and the narrator's client, cache, locale
    results: list[ScoredCandidate],
    scored: list[tuple[Any, Any]],
    parsed: ParsedQuery,
    *,
    client: LlmClient,
    cache: Cache,
    locale: str = DEFAULT_NARRATOR_LOCALE,
) -> list[ScoredCandidate]:
    """Generate §17.3 AI narratives for the top hero rows; merge in-place.

    The narratives land on the first `HERO_NARRATIVE_COUNT` candidates so the
    buyer UI can explain the entire top-10 recommendation set. They run
    concurrently via `asyncio.gather` so the cold-cache cost is one round-trip,
    not ten sequential calls. Per-row failures are caught and logged — the
    result list survives even if one narrative misbehaves, since the UI can
    fall back to `rationale` + score breakdown when `ai_narrative is None`.
    """
    if not results:
        return results

    breakdowns_by_id: dict[str, Any] = {prop.id: bd for prop, bd in scored}
    listings_by_id: dict[str, Any] = {prop.id: prop for prop, _ in scored}

    hero_slice = results[:HERO_NARRATIVE_COUNT]
    narrator = Narrator(client, cache, NarratorSettings(locale=locale))

    async def _one(candidate: ScoredCandidate) -> tuple[str, Any]:
        listing = listings_by_id.get(candidate.listing_id)
        breakdown = breakdowns_by_id.get(candidate.listing_id)
        if listing is None or breakdown is None:
            return candidate.listing_id, None
        try:
            narrative = await generate_narrative(listing, parsed, breakdown, narrator)
        except RedisError as exc:
            # Expected noise — the narrator cache is down; the row goes out without a narrative.
            _logger.warning("narrator cache unavailable", listing_id=candidate.listing_id, error=type(exc).__name__)
            return candidate.listing_id, None
        except Exception:
            # Deliberately broad: `generate_narrative` already turns LLM failures
            # into its heuristic narrative, so what reaches here is a stale cached
            # shape or a bug on one odd listing. The hero narratives run under a
            # single `asyncio.gather`, where one leaked exception would fail the
            # whole search; this row loses its narrative instead, logged at error
            # with the traceback so the bug stays visible.
            _logger.exception("narrator failed", listing_id=candidate.listing_id)
            return candidate.listing_id, None
        return candidate.listing_id, narrative.model_dump()

    # Bounded: see AI_SEARCH_NARRATOR_TIMEOUT_MS. A hung (not failing) LLM socket
    # here used to have no ceiling and could hold the whole search request open.
    narrator_timeout_ms = get_settings().ai_search_narrator_timeout_ms
    try:
        gathered = await asyncio.wait_for(
            asyncio.gather(*(_one(c) for c in hero_slice)),
            timeout=narrator_timeout_ms / 1000,
        )
        narratives = dict(gathered)
    except TimeoutError:
        narrator_logger.warning(
            "narrator phase timed out after %sms; serving rows without ai_narrative",
            narrator_timeout_ms,
        )
        narratives = {}

    updated: list[ScoredCandidate] = []
    for candidate in results:
        narrative_dump = narratives.get(candidate.listing_id)
        if narrative_dump is not None:
            updated.append(candidate.model_copy(update={"ai_narrative": narrative_dump}))
        else:
            updated.append(candidate)
    return updated
