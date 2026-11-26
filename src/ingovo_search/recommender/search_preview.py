"""Semantic homepage preview built from the canonical :class:`ParsedQuery`.

The homepage used to maintain a second, regex-only interpretation stack.  This
module keeps presentation generation pure and deterministic while accepting the
same structured query that drives the real recommender.

It is split four ways: the response models are in `preview_models`, the
strict-constraint summary in `preview_summary`, the image concepts and search
directions in `preview_suggestions`. This module keeps the parser guardrail merge,
the confidence score and `build_search_preview`, and re-exports the public names.
"""

from __future__ import annotations

import hashlib

from ingovo_search.recommender.parser_schema import HardFilters, ParsedQuery
from ingovo_search.recommender.preview_models import ImageConcept, PreviewLocale, SearchDirection, SearchPreview
from ingovo_search.recommender.preview_suggestions import build_directions, build_image_concepts
from ingovo_search.recommender.preview_summary import constraint_summary, constraint_summary_with_keys


def merge_missing_preview_fields(parsed: ParsedQuery, deterministic: ParsedQuery) -> ParsedQuery:
    """Fill only fields the semantic parser left empty.

    The LLM remains authoritative for nuanced ranges and priorities.  The
    deterministic parser is used as a guardrail for explicit facts such as an
    inflected Hungarian property type (for example ``lakást``), which should
    never disappear from the live typing preview.
    """

    hard = parsed.hard_filters
    fallback_hard = deterministic.hard_filters
    hard_updates: dict[str, object] = {}

    for field in ("city", "city_id", "transaction_type", "floor_min", "near"):
        if getattr(hard, field) is None and getattr(fallback_hard, field) is not None:
            hard_updates[field] = getattr(fallback_hard, field)

    for field in ("district_in", "property_type_in", "property_type_exclude"):
        if not getattr(hard, field) and getattr(fallback_hard, field):
            hard_updates[field] = getattr(fallback_hard, field)

    hard_updates.update(_missing_rooms_and_ranges(hard, fallback_hard))

    anchors = parsed.anchors
    fallback_anchors = deterministic.anchors
    anchor_updates = {
        field: getattr(fallback_anchors, field)
        for field in ("price_huf", "area_m2", "rooms")
        if getattr(anchors, field) is None and getattr(fallback_anchors, field) is not None
    }

    geo = parsed.geo_hints
    fallback_geo = deterministic.geo_hints
    geo_updates = {
        field: getattr(fallback_geo, field)
        for field in ("region_name", "side_of_river", "county", "balaton_shore")
        if getattr(geo, field) is None and getattr(fallback_geo, field) is not None
    }

    return parsed.model_copy(
        update={
            "hard_filters": hard.model_copy(update=hard_updates),
            "anchors": anchors.model_copy(update=anchor_updates),
            "geo_hints": geo.model_copy(update=geo_updates),
        }
    )


def _missing_rooms_and_ranges(hard: HardFilters, fallback_hard: HardFilters) -> dict[str, object]:
    """The room count and the area and price ranges, each taken only when the parser set no part of it."""
    updates: dict[str, object] = {}
    if hard.rooms is None and hard.rooms_min is None:
        if fallback_hard.rooms is not None:
            updates["rooms"] = fallback_hard.rooms
        elif fallback_hard.rooms_min is not None:
            updates["rooms_min"] = fallback_hard.rooms_min

    for minimum, maximum in (("area_m2_min", "area_m2_max"), ("price_huf_min", "price_huf_max")):
        if getattr(hard, minimum) is None and getattr(hard, maximum) is None:
            if getattr(fallback_hard, minimum) is not None:
                updates[minimum] = getattr(fallback_hard, minimum)
            if getattr(fallback_hard, maximum) is not None:
                updates[maximum] = getattr(fallback_hard, maximum)
    return updates


def _confidence(parsed: ParsedQuery, query: str) -> float:
    hard = parsed.hard_filters
    recognized = sum(
        value is not None and value != []
        for value in (
            hard.transaction_type,
            hard.property_type_in,
            hard.city,
            hard.district_in,
            hard.rooms,
            hard.rooms_min,
            hard.area_m2_min,
            hard.area_m2_max,
            hard.price_huf_min,
            hard.price_huf_max,
        )
    )
    signal_count = len(parsed.soft_preferences) + len(parsed.soft_prefs_freetext) + len(parsed.poi_near)
    value = min(0.97, 0.32 + recognized * 0.09 + min(signal_count, 5) * 0.055 + min(len(query), 120) / 1200)
    if parsed.conflict_flag:
        value = min(value, 0.45)
    return round(value, 2)


def build_search_preview(query: str, locale: PreviewLocale, parsed: ParsedQuery, request_id: str) -> SearchPreview:
    """Build the stable 8+5 homepage payload from a canonical parsed query."""

    summary, _ = constraint_summary_with_keys(parsed, locale)
    confidence = _confidence(parsed, query)
    clarification: str | None = None
    if parsed.conflict_flag:
        clarification = (
            "A feltételek között ellentmondást találtam. Melyik fontosabb?"
            if locale == "hu"
            else "I found conflicting requirements. Which one matters more?"
        )
    elif confidence < 0.52 and len(query.strip()) >= 10:
        clarification = (
            "Pontosítanád a helyszínt vagy az ingatlantípust?"
            if locale == "hu"
            else "Could you clarify the location or property type?"
        )
    intent_hash = hashlib.sha256(parsed.model_dump_json().encode("utf-8")).hexdigest()[:20]
    return SearchPreview(
        request_id=request_id,
        intent_hash=intent_hash,
        confidence=confidence,
        parsed_query=parsed,
        understood_summary=summary,
        image_concepts=build_image_concepts(parsed, query, locale),
        search_directions=build_directions(parsed, query, locale),
        clarification=clarification,
    )


__all__ = [
    "ImageConcept",
    "PreviewLocale",
    "SearchDirection",
    "SearchPreview",
    "build_search_preview",
    "constraint_summary",
]
