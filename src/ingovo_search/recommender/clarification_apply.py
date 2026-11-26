"""§14 step 7 — applying a clarification answer.

`apply_clarification(parsed, answer)` mutates a copy of `ParsedQuery` based on
the answer's structured `option_id`. The dispatch syntax is a tiny DSL:

- `"district:13"`            → append `"13"` to `hard_filters.district_in`
- `"districts:1,2"`          → append both to `hard_filters.district_in`
- `"relax:price_huf_max"`    → clear `hard_filters.price_huf_max`
- `"relax:area_m2_min"`      → clear `hard_filters.area_m2_min`
- `"relax:rooms_min"`        → clear `hard_filters.rooms_min`
- `"relax:near"`             → clear `hard_filters.near`
- `"expand:location"`        → allow a user-approved farther location pool
- `"softpref:elevator"`      → append `"elevator"` to `soft_preferences`
- `"anchor:price_huf=85000000"` → set `anchors.price_huf` to 85_000_000
- `"city:<settlement_id>"`   → pin `hard_filters.city` / `city_id` to that settlement
- `"skip"`                   → returns the parsed query unchanged

`apply_clarification` returns a new frozen `ParsedQuery`; it does NOT introduce
a new scoring term. Acceptance 3 — `ScoreBreakdown` retains its ten v0 fields.

Each prefix has one handler in `_OPTION_HANDLERS`; `skip`, an id without a
right-hand side and an unknown prefix return the query unchanged. The question
side, `should_clarify`, lives in `clarification_policy`, which also re-exports
`apply_clarification`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from ingovo_search.recommender.clarification_schema import ClarificationAnswer
from ingovo_search.recommender.parser_schema import (
    SOFT_PREF_TAGS,
    LocationCandidate,
    LocationResolution,
    ParsedQuery,
    SoftPrefTag,
)
from ingovo_search.recommender.settlements import settlement_by_id
from ingovo_search.types.errors import ValidationError


def apply_clarification(parsed: ParsedQuery, answer: ClarificationAnswer) -> ParsedQuery:
    """Refine `parsed` based on the user's answer.

    Returns a new frozen `ParsedQuery`. Skips ("skip" option) and unparseable
    `option_id`s return the original query unchanged — the caller can still
    show the result list. Raises `ValidationError` only for malformed payloads
    (e.g. `softpref:not_a_valid_tag`).
    """
    if answer.option_id == "skip":
        return parsed

    prefix, _, rhs = answer.option_id.partition(":")
    if not rhs:
        return parsed

    handler = _OPTION_HANDLERS.get(prefix)
    if handler is not None:
        return handler(parsed, rhs)
    if prefix == "expand" and rhs == "location":
        return parsed.model_copy(update={"allow_distant_location_expansion": True})

    return parsed


def _apply_city(parsed: ParsedQuery, settlement_id: str) -> ParsedQuery:
    settlement = settlement_by_id(settlement_id)
    if settlement is None:
        return parsed
    candidate = LocationCandidate(
        settlement_id=settlement.id,
        name=settlement.name,
        county=settlement.county,
    )
    hard_filters = parsed.hard_filters.model_copy(update={"city": settlement.name, "city_id": settlement.id})
    resolution = LocationResolution(status="resolved", candidates=[candidate], matched_text=settlement.name)
    return parsed.model_copy(update={"hard_filters": hard_filters, "location_resolution": resolution})


def _apply_districts(parsed: ParsedQuery, districts: list[str]) -> ParsedQuery:
    cleaned = [d.strip() for d in districts if d.strip()]
    if not cleaned:
        return parsed
    existing = list(parsed.hard_filters.district_in)
    merged = existing + [d for d in cleaned if d not in existing]
    new_filters = parsed.hard_filters.model_copy(update={"district_in": merged})
    return parsed.model_copy(update={"hard_filters": new_filters})


def _apply_relax(parsed: ParsedQuery, field: str) -> ParsedQuery:
    _RELAXABLE: dict[str, str] = {
        "price_huf_max": "price_huf_max",
        "price_huf_min": "price_huf_min",
        "area_m2_min": "area_m2_min",
        "area_m2_max": "area_m2_max",
        "rooms_min": "rooms_min",
        "rooms": "rooms",
        "floor_min": "floor_min",
        "near": "near",
    }
    target = _RELAXABLE.get(field)
    if target is None:
        return parsed
    new_filters = parsed.hard_filters.model_copy(update={target: None})
    return parsed.model_copy(update={"hard_filters": new_filters})


_SOFTPREF_VOCAB: frozenset[str] = frozenset(SOFT_PREF_TAGS)


def _apply_softpref(parsed: ParsedQuery, tag: str) -> ParsedQuery:
    if tag not in _SOFTPREF_VOCAB:
        raise ValidationError(f"unknown soft-preference tag: {tag!r}")
    if tag in parsed.soft_preferences:
        return parsed
    existing: list[SoftPrefTag] = list(parsed.soft_preferences)
    existing.append(cast(SoftPrefTag, tag))
    return parsed.model_copy(update={"soft_preferences": existing})


def _apply_anchor(parsed: ParsedQuery, rhs: str) -> ParsedQuery:
    field, _, raw_value = rhs.partition("=")
    if not field or not raw_value:
        return parsed
    anchor_field = field.strip()
    try:
        if anchor_field == "price_huf":
            value: int | float | None = int(raw_value)
        elif anchor_field == "area_m2":
            value = float(raw_value)
        elif anchor_field == "rooms":
            value = int(raw_value)
        else:
            return parsed
    except ValueError as exc:
        raise ValidationError(f"invalid anchor value for {anchor_field!r}: {raw_value!r}") from exc
    new_anchors = parsed.anchors.model_copy(update={anchor_field: value})
    return parsed.model_copy(update={"anchors": new_anchors})


def _apply_district_list(parsed: ParsedQuery, rhs: str) -> ParsedQuery:
    """`district:13` and `districts:1,2` both append the comma-separated list."""
    return _apply_districts(parsed, rhs.split(","))


# One handler per DSL prefix; `expand:location` is the only prefix that also
# checks its right-hand side, so `apply_clarification` handles it inline.
_OPTION_HANDLERS: dict[str, Callable[[ParsedQuery, str], ParsedQuery]] = {
    "district": _apply_district_list,
    "districts": _apply_district_list,
    "city": _apply_city,
    "relax": _apply_relax,
    "softpref": _apply_softpref,
    "anchor": _apply_anchor,
}


__all__ = [
    "apply_clarification",
]
