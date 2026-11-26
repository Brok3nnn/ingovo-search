"""M3 hard-filter relaxation — recover from zero-result searches.

When `recommend()` returns an empty candidate list, the §14 search pipeline
walks `RELAXATION_STRATEGY` step-by-step, dropping or widening one filter
per iteration until either:

  - a non-empty candidate set is found (with `relaxed=True` flagged so the
    frontend can render a "lazítottam X-en, ennyi lett" hint), or
  - every relaxable filter is exhausted (still empty → return empty).

The strategy is **deterministic and ordered by user-impact**: cheaper to
drop a price ceiling than a district list, cheaper to drop a district than
the property type, etc. The order mirrors what a human agent would relax
first when asked "I can't find anything that matches exactly":

  1. `price_huf_max` — most common over-restrictor; widen first.
  2. `rooms`        — exact room count drops to "anything close".
  3. `rooms_min`    — minimum room count loosens.
  4. `area_m2_min` / `area_m2_max` — size envelope widens.
  5. `floor_min`    — floor minimum loosens.
  6. `property_type_exclude` — accept previously-excluded types.

City, Budapest district and explicit proximity constraints are never relaxed
here. Nearby geography is built separately from verified settlement distances;
farther expansion requires an explicit clarification answer.

Soft cues (`GeoHints.county`, `GeoHints.balaton_shore`, anchors, soft prefs)
are NEVER relaxed because they don't filter — they only score.

Pure-function module: every transform returns a NEW `HardFilters` (Pydantic
frozen models). The orchestrator that calls `recommend()` lives in
`api/search.py`; this module just provides the steps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ingovo_search.recommender.parser_schema import HardFilters


@dataclass(frozen=True, slots=True)
class RelaxationStep:
    """One pre-defined relaxation move with a stable human-readable label."""

    label: str  # e.g. "max ár"; surfaced to the UI in `relaxation_path`
    apply: Callable[[HardFilters], HardFilters]
    relevant: Callable[[HardFilters], bool]  # True iff the step would change `hf`


def _drop_price_max(hf: HardFilters) -> HardFilters:
    return hf.model_copy(update={"price_huf_max": None})


def _drop_rooms_exact(hf: HardFilters) -> HardFilters:
    return hf.model_copy(update={"rooms": None})


def _drop_rooms_min(hf: HardFilters) -> HardFilters:
    return hf.model_copy(update={"rooms_min": None})


def _drop_area_envelope(hf: HardFilters) -> HardFilters:
    return hf.model_copy(update={"area_m2_min": None, "area_m2_max": None})


def _drop_floor_min(hf: HardFilters) -> HardFilters:
    return hf.model_copy(update={"floor_min": None})


def _drop_property_type_exclude(hf: HardFilters) -> HardFilters:
    return hf.model_copy(update={"property_type_exclude": []})


RELAXATION_STRATEGY: tuple[RelaxationStep, ...] = (
    RelaxationStep("max ár", _drop_price_max, lambda hf: hf.price_huf_max is not None),
    RelaxationStep("pontos szobaszám", _drop_rooms_exact, lambda hf: hf.rooms is not None),
    RelaxationStep("min. szobaszám", _drop_rooms_min, lambda hf: hf.rooms_min is not None),
    RelaxationStep("méret", _drop_area_envelope, lambda hf: hf.area_m2_min is not None or hf.area_m2_max is not None),
    RelaxationStep("min. emelet", _drop_floor_min, lambda hf: hf.floor_min is not None),
    RelaxationStep("kizárt típusok", _drop_property_type_exclude, lambda hf: bool(hf.property_type_exclude)),
)


@dataclass(frozen=True, slots=True)
class RelaxationOutcome:
    """Result of a relaxation walk — the final `HardFilters` plus the labels applied."""

    hard_filters: HardFilters
    applied_labels: list[str]


def stepwise_relax(hard_filters: HardFilters) -> RelaxationOutcome:
    """Apply every relevant relaxation step **at once** (one-shot maximum widening).

    Use this when the caller wants a single fallback try — every relaxable
    filter is dropped/widened in priority order, and the result is returned
    together with the labels of the steps that actually changed something.

    For interactive "try-one-step-at-a-time" search orchestration use the
    `iter_relaxations` generator below instead.
    """
    current = hard_filters
    applied: list[str] = []
    for step in RELAXATION_STRATEGY:
        if step.relevant(current):
            current = step.apply(current)
            applied.append(step.label)
    return RelaxationOutcome(hard_filters=current, applied_labels=applied)


def iter_relaxations(hard_filters: HardFilters):
    """Yield (label, relaxed_hard_filters) one step at a time.

    The orchestrator (e.g. `api/search.py`) re-runs `recommend()` against the
    yielded `HardFilters` until it gets a non-empty result. Skips steps that
    are no-ops for the current filter shape (no `price_huf_max` set → skip).
    """
    current = hard_filters
    for step in RELAXATION_STRATEGY:
        if not step.relevant(current):
            continue
        current = step.apply(current)
        yield step.label, current
