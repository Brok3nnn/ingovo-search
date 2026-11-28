"""Unit tests for hard-filter relaxation."""

from __future__ import annotations

from ingovo_search.recommender.parser_schema import HardFilters, NearFilter
from ingovo_search.recommender.relaxation import (
    RELAXATION_STRATEGY,
    iter_relaxations,
    stepwise_relax,
)


def _empty() -> HardFilters:
    return HardFilters()


def _typical() -> HardFilters:
    """A representative restrictive HardFilters mid-grade query would produce."""
    return HardFilters(
        city="Budapest",
        district_in=["5", "13"],
        property_type_in=["Lakás"],
        rooms=3,
        price_huf_max=80_000_000,
        area_m2_min=50.0,
    )


# --- strategy invariants -----------------------------------------------------


def test_strategy_never_relaxes_location() -> None:
    """Automatic fallback must preserve every explicit geographic constraint."""
    labels = [s.label for s in RELAXATION_STRATEGY]
    assert labels[0] == "max ár"
    assert {"város", "kerület", "közelség-sugár"}.isdisjoint(labels)


def test_strategy_does_not_touch_property_type_in() -> None:
    """`property_type_in` is the strictest cue — dropping it would return apples for an apartment search."""
    labels = [s.label for s in RELAXATION_STRATEGY]
    assert "típus" not in labels
    assert "property_type_in" not in labels


# --- stepwise_relax (one-shot maximum widening) ------------------------------


def test_stepwise_relax_on_empty_filters_is_noop() -> None:
    """Nothing to relax → empty applied list, identical filters back."""
    out = stepwise_relax(_empty())
    assert out.applied_labels == []
    assert out.hard_filters == _empty()


def test_stepwise_relax_typical_drops_every_relevant_filter() -> None:
    """`stepwise_relax` is maximum-widening — every relevant axis ends up cleared,
    while location and `property_type_in` remain strict."""
    out = stepwise_relax(_typical())
    assert out.hard_filters.price_huf_max is None
    assert out.hard_filters.district_in == ["5", "13"]
    assert out.hard_filters.rooms is None
    assert out.hard_filters.area_m2_min is None
    assert out.hard_filters.city == "Budapest"
    assert out.hard_filters.property_type_in == ["Lakás"]  # never relaxed
    assert "max ár" in out.applied_labels
    assert {"kerület", "város"}.isdisjoint(out.applied_labels)


def test_stepwise_relax_preserves_near_radius() -> None:
    hf = HardFilters(near=NearFilter(lat=47.5, lng=19.0, radius_km=30.0))
    out = stepwise_relax(hf)
    assert out.hard_filters.near is not None
    assert out.hard_filters.near.radius_km == 30.0
    assert "közelség-sugár" not in out.applied_labels


def test_stepwise_relax_near_radius_at_cap_is_skipped() -> None:
    """Radius already at the 50 km cap is not relevant — the step is a no-op."""
    hf = HardFilters(near=NearFilter(lat=47.5, lng=19.0, radius_km=50.0))
    out = stepwise_relax(hf)
    assert "közelség-sugár" not in out.applied_labels


# --- iter_relaxations (step-at-a-time orchestration) -------------------------


def test_iter_relaxations_yields_one_change_at_a_time() -> None:
    steps = list(iter_relaxations(_typical()))
    # The labels yielded should match the strategy order, skipping irrelevant ones.
    labels = [label for label, _ in steps]
    assert labels[0] == "max ár"
    assert labels[1] == "pontos szobaszám"
    assert {"város", "kerület"}.isdisjoint(labels)

    # Each step is monotonically more relaxed than the previous.
    _, first_hf = steps[0]
    assert first_hf.price_huf_max is None
    assert first_hf.district_in == ["5", "13"]
    _, second_hf = steps[1]
    assert second_hf.district_in == ["5", "13"]
    assert second_hf.city == "Budapest"


def test_iter_relaxations_empty_filters_yields_nothing() -> None:
    assert list(iter_relaxations(_empty())) == []


def test_iter_relaxations_skips_inapplicable_steps() -> None:
    """A filter without a price_huf_max never sees the 'max ár' label."""
    hf = HardFilters(city="Budapest", rooms=2)
    labels = [label for label, _ in iter_relaxations(hf)]
    assert "max ár" not in labels
    assert "pontos szobaszám" in labels
    assert "város" not in labels
