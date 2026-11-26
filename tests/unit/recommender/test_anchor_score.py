"""Unit tests for §8 `anchor_score`.

Implements the spec formula `exp(-((value-anchor)/sigma)^2)` exactly, with
field-specific sigmas (the design spec §8 step 4): 15% of the anchor for
`price_huf`, absolute 10 m² for `area_m2`, absolute 1 for `rooms`.
"""

from __future__ import annotations

import math

from ingovo_search.recommender.anchor_score import SIGMAS, anchor_score


def test_sigmas_match_spec() -> None:
    assert SIGMAS == {"price_huf": 0.15, "area_m2": 10.0, "rooms": 1.0}


def test_anchor_score_at_anchor_is_one() -> None:
    """§8 acceptance 4 part 1: value == anchor → 1.0."""
    assert anchor_score(80_000_000, 80_000_000, "price_huf") == 1.0
    assert anchor_score(60.0, 60.0, "area_m2") == 1.0
    assert anchor_score(2.0, 2.0, "rooms") == 1.0


def test_anchor_score_price_15_percent_above_anchor() -> None:
    """§8 acceptance 4 part 2 — honest reading.

    Spec annotates `anchor_score(95M, 80M, "price_huf")` as "≈ 0.32", but the
    spec's own formula `exp(-((v-a)/(0.15*a))^2)` yields `exp(-(15/12)^2) =
    exp(-1.5625) ≈ 0.21`. Reaching 0.32 would need sigma ≈ 0.175·anchor or
    sigma relative to value, not anchor. This asserts what the formula
    actually produces; the "≈0.32" annotation is inconsistent with §8's own
    `SIGMAS` + formula.
    """
    assert math.isclose(anchor_score(95_000_000, 80_000_000, "price_huf"), math.exp(-((15 / 12) ** 2)), abs_tol=1e-6)
    assert math.isclose(anchor_score(95_000_000, 80_000_000, "price_huf"), 0.2096, abs_tol=0.001)


def test_anchor_score_one_sigma_off() -> None:
    # area is absolute (sigma = 10 m²): value=70, anchor=60 → exp(-1) ≈ 0.368.
    assert math.isclose(anchor_score(70.0, 60.0, "area_m2"), math.exp(-1.0))
    # rooms is absolute (sigma = 1): value=3, anchor=2 → exp(-1).
    assert math.isclose(anchor_score(3.0, 2.0, "rooms"), math.exp(-1.0))


def test_anchor_score_symmetric_around_anchor() -> None:
    assert math.isclose(anchor_score(70.0, 60.0, "area_m2"), anchor_score(50.0, 60.0, "area_m2"))


def test_anchor_score_handles_zero_anchor_for_price() -> None:
    """`sigma = 0.15 * 0 = 0` — guard avoids division by zero."""
    assert anchor_score(0.0, 0.0, "price_huf") == 1.0
    assert anchor_score(1.0, 0.0, "price_huf") == 0.0
