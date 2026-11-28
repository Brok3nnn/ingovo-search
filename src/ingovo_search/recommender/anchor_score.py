"""Anchor scoring — Gaussian distance from a "körül" / "kb." target.

`anchor_score` is the per-field Gaussian used to score listings against query
`anchors` (the soft "around X" bucket of the parser output). Field-specific
sigmas: price uses a 15% relative band (sigma scales with the anchor), area
and rooms use absolute bands.
"""

from __future__ import annotations

import math
from typing import Literal

AnchorField = Literal["price_huf", "area_m2", "rooms"]

# sigma per field. `price_huf` is multiplicative — `0.15 * anchor` — so a wider
# absolute tolerance applies to bigger prices. The others are absolute bands.
SIGMAS: dict[AnchorField, float] = {"price_huf": 0.15, "area_m2": 10.0, "rooms": 1.0}


def _sigma_for(field: AnchorField, anchor: float) -> float:
    return SIGMAS[field] * anchor if field == "price_huf" else SIGMAS[field]


def anchor_score(value: float, anchor: float, field: AnchorField) -> float:
    """Gaussian distance score: 1.0 at the anchor, decaying with `(value-anchor)/sigma`."""
    sigma = _sigma_for(field, anchor)
    if sigma <= 0:
        return 1.0 if value == anchor else 0.0
    return math.exp(-(((value - anchor) / sigma) ** 2))
