"""Homepage index probe — the payload models.

`SearchProbe` is the `/api/search/probe` response body: the strict-filter count, the
market price band with the budget's position inside it (`PriceStats`), a few real
preview listings with their structured match % (`ProbePreviewListing`) and the grounded
widening moves with their real `COUNT` deltas (`ProbeDirection`). All four are frozen;
`ingovo_search.recommender.probe` computes them and re-exports them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_serializer

from ingovo_search.lib.property_price import json_safe_huf
from ingovo_search.lib.session import RefinementOverrides
from ingovo_search.recommender.parser_schema import ParsedQuery, public_parsed_query_json


class PriceStats(BaseModel):
    """Market price band for the segment, with the budget's position inside it."""

    model_config = {"frozen": True}

    # Number of priced listings in the price-*unconstrained* segment.
    segment_count: int
    currency: str = "HUF"
    min: int | None = None
    p25: int | None = None
    median: int | None = None
    p75: int | None = None
    max: int | None = None
    # The buyer's stated cap (`price_huf_max`) or `~around` anchor, if any.
    budget: int | None = None
    # 0..100: share of the segment priced at or below `budget`. `< 30` reads as a
    # tight budget for this segment, `> 70` as generous. `None` when no budget.
    budget_percentile: int | None = None

    @field_serializer("min", "p25", "median", "p75", "max", "budget")
    def serialize_huf(self, value: int | None) -> int | str | None:
        return json_safe_huf(value)


class ProbePreviewListing(BaseModel):
    """One real listing surfaced in the live preview, with its structured match %."""

    model_config = {"frozen": True}

    listing_id: str
    match_percent: int = Field(ge=1, le=100)
    title: str | None = None
    price: int | None = None
    price_huf: str | None = None
    area_m2: float | None = None
    rooms: str | None = None
    city: str | None = None
    district: str | None = None
    property_type: str | None = None
    transaction_type: str | None = None
    image: str | None = None


class ProbeDirection(BaseModel):
    """One grounded search direction with its real result count and delta.

    ``overrides`` is the exact :class:`RefinementOverrides` the frontend replays
    into ``/api/search`` to *take* this direction; ``None`` on the exact row
    (the query as stated). ``delta_count`` is ``result_count - exact_count`` —
    positive for a widening move.
    """

    model_config = {"frozen": True}

    id: str
    kind: str
    title: str
    result_count: int
    delta_count: int
    overrides: RefinementOverrides | None = None


def _empty_previews() -> list[ProbePreviewListing]:
    return []


def _empty_directions() -> list[ProbeDirection]:
    return []


class SearchProbe(BaseModel):
    """Grounded homepage probe payload — counts, market band, real previews."""

    model_config = {"frozen": True}

    request_id: str
    intent_hash: str
    parsed_query: ParsedQuery
    understood_summary: str
    exact_count: int
    nearby_count: int | None = None
    price_stats: PriceStats
    previews: list[ProbePreviewListing] = Field(default_factory=_empty_previews)
    directions: list[ProbeDirection] = Field(default_factory=_empty_directions)
    insight: str | None = None
    # How many listings were actually scored to pick the previews.
    sample_size: int = 0

    @field_serializer("parsed_query", when_used="json")
    def serialize_parsed_query(self, value: ParsedQuery) -> dict[str, object]:
        """Keep unsafe HUF intent values exact at the JSON/browser boundary."""

        return public_parsed_query_json(value)


__all__ = [
    "PriceStats",
    "ProbeDirection",
    "ProbePreviewListing",
    "SearchProbe",
]
