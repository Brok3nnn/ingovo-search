"""`NormalizedListing` — the canonical listing shape every data source is mapped to.

In the original product, per-agency feed adapters produced it; here the seed command
validates the synthetic listings against it before they are written to `properties`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ingovo_search.lib.property_price import MAX_POSTGRES_BIGINT


class NormalizedListing(BaseModel):
    """Canonical listing shape.

    Validation rules are minimal: a few non-empty strings, a non-negative price,
    a closed `transaction_type` enum. Per-source quirks are absorbed via
    `features` (a JSON dict) and `raw_payload` (anything, preserved).
    """

    source_id: str = Field(min_length=1)  # unique within the source
    source_fingerprint: str = Field(min_length=1)  # content hash for change detection

    title: str = Field(min_length=1)
    description: str = ""

    price: int = Field(ge=0, le=MAX_POSTGRES_BIGINT)  # HUF, integer
    transaction_type: Literal["sale", "rent"]

    city: str = Field(min_length=1)
    district: str | None = None
    address: str | None = None

    rooms: int | None = None
    area: float | None = None  # m²
    floor: int | None = None
    total_floors: int | None = None

    property_type: str | None = None  # panel | tégla | családi ház | sorház | …
    year_built: int | None = None
    heating: str | None = None

    features: dict[str, Any] = Field(default_factory=dict)
    images: list[str] = Field(default_factory=list)

    raw_payload: Any = None  # preserved for debugging and reprocessing
