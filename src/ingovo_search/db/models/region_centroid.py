"""RegionCentroid — learned geographic centroids for fuzzy region names.

Populated by the nightly recompute. Cold-start `llm_hint_districts` cached
here per fuzzy region until enough engagement signal accrues (~50 positives).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from geoalchemy2 import Geography
from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ingovo_search.db.models.base import Base


class RegionCentroid(Base):
    __tablename__ = "region_centroid"

    region_name: Mapped[str] = mapped_column(String, primary_key=True)
    centroid_geo: Mapped[Any | None] = mapped_column(Geography(geometry_type="POINT", srid=4326), nullable=True)
    spread_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_signals: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    llm_hint_districts: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
