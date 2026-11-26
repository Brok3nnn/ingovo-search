"""Poi — points of interest for proximity scoring (schools, transit, shops...).

Loaded from OSM Overpass extracts (`ingestion/poi_osm.py` + `scripts/load_pois.py`).
The §12 nightly job (`jobs/nightly_poi_distances.py`) computes per-listing
nearest-POI distances per category into `properties.features["poi_m"]`, which
the recommender reads for "iskola legyen a közelben"-style soft preferences.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from geoalchemy2 import Geography
from sqlalchemy import DateTime, Index, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from ingovo_search.db.models.base import Base


class Poi(Base):
    __tablename__ = "poi"
    __table_args__ = (
        # (source, source_id) keys re-loads idempotently — OSM elements keep
        # their "node/123" identity across extract runs.
        UniqueConstraint("source", "source_id", name="uq_poi_source_source_id"),
        Index("ix_poi_category", "category"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # One of ingestion.poi_osm.POI_CATEGORIES ("school", "metro", "bus", ...).
    category: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    geo: Mapped[Any] = mapped_column(Geography(geometry_type="POINT", srid=4326), nullable=False)

    source: Mapped[str] = mapped_column(String, default="osm", server_default="osm", nullable=False)
    # OSM element identity, e.g. "node/123456789" or "way/987654321".
    source_id: Mapped[str] = mapped_column(String, nullable=False)
