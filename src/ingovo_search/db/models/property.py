"""Property — the listing row the search filters, scores and returns.

The structured listing fields, the enrichment outputs (all nullable, so the same
table works before and after enrichment), the two Cohere embeddings and the
PostGIS point. `ingovo_search/db/schema.sql` is the DDL for this table; it also
adds the generated `rooms_count` column, which is deliberately not mapped here
(see the comment on `rooms`).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from geoalchemy2 import Geography
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ingovo_search.db.models.base import Base

if TYPE_CHECKING:
    from ingovo_search.db.models.interaction import Interaction


class Property(Base):
    __tablename__ = "properties"
    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uq_properties_source_source_id"),
        Index("ix_properties_city", "city"),
        Index("ix_properties_district", "district"),
        Index("ix_properties_property_type", "property_type"),
        Index("ix_properties_price_huf", "price_huf"),
        CheckConstraint("price_huf IS NULL OR price_huf > 0", name="price_huf_positive"),
        CheckConstraint("currency = 'HUF'", name="currency_huf"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Core listing fields
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Machine translations of the two free-text fields, keyed by site locale. A
    # missing key means the reader falls back to the Hungarian source text.
    title_translations: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    description_translations: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    # The only price column, in whole forints.
    price_huf: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    currency: Mapped[str] = mapped_column(
        CHAR(3),
        default="HUF",
        server_default="HUF",
        nullable=False,
    )
    transaction_type: Mapped[str] = mapped_column(
        String, default="sale", server_default="sale", nullable=False
    )  # sale | rent

    city: Mapped[str] = mapped_column(String, nullable=False)
    district: Mapped[str | None] = mapped_column(String, nullable=True)
    address: Mapped[str | None] = mapped_column(String, nullable=True)

    property_type: Mapped[str | None] = mapped_column(String, nullable=True)  # panel | tégla | családi ház | …
    # ``rooms`` is the source text ("2,5", "2+1"). The numeric ``rooms_count``
    # (GENERATED ALWAYS AS ingovo_room_count(rooms), see schema.sql) is
    # deliberately NOT mapped: ``Base.metadata.create_all`` would create it as a
    # plain column. The hard filter reads it through raw SQL only
    # (``recommender/sql_filter.py``).
    rooms: Mapped[str | None] = mapped_column(String, nullable=True)
    area_m2: Mapped[float | None] = mapped_column(Float, nullable=True)
    floor: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_floors: Mapped[int | None] = mapped_column(Integer, nullable=True)
    year_built: Mapped[int | None] = mapped_column(Integer, nullable=True)
    heating: Mapped[str | None] = mapped_column(String, nullable=True)

    # Visibility: only `active`, unexpired rows are searchable (lib/property_status.py).
    status: Mapped[str] = mapped_column(String, default="active", server_default="active", nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    images: Mapped[list[str]] = mapped_column(ARRAY(String), default=list, server_default="{}", nullable=False)
    # Free-form listing attributes; `features["poi_m"]` holds the metres to the
    # nearest point of interest per category (ingestion/poi_distances.py).
    features: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Amenity booleans (supplied directly or filled in by enrichment)
    has_balcony: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    has_terrace: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    has_parking: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    has_garage: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    has_garden: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    has_basement: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    has_pool: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    elevator: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    air_conditioning: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # Enrichment outputs (an LLM pass over the description; not part of this repository)
    renovation_state: Mapped[str | None] = mapped_column(String, nullable=True)
    orientation: Mapped[str | None] = mapped_column(String, nullable=True)
    noise_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    light_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    red_flags: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    description_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    templated_phrase_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    cleaned_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    one_paragraph_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    short_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Engagement aggregates
    view_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    save_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    contact_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    photo_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)

    # Provenance
    source: Mapped[str | None] = mapped_column(String, nullable=True)  # e.g. "synthetic"
    source_id: Mapped[str | None] = mapped_column(String, nullable=True)

    # Recommender inputs
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_enriched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    factual_embedding: Mapped[list[float] | None] = mapped_column(Vector(1024), nullable=True)
    descriptive_embedding: Mapped[list[float] | None] = mapped_column(Vector(1024), nullable=True)
    geo: Mapped[Any | None] = mapped_column(Geography(geometry_type="POINT", srid=4326), nullable=True)

    interactions: Mapped[list[Interaction]] = relationship(back_populates="property", cascade="all, delete-orphan")
