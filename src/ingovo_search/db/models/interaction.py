"""Interaction — buyer signal captured with query context.

The **query** (not the user) is the pooling key for the recommender; the kNN
aggregation lives in `recommender/pooled_signal.py`.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ingovo_search.db.models.base import Base

if TYPE_CHECKING:
    from ingovo_search.db.models.property import Property


class Interaction(Base):
    __tablename__ = "interactions"
    __table_args__ = (
        Index("ix_interactions_property_id_created_at", "property_id", "created_at"),
        Index("ix_interactions_user_id_created_at", "user_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # user_id captured but ignored by pooled aggregation (used only for in-session reranking).
    user_id: Mapped[str | None] = mapped_column(String, nullable=True)
    property_id: Mapped[str] = mapped_column(String, ForeignKey("properties.id", ondelete="CASCADE"), nullable=False)

    # Query context — the pooling key
    query_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    query_parsed: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    query_embedding: Mapped[list[float] | None] = mapped_column(Vector(1024), nullable=True)

    # thumb_up | thumb_down | favorite | contact | view_long | view_short | hide | save
    # plus refinement_* signals for product analytics (do not enter pooled signal)
    signal_type: Mapped[str] = mapped_column(String, nullable=False)
    signal_weight: Mapped[float] = mapped_column(Float, default=1.0, server_default="1.0", nullable=False)

    property: Mapped[Property] = relationship(back_populates="interactions")
