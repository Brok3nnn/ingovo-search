"""SyntheticEvalRating — relevance ratings of (query, listing) pairs, human or LLM judge.

One row per (query, property, rater) triple. Graded 1-5 per §17.2.5 rating UI.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ingovo_search.db.models.base import Base


class SyntheticEvalRating(Base):
    __tablename__ = "synthetic_eval_rating"
    __table_args__ = (
        UniqueConstraint(
            "query_id",
            "property_id",
            "rater_id",
            name="uq_synthetic_eval_rating_query_property_rater",
        ),
        Index("ix_synthetic_eval_rating_query_id", "query_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    query_id: Mapped[str] = mapped_column(String, nullable=False)
    property_id: Mapped[str] = mapped_column(String, ForeignKey("properties.id", ondelete="CASCADE"), nullable=False)
    rater_id: Mapped[str] = mapped_column(String, nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)  # 0-5
    rating_source: Mapped[str] = mapped_column(String, default="synthetic", server_default="synthetic", nullable=False)
    inventory_source: Mapped[str] = mapped_column(
        String, default="synthetic", server_default="synthetic", nullable=False
    )
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    location_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    price_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    size_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    preference_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    location_match: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    human_match_percent: Mapped[int | None] = mapped_column(Integer, nullable=True)
    predicted_match_percent: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score_version: Mapped[str | None] = mapped_column(String, nullable=True)
    pipeline_versions: Mapped[dict[str, str] | None] = mapped_column(JSONB, nullable=True)
