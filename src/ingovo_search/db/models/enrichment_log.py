"""EnrichmentLog — every LLM call logged for cost and quality monitoring."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Float, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ingovo_search.db.models.base import Base


class EnrichmentLog(Base):
    __tablename__ = "enrichment_log"
    __table_args__ = (
        Index("ix_enrichment_log_created_at", "created_at"),
        Index("ix_enrichment_log_property_id", "property_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    property_id: Mapped[str | None] = mapped_column(String, nullable=True)
    model_used: Mapped[str] = mapped_column(String, nullable=False)  # e.g. "gemini-2.5-flash-lite"
    prompt_version: Mapped[str] = mapped_column(String, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    outcome: Mapped[str] = mapped_column(String, nullable=False)  # success | parse_error | timeout | content_violation
    # Part of `input_tokens` served from a Gemini cache (implicit or explicit), as the
    # provider reported it; NULL when the row carries an estimate.
    cached_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
