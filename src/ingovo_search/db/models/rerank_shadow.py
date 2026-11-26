"""RerankShadowComparison — the Gemini and Jev orders of one live search, side by side.

Written by the search pipeline's shadow task (`AI_SEARCH_RERANK_BACKEND=shadow`).
`semantic_query` is user text: purge old rows (the original deployment kept 30 days).
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ingovo_search.db.models.base import Base


class RerankShadowComparison(Base):
    __tablename__ = "rerank_shadow_comparison"
    __table_args__ = (Index("ix_rerank_shadow_comparison_created_at", "created_at"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    query_hash: Mapped[str] = mapped_column(String, nullable=False)
    semantic_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The rerank head in deterministic order; both orders below are permutations of it.
    candidate_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    # gemini | jev — which order the user saw.
    served_backend: Mapped[str] = mapped_column(String, nullable=False)

    gemini_order: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    jev_order: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    # listing id -> Jev fit on 0..1.
    jev_scores: Mapped[dict[str, float] | None] = mapped_column(JSONB, nullable=True)

    gemini_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    jev_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    jev_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # success | fallback | timeout | transport_error | parse_error | fast_path
    gemini_outcome: Mapped[str | None] = mapped_column(String, nullable=True)
    jev_outcome: Mapped[str | None] = mapped_column(String, nullable=True)
    jev_model: Mapped[str | None] = mapped_column(String, nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String, nullable=True)
