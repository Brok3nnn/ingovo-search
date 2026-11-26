"""SearchLog — every retrieval / search event for ranking quality and latency monitoring.

Separate from `EnrichmentLog` because it is not an LLM-cost record.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ingovo_search.db.models.base import Base


class SearchLog(Base):
    __tablename__ = "search_log"
    __table_args__ = (
        Index("ix_search_log_created_at", "created_at"),
        Index("ix_search_log_query_hash", "query_hash"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user_id: Mapped[str | None] = mapped_column(String, nullable=True)
    query_hash: Mapped[str] = mapped_column(String, nullable=False)
    parsed_query: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    hard_filter_count: Mapped[int] = mapped_column(Integer, nullable=False)
    sql_candidate_count: Mapped[int] = mapped_column(Integer, nullable=False)
    vector_candidate_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    returned_count: Mapped[int] = mapped_column(Integer, nullable=False)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    score_version: Mapped[str] = mapped_column(String, nullable=False)
    top_result_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    # Which reranker served this search (gemini | jev | off), so canary arms can be compared.
    rerank_backend: Mapped[str | None] = mapped_column(String, nullable=True)
    # How the rerank ended: success | fallback | timeout | fast_path | skipped:<reason>.
    rerank_outcome: Mapped[str | None] = mapped_column(String, nullable=True)
