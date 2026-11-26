"""What one buyer search carries between its stages.

``run_search_pipeline`` builds a :class:`SearchRun` once and hands it to every stage
(response cache, embeddings, candidates, rerank, envelope). The stages read the shared
collaborators and options from it and write what they produced back into it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncSession

from ingovo_search.api.search_events import (
    ImageInterpretedEvent,
    IntentParsedEvent,
    SearchProgressEmitter,
    SearchProgressEvent,
)
from ingovo_search.lib.cache import Cache
from ingovo_search.lib.embeddings import EmbeddingClient
from ingovo_search.lib.jev import JevClient
from ingovo_search.lib.llm import LlmClient
from ingovo_search.lib.session import SessionState
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.rerank import RerankResult
from ingovo_search.search.constants import RERANK_CANDIDATE_COUNT
from ingovo_search.search.embedding_cache import SpeculativeCachedEmbedding


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


type RerankBackend = Literal["gemini", "jev", "off"]

# A ``(property row, ScoreBreakdown)`` pair, as ``recommend`` returns them.
type ScoredRow = tuple[Any, Any]

# The Jev client enforces its own timeout; the pipeline ceiling sits a little
# above it so the client's JevError (and its EnrichmentLog row) wins the race
# instead of a bare cancellation.
JEV_TIMEOUT_GRACE_MS = 300


def served_outcome(result: RerankResult | None) -> str:
    """How the rerank ended, for the search log and the shadow row."""
    if result is None:
        return "timeout"
    if result.fast_path:
        return "fast_path"
    if result.fallback:
        return "fallback"
    return "success"


@dataclass(frozen=True, slots=True)
class SearchInput:
    """The session and the parsed query a search starts from."""

    state: SessionState
    raw_parsed: ParsedQuery


@dataclass(frozen=True, slots=True)
class SearchDependencies:
    """The per-request collaborators every stage shares."""

    db_session: AsyncSession
    llm_client: LlmClient
    embedding_client: EmbeddingClient
    cache: Cache


@dataclass(frozen=True, slots=True)
class SearchOptions:
    """How this search runs: what the request asked for and what the settings allow."""

    user_signed_in: bool
    include_narratives: bool = False
    narrative_locale: str = "hu"
    include_rerank: bool = True
    rerank_candidate_count: int = RERANK_CANDIDATE_COUNT
    rerank_rank_only: bool = False
    rerank_timeout_ms: int = 1800
    rerank_backend: RerankBackend = "gemini"
    # `AI_SEARCH_CONDITIONAL_RERANK`: a clear-cut deterministic order skips the LLM rerank.
    conditional_rerank: bool = True
    jev_client: JevClient | None = None
    jev_timeout_ms: int = 1200
    jev_cooldown_s: float = 60.0
    shadow_jev_client: JevClient | None = None
    speculative_query_embedding: SpeculativeCachedEmbedding | None = None


@dataclass(frozen=True, slots=True)
class SearchProgress:
    """Where progress events go (the streaming route) and the two events parsed before the pipeline."""

    emitter: SearchProgressEmitter | None = None
    request_id: str = ""
    intent_event: IntentParsedEvent | None = None
    image_event: ImageInterpretedEvent | None = None


@dataclass(slots=True)
class SearchTimer:
    """The per-step timings of one request, and when the request started."""

    timings: dict[str, int]
    started: float

    @classmethod
    def start(cls, timings_ms: dict[str, int] | None = None, request_started: float | None = None) -> SearchTimer:
        """Continue the request's own timings when it has them; otherwise start fresh here."""
        timings = timings_ms if timings_ms is not None else {}
        started = request_started if request_started is not None else time.perf_counter()
        return cls(timings, started)

    @staticmethod
    def now() -> float:
        return time.perf_counter()

    def record(self, key: str, step_started: float) -> int:
        self.timings[key] = elapsed_ms(step_started)
        return self.timings[key]


def _no_latency() -> int:
    return 0


@dataclass(slots=True)
class SearchRun:
    """One search in flight: its inputs, collaborators, and what the stages have produced so far."""

    state: SessionState
    parsed: ParsedQuery
    original_parsed: ParsedQuery
    parsed_hash: str
    cache_key: str
    deps: SearchDependencies
    options: SearchOptions
    progress: SearchProgress
    timer: SearchTimer
    # Milliseconds since the latency clock started (the response-cache read); set by the pipeline.
    elapsed: Callable[[], int] = _no_latency
    query_emb: list[float] = field(default_factory=list[float])
    freetext_emb: list[float] | None = None
    administrative_city_scope: tuple[str, ...] | None = None
    initial_candidate_ids: list[str] | None = None
    scored: list[ScoredRow] = field(default_factory=list[ScoredRow])
    relaxation_path: list[str] = field(default_factory=list[str])
    relaxation_ms: int = 0
    automatic_distant_expansion: bool = False

    @property
    def timings(self) -> dict[str, int]:
        return self.timer.timings

    async def emit(self, event: SearchProgressEvent) -> None:
        """Best-effort progress delivery must never change search correctness."""
        if self.progress.emitter is None:
            return
        with suppress(Exception):
            await self.progress.emitter(event)
