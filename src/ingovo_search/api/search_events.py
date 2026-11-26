"""Public, user-safe progress events for the streamed buyer search.

These events are an execution trace, not model chain-of-thought. Every payload
is derived from an observable pipeline boundary: parsed public constraints,
candidate counts, applied scope changes, rerank decisions, image interpretation
and the final canonical response.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel, Field

from ingovo_search.search.models import SearchContextItem, SearchResponse


def _empty_constraints() -> list[SearchContextItem]:
    return []


class SearchStartedEvent(BaseModel):
    model_config = {"frozen": True}

    type: Literal["search.started"] = "search.started"
    request_id: str
    query: str


class IntentParsedEvent(BaseModel):
    model_config = {"frozen": True}

    type: Literal["intent.parsed"] = "intent.parsed"
    request_id: str
    summary: str
    confidence: float = Field(ge=0, le=1)
    constraints: list[SearchContextItem] = Field(default_factory=_empty_constraints)
    needs_clarification: bool = False


class CandidatesScoredEvent(BaseModel):
    model_config = {"frozen": True}

    type: Literal["candidates.scored"] = "candidates.scored"
    request_id: str
    candidate_count: int = Field(ge=0)
    exact_count: int = Field(ge=0)


class ScopeExpandedEvent(BaseModel):
    model_config = {"frozen": True}

    type: Literal["scope.expanded"] = "scope.expanded"
    request_id: str
    reason: str
    before_count: int = Field(ge=0)
    after_count: int = Field(ge=0)
    scope: Literal["relaxed", "nearby", "distant"]


class RerankStartedEvent(BaseModel):
    model_config = {"frozen": True}

    type: Literal["rerank.started"] = "rerank.started"
    request_id: str
    candidate_count: int = Field(ge=0)


class RerankCompletedEvent(BaseModel):
    model_config = {"frozen": True}

    type: Literal["rerank.completed"] = "rerank.completed"
    request_id: str
    candidate_count: int = Field(ge=0)


class RerankSkippedEvent(BaseModel):
    model_config = {"frozen": True}

    type: Literal["rerank.skipped"] = "rerank.skipped"
    request_id: str
    candidate_count: int = Field(ge=0)
    reason: Literal["deterministic", "disabled", "no_semantic_query", "timeout", "budget"]


class ImageInterpretedEvent(BaseModel):
    model_config = {"frozen": True}

    type: Literal["image.interpreted"] = "image.interpreted"
    request_id: str
    applied: bool
    reason: Literal["applied", "not_a_property", "unsure", "unavailable"]
    labels: list[str] = Field(default_factory=list)
    summary: str = ""


class SearchResultEvent(BaseModel):
    model_config = {"frozen": True}

    type: Literal["result"] = "result"
    request_id: str
    data: SearchResponse
    timings_ms: dict[str, int] = Field(default_factory=dict)


class SearchErrorEvent(BaseModel):
    model_config = {"frozen": True}

    type: Literal["error"] = "error"
    request_id: str
    code: str
    message: str
    retryable: bool = True


type SearchProgressEvent = (
    SearchStartedEvent
    | IntentParsedEvent
    | CandidatesScoredEvent
    | ScopeExpandedEvent
    | RerankStartedEvent
    | RerankCompletedEvent
    | RerankSkippedEvent
    | ImageInterpretedEvent
    | SearchResultEvent
    | SearchErrorEvent
)

type SearchProgressEmitter = Callable[[SearchProgressEvent], Awaitable[None]]


__all__ = [
    "CandidatesScoredEvent",
    "ImageInterpretedEvent",
    "IntentParsedEvent",
    "RerankCompletedEvent",
    "RerankSkippedEvent",
    "RerankStartedEvent",
    "ScopeExpandedEvent",
    "SearchErrorEvent",
    "SearchProgressEmitter",
    "SearchProgressEvent",
    "SearchResultEvent",
    "SearchStartedEvent",
]
