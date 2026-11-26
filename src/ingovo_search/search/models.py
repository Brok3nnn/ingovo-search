"""The buyer search contract: request, response, envelope and the version boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

from ingovo_search.api.envelope import ErrorBody
from ingovo_search.lib.session import (
    RefinementOverrides,
    SessionSignal,
)
from ingovo_search.recommender.canonical_ranking import (
    CANONICAL_RANKING_VERSION,
)
from ingovo_search.recommender.clarification_policy import (
    Clarification,
    ClarificationAnswer,
)
from ingovo_search.recommender.match_percent import MATCH_PERCENT_VERSION
from ingovo_search.recommender.refinement_suggestions import RefinementSuggestion
from ingovo_search.recommender.rerank_prompt import RERANK_PROMPT_VERSION
from ingovo_search.recommender.score import ScoredCandidate
from ingovo_search.recommender.search_contract import (
    SEARCH_CONTRACT_VERSION,
)
from ingovo_search.search.constants import SCORE_VERSION, SEARCH_RESPONSE_CACHE_VERSION
from ingovo_search.types.errors import ParseError


def _empty_clarification_list() -> list[ClarificationAnswer]:
    return []


def _empty_signal_list() -> list[SessionSignal]:
    return []


def _empty_w_overrides() -> dict[str, float]:
    return {}


class SearchRequest(BaseModel):
    """Buyer-side search payload — see the design spec §14 step 6."""

    model_config = {"frozen": True}

    query: str | None = None
    include_narratives: bool = False
    # Output language for the AI narrative. The narrative is free prose, so the
    # web app's runtime translator cannot localise it — only the model can.
    # Unknown values fall back to Hungarian inside the narrator.
    narrative_locale: str = "hu"
    # Language the *query itself* is written in — the visitor's site language. It
    # is a separate field from `narrative_locale` because they answer different
    # questions (what did they type vs. what should they read), even though the web
    # app sends the same site locale for both. Empty means "same as the narrative
    # locale", which keeps every existing caller working unchanged.
    query_locale: str | None = None
    include_rerank: bool = True
    # Advanced/eval knob: how many top candidates go to the LLM reranker. None
    # uses the production default (`RERANK_CANDIDATE_COUNT`); the §17 sweep sets
    # it to compare top 30 / 20 / 15.
    rerank_candidate_count: int | None = None
    reset_session: bool = False
    prior_clarifications: list[ClarificationAnswer] = Field(default_factory=_empty_clarification_list)
    refinement_overrides: RefinementOverrides | None = None
    session_signals: list[SessionSignal] = Field(default_factory=_empty_signal_list)
    w_overrides: dict[str, float] = Field(default_factory=_empty_w_overrides)


def search_request_from_json(payload: str) -> SearchRequest:
    """The multipart routes' ``payload`` form field: a `SearchRequest` as a JSON string.

    A malformed payload is the reader's own bad input, a `ParseError` that counts
    the bad fields (in Hungarian, like every user-facing message: "Invalid search
    request: N invalid fields.").
    """
    try:
        return SearchRequest.model_validate_json(payload)
    except PydanticValidationError as exc:
        raise ParseError(f"Érvénytelen keresési kérés: {exc.error_count()} hibás mező.") from exc


def _empty_result_list() -> list[ScoredCandidate]:
    return []


def _empty_str_list() -> list[str]:
    return []


def _empty_suggestions() -> list[RefinementSuggestion]:
    return []


class SearchContextItem(BaseModel):
    """One compact, user-facing interpretation of the parsed search query."""

    model_config = {"frozen": True}

    key: str
    label: str
    relaxed: bool = False


def _empty_search_context() -> list[SearchContextItem]:
    return []


class PipelineVersions(BaseModel):
    """Version boundary for every signal that can change public ordering."""

    model_config = {"frozen": True}

    score: str = SCORE_VERSION
    search_contract: str = SEARCH_CONTRACT_VERSION
    match_percent: str = MATCH_PERCENT_VERSION
    canonical_ranking: str = CANONICAL_RANKING_VERSION
    rerank_prompt: str = RERANK_PROMPT_VERSION
    response_cache: str = SEARCH_RESPONSE_CACHE_VERSION


type ImageContextReason = Literal["applied", "not_a_property", "unsure", "unavailable"]


class ImageContextSummary(BaseModel):
    """What the photos attached to this search contributed, for the UI to show.

    `applied` is False whenever the pictures did not steer the ranking — a
    non-property photo, an unsure reading, or a vision call that failed or timed
    out. `reason` names which of those it was so the UI can say something
    truthful instead of silently ignoring an upload the user chose to make.
    """

    model_config = {"frozen": True}

    accepted_images: int = 0
    applied: bool = False
    reason: ImageContextReason = "applied"
    summary: str = ""
    chips: list[str] = Field(default_factory=_empty_str_list)


class SearchResponse(BaseModel):
    """Envelope `data` payload — at most 30 results, optional clarification.

    `relaxed` is True when the initial hard filter produced fewer than the
    first-page target and the pipeline fell back to M3 stepwise relaxation.
    `relaxation_path` carries the human-readable labels of the relaxations
    that were actually applied (in order). `refinement_suggestions` carries
    the M3 one-click filter tweaks the UI offers under the result list.
    """

    model_config = {"frozen": True}

    results: list[ScoredCandidate] = Field(default_factory=_empty_result_list)
    clarification: Clarification | None = None
    total_candidates: int = 0
    score_version: str = SCORE_VERSION
    pipeline_versions: PipelineVersions = Field(default_factory=PipelineVersions)
    relaxed: bool = False
    relaxation_path: list[str] = Field(default_factory=_empty_str_list)
    refinement_suggestions: list[RefinementSuggestion] = Field(default_factory=_empty_suggestions)
    search_context: list[SearchContextItem] = Field(default_factory=_empty_search_context)
    # None for a plain text search; set whenever the request carried photos.
    image_context: ImageContextSummary | None = None


class SearchMeta(BaseModel):
    """Search-specific envelope metadata consumed by latency benchmark tooling."""

    timestamp: str
    request_id: str
    timings_ms: dict[str, int]


class SearchEnvelope[T](BaseModel):
    """Search response envelope with timings, preserving the frontend envelope shape."""

    ok: bool
    data: T | None = None
    error: ErrorBody | None = None
    meta: SearchMeta


def ok_with_timings(
    data: SearchResponse,
    timings_ms: dict[str, int],
    *,
    request_id: str | None = None,
) -> SearchEnvelope[SearchResponse]:
    return SearchEnvelope(
        ok=True,
        data=data,
        error=None,
        meta=SearchMeta(
            timestamp=datetime.now(UTC).isoformat(),
            request_id=request_id or str(uuid4()),
            timings_ms=timings_ms,
        ),
    )
