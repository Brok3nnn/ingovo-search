"""The clarification schema.

`Clarification` is what `should_clarify` asks — the trigger's `ClarificationReason`,
the HU question and its option chips (`ClarificationOption`, whose `id` follows the
option-id DSL in `clarification_apply`) — and `ClarificationAnswer` is what the
frontend posts back. All four are frozen; `clarification_policy` builds the questions
and re-exports these types, `clarification_apply` reads the answer.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ClarificationReason(StrEnum):
    """Why we're asking — drives `apply_clarification` dispatch."""

    CONFLICTING_CONSTRAINTS = "conflicting_constraints"
    AMBIGUOUS_SETTLEMENT = "ambiguous_settlement"
    LOW_PARSER_CONFIDENCE = "low_parser_confidence"
    AMBIGUOUS_GEOGRAPHY = "ambiguous_geography"
    TOO_FEW_RESULTS = "too_few_results"
    TOO_MANY_WEAK_RESULTS = "too_many_weak_results"
    SPLIT_TOP_INTERPRETATIONS = "split_top_interpretations"


class ClarificationOption(BaseModel):
    """One actionable choice the frontend renders as a chip / button.

    `id` follows the DSL described in the module docstring; the frontend
    posts it back verbatim in `ClarificationAnswer.option_id`.
    """

    model_config = {"frozen": True}

    id: str
    label: str


class Clarification(BaseModel):
    """One skippable clarification, returned by `should_clarify`.

    `options` lists actionable choices; "skip" is implicit and rendered by
    the frontend regardless. `reason` lets the frontend choose layout and
    the backend route `apply_clarification`.
    """

    model_config = {"frozen": True}

    reason: ClarificationReason
    question: str
    options: list[ClarificationOption]


class ClarificationAnswer(BaseModel):
    """The user's response to a `Clarification`."""

    model_config = {"frozen": True}

    reason: ClarificationReason
    option_id: str
    free_text: str | None = None


__all__ = [
    "Clarification",
    "ClarificationAnswer",
    "ClarificationOption",
    "ClarificationReason",
]
