"""The §17.3 narrative contract.

`AINarrative` is the JSON object the narrator model must return and the search
envelope carries per result: a headline, a body, an honest compromise, the four
`RationaleChip`s in fixed order and the backend-owned `VsNext`. The LLM path in
`narrator.py` and the heuristic in `narrator_fallback.py` both build it, so the
contract lives apart from either — the same way `parser_schema.py` serves the parser.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

type ChipStatus = Literal["match", "warning", "miss"]
type ChipGroup = Literal["location", "size", "price", "renovation"]


class RationaleChip(BaseModel):
    """One of the four `rationale_chips` the UI renders next to each card."""

    model_config = {"frozen": True}

    group: ChipGroup
    status: ChipStatus
    label: str  # Hungarian, capitalised, e.g. "Lokáció", "Ár"


class VsNext(BaseModel):
    """Optional one-line comparison to the next-ranked candidate."""

    model_config = {"frozen": True}

    compared_to: str  # rank label like "#2"
    summary: str  # one Hungarian sentence


class AINarrative(BaseModel):
    """Per-result narrative payload returned by `generate_narrative`."""

    model_config = {"frozen": True}

    headline: str
    body: str
    compromise: str = ""  # empty when the match is clean
    rationale_chips: list[RationaleChip] = Field(default_factory=lambda: [])
    vs_next: VsNext | None = None
