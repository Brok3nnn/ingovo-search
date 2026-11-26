"""The homepage search preview's response contract.

The three frozen models `build_search_preview` returns — one `SearchPreview` holding the
canonical `ParsedQuery`, the image concepts and the search directions — and the two literal
types the builders share. `/api/search/preview` serializes them as they are, and the main
search reads the summary and confidence for its progress event; the parsed query keeps unsafe
HUF values exact at the JSON boundary through `public_parsed_query_json`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_serializer

from ingovo_search.recommender.parser_schema import ParsedQuery, public_parsed_query_json

type PreviewLocale = Literal["hu", "en"]
type DirectionKind = Literal["exact", "priority", "lifestyle", "alternative", "guided"]


class SearchDirection(BaseModel):
    """One concise, explainable search direction shown in the right column."""

    model_config = {"frozen": True}

    id: str
    kind: DirectionKind
    title: str
    summary: str
    delta_label: str | None = None
    canonical_query: str
    retained_constraints: list[str] = Field(default_factory=list)
    changed_preferences: list[str] = Field(default_factory=list)


class ImageConcept(BaseModel):
    """A semantic visual brief; the web app maps it to an approved stock photo."""

    model_config = {"frozen": True}

    id: str
    title: str
    reason: str
    tags: list[str]
    canonical_query: str


class SearchPreview(BaseModel):
    """Fast homepage interpretation response."""

    model_config = {"frozen": True}

    request_id: str
    intent_hash: str
    confidence: float
    parsed_query: ParsedQuery
    understood_summary: str
    image_concepts: list[ImageConcept]
    search_directions: list[SearchDirection]
    clarification: str | None = None

    @field_serializer("parsed_query", when_used="json")
    def serialize_parsed_query(self, value: ParsedQuery) -> dict[str, object]:
        """Keep unsafe HUF intent values exact at the JSON/browser boundary."""

        return public_parsed_query_json(value)
