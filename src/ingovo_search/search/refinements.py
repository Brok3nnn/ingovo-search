"""Session refinement overrides applied to the parsed query."""

from __future__ import annotations

from typing import Any

from ingovo_search.lib.session import (
    RefinementOverrides,
)
from ingovo_search.recommender.parser_schema import ParsedQuery


def apply_refinements(parsed: ParsedQuery, overrides: RefinementOverrides) -> ParsedQuery:
    updates: dict[str, Any] = {}
    if overrides.hard_filters is not None:
        updates["hard_filters"] = overrides.hard_filters
    if overrides.anchors is not None:
        updates["anchors"] = overrides.anchors
    if overrides.soft_preferences is not None:
        updates["soft_preferences"] = overrides.soft_preferences
    if overrides.soft_prefs_freetext is not None:
        updates["soft_prefs_freetext"] = overrides.soft_prefs_freetext
    if overrides.geo_hints is not None:
        updates["geo_hints"] = overrides.geo_hints
    return parsed.model_copy(update=updates) if updates else parsed
