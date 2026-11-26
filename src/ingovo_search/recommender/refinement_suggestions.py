"""M3 refinement-suggestion generator — 4-6 one-click filter tweaks.

After every `/api/search` response the orchestrator emits a list of
`RefinementSuggestion`s that the UI renders as button-chips ("Lazítok az
áron 20%-kal", "Csak új építésű", "Bővítem az ország egészére"). Clicking
the chip POSTs the suggestion's `overrides` field back into the next
`/api/search` request via `RefinementOverrides`, so the AI re-runs with the
adjusted parsed query.

Heuristics, no LLM — every suggestion is derived deterministically from
`(parsed_query, top_results, relaxation_path)`:

- **Loosen** when the result set is sparse (< `LOOSEN_THRESHOLD`): widen
  price, drop district, drop county, drop Balaton shore, drop named region.
- **Tighten** when the result set is plentiful (> `TIGHTEN_THRESHOLD`):
  suggest the dominant property type, or a common soft preference the user
  did not specify (`new_build`, `balcony`, `parking`, `garden`).
- **Reverse a relaxation** — if the search pipeline had to relax a filter
  (`relaxation_path` non-empty), offer a chip to put the filter back.

Returns at most `MAX_SUGGESTIONS` items, deduplicated by `kind`. The
function is pure: same inputs ⇒ same suggestions, so the §13 search-cache
key can include it.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from ingovo_search.lib.session import RefinementOverrides
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.score import ScoredCandidate

LOOSEN_THRESHOLD = 4  # results below this -> emit loosening suggestions
TIGHTEN_THRESHOLD = 12  # results above this -> emit tightening suggestions
MAX_SUGGESTIONS = 6

type RefinementKind = Literal[
    "loosen_price_20pct",
    "drop_district",
    "drop_county",
    "drop_balaton_shore",
    "drop_region_name",
    "tighten_type",
    "add_new_build",
    "add_balcony",
    "add_parking",
    "add_garden",
    "revert_relaxation",
]


def _empty_overrides_factory() -> RefinementOverrides:
    return RefinementOverrides()


class RefinementSuggestion(BaseModel):
    """One button-chip the UI offers as a one-click query tweak."""

    model_config = {"frozen": True}

    kind: RefinementKind
    label: str  # Hungarian, ≤32 chars, button-text
    overrides: RefinementOverrides = Field(default_factory=_empty_overrides_factory)
    rationale: str | None = None  # optional tooltip; one-liner


# --- helpers ----------------------------------------------------------------


def _dominant_property_type(results: list[ScoredCandidate]) -> str | None:
    """Most-frequent `property.type` across the top results, or None on tie."""
    if not results:
        return None
    types = [r.property.get("type") for r in results if r.property and r.property.get("type")]
    if not types:
        return None
    counts = Counter(types)
    most_common = counts.most_common(2)
    if len(most_common) == 1:
        return str(most_common[0][0])
    # Require a clear majority — at least 2x runner-up — to avoid flapping suggestions.
    top, runner = most_common
    if top[1] >= 2 * runner[1]:
        return str(top[0])
    return None


# --- loosening suggestions --------------------------------------------------


def _format_millions(value_huf: int) -> str:
    """Format one decimal million using integer arithmetic only.

    This label is intentionally approximate, but the approximation itself must
    not be derived from a rounded binary float once a user supplies an exact
    bigint HUF budget.
    """

    whole, remainder = divmod(value_huf, 1_000_000)
    tenths = (remainder + 50_000) // 100_000
    if tenths == 10:
        whole += 1
        tenths = 0
    return str(whole) if tenths == 0 else f"{whole}.{tenths}"


def _suggest_loosen_price(parsed: ParsedQuery) -> RefinementSuggestion | None:
    """If a price ceiling is set, offer a 20% widening."""
    hf = parsed.hard_filters
    if hf.price_huf_max is None:
        return None
    # ceil(price * 6 / 5), without binary-float rounding of an exact HUF cap.
    new_max = (hf.price_huf_max * 6 + 4) // 5
    new_max_m = _format_millions(new_max)
    new_hf = hf.model_copy(update={"price_huf_max": new_max})
    return RefinementSuggestion(
        kind="loosen_price_20pct",
        label=f"Max ár {new_max_m} M-ig",
        overrides=RefinementOverrides(hard_filters=new_hf),
        rationale="Lazítok 20%-kal az ár-felső határon, hátha több találatot ad.",
    )


def _suggest_drop_district(parsed: ParsedQuery) -> RefinementSuggestion | None:
    """If specific districts are set, offer to drop them (keep city)."""
    hf = parsed.hard_filters
    if not hf.district_in:
        return None
    new_hf = hf.model_copy(update={"district_in": []})
    return RefinementSuggestion(
        kind="drop_district",
        label="Bármelyik kerület",
        overrides=RefinementOverrides(hard_filters=new_hf),
        rationale=f"A {', '.join(hf.district_in)} kerület-szűrést elveszem; csak a város marad.",
    )


def _suggest_drop_county(parsed: ParsedQuery) -> RefinementSuggestion | None:
    """If a county hint is set, offer to drop it."""
    if parsed.geo_hints.county is None:
        return None
    new_hints = parsed.geo_hints.model_copy(update={"county": None})
    return RefinementSuggestion(
        kind="drop_county",
        label="Megye nélkül országosan",
        overrides=RefinementOverrides(geo_hints=new_hints),
        rationale=f"A '{parsed.geo_hints.county} megye' korlátot elveszem.",
    )


def _suggest_drop_balaton_shore(parsed: ParsedQuery) -> RefinementSuggestion | None:
    """If a Balaton shore is set, offer 'mindkét part'."""
    if parsed.geo_hints.balaton_shore is None:
        return None
    new_hints = parsed.geo_hints.model_copy(update={"balaton_shore": None})
    return RefinementSuggestion(
        kind="drop_balaton_shore",
        label="Mindkét Balaton-part",
        overrides=RefinementOverrides(geo_hints=new_hints),
    )


def _suggest_drop_region_name(parsed: ParsedQuery) -> RefinementSuggestion | None:
    """If a named region is set, offer to drop it."""
    if parsed.geo_hints.region_name is None:
        return None
    new_hints = parsed.geo_hints.model_copy(update={"region_name": None})
    return RefinementSuggestion(
        kind="drop_region_name",
        label="Régió nélkül",
        overrides=RefinementOverrides(geo_hints=new_hints),
        rationale=f"A '{parsed.geo_hints.region_name}' régió-szűkítést elveszem.",
    )


# --- tightening suggestions -------------------------------------------------


def _suggest_tighten_type(parsed: ParsedQuery, results: list[ScoredCandidate]) -> RefinementSuggestion | None:
    """If no property_type_in is set but the result set has a dominant type, suggest it."""
    if parsed.hard_filters.property_type_in:
        return None
    dom = _dominant_property_type(results)
    if dom is None:
        return None
    # Type must be one of the canonical PropertyType literals; if not, skip.
    canonical = {
        "Lakás",
        "Ház",
        "Iroda",
        "Üzlethelyiség",
        "Telek",
        "Garázs",
        "Ipari",
        "Nyaraló",
        "Vendéglátás",
        "Mezőgazdasági",
    }
    if dom not in canonical:
        return None
    new_hf = parsed.hard_filters.model_copy(update={"property_type_in": [dom]})  # pyright: ignore[reportArgumentType]
    return RefinementSuggestion(
        kind="tighten_type",
        label=f"Csak {dom}",
        overrides=RefinementOverrides(hard_filters=new_hf),
        rationale=f"A találatok többsége {dom} — szűkítsem erre.",
    )


def _suggest_add_soft_pref(
    parsed: ParsedQuery,
    *,
    tag: Literal["new_build", "balcony", "parking", "garden"],
    label: str,
) -> RefinementSuggestion | None:
    """If `tag` is not already in `soft_preferences`, suggest adding it."""
    if tag in parsed.soft_preferences:
        return None
    new_prefs: list[str] = [*parsed.soft_preferences, tag]
    return RefinementSuggestion(
        kind=f"add_{tag}",  # pyright: ignore[reportArgumentType]
        label=label,
        overrides=RefinementOverrides(soft_preferences=new_prefs),  # pyright: ignore[reportArgumentType]
    )


# --- relaxation reversal ----------------------------------------------------


_RELAX_REVERT_LABEL: dict[str, str] = {
    "max ár": "Visszateszem a max árt",
    "kerület": "Visszateszem a kerület-szűrőt",
    "pontos szobaszám": "Visszateszem a szobaszámot",
    "min. szobaszám": "Visszateszem a min. szobaszámot",
    "méret": "Visszateszem a méret-szűrőt",
    "min. emelet": "Visszateszem az emelet-szűrőt",
    "kizárt típusok": "Visszateszem a típus-kizárást",
    "közelség-sugár": "Eredeti közelségi sugár",
    "város": "Visszateszem a város-szűrőt",
}


def _suggest_revert_relaxation(original: ParsedQuery, relaxation_path: list[str]) -> RefinementSuggestion | None:
    """If the pipeline had to relax a filter, offer to restore the strict version."""
    if not relaxation_path:
        return None
    last_relaxed = relaxation_path[-1]  # most-recent step is the one that produced results
    label = _RELAX_REVERT_LABEL.get(last_relaxed, "Visszateszem a szigorú szűrőket")
    return RefinementSuggestion(
        kind="revert_relaxation",
        label=label,
        overrides=RefinementOverrides(hard_filters=original.hard_filters),
        rationale=f"A keresés ezt lazította: {last_relaxed}. Visszateszem a szigorú változatot.",
    )


# --- top-level -------------------------------------------------------------


def _append_new(suggestions: list[RefinementSuggestion], suggestion: RefinementSuggestion | None) -> None:
    """Append `suggestion` unless it is None or a chip of its kind is already listed."""
    if suggestion is not None and not any(existing.kind == suggestion.kind for existing in suggestions):
        suggestions.append(suggestion)


def _loosening(relaxed_parsed: ParsedQuery, suggestions: list[RefinementSuggestion]) -> None:
    """Sparse result set → loosen."""
    for fn in (
        _suggest_loosen_price,
        _suggest_drop_district,
        _suggest_drop_county,
        _suggest_drop_balaton_shore,
        _suggest_drop_region_name,
    ):
        _append_new(suggestions, fn(relaxed_parsed))


def _tightening(
    relaxed_parsed: ParsedQuery, results: list[ScoredCandidate], suggestions: list[RefinementSuggestion]
) -> None:
    """Plentiful → tighten."""
    _append_new(suggestions, _suggest_tighten_type(relaxed_parsed, results))
    for tag, label in (
        ("new_build", "Csak új építésű"),
        ("balcony", "Csak erkélyes"),
        ("parking", "Csak parkolóval"),
        ("garden", "Csak kerttel"),
    ):
        _append_new(
            suggestions,
            _suggest_add_soft_pref(relaxed_parsed, tag=tag, label=label),  # pyright: ignore[reportArgumentType]
        )


def _mixing(
    relaxed_parsed: ParsedQuery, results: list[ScoredCandidate], suggestions: list[RefinementSuggestion]
) -> None:
    """Mid-range → mix: one tighten + one loosen for variety."""
    tighten = _suggest_tighten_type(relaxed_parsed, results)
    if tighten is not None:
        suggestions.append(tighten)
    for fn in (_suggest_drop_county, _suggest_drop_balaton_shore, _suggest_loosen_price):
        _append_new(suggestions, fn(relaxed_parsed))


def suggest_refinements(
    *,
    original_parsed: ParsedQuery,
    relaxed_parsed: ParsedQuery,
    results: list[ScoredCandidate],
    relaxation_path: list[str],
) -> list[RefinementSuggestion]:
    """Compose the per-search refinement chip list.

    `original_parsed` is the parsed query BEFORE relaxation; `relaxed_parsed`
    is what actually drove `recommend()`. We branch on result count to bias
    loosening vs tightening; relaxation reversal is offered first when
    applicable since it directly undoes the pipeline's automatic decision.
    """
    suggestions: list[RefinementSuggestion] = []

    # 1. If the pipeline relaxed a filter, offer the reverse first.
    revert = _suggest_revert_relaxation(original_parsed, relaxation_path)
    if revert is not None:
        suggestions.append(revert)

    n = len(results)
    if n < LOOSEN_THRESHOLD:
        _loosening(relaxed_parsed, suggestions)
    elif n > TIGHTEN_THRESHOLD:
        _tightening(relaxed_parsed, results, suggestions)
    else:
        _mixing(relaxed_parsed, results, suggestions)

    return suggestions[:MAX_SUGGESTIONS]
