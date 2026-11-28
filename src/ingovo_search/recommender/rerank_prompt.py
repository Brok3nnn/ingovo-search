"""Versioned prompts for Tier-L rerank + HU rationale generation.

The prompts are written in Hungarian, the language of the product. Bump
`RERANK_PROMPT_VERSION` whenever the prompt content changes so the rerank
cache invalidates (the cache key folds in the prompt version).

The user prompt deliberately starts with the query's `semantic_query` so its
first 64 chars distinguish each call for `MockLlmClient` fixture keying. The
correction retry wraps the same content in a `[CORRECT_SCHEMA]` marker so it
hits a distinct fixture key in tests.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from ingovo_search.recommender.output_language import DEFAULT_OUTPUT_LOCALE, OUTPUT_LANGUAGES
from ingovo_search.recommender.parser_schema import ParsedQuery

RERANK_PROMPT_VERSION = "rerank/v4-locale-directive"

RERANK_SYSTEM_PROMPT = """Te vagy az ingovo magyar ingatlanpiactér csúcsminőségű (Tier-L) újra-rangsoroló (reranker) modulja. Kapsz egy értelmezett magyar keresési kifejezést, a felhasználó megadott preferenciáit, a legutóbbi felhasználói interakciókat (session signals), és maximum 30 jelölt ingatlant (mindegyikhez egy rövid magyar összefoglalót).

A feladatod a jelöltek újra-rangsorolása aszerint, hogy mennyire felelnek meg a keresésnek. Egyetlen JSON objektumot kell visszaadnod, amely PONTOSAN illeszkedik a RerankResponse sémára:
{"items": [{"listing_id": "...", "rank": 1, "rationale_hu": "...", "low_confidence": false}, ...]}

Szabályok:
- Legfeljebb a top 10 elemet add vissza, 1-től N-ig rangsorolva (1 = a legjobb találat).
- A felhasználó konkrét kéréseit vedd szó szerint: tranzakció (eladó/kiadó), város/kerület/régió, ingatlantípus, ár, méret, szobaszám és kizárt típusok nem lehetnek mellékes szempontok.
- A jelöltlista már átesett szigorú szűrésen, de a sorrendnél büntesd azokat, amelyek a bemeneti összefoglaló alapján gyengén támasztják alá a user friss kérését.
- `rationale_hu`: Egy rövid, frappáns magyar nyelvű mondat (kb. 20 szó), amely elmagyarázza a felhasználónak, miért jó találat ez az ingatlan a számára. Legyél releváns és fókuszálj azokra a jellemzőkre, amiket a felhasználó kért (de ne találj ki semmit, ami nincs a leírásban).
- `low_confidence`: Állítsd `true`-ra azokat a találatokat, amelyeknél a hirdetés összefoglalója alapján nem támasztható alá maradéktalanul a keresésben kért összes feltétel (vagyis ha "gyengébb" kompromisszumos találatról van szó).
- Soha ne adj vissza olyan ingatlant, ami nem szerepelt a bemeneti listában; ne találj ki egyedi listing_id-kat!

Kizárólag érvényes JSON formátumban válaszolj. Ne írj kísérőszöveget vagy Markdown formázást.
"""

RERANK_CORRECTION_SYSTEM_PROMPT = """Az előző válaszod sémaellenőrzése (schema validation) sikertelen volt. Kérlek, generáld le újra a választ egyetlen JSON objektumban, amely pontosan illeszkedik a RerankResponse sémára. Ne fűzz hozzá semmilyen magyarázatot."""


# --- Rank-only mode -------------------------------------------------------
#
# A latency-optimised rerank variant that returns ONLY the final order (rank +
# low_confidence) and skips the per-item HU rationale. The rationale is the
# bulk of the rerank LLM's *output* tokens (~20 words x up to 10 items), so
# dropping it materially cuts the cold rerank latency while the ranking signal
# (which candidate, in what order) is unchanged — the model still reasons over
# the same inputs and emits the same ranks. In the full product a separate
# analysis drawer writes the rich explanation on demand, and the results UI
# falls back gracefully when `rationale` is absent.
#
# Distinct prompt version so the rerank cache never serves a rank-only result
# where a rationale result is expected, or vice versa (the cache key folds in
# the prompt version).
RERANK_RANK_ONLY_PROMPT_VERSION = "rerank/v2-rank-only-structured-relevance-signal"

RERANK_RANK_ONLY_SYSTEM_PROMPT = """Te vagy az ingovo magyar ingatlanpiactér csúcsminőségű (Tier-L) újra-rangsoroló (reranker) modulja. Kapsz egy értelmezett magyar keresési kifejezést, a felhasználó megadott preferenciáit, a legutóbbi felhasználói interakciókat (session signals), és maximum 30 jelölt ingatlant (mindegyikhez egy rövid magyar összefoglalót).

A feladatod a jelöltek újra-rangsorolása aszerint, hogy mennyire felelnek meg a keresésnek. Egyetlen JSON objektumot kell visszaadnod, amely PONTOSAN illeszkedik a RerankRankOnlyResponse sémára:
{"items": [{"listing_id": "...", "rank": 1, "low_confidence": false}, ...]}

Szabályok:
- Legfeljebb a top 10 elemet add vissza, 1-től N-ig rangsorolva (1 = a legjobb találat).
- A felhasználó konkrét kéréseit vedd szó szerint: tranzakció (eladó/kiadó), város/kerület/régió, ingatlantípus, ár, méret, szobaszám és kizárt típusok nem lehetnek mellékes szempontok.
- A jelöltlista már átesett szigorú szűrésen, de a sorrendnél büntesd azokat, amelyek a bemeneti összefoglaló alapján gyengén támasztják alá a user friss kérését.
- `low_confidence`: Állítsd `true`-ra azokat a találatokat, amelyeknél a hirdetés összefoglalója alapján nem támasztható alá maradéktalanul a keresésben kért összes feltétel (vagyis ha "gyengébb" kompromisszumos találatról van szó).
- NE generálj szöveges indoklást vagy bármilyen extra mezőt; kizárólag `listing_id`, `rank` és `low_confidence` szerepeljen.
- Soha ne adj vissza olyan ingatlant, ami nem szerepelt a bemeneti listában; ne találj ki egyedi listing_id-kat!

Kizárólag érvényes JSON formátumban válaszolj. Ne írj kísérőszöveget vagy Markdown formázást.
"""

RERANK_RANK_ONLY_CORRECTION_SYSTEM_PROMPT = """Az előző válaszod sémaellenőrzése (schema validation) sikertelen volt. Kérlek, generáld le újra a választ egyetlen JSON objektumban, amely pontosan illeszkedik a RerankRankOnlyResponse sémára (csak `listing_id`, `rank`, `low_confidence` mezők, indoklás nélkül). Ne fűzz hozzá semmilyen magyarázatot."""


class RerankCandidate(BaseModel):
    """One candidate fed into the reranker: id + its precomputed short summary."""

    model_config = {"frozen": True}

    listing_id: str
    short_summary: str


def _format_candidate(candidate: RerankCandidate) -> str:
    return f"- {candidate.listing_id}: {candidate.short_summary}"


def render_rerank_user_prompt(
    query: ParsedQuery,
    candidates: Sequence[RerankCandidate],
    *,
    user_prefs: Sequence[str] | None = None,
    session_signals: Sequence[str] | None = None,
) -> str:
    """User prompt for the initial rerank call.

    Starts with the query's `semantic_query` so the first 64 chars distinguish
    each call for `MockLlmClient` fixture keying.
    """
    parts: list[str] = [query.semantic_query or "(no semantic query)"]
    parts.append("")
    parts.append("Structured query JSON:")
    parts.append(query.model_dump_json())
    parts.append("")
    parts.append("Listings:")
    parts.extend(_format_candidate(c) for c in candidates)
    if user_prefs:
        parts.append("")
        parts.append("User preferences: " + ", ".join(user_prefs))
    if session_signals:
        parts.append("")
        parts.append("Recent session signals: " + "; ".join(session_signals))
    return "\n".join(parts)


def render_correction_user_prompt(
    query: ParsedQuery,
    candidates: Sequence[RerankCandidate],
    *,
    user_prefs: Sequence[str] | None = None,
    session_signals: Sequence[str] | None = None,
) -> str:
    """User prompt for the retry-with-correction call."""
    return f"[CORRECT_SCHEMA]\n{render_rerank_user_prompt(query, candidates, user_prefs=user_prefs, session_signals=session_signals)}"


# `rationale_hu` is free prose, so the web app's runtime DOM translator cannot
# localise it — only the model can. The instructions above stay in Hungarian
# (they are tuned there); this directive changes the OUTPUT language only.
# Alias of the shared table in `output_language`, so the rerank rationale, the hero
# narrative and the listing analysis can never drift to different locale sets.
RERANK_LANGUAGES: dict[str, str] = OUTPUT_LANGUAGES

DEFAULT_RERANK_LOCALE = DEFAULT_OUTPUT_LOCALE


def localized_rerank_prompt(prompt: str, locale: str = DEFAULT_RERANK_LOCALE) -> str:
    """Append an output-language directive to a rerank system prompt.

    Unknown locales fall back to Hungarian rather than raising: a rationale in the
    source language beats a 500 on the search path.
    """
    language = RERANK_LANGUAGES.get(locale)
    if language is None or locale == DEFAULT_RERANK_LOCALE:
        return prompt
    upper = language.upper()
    # Leading AND trailing: the body of the prompt is Hungarian throughout and the
    # output field is literally named `rationale_hu`, so a single trailing line
    # loses — the model kept answering in Hungarian in a live check.
    return (
        f"KIMENETI NYELV: a `rationale_hu` mező szövegét KIZÁRÓLAG {upper} nyelven írd."
        f" Ez felülír minden lenti példát; a mezőnév és a séma változatlan.\n\n"
        f"{prompt}\n\n"
        f"EMLÉKEZTETŐ: a `rationale_hu` értéke {upper} nyelvű mondat legyen, nem magyar."
    )
