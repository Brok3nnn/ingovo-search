"""Lexical amenity verification — keep the displayed match honest.

A buyer can explicitly ask for a concrete, high-signal amenity ("úszómedencés
villa") that lives only in free text: it is neither a structured filter column
nor a fixed soft-pref tag, so the recommender would otherwise match it by fuzzy
semantic similarity alone and happily show ~83% on a villa that has *no* pool —
while the Ingovo analysis, reading the real listing text, correctly says "nem
biztos, hogy van benne medence". The two disagree because they use different
sources of truth.

This module gives the scorer a tie-break to the *actual listing text*: it
detects which concrete amenities the query explicitly requested, then checks
whether each one is literally present in the listing's text fields. The scorer
uses the resulting ratio to downrank unverifiable matches, and the display layer
caps their visible match %. Purely lexical and dependency-free — it never
removes a candidate on its own; the recommender decides the penalty.
"""

from __future__ import annotations

from typing import Any, cast

from ingovo_search.recommender.parser_schema import ParsedQuery

# Canonical amenity → lowercase stem fragments that match its Hungarian
# spellings/inflections via substring test (accent-sensitive, lowercased). Stems
# are deliberately short so they catch declensions: "medenc" matches medence,
# medencés, úszómedence, kerti medencével, …
_AMENITY_STEMS: dict[str, tuple[str, ...]] = {
    "medence": ("medenc",),
    "kandalló": ("kandall",),
    "panoráma": ("panorám",),
    "szauna": ("szaun",),
    "jakuzzi": ("jakuzzi", "jacuzzi", "pezsgőfürd"),
    "napelem": ("napelem", "napkollektor"),
    "borospince": ("borospinc", "boros pinc"),
    "tetőterasz": ("tetőterasz", "tető terasz"),
    "télikert": ("télikert", "téli kert"),
    "dupla garázs": ("dupla garázs", "két állásos garázs", "kétállásos garázs", "2 állásos garázs"),
    "úszómedence": ("úszómedenc", "úszó medenc"),
}

# `úszómedence` and `medence` are the same real-world feature; collapse so a
# request for either is verified by any pool mention.
_CANONICAL_ALIAS: dict[str, str] = {"úszómedence": "medence"}

# Canonical amenity → structured boolean column on `Property`. When the column
# is populated it is AUTHORITATIVE (overrides the text guess); when it's NULL we
# fall back to the lexical text check. Only `medence`/`has_pool` exists for now.
_STRUCTURED_COLUMN: dict[str, str] = {"medence": "has_pool"}


def _norm(text: str) -> str:
    return (text or "").lower()


def _canonical(amenity: str) -> str:
    return _CANONICAL_ALIAS.get(amenity, amenity)


def query_text(query: ParsedQuery) -> str:
    """All buyer-typed text the amenity request can hide in (semantic + freetext)."""
    parts = [query.semantic_query or "", *(query.soft_prefs_freetext or [])]
    return _norm(" ".join(parts))


def requested_amenities(query: ParsedQuery) -> set[str]:
    """Concrete amenities the buyer explicitly asked for (canonicalised)."""
    text = query_text(query)
    found: set[str] = set()
    for amenity, stems in _AMENITY_STEMS.items():
        if any(stem in text for stem in stems):
            found.add(_canonical(amenity))
    return found


def listing_text(listing: Any) -> str:
    """Concatenated lowercase text of the listing fields that can attest a feature."""
    parts: list[str] = []
    for attr in ("title", "description", "cleaned_description", "one_paragraph_summary", "short_summary"):
        val = getattr(listing, attr, None)
        if isinstance(val, str) and val:
            parts.append(val)
    features = getattr(listing, "features", None)
    if isinstance(features, dict):
        parts.extend(str(v) for v in cast("dict[Any, Any]", features).values() if v is not None)
    elif isinstance(features, list | tuple):
        parts.extend(str(v) for v in cast("list[Any] | tuple[Any, ...]", features) if v is not None)
    return _norm(" ".join(parts))


def _listing_has(amenity: str, text: str) -> bool:
    # Check the canonical amenity plus every alias that maps onto it.
    keys = [amenity] + [src for src, dst in _CANONICAL_ALIAS.items() if dst == amenity]
    return any(stem in text for key in keys for stem in _AMENITY_STEMS.get(key, ()))


def _amenity_present(amenity: str, listing: Any, text: str) -> bool:
    """Is `amenity` attested for `listing`? Prefers the structured boolean column
    (authoritative when non-NULL), falling back to the lexical text check."""
    column = _STRUCTURED_COLUMN.get(amenity)
    if column is not None:
        value = getattr(listing, column, None)
        if value is not None:
            return bool(value)
    return _listing_has(amenity, text)


def amenity_match_ratio(query: ParsedQuery, listing: Any) -> float:
    """Fraction of the buyer's explicitly-requested concrete amenities that the
    listing actually attests (structured column first, then text).

    Returns ``1.0`` (no penalty) when the buyer requested no concrete amenity —
    the overwhelmingly common case — so ordinary searches are untouched.
    """
    requested = requested_amenities(query)
    if not requested:
        return 1.0
    text = listing_text(listing)
    verified = sum(1 for amenity in requested if _amenity_present(amenity, listing, text))
    return verified / len(requested)
