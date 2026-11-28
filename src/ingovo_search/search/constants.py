"""The search pipeline's tuning constants, UI labels and loggers."""

from __future__ import annotations

import logging

import structlog

SCORE_VERSION: str = "v3-authoritative-rank-aligned-match"

SEARCH_CACHE_TTL_SECONDS: int = 600

QUERY_EMBEDDING_CACHE_TTL_SECONDS: int = 60 * 60 * 24 * 7

SEARCH_RESPONSE_CACHE_VERSION: str = "response-v24-running-minimum-match"

RAW_SEARCH_RESPONSE_CACHE_VERSION: str = f"{SEARCH_RESPONSE_CACHE_VERSION}:raw-v1"

RAW_SEARCH_RESPONSE_RANK_ONLY_CACHE_VERSION: str = f"{RAW_SEARCH_RESPONSE_CACHE_VERSION}:rank-only"

RERANK_CANDIDATE_COUNT: int = 12

MIN_VISIBLE_RESULT_COUNT: int = 12

SOFT_PREFERENCE_LABELS: dict[str, str] = {
    "renovated": "Felújított",
    "partially_renovated": "Részben felújított",
    "new_build": "Új építésű",
    "fixer_upper": "Felújítandó",
    "parking": "Parkolás",
    "garage": "Garázs",
    "garden": "Kert",
    "terrace": "Terasz",
    "balcony": "Erkély",
    "basement": "Pince",
    "pool": "Medence",
    "elevator": "Lift",
    "air_conditioning": "Klíma",
    "natural_light": "Világos",
    "sunny": "Napos",
    "view": "Kilátás",
    "central": "Központi elhelyezkedés",
    "quiet": "Csendes",
    "near_metro": "Metró közelében",
    "walkable": "Gyalog jól elérhető",
    "large": "Tágas",
    "small": "Kompakt",
}


SOFT_PREF_FIELD: dict[str, tuple[str, str]] = {
    "parking": ("has_parking", "parkoló"),
    "garage": ("has_garage", "garázs"),
    "garden": ("has_garden", "kert"),
    "terrace": ("has_terrace", "terasz"),
    "balcony": ("has_balcony", "erkély"),
    "basement": ("has_basement", "pince"),
    "pool": ("has_pool", "medence"),
    "elevator": ("elevator", "lift"),
    "air_conditioning": ("air_conditioning", "klíma"),
}

RERANK_DESCRIPTION_MAX_CHARS = 280

HERO_NARRATIVE_COUNT = 10  # how many top results get the full AI narrative

search_logger = structlog.get_logger("ingovo_search.api.search")

narrator_logger = logging.getLogger("ingovo_search.api.search.narrator")

# The two relaxation labels the context chips and the candidate stages share.
EXPLICIT_LOCATION_EXPANSION_LABEL = "távolabbi terület (jóváhagyva)"
AUTOMATIC_LOCATION_FALLBACK_LABEL = "legközelebbi elérhető település"
