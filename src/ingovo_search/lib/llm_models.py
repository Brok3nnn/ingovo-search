"""Single source of truth for LLM model IDs. Swap models here, never at call sites.

Primary models use `gemini-2.5-flash-lite` for cost efficiency. Each tier has an
automatic fallback to `gemini-2.5-flash` when the primary returns 503 (high demand).
The Tier-L rerank is the most likely tier to need a bump if the Hungarian rationale
quality is not good enough.
"""

TIER_S_PARSER: str = "gemini-2.5-flash-lite"
TIER_M_ENRICH: str = "gemini-2.5-flash-lite"
TIER_L_RERANK: str = "gemini-2.5-flash-lite"
# Image understanding for the multimodal search entry point (one call per upload
# batch, cached by image content). `flash-lite` reads photos well enough for the
# coarse "what kind of home is this" signal we merge into the query; bump to
# `gemini-2.5-flash` here if rationale review shows it misreads condition/style.
TIER_V_IMAGE_QUERY: str = "gemini-2.5-flash-lite"
# Explicit retry order for image understanding. Only vision-capable models
# belong here — a text-only fallback would fail every attempt on the same
# request. Verified live against the API on 2026-07-26; `gemini-2.0-flash` is
# deliberately absent because that endpoint is retired (404).
TIER_V_IMAGE_QUERY_CHAIN: tuple[str, ...] = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite",
)

TIER_S_PARSER_FALLBACK: str = "gemini-2.5-flash"
TIER_M_ENRICH_FALLBACK: str = "gemini-2.5-flash"
TIER_L_RERANK_FALLBACK: str = "gemini-2.5-flash"
TIER_V_IMAGE_QUERY_FALLBACK: str = "gemini-2.5-flash"

# TypeSafe Jev, the optional second reranker. Pinned to a versioned id rather than
# the `jev-latest` alias: the alias moves on every release, which would silently
# shift the score thresholds calibrated against this version.
TIER_L_RERANK_JEV: str = "jev-1.13.0"
