"""§8 soft-preference rule table — the full Appendix A vocabulary.

One predicate per soft-pref tag, mapping the listing's structured features to
a `bool`. `softpref_core_match` (in `score.py`) counts how many of a query's
`soft_preferences` evaluate `True` and normalises by count.

`SoftPrefTag` lives in `parser_schema.py` (the parser is the producer of these
tags), so the same Literal both validates parser output and keys these rules.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ingovo_search.recommender.geo_score import poi_distances_from_features
from ingovo_search.recommender.parser_schema import SoftPrefTag

# Predicates use `Any` for the listing param because SQLAlchemy's `Mapped[T]`
# wrapper breaks Protocol structural matching under strict pyright (instance
# access unwraps, class-level annotation does not). `ListingFeatures` in
# `ingovo_search.types.common` still documents the expected shape.
#
# `near_metro` reads the §12 nightly POI job's `features["poi_m"]["metro"]`
# distance (metres, populated from the OSM `poi` table). Missing data ->
# False: the tag simply earns no match credit, which lowers the soft score
# but never excludes the listing.
NEAR_METRO_MAX_M = 800.0


def _nearest_metro_within(listing: Any, max_m: float) -> bool:
    distances = poi_distances_from_features(getattr(listing, "features", None))
    distance = distances.get("metro") if distances else None
    return distance is not None and distance <= max_m


SOFTPREF_RULES: dict[SoftPrefTag, Callable[[Any], bool]] = {
    "renovated": lambda f: f.renovation_state in {"fully_renovated", "new_build"},
    "partially_renovated": lambda f: f.renovation_state in {"partial", "fully_renovated", "new_build"},
    "new_build": lambda f: f.renovation_state == "new_build",
    "fixer_upper": lambda f: f.renovation_state == "original",
    "parking": lambda f: bool(f.has_parking) or bool(f.has_garage),
    "garage": lambda f: bool(f.has_garage),
    "garden": lambda f: bool(f.has_garden),
    "terrace": lambda f: bool(f.has_terrace),
    "balcony": lambda f: bool(f.has_balcony),
    "basement": lambda f: bool(f.has_basement),
    "pool": lambda f: bool(getattr(f, "has_pool", None)),
    "elevator": lambda f: bool(f.elevator),
    "air_conditioning": lambda f: bool(f.air_conditioning),
    "natural_light": lambda f: (f.light_quality_score or 0.0) >= 0.6 or f.orientation in {"S", "SE", "SW", "E"},
    "sunny": lambda f: f.orientation in {"S", "SE", "SW"},
    "view": lambda f: (f.floor or 0) >= 3,
    "central": lambda f: f.district in {"5", "6", "7", "8", "9"},
    "quiet": lambda f: (f.noise_score or 1.0) <= 0.4,
    "near_metro": lambda f: _nearest_metro_within(f, NEAR_METRO_MAX_M),
    "walkable": lambda _f: True,
    "large": lambda f: (f.area_m2 or 0) >= 80,
    "small": lambda f: (f.area_m2 or 0) <= 50,
}
