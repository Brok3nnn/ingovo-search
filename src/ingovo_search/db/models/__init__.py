"""SQLAlchemy models for the tables the search reads and writes.

Importing this module registers every table with `Base.metadata`. The
authoritative DDL (extensions, generated columns, HNSW and GIST indexes) is
`ingovo_search/db/schema.sql`; `tests/unit/db/test_schema_matches_models.py`
keeps the two in step.
"""

from ingovo_search.db.models.base import Base
from ingovo_search.db.models.enrichment_log import EnrichmentLog
from ingovo_search.db.models.interaction import Interaction
from ingovo_search.db.models.poi import Poi
from ingovo_search.db.models.property import Property
from ingovo_search.db.models.region_centroid import RegionCentroid
from ingovo_search.db.models.rerank_shadow import RerankShadowComparison
from ingovo_search.db.models.search_log import SearchLog
from ingovo_search.db.models.synthetic_eval_rating import SyntheticEvalRating

__all__ = [
    "Base",
    "EnrichmentLog",
    "Interaction",
    "Poi",
    "Property",
    "RegionCentroid",
    "RerankShadowComparison",
    "SearchLog",
    "SyntheticEvalRating",
]
