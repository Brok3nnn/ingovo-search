from types import SimpleNamespace

from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.score import ScoreInputs, compute_score


def test_compute_score_tolerates_missing_raw_embeddings() -> None:
    """A not-yet-embedded listing (no raw vector attributes at all) must score
    with zero embedding credit, not crash — freshly ingested rows enter the
    candidate pool before the §5 backfill runs."""
    listing = SimpleNamespace(
        id="L-new",
        created_at=None,
        price_huf=50_000_000,
        area_m2=55.0,
        rooms="2",
        floor=None,
        district=None,
        renovation_state=None,
        orientation=None,
        noise_score=None,
        light_quality_score=None,
        has_balcony=None,
        has_terrace=None,
        has_parking=None,
        has_garage=None,
        has_garden=None,
        has_basement=None,
        has_pool=None,
        elevator=None,
        air_conditioning=None,
        features=None,
        quality_score=None,
    )
    query = ParsedQuery(soft_prefs_freetext=["vízpart"])

    breakdown = compute_score(
        listing, query, ScoreInputs(query_emb=[0.1] * 4, freetext_pref_emb=[0.1] * 4, explore=0.0)
    )

    assert breakdown.content == 0.0
    assert breakdown.factual == 0.0
    assert breakdown.softpref_free == 0.0
    assert breakdown.total >= 0.0
