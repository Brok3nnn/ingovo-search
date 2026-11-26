"""The query and free-text embeddings a search scores against.

The query embedding may already be on its way: the request starts a speculative cache
read of the raw sentence before the parser runs, and it is used when the parser kept the
sentence as the semantic query.
"""

from __future__ import annotations

from ingovo_search.lib.cache import normalise_query
from ingovo_search.lib.embeddings import EMBEDDING_DIM
from ingovo_search.search.embedding_cache import discard_speculative_embedding
from ingovo_search.search.run import SearchRun
from ingovo_search.search.scoring import embed_with_cache


async def _embed(run: SearchRun, text_value: str, timing_prefix: str) -> list[float]:
    return await embed_with_cache(
        text_value=text_value,
        input_type="search_query",
        embedding_client=run.deps.embedding_client,
        cache=run.deps.cache,
        timings=run.timings,
        timing_prefix=timing_prefix,
    )


async def embed_query(run: SearchRun) -> None:
    """The semantic query's embedding: speculative read, cache, or the embedding client."""
    timings = run.timings
    speculative = run.options.speculative_query_embedding
    semantic_query = run.parsed.semantic_query
    if not semantic_query:
        await discard_speculative_embedding(speculative)
        run.query_emb = [0.0] * EMBEDDING_DIM
        timings["embedding_query"] = 0
        timings["embedding_query_cache_hit"] = 0
        timings["embedding_query_speculative_wait"] = 0
        timings["embedding_query_speculative_cache_hit"] = 0
        return

    speculative_used = False
    if speculative is not None and normalise_query(semantic_query) == speculative.normalised_text:
        step_started = run.timer.now()
        cached_embedding = await speculative.task
        run.timer.record("embedding_query_speculative_wait", step_started)
        timings["embedding_query_speculative_cache_hit"] = 1 if cached_embedding is not None else 0
        if cached_embedding is not None:
            speculative_used = True
            timings["embedding_query_cache_read"] = timings["embedding_query_speculative_wait"]
            timings["embedding_query_cache_hit"] = 1
            timings["embedding_query"] = 0
            run.query_emb = cached_embedding.embedding
        else:
            run.query_emb = await _embed(run, semantic_query, "embedding_query")
    else:
        await discard_speculative_embedding(speculative)
        run.query_emb = await _embed(run, semantic_query, "embedding_query")
    if not speculative_used and "embedding_query_speculative_wait" not in timings:
        timings["embedding_query_speculative_wait"] = 0
        timings["embedding_query_speculative_cache_hit"] = 0


async def embed_freetext(run: SearchRun) -> None:
    """The embedding of the free-text soft preferences, when the parser found any."""
    if run.parsed.soft_prefs_freetext:
        run.freetext_emb = await _embed(run, " ".join(run.parsed.soft_prefs_freetext), "embedding_freetext")
        return
    run.timings["embedding_freetext"] = 0
    run.timings["embedding_freetext_cache_hit"] = 0
