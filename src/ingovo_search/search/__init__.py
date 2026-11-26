"""The buyer search orchestration behind ``POST /api/search``.

One request runs as ``request.execute_search_request`` (session, raw cache, parser, photos,
budget) around ``pipeline.run_search_pipeline`` (response cache, embeddings, candidates,
rerank, envelope). The stages share a :class:`~ingovo_search.search.run.SearchRun`.
``stream`` serves the same request as NDJSON progress events, ``clarify`` re-runs the
pipeline with the buyer's answer, ``probe`` is the homepage index preview, and
``scoring`` builds the card rows and the rerankers' inputs.
"""
