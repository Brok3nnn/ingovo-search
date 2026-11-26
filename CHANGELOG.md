# Changelog

All notable changes to this project are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-26

First public release: the AI property search extracted from the Ingovo product as a standalone service.

### Added

- Natural-language query parser: deterministic fast path, Gemini structured output with a correction retry, rule-based post-processing, Hungarian settlement registry.
- SQL and PostGIS hard filters with stepwise scope expansion (relaxation, nearby settlements, nearest city).
- pgvector retrieval over Cohere embeddings and a ten-term deterministic score with calibrated weights.
- Gemini rerank with guardrails (candidate set preserved, bounded movement, timeouts), and an optional TypeSafe Jev reranker with canary and shadow modes.
- Absolute, capped match percentages with per-criterion components.
- FastAPI endpoints for search, streaming search, photo search, clarification, preview and probe.
- `ingovo-search` CLI, `ingovo-search-seed` command, a single idempotent schema file and a docker compose stack.
- 1,800 synthetic listings and an OpenStreetMap points-of-interest extract as demo data.
- Offline test suite, golden snapshots, and the parser, ranking and rerank evals.
