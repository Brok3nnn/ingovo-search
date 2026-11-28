# Changelog

All notable changes to this project are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- The demo listings carried building types (`panel`, `tegla`, …) as `property_type`, so every query that names a property type found nothing on a freshly seeded database; they now carry `Lakás` or `Ház`, with the building type in `features.build_type`.
- A Gemini error other than a timeout failed `POST /api/search`: `GoogleAiClient` now raises `LlmError` once every model failed, and the parser, the rerank and the region hint degrade instead (rule-based parse, deterministic order, no region hint).
- A listing could display a higher match percentage than its own after reranking; the displayed value is now the running minimum down the ranked list.
- The local Postgres image could not load PostGIS ("undefined symbol"); the image now upgrades the server along with the PostGIS package.
- `ingovo-search` printed a stack trace when the database was not running; it now names the database it tried (password hidden) and how to start and seed it, before any model call.

### Changed

- Demo listings name a street, not a building: `address` no longer carries a house number. The coordinates are unchanged.
- `POST /api/search` rejects a `query` longer than 600 characters and a `rerank_candidate_count` outside 1–30.
- `w_overrides` is gone from `/api/search`, `/api/refinement` and the session state: it was stored but never applied to the ranking.
- The CLI logs to stderr, warnings only unless `--verbose`, and `--json` now also applies to `--offline` and `--parse-only`.
- The synthetic data generator writes every description from a local template (the unused Gemini path is gone) and seeds its random generator.
- The scripts that built the bundled points-of-interest extract and the settlement dataset are in `scripts/`.
- `typesafe-sdk` is updated to 0.7.5, and its requirement is now `>=0.7.5,<0.8`.
- The eval scripts print plain-text status lines instead of emoji.

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

[Unreleased]: https://github.com/Brok3nnn/ingovo-search/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Brok3nnn/ingovo-search/releases/tag/v1.0.0
