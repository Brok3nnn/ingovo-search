# Contributing

Thanks for your interest. This repository is a showcase extracted from a larger product, so the scope is the search engine itself: the parser, the filters, retrieval, ranking, the match score and their evals.

## Setup

```bash
uv sync
uv run pytest
```

The offline suite needs no keys, database or Docker. For database-backed work, see "Full stack" in the README (`docker compose up -d`, `uv run ingovo-search-seed`, `uv run pytest -m integration`).

## Before opening a pull request

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

CI runs the same commands. Please also:

- add or update tests with every behaviour change; a test that talks to a database or a real model must be marked `@pytest.mark.integration`;
- run the relevant eval when you change the parser or the ranking (`eval/run_parser_eval.py`, `eval/run_battery.py`), and bump the prompt or score version constants when their content changes so caches invalidate;
- keep SQL out of `ingovo_search.api` (an architecture test enforces it) and keep route handlers short;
- never commit keys: use `.env` (ignored by git) and obvious fakes such as `test-api-key` in tests.

Code, comments and docs are in English. Hungarian is fine, and expected, where it is product behaviour: queries, prompts sent to the model, test cases and user-facing copy.

## Reporting issues

Use the issue templates. For security problems, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.

By contributing you agree that your contributions are licensed under the [AGPL-3.0-only](LICENSE) licence of this repository.
