"""Offline rerank comparison: deterministic vs Gemini vs Jev variants.

For every rated query (`synthetic_eval_rating`) the rerank head is rebuilt the
way `api/search.py` builds it: `ParsedQuery` from the eval record's `expected`
block with the query text as `semantic_query`, real Cohere query embedding,
exploration jitter zeroed, `recommend` top 30, head of `RERANK_CANDIDATE_COUNT`
(12) with `listing_rerank_summary`. Every arm then reorders the same head:

- `deterministic` — the head as it is;
- `gemini` — today's `rerank_and_explain`, through a client that keeps the
  billed token counts;
- `jev:<variant>:a<alpha>` — the real `jev_rerank` per prompt variant, summary
  length and alpha (alpha is a local blend, so one Jev call serves all alphas).

The bounded-rerank guardrail runs on every reranked order, as in production.
Gemini and Jev responses are cached on disk (`eval/out/jev_rerank_cache/`,
keyed by the full request), so a re-run pays nothing and a fresh variant pays
only for itself. The rerank heads are frozen there too (`heads_<n>.json`):
`recommend` scores drift a little between calls, which reorders near-equal
candidates and would miss the cache; `--rebuild-heads` refreshes the snapshot. The adversarial pass injects steering listings and
checks the reranker's raw top 3.

The metric, gate and report logic is unit-tested in
`ingovo_search.eval.jev_rerank_offline` and its `jev_rerank_adversarial`,
`jev_rerank_gates` and `jev_rerank_report` siblings; this file is the live
wiring only.

Run from the repository root (DB env vars + venv), Gemini and Jev fees apply:
    uv run python eval/jev_rerank_offline.py
    uv run python eval/jev_rerank_offline.py --queries 3 --variants score4-hu --no-adversarial
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = REPO_ROOT / "eval" / "golden" / "eval_set.starter.jsonl"
OUT_DIR = REPO_ROOT / "eval" / "out"
CACHE_DIR = OUT_DIR / "jev_rerank_cache"


from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env", override=False)

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typesafe_sdk import Question

from ingovo_search.db.models.synthetic_eval_rating import SyntheticEvalRating
from ingovo_search.db.session import get_async_session_factory
from ingovo_search.eval.gemini_usage import UsageRecordingGeminiClient
from ingovo_search.eval.jev_rerank_adversarial import AdversarialSummary, run_adversarial
from ingovo_search.eval.jev_rerank_gates import evaluate_gates
from ingovo_search.eval.jev_rerank_offline import (
    DETERMINISTIC_ARM,
    GEMINI_ARM,
    JEV_ARM_PREFIX,
    ArmRun,
    OfflineQuery,
    RerankArm,
    Significance,
    arm_significance,
    best_jev_arm,
    run_offline,
)
from ingovo_search.eval.jev_rerank_report import render_markdown
from ingovo_search.lib.cost_tracker import cost_usd_from_tokens
from ingovo_search.lib.embeddings import CohereEmbeddingClient, EmbeddingClient
from ingovo_search.lib.env import get_settings
from ingovo_search.lib.jev import JevClient, JevResponse, JevState, TypeSafeJevClient
from ingovo_search.lib.llm import LlmRequest
from ingovo_search.lib.llm_models import TIER_L_RERANK, TIER_L_RERANK_JEV
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.recommender.ranking_guardrails import enforce_bounded_rerank
from ingovo_search.recommender.recommend import recommend
from ingovo_search.recommender.rerank import RerankDeps, RerankOptions, RerankResult, rerank_and_explain
from ingovo_search.recommender.rerank_jev import JevRerankOptions, jev_rerank
from ingovo_search.recommender.rerank_jev_prompt import (
    NOUL_HU,
    SCORE4_EN,
    SCORE4_EN_SHORT,
    SCORE4_EN_SHORT_R2,
    SCORE4_EN_SHORT_R3,
    SCORE4_HU,
    SCORE4_HU_SHORT,
    SCORE4_HU_SHORT_R2,
    SCORE4_HU_SHORT_R3,
    JevRerankVariant,
)
from ingovo_search.recommender.rerank_prompt import RerankCandidate
from ingovo_search.recommender.score import WEIGHTS
from ingovo_search.search.constants import RERANK_CANDIDATE_COUNT
from ingovo_search.search.scoring import (
    RERANK_MIN_CANDIDATES,
    listing_rerank_summary,
    should_rerank,
)

BASE_VARIANTS: dict[str, JevRerankVariant] = {
    "score4-hu": SCORE4_HU,
    "score4-en": SCORE4_EN,
    "noul-hu": NOUL_HU,
    "score4-hu-short": SCORE4_HU_SHORT,
    "score4-en-short": SCORE4_EN_SHORT,
    "score4-hu-short-r2": SCORE4_HU_SHORT_R2,
    "score4-en-short-r2": SCORE4_EN_SHORT_R2,
    "score4-hu-short-r3": SCORE4_HU_SHORT_R3,
    "score4-en-short-r3": SCORE4_EN_SHORT_R3,
}
# Whole-packet cuts (full / 200 / 120 characters) are opt-in (`--summary-chars`):
# measured 2026-09-28, the packet's title and facts take ~170 characters, so
# they cut the description away first. `DESCRIPTION_CHARS` is the meaningful cut.
SUMMARY_CHARS: tuple[int | None, ...] = (None,)
# Description-only cuts, measured on the rule-v2 variants.
DESCRIPTION_CHARS: tuple[int, ...] = (200, 120)
DESCRIPTION_CUT_BASES: tuple[str, ...] = ("score4-hu-short-r2", "score4-en-short-r2")
ALPHAS: tuple[float, ...] = (1.0, 0.7, 0.5)
# The lean summary is measured on the compact-scale variant only: both cuts
# together are the candidate for the G2 token target.
LEAN_EXTRAS: tuple[tuple[str, int | None], ...] = (
    ("score4-hu-short", None),
    ("score4-hu-short", 200),
    ("score4-hu-short-r2", None),
    ("score4-en-short-r2", None),
    ("score4-hu-short-r3", None),
    ("score4-en-short-r3", None),
)
JEV_TIMEOUT_S = 10.0  # offline: measure the real latency instead of cutting it


class _EmptyRegionHintClient:
    """LlmClient stub: cold-start region hints resolve to 'no districts known' (run_battery convention)."""

    async def complete(self, req: LlmRequest) -> str:
        return '{"districts": []}'


# --- disk caches -----------------------------------------------------------------


class DiskCache:
    """`<key>.json` files under one directory."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, payload: Mapping[str, Any]) -> None:
        (self._dir / f"{key}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CallRecord:
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    from_cache: bool


def _question_json(question: Question) -> Any:
    return question.model_dump(mode="json", exclude_none=True) if hasattr(question, "model_dump") else question


class CachingJevClient:
    """`JevClient` that answers repeated requests from disk and records each call."""

    def __init__(self, inner: JevClient, cache: DiskCache) -> None:
        self._inner = inner
        self._cache = cache
        self.records: list[CallRecord] = []

    async def system_one(
        self, state: JevState, questions: Mapping[str, Question], *, model: str, timeout_s: float
    ) -> JevResponse:
        key = _digest(
            {"model": model, "state": state, "questions": {k: _question_json(q) for k, q in questions.items()}}
        )
        cached = self._cache.get(key)
        if cached is not None:
            response = JevResponse.model_validate(cached["response"])
            self.records.append(self._record(response, cached["latency_ms"], from_cache=True))
            return response
        started = time.perf_counter()
        response = await self._inner.system_one(state, questions, model=model, timeout_s=timeout_s)
        latency_ms = (time.perf_counter() - started) * 1000
        self._cache.set(key, {"response": response.model_dump(), "latency_ms": latency_ms})
        self.records.append(self._record(response, latency_ms, from_cache=False))
        return response

    @staticmethod
    def _record(response: JevResponse, latency_ms: float, *, from_cache: bool) -> CallRecord:
        return CallRecord(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd_from_tokens(
                response.model, input_tokens=response.input_tokens, output_tokens=response.output_tokens
            ),
            from_cache=from_cache,
        )


class CachingGeminiClient:
    """`LlmClient` over `UsageRecordingGeminiClient` with the same disk cache; records billed tokens."""

    def __init__(self, inner: UsageRecordingGeminiClient, cache: DiskCache) -> None:
        self._inner = inner
        self._cache = cache
        self.records: list[CallRecord] = []
        # (real, estimated) input/output token pairs of the live calls, for the token audit.
        self.usage: list[dict[str, float]] = []

    async def complete(self, req: LlmRequest) -> str:
        schema = req.response_schema.__name__ if req.response_schema is not None else None
        key = _digest(
            {
                "model": req.model,
                "system": req.system,
                "user": req.user,
                "schema": schema,
                "t": req.temperature,
                "extra": self._inner.extra_config,
            }
        )
        cached = self._cache.get(key)
        if cached is not None:
            usage = cached["usage"]
            self.records.append(
                CallRecord(usage["input_tokens"], usage["output_tokens"], cached["latency_ms"], usage["cost_usd"], True)
            )
            self.usage.append(usage)
            return cached["text"]
        text = await self._inner.complete(req)
        record = self._inner.records[-1]
        usage = {
            "input_tokens": record.prompt_tokens,
            "output_tokens": record.output_tokens,
            "thoughts_tokens": record.thoughts_tokens,
            "cost_usd": record.cost_usd,
            "estimated_input_tokens": record.estimated_input_tokens,
            "estimated_output_tokens": record.estimated_output_tokens,
            "estimated_cost_usd": record.estimated_cost_usd,
            "latency_ms": record.latency_ms,
        }
        self._cache.set(key, {"text": text, "usage": usage, "latency_ms": record.latency_ms})
        self.records.append(
            CallRecord(record.prompt_tokens, record.output_tokens, record.latency_ms, record.cost_usd, False)
        )
        self.usage.append(usage)
        return text


# --- data -------------------------------------------------------------------------


def load_eval_records() -> dict[str, dict[str, Any]]:
    """query text -> eval record, in file order."""
    records: dict[str, dict[str, Any]] = {}
    for line in EVAL_SET_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["query"]] = record
    return records


async def load_relevance(session: AsyncSession, rater: str | None) -> dict[str, dict[str, float]]:
    """`{query_id: {property_id: rating}}`; several raters on one pair are averaged."""
    stmt = select(SyntheticEvalRating)
    if rater is not None:
        stmt = stmt.where(SyntheticEvalRating.rater_id == rater)
    rows = (await session.execute(stmt)).scalars().all()
    sums: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        sums.setdefault(row.query_id, {}).setdefault(row.property_id, []).append(float(row.rating))
    return {
        query_id: {property_id: sum(values) / len(values) for property_id, values in grades.items()}
        for query_id, grades in sums.items()
    }


def _parsed_for(record: Mapping[str, Any]) -> ParsedQuery:
    """The eval record's `expected` block, with the query text as the semantic query.

    The block carries no `semantic_query`, and without one every reranker takes
    its fast path; the parser would put the query's own words there.
    """
    expected = dict(record["expected"])
    expected.setdefault("semantic_query", record["query"])
    if not expected["semantic_query"]:
        expected["semantic_query"] = record["query"]
    return ParsedQuery.model_validate(expected)


async def build_offline_queries(
    session: AsyncSession,
    records: Sequence[Mapping[str, Any]],
    embedder: EmbeddingClient,
    *,
    top_n: int = 30,
    head: int = RERANK_CANDIDATE_COUNT,
) -> list[OfflineQuery]:
    """Each record's rerank head, built as `api/search.py` builds it."""
    queries: list[OfflineQuery] = []
    weights = {**WEIGHTS, "explore": 0.0}
    client = _EmptyRegionHintClient()
    for record in records:
        text = record["query"]
        parsed = _parsed_for(record)
        query_emb = await embedder.embed(text, "search_query")
        freetext_emb = None
        if parsed.soft_prefs_freetext:
            freetext_emb = await embedder.embed(" ".join(parsed.soft_prefs_freetext), "search_query")
        scored = await recommend(
            session,
            parsed,
            query_emb=query_emb,
            freetext_pref_emb=freetext_emb,
            client=client,  # type: ignore[arg-type]
            top_n=top_n,
            weights=weights,
        )
        if len(scored) < 2:
            # Kept in the snapshot with an empty head, so it is not rebuilt every run.
            print(f"  no rerank head (only {len(scored)} candidates): {text[:60]}", flush=True)
        head_rows = scored[:head]
        queries.append(
            OfflineQuery(
                query_id=text,
                query_text=text,
                parsed=parsed,
                candidates=tuple(
                    RerankCandidate(listing_id=prop.id, short_summary=listing_rerank_summary(prop))
                    for prop, _ in head_rows
                ),
                deterministic_scores={prop.id: breakdown.total for prop, breakdown in head_rows},
                should_rerank=should_rerank(parsed, scored),
                category=record.get("category"),
            )
        )
    return queries


def _heads_path(head: int) -> Path:
    return CACHE_DIR / f"heads_{head}.json"


def _query_to_json(query: OfflineQuery) -> dict[str, Any]:
    return {
        "query_id": query.query_id,
        "query_text": query.query_text,
        "parsed": query.parsed.model_dump(mode="json"),
        "candidates": [{"listing_id": c.listing_id, "short_summary": c.short_summary} for c in query.candidates],
        "deterministic_scores": dict(query.deterministic_scores),
        "should_rerank": query.should_rerank,
        "category": query.category,
    }


def _query_from_json(payload: Mapping[str, Any]) -> OfflineQuery:
    return OfflineQuery(
        query_id=payload["query_id"],
        query_text=payload["query_text"],
        parsed=ParsedQuery.model_validate(payload["parsed"]),
        candidates=tuple(RerankCandidate.model_validate(c) for c in payload["candidates"]),
        deterministic_scores=dict(payload["deterministic_scores"]),
        should_rerank=bool(payload["should_rerank"]),
        category=payload.get("category"),
    )


def save_heads(queries: Sequence[OfflineQuery], *, head: int) -> Path:
    path = _heads_path(head)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([_query_to_json(q) for q in queries], ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_heads(*, head: int) -> dict[str, OfflineQuery]:
    path = _heads_path(head)
    if not path.exists():
        return {}
    return {q["query_id"]: _query_from_json(q) for q in json.loads(path.read_text(encoding="utf-8"))}


async def load_or_build_heads(
    session: AsyncSession,
    records: Sequence[Mapping[str, Any]],
    embedder: EmbeddingClient,
    *,
    head: int = RERANK_CANDIDATE_COUNT,
    rebuild: bool = False,
) -> list[OfflineQuery]:
    """Frozen heads from the snapshot; records missing from it are built and added.

    Records without a usable head are kept too (empty `candidates`); callers filter.
    """
    frozen = {} if rebuild else load_heads(head=head)
    missing = [record for record in records if record["query"] not in frozen]
    built = await build_offline_queries(session, missing, embedder, head=head) if missing else []
    if built or rebuild:
        frozen.update({q.query_id: q for q in built})
        save_heads(list(frozen.values()), head=head)
        print(f"heads snapshot: {len(built)} built, {len(frozen)} total -> {_heads_path(head).relative_to(REPO_ROOT)}")
    else:
        print(f"heads snapshot: {len(frozen)} loaded from {_heads_path(head).relative_to(REPO_ROOT)}")
    return [frozen[record["query"]] for record in records if record["query"] in frozen]


# --- arms -------------------------------------------------------------------------


def _outcome(result: RerankResult) -> str:
    if result.fallback:
        return "fallback"
    if result.fast_path:
        return "fast_path"
    return "success"


def _finish(
    arm: str, query: OfflineQuery, result: RerankResult, records: Sequence[CallRecord], started: float
) -> ArmRun:
    raw = tuple(item.listing_id for item in result.items)
    guarded = tuple(enforce_bounded_rerank(list(query.candidate_ids), list(raw)))
    return ArmRun(
        arm=arm,
        query_id=query.query_id,
        ranked_ids=guarded,
        raw_ranked_ids=raw,
        latency_ms=sum(r.latency_ms for r in records) if records else (time.perf_counter() - started) * 1000,
        input_tokens=sum(r.input_tokens for r in records),
        output_tokens=sum(r.output_tokens for r in records),
        cost_usd=sum(r.cost_usd for r in records),
        outcome=_outcome(result),
        from_cache=bool(records) and all(r.from_cache for r in records),
    )


def deterministic_arm() -> RerankArm:
    async def run(query: OfflineQuery) -> ArmRun:
        return ArmRun(
            DETERMINISTIC_ARM, query.query_id, query.candidate_ids, query.candidate_ids, outcome="deterministic"
        )

    return run


def gemini_arm(client: CachingGeminiClient, *, rank_only: bool, locale: str = "hu") -> RerankArm:
    async def run(query: OfflineQuery) -> ArmRun:
        first = len(client.records)
        started = time.perf_counter()
        result = await rerank_and_explain(
            query.parsed, list(query.candidates), RerankDeps(client), RerankOptions(rank_only=rank_only, locale=locale)
        )
        return _finish(GEMINI_ARM, query, result, client.records[first:], started)

    return run


def jev_arm_name(variant: JevRerankVariant, alpha: float) -> str:
    return f"{JEV_ARM_PREFIX}{variant.prompt_version.removeprefix('rerank-jev/v1-')}:a{alpha}"


def jev_arm(client: CachingJevClient, variant: JevRerankVariant, alpha: float) -> RerankArm:
    name = jev_arm_name(variant, alpha)

    async def run(query: OfflineQuery) -> ArmRun:
        first = len(client.records)
        started = time.perf_counter()
        result = await jev_rerank(
            query.parsed,
            list(query.candidates),
            RerankDeps(client),
            JevRerankOptions(
                deterministic_scores=query.deterministic_scores, variant=variant, alpha=alpha, timeout_s=JEV_TIMEOUT_S
            ),
        )
        return _finish(name, query, result, client.records[first:], started)

    return run


def sweep_variants(
    names: Sequence[str], *, summary_chars: Sequence[int | None] = SUMMARY_CHARS
) -> list[JevRerankVariant]:
    variants: list[JevRerankVariant] = []
    for name in names:
        base = BASE_VARIANTS[name]
        variants.extend(replace(base, summary_chars=chars) for chars in summary_chars)
    for name, chars in LEAN_EXTRAS:
        if name in names:
            variants.append(replace(BASE_VARIANTS[name], summary_chars=chars, lean_summary=True))
    for name in DESCRIPTION_CUT_BASES:
        if name in names:
            variants.extend(replace(BASE_VARIANTS[name], description_chars=chars) for chars in DESCRIPTION_CHARS)
    return variants


# --- main -------------------------------------------------------------------------


def _progress(query: OfflineQuery, run: ArmRun) -> None:
    cache = "cache" if run.from_cache else f"{run.latency_ms:.0f} ms"
    print(f"    {run.arm:<40} {run.outcome:<9} {run.input_tokens:>5} tok  {cache}", flush=True)


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _json_ready(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(v) for v in value]
    return value


async def main_async(args: argparse.Namespace) -> None:
    settings = get_settings()
    if settings.typesafe_api_key is None:
        raise SystemExit("TYPESAFE_API_KEY is not set (environment or .env)")
    records_by_text = load_eval_records()
    cache = DiskCache(CACHE_DIR)
    gemini = CachingGeminiClient(UsageRecordingGeminiClient.from_api_key(settings.gemini_api_key), cache)
    jev = CachingJevClient(TypeSafeJevClient(api_key=settings.typesafe_api_key), cache)
    embedder = CohereEmbeddingClient(api_key=settings.cohere_api_key)

    async with get_async_session_factory()() as session:
        relevance = await load_relevance(session, args.rater)
        rated_records = [records_by_text[text] for text in records_by_text if text in relevance]
        skipped = [text for text in relevance if text not in records_by_text]
        if args.queries is not None:
            rated_records = rated_records[: args.queries]
        print(
            f"rated queries in the eval set: {len(rated_records)} (rated but not in the set: {len(skipped)})",
            flush=True,
        )
        heads = await load_or_build_heads(session, rated_records, embedder, head=args.head, rebuild=args.rebuild_heads)
    queries = [q for q in heads if len(q.candidates) >= args.min_candidates]
    print(f"queries with a rerank head of >= {args.min_candidates}: {len(queries)} of {len(heads)}\n", flush=True)

    variants = sweep_variants(args.variants, summary_chars=args.summary_chars)
    arms: dict[str, RerankArm] = {DETERMINISTIC_ARM: deterministic_arm()}
    if not args.no_gemini:
        arms[GEMINI_ARM] = gemini_arm(gemini, rank_only=settings.ai_search_rerank_rank_only)
    for variant in variants:
        for alpha in args.alphas:
            arms[jev_arm_name(variant, alpha)] = jev_arm(jev, variant, alpha)

    def on_query(query: OfflineQuery, run: ArmRun) -> None:
        if run.arm == DETERMINISTIC_ARM:
            print(f"[{query.query_id[:60]}] should_rerank={query.should_rerank} head={len(query.candidates)}")
        _progress(query, run)

    report = await run_offline(queries, relevance, arms, on_progress=on_query)

    adversarial: tuple[AdversarialSummary, ...] = ()
    if not args.no_adversarial:
        print("\nadversarial pass (alpha 1.0)", flush=True)
        adv_arms: dict[str, RerankArm] = {}
        if not args.no_gemini:
            adv_arms[GEMINI_ARM] = arms[GEMINI_ARM]
        for variant in variants:
            adv_arms[jev_arm_name(variant, 1.0)] = jev_arm(jev, variant, 1.0)
        adversarial = await run_adversarial(queries, adv_arms, on_progress=_progress)

    best = best_jev_arm(report.summaries)
    gate_arm = args.gate_arm or (best.arm if best is not None else None)
    if gate_arm is not None and gate_arm not in arms:
        raise SystemExit(f"--gate-arm {gate_arm!r} is not an arm of this run")
    gates = None
    significance: list[Significance] = []
    if gate_arm is not None and not args.no_gemini:
        # The adversarial pass runs at alpha 1.0 (pure Jev order, the harshest case).
        adv_for_gate = next((a for a in adversarial if a.arm == f"{gate_arm.rsplit(':a', 1)[0]}:a1.0"), None)
        adv_gemini = next((a for a in adversarial if a.arm == GEMINI_ARM), None)
        gates = evaluate_gates(
            report.summaries, jev_arm=gate_arm, adversarial=adv_for_gate, reference_adversarial=adv_gemini
        )
        significance = [
            arm_significance(report, gate_arm, GEMINI_ARM),
            arm_significance(report, gate_arm, GEMINI_ARM, metric="top1_relevance"),
            arm_significance(report, gate_arm, DETERMINISTIC_ARM),
            arm_significance(report, GEMINI_ARM, DETERMINISTIC_ARM),
        ]

    billed_gemini = sum(r.cost_usd for r in gemini.records if not r.from_cache)
    billed_jev = sum(r.cost_usd for r in jev.records if not r.from_cache)
    live_gemini = sum(1 for r in gemini.records if not r.from_cache)
    live_jev = sum(1 for r in jev.records if not r.from_cache)
    meta: dict[str, object] = {
        "Queries": f"{len(queries)} rated queries, {args.head} candidates per head",
        "Rater": args.rater or "all raters (averaged)",
        "Modellek": f"Gemini `{TIER_L_RERANK}` (rank_only={settings.ai_search_rerank_rank_only}), Jev `{TIER_L_RERANK_JEV}`",
        "Jev variants": ", ".join(v.prompt_version.removeprefix("rerank-jev/v1-") for v in variants),
        "Alfa": ", ".join(str(a) for a in args.alphas),
        "Live calls in this run": f"Gemini {live_gemini} (≈ {billed_gemini:.4f} $), Jev {live_jev} (≈ {billed_jev:.4f} $)",
        "Cache": f"{CACHE_DIR.relative_to(REPO_ROOT)} (answers + snapshot of the heads: heads_{args.head}.json)",
        "Kapu-kar": f"{gate_arm} ({'--gate-arm' if args.gate_arm else 'a legjobb nDCG@10'})",
        "G1 (d) criterion": "amended (2026-09-28): instruction-type and all adversarial hits ≤ Gemini's",
    }
    label = args.label or date.today().isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / f"jev_rerank_offline_{label}.md"
    json_path = OUT_DIR / f"jev_rerank_offline_{label}.json"
    md_path.write_text(
        render_markdown(
            report, queries=queries, adversarial=adversarial, gates=gates, significance=significance, meta=meta
        ),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(
            {
                "meta": meta,
                "summaries": _json_ready(list(report.summaries)),
                "adversarial": _json_ready(list(adversarial)),
                "gates": _json_ready(gates) if gates is not None else None,
                "best_jev_arm": best.arm if best is not None else None,
                "gate_arm": gate_arm,
                "significance": _json_ready(significance),
                "gemini_usage": gemini.usage,
                "queries": [
                    {
                        "query_id": q.query_id,
                        "category": q.category,
                        "should_rerank": q.should_rerank,
                        "candidate_ids": list(q.candidate_ids),
                        "rated_candidates": sum(1 for i in q.candidate_ids if i in relevance.get(q.query_id, {})),
                    }
                    for q in queries
                ],
                "metrics": _json_ready(list(report.metrics)),
                "runs": _json_ready(list(report.runs)),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: {md_path.relative_to(REPO_ROOT)} and {json_path.relative_to(REPO_ROOT)}")
    print(f"live calls: Gemini {live_gemini} (~${billed_gemini:.4f}), Jev {live_jev} (~${billed_jev:.4f})")
    if best is not None:
        print(f"best Jev arm by nDCG@10: {best.arm} ({best.mean_ndcg_at_k:.4f}, {best.input_tokens_p50} tok p50)")
    if gates is not None:
        print(f"gates for {gate_arm}:")
        for check in gates.checks:
            print(f"  {'PASS' if check.passed else 'FAIL'} {check.gate}: {check.measured:.4f} vs {check.threshold:.4f}")
    for item in significance:
        print(f"  {item.label} [{item.metric}]: {item.mean:+.4f} [{item.low:+.4f}, {item.high:+.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=int, default=None, help="only the first N rated queries")
    parser.add_argument("--rater", default=None, help="rater_id to use; default: every rater, averaged")
    parser.add_argument(
        "--variants",
        default=",".join(BASE_VARIANTS),
        help=f"comma-separated base variants (default: all of {', '.join(BASE_VARIANTS)})",
    )
    parser.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS), help="comma-separated alphas")
    parser.add_argument(
        "--summary-chars",
        default="",
        help="extra whole-packet cuts to add, comma-separated (e.g. 200,120); the full packet always runs",
    )
    parser.add_argument("--head", type=int, default=RERANK_CANDIDATE_COUNT, help="rerank head size")
    parser.add_argument(
        "--min-candidates",
        type=int,
        default=RERANK_MIN_CANDIDATES,
        help="skip heads smaller than this (production never reranks fewer)",
    )
    parser.add_argument("--rebuild-heads", action="store_true", help="rebuild the frozen rerank heads from the DB")
    parser.add_argument("--no-gemini", action="store_true", help="skip the Gemini arm (no Gemini fees)")
    parser.add_argument("--no-adversarial", action="store_true", help="skip the adversarial pass")
    parser.add_argument("--label", default=None, help="output file label (default: today's date)")
    parser.add_argument("--gate-arm", default=None, help="arm to evaluate the gates for (default: best by nDCG@10)")
    args = parser.parse_args()
    args.variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in args.variants if name not in BASE_VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}; known: {list(BASE_VARIANTS)}")
    args.alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    args.summary_chars = [None, *(int(c) for c in args.summary_chars.split(",") if c.strip())]
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
