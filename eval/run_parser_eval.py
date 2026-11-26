import asyncio
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ingovo_search.lib.llm import GoogleAiClient, LlmClient
from ingovo_search.recommender.parser import parse_query
from ingovo_search.recommender.parser_schema import ParsedQuery

# Keys whose expected list items match by normalized stem containment instead
# of exact equality. The parser normalizes free text unpredictably ("kutyával"
# may come back as "kutyabarát" or "kutyával költözne"), so exact-string
# expectations rot with every prompt revision — expected stems assert the
# *aspect* survived parsing, not its exact wording.
_STEM_MATCH_KEYS = frozenset({"soft_prefs_freetext"})


def _normalize_stem(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def freetext_stem_matches(expected_item: str, actual_items: list[Any]) -> bool:
    """True when any parsed freetext contains the expected stem (accent/case-blind)."""
    stem = _normalize_stem(expected_item)
    return any(isinstance(actual, str) and stem in _normalize_stem(actual) for actual in actual_items)


def check_dict_subset(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[int, int, list[str]]:
    """Recursively checks if 'expected' is a subset of 'actual'. Returns (matches, total, missing_paths)."""
    matches = 0
    total = 0
    missing = []

    for k, expected_v in expected.items():
        if isinstance(expected_v, dict):
            # Recurse
            actual_v = actual.get(k, {})
            if not isinstance(actual_v, dict):
                actual_v = getattr(actual.get(k), "__dict__", {}) if actual.get(k) else {}
            m, t, miss = check_dict_subset(expected_v, actual_v)
            matches += m
            total += t
            missing.extend([f"{k}.{p}" for p in miss])
        elif isinstance(expected_v, list):
            # Check list elements
            actual_v = actual.get(k, [])
            for item in expected_v:
                total += 1
                if k in _STEM_MATCH_KEYS and isinstance(item, str):
                    if freetext_stem_matches(item, actual_v):
                        matches += 1
                    else:
                        missing.append(f"{k}[~{item}]")
                elif item in actual_v:
                    matches += 1
                else:
                    missing.append(f"{k}[{item}]")
        else:
            # Check primitive
            total += 1
            actual_v = actual.get(k)
            if actual_v == expected_v:
                matches += 1
            else:
                missing.append(f"{k} (expected {expected_v!r}, got {actual_v!r})")

    return matches, total, missing


def _load_gemini_api_key() -> str:
    """Read only the one secret the parser eval needs.

    The parser eval needs no database or Redis, so it reads the key directly
    instead of building the full Settings object.
    """
    load_dotenv(Path.cwd() / ".env", override=False)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print(
            "parser eval: GEMINI_API_KEY is missing or empty. "
            "Add the GitHub Actions repository secret before running Parser Eval.",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def build_llm_client() -> LlmClient:
    """Build the real parser-eval LLM client."""
    return GoogleAiClient(api_key=_load_gemini_api_key())


def _load_eval_records(file_path: str) -> list[dict[str, Any]]:
    records = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# §17.1 category semantics. Records carry an optional "category" field; sets
# without categories keep the original global-recall gate for back-compat.
DEFAULT_CATEGORY = "uncategorized"
HARD_CONSTRAINT_CATEGORY = "hard_constraint"
ADVERSARIAL_CATEGORY = "adversarial"
RECALL_THRESHOLD = 0.85

# Free-tier quota discipline (design decision 2026-07-03: stay on the free
# Gemini tier for now). The gate only needs hard_constraint + adversarial, so
# those run first and get the day's quota; quota-exhausted queries count as
# skipped (never as misses); after enough consecutive quota errors the run
# stops burning the model-fallback chain and marks the remainder skipped.
GATED_CATEGORIES = frozenset({HARD_CONSTRAINT_CATEGORY, ADVERSARIAL_CATEGORY})
MIN_GATED_COVERAGE = 0.8
CONSECUTIVE_QUOTA_ABORT = 3

_QUOTA_ERROR_MARKERS = ("RESOURCE_EXHAUSTED", "429")


def is_quota_error(error: BaseException) -> bool:
    """True when the failure is quota/rate-limit exhaustion, not a parser defect."""
    text = str(error)
    return any(marker in text for marker in _QUOTA_ERROR_MARKERS)


def order_records_gate_first(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gated categories first (stable within each group) so quota is spent on what blocks merges."""
    return sorted(
        records,
        key=lambda record: 0 if record.get("category", DEFAULT_CATEGORY) in GATED_CATEGORIES else 1,
    )


# Sections of ParsedQuery where an invented (non-expected) value constrains or
# steers results — the hallucination surface. Soft prefs are deliberately
# excluded: an extra soft tag reorders, a hallucinated hard filter excludes.
_HALLUCINATION_SECTIONS = ("hard_filters", "anchors")

# Keys the parser sets by documented convention rather than by extracting them
# from the query text — never hallucinations (verified on the 2026-07-03 run):
# - `city_id` is derived deterministically from `city` via the settlement
#   registry, so it accompanies every resolved city.
# - `transaction_type` is defaulted (buying intent => "sale") when the query
#   doesn't state eladó/kiadó — a product convention, not an invention.
_DERIVED_OR_DEFAULTED_KEYS = frozenset({"hard_filters.city_id", "hard_filters.transaction_type"})


def _is_set(value: Any) -> bool:
    """True when a parsed field carries a real value (not a schema default)."""
    return value not in (None, [], "", {})


def find_hallucinated_keys(expected: dict[str, Any], parsed: dict[str, Any]) -> list[tuple[str, Any]]:
    """`(key_path, parsed_value)` pairs the parser set in hard_filters/anchors unexpectedly.

    Used for the §17.1 adversarial gate: adversarial queries bait the parser
    into inventing constraints, so any set-but-unexpected key fails the gate.
    Reported (but not gated) for every other category. Keys in
    `_DERIVED_OR_DEFAULTED_KEYS` are exempt. Parsed values are included so
    failures are diagnosable straight from the CI log.
    """
    hallucinated: list[tuple[str, Any]] = []
    for section in _HALLUCINATION_SECTIONS:
        expected_section = expected.get(section) or {}
        parsed_section = parsed.get(section) or {}
        for key, value in parsed_section.items():
            path = f"{section}.{key}"
            if _is_set(value) and key not in expected_section and path not in _DERIVED_OR_DEFAULTED_KEYS:
                hallucinated.append((path, value))
    # poi_near is a list section: an extracted POI requirement the record does
    # not expect is an invention ("családbarát környék" must not become a
    # school-proximity request). Items compare as full dicts.
    expected_poi = expected.get("poi_near") or []
    for item in parsed.get("poi_near") or []:
        if item not in expected_poi:
            hallucinated.append(("poi_near", item))
    return hallucinated


def gate_verdict(
    global_recall: float,
    category_recalls: dict[str, float],
    adversarial_hallucination_count: int,
    *,
    threshold: float = RECALL_THRESHOLD,
    gated_coverage: float = 1.0,
    min_gated_coverage: float = MIN_GATED_COVERAGE,
) -> tuple[bool, list[str]]:
    """§17.1 step 4 gate: hard-constraint recall < threshold OR any adversarial hallucination.

    Falls back to gating on global recall when the eval set has no
    `hard_constraint` category (pre-category sets). `gated_coverage` is the
    fraction of gated-category queries actually evaluated (quota-skipped ones
    excluded); below `min_gated_coverage` the run is inconclusive and fails —
    a quota outage must never silently pass a bad prompt.
    """
    reasons: list[str] = []
    if gated_coverage < min_gated_coverage:
        reasons.append(
            f"inconclusive: only {gated_coverage:.0%} of gated-category queries evaluated "
            f"(minimum {min_gated_coverage:.0%}) — quota exhausted; rerun after the daily reset"
        )
    if HARD_CONSTRAINT_CATEGORY in category_recalls:
        gate_recall = category_recalls[HARD_CONSTRAINT_CATEGORY]
        gate_name = f"{HARD_CONSTRAINT_CATEGORY} recall"
    else:
        gate_recall = global_recall
        gate_name = "global recall"
    if gate_recall < threshold:
        reasons.append(f"{gate_name} {gate_recall:.2%} is below the {threshold:.0%} threshold")
    if adversarial_hallucination_count:
        reasons.append(f"{adversarial_hallucination_count} hallucinated key(s) on {ADVERSARIAL_CATEGORY} queries")
    return (not reasons, reasons)


async def run_eval(file_path: str) -> None:
    client = build_llm_client()
    print("Using GoogleAiClient (Real LLM)")

    total_matches = 0
    total_expected = 0
    total_evaluated = 0
    total_skipped = 0
    # per-category: [matches, expected, evaluated, hallucinated-key count, quota-skipped]
    per_category: dict[str, list[int]] = {}
    adversarial_hallucinations = 0
    consecutive_quota_errors = 0
    quota_aborted = False

    print(f"Loading eval set from: {file_path}")
    records = order_records_gate_first(_load_eval_records(file_path))

    for record in records:
        query = record["query"]
        expected = record["expected"]
        category = record.get("category", DEFAULT_CATEGORY)
        stats = per_category.setdefault(category, [0, 0, 0, 0, 0])

        if quota_aborted:
            stats[4] += 1
            total_skipped += 1
            continue

        print(f"\nQuery [{category}]: {query!r}")
        try:
            parsed: ParsedQuery = await parse_query(query, client=client)
            parsed_dict = parsed.model_dump()
            consecutive_quota_errors = 0

            m, t, missing = check_dict_subset(expected, parsed_dict)
            total_matches += m
            total_expected += t
            total_evaluated += 1
            stats[0] += m
            stats[1] += t
            stats[2] += 1

            hallucinated = find_hallucinated_keys(expected, parsed_dict)
            if hallucinated:
                stats[3] += len(hallucinated)
                if category == ADVERSARIAL_CATEGORY:
                    adversarial_hallucinations += len(hallucinated)
                print(f"  ⚠️ Hallucinated keys: {hallucinated}")

            if missing:
                print(f"  ❌ Missing/Mismatched: {missing}")
            else:
                print("  ✅ Pass")

        except Exception as e:
            if is_quota_error(e):
                stats[4] += 1
                total_skipped += 1
                consecutive_quota_errors += 1
                print("  ⏭ Skipped: quota/rate-limit exhausted")
                if consecutive_quota_errors >= CONSECUTIVE_QUOTA_ABORT:
                    quota_aborted = True
                    print(
                        f"\n⏭ Aborting after {CONSECUTIVE_QUOTA_ABORT} consecutive quota errors; "
                        "remaining queries marked skipped."
                    )
                continue
            consecutive_quota_errors = 0
            print(f"  ❌ Error parsing query: {e}")
            total_expected += len(expected)  # roughly
            total_evaluated += 1
            stats[1] += len(expected)
            stats[2] += 1

    if total_expected == 0:
        print("\nNo queries were evaluated (quota exhausted or empty eval set).")
        sys.exit(1)

    recall = total_matches / total_expected
    category_recalls = {cat: (s[0] / s[1] if s[1] else 0.0) for cat, s in per_category.items() if s[2]}

    gated_total = sum(s[2] + s[4] for cat, s in per_category.items() if cat in GATED_CATEGORIES)
    gated_evaluated = sum(s[2] for cat, s in per_category.items() if cat in GATED_CATEGORIES)
    # Uncategorized sets have no gated categories; coverage then spans the whole run.
    if gated_total:
        gated_coverage = gated_evaluated / gated_total
    else:
        gated_coverage = (
            total_evaluated / (total_evaluated + total_skipped) if (total_evaluated + total_skipped) else 0.0
        )

    print("\n=== EVAL SUMMARY ===")
    print(f"Evaluated: {total_evaluated}  Skipped (quota): {total_skipped}")
    print(f"Global recall (evaluated only): {total_matches}/{total_expected} ({recall:.2%})")
    print("\nPer-category:")
    for cat in sorted(per_category):
        m, t, q, h, sk = per_category[cat]
        cat_recall = f"{m}/{t} ({m / t:.2%})" if t else "—"
        print(f"  {cat:<18} evaluated={q:<3} skipped={sk:<3} recall={cat_recall} hallucinated_keys={h}")

    passed, reasons = gate_verdict(recall, category_recalls, adversarial_hallucinations, gated_coverage=gated_coverage)
    if passed:
        print("\n✅ PASSED: recall gate met and no adversarial hallucinations.")
    else:
        for reason in reasons:
            print(f"\n❌ FAILED: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    eval_file = os.path.join(os.path.dirname(__file__), "golden", "eval_set.starter.jsonl")
    asyncio.run(run_eval(eval_file))
