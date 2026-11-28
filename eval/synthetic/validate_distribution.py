"""Distribution validator.

Reads `synthetic_listings.json` and verifies the realised distribution lands
inside the documented tolerances:

- 1 500-2 000 total rows.
- Tier split within ±3pp of `TIER_PROBABILITIES`.
- Null-field fraction within ±2pp of `NULL_FIELD_FRACTION`.
- Bad-case fraction within ±2pp of `BAD_CASE_FRACTION`.
- City coverage: every city seen in `extract_osm`'s target tables is
  represented at least once (so the geographic spread holds even after
  resampling).

Returns a structured `ValidationReport` whose `ok` flag drives the CLI exit
code. Used by the acceptance gate and by `eval/run_parser_eval.py`'s
sister scripts when they need to spot-check dataset health.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.synthetic.assemble import (
    BAD_CASE_FRACTION,
    DEFAULT_OUTPUT_PATH,
    NULL_FIELD_FRACTION,
    TARGET_COUNT_LOWER,
    TARGET_COUNT_UPPER,
    TIER_PROBABILITIES,
)

TIER_TOLERANCE = 0.03
FRACTION_TOLERANCE = 0.02
REQUIRED_CITIES: tuple[str, ...] = (
    "Budapest",
    "Pécs",
    "Szeged",
    "Debrecen",
    "Miskolc",
    "Győr",
    "Sopron",
)


@dataclass(slots=True)
class ValidationReport:
    ok: bool
    total: int
    tier_counts: dict[str, int] = field(default_factory=dict[str, int])
    null_count: int = 0
    bad_case_count: int = 0
    missing_cities: list[str] = field(default_factory=list[str])
    issues: list[str] = field(default_factory=list[str])

    def summary(self) -> str:
        return (
            f"total={self.total} "
            f"tier={dict(self.tier_counts)} "
            f"null={self.null_count} "
            f"bad={self.bad_case_count} "
            f"missing_cities={self.missing_cities} "
            f"ok={self.ok}"
        )


def validate(listings: list[dict[str, Any]]) -> ValidationReport:
    """Apply all distribution checks. Side-effect free."""
    total = len(listings)
    issues: list[str] = []

    if not TARGET_COUNT_LOWER <= total <= TARGET_COUNT_UPPER:
        issues.append(f"total={total} outside target window [{TARGET_COUNT_LOWER}, {TARGET_COUNT_UPPER}]")

    tier_counts = Counter(str(entry.get("tier", "?")) for entry in listings)
    for tier, expected in TIER_PROBABILITIES.items():
        observed = tier_counts.get(tier, 0) / total if total else 0
        if abs(observed - expected) > TIER_TOLERANCE:
            issues.append(
                f"tier '{tier}' fraction {observed:.3f} differs from target {expected:.3f} by > {TIER_TOLERANCE:.2f}"
            )

    null_count = sum(1 for entry in listings if entry.get("nulled_field"))
    null_fraction = null_count / total if total else 0
    if abs(null_fraction - NULL_FIELD_FRACTION) > FRACTION_TOLERANCE:
        issues.append(
            f"null fraction {null_fraction:.3f} differs from "
            f"target {NULL_FIELD_FRACTION:.3f} by > {FRACTION_TOLERANCE:.2f}"
        )

    bad_count = sum(1 for entry in listings if entry.get("bad_case"))
    bad_fraction = bad_count / total if total else 0
    if abs(bad_fraction - BAD_CASE_FRACTION) > FRACTION_TOLERANCE:
        issues.append(
            f"bad-case fraction {bad_fraction:.3f} differs from "
            f"target {BAD_CASE_FRACTION:.3f} by > {FRACTION_TOLERANCE:.2f}"
        )

    cities = {str(entry.get("city", "")) for entry in listings}
    missing = [c for c in REQUIRED_CITIES if c not in cities]
    if missing:
        issues.append(f"missing required cities: {missing}")

    return ValidationReport(
        ok=not issues,
        total=total,
        tier_counts=dict(tier_counts),
        null_count=null_count,
        bad_case_count=bad_count,
        missing_cities=missing,
        issues=issues,
    )


def load_listings(path: Path = DEFAULT_OUTPUT_PATH) -> list[dict[str, Any]]:
    raw_text = path.read_text(encoding="utf-8")
    raw = json.loads(raw_text)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must hold a JSON array")
    listings: list[dict[str, Any]] = [entry for entry in raw if isinstance(entry, dict)]
    return listings


def main(argv: list[str] | None = None) -> int:
    path = Path(argv[0]) if argv else DEFAULT_OUTPUT_PATH
    if not path.exists():
        print(f"{path} not found — run `python -m eval.synthetic.assemble` first", file=sys.stderr)
        return 1

    report = validate(load_listings(path))
    print(report.summary())
    for issue in report.issues:
        print(f"  - {issue}", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover — CLI shim
    sys.exit(main(sys.argv[1:]))
