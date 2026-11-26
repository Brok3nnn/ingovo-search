"""§17.2.3 HU description generator.

Synthesises Hungarian property descriptions across three deliberate quality
tiers so the §17.2.4 assembler can produce a dataset that mirrors the noise
profile of real-world listings:

- **great** (33%) — rich, neighbourhood detail, specific features. Goes
  through a Tier-L LLM call (`LlmClient` injected; production wires
  `GoogleAiClient`).
- **templated** (34%) — pure-Python template + variable substitution.
  Generic phrasing that mimics the formulaic copy most agency feeds ship.
- **skeletal** (33%) — under 50 words, minimal info. Built from the
  structured fields alone.

Additional noise:

- 15% of all listings have one random structured field deliberately set to
  `None` (`null_one_field`).
- 10% are "obviously bad" cases (`inject_bad_case`): mislabeled type,
  missing photos, contradictions in the description vs. the structured
  fields.

Mass generation is **gated** on a `[Human review required]` task — founder
samples 30 outputs and signs off before the §17.2.4 assembler uses the LLM
path. Until then, the assembler can call `generate_description(tier=...)`
with a `MockLlmClient` to keep CI deterministic.

Pure Python; HU prompt strings live alongside the §4 / §7 / §11 [PLACEHOLDER]
prompts and are tagged with `DESCRIBE_PROMPT_VERSION` so the §13 cache layer
can invalidate cleanly when founder rewrites them.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ingovo_search.lib.llm import LlmClient, LlmRequest
from ingovo_search.lib.llm_models import TIER_L_RERANK

DESCRIBE_PROMPT_VERSION = "describe/v0-skeleton"

DescriptionTier = Literal["great", "templated", "skeletal"]

# Tier distribution per §17.2.3. Numbers must sum to ~1.0; the ±1pp from
# rounding lands on `templated` (the largest bucket).
TIER_PROBABILITIES: dict[DescriptionTier, float] = {
    "great": 0.33,
    "templated": 0.34,
    "skeletal": 0.33,
}

NULL_FIELD_FRACTION = 0.15
BAD_CASE_FRACTION = 0.10

# Structured fields the synthetic generator can wipe to inject `null_one_field`.
NULLABLE_STRUCTURED_FIELDS: tuple[str, ...] = (
    "rooms",
    "area_m2",
    "floor",
    "year_built",
    "heating",
    "renovation_state",
    "orientation",
)


# Templated-tier HU phrases. Picked at random and concatenated; intentionally
# bland to match real boilerplate.
_TEMPLATED_OPENERS: tuple[str, ...] = (
    "Eladásra kínáljuk",
    "Figyelmébe ajánljuk",
    "Csendes környezetben",
    "Kiváló elrendezésű",
    "Tehermentes",
    "Költözhető",
)
_TEMPLATED_CONNECTORS: tuple[str, ...] = (
    "lakás",
    "ingatlan",
    "otthon",
)
_TEMPLATED_AMENITIES: tuple[str, ...] = (
    "felújított",
    "jó állapotú",
    "azonnal beköltözhető",
    "déli fekvésű",
    "panorámás",
    "tágas",
    "világos",
)
_TEMPLATED_CLOSERS: tuple[str, ...] = (
    "A környéken minden infrastruktúra megtalálható.",
    "Tömegközlekedési lehetőségek a közelben.",
    "Iskolák, óvodák, üzletek a környéken.",
    "Csendes utca, parkolási lehetőség.",
    "Befektetésnek és önálló lakhatásra is kiváló.",
)


@dataclass(frozen=True, slots=True)
class ListingSeed:
    """Minimum fields the description generator reads.

    Mirrors the subset of `Property` columns the §17.2.4 assembler emits
    before enrichment runs. Free-form: callers can pass any dict-like object.
    """

    city: str
    district: str | None
    property_type: str  # "panel" | "tegla" | "csaladi_haz" | ...
    rooms: int | None
    area_m2: float | None
    price_huf: int


class _DescribeLlmClient(Protocol):
    async def complete(self, req: LlmRequest) -> str: ...


def _format_int(value: int | float | None) -> str:
    if value is None:
        return "?"
    return f"{int(value):,}".replace(",", " ")


def render_great_user_prompt(seed: ListingSeed) -> str:
    """Tier-L user prompt — `[PLACEHOLDER]` until founder rewrites the HU copy."""
    return (
        "[PLACEHOLDER: HU prompt that asks for a rich, neighbourhood-aware "
        "description of the listing below. Tone: warm, informative, no "
        "marketing fluff. 120-180 words. Mention 2-3 concrete amenities + "
        "1 specific neighbourhood feature.]\n\n"
        f"city={seed.city}, district={seed.district}, property_type="
        f"{seed.property_type}, rooms={seed.rooms}, area_m2={seed.area_m2}, "
        f"price_huf={seed.price_huf}"
    )


GREAT_SYSTEM_PROMPT = (
    "[PLACEHOLDER: HU system prompt for the synthetic 'great-tier' description "
    "generator. Founder finalises the wording during the §17.2.3 30-sample "
    "review.]"
)


async def generate_great_description(seed: ListingSeed, *, client: LlmClient) -> str:
    """Tier-L LLM call. Returns whatever the model emits verbatim."""
    req = LlmRequest(
        model=TIER_L_RERANK,
        system=GREAT_SYSTEM_PROMPT,
        user=render_great_user_prompt(seed),
    )
    return await client.complete(req)


def generate_templated_description(seed: ListingSeed, *, rng: random.Random | None = None) -> str:
    """Mid-tier — formulaic HU sentences stitched from a tiny phrase bank."""
    rng = rng if rng is not None else random.Random()
    opener = rng.choice(_TEMPLATED_OPENERS)
    connector = rng.choice(_TEMPLATED_CONNECTORS)
    amenity = rng.choice(_TEMPLATED_AMENITIES)
    closer = rng.choice(_TEMPLATED_CLOSERS)

    district_clause = f"Budapest {seed.district}. kerület" if seed.district else seed.city
    rooms_clause = f"{seed.rooms} szobás " if seed.rooms is not None else ""
    area_clause = f"{seed.area_m2:.0f} négyzetméteres " if seed.area_m2 is not None else ""

    return f"{opener} ezt a {amenity} {rooms_clause}{area_clause}{connector}t {district_clause} területén. {closer}"


def generate_skeletal_description(seed: ListingSeed) -> str:
    """Sub-50-word description — the kind of feed that triggers the §6 quality gate."""
    parts: list[str] = []
    if seed.rooms is not None:
        parts.append(f"{seed.rooms} szoba")
    if seed.area_m2 is not None:
        parts.append(f"{seed.area_m2:.0f} m²")
    parts.append(seed.city)
    if seed.district is not None:
        parts.append(f"{seed.district}. kerület")
    parts.append(f"{_format_int(seed.price_huf)} Ft")
    return ", ".join(parts) + "."


async def generate_description(
    seed: ListingSeed,
    *,
    tier: DescriptionTier,
    client: LlmClient,
    rng: random.Random | None = None,
) -> str:
    """Dispatch by tier — caller decides upstream what fraction goes where."""
    if tier == "great":
        return await generate_great_description(seed, client=client)
    if tier == "templated":
        return generate_templated_description(seed, rng=rng)
    return generate_skeletal_description(seed)


def choose_tier(*, rng: random.Random) -> DescriptionTier:
    """Sample a tier per `TIER_PROBABILITIES`. Deterministic when `rng` is seeded."""
    roll = rng.random()
    cumulative = 0.0
    for tier, probability in TIER_PROBABILITIES.items():
        cumulative += probability
        if roll < cumulative:
            return tier
    return "templated"  # rounding fallback


def null_one_field(seed: ListingSeed, *, rng: random.Random) -> ListingSeed:
    """Wipe one random structured field on the seed.

    Operates on a copy — the caller's seed stays untouched. Only fields the
    `ListingSeed` schema actually carries are eligible (subset of
    `NULLABLE_STRUCTURED_FIELDS`).
    """
    eligible = tuple(f for f in NULLABLE_STRUCTURED_FIELDS if hasattr(seed, f))
    if not eligible:
        return seed
    field_name = rng.choice(eligible)
    return _replace_seed(seed, **{field_name: None})


def inject_bad_case(seed: ListingSeed, description: str, *, rng: random.Random) -> tuple[ListingSeed, str]:
    """Apply one of three bad-case mutations and return the updated `(seed, description)`."""
    choice = rng.choice(("mislabeled_type", "contradiction", "missing_field"))

    if choice == "mislabeled_type":
        wrong = "csaladi_haz" if seed.property_type != "csaladi_haz" else "panel"
        return _replace_seed(seed, property_type=wrong), description

    if choice == "contradiction":
        # Description claims 4 rooms while the structured field says something else.
        return seed, description + " A lakásban 4 nagy szoba található."

    return _replace_seed(seed, area_m2=None), description


def _replace_seed(seed: ListingSeed, **changes: Any) -> ListingSeed:
    """`dataclasses.replace` shim — `ListingSeed` is frozen + slotted."""
    current = {
        "city": seed.city,
        "district": seed.district,
        "property_type": seed.property_type,
        "rooms": seed.rooms,
        "area_m2": seed.area_m2,
        "price_huf": seed.price_huf,
    }
    current.update(changes)
    return ListingSeed(**current)


@dataclass(frozen=True, slots=True)
class DescriptionRecord:
    """One generated listing's (seed, description, tier, noise-flag) tuple."""

    seed: ListingSeed
    description: str
    tier: DescriptionTier
    nulled_field: str | None = None
    bad_case: bool = False


async def generate_dataset(
    seeds: Sequence[ListingSeed],
    *,
    client: LlmClient,
    rng: random.Random,
) -> list[DescriptionRecord]:
    """Generate one description per seed, applying tier + null + bad-case noise."""
    records: list[DescriptionRecord] = []
    n = len(seeds)
    bad_case_count = round(n * BAD_CASE_FRACTION)
    null_count = round(n * NULL_FIELD_FRACTION)
    bad_indices = set(rng.sample(range(n), k=bad_case_count)) if bad_case_count else set()
    null_indices = set(rng.sample(range(n), k=null_count)) if null_count else set()

    for index, seed in enumerate(seeds):
        applied_null: str | None = None
        if index in null_indices:
            new_seed = null_one_field(seed, rng=rng)
            # Identify which field was nulled by comparing to the original.
            for candidate in NULLABLE_STRUCTURED_FIELDS:
                if hasattr(seed, candidate) and getattr(seed, candidate) != getattr(
                    new_seed, candidate, getattr(seed, candidate)
                ):
                    applied_null = candidate
                    break
            seed = new_seed

        tier = choose_tier(rng=rng)
        description = await generate_description(seed, tier=tier, client=client, rng=rng)

        bad = False
        if index in bad_indices:
            seed, description = inject_bad_case(seed, description, rng=rng)
            bad = True

        records.append(
            DescriptionRecord(
                seed=seed,
                description=description,
                tier=tier,
                nulled_field=applied_null,
                bad_case=bad,
            )
        )

    return records
