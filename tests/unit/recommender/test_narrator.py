"""Unit tests for the AI narrator."""

from __future__ import annotations

import hashlib
import json

import pytest
from structlog.testing import capture_logs

from ingovo_search.lib.cache import MemoryCache
from ingovo_search.lib.llm import LlmClient, LlmRequest
from ingovo_search.recommender.narrator import (
    NARRATOR_CACHE_TTL_SECONDS,
    Narrator,
    NarratorSettings,
    attach_vs_next,
    generate_narrative,
    narrator_cache_key,
)
from ingovo_search.recommender.narrator_fallback import fallback_narrative
from ingovo_search.recommender.narrator_prompt import (
    NARRATOR_PROMPT_VERSION,
    NARRATOR_SYSTEM_PROMPT,
    _derive_mood,  # pyright: ignore[reportPrivateUsage]
    narrator_system_prompt,
    render_user_prompt,
)
from ingovo_search.recommender.narrator_schema import AINarrative, RationaleChip, VsNext
from ingovo_search.recommender.parser_schema import GeoHints, HardFilters, ParsedQuery
from ingovo_search.recommender.score import ScoreBreakdown

# --- mini mock ---------------------------------------------------------------


class _QueueLlmClient(LlmClient):
    """Returns successive responses from `queue` per `complete()` call.

    Simpler than `MockLlmClient` for the narrator's dynamic prompt, where the
    fixture-key approach would force callers to mirror the user-prompt builder.
    """

    def __init__(self, queue: list[str]) -> None:
        self._queue = list(queue)
        self.calls: list[LlmRequest] = []

    async def complete(self, req: LlmRequest) -> str:
        self.calls.append(req)
        if not self._queue:
            raise AssertionError("LLM queue exhausted")
        return self._queue.pop(0)


# --- helpers -----------------------------------------------------------------


class _FakeListing:
    """Minimal duck-typed stand-in for `Property`."""

    def __init__(self, **fields: object) -> None:
        for name, value in fields.items():
            setattr(self, name, value)

    # The stand-in deliberately carries whatever fields a test needs, so
    # state that contract instead of declaring every `Property` column.
    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)


def _listing() -> _FakeListing:
    return _FakeListing(
        id="prop-001",
        title="Panorámás családi ház",
        property_type="Ház",
        transaction_type="sale",
        city="Balatonfüred",
        district=None,
        area_m2=180.0,
        rooms="4",
        price_huf=148_000_000,
        currency="HUF",
        year_built=2023,
        renovation_state="új építésű",
        description="Csendes zsákutca végén, panorámás kilátás a tóra.",
    )


def _query() -> ParsedQuery:
    return ParsedQuery(
        hard_filters=HardFilters(
            transaction_type="sale",
            property_type_in=["Ház"],
            price_huf_max=150_000_000,
        ),
        geo_hints=GeoHints(county="Veszprém", balaton_shore="Északi"),
        soft_preferences=["new_build"],
        soft_prefs_freetext=["csendes"],
        semantic_query="csendes új építésű ház Balaton északi part",
    )


def _breakdown() -> ScoreBreakdown:
    return ScoreBreakdown(
        content=0.18,
        factual=0.12,
        softpref_core=0.08,
        softpref_free=0.05,
        anchor=0.07,
        quality=0.0,
        geo=0.15,
        pooled=0.03,
        recency=0.05,
        explore=0.02,
        total=0.75,
    )


_VALID_NARRATIVE_JSON = json.dumps(
    {
        "headline": "Csendes zsákutcát kértél, ez pont olyan.",
        "body": "Balatonfüreden, panorámás családi ház 180 m²-en. Új építésű 2023-ban.",
        "compromise": "Az ár 148 M, ami a 150 M-es budget alatt van.",
        "rationale_chips": [
            {"group": "location", "status": "match", "label": "Lokáció"},
            {"group": "size", "status": "match", "label": "Méret"},
            {"group": "price", "status": "match", "label": "Ár"},
            {"group": "renovation", "status": "match", "label": "Felújítás"},
        ],
    },
    ensure_ascii=False,
)


# --- generate_narrative ------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_narrative_happy_path() -> None:
    client = _QueueLlmClient([_VALID_NARRATIVE_JSON])
    narrative = await generate_narrative(_listing(), _query(), _breakdown(), Narrator(client))
    assert narrative.headline.startswith("Csendes")
    assert "180 m²" in narrative.body
    assert len(narrative.rationale_chips) == 4
    assert narrative.vs_next is None
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_generate_narrative_strips_markdown_fences() -> None:
    """Some models wrap JSON in ```json fences; the coercer must strip them."""
    fenced = "```json\n" + _VALID_NARRATIVE_JSON + "\n```"
    client = _QueueLlmClient([fenced])
    narrative = await generate_narrative(_listing(), _query(), _breakdown(), Narrator(client))
    assert narrative.headline.startswith("Csendes")


@pytest.mark.asyncio
async def test_generate_narrative_strips_hallucinated_vs_next() -> None:
    """The backend owns `vs_next`; any LLM-emitted value must be dropped."""
    raw = json.loads(_VALID_NARRATIVE_JSON)
    raw["vs_next"] = {"compared_to": "#2", "summary": "BOGUS"}
    client = _QueueLlmClient([json.dumps(raw, ensure_ascii=False)])
    narrative = await generate_narrative(_listing(), _query(), _breakdown(), Narrator(client))
    assert narrative.vs_next is None


@pytest.mark.asyncio
async def test_generate_narrative_retries_on_invalid_json_then_succeeds() -> None:
    """A first-attempt schema miss is retried; second-attempt success returns cleanly."""
    client = _QueueLlmClient(["not json at all", _VALID_NARRATIVE_JSON])
    narrative = await generate_narrative(_listing(), _query(), _breakdown(), Narrator(client))
    assert narrative.headline.startswith("Csendes")
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_generate_narrative_returns_fallback_after_two_failures() -> None:
    """Two consecutive invalid responses → deterministic heuristic narrative,
    never an exception. The hero card always has something to render."""
    client = _QueueLlmClient(["not json", "still not"])
    narrative = await generate_narrative(_listing(), _query(), _breakdown(), Narrator(client))
    # Fallback shape: 4 chips, non-empty headline + body, no vs_next.
    assert len(narrative.rationale_chips) == 4
    assert narrative.headline
    assert narrative.body
    assert narrative.vs_next is None
    # The fallback for the test listing (Veszprém + Balaton-north match):
    # location chip should be `match` because city/district align with the query.
    location_chip = next(c for c in narrative.rationale_chips if c.group == "location")
    assert location_chip.status == "match"


@pytest.mark.asyncio
async def test_generate_narrative_fallback_is_not_cached() -> None:
    """A transient LLM outage must NOT poison the 30-day narrator cache."""
    cache = MemoryCache()
    client = _QueueLlmClient(["bad", "still bad"])
    await generate_narrative(_listing(), _query(), _breakdown(), Narrator(client, cache=cache))
    # Next call with a working LLM should not hit a cached fallback — should
    # call the LLM again and use the real response.
    client2 = _QueueLlmClient([_VALID_NARRATIVE_JSON])
    narrative2 = await generate_narrative(_listing(), _query(), _breakdown(), Narrator(client2, cache=cache))
    assert narrative2.headline.startswith("Csendes")  # real LLM response
    assert len(client2.calls) == 1


@pytest.mark.asyncio
async def test_generate_narrative_uses_cache_on_hit() -> None:
    """Cache hit returns immediately without LLM contact."""
    cache = MemoryCache()
    cached = AINarrative(
        headline="cached",
        body="from-cache",
        rationale_chips=[
            RationaleChip(group="location", status="match", label="Lokáció"),
            RationaleChip(group="size", status="match", label="Méret"),
            RationaleChip(group="price", status="match", label="Ár"),
            RationaleChip(group="renovation", status="match", label="Felújítás"),
        ],
    )
    pq_hash = hashlib.sha256(_query().model_dump_json().encode("utf-8")).hexdigest()
    key = narrator_cache_key(
        "prop-001", pq_hash, _breakdown(), NarratorSettings("gemini-2.5-flash-lite", NARRATOR_PROMPT_VERSION)
    )
    await cache.set(key, cached.model_dump_json().encode("utf-8"))

    client = _QueueLlmClient([])  # any LLM call would crash on empty queue
    narrative = await generate_narrative(_listing(), _query(), _breakdown(), Narrator(client, cache=cache))
    assert narrative.headline == "cached"
    assert client.calls == []


@pytest.mark.asyncio
async def test_generate_narrative_writes_through_cache_on_miss() -> None:
    """Cache miss → LLM call → stored under the same key for next time."""
    cache = MemoryCache()
    client = _QueueLlmClient([_VALID_NARRATIVE_JSON])
    await generate_narrative(_listing(), _query(), _breakdown(), Narrator(client, cache=cache))

    # Second call should not touch the LLM
    client2 = _QueueLlmClient([])
    narrative2 = await generate_narrative(_listing(), _query(), _breakdown(), Narrator(client2, cache=cache))
    assert narrative2.headline.startswith("Csendes")
    assert client2.calls == []


class _WriteFailingCache(MemoryCache):
    """Reads like an empty cache; every write is recorded, then raises like a dropped Redis link."""

    def __init__(self) -> None:
        super().__init__()
        self.attempted: list[tuple[str, bytes, int | None]] = []

    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        self.attempted.append((key, value, ttl_seconds))
        raise ConnectionError("cache down")


@pytest.mark.asyncio
async def test_generate_narrative_keeps_the_narrative_when_the_cache_write_fails() -> None:
    """A failed write-through is logged; it neither costs a second LLM call nor the narrative."""
    client = _QueueLlmClient([_VALID_NARRATIVE_JSON])
    cache = _WriteFailingCache()
    with capture_logs() as logs:
        narrative = await generate_narrative(_listing(), _query(), _breakdown(), Narrator(client, cache=cache))

    assert narrative.headline.startswith("Csendes")
    assert len(client.calls) == 1
    [(key, value, ttl_seconds)] = cache.attempted
    assert key.startswith("narrator:")
    assert AINarrative.model_validate_json(value) == narrative
    assert ttl_seconds == NARRATOR_CACHE_TTL_SECONDS
    failures = [entry for entry in logs if entry.get("event") == "narrator cache write failed"]
    assert len(failures) == 1
    assert failures[0]["log_level"] == "warning"
    assert failures[0]["listing_id"] == "prop-001"
    assert failures[0]["error_type"] == "ConnectionError"
    assert failures[0]["error"] == "cache down"


@pytest.mark.asyncio
async def test_generate_narrative_logs_each_failed_attempt_before_the_fallback() -> None:
    """Both failed attempts and the heuristic fallback leave a warning each, in order."""
    client = _QueueLlmClient(["not json", "```\n" + _VALID_NARRATIVE_JSON[:-1] + "\n```"])
    with capture_logs() as logs:
        await generate_narrative(
            _listing(), _query(), _breakdown(), Narrator(client, settings=NarratorSettings(model="m"))
        )

    events = [(entry["event"], entry.get("attempt")) for entry in logs]
    assert events == [
        ("narrator attempt failed", 1),
        ("narrator attempt failed", 2),
        ("narrator falling back to heuristic narrative", None),
    ]
    assert {entry["model"] for entry in logs} == {"m"}
    assert "FIGYELEM: az előző válasz sémahibás volt." in client.calls[1].user


@pytest.mark.asyncio
async def test_generate_narrative_strips_an_untagged_markdown_fence() -> None:
    """A bare ``` fence (no `json` tag) is stripped the same way."""
    client = _QueueLlmClient(["```\n" + _VALID_NARRATIVE_JSON + "\n```"])
    narrative = await generate_narrative(_listing(), _query(), _breakdown(), Narrator(client))
    assert narrative.headline.startswith("Csendes")


# --- narrator_cache_key ------------------------------------------------------


def test_narrator_cache_key_is_stable_across_breakdown_noise() -> None:
    """Sub-permille breakdown diffs must collide on the cache (rounding to 3dp)."""
    bd1 = _breakdown()
    bd2 = ScoreBreakdown(**{**bd1.model_dump(), "content": 0.18001})  # tiny diff
    k1 = narrator_cache_key("p", "q", bd1, NarratorSettings("m", "v"))
    k2 = narrator_cache_key("p", "q", bd2, NarratorSettings("m", "v"))
    assert k1 == k2


def test_narrator_cache_key_changes_with_prompt_version() -> None:
    bd = _breakdown()
    k1 = narrator_cache_key("p", "q", bd, NarratorSettings("m", "narrator/v1"))
    k2 = narrator_cache_key("p", "q", bd, NarratorSettings("m", "narrator/v2"))
    assert k1 != k2


# --- output locale ----------------------------------------------------------


def test_narrator_cache_key_separates_locales() -> None:
    """A German narrative must not be served from the Hungarian cache entry."""
    bd = _breakdown()
    assert narrator_cache_key("p", "q", bd, NarratorSettings("m", "v", "hu")) != narrator_cache_key(
        "p", "q", bd, NarratorSettings("m", "v", "de")
    )


def test_narrator_system_prompt_pins_the_output_language() -> None:
    directive = narrator_system_prompt("de")
    assert NARRATOR_SYSTEM_PROMPT in directive
    # Leading and trailing, so the Hungarian prompt body cannot outweigh it.
    assert directive.startswith("KIMENETI NYELV:")
    assert directive.count("NÉMET") == 2


@pytest.mark.parametrize("locale", ["hu", "klingon", ""])
def test_narrator_system_prompt_falls_back_to_hungarian(locale: str) -> None:
    """Hungarian is the source language, so an unknown locale must not add a directive."""
    assert narrator_system_prompt(locale) == NARRATOR_SYSTEM_PROMPT


# --- attach_vs_next ---------------------------------------------------------


# --- fallback_narrative ------------------------------------------------------


def test_fallback_narrative_includes_listing_facts() -> None:
    """Fallback must expose price + location + type — these are the facts the
    hero card needs to be useful even without an LLM-generated story."""
    narrative = fallback_narrative(_listing(), _query())
    assert "Balatonfüred" in narrative.body
    assert "148" in narrative.body  # 148 M Ft
    assert len(narrative.rationale_chips) == 4


def test_fallback_narrative_marks_over_budget_price_as_miss() -> None:
    """A listing 20% over the price ceiling lands as a `miss` chip."""
    listing = _FakeListing(
        id="p",
        title="Drága ház",
        property_type="Ház",
        transaction_type="sale",
        city="Budapest",
        district=None,
        area_m2=120.0,
        rooms="4",
        price_huf=200_000_000,  # 50/150M = 33% over
        currency="HUF",
        year_built=2024,
        renovation_state="új építésű",
        description="…",
    )
    query = ParsedQuery(hard_filters=HardFilters(property_type_in=["Ház"], price_huf_max=150_000_000))
    narrative = fallback_narrative(listing, query)
    price_chip = next(c for c in narrative.rationale_chips if c.group == "price")
    assert price_chip.status == "miss"
    # Compromise sentence should mention "Ár" since one chip is a miss.
    assert "ár" in narrative.compromise.lower()


def test_fallback_narrative_reads_the_canonical_huf_amount_and_ignores_a_legacy_payload_key() -> None:
    # A row has no `price` attribute since Alembic 0061; a payload-shaped listing
    # may still carry the key and must not outrank the canonical amount.
    listing = _listing()
    listing.price = 200_000_000.0
    listing.price_huf = 140_000_000

    narrative = fallback_narrative(listing, _query())

    assert "140 M Ft" in narrative.body
    price_chip = next(chip for chip in narrative.rationale_chips if chip.group == "price")
    assert price_chip.status == "match"


def test_fallback_narrative_renovation_status_from_listing_state() -> None:
    """The renovation chip follows the listing's `renovation_state` literally."""
    query = ParsedQuery(hard_filters=HardFilters())
    for state, expected in (
        ("új építésű", "match"),
        ("felújítandó", "miss"),
        ("részben felújított", "warning"),
        (None, "warning"),
    ):
        listing = _FakeListing(
            id="p",
            title="t",
            property_type="Lakás",
            transaction_type="sale",
            city="Budapest",
            district=None,
            area_m2=50.0,
            rooms="2",
            price_huf=50_000_000,
            currency="HUF",
            year_built=2020,
            renovation_state=state,
            description="",
        )
        narrative = fallback_narrative(listing, query)
        renovation_chip = next(c for c in narrative.rationale_chips if c.group == "renovation")
        assert renovation_chip.status == expected, f"state={state!r}"


def test_attach_vs_next_returns_new_instance() -> None:
    """Models are frozen, so attaching `vs_next` returns a copy, not a mutation."""
    base = AINarrative(headline="h", body="b")
    updated = attach_vs_next(base, compared_to="#2", summary="ez 5M drágább")
    assert base.vs_next is None
    assert updated.vs_next == VsNext(compared_to="#2", summary="ez 5M drágább")
    assert updated.headline == base.headline


# --- Mood derivation --------------------------------------------------------


def _bd(**overrides: float) -> ScoreBreakdown:
    """Helper: build a ScoreBreakdown with sensible defaults, overrideable per term."""
    base = dict(
        content=0.0,
        factual=0.0,
        softpref_core=0.0,
        softpref_free=0.0,
        anchor=0.0,
        quality=0.0,
        geo=0.0,
        pooled=0.0,
        recency=0.0,
        explore=0.0,
        total=0.0,
    )
    base.update(overrides)
    return ScoreBreakdown(**base)


def test_derive_mood_enthusiastic_when_total_and_fit_are_strong() -> None:
    """Strong total + strong fit signals → the agent is genuinely sold."""
    bd = _bd(content=0.18, factual=0.12, softpref_core=0.08, softpref_free=0.05, total=0.86)
    assert _derive_mood(bd) == "enthusiastic"


def test_derive_mood_experienced_when_location_carries_a_weak_fit() -> None:
    """Listing leans on district strength, fit is weak → district-savvy voice."""
    bd = _bd(content=0.05, factual=0.05, geo=0.20, pooled=0.15, total=0.62)
    assert _derive_mood(bd) == "experienced"


def test_derive_mood_cautious_when_total_is_low() -> None:
    """Low overall score → cautious, compromise-aware tone."""
    bd = _bd(content=0.08, factual=0.06, geo=0.05, total=0.42)
    assert _derive_mood(bd) == "cautious"


def test_derive_mood_cautious_when_fit_is_very_weak() -> None:
    """Even with a decent total, very weak fit signals → cautious."""
    bd = _bd(content=0.04, factual=0.03, anchor=0.10, quality=0.10, total=0.60)
    assert _derive_mood(bd) == "cautious"


def test_derive_mood_factual_is_the_safe_default() -> None:
    """Mid-range total, mid fit, no standout signal → neutral 'factual' tone."""
    bd = _bd(content=0.12, factual=0.10, softpref_core=0.06, softpref_free=0.04, geo=0.10, total=0.68)
    assert _derive_mood(bd) == "factual"


def test_render_user_prompt_includes_the_mood_directive() -> None:
    """The mood ends up in the user message so the model is forced to honour it."""
    bd = _bd(content=0.18, factual=0.12, softpref_core=0.08, softpref_free=0.05, total=0.86)
    user_msg = render_user_prompt(_listing(), _query(), bd)
    assert "HANGULAT (kötelező): enthusiastic" in user_msg
    # And the closing instruction repeats the mood for emphasis.
    assert "'enthusiastic' hangulatot" in user_msg


def test_render_user_prompt_routes_different_breakdowns_to_different_moods() -> None:
    """A weaker breakdown produces a different mood directive in the prompt."""
    cautious_bd = _bd(content=0.04, factual=0.03, total=0.40)
    msg = render_user_prompt(_listing(), _query(), cautious_bd)
    assert "HANGULAT (kötelező): cautious" in msg
