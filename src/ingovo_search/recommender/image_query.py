"""Image-driven search — a photo becomes extra *soft* signal on the query.

The buyer attaches one or more photos ("ilyet szeretnék") next to (or instead
of) a sentence. One Tier-V vision call turns the batch into an `ImageQuery`:
what kind of home it is, which fixed-vocab soft-preference tags are actually
visible, a few long-tail Hungarian style phrases, and one sentence the UI can
show back ("ezt olvastuk ki a képből").

Two rules shape the whole module, and both exist because a photo is a much
weaker evidence source than a typed sentence:

1. **A picture never sets a hard constraint.** Price, city, district, room
   count and floor area are not visible in a photo — any model that claims
   otherwise is hallucinating. `merge_image_query` touches `soft_preferences`,
   `soft_prefs_freetext` and `semantic_query` only; the single exception is
   `property_type_in`, and only when the typed query left it empty.
2. **Typed text wins every conflict.** The merge is additive; nothing the
   parser produced is overwritten or removed.

`analyze_images` is cached on the image bytes themselves, so re-submitting the
same photo (a refinement, a back-navigation, a retry) costs nothing.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from typing import Literal

import structlog
from pydantic import BaseModel, Field, ValidationError

from ingovo_search.lib.cache import Cache, get_json, set_json
from ingovo_search.lib.llm import LlmClient, LlmImage, LlmRequest
from ingovo_search.lib.llm_cost_log import llm_call_row, record_llm_calls_later
from ingovo_search.lib.llm_models import TIER_V_IMAGE_QUERY, TIER_V_IMAGE_QUERY_CHAIN
from ingovo_search.lib.llm_usage import last_llm_usage, reset_last_llm_usage
from ingovo_search.recommender.parser_schema import (
    SOFT_PREF_TAGS,
    HardFilters,
    ParsedQuery,
    PropertyType,
    SoftPrefTag,
)
from ingovo_search.types.errors import LlmError

IMAGE_QUERY_PROMPT_VERSION = "image_query/v1.0"
IMAGE_QUERY_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60

# Below this the vision call is treated as "did not really understand the
# picture" and contributes nothing — better a text-only search than a query
# steered by a guess.
MIN_MERGE_CONFIDENCE = 0.35

MAX_FREETEXT_PHRASES = 5
MAX_FREETEXT_PHRASE_CHARS = 48
MAX_SUMMARY_CHARS = 180
# Ceiling on the merged freetext list so a photo batch cannot crowd out the
# phrases the user actually typed.
MAX_MERGED_FREETEXT = 12

_logger = structlog.get_logger("ingovo_search.recommender.image_query")

# The composer's two real languages (mirrors `SuggestionLocale` on the web
# side): Hungarian, and English for every other locale.
type ImageLocale = Literal["hu", "en"]

type ImageSubject = Literal[
    "exterior",
    "living_room",
    "kitchen",
    "bedroom",
    "bathroom",
    "garden",
    "view",
    "floor_plan",
    "other",
]


def _empty_softpref_list() -> list[SoftPrefTag]:
    return []


def _empty_str_list() -> list[str]:
    return []


class ImageQuery(BaseModel):
    """What one batch of uploaded photos contributes to a search.

    Every field is optional/defaulted: a structurally incomplete model response
    still validates, and an empty `ImageQuery` merges as a no-op.
    """

    model_config = {"frozen": True}

    # False for anything that is not a home: a person, a screenshot, a meme, a
    # document. The endpoint reports this back so the UI can say "ez nem tűnik
    # ingatlanfotónak" ("this does not look like a property photo") instead of
    # silently searching on noise.
    is_property_photo: bool = False
    subject: ImageSubject = "other"
    property_type: PropertyType | None = None
    soft_preferences: list[SoftPrefTag] = Field(default_factory=_empty_softpref_list)
    soft_prefs_freetext: list[str] = Field(default_factory=_empty_str_list)
    # One short Hungarian sentence, shown to the user under the thumbnails.
    summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


IMAGE_QUERY_SYSTEM_PROMPT = """Ingatlanfotó-elemző vagy egy magyar ingatlankereső motorban.
A felhasználó feltöltött egy vagy több képet, mert ILYEN otthont keres.
A feladatod: kizárólag azt írd le, ami a képeken TÉNYLEGESEN LÁTHATÓ.

SZIGORÚ TILTÁSOK — ezeket soha ne találd ki fényképből:
- ár, bérleti díj, rezsi
- város, kerület, utca, cím, ország
- alapterület négyzetméterben, szobaszám, emelet
- eladó vagy kiadó-e az ingatlan
Ha ilyesmit vélsz felismerni, akkor is hagyd ki. Ezeket a felhasználó mondata adja meg, nem a kép.

MEZŐK:
- is_property_photo: true, ha a kép egy lakóingatlant, annak belsejét, kertjét, kilátását
  vagy alaprajzát mutatja. Minden másra (ember, állat, dokumentum, képernyőkép, mém,
  autó, étel) false — és ilyenkor minden más mező maradjon üres/alapértelmezett.
- subject: mit ábrázol a kép elsősorban.
- property_type: csak ha egyértelműen eldönthető. Családi ház / ikerház / sorház
  külső képe -> "Ház". Társasházi vagy panel lakás belső tere -> "Lakás". Ha bizonytalan
  vagy csak egy szobabelsőt látsz kontextus nélkül, hagyd null-on.
- soft_preferences: KIZÁROLAG a lenti szótárból, és csak akkor, ha a képen látszik.
  Ne következtess: az, hogy egy nappali szép, nem jelenti azt, hogy "renovated".
- soft_prefs_freetext: 2-5 rövid magyar kifejezés a stílusról, anyagokról, hangulatról,
  amit a fix szótár nem fed le. Példák: "skandináv stílus", "faburkolat", "tetőtéri ferde fal",
  "nyitott konyha", "ipari loft hangulat", "boltíves ablak", "vörös tégla homlokzat".
  Ne ismételd a soft_preferences tagjeit magyarul.
- summary: EGY rövid magyar mondat arról, mit keres a felhasználó a kép alapján.
  A felhasználónak mutatjuk meg. Példa: "Világos, modern nappali nagy ablakokkal és faparkettával."
- confidence: 0.0-1.0, mennyire vagy biztos az egész elemzésben. Rossz minőségű,
  sötét, vagy nehezen értelmezhető képnél legyen alacsony (0.3 alatt).

SOFT_PREFERENCES SZÓTÁR (csak ezek használhatók):
renovated, partially_renovated, new_build, fixer_upper, parking, garage, garden,
terrace, balcony, basement, pool, elevator, air_conditioning, natural_light,
sunny, view, central, quiet, near_metro, walkable, large, small

Néhány tag jelentése ebben a kontextusban:
- natural_light / sunny: nagy ablakfelület, erős természetes fény a képen
- view: az ablakból/teraszról látható panoráma
- fixer_upper: láthatóan felújításra szoruló, lelakott állapot
- new_build: újépítésű, még csupasz vagy frissen átadott állapot
- large / small: a tér láthatóan tágas, illetve szűkös

Csak a megadott JSON sémának megfelelő választ adj."""

IMAGE_QUERY_CORRECTION_SYSTEM_PROMPT = (
    "Az előző válaszod sémaellenőrzése (schema validation) sikertelen volt. Nézd meg újra "
    "a képeket, és add vissza a választ egyetlen JSON objektumban, amely pontosan illeszkedik "
    "az ImageQuery sémára. Ne fűzz hozzá semmilyen magyarázatot.\n\n" + IMAGE_QUERY_SYSTEM_PROMPT
)


def render_image_query_user_prompt(image_count: int, locale: ImageLocale = "hu") -> str:
    """Instruction that follows the inline image parts.

    The prompt itself stays Hungarian — it is the analyst's brief, and a single
    wording keeps the eval comparable. Only the *free text* the user will read
    (`summary`, `soft_prefs_freetext`) follows the site language.
    """
    language_rule = (
        "A `summary` és a `soft_prefs_freetext` mezőket MAGYARUL írd."
        if locale == "hu"
        else "Write the `summary` and `soft_prefs_freetext` fields in ENGLISH, even though these instructions are Hungarian."
    )
    if image_count == 1:
        subject = "Elemezd a fenti képet a rendszerüzenet szabályai szerint, és add vissza a JSON-t."
    else:
        subject = (
            f"A fenti {image_count} kép ugyanannak a keresésnek a része — együtt, egyetlen "
            "elemzésként értelmezd őket, és add vissza a JSON-t a rendszerüzenet szabályai szerint. "
            "Csak olyasmit vegyél fel, ami legalább az egyik képen látható."
        )
    return f"{subject}\n{language_rule}"


def image_query_cache_key(
    images: Sequence[LlmImage], *, prompt_version: str, model: str, locale: ImageLocale = "hu"
) -> str:
    """Content-addressed key: the same photos always hit the same entry.

    Digests are sorted so re-uploading the same batch in a different order —
    which happens whenever a user removes and re-adds a thumbnail — still hits.
    The locale is part of the key because it changes the language of the text
    the user is shown.
    """
    digests = sorted(hashlib.sha256(image.data).hexdigest() for image in images)
    fingerprint = hashlib.sha256("|".join(digests).encode("utf-8")).hexdigest()
    return f"image_query:{prompt_version}:{model}:{locale}:{fingerprint}"


def _sanitise(raw: ImageQuery) -> ImageQuery:
    """Clamp the model's output to what the rest of the pipeline can trust.

    A non-property photo is reduced to an empty result outright: whatever tags
    a meme or a screenshot produced are noise by definition.
    """
    if not raw.is_property_photo:
        return ImageQuery(is_property_photo=False, summary=raw.summary.strip()[:MAX_SUMMARY_CHARS])

    seen_tags: set[str] = set()
    tags: list[SoftPrefTag] = []
    for tag in raw.soft_preferences:
        if tag in SOFT_PREF_TAGS and tag not in seen_tags:
            seen_tags.add(tag)
            tags.append(tag)

    # Freetext is for what the fixed vocabulary cannot express. The prompt says
    # so, but models still hand back "medence" next to the `pool` tag — which
    # then double-counts in scoring and shows up twice in any list built from
    # both. Drop any phrase that is just a tag label in either language.
    tag_labels = {label.casefold() for tag in tags for label in (SOFT_PREF_LABELS_HU[tag], SOFT_PREF_LABELS_EN[tag])}

    seen_phrases: set[str] = set()
    phrases: list[str] = []
    for phrase in raw.soft_prefs_freetext:
        cleaned = " ".join(phrase.split())[:MAX_FREETEXT_PHRASE_CHARS].strip()
        key = cleaned.casefold()
        if not cleaned or key in seen_phrases or key in tag_labels:
            continue
        seen_phrases.add(key)
        phrases.append(cleaned)
        if len(phrases) >= MAX_FREETEXT_PHRASES:
            break

    return raw.model_copy(
        update={
            "soft_preferences": tags,
            "soft_prefs_freetext": phrases,
            "summary": " ".join(raw.summary.split())[:MAX_SUMMARY_CHARS],
        }
    )


async def _vision_attempt(
    images: Sequence[LlmImage],
    *,
    client: LlmClient,
    model: str,
    locale: ImageLocale,
    attempt: int,
) -> tuple[ImageQuery | None, ValidationError | None]:
    """One vision call and its cost row: the sanitised result, or the schema error."""
    is_correction = attempt > 1
    req = LlmRequest(
        model=model,
        system=IMAGE_QUERY_CORRECTION_SYSTEM_PROMPT if is_correction else IMAGE_QUERY_SYSTEM_PROMPT,
        user=render_image_query_user_prompt(len(images), locale),
        response_schema=ImageQuery,
        temperature=0.0,
        images=list(images),
        fallback_models=list(TIER_V_IMAGE_QUERY_CHAIN),
    )
    started = time.monotonic()
    reset_last_llm_usage()
    raw_response = await client.complete(req)
    usage = last_llm_usage()
    parsed: ImageQuery | None = None
    error: ValidationError | None = None
    try:
        parsed = _sanitise(ImageQuery.model_validate_json(raw_response))
    except ValidationError as exc:
        error = exc
        await _logger.awarning(
            "image_query schema validation failed",
            attempt=attempt,
            error=str(exc)[:200],
        )
    # Cost row, written in the
    # background: this call sits in front of the search.
    record_llm_calls_later(
        [
            llm_call_row(
                usage=usage,
                requested_model=model,
                prompt_version=IMAGE_QUERY_PROMPT_VERSION,
                latency_ms=int((time.monotonic() - started) * 1000),
                outcome="success" if parsed is not None else "parse_error",
                prompt_text=req.system + req.user,
                completion_text=raw_response,
            )
        ]
    )
    return parsed, error


async def analyze_images(
    images: Sequence[LlmImage],
    *,
    client: LlmClient,
    cache: Cache | None = None,
    model: str = TIER_V_IMAGE_QUERY,
    locale: ImageLocale = "hu",
) -> ImageQuery:
    """One vision call per photo batch, cached on the image bytes.

    Raises `LlmError` when the model is unreachable or answers with something
    that is not a valid `ImageQuery` — the caller turns that into a degraded
    text-only search rather than failing the whole request.
    """
    if not images:
        return ImageQuery()

    cache_key = image_query_cache_key(images, prompt_version=IMAGE_QUERY_PROMPT_VERSION, model=model, locale=locale)
    if cache is not None:
        cached = await get_json(cache, cache_key, ImageQuery)
        if cached is not None:
            await _logger.adebug("image_query cache hit", cache_key=cache_key)
            return cached

    # Two attempts, mirroring the text parser: the first asks normally, the
    # second re-asks with a schema-correction prompt. Transport failures are
    # handled a layer down — `LlmRequest.fallback_models` makes the client walk
    # the vision-capable chain before it gives up, so a single model being
    # overloaded or rate-limited never costs us the photo.
    last_error: ValidationError | None = None
    parsed: ImageQuery | None = None

    for attempt in (1, 2):
        parsed, error = await _vision_attempt(images, client=client, model=model, locale=locale, attempt=attempt)
        if parsed is not None:
            break
        last_error = error

    if parsed is None:
        raise LlmError(f"image analysis returned an unusable payload: {last_error}")

    if cache is not None:
        await set_json(cache, cache_key, parsed, ttl_seconds=IMAGE_QUERY_CACHE_TTL_SECONDS)
    await _logger.ainfo(
        "image_query analysed",
        image_count=len(images),
        is_property_photo=parsed.is_property_photo,
        property_type=parsed.property_type,
        tag_count=len(parsed.soft_preferences),
        confidence=parsed.confidence,
    )
    return parsed


def merge_image_query(parsed: ParsedQuery, image: ImageQuery) -> ParsedQuery:
    """Fold image-derived *soft* signal into a parsed query. Purely additive.

    No-op when the photo was not a property photo or the vision call was not
    confident enough. `property_type_in` is the only hard filter this can set,
    and only into an empty slot — if the sentence said "házat keresek" and the
    photo shows a flat, the sentence stands.
    """
    if not image.is_property_photo or image.confidence < MIN_MERGE_CONFIDENCE:
        return parsed

    tags: list[SoftPrefTag] = list(parsed.soft_preferences)
    for tag in image.soft_preferences:
        if tag not in tags:
            tags.append(tag)

    existing = {phrase.casefold() for phrase in parsed.soft_prefs_freetext}
    freetext = list(parsed.soft_prefs_freetext)
    for phrase in image.soft_prefs_freetext:
        if phrase.casefold() in existing:
            continue
        existing.add(phrase.casefold())
        freetext.append(phrase)
    freetext = freetext[:MAX_MERGED_FREETEXT]

    hard: HardFilters = parsed.hard_filters
    if (
        image.property_type is not None
        and not hard.property_type_in
        and image.property_type not in hard.property_type_exclude
    ):
        hard = hard.model_copy(update={"property_type_in": [image.property_type]})

    semantic = parsed.semantic_query.strip()
    if image.summary and image.summary.casefold() not in semantic.casefold():
        semantic = f"{semantic} {image.summary}".strip()

    return parsed.model_copy(
        update={
            "hard_filters": hard,
            "soft_preferences": tags,
            "soft_prefs_freetext": freetext,
            "semantic_query": semantic,
        }
    )


# Labels for the fixed vocab, so the UI can show what a photo contributed
# without shipping the tag identifiers to the browser. The free-text phrases
# next to these already come back in the requested language from the model.
SOFT_PREF_LABELS_EN: dict[str, str] = {
    "renovated": "renovated",
    "partially_renovated": "partly renovated",
    "new_build": "new build",
    "fixer_upper": "needs renovation",
    "parking": "parking",
    "garage": "garage",
    "garden": "garden",
    "terrace": "terrace",
    "balcony": "balcony",
    "basement": "basement",
    "pool": "pool",
    "elevator": "lift",
    "air_conditioning": "air conditioning",
    "natural_light": "lots of natural light",
    "sunny": "sunny",
    "view": "view",
    "central": "central",
    "quiet": "quiet",
    "near_metro": "near metro",
    "walkable": "walkable",
    "large": "spacious",
    "small": "compact",
}

PROPERTY_TYPE_LABELS_EN: dict[str, str] = {
    "Lakás": "Apartment",
    "Ház": "House",
    "Iroda": "Office",
    "Üzlethelyiség": "Retail unit",
    "Telek": "Plot",
    "Garázs": "Garage",
    "Ipari": "Industrial",
    "Nyaraló": "Holiday home",
    "Vendéglátás": "Hospitality",
    "Mezőgazdasági": "Agricultural",
}

SOFT_PREF_LABELS_HU: dict[str, str] = {
    "renovated": "felújított",
    "partially_renovated": "részben felújított",
    "new_build": "újépítésű",
    "fixer_upper": "felújítandó",
    "parking": "parkolás",
    "garage": "garázs",
    "garden": "kert",
    "terrace": "terasz",
    "balcony": "erkély",
    "basement": "pince",
    "pool": "medence",
    "elevator": "lift",
    "air_conditioning": "klíma",
    "natural_light": "sok természetes fény",
    "sunny": "napos",
    "view": "panoráma",
    "central": "központi",
    "quiet": "csendes",
    "near_metro": "metró közel",
    "walkable": "gyalogosbarát",
    "large": "tágas",
    "small": "kompakt",
}


def image_query_chips(image: ImageQuery, locale: ImageLocale = "hu") -> list[str]:
    """Short labels of everything the photo contributed, for the UI."""
    if not image.is_property_photo or image.confidence < MIN_MERGE_CONFIDENCE:
        return []
    tag_labels = SOFT_PREF_LABELS_HU if locale == "hu" else SOFT_PREF_LABELS_EN
    chips: list[str] = []
    if image.property_type is not None:
        chips.append(
            image.property_type
            if locale == "hu"
            else PROPERTY_TYPE_LABELS_EN.get(image.property_type, image.property_type)
        )
    chips.extend(tag_labels.get(tag, tag) for tag in image.soft_preferences)
    chips.extend(image.soft_prefs_freetext)
    return chips
