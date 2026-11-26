"""Tests for image-driven search — analysis, sanitisation and the merge rules.

The merge tests are the important ones: they pin the guarantee that a photo can
never introduce a price, a location, a room count or a transaction type, and
that a typed sentence always beats what the picture suggested.
"""

import pytest

from ingovo_search.lib.cache import MemoryCache
from ingovo_search.lib.llm import LlmImage, LlmRequest, MockLlmClient, fixture_key
from ingovo_search.lib.llm_models import TIER_V_IMAGE_QUERY
from ingovo_search.recommender.image_query import (
    IMAGE_QUERY_PROMPT_VERSION,
    MAX_FREETEXT_PHRASES,
    MAX_SUMMARY_CHARS,
    ImageQuery,
    analyze_images,
    image_query_cache_key,
    image_query_chips,
    merge_image_query,
    render_image_query_user_prompt,
)
from ingovo_search.recommender.parser_schema import Anchors, GeoHints, HardFilters, ParsedQuery
from ingovo_search.types.errors import LlmError

JPEG = LlmImage(mime_type="image/jpeg", data=b"\xff\xd8\xff\xe0 fake jpeg bytes")
PNG = LlmImage(mime_type="image/png", data=b"\x89PNG\r\n\x1a\n fake png bytes")


def _confident(**overrides: object) -> ImageQuery:
    base = {
        "is_property_photo": True,
        "confidence": 0.8,
        "summary": "Világos, modern nappali nagy ablakokkal.",
    }
    return ImageQuery.model_validate(base | overrides)


def _mock_client(response: ImageQuery, *, images: list[LlmImage]) -> MockLlmClient:
    req = LlmRequest(
        model=TIER_V_IMAGE_QUERY,
        system="",
        user=render_image_query_user_prompt(len(images)),
        images=images,
    )
    return MockLlmClient({fixture_key(req): response.model_dump_json()})


# --- cache key -------------------------------------------------------------


def test_cache_key_is_content_addressed_and_order_independent() -> None:
    forward = image_query_cache_key([JPEG, PNG], prompt_version="v1", model="m")
    reversed_batch = image_query_cache_key([PNG, JPEG], prompt_version="v1", model="m")
    assert forward == reversed_batch


def test_cache_key_changes_with_prompt_version_content_and_locale() -> None:
    base = image_query_cache_key([JPEG], prompt_version="v1", model="m")
    assert base != image_query_cache_key([JPEG], prompt_version="v2", model="m")
    assert base != image_query_cache_key([PNG], prompt_version="v1", model="m")
    # The locale changes the language of the text the user is shown, so an `en`
    # analysis must never be served from the `hu` entry.
    assert base != image_query_cache_key([JPEG], prompt_version="v1", model="m", locale="en")


# --- analyze_images --------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_batch_never_calls_the_model() -> None:
    result = await analyze_images([], client=MockLlmClient())
    assert result == ImageQuery()


@pytest.mark.asyncio
async def test_analysis_is_cached_on_image_bytes() -> None:
    cache = MemoryCache()
    response = _confident(soft_preferences=["balcony"])
    client = _mock_client(response, images=[JPEG])

    first = await analyze_images([JPEG], client=client, cache=cache)
    assert first.soft_preferences == ["balcony"]

    # An empty MockLlmClient raises if it is called at all, so a second hit that
    # succeeds proves the result came from the cache.
    second = await analyze_images([JPEG], client=MockLlmClient(), cache=cache)
    assert second == first
    assert await cache.get(
        image_query_cache_key([JPEG], prompt_version=IMAGE_QUERY_PROMPT_VERSION, model=TIER_V_IMAGE_QUERY)
    )


@pytest.mark.asyncio
async def test_unusable_model_output_raises_llm_error() -> None:
    req = LlmRequest(model=TIER_V_IMAGE_QUERY, system="", user=render_image_query_user_prompt(1), images=[JPEG])
    client = MockLlmClient({fixture_key(req): '{"confidence": "nem szám"}'})
    with pytest.raises(LlmError):
        await analyze_images([JPEG], client=client)


@pytest.mark.asyncio
async def test_non_property_photo_is_reduced_to_nothing() -> None:
    noisy = ImageQuery(
        is_property_photo=False,
        confidence=0.9,
        property_type="Ház",
        soft_preferences=["pool", "garden"],
        soft_prefs_freetext=["mediterrán villa"],
        summary="Ez egy macska.",
    )
    result = await analyze_images([JPEG], client=_mock_client(noisy, images=[JPEG]))

    assert result.is_property_photo is False
    assert result.property_type is None
    assert result.soft_preferences == []
    assert result.soft_prefs_freetext == []
    assert result.summary == "Ez egy macska."


@pytest.mark.asyncio
async def test_sanitiser_drops_invented_tags_and_caps_freetext() -> None:
    raw = _confident(
        soft_preferences=["balcony", "balcony", "garden"],
        soft_prefs_freetext=[
            "skandináv   stílus",
            "SKANDINÁV STÍLUS",
            "faburkolat",
            "nyitott konyha",
            "boltíves ablak",
            "tetőtéri ferde fal",
            "vörös tégla homlokzat",
        ],
        summary="x" * (MAX_SUMMARY_CHARS + 80),
    )
    result = await analyze_images([JPEG], client=_mock_client(raw, images=[JPEG]))

    assert result.soft_preferences == ["balcony", "garden"]
    assert result.soft_prefs_freetext[0] == "skandináv stílus"
    assert len(result.soft_prefs_freetext) == MAX_FREETEXT_PHRASES
    assert len(result.summary) == MAX_SUMMARY_CHARS


# --- merge -----------------------------------------------------------------


def test_merge_is_a_no_op_for_a_non_property_photo() -> None:
    parsed = ParsedQuery(semantic_query="csendes lakás")
    merged = merge_image_query(parsed, ImageQuery(is_property_photo=False, confidence=0.9))
    assert merged == parsed


def test_merge_is_a_no_op_below_the_confidence_floor() -> None:
    parsed = ParsedQuery(semantic_query="csendes lakás")
    weak = _confident(confidence=0.1, soft_preferences=["view"])
    assert merge_image_query(parsed, weak) == parsed


def test_merge_never_touches_price_location_rooms_or_transaction() -> None:
    parsed = ParsedQuery(
        hard_filters=HardFilters(
            city="Budapest",
            district_in=["7"],
            transaction_type="rent",
            rooms_min=2,
            price_huf_max=400_000,
            area_m2_min=50,
        ),
        anchors=Anchors(price_huf=380_000),
        geo_hints=GeoHints(side_of_river="Pest"),
    )
    merged = merge_image_query(parsed, _confident(soft_preferences=["balcony"], property_type="Ház"))

    assert merged.hard_filters.city == "Budapest"
    assert merged.hard_filters.district_in == ["7"]
    assert merged.hard_filters.transaction_type == "rent"
    assert merged.hard_filters.rooms_min == 2
    assert merged.hard_filters.price_huf_max == 400_000
    assert merged.hard_filters.area_m2_min == 50
    assert merged.anchors == parsed.anchors
    assert merged.geo_hints == parsed.geo_hints


def test_merge_adds_property_type_only_into_an_empty_slot() -> None:
    empty = ParsedQuery()
    assert merge_image_query(empty, _confident(property_type="Ház")).hard_filters.property_type_in == ["Ház"]

    typed = ParsedQuery(hard_filters=HardFilters(property_type_in=["Lakás"]))
    assert merge_image_query(typed, _confident(property_type="Ház")).hard_filters.property_type_in == ["Lakás"]


def test_merge_respects_an_explicit_property_type_exclusion() -> None:
    parsed = ParsedQuery(hard_filters=HardFilters(property_type_exclude=["Ház"]))
    merged = merge_image_query(parsed, _confident(property_type="Ház"))
    assert merged.hard_filters.property_type_in == []


def test_merge_unions_soft_signal_without_duplicates() -> None:
    parsed = ParsedQuery(
        soft_preferences=["quiet", "balcony"],
        soft_prefs_freetext=["kutyabarát", "Skandináv stílus"],
        semantic_query="csendes erkélyes lakás",
    )
    image = _confident(
        soft_preferences=["balcony", "natural_light"],
        soft_prefs_freetext=["skandináv stílus", "faburkolat"],
        summary="Világos, modern nappali nagy ablakokkal.",
    )
    merged = merge_image_query(parsed, image)

    assert merged.soft_preferences == ["quiet", "balcony", "natural_light"]
    assert merged.soft_prefs_freetext == ["kutyabarát", "Skandináv stílus", "faburkolat"]
    assert merged.semantic_query == "csendes erkélyes lakás Világos, modern nappali nagy ablakokkal."


def test_merge_does_not_repeat_a_summary_already_in_the_semantic_query() -> None:
    parsed = ParsedQuery(semantic_query="Világos, modern nappali nagy ablakokkal.")
    merged = merge_image_query(parsed, _confident())
    assert merged.semantic_query == "Világos, modern nappali nagy ablakokkal."


def test_image_only_search_gets_its_whole_query_from_the_photo() -> None:
    merged = merge_image_query(
        ParsedQuery(),
        _confident(property_type="Lakás", soft_preferences=["natural_light"], soft_prefs_freetext=["faburkolat"]),
    )
    assert merged.semantic_query == "Világos, modern nappali nagy ablakokkal."
    assert merged.soft_preferences == ["natural_light"]
    assert merged.hard_filters.property_type_in == ["Lakás"]


# --- UI chips --------------------------------------------------------------


def test_chips_are_hungarian_and_ordered_type_then_tags_then_freetext() -> None:
    chips = image_query_chips(
        _confident(
            property_type="Ház",
            soft_preferences=["garden", "natural_light"],
            soft_prefs_freetext=["vörös tégla homlokzat"],
        )
    )
    assert chips == ["Ház", "kert", "sok természetes fény", "vörös tégla homlokzat"]


def test_chips_follow_the_requested_locale() -> None:
    analysis = _confident(
        property_type="Ház",
        soft_preferences=["garden", "natural_light"],
        soft_prefs_freetext=["red brick facade"],
    )
    assert image_query_chips(analysis, "en") == [
        "House",
        "garden",
        "lots of natural light",
        "red brick facade",
    ]


def test_english_prompt_asks_for_english_free_text() -> None:
    assert "ENGLISH" in render_image_query_user_prompt(1, "en")
    assert "MAGYARUL" in render_image_query_user_prompt(1, "hu")


def test_chips_are_empty_when_the_analysis_will_not_be_applied() -> None:
    assert image_query_chips(ImageQuery(is_property_photo=False, confidence=0.9)) == []
    assert image_query_chips(_confident(confidence=0.1, soft_preferences=["view"])) == []


@pytest.mark.asyncio
async def test_freetext_that_only_repeats_a_tag_label_is_dropped() -> None:
    """Models hand back "medence" alongside the `pool` tag; it must not survive.

    A duplicate double-counts in scoring and shows up twice in any list built
    from tags + freetext together.
    """
    raw = _confident(
        soft_preferences=["pool", "garden"],
        soft_prefs_freetext=["medence", "pool", "kert", "mediterrán hangulat"],
    )
    result = await analyze_images([JPEG], client=_mock_client(raw, images=[JPEG]))

    assert result.soft_preferences == ["pool", "garden"]
    assert result.soft_prefs_freetext == ["mediterrán hangulat"]


def test_chips_never_repeat_a_label() -> None:
    chips = image_query_chips(_confident(soft_preferences=["pool"], soft_prefs_freetext=["napfényes terasz"]))
    assert len(chips) == len(set(chips))
