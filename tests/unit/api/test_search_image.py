"""Unit tests for image-carrying search — `POST /api/search/with-images`.

Three halves:

- upload validation at the trust boundary — count, size, and the magic-byte
  sniff that decides the format regardless of what the browser claimed;
- the merge: photos reach the recommender as soft signal on the parsed query;
- the fallbacks: a vision call that fails or times out must degrade the search
  to text-only rather than break it, and say so in the response.
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests._fakes import fake_db_session

from ingovo_search.api.deps import (
    get_db_session,
    get_embedding_client,
    get_llm_client,
    get_session_cache,
    get_session_cookie_secure,
    get_session_secret,
    get_session_store,
)
from ingovo_search.api.envelope import install_exception_handlers
from ingovo_search.api.search import router as search_router
from ingovo_search.api.search_image import MAX_BYTES_PER_IMAGE, MAX_IMAGES, _sniff_mime, read_search_images
from ingovo_search.lib.cache import Cache, MemoryCache
from ingovo_search.lib.embeddings import MockEmbeddingClient
from ingovo_search.lib.llm import MockLlmClient
from ingovo_search.lib.session import SessionStore
from ingovo_search.recommender.image_query import ImageQuery
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.types.errors import LlmError

SECRET = b"x" * 32

JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"jpeg payload" * 8
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"png payload" * 8
WEBP_BYTES = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"webp payload" * 8
HEIC_BYTES = b"\x00\x00\x00\x18ftypheic" + b"heic payload" * 8


# FastAPI resolves an override's own signature, and these classes' constructors take parameters it
# would turn into request parameters, so each override is a parameterless factory (fresh per request).
def _mock_llm_client() -> MockLlmClient:
    return MockLlmClient()


def _mock_embedding_client() -> MockEmbeddingClient:
    return MockEmbeddingClient()


def _parsed_from(mock: AsyncMock) -> ParsedQuery:
    """The ParsedQuery the recommender was actually called with."""
    call = mock.await_args
    assert call is not None, "expected recommend() to have been awaited"
    return call.args[1]


@pytest.fixture
def session_cache() -> Cache:
    return MemoryCache()


@pytest.fixture
def analysis() -> ImageQuery:
    return ImageQuery(
        is_property_photo=True,
        confidence=0.8,
        property_type="Ház",
        soft_preferences=["garden"],
        soft_prefs_freetext=["vörös tégla homlokzat"],
        summary="Kertes családi ház vörös tégla homlokzattal.",
    )


@pytest.fixture
def app(session_cache: Cache) -> FastAPI:
    application = FastAPI()
    install_exception_handlers(application)
    application.include_router(search_router)
    application.dependency_overrides[get_session_store] = lambda: SessionStore(session_cache)
    application.dependency_overrides[get_session_cache] = lambda: session_cache
    application.dependency_overrides[get_session_secret] = lambda: SECRET
    application.dependency_overrides[get_session_cookie_secure] = lambda: False
    application.dependency_overrides[get_llm_client] = _mock_llm_client
    application.dependency_overrides[get_embedding_client] = _mock_embedding_client
    application.dependency_overrides[get_db_session] = fake_db_session
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def mock_analyze(analysis: ImageQuery):
    with patch("ingovo_search.search.request.analyze_images", new_callable=AsyncMock) as mock:
        mock.return_value = analysis
        yield mock


@pytest.fixture(autouse=True)
def mock_pipeline():
    """Same stubbing as `test_search.py` — the recommender itself is not under test."""
    with (
        patch("ingovo_search.search.request.parse_query", new_callable=AsyncMock) as mock_parse,
        patch("ingovo_search.search.candidates.recommend", new_callable=AsyncMock) as mock_rec,
        patch("ingovo_search.search.pipeline.should_clarify") as mock_clarify,
        patch("ingovo_search.search.reranking.rerank_and_explain", new_callable=AsyncMock),
    ):
        mock_parse.return_value = ParsedQuery()
        mock_rec.return_value = []
        mock_clarify.return_value = None
        yield {"parse_query": mock_parse, "recommend": mock_rec}


def _post(client: TestClient, *, query: str = "kertes ház", files: Any = None) -> Any:
    return client.post(
        "/api/search/with-images",
        data={"payload": f'{{"query": "{query}"}}', "locale": "hu"},
        files=files if files is not None else {"files": ("otthon.jpg", JPEG_BYTES, "image/jpeg")},
    )


# --- magic-byte sniff ------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (JPEG_BYTES, "image/jpeg"),
        (PNG_BYTES, "image/png"),
        (WEBP_BYTES, "image/webp"),
        (HEIC_BYTES, "image/heic"),
        (b"%PDF-1.7 not an image", None),
        (b"", None),
    ],
)
def test_sniff_mime_identifies_by_content(payload: bytes, expected: str | None) -> None:
    assert _sniff_mime(payload[:16]) == expected


@pytest.mark.asyncio
async def test_read_search_images_ignores_an_empty_file_part() -> None:
    """Browsers submit an empty part for an untouched file input."""
    assert await read_search_images([]) == []


# --- upload validation -----------------------------------------------------


def test_upload_trusts_content_over_the_declared_mime(client: TestClient, mock_analyze: AsyncMock) -> None:
    """A PNG announced as JPEG is still read as a PNG."""
    _post(client, files={"files": ("a.jpg", PNG_BYTES, "image/jpeg")})
    call = mock_analyze.await_args
    assert call is not None
    assert [image.mime_type for image in call.args[0]] == ["image/png"]


def test_upload_rejects_a_declared_non_image(client: TestClient) -> None:
    response = _post(client, files={"files": ("cv.pdf", JPEG_BYTES, "application/pdf")})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_input"


def test_upload_rejects_a_disguised_non_image(client: TestClient) -> None:
    response = _post(client, files={"files": ("x.jpg", b"%PDF-1.7 payload", "image/jpeg")})
    assert response.status_code == 400
    assert "kép" in response.json()["error"]["message"]


def test_upload_rejects_an_oversized_image(client: TestClient) -> None:
    oversized = JPEG_BYTES + b"\x00" * MAX_BYTES_PER_IMAGE
    response = _post(client, files={"files": ("big.jpg", oversized, "image/jpeg")})
    assert response.status_code == 400
    assert "nagy" in response.json()["error"]["message"]


def test_upload_rejects_too_many_images(client: TestClient) -> None:
    files = [("files", (f"{i}.jpg", JPEG_BYTES, "image/jpeg")) for i in range(MAX_IMAGES + 1)]
    response = _post(client, files=files)
    assert response.status_code == 400
    assert str(MAX_IMAGES) in response.json()["error"]["message"]


@pytest.mark.usefixtures("mock_analyze")
def test_upload_accepts_a_full_batch(client: TestClient) -> None:
    files = [("files", (f"{i}.jpg", JPEG_BYTES, "image/jpeg")) for i in range(MAX_IMAGES)]
    response = _post(client, files=files)
    assert response.status_code == 200
    assert response.json()["data"]["image_context"]["accepted_images"] == MAX_IMAGES


def test_a_malformed_payload_field_is_a_400(client: TestClient) -> None:
    response = client.post(
        "/api/search/with-images",
        data={"payload": "{not json", "locale": "hu"},
        files={"files": ("a.jpg", JPEG_BYTES, "image/jpeg")},
    )
    assert response.status_code == 400


# --- the merge -------------------------------------------------------------


@pytest.mark.usefixtures("mock_analyze")
def test_photos_reach_the_recommender_as_soft_signal(client: TestClient, mock_pipeline: dict[str, AsyncMock]) -> None:
    response = _post(client, query="kertes ház Debrecenben")

    assert response.status_code == 200
    parsed = _parsed_from(mock_pipeline["recommend"])
    assert parsed.hard_filters.property_type_in == ["Ház"]
    assert parsed.soft_preferences == ["garden"]
    assert "vörös tégla homlokzat" in parsed.soft_prefs_freetext
    assert "Kertes családi ház" in parsed.semantic_query


@pytest.mark.usefixtures("mock_analyze")
def test_the_response_explains_what_the_photo_contributed(client: TestClient) -> None:
    context = _post(client).json()["data"]["image_context"]
    assert context["applied"] is True
    assert context["reason"] == "applied"
    assert context["accepted_images"] == 1
    assert context["chips"] == ["Ház", "kert", "vörös tégla homlokzat"]
    assert context["summary"] == "Kertes családi ház vörös tégla homlokzattal."


@pytest.mark.usefixtures("mock_analyze")
def test_an_image_only_search_skips_the_parser_entirely(
    client: TestClient, mock_pipeline: dict[str, AsyncMock]
) -> None:
    response = _post(client, query="")

    assert response.status_code == 200
    mock_pipeline["parse_query"].assert_not_awaited()
    assert _parsed_from(mock_pipeline["recommend"]).semantic_query == "Kertes családi ház vörös tégla homlokzattal."


def test_a_plain_json_search_carries_no_image_context(client: TestClient) -> None:
    response = client.post("/api/search", json={"query": "csendes lakás"})
    assert response.status_code == 200
    assert response.json()["data"]["image_context"] is None


# --- fallbacks -------------------------------------------------------------


def test_a_failed_vision_call_still_returns_results(client: TestClient, mock_pipeline: dict[str, AsyncMock]) -> None:
    """Every model has already been tried by the time the error surfaces."""
    with patch("ingovo_search.search.request.analyze_images", new_callable=AsyncMock) as mock:
        mock.side_effect = LlmError("all vision models failed")
        response = _post(client, query="kertes ház Debrecenben")

    assert response.status_code == 200
    context = response.json()["data"]["image_context"]
    assert context["applied"] is False
    assert context["reason"] == "unavailable"
    assert _parsed_from(mock_pipeline["recommend"]) == ParsedQuery()


def test_a_slow_vision_call_times_out_into_a_text_only_search(
    client: TestClient, mock_pipeline: dict[str, AsyncMock]
) -> None:
    async def never_finishes(*_args: object, **_kwargs: object) -> ImageQuery:
        await asyncio.sleep(30)
        raise AssertionError("should have been cancelled")

    with (
        patch("ingovo_search.search.request.analyze_images", new=never_finishes),
        patch("ingovo_search.search.request.get_settings") as mock_settings,
    ):
        mock_settings.return_value = mock_settings.return_value
        settings = mock_settings.return_value
        settings.ai_search_image_timeout_ms = 50
        settings.ai_search_parser_timeout_ms = 2500
        settings.ai_search_rerank_timeout_ms = 700
        settings.ai_search_rerank_rank_only = False
        settings.llm_daily_budget_usd = 25.0
        response = _post(client, query="kertes ház Debrecenben")

    assert response.status_code == 200
    context = response.json()["data"]["image_context"]
    assert context["reason"] == "unavailable"
    assert _parsed_from(mock_pipeline["recommend"]) == ParsedQuery()


def test_a_non_property_photo_is_reported_and_ignored(client: TestClient, mock_pipeline: dict[str, AsyncMock]) -> None:
    with patch("ingovo_search.search.request.analyze_images", new_callable=AsyncMock) as mock:
        mock.return_value = ImageQuery(is_property_photo=False, confidence=0.9, summary="Ez egy macska.")
        response = _post(client, query="kertes ház")

    context = response.json()["data"]["image_context"]
    assert context["applied"] is False
    assert context["reason"] == "not_a_property"
    assert context["summary"] == "Ez egy macska."
    assert _parsed_from(mock_pipeline["recommend"]) == ParsedQuery()


def test_an_unsure_reading_is_reported_and_ignored(client: TestClient, mock_pipeline: dict[str, AsyncMock]) -> None:
    with patch("ingovo_search.search.request.analyze_images", new_callable=AsyncMock) as mock:
        mock.return_value = ImageQuery(is_property_photo=True, confidence=0.1, summary="Homályos kép.")
        response = _post(client, query="kertes ház")

    context = response.json()["data"]["image_context"]
    assert context["applied"] is False
    assert context["reason"] == "unsure"
    assert _parsed_from(mock_pipeline["recommend"]) == ParsedQuery()
