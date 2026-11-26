"""Upload validation for image-driven search.

Photos ride along with the search itself (`POST /api/search/with-images`) rather
than through a separate analyse-then-search round trip: one request carries the
sentence and the pictures together, and the vision call runs *inside* that
request, concurrently with the parser.

This module owns the trust boundary — how many images, how big, and whether the
bytes really are an image. Nothing here is persisted: the uploads live only for
the duration of the request, and only the derived tags and one sentence ever
leave it.

The `ParseError` messages are shown to the visitor as-is, so they are Hungarian
product copy (for example "Az egyik feltöltött fájl üres." = "One of the uploaded
files is empty.").
"""

from __future__ import annotations

from fastapi import UploadFile

from ingovo_search.lib.llm import LlmImage
from ingovo_search.types.errors import ParseError

MAX_IMAGES = 4
MAX_BYTES_PER_IMAGE = 6 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024

# Formats Gemini reads inline. HEIC/HEIF are here because that is what an
# iPhone hands over when a user picks a photo from their library.
ALLOWED_MIME_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/heic",
        "image/heif",
    }
)


def _sniff_mime(head: bytes) -> str | None:
    """Identify an image by its magic bytes.

    The browser-supplied `content_type` is attacker-controlled, so it is only
    used as a first cheap gate — this sniff is what actually decides. Anything
    unrecognised is rejected rather than passed to the model.
    """
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"}:
            return "image/heic"
        if brand in {b"heif", b"hevm"}:
            return "image/heif"
    return None


async def _read_validated_image(upload: UploadFile) -> LlmImage:
    """Read one upload fully, enforcing the size and format limits."""
    declared = (upload.content_type or "").split(";")[0].strip().lower()
    if declared and declared not in ALLOWED_MIME_TYPES:
        raise ParseError(f"Nem támogatott képformátum: {declared}. JPEG, PNG, WEBP vagy HEIC képet tölts fel.")

    data = await upload.read()
    if not data:
        raise ParseError("Az egyik feltöltött fájl üres.")
    if len(data) > MAX_BYTES_PER_IMAGE:
        limit_mb = MAX_BYTES_PER_IMAGE // (1024 * 1024)
        raise ParseError(f"A kép túl nagy (max. {limit_mb} MB képenként).")

    sniffed = _sniff_mime(data[:16])
    if sniffed is None:
        raise ParseError("A feltöltött fájl nem felismerhető kép. JPEG, PNG, WEBP vagy HEIC képet tölts fel.")
    return LlmImage(mime_type=sniffed, data=data)


async def read_search_images(files: list[UploadFile]) -> list[LlmImage]:
    """Validate and read an attachment batch.

    Raises `ParseError` (→ 400 `invalid_input`) on anything the user can fix by
    picking different files. A rejected upload fails the whole search request
    rather than silently dropping the photo — the user chose to attach it, so a
    silent drop would be a lie. Failures *inside* the vision call are different:
    those degrade to a text-only search, see `api/search.py`.
    """
    real_files = [upload for upload in files if upload.filename or upload.size]
    if not real_files:
        return []
    if len(real_files) > MAX_IMAGES:
        raise ParseError(f"Legfeljebb {MAX_IMAGES} képet tölthetsz fel egyszerre.")

    images: list[LlmImage] = []
    total_bytes = 0
    for upload in real_files:
        image = await _read_validated_image(upload)
        total_bytes += len(image.data)
        if total_bytes > MAX_TOTAL_BYTES:
            limit_mb = MAX_TOTAL_BYTES // (1024 * 1024)
            raise ParseError(f"A képek összmérete túl nagy (max. {limit_mb} MB).")
        images.append(image)
    return images
