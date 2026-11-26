"""The photos attached to a search: the vision call and how its reading is folded in."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass

from ingovo_search.lib.llm import LlmImage
from ingovo_search.recommender.image_query import ImageLocale, ImageQuery, image_query_chips, merge_image_query
from ingovo_search.recommender.parser_schema import ParsedQuery
from ingovo_search.search.constants import search_logger
from ingovo_search.search.models import ImageContextSummary


@dataclass(frozen=True, slots=True)
class SearchImages:
    """The photos attached to the search and the language their labels follow."""

    files: list[LlmImage]
    locale: ImageLocale


async def cancel_image_task(task: asyncio.Task[ImageQuery] | None) -> None:
    """Cancel a still-running vision call and swallow its outcome."""
    if task is None or task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task


async def await_image_analysis(
    task: asyncio.Task[ImageQuery],
    *,
    parsed: ParsedQuery,
    images: SearchImages,
    timeout_ms: int,
    request_id: str = "",
) -> tuple[ParsedQuery, ImageContextSummary]:
    """Collect the vision result and merge it, or degrade without failing.

    Attaching a photo must never be able to break a search that would otherwise
    work. Transport failures are already retried across the vision model chain
    one layer down (`LlmRequest.fallback_models`), and schema failures get a
    correction retry inside `analyze_images` — so by the time an exception
    reaches here, every model has been tried. The search then runs on the typed
    sentence alone, and the response says so via `reason="unavailable"` instead
    of pretending the upload counted.
    """
    image_count = len(images.files)
    try:
        analysis = await asyncio.wait_for(task, timeout=timeout_ms / 1000) if timeout_ms > 0 else await task
    except TimeoutError:
        await cancel_image_task(task)
        search_logger.warning(
            "image analysis timed out; searching on the text alone",
            extra={"request_id": request_id, "timeout_ms": timeout_ms, "image_count": image_count},
        )
        return parsed, ImageContextSummary(accepted_images=image_count, applied=False, reason="unavailable")
    except Exception as exc:  # noqa: BLE001
        # Deliberate fail-soft boundary (see the docstring): `GoogleAiClient.complete`
        # re-raises the provider SDK's own exception once the vision chain is
        # exhausted, and a stale cache payload surfaces as a pydantic error, so
        # the set is open-ended. Swallowing is correct — the photo is optional
        # and the typed query still runs — but the traceback goes to the log.
        search_logger.warning(
            "image analysis failed on every model; searching on the text alone",
            extra={
                "request_id": request_id,
                "image_count": image_count,
                "error": f"{type(exc).__name__}: {exc}"[:200],
            },
            exc_info=exc,
        )
        return parsed, ImageContextSummary(accepted_images=image_count, applied=False, reason="unavailable")

    # Labels follow the site language: the fixed-vocab tags have no text of their
    # own, so if they were always emitted in Hungarian an English page would show
    # a row of half-translated chips.
    chips = image_query_chips(analysis, images.locale)
    if not analysis.is_property_photo:
        return parsed, ImageContextSummary(
            accepted_images=image_count,
            applied=False,
            reason="not_a_property",
            summary=analysis.summary,
        )
    if not chips:
        # A property photo the model could not read with enough confidence.
        return parsed, ImageContextSummary(
            accepted_images=image_count,
            applied=False,
            reason="unsure",
            summary=analysis.summary,
        )

    return (
        merge_image_query(parsed, analysis),
        ImageContextSummary(
            accepted_images=image_count,
            applied=True,
            reason="applied",
            summary=analysis.summary,
            chips=chips,
        ),
    )
