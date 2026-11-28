"""Generic async retry helper with exponential backoff.

Used by the `photo_mirror` and `geocode_address` to wrap transient HTTP failures.
Callers decide whether `RetryExhaustedError` is recoverable (fall back to original
input) or fatal (let propagate).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from ingovo_search.types.errors import IngovoError


class RetryExhaustedError(IngovoError):
    """All retry attempts exhausted.

    The original exception is chained via `__cause__`; inspect it via
    `err.__cause__` to recover the underlying failure for logging or fallback
    decisions.
    """


async def async_with_retry[T](
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int = 3,
    backoff_s: float = 1.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    max_backoff_s: float = 30.0,
) -> T:
    """Run `fn()` with exponential backoff on retryable failures.

    The first attempt runs immediately; each subsequent attempt waits
    `backoff_s * 2 ** (attempt - 1)`, capped at `max_backoff_s`. After
    `retries` total attempts have failed, raises `RetryExhaustedError`
    with the last exception chained via `__cause__`.

    Non-retryable exceptions (any class outside `retryable_exceptions`)
    propagate immediately without retry — use this to let programmer
    errors fail loudly while transient I/O retries.
    """
    if retries < 1:
        raise ValueError(f"retries must be >= 1, got {retries}")

    last_exc: BaseException | None = None
    for attempt in range(retries):
        try:
            return await fn()
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt < retries - 1:
                delay = min(backoff_s * (2**attempt), max_backoff_s)
                await asyncio.sleep(delay)

    raise RetryExhaustedError(f"all {retries} attempts failed") from last_exc
