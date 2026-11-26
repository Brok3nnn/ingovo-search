"""Unit tests for `async_with_retry`."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ingovo_search.lib.retry import RetryExhaustedError, async_with_retry


async def test_returns_value_on_first_try() -> None:
    fn = AsyncMock(return_value=42)
    result = await async_with_retry(fn)

    assert result == 42
    assert fn.call_count == 1


async def test_retries_then_succeeds() -> None:
    calls = 0

    async def flaky() -> int:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise ConnectionError("transient")
        return 42

    result = await async_with_retry(flaky, backoff_s=0.0)

    assert result == 42
    assert calls == 2


async def test_raises_retry_exhausted_after_max_attempts() -> None:
    async def always_fails() -> None:
        raise ConnectionError("permanent")

    with pytest.raises(RetryExhaustedError) as exc_info:
        await async_with_retry(always_fails, retries=3, backoff_s=0.0)

    assert isinstance(exc_info.value.__cause__, ConnectionError)


async def test_non_retryable_exception_propagates_immediately() -> None:
    """`ValueError` is not in the retryable set → propagates without retry."""
    calls = 0

    async def value_error() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("not retryable")

    with pytest.raises(ValueError, match="not retryable"):
        await async_with_retry(
            value_error,
            retries=3,
            backoff_s=0.0,
            retryable_exceptions=(ConnectionError,),
        )

    assert calls == 1


async def test_backoff_grows_exponentially() -> None:
    """Sleeps between attempts follow `backoff_s * 2 ** attempt`, capped."""
    calls = 0

    async def flaky() -> None:
        nonlocal calls
        calls += 1
        raise ConnectionError("transient")

    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    with patch("ingovo_search.lib.retry.asyncio.sleep", side_effect=fake_sleep), pytest.raises(RetryExhaustedError):
        await async_with_retry(flaky, retries=4, backoff_s=1.0)

    # 4 attempts → 3 sleeps at 1s, 2s, 4s
    assert delays == [1.0, 2.0, 4.0]


async def test_backoff_capped_at_max() -> None:
    calls = 0

    async def flaky() -> None:
        nonlocal calls
        calls += 1
        raise ConnectionError("transient")

    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    with patch("ingovo_search.lib.retry.asyncio.sleep", side_effect=fake_sleep), pytest.raises(RetryExhaustedError):
        await async_with_retry(flaky, retries=6, backoff_s=10.0, max_backoff_s=15.0)

    # Would be 10, 20, 40, 80, 160 — but capped at 15.
    assert delays == [10.0, 15.0, 15.0, 15.0, 15.0]


async def test_zero_retries_rejected() -> None:
    async def fn() -> None:
        return None

    with pytest.raises(ValueError, match="retries must be >= 1"):
        await async_with_retry(fn, retries=0)
