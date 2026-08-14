import asyncio
import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")

_TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")


def with_retry(fn: Callable[[], T], attempts: int = 3) -> T:
    """Call fn(), retrying transient API failures with exponential backoff.

    Retries only on transient errors (503/429/UNAVAILABLE/RESOURCE_EXHAUSTED).
    Non-transient errors (bad model, auth) raise immediately. Shared by all
    agents so resilience is consistent across the system.
    """
    last_err = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if any(marker in str(e) for marker in _TRANSIENT_MARKERS):
                time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff
                continue
            raise
    raise RuntimeError(f"Failed after {attempts} retries: {last_err}")


async def with_retry_async(fn: Callable[[], Awaitable[T]], attempts: int = 3) -> T:
    """Async twin of with_retry. Awaits fn(), retrying transient failures
    with exponential backoff using a non-blocking async sleep."""
    last_err = None
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as e:
            last_err = e
            if any(marker in str(e) for marker in _TRANSIENT_MARKERS):
                await asyncio.sleep(2 ** attempt)  # non-blocking
                continue
            raise
    raise RuntimeError(f"Failed after {attempts} retries: {last_err}")