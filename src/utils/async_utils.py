"""
Async Utilities — Semaphore, gather helpers for concurrent processing.
"""

import asyncio
from typing import Any, Callable, Coroutine, List, TypeVar

from loguru import logger

T = TypeVar("T")


async def gather_with_semaphore(
    tasks: List[Coroutine],
    max_concurrent: int = 4,
    return_exceptions: bool = True,
) -> List[Any]:
    """
    Run async tasks with a concurrency limit.

    Args:
        tasks: List of coroutines to execute.
        max_concurrent: Maximum number of concurrent tasks.
        return_exceptions: Whether to return exceptions instead of raising.

    Returns:
        List of results in the same order as tasks.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(
        *[limited_task(task) for task in tasks],
        return_exceptions=return_exceptions,
    )


async def retry_async(
    func: Callable[..., Coroutine],
    *args,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    **kwargs,
) -> Any:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry.
        max_retries: Maximum number of retries.
        delay: Initial delay between retries in seconds.
        backoff: Multiplier for delay after each retry.

    Returns:
        Function result.
    """
    last_error = None
    current_delay = delay

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {current_delay:.1f}s..."
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(f"All {max_retries + 1} attempts failed: {e}")

    raise last_error


class ProgressTracker:
    """Track progress across async tasks."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.completed = 0
        self.description = description
        self._lock = asyncio.Lock()

    async def increment(self):
        """Increment completed count."""
        async with self._lock:
            self.completed += 1
            logger.info(
                f"{self.description}: {self.completed}/{self.total} "
                f"({self.completed/self.total*100:.0f}%)"
            )

    @property
    def progress(self) -> float:
        """Current progress as a fraction (0.0-1.0)."""
        return self.completed / self.total if self.total > 0 else 0.0
