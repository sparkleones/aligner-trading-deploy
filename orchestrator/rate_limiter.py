"""Async token-bucket rate limiter for SEBI OPS compliance (max 9 OPS)."""

import asyncio
import time


class AsyncRateLimiter:
    """Token-bucket rate limiter capped at max_ops per second."""

    def __init__(self, max_ops: int = 9, window_seconds: float = 1.0):
        self.max_ops = max_ops
        self.window = window_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        async with self._lock:
            now = time.monotonic()
            # Purge old timestamps outside the window
            self._timestamps = [
                t for t in self._timestamps if now - t < self.window
            ]
            if len(self._timestamps) >= self.max_ops:
                wait = self._timestamps[0] + self.window - now
                if wait > 0:
                    await asyncio.sleep(wait)
            self._timestamps.append(time.monotonic())

    @property
    def current_usage(self) -> int:
        now = time.monotonic()
        return sum(1 for t in self._timestamps if now - t < self.window)
