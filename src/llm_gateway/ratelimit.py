"""Per-tenant sliding-window rate limiting in Redis (requests per minute).

A fixed one-minute window lets a client send its whole limit at 0:59 and again at 1:00,
twice the limit in two seconds. This uses the sliding-window counter instead: one counter
per minute window (`INCR` + `EXPIRE`, shared across all gateway replicas), and the count
for the last 60 s is estimated as

    current window + previous window x (share of the previous window still in the last 60 s)

which assumes the previous window's requests were spread evenly. Rejected requests are
not counted, so a client that keeps retrying is let back in as soon as its rate drops.

If Redis is unreachable the limiter fails open: requests are allowed and the error is
logged.
"""

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    # Rejected: seconds until one more request fits. Allowed: until the window rolls over.
    reset_seconds: int

    def headers(self) -> dict[str, str]:
        """The de facto standard `X-RateLimit-*` headers (as GitHub and OpenAI send)."""
        if self.limit <= 0:
            return {}
        return {
            "x-ratelimit-limit": str(self.limit),
            "x-ratelimit-remaining": str(self.remaining),
            "x-ratelimit-reset": str(self.reset_seconds),
        }


class RateLimiter:
    def __init__(
        self,
        redis: Redis,
        limit_per_minute: int,
        clock: Callable[[], float] = time.time,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.redis = redis
        self.limit = limit_per_minute
        self.clock = clock
        self.on_error = on_error or (lambda op: None)

    async def hit(self, tenant: str) -> RateLimitResult:
        if self.limit <= 0:
            return RateLimitResult(True, 0, 0, 0)
        now = self.clock()
        window = int(now // WINDOW_SECONDS)
        into_window = (now % WINDOW_SECONDS) / WINDOW_SECONDS
        previous_weight = 1.0 - into_window
        key = f"llmgw:rl:{tenant}:{window}"
        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.incr(key)
                # Kept for two windows: the next window reads it as its previous count.
                pipe.expire(key, 2 * WINDOW_SECONDS, nx=True)
                pipe.get(f"llmgw:rl:{tenant}:{window - 1}")
                current, _, previous = await pipe.execute()
            previous = int(previous or 0)
            estimate = current + previous * previous_weight
            if estimate > self.limit:
                await self.redis.decr(key)  # rejected requests don't count
        except (RedisError, OSError) as exc:
            self.on_error("rate_limit")
            logger.warning("rate limiter unavailable, allowing request: %s", exc)
            return RateLimitResult(True, self.limit, self.limit, WINDOW_SECONDS)
        if estimate > self.limit:
            return RateLimitResult(
                False, self.limit, 0, _retry_after(current - 1, previous, into_window, self.limit)
            )
        remaining = max(int(self.limit - estimate), 0)
        reset = math.ceil((1 - into_window) * WINDOW_SECONDS)  # when this window rolls over
        return RateLimitResult(True, self.limit, remaining, max(1, reset))


def _retry_after(current: int, previous: int, into_window: float, limit: int) -> int:
    """Seconds until one more request fits, i.e. until current + 1 + previous x weight <= limit.

    `current` is this window's admitted count, `previous` the previous window's.
    """
    room = limit - 1 - current
    if room >= 0:  # wait for the previous window's weight to fall to room / previous
        seconds = (1 - into_window - room / previous) * WINDOW_SECONDS
    else:  # this window alone is full: wait into the next one, where it is the previous
        seconds = (1 - into_window + 1 - (limit - 1) / current) * WINDOW_SECONDS
    return max(1, math.ceil(seconds))
