"""Per-tenant fixed-window rate limiting in Redis (requests per minute).

Each tenant gets one counter per minute window (`INCR` + `EXPIRE`), so the limit is
shared across all gateway replicas. If Redis is unreachable the limiter fails open:
requests are allowed and the error is logged.
"""

import logging
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
    reset_seconds: int


class RateLimiter:
    def __init__(
        self, redis: Redis, limit_per_minute: int, clock: Callable[[], float] = time.time
    ) -> None:
        self.redis = redis
        self.limit = limit_per_minute
        self.clock = clock

    async def hit(self, tenant: str) -> RateLimitResult:
        if self.limit <= 0:
            return RateLimitResult(True, 0, 0, 0)
        now = self.clock()
        window = int(now // WINDOW_SECONDS)
        reset = WINDOW_SECONDS - int(now % WINDOW_SECONDS)
        key = f"llmgw:rl:{tenant}:{window}"
        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.incr(key)
                pipe.expire(key, WINDOW_SECONDS, nx=True)
                count, _ = await pipe.execute()
        except RedisError as exc:
            logger.warning("rate limiter unavailable, allowing request: %s", exc)
            return RateLimitResult(True, self.limit, self.limit, reset)
        return RateLimitResult(count <= self.limit, self.limit, max(self.limit - count, 0), reset)
