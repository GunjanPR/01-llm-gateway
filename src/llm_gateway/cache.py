"""Exact-match response cache in Redis.

The key is a SHA-256 of the canonical request, so identical requests (same model,
messages and sampling parameters) return the stored response without a model call.
Redis failures never fail a request: the cache is skipped and the error is logged.
"""

import hashlib
import json
import logging
from collections.abc import Callable
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

KEY_PREFIX = "llmgw:cache:"
# Fields that change transport, not the answer.
IGNORED_FIELDS = {"stream", "stream_options", "user"}


def cache_key(payload: dict[str, Any], namespace: str = "") -> str:
    """Key for a request; `namespace` (e.g. the tenant id) keeps entries separate."""
    relevant = {k: v for k, v in payload.items() if k not in IGNORED_FIELDS}
    relevant["__namespace__"] = namespace
    canonical = json.dumps(relevant, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return KEY_PREFIX + hashlib.sha256(canonical.encode()).hexdigest()


def is_cacheable(payload: dict[str, Any], mode: str) -> bool:
    if mode == "off" or payload.get("stream"):
        return False
    if mode == "deterministic":
        return payload.get("temperature") == 0 or "seed" in payload
    return True


class ResponseCache:
    def __init__(
        self,
        redis: Redis,
        ttl_seconds: int,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.redis = redis
        self.ttl_seconds = ttl_seconds
        self.on_error = on_error or (lambda op: None)

    async def get(self, key: str) -> dict[str, Any] | None:
        try:
            raw = await self.redis.get(key)
        except (RedisError, OSError) as exc:
            self.on_error("cache_get")
            logger.warning("cache get failed, skipping cache: %s", exc)
            return None
        if not raw:
            return None
        try:
            entry = json.loads(raw)
        except ValueError:
            entry = None
        if not isinstance(entry, dict):
            self.on_error("cache_corrupt")
            logger.warning("dropping unreadable cache entry %s", key)
            return None
        return entry

    async def set(self, key: str, value: dict[str, Any]) -> None:
        try:
            await self.redis.set(key, json.dumps(value), ex=self.ttl_seconds)
        except (RedisError, OSError) as exc:
            self.on_error("cache_set")
            logger.warning("cache set failed: %s", exc)
