import fakeredis
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from llm_gateway.cache import ResponseCache, cache_key, is_cacheable
from llm_gateway.ratelimit import RateLimiter


def test_cache_key_ignores_transport_fields_and_key_order():
    a = {"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": False}
    b = {"messages": [{"role": "user", "content": "x"}], "model": "m", "user": "u1"}
    assert cache_key(a) == cache_key(b)
    assert cache_key(a) != cache_key({**a, "temperature": 0.5})


def test_cacheable_modes():
    assert is_cacheable({"temperature": 1}, "all")
    assert not is_cacheable({"stream": True}, "all")
    assert not is_cacheable({"temperature": 1}, "deterministic")
    assert is_cacheable({"temperature": 0}, "deterministic")
    assert is_cacheable({"seed": 1}, "deterministic")
    assert not is_cacheable({}, "off")


@pytest.mark.anyio
async def test_cache_roundtrip_with_ttl():
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    cache = ResponseCache(redis, ttl_seconds=30)
    await cache.set("k", {"a": 1})
    assert await cache.get("k") == {"a": 1}
    assert 0 < await redis.ttl("k") <= 30
    assert await cache.get("missing") is None


@pytest.mark.anyio
async def test_rate_limiter_counts_per_tenant_and_window():
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    now = [120.0]
    limiter = RateLimiter(redis, limit_per_minute=2, clock=lambda: now[0])
    assert [(await limiter.hit("t1")).allowed for _ in range(3)] == [True, True, False]
    assert (await limiter.hit("t2")).allowed
    now[0] = 180.0  # next window
    assert (await limiter.hit("t1")).allowed


class DownRedis:
    """Stands in for an unreachable Redis server."""

    def pipeline(self, *args, **kwargs):
        raise RedisConnectionError("connection refused")

    async def get(self, *args, **kwargs):
        raise RedisConnectionError("connection refused")

    async def set(self, *args, **kwargs):
        raise RedisConnectionError("connection refused")


@pytest.mark.anyio
async def test_redis_outage_fails_open():
    assert (await RateLimiter(DownRedis(), limit_per_minute=1).hit("t")).allowed
    cache = ResponseCache(DownRedis(), ttl_seconds=1)
    assert await cache.get("k") is None
    await cache.set("k", {"a": 1})  # does not raise
