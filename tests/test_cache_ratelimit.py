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
    now[0] = 240.0  # two windows later: nothing left in the last 60 s
    assert (await limiter.hit("t1")).allowed


@pytest.mark.anyio
async def test_rate_limiter_blocks_the_burst_across_a_window_boundary():
    """A fixed window admits the limit at 0:59 and again at 1:00 (2x in two seconds)."""
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    now = [119.0]
    limiter = RateLimiter(redis, limit_per_minute=10, clock=lambda: now[0])
    assert sum([(await limiter.hit("t")).allowed for _ in range(10)]) == 10
    now[0] = 121.0  # the next window has just started
    admitted = [(await limiter.hit("t")).allowed for _ in range(10)]
    assert sum(admitted) == 0
    rejected = await limiter.hit("t")
    # 10 x (1 - 1/60) = 9.83 still counts, so the next request fits once that is <= 9:
    # at 6 s into the window, 5 s from now.
    assert not rejected.allowed and rejected.reset_seconds == 5
    now[0] = 126.0
    assert (await limiter.hit("t")).allowed


@pytest.mark.anyio
async def test_rate_limiter_steady_rate_over_many_windows():
    """At twice the limit, about the limit gets through per minute, and never 2x."""
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    now = [0.0]
    limiter = RateLimiter(redis, limit_per_minute=30, clock=lambda: now[0])
    admitted_at = []
    for i in range(60 * 10):  # one request per second for 10 minutes
        now[0] = float(i)
        if (await limiter.hit("t")).allowed:
            admitted_at.append(now[0])
    worst = max(sum(1 for t in admitted_at if s <= t < s + 60) for s in range(0, 540))
    assert worst <= 31  # any 60 s span, not just aligned windows
    assert len(admitted_at) >= 30 * 9


@pytest.mark.anyio
async def test_rate_limit_headers():
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    limiter = RateLimiter(redis, limit_per_minute=5, clock=lambda: 130.0)
    result = await limiter.hit("t")
    assert result.headers() == {
        "x-ratelimit-limit": "5",
        "x-ratelimit-remaining": "4",
        "x-ratelimit-reset": "50",
    }
    assert (await RateLimiter(redis, 0).hit("t")).headers() == {}  # limiter off


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
