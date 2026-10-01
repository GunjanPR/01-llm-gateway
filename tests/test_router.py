import httpx
import pytest

from conftest import Upstream, fail, make_backend, ok
from llm_gateway.backends import BackendError
from llm_gateway.router import NoBackendAvailable, Router

CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def names(backends):
    return [b.name for b in backends]


def test_cost_policy_prefers_cheapest():
    cheap = make_backend("cheap", Upstream(ok()), cost_in=0.0)
    pricey = make_backend("pricey", Upstream(ok()), cost_in=1.0)
    assert names(Router([pricey, cheap], policy="cost").candidates("m")) == ["cheap", "pricey"]


def test_quality_policy_prefers_best():
    low = make_backend("low", Upstream(ok()), quality=0.2)
    high = make_backend("high", Upstream(ok()), quality=0.9)
    assert names(Router([low, high], policy="quality").candidates("m")) == ["high", "low"]


def test_latency_policy_uses_observed_latency():
    a, b = make_backend("a", Upstream(ok())), make_backend("b", Upstream(ok()))
    router = Router([a, b], policy="latency")
    router.record_success(a, 2.0)
    router.record_success(b, 0.5)
    assert names(router.candidates("m")) == ["b", "a"]


def test_only_backends_serving_the_model():
    a = make_backend("a", Upstream(ok()), models={"x": "x"})
    b = make_backend("b", Upstream(ok()))
    assert names(Router([a, b]).candidates("y")) == ["b"]


def test_circuit_breaker_opens_and_recovers():
    clock = Clock()
    a, b = make_backend("a", Upstream(ok())), make_backend("b", Upstream(ok()))
    router = Router([a, b], breaker_failures=2, breaker_cooldown_seconds=10, clock=clock)
    for _ in range(2):
        router.record_failure(a, BackendError("a", "boom", 500))
    assert not router.is_available(a)
    assert names(router.candidates("m")) == ["b", "a"]  # open breakers go last
    clock.now = 11
    assert router.is_available(a)


@pytest.mark.anyio
async def test_run_retries_then_falls_back():
    bad, good = Upstream(fail(502)), Upstream(ok())
    router = Router(
        [make_backend("bad", bad), make_backend("good", good)],
        max_retries=2,
        retry_backoff_seconds=0,
    )
    _, backend = await router.run("m", lambda b: b.chat(CHAT))
    assert backend.name == "good"
    assert len(bad.requests) == 3


@pytest.mark.anyio
async def test_run_skips_backend_once_breaker_opens():
    bad = Upstream(fail(500))
    router = Router(
        [make_backend("bad", bad), make_backend("good", Upstream(ok()))],
        max_retries=5,
        retry_backoff_seconds=0,
        breaker_failures=2,
    )
    await router.run("m", lambda b: b.chat(CHAT))
    assert len(bad.requests) == 2


@pytest.mark.anyio
async def test_run_raises_when_everything_fails():
    router = Router([make_backend("a", Upstream(fail(500)))], max_retries=0)
    with pytest.raises(NoBackendAvailable) as info:
        await router.run("m", lambda b: b.chat(CHAT))
    assert info.value.errors[0].status_code == 500


@pytest.mark.anyio
async def test_connection_errors_are_backend_errors():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    backend = make_backend("down", Upstream(refuse))
    with pytest.raises(BackendError) as info:
        await backend.chat(CHAT)
    assert info.value.status_code is None
    assert info.value.kind == "connection"
