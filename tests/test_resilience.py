"""Failure handling and graceful degradation: load shedding, bad upstream replies,
oversized requests, Redis outages, crashes, health status and request ids."""

import asyncio
import hashlib
import json
import logging
import uuid
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from redis.asyncio import Redis
from starlette.requests import Request

from conftest import AUTH, Upstream, fail, make_backend, ok
from llm_gateway.cache import cache_key
from llm_gateway.config import BackendConfig
from llm_gateway.logs import JsonFormatter
from llm_gateway.main import ClientDisconnected, create_app, until_disconnect
from llm_gateway.router import Overloaded, Router

CHAT = {"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]}


class GatedBackend:
    """A backend whose calls block until the test releases them."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.config = BackendConfig(name=name, base_url=f"http://{name}/v1")
        self.gate = asyncio.Event()
        self.calls = 0

    def serves(self, model: str) -> bool:
        return True

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        await self.gate.wait()
        return {"from": self.name}

    async def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def chat_stream(self, payload: dict[str, Any]):
        self.calls += 1
        await self.gate.wait()
        for line in ["data: a", "data: b", "data: [DONE]"]:
            yield line

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


# ---- router: per-backend concurrency caps ----


@pytest.mark.anyio
async def test_saturated_backend_spills_over_to_the_next():
    a, b = GatedBackend("a"), GatedBackend("b")
    router = Router([a, b], max_concurrency=1, queue_timeout_seconds=1)
    first = asyncio.create_task(router.run("m", lambda x: x.chat({})))
    await _settle()
    b.gate.set()
    _, used = await router.run("m", lambda x: x.chat({}))
    assert used.name == "b"  # "a" was full, so the request went to "b" instead of queueing
    a.gate.set()
    assert (await first)[1].name == "a"
    assert router.state["a"].inflight == router.state["b"].inflight == 0


@pytest.mark.anyio
async def test_queues_for_a_slot_then_proceeds():
    a = GatedBackend("a")
    waits: list[float] = []
    router = Router([a], max_concurrency=1, queue_timeout_seconds=5, on_queue_wait=waits.append)
    first = asyncio.create_task(router.run("m", lambda x: x.chat({})))
    second = asyncio.create_task(router.run("m", lambda x: x.chat({})))
    await _settle()
    assert a.calls == 1  # the second request is waiting, not running
    a.gate.set()
    await asyncio.gather(first, second)
    assert a.calls == 2 and len(waits) == 1


@pytest.mark.anyio
async def test_overloaded_after_queue_timeout():
    a = GatedBackend("a")
    waits: list[float] = []
    router = Router([a], max_concurrency=1, queue_timeout_seconds=0.05, on_queue_wait=waits.append)
    first = asyncio.create_task(router.run("m", lambda x: x.chat({})))
    await _settle()
    with pytest.raises(Overloaded):
        await router.run("m", lambda x: x.chat({}))
    assert waits and waits[0] >= 0.05
    a.gate.set()
    await first


@pytest.mark.anyio
async def test_stream_holds_its_slot_until_closed():
    a = GatedBackend("a")
    a.gate.set()
    router = Router([a], max_concurrency=1)
    _, lines, _ = await router.open_stream("m", {})
    assert router.state["a"].inflight == 1
    assert [line async for line in lines] == ["data: a", "data: b", "data: [DONE]"]
    assert router.state["a"].inflight == 0


@pytest.mark.anyio
async def test_slot_is_freed_when_a_call_fails():
    router = Router([make_backend("a", Upstream(fail(500)))], max_retries=0, max_concurrency=1)
    with pytest.raises(Exception):  # noqa: B017 - NoBackendAvailable
        await router.run("m", lambda b: b.chat({"model": "m", "messages": []}))
    assert router.state["a"].inflight == 0


# ---- API ----


def test_load_shedding_returns_503_with_retry_after(client_factory):
    router_full = GatedBackend("a")
    client = client_factory([router_full], backend_max_concurrency=1)
    client.app.state.router.state["a"].inflight = 1  # simulate a request holding the slot
    client.app.state.router.queue_timeout_seconds = 0.01
    resp = client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "1"
    assert resp.json()["error"]["type"] == "overloaded_error"
    metrics = client.get("/metrics").text
    assert 'llmgw_load_shed_total{route="chat"} 1.0' in metrics
    assert "llmgw_backend_queue_wait_seconds_count 1.0" in metrics


def test_non_json_upstream_reply_is_a_502_and_falls_back(client_factory):
    def html(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>proxy error</html>")

    client = client_factory(
        [make_backend("broken", Upstream(html)), make_backend("good", Upstream(ok()))]
    )
    resp = client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
    assert resp.status_code == 200 and resp.headers["x-gateway-backend"] == "good"

    only_broken = client_factory([make_backend("broken", Upstream(html))])
    no_cache = {**AUTH, "Cache-Control": "no-cache"}
    resp = only_broken.post("/v1/chat/completions", json=CHAT, headers=no_cache)
    assert resp.status_code == 502
    assert "invalid JSON" in resp.json()["error"]["message"]


def test_oversized_request_is_rejected_before_parsing(client_factory):
    upstream = Upstream(ok())
    client = client_factory([make_backend("a", upstream)], max_request_bytes=1000)
    big = {**CHAT, "messages": [{"role": "user", "content": "x" * 2000}]}
    resp = client.post("/v1/chat/completions", json=big, headers=AUTH)
    assert resp.status_code == 413
    assert not upstream.requests


def test_non_utf8_body_is_a_400(client_factory):
    client = client_factory([make_backend("a", Upstream(ok()))])
    resp = client.post("/v1/chat/completions", content=b"\xff\xfe{", headers=AUTH)
    assert resp.status_code == 400


def test_redis_outage_fails_open_and_is_counted(settings):
    dead = Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2)
    app = create_app(settings, backends=[make_backend("a", Upstream(ok()))], redis=dead)
    with TestClient(app) as dead_client:
        resp = dead_client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
        assert resp.status_code == 200  # served without cache or rate limiting
        metrics = dead_client.get("/metrics").text
        health = dead_client.get("/health").json()
    assert 'llmgw_redis_errors_total{op="cache_get"} 1.0' in metrics
    assert 'llmgw_redis_errors_total{op="rate_limit"} 1.0' in metrics
    assert health["status"] == "degraded" and health["redis"] is False


def test_corrupt_cache_entry_is_a_miss(client_factory, redis):
    upstream = Upstream(ok())
    client = client_factory([make_backend("a", upstream)])
    tenant = hashlib.sha256(b"test-key").hexdigest()[:12]
    asyncio.run(redis.set(cache_key({**CHAT}, tenant), "not json{"))
    resp = client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
    assert resp.status_code == 200 and resp.headers["x-gateway-cache"] == "MISS"
    assert 'llmgw_redis_errors_total{op="cache_corrupt"} 1.0' in client.get("/metrics").text


def test_unhandled_error_is_a_json_500_with_request_id(client_factory):
    class Exploding(GatedBackend):
        async def chat(self, payload):
            raise RuntimeError("bug")

    client = client_factory([Exploding("a")])
    client_raise = type(client)(client.app, raise_server_exceptions=False)
    resp = client_raise.post("/v1/chat/completions", json=CHAT, headers=AUTH)
    assert resp.status_code == 500
    body = resp.json()["error"]
    assert body["type"] == "internal_error"
    assert resp.headers["x-request-id"] in body["message"]


def test_health_status_reflects_backends(client_factory):
    client = client_factory(
        [make_backend("up", Upstream(ok())), make_backend("down", Upstream(fail(503)))]
    )
    health = client.get("/health").json()
    assert health["status"] == "degraded"
    assert {b["name"]: b["reachable"] for b in health["backends"]} == {"up": True, "down": False}
    assert 'llmgw_backend_reachable{backend="down"} 0.0' in client.get("/metrics").text

    all_down = client_factory([make_backend("down", Upstream(fail(503)))])
    assert all_down.get("/health").json()["status"] == "down"


def test_client_request_id_is_echoed_when_it_is_a_uuid(client_factory):
    client = client_factory([make_backend("a", Upstream(ok()))])
    rid = str(uuid.uuid4())
    resp = client.get("/livez", headers={"X-Request-ID": rid})
    assert resp.headers["x-request-id"] == rid
    other = client.get("/livez", headers={"X-Request-ID": "not-a-uuid"})
    assert other.headers["x-request-id"] != "not-a-uuid"


def test_http_responses_counted_by_route_template(client_factory):
    client = client_factory([make_backend("a", Upstream(ok()))])
    client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
    client.post("/v1/chat/completions", json=CHAT)  # 401
    client.get("/nope")
    metrics = client.get("/metrics").text
    assert 'llmgw_http_responses_total{path="/v1/chat/completions",status="2xx"} 1.0' in metrics
    assert 'llmgw_http_responses_total{path="/v1/chat/completions",status="4xx"} 1.0' in metrics
    assert 'llmgw_http_responses_total{path="unmatched",status="4xx"} 1.0' in metrics
    assert 'llmgw_build_info{version="' in metrics


def test_access_log_has_structured_fields(client_factory, caplog):
    client = client_factory([make_backend("a", Upstream(ok()))])
    with caplog.at_level(logging.INFO, logger="llm_gateway.access"):
        resp = client.post("/v1/chat/completions", json=CHAT, headers=AUTH)
    record = next(r for r in caplog.records if r.name == "llm_gateway.access")
    line = json.loads(JsonFormatter().format(record))
    assert line["request_id"] == resp.headers["x-request-id"]
    assert line["status"] == 200 and line["model"] == "llama3.2" and line["backend"] == "a"
    assert line["cache"] == "MISS" and "latency_ms" in line


# ---- client disconnects ----


def _request(disconnect_after: float | None):
    async def receive():
        if disconnect_after is None:
            await asyncio.Event().wait()  # never disconnects
        await asyncio.sleep(disconnect_after)
        return {"type": "http.disconnect"}

    return Request({"type": "http", "method": "POST", "headers": []}, receive)


@pytest.mark.anyio
async def test_disconnect_cancels_the_backend_call_and_frees_the_slot():
    a = GatedBackend("a")  # never answers unless released
    router = Router([a], max_concurrency=1)
    with pytest.raises(ClientDisconnected):
        await until_disconnect(_request(0.01), router.run("m", lambda b: b.chat({})))
    assert router.state["a"].inflight == 0


@pytest.mark.anyio
async def test_result_is_returned_when_the_client_stays():
    a = GatedBackend("a")
    a.gate.set()
    result, _ = await until_disconnect(_request(None), Router([a]).run("m", lambda b: b.chat({})))
    assert result == {"from": "a"}
