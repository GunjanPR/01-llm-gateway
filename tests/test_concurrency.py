"""Concurrent load through the whole app: request coalescing, concurrency caps, and
counters that stay consistent when hundreds of requests overlap."""

import asyncio
import json

import httpx
import pytest

from conftest import API_KEY, chat_response
from llm_gateway.backends import OpenAICompatibleBackend
from llm_gateway.config import BackendConfig
from llm_gateway.main import create_app

pytestmark = pytest.mark.anyio
AUTH = {"Authorization": f"Bearer {API_KEY}"}


class SlowUpstream:
    """Async scripted backend: tracks concurrent calls; can fail the first N calls."""

    def __init__(self, delay: float = 0.02, fail_first: int = 0) -> None:
        self.delay = delay
        self.fail_first = fail_first
        self.calls = 0
        self.active = 0
        self.peak = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            if self.calls <= self.fail_first:
                return httpx.Response(500, json={"error": {"message": "boom"}})
            body = json.loads(request.content)
            return httpx.Response(200, json=chat_response(body["messages"][0]["content"]))
        finally:
            self.active -= 1


def _app(settings, redis, upstream: SlowUpstream, **overrides):
    backend = OpenAICompatibleBackend(
        BackendConfig(name="slow", base_url="http://slow.test/v1"),
        transport=httpx.MockTransport(upstream),
    )
    return create_app(settings.model_copy(update=overrides), backends=[backend], redis=redis)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")


def _chat(text: str) -> dict:
    return {"model": "llama3.2", "messages": [{"role": "user", "content": text}]}


async def test_identical_concurrent_requests_share_one_backend_call(settings, redis):
    upstream = SlowUpstream(delay=0.05)
    async with _client(_app(settings, redis, upstream)) as client:
        responses = await asyncio.gather(
            *(
                client.post("/v1/chat/completions", json=_chat("same"), headers=AUTH)
                for _ in range(20)
            )
        )
        metrics = (await client.get("/metrics")).text
    assert upstream.calls == 1
    assert all(r.status_code == 200 for r in responses)
    states = sorted(r.headers["x-gateway-cache"] for r in responses)
    assert states == ["COALESCED"] * 19 + ["MISS"]
    assert len({r.json()["choices"][0]["message"]["content"] for r in responses}) == 1
    assert 'llmgw_cache_requests_total{result="coalesced"} 19.0' in metrics


async def test_followers_answer_themselves_when_the_leader_fails(settings, redis):
    upstream = SlowUpstream(delay=0.05, fail_first=1)
    app = _app(settings, redis, upstream, backend_max_retries=0)
    async with _client(app) as client:
        responses = await asyncio.gather(
            *(
                client.post("/v1/chat/completions", json=_chat("same"), headers=AUTH)
                for _ in range(5)
            )
        )
    codes = sorted(r.status_code for r in responses)
    assert codes == [200, 200, 200, 200, 502]  # only the leader saw the failure


async def test_hundreds_of_overlapping_requests_respect_the_cap(settings, redis):
    upstream = SlowUpstream(delay=0.01)
    app = _app(settings, redis, upstream, backend_max_concurrency=8)
    async with _client(app) as client:
        responses = await asyncio.gather(
            *(
                client.post("/v1/chat/completions", json=_chat(f"q{i}"), headers=AUTH)
                for i in range(300)
            )
        )
        metrics = (await client.get("/metrics")).text
    assert [r.status_code for r in responses] == [200] * 300
    assert upstream.calls == 300 and upstream.peak <= 8
    assert app.state.router.state["slow"].inflight == 0
    assert 'llmgw_requests_total{model="llama3.2",outcome="ok",route="chat"} 300.0' in metrics
    assert "llmgw_inflight_requests 1.0" in metrics  # only the /metrics request itself
    # Every reply belongs to its own request: no cross-talk under concurrency.
    for i, r in enumerate(responses):
        assert r.json()["choices"][0]["message"]["content"] == f"q{i}"
