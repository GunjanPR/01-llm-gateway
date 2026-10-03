"""End-to-end tests against the real stack: a running gateway, Redis and Ollama.

Run with `make test-integration` after `make up` and `make run`. Configure with:
    GATEWAY_URL (default http://127.0.0.1:8000), GATEWAY_TEST_KEY (one of GATEWAY_API_KEYS),
    GATEWAY_TEST_MODEL (default llama3.2), GATEWAY_TEST_EMBED_MODEL (default nomic-embed-text)
"""

import json
import os
import uuid

import httpx
import pytest

pytestmark = pytest.mark.integration

URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000")
KEY = os.environ.get("GATEWAY_TEST_KEY", "")
MODEL = os.environ.get("GATEWAY_TEST_MODEL", "llama3.2")
EMBED_MODEL = os.environ.get("GATEWAY_TEST_EMBED_MODEL", "nomic-embed-text")


@pytest.fixture(scope="module")
def client():
    if not KEY:
        pytest.fail("set GATEWAY_TEST_KEY to one of the gateway's GATEWAY_API_KEYS")
    with httpx.Client(
        base_url=URL, headers={"Authorization": f"Bearer {KEY}"}, timeout=120
    ) as client:
        yield client


def chat(prompt: str, **extra) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0,
        **extra,
    }


def test_health_sees_redis_and_a_backend(client):
    body = client.get("/health").json()
    assert body["redis"] is True
    assert any(b["reachable"] for b in body["backends"])
    assert body["status"] in {"ok", "degraded"}


def test_identical_concurrent_requests_are_coalesced():
    """Two identical requests at once on a cold cache: one generation, shared."""
    import asyncio

    body = chat(f"Reply with the single word OK. ({uuid.uuid4()})")

    async def both():
        async with httpx.AsyncClient(
            base_url=URL, headers={"Authorization": f"Bearer {KEY}"}, timeout=120
        ) as c:
            return await asyncio.gather(
                *(c.post("/v1/chat/completions", json=body) for _ in range(2))
            )

    responses = asyncio.run(both())
    assert [r.status_code for r in responses] == [200, 200]
    assert sorted(r.headers["x-gateway-cache"] for r in responses) == ["COALESCED", "MISS"]
    assert responses[0].json() == responses[1].json()


def test_client_disconnect_frees_the_backend():
    """A client that gives up must not leave its call running on the backend."""
    body = chat(f"Write a long story about a lighthouse. ({uuid.uuid4()})", max_tokens=400)
    with pytest.raises(httpx.ReadTimeout), httpx.Client(timeout=0.3) as quick:
        quick.post(
            f"{URL}/v1/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {KEY}", "Cache-Control": "no-cache"},
        )
    import time

    time.sleep(0.5)
    health = httpx.get(f"{URL}/health", timeout=10).json()
    assert all(b["inflight"] == 0 for b in health["backends"])


def test_real_completion_then_cache_hit(client):
    body = chat(f"Reply with the single word OK. ({uuid.uuid4()})")
    first = client.post("/v1/chat/completions", json=body)
    assert first.status_code == 200, first.text
    assert first.json()["choices"][0]["message"]["content"].strip()
    assert first.json()["usage"]["completion_tokens"] > 0
    assert first.headers["x-gateway-cache"] == "MISS"
    second = client.post("/v1/chat/completions", json=body)
    assert second.headers["x-gateway-cache"] == "HIT"
    assert float(second.headers["x-gateway-latency-ms"]) < float(
        first.headers["x-gateway-latency-ms"]
    )


def test_real_streaming(client):
    body = chat("Count from 1 to 5.", stream=True)
    text = []
    with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data:") and not line.endswith("[DONE]"):
                for choice in json.loads(line[5:]).get("choices", []):
                    text.append(choice.get("delta", {}).get("content") or "")
    assert "".join(text).strip()


def test_real_embeddings(client):
    resp = client.post("/v1/embeddings", json={"model": EMBED_MODEL, "input": "gene expression"})
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["data"][0]["embedding"]) > 100


def test_metrics_exposed(client):
    text = client.get("/metrics").text
    assert "llmgw_requests_total" in text
    assert "llmgw_ttft_seconds_bucket" in text


def test_real_guardrails_redact_before_the_model(client):
    body = chat("My email is jane.doe@example.com. Repeat my email back to me exactly.")
    resp = client.post("/v1/chat/completions", json=body, headers={"Cache-Control": "no-cache"})
    assert resp.status_code == 200, resp.text
    assert "pii_email=1" in resp.headers["x-gateway-guardrails"]
    assert "jane.doe@example.com" not in resp.json()["choices"][0]["message"]["content"]


def test_real_semantic_cache_reuses_a_paraphrase(client):
    session = uuid.uuid4().hex  # fresh conversation scope so earlier runs can't interfere
    system = {"role": "system", "content": f"Session {session}."}

    def ask(text: str) -> httpx.Response:
        body = chat(text)
        body["messages"] = [system, *body["messages"]]
        return client.post("/v1/chat/completions", json=body)

    first = ask("What is the capital of Italy?")
    assert first.headers["x-gateway-cache"] == "MISS"
    second = ask("Which city is the capital of Italy?")
    assert second.headers["x-gateway-cache"] == "SEMANTIC_HIT", second.headers
    assert second.json() == first.json()
    third = ask("What is the capital of Spain?")
    assert third.headers["x-gateway-cache"] == "MISS"  # near-miss must not reuse Italy


def test_real_usage_ledger_records_requests(client):
    before = client.get("/v1/usage", params={"days": 1})
    assert before.status_code == 200, before.text
    body = chat(f"Reply OK. ({uuid.uuid4()})")
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.headers["x-request-id"]
    after = client.get("/v1/usage", params={"days": 1}).json()
    assert after["totals"]["requests"] == before.json()["totals"]["requests"] + 1
    assert after["totals"]["completion_tokens"] > before.json()["totals"]["completion_tokens"]
