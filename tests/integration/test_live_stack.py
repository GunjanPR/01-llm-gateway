"""End-to-end tests against the real stack: a running gateway, Redis and Ollama.

Run with `make test-integration` after `make up` and `make run`. Configure with:
    GATEWAY_URL (default http://localhost:8000), GATEWAY_TEST_KEY (one of GATEWAY_API_KEYS),
    GATEWAY_TEST_MODEL (default llama3.2), GATEWAY_TEST_EMBED_MODEL (default nomic-embed-text)
"""

import json
import os
import uuid

import httpx
import pytest

pytestmark = pytest.mark.integration

URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")
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
