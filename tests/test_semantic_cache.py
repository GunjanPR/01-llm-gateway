import json
import math

import httpx
from redis.exceptions import ResponseError

from conftest import AUTH, Upstream, chat_response, make_backend
from llm_gateway.semantic_cache import QUERY_PREFIX, VERIFY_SYSTEM, split_request


class MemoryIndex:
    """In-process stand-in for RedisVectorIndex (brute-force cosine)."""

    def __init__(self, fail: bool = False) -> None:
        self.entries: list[tuple[str, list[float], dict]] = []
        self.fail = fail

    async def search(self, scope, vector):
        if self.fail:
            raise ResponseError("unknown command 'FT.SEARCH'")
        best = None
        for s, v, entry in self.entries:
            if s != scope:
                continue
            sim = sum(a * b for a, b in zip(v, vector, strict=True)) / (
                math.hypot(*v) * math.hypot(*vector)
            )
            if best is None or sim > best[0]:
                best = (sim, entry)
        return best

    async def add(self, scope, vector, entry, ttl_seconds):
        if self.fail:
            raise ResponseError("unknown command 'HSET'")
        self.entries.append((scope, vector, entry))


def semantic_upstream(vectors: dict[str, list[float]], verdict: str = "YES") -> Upstream:
    """Embeddings come from `vectors`; the verifier answers `verdict`."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            text = body["input"].removeprefix(QUERY_PREFIX)
            return httpx.Response(200, json={"data": [{"embedding": vectors[text]}]})
        if body["messages"][0].get("content") == VERIFY_SYSTEM:
            return httpx.Response(200, json=chat_response(verdict))
        return httpx.Response(
            200, json=chat_response(f"answer to {body['messages'][-1]['content']}")
        )

    return Upstream(handler)


def ask(client, text, **extra):
    body = {"model": "llama3.2", "messages": [{"role": "user", "content": text}], **extra}
    return client.post("/v1/chat/completions", json=body, headers=AUTH)


def generations(upstream: Upstream) -> int:
    return sum(
        1
        for r in upstream.requests
        if r["path"].endswith("/chat/completions")
        and r["body"]["messages"][0].get("content") != VERIFY_SYSTEM
    )


VECTORS = {
    "What is the capital of France?": [1.0, 0.0, 0.0],
    "Which city is the capital of France?": [0.99, 0.1, 0.0],  # cosine ~0.995
    "How do vaccines work?": [0.0, 1.0, 0.0],  # cosine 0 to the others
}


def test_split_request_scopes_by_model_params_and_history():
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0}
    scope, query = split_request(base)
    assert query == "hi"
    assert split_request({**base, "stream": True, "user": "u"})[0] == scope
    assert split_request({**base, "temperature": 0.7})[0] != scope
    assert split_request({**base, "model": "other"})[0] != scope
    with_system = {
        **base,
        "messages": [{"role": "system", "content": "be terse"}, *base["messages"]],
    }
    assert split_request(with_system)[0] != scope
    assert split_request({**base, "messages": [{"role": "assistant", "content": "x"}]}) is None


def test_verified_paraphrase_is_served_from_cache(client_factory):
    upstream, index = semantic_upstream(VECTORS), MemoryIndex()
    client = client_factory(
        [make_backend("a", upstream)], vector_index=index, semantic_cache_enabled=True
    )
    first = ask(client, "What is the capital of France?")
    assert first.headers["x-gateway-cache"] == "MISS"
    second = ask(client, "Which city is the capital of France?")
    assert second.headers["x-gateway-cache"] == "SEMANTIC_HIT"
    assert float(second.headers["x-gateway-similarity"]) > 0.99
    assert second.json() == first.json()
    assert generations(upstream) == 1
    metrics = client.get("/metrics").text
    assert 'llmgw_semantic_cache_total{outcome="hit"} 1.0' in metrics


def test_verifier_rejection_means_a_fresh_answer(client_factory):
    upstream = semantic_upstream(VECTORS, verdict="NO")
    client = client_factory(
        [make_backend("a", upstream)], vector_index=MemoryIndex(), semantic_cache_enabled=True
    )
    ask(client, "What is the capital of France?")
    second = ask(client, "Which city is the capital of France?")
    assert second.headers["x-gateway-cache"] == "MISS"
    assert generations(upstream) == 2
    assert 'llmgw_semantic_cache_total{outcome="rejected"} 1.0' in client.get("/metrics").text


def test_dissimilar_prompt_skips_the_verifier(client_factory):
    upstream = semantic_upstream(VECTORS)
    client = client_factory(
        [make_backend("a", upstream)], vector_index=MemoryIndex(), semantic_cache_enabled=True
    )
    ask(client, "What is the capital of France?")
    assert ask(client, "How do vaccines work?").headers["x-gateway-cache"] == "MISS"
    verifier_calls = [
        r
        for r in upstream.requests
        if r["body"].get("messages", [{}])[0].get("content") == VERIFY_SYSTEM
    ]
    assert verifier_calls == []


def test_different_sampling_params_never_share_answers(client_factory):
    upstream = semantic_upstream(VECTORS)
    client = client_factory(
        [make_backend("a", upstream)], vector_index=MemoryIndex(), semantic_cache_enabled=True
    )
    ask(client, "What is the capital of France?", temperature=0)
    resp = ask(client, "Which city is the capital of France?", temperature=1)
    assert resp.headers["x-gateway-cache"] == "MISS"


def test_index_failure_is_a_miss_not_an_error(client_factory):
    upstream = semantic_upstream(VECTORS)
    client = client_factory(
        [make_backend("a", upstream)],
        vector_index=MemoryIndex(fail=True),
        semantic_cache_enabled=True,
    )
    assert ask(client, "What is the capital of France?").status_code == 200
    resp = ask(client, "Which city is the capital of France?")
    assert resp.status_code == 200
    assert resp.headers["x-gateway-cache"] == "MISS"
    assert 'llmgw_semantic_cache_total{outcome="error"} 2.0' in client.get("/metrics").text


def test_cache_bypass_also_skips_semantic_cache(client_factory):
    upstream = semantic_upstream(VECTORS)
    client = client_factory(
        [make_backend("a", upstream)], vector_index=MemoryIndex(), semantic_cache_enabled=True
    )
    ask(client, "What is the capital of France?")
    body = {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "Which city is the capital of France?"}],
    }
    resp = client.post(
        "/v1/chat/completions", json=body, headers={**AUTH, "Cache-Control": "no-cache"}
    )
    assert resp.headers["x-gateway-cache"] == "BYPASS"
    assert not any(r["path"].endswith("/embeddings") for r in upstream.requests[1:])
