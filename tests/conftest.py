"""Shared fixtures.

Unit tests drive the real `OpenAICompatibleBackend` against scripted HTTP responses
(httpx.MockTransport) and an in-process Redis (fakeredis), so they run in CI without
a GPU. Tests against real Ollama/Redis are marked `integration`.
"""

import json
from collections.abc import Callable, Iterator
from typing import Any

import fakeredis
import httpx
import pytest
from fastapi.testclient import TestClient

from llm_gateway.backends import OpenAICompatibleBackend
from llm_gateway.config import BackendConfig, Settings
from llm_gateway.main import create_app

API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def chat_response(content: str, model: str = "m", prompt: int = 5, completion: int = 7) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def sse(chunks: list[str], usage: dict | None = None) -> str:
    lines = [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": c}}]}) for c in chunks
    ]
    if usage:
        lines.append("data: " + json.dumps({"choices": [], "usage": usage}))
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


class Upstream:
    """A scripted upstream server: records requests, replies via `handler`."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append({"path": request.url.path, "body": body})
        return self.handler(request)


def ok(content: str = "hello") -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": []})
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                    "model": "e",
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                },
            )
        body = json.loads(request.content)
        if body.get("stream"):
            text = sse(["hel", "lo"], {"prompt_tokens": 4, "completion_tokens": 2})
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=chat_response(content, body["model"]))

    return handler


def fail(status: int = 500) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": f"upstream {status}"}})

    return handler


def make_backend(
    name: str,
    upstream: Upstream,
    models: dict[str, str] | None = None,
    cost_in: float = 0.0,
    cost_out: float = 0.0,
    quality: float = 0.5,
) -> OpenAICompatibleBackend:
    config = BackendConfig(
        name=name,
        base_url=f"http://{name}.test/v1",
        models=models or {"*": "*"},
        cost_per_1k_input=cost_in,
        cost_per_1k_output=cost_out,
        quality=quality,
    )
    return OpenAICompatibleBackend(config, transport=httpx.MockTransport(upstream))


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        gateway_api_keys=API_KEY,
        rate_limit_rpm=1000,
        retry_backoff_seconds=0,
        backend_max_retries=1,
        default_model="llama3.2",
        semantic_cache_enabled=False,  # enabled per test in test_semantic_cache.py
    )


@pytest.fixture
def redis() -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(decode_responses=True)


@pytest.fixture
def client_factory(
    settings: Settings, redis: fakeredis.FakeAsyncRedis
) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def factory(
        backends: list[OpenAICompatibleBackend],
        vector_index: Any = None,
        ledger_writer: Any = None,
        **overrides: Any,
    ) -> TestClient:
        app = create_app(
            settings.model_copy(update=overrides),
            backends=backends,
            redis=redis,
            vector_index=vector_index,
            ledger_writer=ledger_writer,
        )
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)
