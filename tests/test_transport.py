"""The real backend client over both HTTP transports, against a local aiohttp server.

Most tests script upstream replies with httpx.MockTransport, which bypasses the network
layer. These run real sockets so the aiohttp transport (the default) is covered too.
"""

import asyncio
import json
import socket

import pytest
from aiohttp import web

from llm_gateway.backends import BackendError, OpenAICompatibleBackend
from llm_gateway.config import BackendConfig

pytestmark = pytest.mark.anyio
TRANSPORTS = ["httpx", "aiohttp"]


async def _chat(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    if body["model"] == "bad":
        return web.json_response({"error": {"message": "unknown model"}}, status=400)
    if body["model"] == "slow":
        await asyncio.sleep(1)
    if not body.get("stream"):
        return web.json_response(
            {
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                "seen_model": body["model"],
                "auth": request.headers.get("Authorization"),
            }
        )
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    for word in ["a", "b", "c"]:
        chunk = {"choices": [{"index": 0, "delta": {"content": word}}]}
        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
    await response.write(b"data: [DONE]\n\n")
    return response


async def _models(_: web.Request) -> web.Response:
    return web.json_response({"object": "list", "data": []})


@pytest.fixture
async def upstream():
    app = web.Application()
    app.router.add_post("/v1/chat/completions", _chat)
    app.router.add_get("/v1/models", _models)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}/v1"
    await runner.cleanup()


def _backend(base_url: str, transport: str, **kw) -> OpenAICompatibleBackend:
    config = BackendConfig(
        name="b", base_url=base_url, api_key="sk", models={"llama3.2": "llama3.2:3b"}, **kw
    )
    return OpenAICompatibleBackend(config.model_copy(update={"http_transport": transport}))


@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_chat_and_health(upstream, transport):
    backend = _backend(upstream, transport)
    try:
        result = await backend.chat({"model": "llama3.2", "messages": []})
        assert result["seen_model"] == "llama3.2:3b"
        assert result["auth"] == "Bearer sk"
        assert await backend.health()
    finally:
        await backend.aclose()


@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_stream_lines(upstream, transport):
    backend = _backend(upstream, transport)
    try:
        lines = [line async for line in backend.chat_stream({"model": "llama3.2"})]
    finally:
        await backend.aclose()
    assert len(lines) == 4 and lines[-1] == "data: [DONE]"


@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_client_error_status_is_kept(upstream, transport):
    backend = _backend(upstream, transport)
    try:
        with pytest.raises(BackendError) as err:
            await backend.chat({"model": "bad", "messages": []})
    finally:
        await backend.aclose()
    assert err.value.status_code == 400 and err.value.is_client_error
    assert err.value.message == "unknown model"


@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_read_timeout_is_a_backend_error(upstream, transport):
    backend = _backend(upstream, transport, timeout_seconds=0.2)
    try:
        with pytest.raises(BackendError) as err:
            await backend.chat({"model": "slow", "messages": []})
    finally:
        await backend.aclose()
    assert err.value.kind == "connection"


@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_refused_connection_is_a_backend_error(transport):
    with socket.socket() as s:  # a port nobody listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    backend = _backend(f"http://127.0.0.1:{port}/v1", transport)
    try:
        with pytest.raises(BackendError) as err:
            await backend.chat({"model": "llama3.2", "messages": []})
        assert not await backend.health()
    finally:
        await backend.aclose()
    assert err.value.kind == "connection"
