r"""A fake OpenAI-compatible backend with fixed latency, for measuring the gateway itself.

Real model latency (tens of ms to seconds) hides what the gateway costs. Pointing the
gateway at this server isolates its own overhead and its throughput ceiling, and makes
failure scenarios reproducible without a GPU.

    python scripts/mock_upstream.py --port 9100 --delay-ms 20
    OLLAMA_BASE_URL=http://127.0.0.1:9100 python -m llm_gateway
    python scripts/bench.py --label mock --api-key $KEY --requests 2000 --concurrency 32 --no-cache

Flags: --delay-ms (fixed latency per request), --jitter-ms (uniform extra latency),
--error-rate (fraction answered with HTTP 500), --tokens (streamed chunks per answer),
--parallel (requests generated at once; the rest queue, like a GPU).
"""

import argparse
import asyncio
import json
import random
import time

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

ARGS = argparse.Namespace(delay_ms=0.0, jitter_ms=0.0, error_rate=0.0, tokens=8, parallel=0)
_slots: asyncio.Semaphore | None = None


async def _generate(delay: float) -> None:
    if _slots is None:
        await asyncio.sleep(delay)
        return
    async with _slots:
        await asyncio.sleep(delay)


async def _disconnected(request: Request) -> None:
    while (await request.receive())["type"] != "http.disconnect":
        pass


async def _wait(request: Request) -> bool:
    """Simulate generation; False if the client disconnected first.

    With --parallel N, like a GPU, only N requests run at once and the rest queue
    (Ollama's OLLAMA_NUM_PARALLEL), so latency grows with load. Like Ollama, generation
    stops when the client disconnects, freeing the slot.
    """
    delay = (ARGS.delay_ms + random.uniform(0, ARGS.jitter_ms)) / 1000
    if not delay:
        return True
    work = asyncio.ensure_future(_generate(delay))
    gone = asyncio.ensure_future(_disconnected(request))
    await asyncio.wait({work, gone}, return_when=asyncio.FIRST_COMPLETED)
    gone.cancel()
    if work.done():
        return True
    work.cancel()
    return False


def _usage(completion: int) -> dict:
    return {"prompt_tokens": 12, "completion_tokens": completion, "total_tokens": 12 + completion}


async def chat(request: Request) -> Response:
    body = await request.json()
    if not await _wait(request):
        return Response(status_code=499)
    if random.random() < ARGS.error_rate:
        return JSONResponse({"error": {"message": "injected failure"}}, 500)
    model = body.get("model", "mock")
    if not body.get("stream"):
        return JSONResponse(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok " * ARGS.tokens},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _usage(ARGS.tokens),
            }
        )

    async def events():
        for _ in range(ARGS.tokens):
            chunk = {"model": model, "choices": [{"index": 0, "delta": {"content": "ok "}}]}
            yield f"data: {json.dumps(chunk)}\n\n"
        yield f"data: {json.dumps({'choices': [], 'usage': _usage(ARGS.tokens)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


async def embeddings(request: Request) -> Response:
    await request.json()
    if not await _wait(request):
        return Response(status_code=499)
    vector = [0.0] * 768
    return JSONResponse(
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": vector}],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }
    )


async def models(_: Request) -> Response:
    return JSONResponse({"object": "list", "data": [{"id": "mock", "object": "model"}]})


app = Starlette(
    routes=[
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/v1/embeddings", embeddings, methods=["POST"]),
        Route("/v1/models", models),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--delay-ms", type=float, default=0.0)
    parser.add_argument("--jitter-ms", type=float, default=0.0)
    parser.add_argument("--error-rate", type=float, default=0.0)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--parallel", type=int, default=0, help="0 = unlimited")
    args = parser.parse_args()
    vars(ARGS).update(vars(args))
    global _slots
    _slots = asyncio.Semaphore(args.parallel) if args.parallel else None
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
