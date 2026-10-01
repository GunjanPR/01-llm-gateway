r"""Load-test an OpenAI-compatible endpoint: the gateway, or a backend directly (baseline).

Examples:
    # Through the gateway, cache bypassed: compare against the direct run for overhead
    python scripts/bench.py --label gateway --base-url http://localhost:8000/v1 \
        --api-key $KEY --requests 100 --concurrency 4 --no-cache

    # Same workload straight to Ollama: the baseline
    python scripts/bench.py --label direct --base-url http://localhost:11434/v1 \
        --requests 100 --concurrency 4

    # Cache effect: 50% of requests repeat an earlier prompt
    python scripts/bench.py --label cache --base-url http://localhost:8000/v1 \
        --api-key $KEY --repeat-ratio 0.5

    # Streaming, to measure time-to-first-token
    python scripts/bench.py --label stream --base-url http://localhost:8000/v1 \
        --api-key $KEY --stream

Results are printed and appended as JSON lines to results/bench.jsonl.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

TOPICS = [
    "photosynthesis", "TCP congestion control", "the French Revolution", "black holes",
    "CRISPR", "inflation", "transformers in NLP", "plate tectonics", "vaccines", "blockchains",
    "the Krebs cycle", "gradient descent", "volcanoes", "the immune system", "quantum tunneling",
]  # fmt: skip


def make_prompts(n: int, repeat_ratio: float, seed: int) -> list[str]:
    rng = random.Random(seed)
    prompts: list[str] = []
    for i in range(n):
        if prompts and rng.random() < repeat_ratio:
            prompts.append(rng.choice(prompts))
        else:
            prompts.append(f"In two sentences, explain {rng.choice(TOPICS)}. (variant {i})")
    return prompts


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


async def one_request(client: httpx.AsyncClient, args, prompt: str) -> dict:
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": args.stream,
    }
    if args.stream:
        body["stream_options"] = {"include_usage": True}
    headers = {"Cache-Control": "no-cache"} if args.no_cache else {}
    start = time.perf_counter()
    result = {"ok": False, "latency": None, "ttft": None, "tokens": 0, "cache": None}
    try:
        if args.stream:
            async with client.stream(
                "POST", "/chat/completions", json=body, headers=headers
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:") or line.endswith("[DONE]"):
                        continue
                    chunk = json.loads(line[5:])
                    if result["ttft"] is None and chunk.get("choices"):
                        result["ttft"] = time.perf_counter() - start
                    if chunk.get("usage"):
                        result["tokens"] = chunk["usage"].get("completion_tokens", 0)
        else:
            resp = await client.post("/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            result["tokens"] = (resp.json().get("usage") or {}).get("completion_tokens", 0)
            result["cache"] = resp.headers.get("x-gateway-cache")
        result["ok"] = True
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:200]
    result["latency"] = time.perf_counter() - start
    return result


async def run(args) -> dict:
    prompts = make_prompts(args.requests + args.warmup, args.repeat_ratio, args.seed)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(
        base_url=args.base_url, headers=headers, timeout=args.timeout, limits=limits
    ) as client:
        for prompt in prompts[: args.warmup]:
            await one_request(client, args, prompt)
        semaphore = asyncio.Semaphore(args.concurrency)

        async def bounded(prompt: str) -> dict:
            async with semaphore:
                return await one_request(client, args, prompt)

        start = time.perf_counter()
        results = await asyncio.gather(*(bounded(p) for p in prompts[args.warmup :]))
        wall = time.perf_counter() - start

    ok = [r for r in results if r["ok"]]
    latencies = [r["latency"] for r in ok]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    tokens = sum(r["tokens"] for r in ok)
    summary = {
        "label": args.label,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "model": args.model,
        "requests": len(results),
        "concurrency": args.concurrency,
        "stream": args.stream,
        "max_tokens": args.max_tokens,
        "repeat_ratio": args.repeat_ratio,
        "no_cache": args.no_cache,
        "errors": len(results) - len(ok),
        "throughput_rps": len(ok) / wall if wall else 0.0,
        "completion_tokens_per_s": tokens / wall if wall else 0.0,
        "latency_p50_s": pct(latencies, 0.50),
        "latency_p95_s": pct(latencies, 0.95),
        "latency_p99_s": pct(latencies, 0.99),
        "latency_mean_s": statistics.fmean(latencies) if latencies else float("nan"),
        "ttft_p50_s": pct(ttfts, 0.50),
        "ttft_p95_s": pct(ttfts, 0.95),
        "cache_hits": sum(1 for r in ok if r["cache"] == "HIT"),
        "sample_errors": [r["error"] for r in results if not r["ok"]][:3],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="name for this run, e.g. gateway/direct")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeat-ratio", type=float, default=0.0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="send Cache-Control: no-cache")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="results/bench.jsonl")
    args = parser.parse_args()

    summary = asyncio.run(run(args))
    for key, value in summary.items():
        print(f"{key:>26}: {value:.4f}" if isinstance(value, float) else f"{key:>26}: {value}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
