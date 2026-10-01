"""Measure availability while a backend dies under load.

Sends a steady stream of chat requests through the gateway for --duration seconds and
records, per request: time, status, the backend that served it, and latency. Kill the
primary backend while this runs (for example, stop a second Ollama node) and the
summary shows how many requests failed and how latency behaved around the failure.

    python scripts/failover.py --api-key $KEY --duration 60 --concurrency 2
"""

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx


async def worker(client: httpx.AsyncClient, args, t0: float, results: list[dict], n: int) -> None:
    i = 0
    while time.perf_counter() - t0 < args.duration:
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": f"Say a random word. ({n}-{i})"}],
            "max_tokens": args.max_tokens,
        }
        start = time.perf_counter()
        try:
            resp = await client.post(
                "/chat/completions", json=body, headers={"Cache-Control": "no-cache"}
            )
            status, backend = resp.status_code, resp.headers.get("x-gateway-backend", "-")
        except httpx.HTTPError as exc:
            status, backend = 0, type(exc).__name__
        results.append(
            {
                "t": round(start - t0, 3),
                "status": status,
                "backend": backend,
                "latency": round(time.perf_counter() - start, 3),
            }
        )
        i += 1


async def run(args) -> dict:
    results: list[dict] = []
    headers = {"Authorization": f"Bearer {args.api_key}"}
    async with httpx.AsyncClient(base_url=args.base_url, headers=headers, timeout=120) as client:
        t0 = time.perf_counter()
        await asyncio.gather(
            *(worker(client, args, t0, results, n) for n in range(args.concurrency))
        )
    results.sort(key=lambda r: r["t"])
    ok = [r for r in results if r["status"] == 200]
    per_window: dict[int, Counter] = defaultdict(Counter)
    for r in results:
        per_window[int(r["t"] // args.window)][
            r["backend"] if r["status"] == 200 else "FAILED"
        ] += 1
    return {
        "label": args.label,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "requests": len(results),
        "succeeded": len(ok),
        "availability": len(ok) / len(results) if results else 0.0,
        "served_by": dict(Counter(r["backend"] for r in ok)),
        "max_latency_s": max((r["latency"] for r in ok), default=None),
        "timeline": {
            f"{w * args.window}-{(w + 1) * args.window}s": dict(c)
            for w, c in sorted(per_window.items())
        },
        "requests_detail": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="failover")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--window", type=float, default=5, help="timeline bucket in seconds")
    parser.add_argument("--out", default="results/failover.jsonl")
    args = parser.parse_args()

    summary = asyncio.run(run(args))
    printable = {k: v for k, v in summary.items() if k != "requests_detail"}
    print(json.dumps(printable, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
