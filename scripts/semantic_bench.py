"""End-to-end semantic cache evaluation through the running gateway.

For each labelled pair in eval/semantic_pairs.jsonl: send prompt A (a miss; its answer is
stored), then prompt B, and record whether B was served from the semantic cache. A hit
on a "same" pair is a correct reuse; a hit on a "different" pair is a false hit (a wrong
answer). Each pair runs in its own conversation (a per-pair system message), so pairs
cannot match each other's entries.

    python scripts/semantic_bench.py --api-key $KEY
"""

import argparse
import json
import re
import statistics
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx


def stage_means(metrics_text: str) -> dict[str, float]:
    sums, counts = {}, {}
    for line in metrics_text.splitlines():
        m = re.match(r'llmgw_semantic_cache_seconds_(sum|count)\{stage="(\w+)"\} (\S+)', line)
        if m:
            (sums if m[1] == "sum" else counts)[m[2]] = float(m[3])
    return {k: sums[k] / counts[k] * 1000 for k in sums if counts.get(k)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--pairs", default="eval/semantic_pairs.jsonl")
    parser.add_argument("--out", default="results/semantic_bench.json")
    args = parser.parse_args()

    pairs = [json.loads(line) for line in Path(args.pairs).read_text().splitlines() if line]
    run = uuid.uuid4().hex[:8]
    headers = {"Authorization": f"Bearer {args.api_key}"}
    rows = []
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=180) as client:
        before = stage_means(client.get("/metrics").text)
        for i, pair in enumerate(pairs):
            system = {
                "role": "system",
                "content": f"You are a helpful assistant. Session {run}-{i}.",
            }

            def send(text: str, system: dict = system) -> httpx.Response:
                body = {
                    "model": args.model,
                    "messages": [system, {"role": "user", "content": text}],
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                }
                resp = client.post("/v1/chat/completions", json=body)
                resp.raise_for_status()
                return resp

            first = send(pair["a"])
            second = send(pair["b"])
            hit = second.headers["x-gateway-cache"] == "SEMANTIC_HIT"
            rows.append(
                {
                    **pair,
                    "first_cache": first.headers["x-gateway-cache"],
                    "first_latency_ms": float(first.headers["x-gateway-latency-ms"]),
                    "second_cache": second.headers["x-gateway-cache"],
                    "second_latency_ms": float(second.headers["x-gateway-latency-ms"]),
                    "similarity": float(second.headers.get("x-gateway-similarity", "nan")),
                    "outcome": ("correct_hit" if pair["same"] else "FALSE_HIT")
                    if hit
                    else ("missed_paraphrase" if pair["same"] else "correct_miss"),
                }
            )
            print(f"{i:>2} {rows[-1]['outcome']:<17} {pair['a']} | {pair['b']}")
        after_text = client.get("/metrics").text

    same = [r for r in rows if r["same"]]
    diff = [r for r in rows if not r["same"]]
    hits = [r for r in rows if r["second_cache"] == "SEMANTIC_HIT"]
    misses = [r for r in rows if r["second_cache"] != "SEMANTIC_HIT"]
    summary = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": args.model,
        "pairs": len(rows),
        "paraphrase_hit_rate": sum(r["outcome"] == "correct_hit" for r in same) / len(same),
        "paraphrase_hits": f"{sum(r['outcome'] == 'correct_hit' for r in same)}/{len(same)}",
        "false_hits": f"{sum(r['outcome'] == 'FALSE_HIT' for r in diff)}/{len(diff)}",
        "semantic_hit_latency_ms_median": statistics.median(r["second_latency_ms"] for r in hits)
        if hits
        else None,
        "miss_latency_ms_median": statistics.median(r["second_latency_ms"] for r in misses),
        "stage_mean_ms_this_run": stage_means(after_text),
        "stage_mean_ms_before_run": before,
    }
    print(json.dumps(summary, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
