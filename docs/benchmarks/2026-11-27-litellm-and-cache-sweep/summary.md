# LiteLLM baseline and cache repeat-ratio sweep (2026-11-27)

Two questions: how does this gateway's per-request cost compare with an existing open-source
gateway (LiteLLM proxy) on the same backend, and how does throughput change with the share of
repeated prompts when the response cache is on.

## Setup

- Desktop "XTRRAGE": Ryzen 7 9800X3D, 31 GB RAM, RTX 5070 Ti 16 GB (driver 621.40), Windows 11 Home 10.0.26300, mains power. Nothing else running on the CPU or GPU; the 02 OpenSearch/Qdrant containers were stopped.
- Backend: Ollama 0.40.18 with `llama3.2` (3B, Q4_K_M) on the GPU, default context length.
- This gateway: `python -m llm_gateway --port 8100` at commit `6f4ff89`, with `RATE_LIMIT_RPM=0 SEMANTIC_CACHE_ENABLED=false OLLAMA_BASE_URL=http://127.0.0.1:11434`, tracing export off, a generated bench key. Redis 8 and Postgres 16 (usage ledger) in Docker Desktop (engine 29.9.3).
- LiteLLM proxy 1.111.0 in its own venv: `litellm --model ollama_chat/llama3.2 --port 4000`, no cache, no logging callbacks, a generated `LITELLM_MASTER_KEY` (1.111 refuses to start without one). `ollama_chat/` makes LiteLLM call Ollama's native `/api/chat`; the gateway and the direct runs use Ollama's OpenAI-compatible `/v1`.
- Load generator `scripts/bench.py` (2 warm-up requests per run, not counted), plus a 20-request warm-up per target first (`warmup.jsonl`). Comparison runs send `Cache-Control: no-cache` to both gateways.
- **0 errors across 3,100 measured requests.** Per-run summaries: `raw.jsonl`. Gateway cache counters after the sweep: `gateway_cache_metrics.txt`.

## 1-token requests (concurrency 1, `--max-tokens 1`, 3 × 100 per target, alternating)

| Run | direct p50 / p95 | gateway p50 / p95 | LiteLLM p50 / p95 |
|---|---:|---:|---:|
| 1 | 16.5 / 18.7 ms | 17.5 / 20.2 ms | 33.6 / 37.0 ms |
| 2 | 16.6 / 19.1 ms | 17.4 / 20.1 ms | 33.4 / 35.0 ms |
| 3 | 16.5 / 18.6 ms | 17.3 / 18.9 ms | 34.7 / 39.3 ms |
| **Median of run p50s** | **16.5 ms** | **17.4 ms** | **33.6 ms** |
| Added over direct | - | **+0.9 ms** | **+17.1 ms** |
| Throughput (mean) | 60.2 req/s | 57.1 req/s | 29.3 req/s |

The earlier XTRRAGE run (`2026-11-27-xtrrage`) measured +3.1 ms for the gateway with tracing export on and failing in the background; this run had it off. Part of LiteLLM's extra time may come from the different Ollama endpoint (`/api/chat` vs `/v1`), so the +17 ms is "LiteLLM as configured here", not proxy code alone.

## Load (`--requests 200 --max-tokens 64`, one run each, alternating)

| Concurrency | Target | p50 | p95 | p99 | req/s | tok/s |
|---|---|---:|---:|---:|---:|---:|
| 4 | direct | 948 ms | 1001 ms | 1013 ms | 4.21 | 267 |
| 4 | gateway | 910 ms | 916 ms | 919 ms | 4.42 | 281 |
| 4 | LiteLLM | 912 ms | 924 ms | 928 ms | 4.41 | 280 |
| 16 | direct | 3634 ms | 3649 ms | 3658 ms | 4.43 | 281 |
| 16 | gateway | 3618 ms | 3649 ms | 3657 ms | 4.43 | 281 |
| 16 | LiteLLM | 3613 ms | 3659 ms | 5498 ms | 4.43 | 281 |

With 64-token answers the GPU is the bottleneck: all three reach the same ~4.4 req/s, and at concurrency 16 requests queue in Ollama (p50 ≈ 3.6 s). The direct run at concurrency 4 was the first 64-token run after the 1-token runs and came out ~4% slower; with one run per cell, differences of a few percent are noise. LiteLLM's p99 at concurrency 16 (5.5 s) is a single run's tail.

## Cache repeat-ratio sweep (this gateway, cache on, `--requests 200 --concurrency 4 --max-tokens 64`)

Gateway cache keys (`llmgw:cache:*`) were deleted before each run so runs do not share entries. Hits are responses with `x-cache: HIT`.

| Repeat ratio | Cache hits | Hit rate | p50 | p95 | req/s | tok/s | vs ratio 0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0 | 0% | 906 ms | 914 ms | 4.44 | 282 | 1.00× |
| 0.2 | 40 | 20% | 903 ms | 915 ms | 5.70 | 361 | 1.28× |
| 0.4 | 78 | 39% | 685 ms | 914 ms | 7.79 | 495 | 1.75× |
| 0.6 | 111 | 56% | 7.5 ms | 914 ms | 11.02 | 703 | 2.48× |
| 0.8 | 147 | 74% | 4.7 ms | 901 ms | 19.37 | 1232 | 4.36× |

- Hit rates track the repeat ratio, a little below it, because a repeat whose original is still generating is coalesced (waits for that answer) rather than counted as a hit. The gateway counted 376 hits and 28 coalesced requests over the sweep.
- p95 stays at the uncached generation time (~0.91 s) at every ratio, because misses still generate. p50 drops to a few milliseconds once more than half the requests are hits.
- Throughput rises roughly as 1 / (1 − hit rate) for a backend-bound workload: 4.4 → 19.4 req/s at 74% hits.
