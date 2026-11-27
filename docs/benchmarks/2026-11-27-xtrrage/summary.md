# Gateway re-measurement on the new desktop (2026-11-27)

Same method as the README's results section and `2026-10-03-overhead`, on new hardware.

## Setup

- Desktop "XTRRAGE": AMD Ryzen 7 9800X3D (8 cores / 16 threads), 31 GB RAM, NVIDIA GeForce RTX 5070 Ti (16 GB, driver 621.40), Windows 11 Home 10.0.26300, mains power.
- Ollama 0.35.0 with `llama3.2` (3B) on the GPU. Redis 8 and Postgres 16 in Docker Desktop 4.100.1 (engine 29.9.3).
- Gateway on the host: `python -m llm_gateway --port 8100`, code from `main` as copied from the laptop, `.env` from the laptop, with overrides `RATE_LIMIT_RPM=0 SEMANTIC_CACHE_ENABLED=false OLLAMA_BASE_URL=http://127.0.0.1:11434` and a generated bench key.
- `.env` has OpenTelemetry tracing on, and no Jaeger was running, so span export failed in the background. The README measured tracing on vs off at about +0.1 ms.
- Load generator `scripts/bench.py`, direct and gateway runs alternating, 20-request warm-up first.
- **0 errors across 920 measured requests.** Raw per-run summaries: `raw.jsonl`.

## Results vs the laptop (RTX 4050 Laptop at a 20 W cap, 2026-10-02 / 10-04)

| What | Laptop direct | Laptop gateway | **Desktop direct** | **Desktop gateway** | Desktop difference |
|---|---:|---:|---:|---:|---|
| 1-token p50, 3×100, median of run p50s | 41.5 ms (40.8 on 10-04) | 45.7 ms (42.6) | **16.2 ms** | **19.3 ms** | **+3.1 ms** |
| Streaming TTFT p50, 2×30, mean of run p50s | 56.6 ms (55.0) | 55.2 ms (54.6) | **20.3 ms** | **22.9 ms** | +2.6 ms |
| Load: 4 clients, 64 tokens, 2×40, mean of run p50s | 9.53 s | 9.64 s | **0.93 s** | **0.92 s** | -1.1%, noise |
| Load throughput | 0.42 req/s, 26.5 tok/s | 0.42 req/s, 26.5 tok/s | **4.24 req/s, 270 tok/s** | **4.34 req/s, 276 tok/s** | none |
| Cache: 50% repeats, 4 clients, 40 requests | - | 17 hits, 0.74 vs 0.42 req/s (+74%) | - | **17 hits, 10.61 vs 4.34 req/s (+144%)** | |
| In-process overhead histogram | - | 306-317 req, mean 0.34-0.39 ms | - | **409 req, mean 0.37 ms; 50.9% ≤ 0.25 ms, 93.6% ≤ 0.5 ms, 99.8% ≤ 5 ms** | |

## Per-run p50 (ms)

| Run | direct | gateway |
|---|---:|---:|
| 1-token 1 | 16.2 | 19.7 |
| 1-token 2 | 16.1 | 19.3 |
| 1-token 3 | 17.1 | 17.9 |
| stream TTFT 1 | 20.3 | 23.6 |
| stream TTFT 2 | 20.2 | 22.1 |
| load 1 | 953.8 | 939.4 |
| load 2 | 910.9 | 905.1 |

## Reading it

- Generation got about **10× faster** (26.5 → ~270 tok/s), so 64-token answers dropped from ~9.5 s to ~0.93 s. Part of that is the laptop's 20 W power cap, not only the newer GPU.
- The gateway's own cost is unchanged: about 0.37 ms in-process and **about 2-4 ms** client-side at p50, the same range as the laptop's +1.8 to +4.2 ms. With a faster backend, that is now a larger *share* of a tiny 1-token request (~19% vs ~5-10%) and still noise on real answers.
- The cache got the same 17/40 hits; the throughput gain measured +144% here vs +74% on the laptop. With only 40 requests in one run, treat that gap as indicative, not precise.
