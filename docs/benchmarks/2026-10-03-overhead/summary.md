# Gateway overhead re-measurement (branch head)

This re-measures the README's headline overhead with the code on `claude/project-thread-x3jux6`, using the README's own method.

## Setup

- Run on 2026-10-04 on an RTX 4050 Laptop GPU (6 GB), Windows 11, on AC power, with no other GPU jobs.
- Before the run, 10 direct 1-token requests (after a 20-request warm-up) took 32-40 ms each.
- Gateway: started with `python -m llm_gateway --port 8100` using branch code (`PYTHONPATH=<worktree>/src`) and the `.env` from `main`.
  - Overrides: `RATE_LIMIT_RPM=0 SEMANTIC_CACHE_ENABLED=false`, plus a generated bench key in `GATEWAY_API_KEYS`.
  - Port 8000 was held by another gateway instance, which was left untouched.
- Load generator: `scripts/bench.py`. Direct runs go to `http://127.0.0.1:11434/v1`. Gateway runs add `--no-cache`.
- 1-token runs: 3×100 requests, concurrency 1, `--max-tokens 1`, alternating direct/gateway.
- Streaming runs: 2×30 requests, `--stream --max-tokens 64`, concurrency 1, alternating.
- All 660 requests succeeded. Raw per-run summaries are in `raw.jsonl`.

## 1-token latency, p50 per run (ms)

| Run | direct | gateway |
|---|---:|---:|
| 1 | 46.0 | 42.6 |
| 2 | 40.8 | 42.4 |
| 3 | 40.8 | 42.6 |
| **Median of run p50s** | **40.8** | **42.6** |

Gateway minus direct: **+1.8 ms at p50**, including the extra HTTP hop. The README currently says +4.2 ms (41.5 → 45.7 ms).

The first direct run (p95 158 ms) includes GPU clock ramp-up. Runs 2–3 have direct p95 of 47.7–48.9 ms and gateway p95 of 47.9–50.4 ms.

## Streaming TTFT, p50 per run (ms)

| Run | direct | gateway |
|---|---:|---:|
| 1 | 55.5 | 58.2 |
| 2 | 54.4 | 51.0 |
| **Mean of run p50s** | **55.0** | **54.6** |

The difference is -0.4 ms, within run-to-run noise (gateway runs were 51.0 and 58.2 ms). TTFT p95: direct 67.6 / 64.3 ms, gateway 63.8 / 60.1 ms.

## In-process overhead (`llmgw_gateway_overhead_seconds`, route=chat)

Scraped from `/metrics` at the end of the run. Full lines are in `gateway_overhead_histogram.txt`.

- 306 requests observed. 0 were under 0.25 ms, 299 (97.7%) were 0.5 ms or less, and all 306 were 1 ms or less.
- Mean: 0.337 ms (sum 0.1030 s / 306).
- The README currently says p50 < 0.25 ms, p95 < 0.5 ms, mean 0.39 ms. The new p50 falls in the 0.25–0.5 ms bucket, so the histogram can't confirm "< 0.25 ms".
