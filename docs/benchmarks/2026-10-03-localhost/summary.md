# Localhost fix: before/after benchmark

Measures the Windows `localhost` -> `::1` connect delay fix on branch `claude/project-thread-x3jux6`.
Run on 2026-10-04 with the GPU otherwise idle. A pre-check of 10 direct 1-token requests took 33-162 ms each.

## Setup

- Laptop: RTX 4050 (6 GB), Windows 11, on AC power. Ollama runs natively and listens on `127.0.0.1:11434` only.
  Redis runs in Docker.
- Model `llama3.2`, `max_tokens=1`, `temperature=0`. Every prompt is unique, and requests send
  `Cache-Control: no-cache`, so the exact cache never answers.
- Gateway env overrides: `RATE_LIMIT_RPM=0 SEMANTIC_CACHE_ENABLED=false OTEL_EXPORTER_OTLP_ENDPOINT= LEDGER_DATABASE_URL=`.
  The `.env` keeps `OLLAMA_BASE_URL=http://localhost:11434` for both gateway variants.
- Variants:
  - **BEFORE**: `main` code.
  - **AFTER**: this branch's code (same venv, `PYTHONPATH=<worktree>/src`), plus `BACKEND_MAX_CONCURRENCY=0` so the concurrency cap does not affect the comparison.
  - **DIRECT**: no gateway, straight to `http://127.0.0.1:11434/v1`.
- Each gateway was started with `python -m llm_gateway --port 8100`, with no reload. Port 8000 was already
  taken by another gateway instance, which was left untouched. Each gateway was stopped after its variant.
- The client used one persistent `httpx.AsyncClient`. Phases:
  - **first**: the first request after `/livez` answers.
  - **idle**: 25 requests, each after 7 s of sleep.
  - **burst**: 50 back-to-back requests.
  - **concurrent**: 40 requests at concurrency 8.
- Order: BEFORE, AFTER, DIRECT, then the same again (2 rounds, pooled in the table below).
- `client` is the end-to-end time measured by the client. `gateway` is the `x-gateway-latency-ms` header. All times are in ms.
- Non-200 responses: 0.

## Results (median / p95, both rounds pooled)

| Phase | Variant | n | client median | client p95 | gateway median | gateway p95 |
|---|---|---:|---:|---:|---:|---:|
| first | BEFORE | 2 | 533.8 | 618.0 | 498.7 | 556.8 |
| first | AFTER | 2 | 178.7 | 178.9 | 171.2 | 171.3 |
| first | DIRECT | 2 | 167.2 | 169.8 | n/a | n/a |
| idle | BEFORE | 50 | 589.8 | 604.8 | 578.2 | 594.0 |
| idle | AFTER | 50 | 344.0 | 348.5 | 333.6 | 336.9 |
| idle | DIRECT | 50 | 343.0 | 347.4 | n/a | n/a |
| burst | BEFORE | 100 | 44.4 | 159.2 | 40.0 | 154.8 |
| burst | AFTER | 100 | 38.7 | 154.7 | 34.8 | 151.1 |
| burst | DIRECT | 100 | 37.6 | 153.3 | n/a | n/a |
| concurrent | BEFORE | 80 | 270.4 | 476.4 | 265.5 | 436.8 |
| concurrent | AFTER | 80 | 260.2 | 289.9 | 255.4 | 285.3 |
| concurrent | DIRECT | 80 | 265.7 | 272.5 | n/a | n/a |

### First request per run

- round 1 BEFORE: client 618.0 ms, gateway 556.8 ms
- round 1 AFTER: client 178.9 ms, gateway 171.3 ms
- round 1 DIRECT: client 164.6 ms
- round 2 BEFORE: client 449.6 ms, gateway 440.6 ms
- round 2 AFTER: client 178.5 ms, gateway 171.0 ms
- round 2 DIRECT: client 169.8 ms

Raw per-request data: `raw.jsonl` (fields: round, variant, phase, i, client_ms, gateway_ms, status).

## Reading the results

- **first** and **idle** show the fix. BEFORE pays about +250 to +350 ms on a fresh backend connection, because `localhost` tries `::1` first and Ollama only listens on IPv4. AFTER matches DIRECT within about 10 ms.
  - first: BEFORE 534 ms, AFTER 179 ms, DIRECT 167 ms.
  - idle: BEFORE 590 ms, AFTER 344 ms, DIRECT 343 ms.
- **burst**: AFTER is 5.7 ms faster at the median, 44.4 ms vs 38.7 ms. DIRECT is 37.6 ms.
- **concurrent**: AFTER's p95 drops from 476 ms to 290 ms (DIRECT 273 ms). New pooled connections no longer pay the `::1` delay.
- DIRECT's idle median (343 ms vs 38 ms in burst) is Ollama/GPU wake-up after 7 s idle, not contention. No other GPU jobs were running.
- AFTER = branch commit `120cddb`. BEFORE = `main` at `350b8f8`.
