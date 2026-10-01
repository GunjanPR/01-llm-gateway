# Production LLM Inference Gateway

An OpenAI-compatible gateway in front of vLLM/Ollama (and optional hosted APIs). It handles routing, caching, rate limiting, fallback, and cost tracking, with full observability.

> Part of the Live Projects portfolio. See [SHARED_STANDARDS.md](../SHARED_STANDARDS.md) for conventions and the milestone definition of done.

## Overview

Applications that call LLMs directly end up re-implementing auth, retries, caching, and
cost tracking in every service. This gateway puts all of that behind one
**OpenAI-compatible endpoint**. Any OpenAI SDK or HTTP client works unchanged, including
the [Healthcare RAG](../02-healthcare-rag/) project. Requests can go to local models
(Ollama on your GPU, vLLM) or to a hosted provider.

**What it does today**

| Feature | How |
|---|---|
| OpenAI-compatible API | `POST /v1/chat/completions` (JSON + SSE streaming), `POST /v1/embeddings`, `GET /v1/models` |
| Auth | Bearer API keys (`GATEWAY_API_KEYS`), constant-time comparison; with no keys configured, every request is rejected |
| Per-tenant rate limiting | Fixed one-minute window per API key in Redis (`INCR` + `EXPIRE`), shared across replicas, with a `Retry-After` header |
| Routing policies | `cost` (cheapest $/1K tokens), `latency` (lowest observed EWMA latency), `quality` (highest configured score) |
| Reliability | Retries with exponential backoff, fallback to the next backend, and a per-backend circuit breaker. Streams fall back only until the first token arrives. Bad requests (400/413/422) go straight back to the client, without retries. |
| Model aliasing | Maps a public model name to each backend's own name, e.g. `llama3.2` → `llama3.2` on Ollama and `llama-3.1-8b-instant` on Groq, so fallback works across providers |
| Response cache | Exact-match Redis cache keyed by a SHA-256 of the canonical request. Modes: `all`, `deterministic`, `off`. `Cache-Control: no-cache` bypasses it. |
| Cost tracking | Token usage from the backend, or estimated if the backend doesn't report it, times $/1K pricing per backend. The cost of each request goes in the `x-gateway-cost-usd` header. |
| Resilience | If Redis goes down, cache and rate limiting are skipped (fail open) instead of failing requests |
| Metrics | Prometheus `/metrics`: requests by outcome, latency histograms, TTFT, tokens/s, tokens, spend, cache hits and savings, backend errors, breaker state |
| Dashboards | A provisioned Grafana dashboard (15 panels) |

Every response carries `x-gateway-backend`, `x-gateway-cache` (`HIT`/`MISS`/`BYPASS`),
`x-gateway-cost-usd` and `x-gateway-latency-ms` headers.

## Tech Stack

| Layer | Tools |
|---|---|
| API | FastAPI (async), Uvicorn, httpx |
| Cache / rate limits | Redis 7 |
| Model backends | Ollama (native, GPU), vLLM, any OpenAI-compatible hosted API (e.g. Groq) |
| Observability | Prometheus, Grafana |
| Load testing | `scripts/bench.py` (async httpx: p50/p95/p99, TTFT, throughput) |
| Infra | Docker, Docker Compose, GitHub Actions |

## Architecture

| Component | Module | Responsibility |
|---|---|---|
| **API layer** | `main.py` | OpenAI-compatible routes, validation, SSE streaming, OpenAI-style errors |
| **Auth & rate limiter** | `main.py`, `ratelimit.py` | API-key check, per-tenant limit in Redis |
| **Cache** | `cache.py` | Exact-match response cache in Redis |
| **Router** | `router.py` | Ranks backends by policy; retries, fallback, circuit breaker |
| **Backend adapter** | `backends.py` | One httpx client per OpenAI-compatible upstream; maps errors to retryable or client errors |
| **Telemetry** | `metrics.py` | Prometheus metrics, token accounting, cost |
| **Config** | `config.py` | `pydantic-settings` from env / `.env`; builds the backend list |

```
          clients (curl, OpenAI SDKs, 02-healthcare-rag)
                         │  OpenAI-compatible HTTP + Bearer key
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    LLM Gateway (FastAPI)                     │
│                                                              │
│  auth ─► rate limit ─► cache lookup ──hit──► response        │
│            │               │ miss                            │
│            ▼               ▼                                 │
│          Redis ◄──── router: rank by policy ─► retry ─►      │
│        (counters,     fallback ─► circuit breaker            │
│         cache)            │                                  │
│  /metrics ─► Prometheus ─► Grafana                           │
└───────────────────────────┼──────────────────────────────────┘
              ┌─────────────┼──────────────────┐
              ▼             ▼                  ▼
         vLLM server   Ollama (GPU)    hosted API (Groq, ...)
```

## Setup

### Prerequisites
| Tool | Install (Windows) | Check |
|---|---|---|
| Python 3.11 | `winget install -e --id Python.Python.3.11` | `py -3.11 --version` |
| WSL2 (needed by Docker) | `wsl --install` in an admin PowerShell, then reboot | `wsl --status` |
| Docker Desktop | `winget install -e --id Docker.DockerDesktop` | `docker run --rm hello-world` |
| Ollama | `winget install -e --id Ollama.Ollama` | `ollama list` |
| Models | `ollama pull llama3.2`, `ollama pull llama3.2:1b`, `ollama pull nomic-embed-text` | `ollama list` |
| GNU Make (optional) | `choco install make -y` (admin) | `make --version` |

On macOS/Linux, install the same tools with your package manager. vLLM is optional and
needs Linux or WSL2 with an NVIDIA GPU.

### Install
```bash
git clone https://github.com/GunjanPR/01-llm-gateway.git && cd 01-llm-gateway
py -3.11 -m venv .venv            # macOS/Linux: python3.11 -m venv .venv
source .venv/Scripts/activate     # PowerShell: .venv\Scripts\Activate.ps1; macOS/Linux: source .venv/bin/activate
make install                      # = pip install -e ".[dev]"
```

### Environment config
```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # paste into GATEWAY_API_KEYS
```
Every variable is documented in [.env.example](.env.example). The ones you'll most often change:

- `GATEWAY_API_KEYS`: the client keys the gateway accepts. **This is required**; the gateway
  rejects every request if it's empty.
- `OLLAMA_MODELS`: the models Ollama serves. A `public=upstream` entry maps a gateway model
  name to an Ollama tag.
- `ROUTING_POLICY`: `cost`, `latency` or `quality`.
- `CACHE_MODE`: `all`, `deterministic` or `off`.
- `HOSTED_*`: an optional second provider. Set `HOSTED_BASE_URL`, `HOSTED_API_KEY`,
  `HOSTED_MODELS` (for example `llama3.2=llama-3.1-8b-instant`) and its prices, to get real
  cross-provider fallback.

## How to Run

> **Windows note:** `make` isn't installed by default. Use `choco install make`, run from
> **Git Bash**, or run the command shown next to each target in the [Makefile](Makefile).

### 1. Start the infrastructure (Redis, Prometheus, Grafana)
```bash
make up                 # = docker compose up -d redis prometheus grafana
```

### 2. Start the gateway
Ollama runs natively, so it uses your GPU directly. The gateway runs on the host:
```bash
make run                # = python -m llm_gateway --reload  → http://localhost:8000
```
Or run everything, gateway included, in Docker. Inside a container the gateway reaches
the host's Ollama at `host.docker.internal`:
```bash
make up-all             # = docker compose up -d --build
```

### 3. Call it
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_KEY" -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "messages": [{"role": "user", "content": "Hello"}]}' -i
```
Or use the OpenAI SDK:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<gateway key>")
client.chat.completions.create(model="llama3.2", messages=[{"role": "user", "content": "Hi"}])
```

| URL | What |
|---|---|
| http://localhost:8000/health | Redis and backend reachability, circuit-breaker state |
| http://localhost:8000/metrics | Prometheus metrics |
| http://localhost:8000/docs | OpenAPI UI |
| http://localhost:9090 | Prometheus |
| http://localhost:3000 | Grafana: **LLM Gateway** dashboard (admin password from `GRAFANA_ADMIN_PASSWORD`) |

### Tests
```bash
make lint               # ruff check + ruff format --check
make test               # unit tests (no GPU or services needed)
make test-integration   # end-to-end against the running gateway + Redis + Ollama
                        #   needs GATEWAY_TEST_KEY=<one of GATEWAY_API_KEYS> in the environment
```
The unit tests run the real `OpenAICompatibleBackend` against scripted HTTP responses
(`httpx.MockTransport`) and an in-process Redis (`fakeredis`), so CI needs no GPU. The
integration tests in `tests/integration/` hit the real stack.

### Benchmarks
```bash
make bench KEY=<gateway key>
```
This runs the same workload four times and appends one JSON line per run to
`results/bench.jsonl`:
1. `direct`: straight to Ollama. This is the baseline.
2. `gateway`: through the gateway with the cache bypassed, to measure gateway overhead.
3. `gateway-cache`: half the prompts repeat an earlier one, to measure the cache's effect.
4. `gateway-stream`: streaming, to measure TTFT.

Each run records p50/p95/p99 latency, TTFT, throughput and tokens/s. See
[scripts/bench.py](scripts/bench.py) for every option.

## Evaluation & Results

- **Baseline:** the same requests sent directly to Ollama, with no gateway.
- **Metrics:** p50/p95 latency overhead the gateway adds, TTFT, completion tokens/s, cache
  hit rate and latency on hits, cost per 1K tokens, and availability when a backend fails.
- **Method:** `scripts/bench.py` with a fixed seed and prompt set. Failover is tested by
  stopping a backend during a run.
- **Results:** *not measured yet.* Real numbers on the RTX 4050 laptop GPU will go here
  once the stack is running.

## Milestones

Milestones follow [SHARED_STANDARDS.md](../SHARED_STANDARDS.md). Each one has a definition of done there.

### M1: Scaffold & baseline
- [x] Repo scaffold: `pyproject.toml`, Makefile, `.gitignore`, `.env.example`
- [x] Smoke test passes (`make test`)
- [x] Baseline verified live: passthrough `/v1/chat/completions` to Ollama (`llama3.2`, RTX 4050)

### M2: Core functionality
- [x] OpenAI-compatible chat + embeddings routes with SSE streaming
- [x] API-key auth
- [x] One adapter for every OpenAI-compatible backend (Ollama, vLLM, hosted); several Ollama nodes via a comma-separated `OLLAMA_BASE_URL`
- [x] Routing policy (cost / latency / quality) with retries, fallback and circuit breaker
- [x] Model aliasing across providers
- [ ] Guardrails: PII redaction, prompt-injection checks

### M3: Data / storage layer
- [x] Redis per-tenant rate limiting (fixed one-minute window)
- [x] Exact-match response cache (Redis, TTL, modes)
- [ ] Semantic cache (embedding similarity)
- [ ] Usage / cost ledger per tenant (PostgreSQL)

### M4: Evaluation & testing
- [x] Unit tests for router, limiter, cache, config and API (38 tests)
- [x] Integration tests against the real stack written (`tests/integration/`)
- [x] Integration tests run green against real Ollama + Redis (5/5)
- [x] Benchmark harness (`scripts/bench.py`) with a direct-to-backend baseline
- [x] Failover harness (`scripts/failover.py`) for killing a backend under load
- [ ] Benchmarks run on mains power; failover measured

### M5: Observability / monitoring
- [x] Prometheus metrics: TTFT, tokens/sec, p50/p95 latency, cache hit rate, cost
- [x] Grafana dashboard provisioned (`deploy/grafana/`)
- [ ] Dashboard verified with live traffic (screenshot in README)
- [ ] OpenTelemetry tracing

### M6: Containerization & CI/CD
- [x] Dockerfile and `docker-compose.yml` (gateway, Redis, Prometheus, Grafana; optional Ollama profile)
- [x] `make up` verified locally (Redis, Prometheus scraping the gateway, Grafana dashboard provisioned)
- [ ] `make up-all` (gateway container → host Ollama) verified locally
- [x] GitHub Actions: lint, test, image build and container smoke test
- [ ] Kubernetes manifests (optional)

### M7: Docs, benchmarks & demo
- [ ] README with measured results against the baseline
- [ ] Benchmark report (latency overhead, cache savings, failover)
- [ ] Demo recording or public deploy

## Project Layout
```
01-llm-gateway/
├── src/llm_gateway/
│   ├── __main__.py        # python -m llm_gateway
│   ├── main.py            # FastAPI app factory + routes
│   ├── router.py          # policy ranking, retries, fallback, circuit breaker
│   ├── backends.py        # OpenAI-compatible upstream adapter
│   ├── cache.py           # Redis response cache
│   ├── ratelimit.py       # Redis per-tenant rate limiter
│   ├── metrics.py         # Prometheus metrics, tokens, cost
│   └── config.py          # settings + backend list
├── tests/                 # unit tests; tests/integration/ = live stack
├── scripts/bench.py       # load test / benchmark
├── scripts/failover.py    # availability while a backend is killed
├── deploy/                # Prometheus config, Grafana provisioning + dashboard
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── .env.example
```
