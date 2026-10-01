# Production LLM Inference Gateway

An OpenAI-compatible gateway in front of vLLM/Ollama (and optional hosted APIs). It handles routing, caching, rate limiting, fallback, and cost tracking, with full observability.

> Part of the Live Projects portfolio. See [SHARED_STANDARDS.md](../SHARED_STANDARDS.md) for conventions and the milestone definition of done.

## Overview

Applications that call LLMs directly end up re-implementing auth, retries, caching, and
cost tracking in every service. This gateway puts all of that behind one
**OpenAI-compatible endpoint** (`/v1/chat/completions`, `/v1/embeddings`). Any client,
including the [Healthcare RAG](../02-healthcare-rag/) project, can then use local
models through vLLM or Ollama, or hosted models, without code changes.

Planned features:
- API-key auth and per-tenant rate limiting (Redis token bucket)
- Model routing by cost, latency, or quality policy, with fallback and retries
- Exact and semantic response caching
- Streaming responses (SSE)
- Guardrails: PII redaction and prompt-injection checks
- Usage ledger and cost per 1K tokens
- Prometheus metrics (TTFT, tokens/sec, p50/p95 latency, cache hit rate) and Grafana dashboards

## Tech Stack

| Layer | Tools |
|---|---|
| API | FastAPI (async), Uvicorn, httpx |
| Cache / rate limits | Redis |
| Model backends | vLLM (GPU) or Ollama (CPU/laptop), optional hosted APIs |
| Observability | Prometheus, Grafana, OpenTelemetry |
| Load testing | Locust or k6 |
| Infra | Docker Compose, Kubernetes (later milestones) |

## Architecture

| Component | Responsibility |
|---|---|
| **API layer** | OpenAI-compatible routes, request validation, SSE streaming |
| **Auth & rate limiter** | API-key lookup, per-tenant token-bucket limits in Redis |
| **Cache** | Exact-match (hash of normalized request) and semantic (embedding similarity) caches in Redis |
| **Router** | Chooses a backend by policy (cost / latency / quality); handles retries and fallback |
| **Backend adapters** | Thin clients for vLLM, Ollama, and hosted APIs behind one interface |
| **Guardrails** | PII redaction and prompt-injection checks, before and after the model call |
| **Telemetry** | Prometheus `/metrics`, OpenTelemetry traces, usage/cost ledger |

```
          clients (curl, SDKs, 02-healthcare-rag)
                         │  OpenAI-compatible HTTP
                         ▼
┌───────────────────────────────────────────────────────────┐
│                  LLM Gateway (FastAPI)                    │
│                                                           │
│  auth + rate limit ──► guardrails ──► cache lookup ──┐    │
│        │                                   hit │     │miss│
│        ▼                                       ▼     ▼    │
│     Redis ◄──────────── response cache ◄──── router       │
│                                         (policy, retry,   │
│                                          fallback)        │
│  /metrics, traces ──► Prometheus ──► Grafana   │          │
└────────────────────────────────────────────────┼──────────┘
                    ┌────────────────┬───────────┴────────┐
                    ▼                ▼                    ▼
               vLLM server      Ollama server       hosted API (optional)
```

## Setup

### Prerequisites
- Python 3.11
- Redis 7+ (local install or `docker run -p 6379:6379 redis:7`)
- At least one model backend:
  - **Ollama** (easiest on a laptop): install from https://ollama.com, then `ollama pull llama3.2`
  - **vLLM** (Linux + NVIDIA GPU, or WSL2): `pip install vllm`, then
    `vllm serve <model> --port 8001`
- Optional: Docker Desktop (for Prometheus + Grafana, and the full stack from M6)
- Optional: GNU Make (see the Windows note below)

### Install
```bash
git clone <repo-url> 01-llm-gateway && cd 01-llm-gateway
py -3.11 -m venv .venv            # Windows; on macOS/Linux use: python3.11 -m venv .venv
source .venv/Scripts/activate     # Git Bash; PowerShell: .venv\Scripts\Activate.ps1; macOS/Linux: source .venv/bin/activate
make install                      # = pip install -e ".[dev]"
```

### Environment config
```bash
cp .env.example .env              # then fill in values; .env is git-ignored
```
Key variables: `GATEWAY_API_KEYS` (comma-separated client keys), `REDIS_URL`,
`OLLAMA_BASE_URL` / `VLLM_BASE_URL`, `DEFAULT_MODEL`, and `ROUTING_POLICY`.
Hosted-API keys are optional and can stay blank.

## How to Run

> **Windows note:** `make` isn't installed on Windows by default. Install it with `choco install make`, or run the
> targets from **Git Bash**. You can also run the commands in the `Makefile` yourself
> (for example, `python -m pytest` instead of `make test`).

### Run the gateway
```bash
# 1. Start dependencies
docker run -d --name redis -p 6379:6379 redis:7
ollama serve                          # or: vllm serve <model> --port 8001

# 2. Start the gateway on http://localhost:8000
make run                              # = uvicorn llm_gateway.main:app --reload --port 8000
```
> `make run` starts working at **M2**, once `src/llm_gateway/main.py` (the FastAPI app) is added.

### Try it
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer <a key from GATEWAY_API_KEYS>" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "messages": [{"role": "user", "content": "Hello"}]}'
```
Metrics are served at `http://localhost:8000/metrics`. Grafana dashboards come in M5.

### Lint and test
```bash
make lint     # ruff check + ruff format --check
make test     # pytest (unit tests only; integration tests are marked `integration`)
make format   # auto-fix lint and formatting
```

## Evaluation & Results

- **Baseline:** a direct client-to-backend call with no gateway.
- **Metrics:** p50/p95 latency overhead the gateway adds, TTFT, tokens/sec, cache hit rate,
  cost per 1K tokens, and availability under injected backend failures.
- **Method:** Locust/k6 load scenarios with fixed prompt sets and failure injection.
- **Results:** *targets until measured*: cache-driven cost reduction, and no availability
  loss when one backend fails.

## Milestones

Milestones follow [SHARED_STANDARDS.md](../SHARED_STANDARDS.md). Each one has a definition of done there.

### M1: Scaffold & baseline
- [x] Repo scaffold: `pyproject.toml`, Makefile, `.gitignore`, `.env.example`
- [x] Smoke test passes (`make test`)
- [ ] Baseline: passthrough `/v1/chat/completions` to a single Ollama backend

### M2: Core functionality
- [ ] OpenAI-compatible chat + embeddings routes with SSE streaming
- [ ] API-key auth
- [ ] Backend adapters (Ollama, vLLM) behind a common interface
- [ ] Routing policy (cost / latency / quality) with retries and fallback

### M3: Data / storage layer
- [ ] Redis per-tenant rate limiting (token bucket)
- [ ] Exact-match response cache
- [ ] Semantic cache (embedding similarity)
- [ ] Usage / cost ledger

### M4: Evaluation & testing
- [ ] Unit tests for router, limiter, cache, and guardrails
- [ ] Integration tests against a mock backend
- [ ] Locust/k6 load tests with failure injection; baseline = direct backend call

### M5: Observability / monitoring
- [ ] Prometheus metrics: TTFT, tokens/sec, p50/p95 latency, cache hit rate, cost/1K tokens
- [ ] OpenTelemetry tracing
- [ ] Grafana dashboard JSON committed

### M6: Containerization & CI/CD
- [ ] Dockerfile and `docker compose up` (gateway, Redis, Ollama, Prometheus, Grafana)
- [ ] GitHub Actions: lint + test
- [ ] Kubernetes manifests (optional)

### M7: Docs, benchmarks & demo
- [ ] README with measured results against the baseline
- [ ] Benchmark report (latency overhead, cache savings, failover)
- [ ] Demo recording or public deploy

## Project Layout
```
01-llm-gateway/
├── src/llm_gateway/__init__.py
├── tests/test_smoke.py
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md
```
