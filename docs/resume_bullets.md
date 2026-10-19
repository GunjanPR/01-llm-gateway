# Resume bullets (LLM Gateway)

Every number here is measured; the source is linked. The Resumes folder is not edited
from this repo; copy what you want from here.

- Built an OpenAI-compatible LLM gateway (FastAPI, Redis, PostgreSQL, Prometheus/Grafana,
  OpenTelemetry) in front of Ollama/vLLM with cost/latency/quality routing, retries,
  circuit breakers, exact and semantic caching, PII guardrails and a per-tenant usage
  ledger; adds **+1.8 ms at p50** over calling Ollama directly on 1-token requests
  (+4.2 ms before the optimization pass; README, local results).
- Cut the gateway's CPU per uncached request by **60% (3.98 → 1.61 ms)** after profiling
  with py-spy: replaced Starlette's BaseHTTPMiddleware with plain ASGI middleware and
  moved backend calls to an aiohttp transport (docs/benchmarks/2026-10-03-cpu-profile).
- Added load shedding with spillover and cancel-on-disconnect: at 3× overload with a 2 s
  client timeout, goodput went from **1.9 to 19.3 req/s** on one server (its maximum)
  and **38.6 req/s** on two (docs/benchmarks/2026-10-03-overload).
- Coalesced identical in-flight requests, halving backend calls (**186 → 86**) and doubling
  throughput (**33 → 68 req/s**) on a cold-cache burst (docs/benchmarks/2026-10-03-coalescing).
- Kept **100% availability (469/469 requests)** when a backend was killed under load
  (README, failover).
- Diagnosed a Windows `localhost` → IPv6 fallback that cost **~2 s per raw connection**
  (2,052 ms vs 0.4 ms) and fixed it in config handling, cutting first-request latency
  **534 → 179 ms** and post-idle latency **590 → 344 ms**, within 10 ms of calling the
  model directly (docs/benchmarks/2026-10-03-localhost).
- Extended PII redaction to streamed answers, where an entity arrives split across chunks:
  leaked entities fell from **2,010 of 2,010 to 0** on a 500-answer benchmark, with text
  held back a median of 4 characters and about 11 µs of CPU per chunk; a property test
  checks any chunking matches one-shot redaction (docs/benchmarks/2026-10-19-stream-redaction).
- Wrote 13 Prometheus alert rules unit-tested with promtool, and 159 unit tests including
  failure-injection, concurrency and property-based tests.

Caveat to keep in mind in interviews: the CPU, overload and coalescing numbers use a mock
backend on a cloud VM to isolate the gateway; the overhead, failover and localhost numbers are on
the RTX 4050 laptop with real Ollama.
