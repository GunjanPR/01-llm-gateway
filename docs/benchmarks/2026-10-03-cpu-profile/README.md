# Gateway CPU cost per request (2026-10-03)

**What:** CPU time the gateway process spends per request, with the model replaced by
`scripts/mock_upstream.py` (0 ms latency), so only the gateway's own work is measured.

**Setup:** 4-vCPU Linux cloud VM (slower per core than the laptop), Python 3.11, Redis 7
local, rate limiter off, semantic cache off, tracing off. Load: `scripts/bench.py`,
concurrency 32, 3,000 requests per run (2,000 for streaming). The client runs on the same
machine, so throughput (rps) is client-bound and noisy; the stable number is **gateway
CPU ms per request** (`utime+stime` of the gateway process from `/proc`, divided by
requests), measured by [cpu_bench.sh](cpu_bench.sh). Variants alternated, 0 errors.

| Change | Cache miss | Cache hit | Streaming |
|---|---|---|---|
| Before (`@app.middleware("http")`, httpx transport) | 3.98 ms | 1.63 ms | – |
| Plain ASGI middleware | 3.25 ms (−18%) | 1.12 ms (−31%) | 4.15 ms |
| + aiohttp transport under httpx | **1.61 ms (−60% total)** | 1.18 ms | **2.15 ms (−48%)** |

Means of 2 runs (middleware) and 3 runs (transport); raw lines in
[middleware_ab.txt](middleware_ab.txt) and [transport_ab.txt](transport_ab.txt).

**Why:** py-spy showed Starlette's `BaseHTTPMiddleware` (an extra task group and memory
streams per request) and httpcore's connection pool (it scans every pooled connection,
with a `select()` per socket, on each request) as the two biggest costs. The middleware
is now plain ASGI, and backend calls go through `httpx_aiohttp.AiohttpTransport`, keeping
the httpx API. `BACKEND_HTTP_TRANSPORT=httpx` switches back.

**What it means on real hardware:** with a real model the GPU dominates latency, so this
doesn't change single-request latency much. It raises how many concurrent requests one
gateway process can proxy before the CPU becomes the limit: at 1.6 ms per uncached
request, one core handles about 600 requests/s, up from about 250.
