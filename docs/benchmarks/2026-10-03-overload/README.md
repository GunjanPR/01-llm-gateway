# Overload: load shedding, spillover and disconnect cancellation (2026-10-03)

**Scenario:** offered load is about 3× what the model servers can do, and clients give up
after 2 s (a typical interactive timeout). Each model server is `scripts/mock_upstream.py
--delay-ms 200 --parallel 4`: like a GPU, it generates 4 answers at once and queues the
rest, so it serves at most 20 requests/s. Client: `scripts/bench.py`, 64 concurrent users,
600 requests, `--timeout 2`. Like Ollama, the mock stops generating when its client
disconnects. Mocks restarted before every run; 2 rounds; script in
[overload.sh](overload.sh), raw results in [raw.jsonl](raw.jsonl).

- **Before:** commit a86949b (no concurrency cap, no cancellation on disconnect).
- **After:** `BACKEND_MAX_CONCURRENCY=4`, `BACKEND_QUEUE_TIMEOUT_SECONDS=1`,
  cancellation on disconnect (default on).

| Setup | Succeeded (round 1 / 2) | Goodput | p95 of successes | What failed |
|---|---|---|---|---|
| 1 server, before | 36 / 36 of 600 | 1.9 req/s | 1.94 s | client timeouts |
| 1 server, after | 191 / 188 of 600 | **19.3 req/s** (server max: 20) | 1.21–1.23 s | fast 503 + `Retry-After` |
| 2 servers, before | 36 / 36 of 600 | 1.8 req/s | 1.97 s | client timeouts |
| 2 servers, after | 367 / 359 of 600 | **38.6 req/s** (max: 40) | 1.21–1.22 s | fast 503 + `Retry-After` |

**Why "before" collapses:** when a client times out, the gateway kept its backend call
running, so the GPU kept generating answers nobody would read. Those orphans filled the
server's queue, every new request waited behind them, and nearly all of them timed out
too. Goodput fell to under 10% of capacity. With a second server it got no better,
because the cost policy ranks both equally and config order sent everything to the first.

**What changed:** each backend takes at most `BACKEND_MAX_CONCURRENCY` calls at once.
When it is full, the request goes to the next backend (spillover), so the second server
is used; when all are full, it waits up to `BACKEND_QUEUE_TIMEOUT_SECONDS` and then gets
503 with `Retry-After: 1` instead of a timeout. If the client disconnects, the backend call
is cancelled and its slot freed. The servers stay at full capacity and every request
that is admitted finishes within the client's deadline.

**Caveat:** synthetic backends with fixed 200 ms generation. On a real GPU, set
`BACKEND_MAX_CONCURRENCY` to roughly what the server runs in parallel (Ollama's
`OLLAMA_NUM_PARALLEL`) plus a small queue.
