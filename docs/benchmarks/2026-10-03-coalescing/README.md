# Request coalescing: identical concurrent misses share one backend call (2026-10-03)

**Scenario:** a burst of popular prompts on a cold cache. 400 requests from 64 concurrent
users; 80% repeat an earlier prompt (`scripts/bench.py --repeat-ratio 0.8`). Backend:
`scripts/mock_upstream.py --delay-ms 500 --parallel 8` (a GPU that runs 8 answers at once).
Concurrency cap off in both runs. Backend calls are counted from the gateway's
`llmgw_upstream_latency_seconds_count`. Two rounds; script [coalesce.sh](coalesce.sh).

| | Backend calls | Throughput | p50 | p95 | Errors |
|---|---|---|---|---|---|
| Before (exact cache only) | 186 / 181 | 32.6 / 34.0 req/s | 0.02 s | 4.01 s | 0 |
| After (coalescing) | **86 / 86** | **67.9 / 68.2 req/s** | 0.58 / 0.48 s | **2.02 / 2.40 s** | 0 |

**Why:** the exact cache only helps once the first answer is stored. Until then, every
copy of a popular prompt that arrives while the first is still generating went to the
GPU too: about 100 duplicate generations in this run. Now the first request for a cache
key is the leader, and identical requests that arrive while it runs wait for its result
(`x-gateway-cache: COALESCED`).

**Trade-off:** the median rose because requests that would have been instant cache hits
late in the "before" run now arrive while the leader is still generating, and wait for
it. The whole burst finishes in half the time, and p95 halves. If the leader fails or its
client disconnects, each follower makes its own call, so one bad request doesn't fail the
others. Only requests that would be cached are coalesced; streaming and
`Cache-Control: no-cache` requests are not.
