#!/bin/bash
REPO=${REPO:-$(git rev-parse --show-toplevel)}
. $REPO/.venv/bin/activate
B=$REPO/scripts/bench.py
for round in 1 2; do
for v in "before /tmp/wt-pre/src" "after $REPO/src"; do
  set -- $v
  /tmp/prof/mock.sh --delay-ms 500 --parallel 8
  BACKEND_MAX_CONCURRENCY=0 /tmp/prof/restart.sh $2
  out=$(python $B --label coalesce-$1 --api-key k --requests 400 --concurrency 64 --repeat-ratio 0.8 --warmup 0 --out /tmp/prof/coalesce.jsonl)
  calls=$(curl -s 127.0.0.1:8000/metrics | awk '/^llmgw_upstream_latency_seconds_count\{.*route="chat"/{s+=$2} END{print s}')
  echo "$1: backend_calls=$calls $(echo "$out" | awk '/throughput_rps|latency_p50_s|latency_p95_s|cache_hits| errors/{printf "%s=%s ",$1,$2}')"
done; done
