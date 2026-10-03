#!/bin/bash
. /home/claude/01-llm-gateway/.venv/bin/activate
B=/home/claude/01-llm-gateway/scripts/bench.py
for round in 1 2; do
for v in "before /tmp/claude-0/wt-pre/src" "after /home/claude/01-llm-gateway/src"; do
  set -- $v
  /tmp/claude-0/prof/mock.sh --delay-ms 500 --parallel 8
  BACKEND_MAX_CONCURRENCY=0 /tmp/claude-0/prof/restart.sh $2
  out=$(python $B --label coalesce-$1 --api-key k --requests 400 --concurrency 64 --repeat-ratio 0.8 --warmup 0 --out /tmp/claude-0/prof/coalesce.jsonl)
  calls=$(curl -s 127.0.0.1:8000/metrics | awk '/^llmgw_upstream_latency_seconds_count\{.*route="chat"/{s+=$2} END{print s}')
  echo "$1: backend_calls=$calls $(echo "$out" | awk '/throughput_rps|latency_p50_s|latency_p95_s|cache_hits| errors/{printf "%s=%s ",$1,$2}')"
done; done
