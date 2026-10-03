#!/bin/bash
# Overload experiment: each mock "GPU" runs 4 requests at once x 200 ms = 20 req/s.
# Client: 64 concurrent users, 2 s timeout (an SLA), 600 requests.
. /home/claude/01-llm-gateway/.venv/bin/activate
B=/home/claude/01-llm-gateway/scripts/bench.py
OUT=/tmp/claude-0/prof/overload.jsonl
mocks() {
  pkill -f "mock_upstrea[m]"; sleep 1
  for port in 9100 9101; do
    nohup python /home/claude/01-llm-gateway/scripts/mock_upstream.py --port $port --delay-ms 200 --parallel 4 > /dev/null 2>&1 &
  done
  sleep 2
}
run() {  # label, src, env...
  label=$1; src=$2; shift 2
  mocks
  env "$@" /tmp/claude-0/prof/restart.sh $src
  python $B --label $label --api-key k --requests 600 --concurrency 64 --no-cache --timeout 2 --warmup 0 --out $OUT > /dev/null
  python3 -c "
import json; r=[json.loads(l) for l in open('$OUT')][-1]
ok=r['requests']-r['errors']
print(f\"{r['label']:<26} ok={ok:>4}/{r['requests']} goodput={r['throughput_rps']:5.1f}/s p50={r['latency_p50_s']*1000:5.0f}ms p95={r['latency_p95_s']*1000:5.0f}ms first_err={(r['sample_errors'] or ['-'])[0][:40]}\")"
}
BASE=/tmp/claude-0/wt-base/src
NEW=/home/claude/01-llm-gateway/src
TWO=OLLAMA_BASE_URL=http://127.0.0.1:9100,http://127.0.0.1:9101
for round in 1 2; do
run before-1node $BASE
run after-1node $NEW BACKEND_MAX_CONCURRENCY=4 BACKEND_QUEUE_TIMEOUT_SECONDS=1
run before-2nodes $BASE $TWO
run after-2nodes $NEW BACKEND_MAX_CONCURRENCY=4 BACKEND_QUEUE_TIMEOUT_SECONDS=1 $TWO
done
