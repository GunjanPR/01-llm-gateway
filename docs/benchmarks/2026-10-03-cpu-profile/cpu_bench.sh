#!/bin/bash
REPO=${REPO:-$(git rev-parse --show-toplevel)}
# usage: cpu_bench.sh label extra-bench-args...  (gateway on :8000 must be running)
label=$1; shift
. $REPO/.venv/bin/activate
GW=$(/tmp/prof/gwpid.sh)
t0=$(awk '{print $14+$15}' /proc/$GW/stat)
out=$(python $REPO/scripts/bench.py --label $label --api-key k --out res.jsonl "$@")
t1=$(awk '{print $14+$15}' /proc/$GW/stat)
n=$(echo "$out" | awk '/ requests:/{print $2}')
rps=$(echo "$out" | awk '/throughput_rps/{print $2}')
p50=$(echo "$out" | awk '/latency_p50_s/{print $2*1000}')
p99=$(echo "$out" | awk '/latency_p99_s/{print $2*1000}')
err=$(echo "$out" | awk '/ errors:/{print $2}')
tck=$(getconf CLK_TCK)
echo "$label n=$n err=$err rps=$rps p50=${p50}ms p99=${p99}ms gw_cpu_per_req=$(python3 -c "print(round(($t1-$t0)/$tck*1000/$n,3))")ms"
