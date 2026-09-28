#!/usr/bin/env bash
set -euo pipefail
cd /home/lipeize/JSCC_FOV120_20260924
base=experiments/ELLIPSE500x300_H120
first_pid=${1:?Pass active A218/slab00 Python PID}
cross_pid=${2:?Pass active C440to218 GPU1 lane PID}

receipt() { [[ -f "$base/generated/ScatterSlabs/$1/slab0$2/complete.json" ]]; }
run_slab() {
  local response=$1 index=$2 gpu=$3
  if receipt "$response" "$index"; then
    echo "Already complete: $response/$index"; return
  fi
  .venv/bin/python -u "$base/run_scatter_slabs.py" run "$response" "$index" --cuda "$gpu"
}

# These disjoint queues only use the idle GPUs; GPU1 retains the existing cross lane.
(run_slab A218 1 2; run_slab A218 2 2) > "$base/scatter_gpu2.log" 2>&1 & p2=$!
(run_slab A218 3 3; run_slab A440 0 3) > "$base/scatter_gpu3.log" 2>&1 & p3=$!
(run_slab A440 1 4; run_slab A440 2 4) > "$base/scatter_gpu4.log" 2>&1 & p4=$!

for _ in {1..720}; do
  if receipt A218 0; then break; fi
  kill -0 "$first_pid" 2>/dev/null || { echo "Initial A218 slab failed" >&2; exit 1; }
  sleep 60
done
receipt A218 0 || { echo "Initial A218 slab timed out" >&2; exit 1; }
run_slab A440 3 0 > "$base/scatter_gpu0_tail.log" 2>&1
wait "$p2"; wait "$p3"; wait "$p4"

for _ in {1..720}; do
  ready=true
  for index in 0 1 2 3; do receipt C440to218 "$index" || ready=false; done
  if "$ready"; then break; fi
  kill -0 "$cross_pid" 2>/dev/null || { echo "Cross lane exited without all receipts" >&2; exit 1; }
  sleep 60
done
for response in A218 A440 C440to218; do
  for index in 0 1 2 3; do receipt "$response" "$index" || exit 1; done
  .venv/bin/python -u "$base/run_scatter_slabs.py" stitch "$response"
done
.venv/bin/python -u "$base/run_factor_conversion.py"
echo ELLIPSE_SCATTER_AND_FACTORS_COMPLETE
