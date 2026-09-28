#!/usr/bin/env bash
set -euo pipefail
cd /home/lipeize/JSCC_FOV120_20260924
base=experiments/ELLIPSE500x300_H120
first_pid=${1:?Pass PID of the A218 slab00 Python controller}
first="$base/generated/ScatterSlabs/A218/slab00/complete.json"
for _ in {1..720}; do
  if [[ -f "$first" ]]; then break; fi
  if ! kill -0 "$first_pid" 2>/dev/null; then
    echo "Initial A218 slab exited without a completion receipt" >&2
    exit 1
  fi
  sleep 60
done
[[ -f "$first" ]] || { echo "Initial slab timed out" >&2; exit 1; }
for _ in {1..720}; do
  if grep -q ELLIPSE_SCATTER_SLABS_PREPARED "$base/scatter_prepare.log"; then break; fi
  sleep 60
done
grep -q ELLIPSE_SCATTER_SLABS_PREPARED "$base/scatter_prepare.log" || {
  echo "Slab preparation failed or timed out" >&2; exit 1;
}
for response in A218 A440 C440to218; do
  for index in 0 1 2 3; do
    if [[ "$response" == A218 && "$index" == 0 ]]; then continue; fi
    .venv/bin/python -u "$base/run_scatter_slabs.py" run "$response" "$index" --cuda 0
  done
  .venv/bin/python -u "$base/run_scatter_slabs.py" stitch "$response"
done
.venv/bin/python -u "$base/run_factor_conversion.py"
echo ELLIPSE_SCATTER_AND_FACTORS_COMPLETE
