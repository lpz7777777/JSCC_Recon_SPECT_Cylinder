#!/usr/bin/env bash
set -euo pipefail
cd /home/lipeize/JSCC_FOV120_20260924
for response in A218 A440 C440to218; do
  for index in 0 1 2 3; do
    if [[ "$response" == A218 && "$index" == 0 ]]; then continue; fi
    .venv/bin/python -u experiments/ELLIPSE500x300_H120/run_scatter_slabs.py \
      prepare "$response" "$index"
  done
done
echo ELLIPSE_SCATTER_SLABS_PREPARED
