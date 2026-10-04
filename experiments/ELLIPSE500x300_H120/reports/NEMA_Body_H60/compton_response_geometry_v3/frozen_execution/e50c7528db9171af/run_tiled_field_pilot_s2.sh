#!/usr/bin/env bash
set -euo pipefail
exec 9>/home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/tiled_field_pilot_s2.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
cd /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/tiled_field_releases/e50c7528db9171af
timeout --signal=TERM --kill-after=30s 45m /home/lipeize/JSCC_FOV120_20260924/.venv/bin/python generate_compton_tiled_field_v3.py --root /home/lipeize/JSCC_FOV120_20260924 --plan /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/tiled_field_releases/e50c7528db9171af/tiled_production_plan.json --field /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/A440_guard_field --output /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/A440_tiled_field --phase pilot --shard 2 --shards 4 --max-new-tiles 21 --cuda 0 --physical-gpu 3
