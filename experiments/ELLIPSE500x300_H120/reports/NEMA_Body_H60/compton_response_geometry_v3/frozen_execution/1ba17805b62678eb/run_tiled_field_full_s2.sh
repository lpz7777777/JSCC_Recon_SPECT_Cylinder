#!/usr/bin/env bash
set -euo pipefail
exec 9>/home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/tiled_field_full_s2.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
cd /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/tiled_field_releases/1ba17805b62678eb
timeout --signal=TERM --kill-after=30s 24h /home/lipeize/JSCC_FOV120_20260924/.venv/bin/python generate_compton_tiled_field_v3.py --root /home/lipeize/JSCC_FOV120_20260924 --plan /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/tiled_field_releases/1ba17805b62678eb/tiled_production_plan.json --field /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/A440_guard_field --output /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/A440_tiled_field --phase full --shard 2 --shards 4 --max-new-tiles 10720 --cuda 0 --physical-gpu 3 --run-id full_batch1 --stop-after-seconds 72000
