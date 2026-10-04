#!/usr/bin/env bash
set -euo pipefail
exec 9>/home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/tile_pilot.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/tile_pilot_releases/f7980c4c5ea40eaa
timeout --signal=TERM --kill-after=30s 45m /home/lipeize/JSCC_FOV120_20260924/.venv/bin/python generate_compton_a_tile_pilot_v3.py --root /home/lipeize/JSCC_FOV120_20260924 --plan /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/tile_pilot_releases/f7980c4c5ea40eaa/A_tile_plan.json --field /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/A440_guard_field --output /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/A440_tile_pilot --cuda 0
