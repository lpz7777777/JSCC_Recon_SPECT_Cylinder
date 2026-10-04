#!/usr/bin/env bash
set -euo pipefail
exec 9>/home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/guard_column.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/guard_column_releases/6e7adfc3ead95a5d
timeout --signal=TERM --kill-after=30s 45m /home/lipeize/JSCC_FOV120_20260924/.venv/bin/python generate_compton_a_column_v3.py --root /home/lipeize/JSCC_FOV120_20260924 --patches /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/A440_near_patch_ultrafine --output /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/A440_near_column --cuda 0
