#!/usr/bin/env bash
set -euo pipefail
exec 9>/home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/regional_a_g2.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/regional_a_g2_releases/63d405cde5eef50a
timeout --signal=TERM --kill-after=30s 45m /home/lipeize/JSCC_FOV120_20260924/.venv/bin/python generate_compton_regional_a_v3.py --root /home/lipeize/JSCC_FOV120_20260924 --plan /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/regional_a_g2_releases/63d405cde5eef50a/regional_a_plan.json --group 2 --output /home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/regional_a_g2 --cuda 0
