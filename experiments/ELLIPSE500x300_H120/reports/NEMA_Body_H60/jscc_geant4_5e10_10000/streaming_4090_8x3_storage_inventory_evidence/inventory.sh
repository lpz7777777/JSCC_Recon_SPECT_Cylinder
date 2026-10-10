#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load miniforge3/25.11.0-1
srun --label --kill-on-bad-exit=1 /data/home/scxi717/.conda/envs/torch/bin/python /data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/generated/jscc_geant4_5e10_10000/storage_inspection/streaming_4090_8x3_storage_inventory/inventory.py
