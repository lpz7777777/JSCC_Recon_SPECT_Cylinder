#!/usr/bin/env bash
#SBATCH --job-name=ELLIPSE_XCAT_collect
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=collect_xcat.%j.out
#SBATCH --error=collect_xcat.%j.err
set -euo pipefail
ROOT=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928
cd "$ROOT"
python3 experiments/FOV120/workflow.py collect \
  --manifest experiments/ELLIPSE500x300_H120/generated/XCAT_1e9_fullx/jobs.json \
  --dataset XCAT --level 1e9 \
  --destination experiments/ELLIPSE500x300_H120/generated
