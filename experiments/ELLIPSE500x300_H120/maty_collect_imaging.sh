#!/usr/bin/env bash
#SBATCH --job-name=ELLIPSE_collect_1e9
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=collect_imaging.%A_%a.out
#SBATCH --error=collect_imaging.%A_%a.err
set -euo pipefail
ROOT=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928
cd "$ROOT"
case "$SLURM_ARRAY_TASK_ID" in
  0) dataset=CircleNewDist ;;
  1) dataset=EllipseUniform ;;
  2) dataset=EllipseContrast ;;
  *) echo "Invalid collector index" >&2; exit 2 ;;
esac
python3 experiments/ELLIPSE500x300_H120/simulation.py collect \
  --dataset "$dataset" --level 1e9
