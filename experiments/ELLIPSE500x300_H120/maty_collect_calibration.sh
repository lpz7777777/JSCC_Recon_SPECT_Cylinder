#!/usr/bin/env bash
#SBATCH --job-name=ELLIPSE_collect_cal
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=collect_calibration.%j.out
#SBATCH --error=collect_calibration.%j.err
set -euo pipefail
ROOT=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928
cd "$ROOT"
base=experiments/ELLIPSE500x300_H120/generated
for dataset in calibration_218 calibration_440 sensitivity_440 sensitivity_validation_440; do
  marker="$base/collections/${dataset}_all.json"
  if [[ -f "$marker" ]]; then continue; fi
  python3 experiments/ELLIPSE500x300_H120/simulation.py collect \
    --dataset "$dataset" --level all
done
python3 experiments/ELLIPSE500x300_H120/bundle_calibration_inputs.py create \
  --data "$base" --archive "$base/calibration_transfer.tar.gz"
echo ELLIPSE_CALIBRATION_COLLECTION_COMPLETE
