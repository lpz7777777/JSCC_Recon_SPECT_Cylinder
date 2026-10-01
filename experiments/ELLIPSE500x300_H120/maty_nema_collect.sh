#!/usr/bin/env bash
#SBATCH --job-name=ELLIPSE_NEMA_COLLECT
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
set -euo pipefail
root=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928
cd "$root"
level=${NEMA_LEVEL:?Set NEMA_LEVEL explicitly}
case "$level" in 1e9|5e9|1e10) ;; *) echo "Invalid NEMA level" >&2; exit 1 ;; esac
base=experiments/ELLIPSE500x300_H120
manifest="$base/generated/NEMA_Body_H60/Simulation_${level}/jobs.json"
python3 "$base/validate_nema_simulation.py" "$manifest" --stage complete
python3 experiments/FOV120/workflow.py collect --manifest "$manifest" \
  --dataset NEMA_Body_H60 --level "$level" --destination "$base/generated"
python3 "$base/package_imaging.py" --generated "$base/generated" \
  --datasets NEMA_Body_H60 --level "$level" --archive-stem "nema_h60_imaging_${level}"
