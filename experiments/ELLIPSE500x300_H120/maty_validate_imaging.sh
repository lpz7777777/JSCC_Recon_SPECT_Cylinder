#!/usr/bin/env bash
#SBATCH --job-name=ELLIPSE_gate_1e9
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=validate_imaging.%j.out
#SBATCH --error=validate_imaging.%j.err
set -euo pipefail
ROOT=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928
cd "$ROOT"
python3 experiments/ELLIPSE500x300_H120/validate_1e9_inputs.py \
  --root experiments/ELLIPSE500x300_H120/generated \
  --report experiments/ELLIPSE500x300_H120/generated/Rule_1e9_input_closure.json
