#!/usr/bin/env bash
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=array.%A_%a.out
#SBATCH --error=array.%A_%a.err
set -eo pipefail
module load compilers/gcc/v12.2.0
export Geant4_DIR=/apps/soft/geant/geant4-v11.1.0/lib64/cmake/Geant4
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
ROOT=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928
cd "$ROOT"
offset=${ELLIPSE_JOB_INDEX_OFFSET:?Set the immutable task offset}
[[ "$offset" =~ ^[0-9]+$ ]] || exit 2
index=$((SLURM_ARRAY_TASK_ID + offset))
python3 experiments/ELLIPSE500x300_H120/simulation.py run --index "$index" \
  --executable Geant4Build/gamma01 \
  --crystal Geant4Sim/Geant4Code/CrystalMatrix.txt
