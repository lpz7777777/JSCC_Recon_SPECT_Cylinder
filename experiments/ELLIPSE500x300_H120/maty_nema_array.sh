#!/usr/bin/env bash
#SBATCH --job-name=ELLIPSE_NEMA_H60
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928/experiments/ELLIPSE500x300_H120/generated/NEMA_Body_H60/Simulation_1e9/logs/%x.%A_%a.out
#SBATCH --error=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928/experiments/ELLIPSE500x300_H120/generated/NEMA_Body_H60/Simulation_1e9/logs/%x.%A_%a.err
set -euo pipefail
source /etc/profile.d/modules.sh
module load compilers/gcc/v12.2.0
export Geant4_DIR=/apps/soft/geant/geant4-v11.1.0/lib64/cmake/Geant4
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh

root=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928
cd "$root"
manifest=experiments/ELLIPSE500x300_H120/generated/NEMA_Body_H60/Simulation_1e9/jobs.json
if [[ ${NEMA_SMOKE:-0} == 1 ]]; then
  python3 experiments/FOV120/workflow.py run \
    --manifest "$manifest" --index "${SLURM_ARRAY_TASK_ID:?Array index required}" \
    --executable Geant4Build/gamma01 \
    --crystal Geant4Sim/Geant4Code/CrystalMatrix.txt --smoke
else
  python3 experiments/FOV120/workflow.py run \
    --manifest "$manifest" --index "${SLURM_ARRAY_TASK_ID:?Array index required}" \
    --executable Geant4Build/gamma01 \
    --crystal Geant4Sim/Geant4Code/CrystalMatrix.txt
fi
