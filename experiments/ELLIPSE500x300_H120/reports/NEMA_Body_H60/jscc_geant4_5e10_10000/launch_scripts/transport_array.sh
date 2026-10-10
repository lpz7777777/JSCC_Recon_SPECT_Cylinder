#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load compilers/gcc/v12.2.0
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[[ "${SLURM_CPUS_PER_TASK}" == 1 && "${SLURM_NNODES}" == 1 && "${SLURM_NTASKS}" == 1 ]]
index="${SLURM_ARRAY_TASK_ID}"
printf -v suffix "worker_%04d" "$index"
srun --chdir=/tmp --kill-on-bad-exit=1 --wait=0 python3 /WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928/experiments/ELLIPSE500x300_H120/generated/jscc_geant4_5e10_10000/transport_array_releases/bd18503c61de2012/jscc_5e10_transport.py worker --release /WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928/experiments/ELLIPSE500x300_H120/generated/jscc_geant4_5e10_10000/transport_array_releases/bd18503c61de2012 --simulation /WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928/experiments/ELLIPSE500x300_H120/generated/jscc_geant4_5e10_10000/source_registry --output /WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928/experiments/ELLIPSE500x300_H120/generated/jscc_geant4_5e10_10000/transport_array_bd18503c61de2012/"$suffix" --index "$index" --limit 32348
