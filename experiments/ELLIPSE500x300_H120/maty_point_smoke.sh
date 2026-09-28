#!/usr/bin/env bash
#SBATCH --job-name=ELLIPSE_point_smoke
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=point_smoke.%j.out
#SBATCH --error=point_smoke.%j.err
set -eo pipefail
module load compilers/gcc/v12.2.0
export Geant4_DIR=/apps/soft/geant/geant4-v11.1.0/lib64/cmake/Geant4
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
ROOT=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928
cd "$ROOT"
for index in 0 3560 3580 3960; do
  python3 experiments/FOV120/workflow.py run \
    --manifest experiments/ELLIPSE500x300_H120/generated/PointScan/jobs.json \
    --index "$index" --smoke --executable Geant4Build/gamma01 \
    --crystal Geant4Sim/Geant4Code/CrystalMatrix.txt
done
echo ELLIPSE_POINT_SMOKE_COMPLETE
