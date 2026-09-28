#!/usr/bin/env bash
#SBATCH --job-name=ELLIPSE_sources_smoke
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=source_smoke.%j.out
#SBATCH --error=source_smoke.%j.err
set -eo pipefail
module load compilers/gcc/v12.2.0
export Geant4_DIR=/apps/soft/geant/geant4-v11.1.0/lib64/cmake/Geant4
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
ROOT=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928
cd "$ROOT"
for index in 0 400 800 1200 1400; do
  python3 experiments/ELLIPSE500x300_H120/simulation.py run --index "$index" --smoke \
    --executable Geant4Build/gamma01 \
    --crystal Geant4Sim/Geant4Code/CrystalMatrix.txt
done
echo ELLIPSE_SELECTED_SMOKE_COMPLETE
