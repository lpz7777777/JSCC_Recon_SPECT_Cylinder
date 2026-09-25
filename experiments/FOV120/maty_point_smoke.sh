#!/usr/bin/env bash
#SBATCH --job-name=FOV120_point_smoke
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=experiments/FOV120/generated/point_smoke.%j.out
#SBATCH --error=experiments/FOV120/generated/point_smoke.%j.err
set -eo pipefail
module load compilers/gcc/v12.2.0
export Geant4_DIR=/apps/soft/geant/geant4-v11.1.0/lib64/cmake/Geant4
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
set -u
# Both energies, central source and r=135/z=57 mm corner of the scan.
for index in 400 480 481 561; do
  python3 experiments/FOV120/workflow.py run --index "$index" --smoke \
    --executable experiments/FOV120/generated/Geant4Build/gamma01 \
    --crystal Geant4Sim/Geant4Code/CrystalMatrix.txt
done
echo FOV120_POINT_SMOKE_COMPLETE
