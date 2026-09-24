#!/usr/bin/env bash
# Run from the isolated FOV120 repository root on the maty supercomputer.
#SBATCH --job-name=FOV120_build_smoke
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=experiments/FOV120/generated/build_smoke.%j.out
#SBATCH --error=experiments/FOV120/generated/build_smoke.%j.err
set -eo pipefail
module load compilers/gcc/v12.2.0
export Geant4_DIR=/apps/soft/geant/geant4-v11.1.0/lib64/cmake/Geant4
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
set -u
CMAKE=/apps/tools/cmake/v3.25.2/bin/cmake
BUILD=experiments/FOV120/generated/Geant4Build
hostname
date -Is
"$CMAKE" -S Geant4Sim/Geant4Code -B "$BUILD" -DWITH_GEANT4_UIVIS=OFF -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/apps/compilers/gcc/v12.2.0/bin/gcc \
    -DCMAKE_CXX_COMPILER=/apps/compilers/gcc/v12.2.0/bin/g++
"$CMAKE" --build "$BUILD" --parallel 2
sha256sum "$BUILD/gamma01" Geant4Sim/Geant4Code/CrystalMatrix.txt
for index in 0 100 562 962 1362 1462; do
    python3 experiments/FOV120/workflow.py run --index "$index" \
        --executable "$BUILD/gamma01" --crystal Geant4Sim/Geant4Code/CrystalMatrix.txt --smoke
done
date -Is
echo FOV120_CLUSTER_SMOKE_COMPLETE
