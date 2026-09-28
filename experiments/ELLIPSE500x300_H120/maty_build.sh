#!/usr/bin/env bash
#SBATCH --job-name=ELLIPSE_G4_build
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=build.%j.out
#SBATCH --error=build.%j.err
set -eo pipefail
module load compilers/gcc/v12.2.0
export Geant4_DIR=/apps/soft/geant/geant4-v11.1.0/lib64/cmake/Geant4
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
set -u
CMAKE=/apps/tools/cmake/v3.25.2/bin/cmake
ROOT=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928
cd "$ROOT"
"$CMAKE" -S Geant4Sim/Geant4Code -B Geant4Build -DWITH_GEANT4_UIVIS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/apps/compilers/gcc/v12.2.0/bin/gcc \
  -DCMAKE_CXX_COMPILER=/apps/compilers/gcc/v12.2.0/bin/g++
"$CMAKE" --build Geant4Build --parallel 2
sha256sum Geant4Build/gamma01 Geant4Sim/Geant4Code/CrystalMatrix.txt
