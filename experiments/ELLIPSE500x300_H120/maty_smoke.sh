#!/usr/bin/env bash
#SBATCH --job-name=ELLIPSE_G4_smoke
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=smoke.%j.out
#SBATCH --error=smoke.%j.err
set -eo pipefail
module load compilers/gcc/v12.2.0
export Geant4_DIR=/apps/soft/geant/geant4-v11.1.0/lib64/cmake/Geant4
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
ROOT=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928
cd "$ROOT"
folder="generated/Smoke/${SLURM_JOB_ID}"
mkdir -p "$folder"
cp Geant4Sim/Geant4Code/CrystalMatrix.txt "$folder/CrystalMatrix.txt"
cp experiments/ELLIPSE500x300_H120/smoke_ellipse.mac "$folder/run.mac"
cd "$folder"
"$ROOT/Geant4Build/gamma01" run.mac > console.log 2>&1
python3 - <<'PY'
import numpy as np
counts=np.loadtxt('PrimaryCount.csv',delimiter=',',dtype=np.int64).reshape(-1)
assert len(counts)==3 and counts.sum()==10000 and counts[2]==0,counts
assert 2800<counts[0]<3300,counts
for energy in (218,440):
    values=np.loadtxt(f'CntStat_{energy}.csv',delimiter=',',dtype=np.int64)
    assert values.size==10496 and np.all(values>=0)
print('ELLIPSE_SOURCE_SMOKE_PASSED',counts.tolist())
PY
