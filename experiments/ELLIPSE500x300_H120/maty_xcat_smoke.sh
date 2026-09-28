#!/usr/bin/env bash
#SBATCH --job-name=ELLIPSE_XCAT_smoke
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=xcat_smoke.%j.out
#SBATCH --error=xcat_smoke.%j.err
set -eo pipefail
module load compilers/gcc/v12.2.0
export Geant4_DIR=/apps/soft/geant/geant4-v11.1.0/lib64/cmake/Geant4
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
ROOT=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928
folder="$ROOT/generated/XCATSmoke/$SLURM_JOB_ID"
mkdir -p "$folder"
cp "$ROOT/Geant4Sim/Geant4Code/CrystalMatrix.txt" "$folder/CrystalMatrix.txt"
cp "$ROOT/xcat_smoke.mac" "$folder/run.mac"
cd "$folder"
"$ROOT/Geant4Build/gamma01" run.mac > console.log 2>&1
python3 - <<'PY'
import numpy as np
p=np.loadtxt('PrimaryCount.csv',delimiter=',',dtype=np.int64).reshape(-1)
assert p.size==3 and p.sum()==10000 and p[2]==0,p
for e in (218,440):
    c=np.loadtxt(f'CntStat_{e}.csv',delimiter=',',dtype=np.int64)
    assert c.size==10496 and np.all(c>=0)
print('ELLIPSE_XCAT_SMOKE_PASSED',p.tolist())
PY
