#!/usr/bin/env bash
# Submit only after all six cluster smoke workers pass.
#SBATCH --job-name=FOV120_mono_pilot
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=experiments/FOV120/generated/pilot.%A_%a.out
#SBATCH --error=experiments/FOV120/generated/pilot.%A_%a.err
set -eo pipefail
module load compilers/gcc/v12.2.0
export Geant4_DIR=/apps/soft/geant/geant4-v11.1.0/lib64/cmake/Geant4
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
set -u
python3 - <<'PY'
import json
from pathlib import Path
from experiments.FOV120.workflow import digest
root=Path('experiments/FOV120/generated')
executable=digest(root/'Geant4Build/gamma01')
crystal=digest(Path('Geant4Sim/Geant4Code/CrystalMatrix.txt'))
for index in (0,100,562,962,1362,1462):
    record=json.loads((root/f'Simulation/smoke/{index:05d}/worker.json').read_text())
    if (record['status']!='complete' or record['simulated_photons']!=10000
        or sum(record['primary_counts'])!=10000
        or record['executable_sha256']!=executable or record['crystal_sha256']!=crystal):
        raise RuntimeError(f'Cluster smoke failed or binary changed: {index}')
PY
# Slurm MaxArraySize=1001 on maty; preserve immutable manifest indices via offset.
offset=${FOV120_JOB_INDEX_OFFSET:-0}
[[ "$offset" =~ ^[0-9]+$ ]] || { echo "Invalid FOV120_JOB_INDEX_OFFSET" >&2; exit 2; }
index=$((SLURM_ARRAY_TASK_ID + offset))
python3 experiments/FOV120/workflow.py run --index "$index" \
    --executable experiments/FOV120/generated/Geant4Build/gamma01 \
    --crystal Geant4Sim/Geant4Code/CrystalMatrix.txt
