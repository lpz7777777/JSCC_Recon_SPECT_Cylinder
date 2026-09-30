#!/usr/bin/env bash
# Selected 10 independent point datasets, 20 views each, 1e7 primaries/dataset.
# Submit with --array=0-199%40; MaxArraySize on maty is 1001.
#SBATCH --job-name=ELLIPSE_point_QA
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=point_selected.%A_%a.out
#SBATCH --error=point_selected.%A_%a.err
set -euo pipefail
module load compilers/gcc/v12.2.0
export Geant4_DIR=/apps/soft/geant/geant4-v11.1.0/lib64/cmake/Geant4
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
root=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928
cd "$root"
task=${SLURM_ARRAY_TASK_ID:?Run as a Slurm array}
if (( task < 40 )); then
  index=$task
elif (( task < 120 )); then
  index=$((3520+task-40))
elif (( task < 200 )); then
  index=$((3920+task-120))
else
  echo "Unapproved selected point task $task" >&2
  exit 2
fi
python3 experiments/FOV120/workflow.py run \
  --manifest experiments/ELLIPSE500x300_H120/generated/PointScan/jobs.json \
  --index "$index" --executable Geant4Build/gamma01 \
  --crystal Geant4Sim/Geant4Code/CrystalMatrix.txt
