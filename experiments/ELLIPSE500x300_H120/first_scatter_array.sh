#!/usr/bin/env bash
#SBATCH --job-name=FIRST_SCATTER_V2
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00
set -eo pipefail
source /etc/profile.d/modules.sh
module load compilers/gcc/v12.2.0
export Geant4_DIR=/apps/soft/geant/geant4-v11.1.0/lib64/cmake/Geant4
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
set -u
cd "${SLURM_SUBMIT_DIR:?}"
python3 first_scatter_workflow.py run --base "$PWD" --index "${SLURM_ARRAY_TASK_ID:?}"
