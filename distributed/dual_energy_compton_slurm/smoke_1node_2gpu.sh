#!/bin/bash
#SBATCH -J JSCC_KB_smoke
#SBATCH -p gpu_5090
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --qos=gpugpu
#SBATCH --time=01:00:00
#SBATCH --output=distributed/dual_energy_compton_slurm/logs/smoke_%j.log
#SBATCH --error=distributed/dual_energy_compton_slurm/logs/smoke_%j.err

set -euo pipefail
module load cuda/12.9
module load miniforge3/25.11.0-1
source activate torch

REPO_ROOT=${JSCC_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO_ROOT"
mkdir -p distributed/dual_energy_compton_slurm/logs
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

torchrun --standalone --nproc_per_node=2 \
    distributed/dual_energy_compton_python/main_dist_dual_energy_compton.py \
    --count-level 1e10 \
    --iterations 2 \
    --save-step 1 \
    --max-events-per-view 256 \
    --materialize-block-events 64 \
    --theta-stride 1 \
    --z-stride 1 \
    --output-dir Results/Reconstruction/Distributed_JSCC_ComptonValidation_Smoke_2GPU
