#!/usr/bin/env bash
# Two ranks on distinct nodes; validates collectives against the serial MLEM reference.
#SBATCH --job-name=FOV120_NCCL_2node
#SBATCH --partition=gpu_5090
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --qos=gpugpu
#SBATCH --time=00:10:00
#SBATCH --output=experiments/FOV120/generated/nccl_2node.%j.out
#SBATCH --error=experiments/FOV120/generated/nccl_2node.%j.err
set -eo pipefail
: "${JSCC_REPO_ROOT:?Set isolated FOV120 repository root}"
cd "$JSCC_REPO_ROOT"
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate torch
set -u
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 JSCC_TEST_BACKEND=nccl
test "$SLURM_NNODES" -eq 2
master=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
port=$((50000+SLURM_JOB_ID%10000))
srun --kill-on-bad-exit=1 hostname
srun --kill-on-bad-exit=1 torchrun --nnodes=2 --nproc_per_node=1 \
  --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$master:$port" \
  distributed/dual_energy_compton_python/validate_synthetic_distributed.py
