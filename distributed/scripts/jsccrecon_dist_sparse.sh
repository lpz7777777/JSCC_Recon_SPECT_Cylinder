#!/bin/bash
#SBATCH -J JSCC_Sparse_LPZ
#SBATCH -p gpu_5090
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --qos=gpugpu
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err

module load cuda/12.9
module load miniforge3/25.11.0-1
source activate torch

find_repo_root() {
    local start_dir current depth
    for start_dir in "${JSCC_REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
        [[ -n "$start_dir" ]] || continue
        current=$(cd "$start_dir" 2>/dev/null && pwd) || continue
        depth=0
        while [[ "$current" != "/" && $depth -lt 8 ]]; do
            if [[ -f "$current/distributed/python/main_dist_sparse.py" && -f "$current/distributed/python/_path_setup.py" ]]; then
                echo "$current"
                return 0
            fi
            current=$(dirname "$current")
            depth=$((depth + 1))
        done
    done
    return 1
}

REPO_ROOT=$(find_repo_root) || {
    echo "Failed to locate repo root. Please export JSCC_REPO_ROOT=/path/to/repo before sbatch." >&2
    exit 1
}
cd "$REPO_ROOT"

set -e
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

MASTER_NODE=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(shuf -i 50000-60000 -n 1)
GPUS_PER_NODE=8
ENTRYPOINT="$REPO_ROOT/distributed/python/main_dist_sparse.py"

if [[ ! -f "$ENTRYPOINT" ]]; then
    echo "Entry point not found: $ENTRYPOINT" >&2
    exit 1
fi

echo "Starting distributed sparse JSCC job on nodes: $SLURM_JOB_NODELIST"
echo "Master Node: $MASTER_NODE, Port: $MASTER_PORT"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "Repo Root: $REPO_ROOT"
echo "Entrypoint: $ENTRYPOINT"

srun --kill-on-bad-exit=1 \
     --gres=gpu:$GPUS_PER_NODE \
     torchrun \
     --nnodes=$SLURM_NNODES \
     --nproc_per_node=$GPUS_PER_NODE \
     --rdzv_id=$SLURM_JOB_ID \
     --rdzv_endpoint=$MASTER_NODE:$MASTER_PORT \
     --rdzv_backend=c10d \
     "$ENTRYPOINT"
