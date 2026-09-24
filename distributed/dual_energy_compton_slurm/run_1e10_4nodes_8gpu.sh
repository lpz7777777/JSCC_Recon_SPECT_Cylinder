#!/bin/bash
#SBATCH -J JSCC_1e10_KB
#SBATCH -p gpu_5090
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --qos=gpugpu
#SBATCH --time=48:00:00
#SBATCH --output=distributed/dual_energy_compton_slurm/logs/%j.log
#SBATCH --error=distributed/dual_energy_compton_slurm/logs/%j.err

set -euo pipefail

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
            if [[ -f "$current/distributed/dual_energy_compton_python/main_dist_dual_energy_compton.py" ]]; then
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
    echo "Set JSCC_REPO_ROOT to the repository path before sbatch." >&2
    exit 1
}
cd "$REPO_ROOT"
mkdir -p distributed/dual_energy_compton_slurm/logs

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_bond_0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

MASTER_NODE=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((50000 + SLURM_JOB_ID % 10000))
GPUS_PER_NODE=8
ENTRYPOINT="$REPO_ROOT/distributed/dual_energy_compton_python/main_dist_dual_energy_compton.py"

python distributed/dual_energy_compton_python/preflight.py \
    --count-level 1e10 --world-size $((SLURM_NNODES * GPUS_PER_NODE))

srun --kill-on-bad-exit=1 --gres=gpu:$GPUS_PER_NODE \
    torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_endpoint="$MASTER_NODE:$MASTER_PORT" \
    --rdzv_backend=c10d \
    "$ENTRYPOINT" \
    --count-level 1e10 \
    --iterations 1000 \
    --save-step 50 \
    --theta-stride 1 \
    --z-stride 1 \
    --energy-resolution-fwhm 0.13 \
    --energy-resolution-reference-kev 511 \
    --energy-threshold-sum-mev 0.350 \
    --materialize-device cuda \
    --output-dir Results/Reconstruction/Distributed_JSCC_ComptonValidation_Geant4_1e10_Iter1000 \
    "$@"
