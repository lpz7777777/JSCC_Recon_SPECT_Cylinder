#!/bin/bash
#SBATCH -J JSCC_Dist_LPZ
#SBATCH -p gpu_5090
#SBATCH -N 1                   # 申请的节点数量
#SBATCH --ntasks-per-node=1    # 每个节点启动一个任务（由 torchrun 进一步分发 GPU 进程）
#SBATCH --gres=gpu:8           # 每个节点使用的 GPU 数量
#SBATCH --qos=gpugpu           # 多机多卡必须指定的 QoS 参数
#SBATCH --time=48:00:00        # 康普顿数据量大，建议适当延长限时
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err

# --- 加载环境 ---
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
            if [[ -f "$current/distributed/python/main_dist.py" && -f "$current/distributed/python/_path_setup.py" ]]; then
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

# --- 错误处理与调试配置 ---
set -e
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# --- 网络与分布式环境配置 ---
export NCCL_DEBUG=INFO 
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3        
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# 动态获取主节点地址和随机端口
MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=$(shuf -i 50000-60000 -n 1)

GPUS_PER_NODE=8
ENTRYPOINT="$REPO_ROOT/distributed/python/main_dist.py"

if [[ ! -f "$ENTRYPOINT" ]]; then
    echo "Entry point not found: $ENTRYPOINT" >&2
    exit 1
fi

echo "Starting Distributed JSCC Job on nodes: $SLURM_JOB_NODELIST"
echo "Master Node: $MASTER_NODE, Port: $MASTER_PORT"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "Repo Root: $REPO_ROOT"

# --- 使用 srun 启动 torchrun ---
# 增加 --gres=gpu:$GPUS_PER_NODE 确保 srun 启动的子任务能看到完整的 GPU 资源
srun --kill-on-bad-exit=1 \
     --gres=gpu:$GPUS_PER_NODE \
     torchrun \
     --nnodes=$SLURM_NNODES \
     --nproc_per_node=$GPUS_PER_NODE \
     --rdzv_id=$SLURM_JOB_ID \
     --rdzv_endpoint=$MASTER_NODE:$MASTER_PORT \
     --rdzv_backend=c10d \
     "$ENTRYPOINT" "$@"
