#!/bin/bash
#SBATCH -J JSCC_Dist_LPZ
#SBATCH -p gpu 
#SBATCH -N 2                   # 申请的节点数量
#SBATCH --ntasks-per-node=1    # 每个节点启动一个任务（由 torchrun 进一步分发 GPU 进程）
#SBATCH --gres=gpu:4           # 每个节点使用的 GPU 数量
#SBATCH --qos=gpugpu           # 多机多卡必须指定的 QoS 参数
#SBATCH --cpus-per-task=16     # 每个任务分配的 CPU 核心数
#SBATCH --time=48:00:00        # 康普顿数据量大，建议适当延长限时
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err

# --- 加载环境 ---
module load cuda/12.9
module load miniforge/25.3.0-3            
source activate torch

# --- 错误处理与调试配置 ---
set -e
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# --- 网络与分布式环境配置 ---
export NCCL_IB_DISABLE=0          
# 【关键修改】：不再强行指定 ib0，而是排除掉回环网卡和 Docker 网卡，让 NCCL 自动探测可用网卡
export NCCL_SOCKET_IFNAME=^lo,docker0
# 如果已知高速网卡的特定前缀（如 hsn, mlx），也可以使用 export NCCL_SOCKET_IFNAME=hsn,mlx,eth
export NCCL_DEBUG=INFO            
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 动态获取主节点地址和随机端口
MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=$(shuf -i 50000-60000 -n 1)

# 与 #SBATCH --gres=gpu:4 保持一致
GPUS_PER_NODE=4

echo "Starting Distributed JSCC Job on nodes: $SLURM_JOB_NODELIST"
echo "Master Node: $MASTER_NODE, Port: $MASTER_PORT"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"

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
     main_dist.py