#!/bin/bash
#SBATCH -J JSCCSD_CPU
#SBATCH -p amd_m9_768
#SBATCH -N 4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=168:00:00
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err

# ============================================================================
# BSCC-M9 超算集群 —— 分布式 CPU 稀疏 Compton JSCC-SD-only 重建
#
# 集群配置：
#   队列: amd_m9_768
#   单节点: 256 核 AMD EPYC 9755 @ 2.7GHz, 768 GB 内存
#   互连: InfiniBand (IB)
#
# 资源布局：
#   16 procs/node × 16 threads/proc = 256 cores/node (满占)
#   768 GB / 16 procs = 48 GB/proc
#   4 nodes × 16 procs = 64 total ranks
#
# 调整节点数：
#   修改 #SBATCH -N 即可。例如 -N 8 → 128 ranks, -N 1 → 16 ranks (单节点)
#
# 调整每节点进程数（影响每进程内存和线程数）：
#   --ntasks-per-node=8  --cpus-per-task=32  → 8×32=256, 96GB/proc
#   --ntasks-per-node=16 --cpus-per-task=16  → 16×16=256, 48GB/proc
#   --ntasks-per-node=32 --cpus-per-task=8   → 32×8=256, 24GB/proc
#
# 用法：
#   sbatch jsccrecon_dist_sparse_jsccsd_only_cpu.sh [--jsccsd-iter 5000 ...]
# ============================================================================

module load miniforge3/25.11.0-1
source activate torch

# ---------- 查找工程根目录 ----------
find_repo_root() {
    local start_dir current depth
    for start_dir in "${JSCC_REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
        [[ -n "$start_dir" ]] || continue
        current=$(cd "$start_dir" 2>/dev/null && pwd) || continue
        depth=0
        while [[ "$current" != "/" && $depth -lt 8 ]]; do
            if [[ -f "$current/distributed/python/main_dist_sparse_jsccsd_only_cpu.py" && -f "$current/distributed/python/_path_setup.py" ]]; then
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

# ---------- 环境变量 ----------
export PYTHONUNBUFFERED=1

# OpenMP / BLAS 线程数 = SLURM 分配的 cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# GLOO 后端网络配置（IB 互连优化）
# IB 网卡通常映射为 ib0 或 bond0，通过 IPoIB 提供 TCP 通信
# 如果 IB 接口名不同，请通过 GLOO_SOCKET_IFNAME 环境变量指定
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ib0}
export GLOO_IB_DISABLE=0

# ---------- 并行配置 ----------
PROCS_PER_NODE=${SLURM_NTASKS_PER_NODE:-16}
TOTAL_PROCS=$((SLURM_NNODES * PROCS_PER_NODE))
CPUS_PER_PROC=${SLURM_CPUS_PER_TASK:-16}
MEM_PER_NODE_GB=768
MEM_PER_PROC_GB=$((MEM_PER_NODE_GB / PROCS_PER_NODE))

ENTRYPOINT="$REPO_ROOT/distributed/python/main_dist_sparse_jsccsd_only_cpu.py"

if [[ ! -f "$ENTRYPOINT" ]]; then
    echo "Entry point not found: $ENTRYPOINT" >&2
    exit 1
fi

MASTER_NODE=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(shuf -i 50000-60000 -n 1)

echo "============================================"
echo "BSCC-M9 Distributed CPU Sparse JSCCSD-Only"
echo "============================================"
echo "Queue:    amd_m9_768"
echo "Nodes:    $SLURM_JOB_NODELIST ($SLURM_NNODES nodes)"
echo "Master:   $MASTER_NODE:$MASTER_PORT"
echo "Layout:   $PROCS_PER_NODE procs/node × $CPUS_PER_PROC threads/proc = $((PROCS_PER_NODE * CPUS_PER_PROC)) cores/node"
echo "Total:    $TOTAL_PROCS ranks × $CPUS_PER_PROC threads = $((TOTAL_PROCS * CPUS_PER_PROC)) CPU threads"
echo "Memory:   ~${MEM_PER_PROC_GB} GB/proc (768 GB/node ÷ $PROCS_PER_NODE)"
echo "Network:  IB ($GLOO_SOCKET_IFNAME)"
echo "Root:     $REPO_ROOT"
echo "============================================"

# ---------- 启动分布式训练 ----------
srun --kill-on-bad-exit=1 \
     --cpu-bind=cores \
     torchrun \
     --nnodes=$SLURM_NNODES \
     --nproc_per_node=$PROCS_PER_NODE \
     --rdzv_id=$SLURM_JOB_ID \
     --rdzv_endpoint=$MASTER_NODE:$MASTER_PORT \
     --rdzv_backend=c10d \
     "$ENTRYPOINT" \
     --num-threads $CPUS_PER_PROC \
     "$@"