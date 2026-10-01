#!/usr/bin/env bash
#SBATCH -J ellipse_nccl_diag
#SBATCH -p gpu_4090
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpugpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/logs/nccl.%j.out
#SBATCH --error=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/logs/nccl.%j.err
set -euo pipefail
root=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor
cd "$root"
source /etc/profile.d/modules.sh
module load cuda/12.9 miniforge3/25.11.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch
export PYTHONUNBUFFERED=1 NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
master=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
port=$((50000+SLURM_JOB_ID%10000))
export ELLIPSE_MASTER_ADDR="$master" ELLIPSE_MASTER_PORT="$port"
export TORCH_ELASTIC_WORKER_IDENTICAL=1
srun --kill-on-bad-exit=1 bash -c '
  exec torchrun --nnodes="$SLURM_NNODES" --nproc_per_node=1 \
    --node_rank="$SLURM_PROCID" --master_addr="$ELLIPSE_MASTER_ADDR" \
    --master_port="$ELLIPSE_MASTER_PORT" --rdzv_backend=static \
    --rdzv_conf=timeout=120 --rdzv_id="$SLURM_JOB_ID" --max_restarts=0 "$@"
' bash \
  experiments/ELLIPSE500x300_H120/diagnose_nccl.py
