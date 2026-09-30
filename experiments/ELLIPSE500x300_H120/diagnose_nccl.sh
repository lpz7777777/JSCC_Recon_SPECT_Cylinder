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
master=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
port=$((50000+SLURM_JOB_ID%10000))
srun --kill-on-bad-exit=1 torchrun --nnodes="$SLURM_NNODES" --nproc_per_node=1 \
  --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$master:$port" \
  experiments/ELLIPSE500x300_H120/diagnose_nccl.py
