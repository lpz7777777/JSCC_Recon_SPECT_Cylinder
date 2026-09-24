#!/usr/bin/env bash
#SBATCH -J JSCC_FOV120
#SBATCH -p gpu_5090
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --qos=gpugpu
#SBATCH --time=48:00:00
set -euo pipefail
: "${JSCC_REPO_ROOT:?Set repository root}"
: "${FOV120_ACCEPTED_EVENTS:?Set event estimate from pilot acceptance rate}"
: "${FOV120_GPU_GIB:?Set actual per-GPU memory GiB}"
cd "$JSCC_REPO_ROOT"
# Use the same installed environment as the existing distributed launchers.
module load cuda/12.9
module load miniforge3/25.11.0-1
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate torch
base=experiments/FOV120/generated
dataset=${FOV120_DATASET:-XCAT}
count=${FOV120_COUNT_LEVEL:-1e10}
gpus=${FOV120_GPUS_PER_NODE:-8}
common=(--experiment-config experiments/FOV120/config.json --factors-dir "$base/Factors"
        --cntstat-dir "$base/CntStat" --list-dir "$base/List"
        --data-file-name "$dataset" --count-level "$count")
python distributed/dual_energy_compton_python/preflight.py "${common[@]}" \
  --world-size "$((SLURM_NNODES*gpus))" --estimated-accepted-events "$FOV120_ACCEPTED_EVENTS" \
  --gpu-memory-gib "$FOV120_GPU_GIB"
master=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
port=$((50000+SLURM_JOB_ID%10000))
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
srun --kill-on-bad-exit=1 torchrun --nnodes="$SLURM_NNODES" --nproc_per_node="$gpus" \
  --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$master:$port" \
  distributed/dual_energy_compton_python/main_dist_dual_energy_compton.py "${common[@]}" \
  --iterations 1000 --save-step 50 --theta-stride 1 --z-stride 1 \
  --energy-resolution-fwhm .13 --energy-resolution-reference-kev 511 \
  --energy-threshold-sum-mev .350 --materialize-device cuda \
  --output-dir "$base/Results/${dataset}_${count}_${SLURM_JOB_ID}" "$@"
