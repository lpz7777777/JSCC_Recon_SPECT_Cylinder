#!/usr/bin/env bash
#SBATCH -J NEMA_response_cut3
#SBATCH -p gpu_4090,gpu_5090
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --qos=gpugpu
#SBATCH --time=48:00:00
#SBATCH --output=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/logs/response_cut3.%j.out
#SBATCH --error=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/logs/response_cut3.%j.err
set -euo pipefail
release=$(cd -- "$(dirname -- "$0")" && pwd)
: "${RESPONSE_RELEASE:?Use the frozen deployed release}"
release="$RESPONSE_RELEASE"
root=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor
base="$root/experiments/ELLIPSE500x300_H120"
study="$base/generated/response_mismatch_cut3_v1"
cd "$root"
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate torch
[[ "$SLURM_NNODES" == 8 && "${SLURM_GPUS_ON_NODE:-1}" == 1 ]]
export JSCC_PROJECT_ROOT="$release"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=bond0 TORCH_ELASTIC_WORKER_IDENTICAL=1
if [[ -n ${SLURM_MEM_PER_NODE:-} ]]; then
  export ABLATION_HOST_ALLOCATED_BYTES=$((SLURM_MEM_PER_NODE*1024*1024))
else
  : "${SLURM_MEM_PER_CPU:?Actual granted memory must be available}"
  export ABLATION_HOST_ALLOCATED_BYTES=$((SLURM_MEM_PER_CPU*SLURM_CPUS_PER_TASK*1024*1024))
fi
export ELLIPSE_MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export ELLIPSE_MASTER_PORT=$((50000+SLURM_JOB_ID%10000))
export ELLIPSE_GPUS_PER_NODE=1
scontrol show job "$SLURM_JOB_ID" > "$study/allocation_${SLURM_JOB_ID}.txt"
baseline="$base/generated/Results/NEMA_Body_H60_5e9_1644876"
config="$release/response_mismatch_cut3_v1.json"
for phase in regression pilot formal; do
  case "$phase" in
    regression) iterations=50; save=50; mode=(--baseline-regression);;
    pilot) iterations=10; save=10; mode=(--pilot-only --compton-sensitivity "$study/scan/Sensi_d");;
    formal) iterations=10000; save=50; mode=(--compton-sensitivity "$study/scan/Sensi_d");;
  esac
  output="$study/${phase}_${SLURM_JOB_ID}"
  echo "RESPONSE_PHASE $phase $iterations"
  srun --kill-on-bad-exit=1 bash -c '
    exec torchrun --nnodes="$SLURM_NNODES" --nproc_per_node=1 \
      --node_rank="$SLURM_PROCID" --master_addr="$ELLIPSE_MASTER_ADDR" \
      --master_port="$ELLIPSE_MASTER_PORT" --rdzv_backend=static --rdzv_conf=timeout=300 \
      --rdzv_id="${SLURM_JOB_ID}_$1" --max_restarts=0 "${@:2}"
  ' bash "$phase" "$release/run_reconstruction.py" \
    --factors "$base/generated/FactorsCalibrated" --geometry "$base/generated/Geometry/geometry.npz" \
    --data-root "$base/generated" --dataset NEMA_Body_H60 --level 5e9 --channels compton-jscc \
    --response-filter-config "$config" --iterations "$iterations" --save-step "$save" \
    --output "$output" "${mode[@]}"
  python "$release/verify_response_mismatch.py" --result "$output" --config "$config" \
    --baseline "$baseline" --geometry "$base/generated/Geometry/geometry.npz" --mode "$phase"
  echo "RESPONSE_PHASE_PASSED $phase"
done
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,State,Elapsed,MaxRSS,AllocTRES -P \
  > "$study/accounting_${SLURM_JOB_ID}.txt"
echo "RESPONSE_STUDY_COMPLETE $SLURM_JOB_ID"
