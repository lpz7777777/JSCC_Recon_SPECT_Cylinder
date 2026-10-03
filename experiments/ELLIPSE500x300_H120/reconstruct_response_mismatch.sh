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
: "${RESPONSE_RELEASE:?Use the frozen deployed release}"
release="$RESPONSE_RELEASE"
root=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor
base="$root/experiments/ELLIPSE500x300_H120"
study="$base/generated/response_mismatch_cut3_v1"
export RESPONSE_PROJECT_ROOT="$root"
export RESPONSE_PYTHON=/data/home/scxi717/.conda/envs/torch/bin/python
# Do not start a process group until every node can see the shared paths.
# This also avoids relying on conda activation propagating through srun.
for attempt in 1 2 3 4 5 6; do
  if [[ -d $root && -x $RESPONSE_PYTHON ]]; then break; fi
  sleep 5
done
test -d "$root" && test -x "$RESPONSE_PYTHON"
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
srun --chdir=/tmp --label --kill-on-bad-exit=1 bash -c '
  echo "RESPONSE_NODE_PREFLIGHT $(hostname) rank=$SLURM_PROCID"
  base="$RESPONSE_PROJECT_ROOT/experiments/ELLIPSE500x300_H120"
  ready=0
  for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
    if [[ -x $RESPONSE_PYTHON && -r $RESPONSE_RELEASE/run_reconstruction.py &&
          -r $RESPONSE_RELEASE/response_mismatch_cut3_v1.json &&
          -r $base/generated/Geometry/geometry.npz &&
          -r $base/generated/FactorsCalibrated/440keV_RotateNum20/SysMat_polar &&
          -r $base/generated/response_mismatch_cut3_v1/scan/Sensi_d &&
          -r $base/generated/List/218-440keV_RotateNum20_Geant4JSCC/List_NEMA_Body_H60_5e9/1.csv &&
          -r $base/generated/List/218-440keV_RotateNum20_Geant4JSCC/List_NEMA_Body_H60_5e9/20.csv ]]; then
      ready=1; break
    fi
    echo "RESPONSE_NODE_PATH_WAIT $(hostname) attempt=$attempt"
    sleep 5
  done
  if [[ $ready != 1 ]]; then
    echo "RESPONSE_NODE_PATH_FAILED $(hostname)" >&2
    exit 78
  fi
  timeout --signal=TERM --kill-after=10s 60s "$RESPONSE_PYTHON" -c "import socket,torch; print(\"RESPONSE_NODE_CUDA_OK\",socket.gethostname(),torch.__version__,torch.cuda.device_count(),flush=True); assert torch.cuda.is_available() and torch.cuda.device_count()==1"
'
for phase in regression pilot formal; do
  case "$phase" in
    regression) iterations=50; save=50; mode=(--baseline-regression);;
    pilot) iterations=10; save=10; mode=(--pilot-only --compton-sensitivity "$study/scan/Sensi_d");;
    formal) iterations=10000; save=50; mode=(--compton-sensitivity "$study/scan/Sensi_d");;
  esac
  output="$study/${phase}_${SLURM_JOB_ID}"
  case "$phase" in regression|pilot) limit=2h;; formal) limit=48h;; esac
  echo "RESPONSE_PHASE $phase $iterations"
  timeout --signal=TERM --kill-after=60s "$limit" \
    srun --chdir=/tmp --label --kill-on-bad-exit=1 bash -c '
    exec "$RESPONSE_PYTHON" -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc_per_node=1 \
      --node_rank="$SLURM_PROCID" --master_addr="$ELLIPSE_MASTER_ADDR" \
      --master_port="$ELLIPSE_MASTER_PORT" --rdzv_backend=static --rdzv_conf=timeout=300 \
      --rdzv_id="${SLURM_JOB_ID}_$1" --max_restarts=0 "${@:2}"
  ' bash "$phase" "$release/run_reconstruction.py" \
    --factors "$base/generated/FactorsCalibrated" --geometry "$base/generated/Geometry/geometry.npz" \
    --data-root "$base/generated" --dataset NEMA_Body_H60 --level 5e9 --channels compton-jscc \
    --response-filter-config "$config" --iterations "$iterations" --save-step "$save" \
    --output "$output" "${mode[@]}"
  "$RESPONSE_PYTHON" "$release/verify_response_mismatch.py" --result "$output" --config "$config" \
    --baseline "$baseline" --geometry "$base/generated/Geometry/geometry.npz" --mode "$phase"
  echo "RESPONSE_PHASE_PASSED $phase"
done
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,State,Elapsed,MaxRSS,AllocTRES -P \
  > "$study/accounting_${SLURM_JOB_ID}.txt"
echo "RESPONSE_STUDY_COMPLETE $SLURM_JOB_ID"
