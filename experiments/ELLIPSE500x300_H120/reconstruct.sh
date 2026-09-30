#!/usr/bin/env bash
#SBATCH -J JSCC_ELLIPSE
#SBATCH -p gpu_5090
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --qos=gpugpu
#SBATCH --time=48:00:00
#SBATCH --output=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/logs/recon.%j.out
#SBATCH --error=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/logs/recon.%j.err
set -euo pipefail
: "${ELLIPSE_ACCEPTED_EVENTS:?Use measured accepted-event count from 1e9 or 1e10 input}"
: "${ELLIPSE_GPU_GIB:?Provide GPU capacity in GiB}"
: "${ELLIPSE_HOST_GIB:?Provide granted host memory per node in GiB}"
root=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor
base="$root/experiments/ELLIPSE500x300_H120"
cd "$root"
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate torch
gpus=${ELLIPSE_GPUS_PER_NODE:-3}
if [[ -n ${SLURM_GPUS_ON_NODE:-} && "$SLURM_GPUS_ON_NODE" != "$gpus" ]]; then
  echo "GPU allocation mismatch: ${SLURM_GPUS_ON_NODE} versus ${gpus}" >&2
  exit 1
fi
world=$((SLURM_NNODES*gpus))
python "$base/resource_budget.py" --accepted-events "$ELLIPSE_ACCEPTED_EVENTS" \
  --gpu-gib "$ELLIPSE_GPU_GIB" --host-gib "$ELLIPSE_HOST_GIB" \
  --nodes "$SLURM_NNODES" --gpus-per-node "$gpus" > "$base/logs/budget.${SLURM_JOB_ID}.json"
python - "$base/logs/budget.${SLURM_JOB_ID}.json" "$world" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))
entry=next((r for r in rows if r['world_size']==int(sys.argv[2])),None)
if not entry or not entry['gpu_20_percent_margin_pass'] or not entry['host_20_percent_margin_pass']:
    raise SystemExit('Resource estimate fails the 20% margin; choose a larger allocation')
print(entry)
PY
dataset=${ELLIPSE_DATASET:?Set CircleNewDist, EllipseUniform, EllipseContrast, XCAT or NEMA_Body_H60}
level=${ELLIPSE_LEVEL:?Set 1e9 or 1e10}
mode=()
iterations=10000
save_step=50
if [[ ${ELLIPSE_PILOT:-0} == 1 ]]; then
  iterations=10
  save_step=10
  mode=(--pilot-only)
fi
if [[ ${ELLIPSE_DRY_RUN:-0} == 1 ]]; then
  mode+=(--dry-run)
fi
master=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
port=$((50000+SLURM_JOB_ID%10000))
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
# Some 4090 nodes otherwise select an unreachable 169.254/16 interface.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
command=(srun --kill-on-bad-exit=1 torchrun --nnodes="$SLURM_NNODES" --nproc_per_node="$gpus" \
  --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$master:$port" \
  "$base/run_reconstruction.py" \
  --factors "$base/generated/FactorsCalibrated" --geometry "$base/generated/Geometry/geometry.npz" \
  --data-root "$base/generated" --dataset "$dataset" --level "$level" \
  --iterations "$iterations" --save-step "$save_step" \
  --output "$base/generated/Results/${dataset}_${level}_${SLURM_JOB_ID}")
if [[ ${#mode[@]} -gt 0 ]]; then
  command+=("${mode[@]}")
fi
"${command[@]}"
