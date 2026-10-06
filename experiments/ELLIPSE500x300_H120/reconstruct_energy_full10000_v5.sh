#!/usr/bin/env bash
set -euo pipefail
: "${ENERGY_FULL_V5_RELEASE:?}" "${ENERGY_FULL_V5_EXECUTION:?}" "${ENERGY_FULL_V5_PHASE_SECONDS:?}" "${ENERGY_FULL_V5_TOTAL_SECONDS:?}"
release="$ENERGY_FULL_V5_RELEASE"
base=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120
study="$base/generated/compton_energy_probability_v5_5e9_full10000"
export RESPONSE_PYTHON=/data/home/scxi717/.conda/envs/torch/bin/python
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
[[ "$SLURM_NNODES" == 8 ]]
export JSCC_PROJECT_ROOT="$release" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=bond0 TORCH_ELASTIC_WORKER_IDENTICAL=1
export ELLIPSE_MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export ELLIPSE_MASTER_PORT=$((50000+SLURM_JOB_ID%10000))
scontrol show job "$SLURM_JOB_ID" > "$study/formal_allocation_${SLURM_JOB_ID}.txt"
export ABLATION_HOST_ALLOCATED_BYTES=$("$RESPONSE_PYTHON" "$release/verify_first_scatter.py" --allocation-only "$study/formal_allocation_${SLURM_JOB_ID}.txt" --nodes 8)
srun --chdir=/tmp --label --kill-on-bad-exit=1 bash -c '
  timeout --signal=TERM --kill-after=10s 60s "$RESPONSE_PYTHON" -c "import socket,torch; from pathlib import Path; p=Path(\"$ENERGY_FULL_V5_RELEASE\"); assert (p/\"contract.json\").is_file() and (p/\"whole_geometry.npz\").is_file(); assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(socket.gethostname(),flush=True)"
'
authority=()
if [[ "$ENERGY_FULL_V5_EXECUTION" == validation ]]; then
  iterations=10; save=10
elif [[ "$ENERGY_FULL_V5_EXECUTION" == formal ]]; then
  : "${ENERGY_FULL_V5_AUTHORITY:?}" "${ENERGY_FULL_V5_AUTHORITY_SHA:?}"
  iterations=10000; save=50
  authority=(--authority "$ENERGY_FULL_V5_AUTHORITY" --authority-sha256 "$ENERGY_FULL_V5_AUTHORITY_SHA")
else exit 2; fi
output="$study/${ENERGY_FULL_V5_EXECUTION}_continuous_energy_${SLURM_JOB_ID}"
echo "ENERGY_FULL10000_V5_PHASE $ENERGY_FULL_V5_EXECUTION continuous_energy six_channels"
timeout --signal=TERM --kill-after=60s "${ENERGY_FULL_V5_TOTAL_SECONDS}s" srun --chdir=/tmp --label --kill-on-bad-exit=1 bash -c '
  exec "$RESPONSE_PYTHON" -m torch.distributed.run --nnodes=8 --nproc_per_node=1 --node_rank="$SLURM_PROCID" \
    --master_addr="$ELLIPSE_MASTER_ADDR" --master_port="$ELLIPSE_MASTER_PORT" --rdzv_backend=static \
    --rdzv_conf=timeout=300 --rdzv_id="${SLURM_JOB_ID}_full" --max_restarts=0 "${@:1}"
' bash "$release/run_energy_full10000_v5.py" --contract "$release/contract.json" \
  --factors "$base/generated/FactorsCalibrated" --input-root "$base/generated" \
  --output "$output" --model continuous_energy --mode "$ENERGY_FULL_V5_EXECUTION" \
  --iterations "$iterations" --save-step "$save" "${authority[@]}"
"$RESPONSE_PYTHON" "$release/verify_energy_full10000_v5.py" --result "$output" --contract "$release/contract.json" \
  --allocation "$study/formal_allocation_${SLURM_JOB_ID}.txt" --mode "$ENERGY_FULL_V5_EXECUTION"
echo "ENERGY_FULL10000_V5_ENTRY_COMPLETE $ENERGY_FULL_V5_EXECUTION"
