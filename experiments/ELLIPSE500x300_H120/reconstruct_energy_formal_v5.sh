#!/usr/bin/env bash
set -euo pipefail
: "${ENERGY_V5_FORMAL_RELEASE:?}" "${ENERGY_V5_EXECUTION:?}" "${ENERGY_V5_PHASE_SECONDS:?}"
release="$ENERGY_V5_FORMAL_RELEASE"
base=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120
study="$base/generated/compton_energy_probability_v5"
old="$base/generated/compton_first_scatter_v2"
export RESPONSE_PYTHON=/data/home/scxi717/.conda/envs/torch/bin/python
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
[[ "$SLURM_NNODES" == 4 ]]
export JSCC_PROJECT_ROOT="$release" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=bond0 TORCH_ELASTIC_WORKER_IDENTICAL=1
export ELLIPSE_MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export ELLIPSE_MASTER_PORT=$((50000+SLURM_JOB_ID%10000))
scontrol show job "$SLURM_JOB_ID" > "$study/formal_allocation_${SLURM_JOB_ID}.txt"
export ABLATION_HOST_ALLOCATED_BYTES=$("$RESPONSE_PYTHON" "$release/verify_first_scatter.py" --allocation-only "$study/formal_allocation_${SLURM_JOB_ID}.txt" --nodes 4)
srun --chdir=/tmp --label --kill-on-bad-exit=1 bash -c '
  timeout --signal=TERM --kill-after=10s 60s "$RESPONSE_PYTHON" -c "import socket,torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(socket.gethostname(),flush=True)"
  [[ -r "$ENERGY_V5_FORMAL_RELEASE/contract.json" ]]
'
authority=()
if [[ "$ENERGY_V5_EXECUTION" == validation ]]; then
  iterations=10; save=10
elif [[ "$ENERGY_V5_EXECUTION" == formal ]]; then
  : "${ENERGY_V5_AUTHORITY:?}" "${ENERGY_V5_AUTHORITY_SHA:?}"
  iterations=2000; save=50
  authority=(--authority "$ENERGY_V5_AUTHORITY" --authority-sha256 "$ENERGY_V5_AUTHORITY_SHA")
else exit 2; fi
for model in angular continuous_energy; do
  output="$study/${ENERGY_V5_EXECUTION}_${model}_${SLURM_JOB_ID}"
  echo "ENERGY_V5_FORMAL_PHASE $ENERGY_V5_EXECUTION $model"
  timeout --signal=TERM --kill-after=60s "${ENERGY_V5_PHASE_SECONDS}s" srun --chdir=/tmp --label --kill-on-bad-exit=1 bash -c '
    exec "$RESPONSE_PYTHON" -m torch.distributed.run --nnodes=4 --nproc_per_node=1 --node_rank="$SLURM_PROCID" \
      --master_addr="$ELLIPSE_MASTER_ADDR" --master_port="$ELLIPSE_MASTER_PORT" --rdzv_backend=static \
      --rdzv_conf=timeout=300 --rdzv_id="${SLURM_JOB_ID}_$1" --max_restarts=0 "${@:2}"
  ' bash "$model" "$release/run_energy_formal_v5.py" --contract "$release/contract.json" \
    --factors "$base/generated/FactorsCalibrated" --input-root "$old/recon_inputs/ideal" \
    --output "$output" --model "$model" --mode "$ENERGY_V5_EXECUTION" \
    --iterations "$iterations" --save-step "$save" "${authority[@]}"
  "$RESPONSE_PYTHON" "$release/verify_energy_formal_v5.py" --result "$output" --contract "$release/contract.json" \
    --allocation "$study/formal_allocation_${SLURM_JOB_ID}.txt" --mode "$ENERGY_V5_EXECUTION"
done
echo "ENERGY_V5_FORMAL_ENTRY_COMPLETE $ENERGY_V5_EXECUTION"
