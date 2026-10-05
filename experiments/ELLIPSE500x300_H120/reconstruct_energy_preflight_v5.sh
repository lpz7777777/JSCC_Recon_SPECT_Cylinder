#!/usr/bin/env bash
# Three bounded phases only; no paired/formal submission in this entry.
set -euo pipefail
: "${ENERGY_V5_RELEASE:?Frozen release required}"
release="$ENERGY_V5_RELEASE"
root=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor
base="$root/experiments/ELLIPSE500x300_H120"
study="$base/generated/compton_energy_probability_v5"
old="$base/generated/compton_first_scatter_v2"
export RESPONSE_PYTHON=/data/home/scxi717/.conda/envs/torch/bin/python
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
[[ "$SLURM_NNODES" == 4 || "$SLURM_NNODES" == 8 ]]
export JSCC_PROJECT_ROOT="$release" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=bond0 TORCH_ELASTIC_WORKER_IDENTICAL=1
export ELLIPSE_MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export ELLIPSE_MASTER_PORT=$((50000+SLURM_JOB_ID%10000))
scontrol show job "$SLURM_JOB_ID" > "$study/allocation_${SLURM_JOB_ID}.txt"
srun --chdir=/tmp --label --kill-on-bad-exit=1 bash -c '
  timeout --signal=TERM --kill-after=10s 60s "$RESPONSE_PYTHON" -c "import socket,torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(socket.gethostname(),flush=True)"
  [[ -r "$ENERGY_V5_RELEASE/contract.json" && -r "$ENERGY_V5_RELEASE/whole_geometry.npz" ]]
'
export ABLATION_HOST_ALLOCATED_BYTES=$("$RESPONSE_PYTHON" "$release/verify_first_scatter.py" \
  --allocation-only "$study/allocation_${SLURM_JOB_ID}.txt" --nodes "$SLURM_NNODES")
phases=(regression angular continuous_energy)
if "$RESPONSE_PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("regression_reuse") else 1)' "$release/contract.json"; then
  "$RESPONSE_PYTHON" "$release/verify_energy_preflight_v5.py" --contract "$release/contract.json" \
    --reuse-regression-only --reuse-receipt "$study/regression_reuse_${SLURM_JOB_ID}.json"
  phases=(angular continuous_energy)
fi
for phase in "${phases[@]}"; do
  output="$study/preflight_${phase}_${SLURM_JOB_ID}"
  echo "ENERGY_V5_PREFLIGHT_PHASE $phase"
  if [[ "$phase" == regression ]]; then
    args=("$release/run_reconstruction.py" --factors "$base/generated/FactorsCalibrated"
      --geometry "$release/geometry.npz" --data-root "$old/recon_inputs/ideal" --dataset NEMA_Body_H60 --level 1e9
      --channels compton-jscc --response-filter-config "$release/legacy_regression/R1.json"
      --iterations 50 --save-step 50 --output "$output" --compton-sensitivity "$old/analysis/ideal/Sensi_d"
      --baseline-regression)
  else
    args=("$release/run_energy_preflight_v5.py" --contract "$release/contract.json"
      --factors "$base/generated/FactorsCalibrated" --input-root "$old/recon_inputs/ideal"
      --output "$output" --model "$phase")
  fi
  timeout --signal=TERM --kill-after=60s 25m srun --chdir=/tmp --label --kill-on-bad-exit=1 bash -c '
    exec "$RESPONSE_PYTHON" -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc_per_node=1 \
      --node_rank="$SLURM_PROCID" --master_addr="$ELLIPSE_MASTER_ADDR" --master_port="$ELLIPSE_MASTER_PORT" \
      --rdzv_backend=static --rdzv_conf=timeout=300 --rdzv_id="${SLURM_JOB_ID}_$1" --max_restarts=0 "${@:2}"
  ' bash "$phase" "${args[@]}"
  if [[ "$phase" == regression ]]; then
    "$RESPONSE_PYTHON" "$release/verify_first_scatter.py" --result "$output" \
      --config "$release/legacy_regression/R1.json" --geometry "$release/geometry.npz" \
      --baseline "$old/formal_ideal_1660254" --mode regression \
      --allocation "$study/allocation_${SLURM_JOB_ID}.txt"
  else
    "$RESPONSE_PYTHON" "$release/verify_energy_preflight_v5.py" --result "$output" \
      --contract "$release/contract.json" --allocation "$study/allocation_${SLURM_JOB_ID}.txt"
  fi
done
echo ENERGY_V5_PREFLIGHT_COMPLETE_NO_FORMAL
