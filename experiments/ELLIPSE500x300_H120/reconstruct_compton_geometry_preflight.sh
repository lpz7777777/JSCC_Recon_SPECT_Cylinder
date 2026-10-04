#!/usr/bin/env bash
# This bounded entry has no formal-imaging phase.
set -euo pipefail
: "${GEOMETRY_V3_RELEASE:?Frozen release required}"
release="$GEOMETRY_V3_RELEASE"
root=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor
base="$root/experiments/ELLIPSE500x300_H120"
study="$base/generated/compton_response_geometry_v3"
old="$base/generated/compton_first_scatter_v2"
export RESPONSE_PROJECT_ROOT="$root" RESPONSE_PYTHON=/data/home/scxi717/.conda/envs/torch/bin/python
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
[[ "$SLURM_NNODES" == 4 || "$SLURM_NNODES" == 8 ]]
[[ "${SLURM_GPUS_ON_NODE:-1}" == 1 ]]
export JSCC_PROJECT_ROOT="$release" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=bond0 TORCH_ELASTIC_WORKER_IDENTICAL=1
export ELLIPSE_MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export ELLIPSE_MASTER_PORT=$((50000+SLURM_JOB_ID%10000))
scontrol show job "$SLURM_JOB_ID" > "$study/allocation_${SLURM_JOB_ID}.txt"
srun --chdir=/tmp --label --kill-on-bad-exit=1 bash -c '
  timeout --signal=TERM --kill-after=10s 60s "$RESPONSE_PYTHON" -c "import socket,torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(socket.gethostname(),flush=True)"
  [[ -r "$GEOMETRY_V3_RELEASE/R1.json" && -r "$GEOMETRY_V3_RELEASE/geometry.npz" ]]
'
export ABLATION_HOST_ALLOCATED_BYTES=$("$RESPONSE_PYTHON" "$release/verify_first_scatter.py" \
  --allocation-only "$study/allocation_${SLURM_JOB_ID}.txt" --nodes "$SLURM_NNODES")
for phase in regression pilot; do
  if [[ "$phase" == regression ]]; then
    iterations=50;save=50;sensi="$old/analysis/ideal/Sensi_d";mode=(--baseline-regression)
  else
    iterations=10;save=10;sensi="$release/Sensi_d";mode=(--pilot-only)
  fi
  output="$study/preflight_${phase}_${SLURM_JOB_ID}"
  echo "COMPTON_GEOMETRY_PREFLIGHT $phase"
  timeout --signal=TERM --kill-after=60s 35m srun --chdir=/tmp --label --kill-on-bad-exit=1 bash -c '
    exec "$RESPONSE_PYTHON" -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc_per_node=1 \
      --node_rank="$SLURM_PROCID" --master_addr="$ELLIPSE_MASTER_ADDR" --master_port="$ELLIPSE_MASTER_PORT" \
      --rdzv_backend=static --rdzv_conf=timeout=300 --rdzv_id="${SLURM_JOB_ID}_$1" --max_restarts=0 "${@:2}"
  ' bash "$phase" "$release/run_reconstruction.py" --factors "$base/generated/FactorsCalibrated" \
    --geometry "$release/geometry.npz" --data-root "$old/recon_inputs/ideal" --dataset NEMA_Body_H60 --level 1e9 \
    --channels compton-jscc --response-filter-config "$release/R1.json" --iterations "$iterations" \
    --save-step "$save" --output "$output" --compton-sensitivity "$sensi" "${mode[@]}"
  "$RESPONSE_PYTHON" "$release/verify_first_scatter.py" --result "$output" --config "$release/R1.json" \
    --geometry "$release/geometry.npz" --baseline "$old/formal_ideal_1660254" --mode "$phase" \
    --allocation "$study/allocation_${SLURM_JOB_ID}.txt"
done
echo COMPTON_GEOMETRY_PREFLIGHT_COMPLETE_NO_FORMAL
