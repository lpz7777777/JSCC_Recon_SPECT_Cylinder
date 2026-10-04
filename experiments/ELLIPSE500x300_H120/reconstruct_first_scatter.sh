#!/usr/bin/env bash
# Submit only through first_scatter_imaging.py after the independent gates pass.
#SBATCH -J NEMA_first_scatter_v2
#SBATCH -p gpu_4090,gpu_5090
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --qos=gpugpu
#SBATCH --time=06:00:00
set -euo pipefail
: "${FIRST_SCATTER_RELEASE:?Frozen release is required}"
release="$FIRST_SCATTER_RELEASE"
root=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor
base="$root/experiments/ELLIPSE500x300_H120"
study="$base/generated/compton_first_scatter_v2"
export RESPONSE_PROJECT_ROOT="$root"
export RESPONSE_PYTHON=/data/home/scxi717/.conda/envs/torch/bin/python
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
[[ "$SLURM_NNODES" == 4 || "$SLURM_NNODES" == 8 ]]
[[ "${SLURM_GPUS_ON_NODE:-1}" == 1 ]]
export JSCC_PROJECT_ROOT="$release" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=bond0 TORCH_ELASTIC_WORKER_IDENTICAL=1
export ELLIPSE_MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export ELLIPSE_MASTER_PORT=$((50000+SLURM_JOB_ID%10000)) ELLIPSE_GPUS_PER_NODE=1
scontrol show job "$SLURM_JOB_ID" > "$study/allocation_${SLURM_JOB_ID}.txt"
baseline="$base/generated/Results/NEMA_Body_H60_1e9_1643142"
srun --chdir=/tmp --label --kill-on-bad-exit=1 bash -c '
  ready=0
  for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
    if [[ -x $RESPONSE_PYTHON && -r $FIRST_SCATTER_RELEASE/validation_gate.json &&
          -r $FIRST_SCATTER_RELEASE/legacy.json && -r $FIRST_SCATTER_RELEASE/ideal.json &&
          -r $FIRST_SCATTER_RELEASE/geometry.npz &&
          -r $RESPONSE_PROJECT_ROOT/experiments/ELLIPSE500x300_H120/generated/FactorsCalibrated/440keV_RotateNum20/SysMat_polar ]]; then
      ready=1;break
    fi
    sleep 5
  done
  [[ $ready == 1 ]]
  timeout --signal=TERM --kill-after=10s 60s "$RESPONSE_PYTHON" -c "import socket,torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(socket.gethostname(),torch.__version__,flush=True)"
'
export ABLATION_HOST_ALLOCATED_BYTES=$("$RESPONSE_PYTHON" "$release/verify_first_scatter.py" \
  --allocation-only "$study/allocation_${SLURM_JOB_ID}.txt" --nodes "$SLURM_NNODES")
# Both full-data pilots precede both formal runs. A failed pilot holds imaging.
for step in regression_legacy pilot_legacy pilot_ideal formal_legacy formal_ideal; do
  phase=${step%%_*}; group=${step##*_}
  config="$release/$group.json"
  data="$study/recon_inputs/$group"
  case "$phase" in
    regression) iterations=50;save=50;mode=(--baseline-regression);limit=2h;;
    pilot) iterations=10;save=10;mode=(--pilot-only --compton-sensitivity "$study/analysis/$group/Sensi_d");limit=2h;;
    formal) iterations=2000;save=50;mode=(--compton-sensitivity "$study/analysis/$group/Sensi_d");limit=4h;;
  esac
  output="$study/${step}_${SLURM_JOB_ID}"
  echo "FIRST_SCATTER_PHASE $step $iterations"
  timeout --signal=TERM --kill-after=60s "$limit" srun --chdir=/tmp --label --kill-on-bad-exit=1 bash -c '
    exec "$RESPONSE_PYTHON" -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc_per_node=1 \
      --node_rank="$SLURM_PROCID" --master_addr="$ELLIPSE_MASTER_ADDR" --master_port="$ELLIPSE_MASTER_PORT" \
      --rdzv_backend=static --rdzv_conf=timeout=300 --rdzv_id="${SLURM_JOB_ID}_$1" --max_restarts=0 "${@:2}"
  ' bash "$step" "$release/run_reconstruction.py" --factors "$base/generated/FactorsCalibrated" \
    --geometry "$release/geometry.npz" --data-root "$data" --dataset NEMA_Body_H60 --level 1e9 \
    --channels compton-jscc --response-filter-config "$config" --iterations "$iterations" \
    --save-step "$save" --output "$output" "${mode[@]}"
  "$RESPONSE_PYTHON" "$release/verify_first_scatter.py" --result "$output" --config "$config" \
    --geometry "$release/geometry.npz" --baseline "$baseline" --mode "$phase" \
    --allocation "$study/allocation_${SLURM_JOB_ID}.txt"
done
echo FIRST_SCATTER_PAIRED_COMPLETE
