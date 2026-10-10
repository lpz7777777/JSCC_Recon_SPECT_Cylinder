#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
[[ "$SLURM_NNODES" == 8 && "$SLURM_NTASKS" == 8 ]]
export JSCC_PROJECT_ROOT=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/generated/jscc_geant4_5e10_10000/trial_selection_releases/1516a532adb59f94 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=bond0 TORCH_ELASTIC_WORKER_IDENTICAL=1
export JSCC_PHASE_SECONDS=3600 JSCC_PREPARE_SECONDS=10800
export JSCC_MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export JSCC_MASTER_PORT=$((40000+SLURM_JOB_ID%10000))
allocation=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/generated/jscc_geant4_5e10_10000/allocations/selection_5090_8x2_trial_"${SLURM_JOB_ID}.txt"
scontrol show job "$SLURM_JOB_ID" > "$allocation"
export JSCC_HOST_BYTES_NODE=$(cd "$JSCC_PROJECT_ROOT" && /data/home/scxi717/.conda/envs/torch/bin/python -c 'import sys;from jscc_5e10_common import host_allocated_bytes;print(host_allocated_bytes(open(sys.argv[1]).read(),8))' "$allocation")
srun --chdir=/tmp --label --kill-on-bad-exit=1 /data/home/scxi717/.conda/envs/torch/bin/python -c 'import torch;assert torch.cuda.device_count()==2;print("TWO_VISIBLE_GPUS",flush=True)'
output=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/generated/jscc_geant4_5e10_10000/selection_5090_8x2_trial_"${SLURM_JOB_ID}"
timeout --signal=TERM --kill-after=60s 14220s srun --chdir=/tmp --label --kill-on-bad-exit=1 bash -c '
  exec /data/home/scxi717/.conda/envs/torch/bin/python -m torch.distributed.run --nnodes=8 --nproc_per_node=2 --node_rank="$SLURM_PROCID"     --master_addr="$JSCC_MASTER_ADDR" --master_port="$JSCC_MASTER_PORT" --rdzv_backend=static     --rdzv_conf=timeout=600 --rdzv_id="${SLURM_JOB_ID}_three" --max_restarts=0 "${@:1}"
' bash /data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/generated/jscc_geant4_5e10_10000/trial_selection_releases/1516a532adb59f94/jscc_5e10_selection.py --release /data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/generated/jscc_geant4_5e10_10000/trial_selection_releases/1516a532adb59f94 --input /data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/generated/jscc_geant4_5e10_10000/input --factors /data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/generated/FactorsCalibrated --output "$output" --allocation "$allocation"
echo JSCC5E10_COMPUTATION_COMPLETED selection_5090_8x2_trial
