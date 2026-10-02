#!/usr/bin/env bash
#SBATCH --job-name=NEMA_ablation
#SBATCH --partition=gpu_4090,gpu_5090
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --qos=gpugpu
#SBATCH --time=48:00:00
set -euo pipefail
: "${ABLATION_RELEASE:?Immutable release path required}"
: "${ABLATION_PHASE:?pipeline, pilot, short, or formal required}"
root=/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor
base="$root/experiments/ELLIPSE500x300_H120"
study="$base/generated/SpikeAblation/NEMA_5e9_SPIKE_ABLATION_V1/study.json"
export JSCC_PROJECT_ROOT="$root" PYTHONUNBUFFERED=1 NCCL_SOCKET_IFNAME=bond0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} TORCH_ELASTIC_WORKER_IDENTICAL=1
export ABLATION_PYTHON=/data/home/scxi717/.conda/envs/torch/bin/python
# Some compute nodes expose shared paths only after the first access. Do not
# initialize a process group before every allocated node passes its path check.
for attempt in 1 2 3 4 5 6; do
  if [[ -d $root && -x $ABLATION_PYTHON ]]; then break; fi
  sleep 5
done
test -d "$root" && test -x "$ABLATION_PYTHON"
cd "$root"
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch
if [[ $ABLATION_PHASE == pipeline ]]; then
  # Account MaxSubmitJobs=50: one submission executes all five gated contrasts.
  # Timeout applies per phase; the outer wall limit covers their sum.
  for index in 0 1 2 3 4; do
    variant=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["variants"][int(sys.argv[2])]["id"])' "$study" "$index")
    for phase in pilot short formal; do
      gate="$(dirname "$study")/gates/${variant}.${phase}.json"
      if [[ -f $gate ]]; then
        python - "$gate" "$study" "$ABLATION_RELEASE/release.json" <<'PY'
import hashlib,json,pathlib,sys
gate=json.load(open(sys.argv[1]))
assert gate['passed']
assert gate['study_sha256']==hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest()
assert gate['release_sha256']==hashlib.sha256(pathlib.Path(sys.argv[3]).read_bytes()).hexdigest()
PY
        echo "SPIKE_ABLATION_ALREADY_VERIFIED $variant $phase"
        continue
      fi
      case "$phase" in pilot) limit=2h ;; short) limit=4h ;; formal) limit=48h ;; esac
      timeout --signal=TERM --kill-after=60s "$limit" env ABLATION_PHASE="$phase" \
        ABLATION_VARIANT_INDEX="$index" bash "$ABLATION_RELEASE/reconstruct_spike_ablation.sh"
    done
  done
  echo "SPIKE_ABLATION_ALL_FIVE_FORMAL_GATES_PASSED"
  exit 0
fi
# This cluster forbids explicit --mem flags and assigns memory per GPU.
# Freeze the actual grant, including the fallback if Slurm omits the env var.
export ABLATION_HOST_ALLOCATED_BYTES=$(python - <<'PY'
import os,re,subprocess
value=os.environ.get('SLURM_MEM_PER_NODE')
if value and int(value)>0:
    print(int(value)*1024**2)
else:
    value=os.environ.get('SLURM_MEM_PER_CPU')
    if value and int(value)>0:
        # One rank/task per node, six granted CPUs; no unrequested host RAM.
        print(int(value)*int(os.environ['SLURM_CPUS_PER_TASK'])*1024**2)
    else:
        text=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'-o'],text=True)
        match=re.search(r'\bMinMemory(Node|CPU)=([0-9.]+)([KMGT])',text)
        if not match:
            raise SystemExit('Cannot determine actual node memory grant')
        size=float(match[2])*1024**({'K':1,'M':2,'G':3,'T':4}[match[3]])
        if match[1]=='CPU':
            size*=int(os.environ['SLURM_CPUS_PER_TASK'])
        print(int(size))
PY
)
python - "$ABLATION_HOST_ALLOCATED_BYTES" <<'PY'
import sys
assert int(sys.argv[1])>=55*1024**3, 'Host grant below the conservative 55GiB budget'
PY
index=${ABLATION_VARIANT_INDEX:-${SLURM_ARRAY_TASK_ID:?Variant index required}}
variant=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["variants"][int(sys.argv[2])]["id"])' "$study" "$index")
python - "$ABLATION_RELEASE" "$study" "$variant" "$ABLATION_PHASE" <<'PY'
import hashlib,json,pathlib,sys
release,study=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2])
record=json.load(open(release/'release.json'))
for name,row in record['files'].items():
    path=release/name
    assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],name
parent={'pilot':None,'short':'pilot','formal':'short'}[sys.argv[4]]
if parent:
    gate=json.load(open(study.parent/'gates'/f'{sys.argv[3]}.{parent}.json'))
    assert gate['passed'] and gate['study_sha256']==hashlib.sha256(study.read_bytes()).hexdigest()
    assert gate['release_sha256']==hashlib.sha256((release/'release.json').read_bytes()).hexdigest()
PY
case "$ABLATION_PHASE" in
  pilot) iterations=10; save_step=10; mode=(--pilot-only) ;;
  short) iterations=200; save_step=50; mode=(--study-short) ;;
  formal) iterations=10000; save_step=50; mode=() ;;
  *) exit 2 ;;
esac
python "$ABLATION_RELEASE/resource_budget.py" --accepted-events 484936 --nodes "$SLURM_NNODES" \
  --gpus-per-node 1 --gpu-gib 22 --host-gib 55 > "$base/logs/ablation-budget.${SLURM_JOB_ID}.json"
python - "$base/logs/ablation-budget.${SLURM_JOB_ID}.json" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))[0]
assert row['gpu_20_percent_margin_pass'] and row['host_20_percent_margin_pass']
PY
master=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export ELLIPSE_MASTER_ADDR="$master" ELLIPSE_MASTER_PORT=$((50000+SLURM_JOB_ID%10000))
study_job=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
output="$base/generated/Results/NEMA_Body_H60_5e9_${variant}_${ABLATION_PHASE}_${study_job}_${index}"
echo "SPIKE_ABLATION_START $variant $ABLATION_PHASE $output"
srun --chdir=/tmp --label --kill-on-bad-exit=1 bash -c '
  echo "SPIKE_NODE_PREFLIGHT $(hostname) rank=$SLURM_PROCID"
  base="$JSCC_PROJECT_ROOT/experiments/ELLIPSE500x300_H120"
  ready=0
  for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
    if [[ -x $ABLATION_PYTHON && -r $ABLATION_RELEASE/run_reconstruction.py &&
          -r $base/generated/Geometry/geometry.npz &&
          -r $base/generated/FactorsCalibrated/440keV_RotateNum20/SysMat_polar &&
          -r $base/generated/List/218-440keV_RotateNum20_Geant4JSCC/List_NEMA_Body_H60_5e9/20.csv ]]; then
      ready=1; break
    fi
    echo "SPIKE_NODE_PATH_WAIT $(hostname) attempt=$attempt"
    sleep 5
  done
  if [[ $ready != 1 ]]; then
    echo "SPIKE_NODE_PATH_FAILED $(hostname)" >&2
    exit 78
  fi
  "$ABLATION_PYTHON" -c "import socket,torch; print(\"SPIKE_NODE_CUDA_OK\",socket.gethostname(),torch.__version__,torch.cuda.device_count(),flush=True); assert torch.cuda.is_available() and torch.cuda.device_count()==1"
'
srun --chdir=/tmp --label --kill-on-bad-exit=1 bash -c '
  exec "$ABLATION_PYTHON" -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc_per_node=1 \
    --node_rank="$SLURM_PROCID" --master_addr="$ELLIPSE_MASTER_ADDR" \
    --master_port="$ELLIPSE_MASTER_PORT" --rdzv_backend=static \
    --rdzv_conf=timeout=300 --rdzv_id="$SLURM_JOB_ID" --max_restarts=0 "$@"
' bash "$ABLATION_RELEASE/run_reconstruction.py" \
  --factors "$base/generated/FactorsCalibrated" --geometry "$base/generated/Geometry/geometry.npz" \
  --data-root "$base/generated" --dataset NEMA_Body_H60 --level 5e9 \
  --output "$output" --iterations "$iterations" --save-step "$save_step" \
  --study-json "$study" --variant "$variant" "${mode[@]}"
if [[ $ABLATION_PHASE == formal ]]; then
  python "$base/verify_formal_result.py" "$output"
fi
python "$ABLATION_RELEASE/verify_spike_ablation.py" --experiment-root "$base" \
  --study-json "$study" --variant "$variant" --phase "$ABLATION_PHASE" --result "$output" \
  --job "$SLURM_JOB_ID"
