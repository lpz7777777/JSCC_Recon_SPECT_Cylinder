#!/bin/bash
set -euo pipefail

find_repo_root() {
    local entrypoint_rel="$1"
    local start_dir current depth

    for start_dir in "${JSCC_REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
        [[ -n "$start_dir" ]] || continue
        current=$(cd "$start_dir" 2>/dev/null && pwd) || continue
        depth=0
        while [[ "$current" != "/" && $depth -lt 8 ]]; do
            if [[ -f "$current/$entrypoint_rel" && -f "$current/distributed/python/_path_setup.py" ]]; then
                echo "$current"
                return 0
            fi
            current=$(dirname "$current")
            depth=$((depth + 1))
        done
    done
    return 1
}

setup_tansuo1000_env() {
    module load gpu/cuda/v12
    source /apps/soft/anaconda3/bin/activate
    conda activate pytorch-gpu
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
}

run_torch_distributed() {
    local entrypoint_rel="$1"
    shift || true

    local repo_root
    repo_root=$(find_repo_root "$entrypoint_rel") || {
        echo "Failed to locate repo root. Please export JSCC_REPO_ROOT=/path/to/repo before sbatch." >&2
        exit 1
    }

    setup_tansuo1000_env
    cd "$repo_root"

    local gpus_per_node
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
        gpus_per_node="${SLURM_GPUS_ON_NODE}"
    else
        gpus_per_node=8
    fi

    local entrypoint_abs="$repo_root/$entrypoint_rel"
    if [[ ! -f "$entrypoint_abs" ]]; then
        echo "Entry point not found: $entrypoint_abs" >&2
        exit 1
    fi

    echo "Repo Root: $repo_root"
    echo "Entrypoint: $entrypoint_abs"
    echo "GPUS_PER_NODE: $gpus_per_node"
    echo "Node List: ${SLURM_JOB_NODELIST:-single-node}"
    
    local nnodes
    nnodes="${SLURM_NNODES:-1}"

    if [[ "$nnodes" -le 1 ]]; then
        echo "Launching single-node distributed job"
        torchrun \
            --standalone \
            --nproc_per_node="$gpus_per_node" \
            "$entrypoint_abs" "$@"
        return
    fi

    local master_node master_port
    master_node=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
    master_port="${MASTER_PORT:-29500}"

    echo "Launching multi-node distributed job"
    echo "NNODES: $nnodes"
    echo "MASTER_NODE: $master_node"
    echo "MASTER_PORT: $master_port"

    srun --kill-on-bad-exit=1 \
        --ntasks="$nnodes" \
        --ntasks-per-node=1 \
        torchrun \
        --nnodes="$nnodes" \
        --nproc_per_node="$gpus_per_node" \
        --rdzv_id="${SLURM_JOB_ID:-jscc_job}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$master_node:$master_port" \
        "$entrypoint_abs" "$@"
}
