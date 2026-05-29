#!/bin/bash
#SBATCH -J JSCC_TStream_LPZ
#SBATCH -N 1
#SBATCH -o log/stdout.%j
#SBATCH -e log/stderr.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH -p gnall
#SBATCH --no-requeue
#SBATCH --time=05:00:00

find_common_sh() {
    local start_dir current depth candidate
    for start_dir in "${JSCC_REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
        [[ -n "$start_dir" ]] || continue
        current=$(cd "$start_dir" 2>/dev/null && pwd) || continue
        depth=0
        while [[ "$current" != "/" && $depth -lt 8 ]]; do
            candidate="$current/distributed/scripts_tansuo1000/_common.sh"
            if [[ -f "$candidate" ]]; then
                echo "$candidate"
                return 0
            fi
            current=$(dirname "$current")
            depth=$((depth + 1))
        done
    done
    return 1
}

COMMON_SH=$(find_common_sh) || {
    echo "Failed to locate distributed/scripts_tansuo1000/_common.sh. Please export JSCC_REPO_ROOT=/path/to/repo before sbatch." >&2
    exit 1
}
source "$COMMON_SH"
run_torch_distributed "distributed/python/main_dist_tstream.py" "$@"
