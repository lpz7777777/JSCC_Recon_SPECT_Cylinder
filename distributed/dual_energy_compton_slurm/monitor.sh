#!/bin/bash
set -euo pipefail

JOB_ID=${1:?Usage: monitor.sh JOB_ID}
squeue -j "$JOB_ID" -o "%.18i %.12P %.28j %.2t %.10M %.10l %.4D %R"
echo
LOG_DIR="distributed/dual_energy_compton_slurm/logs"
for output in "${LOG_DIR}/${JOB_ID}.log" "${LOG_DIR}/smoke_${JOB_ID}.log"; do
    if [[ -f "$output" ]]; then
        echo "===== $output ====="
        tail -n 40 "$output"
    fi
done
for error in "${LOG_DIR}/${JOB_ID}.err" "${LOG_DIR}/smoke_${JOB_ID}.err"; do
    if [[ -f "$error" ]]; then
        echo "===== $error ====="
        tail -n 20 "$error"
    fi
done
