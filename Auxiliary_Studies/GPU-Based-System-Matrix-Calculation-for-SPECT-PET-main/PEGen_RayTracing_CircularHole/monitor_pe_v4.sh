#!/usr/bin/env bash

set -euo pipefail

PROGRESS_PATH="${1:-PE_v4_progress.json}"
INTERVAL_SECONDS="${2:-10}"

while true; do
    if [[ ! -f "$PROGRESS_PATH" ]]; then
        printf 'Waiting for %s\n' "$PROGRESS_PATH"
        sleep "$INTERVAL_SECONDS"
        continue
    fi
    set +e
    python3 - "$PROGRESS_PATH" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text())
total = state["total_rows"]
fraction = state["completed_rows"] / total if total else 0.0
eta = state["eta_seconds"]
hours, remainder = divmod(max(0, int(eta)), 3600)
minutes, seconds = divmod(remainder, 60)
print(
    f'[{state["last_update"]}] status={state["status"]} '
    f'rows={state["completed_rows"]}/{total} ({fraction:.2%}) '
    f'rate={state["elements_per_second"] / 1e6:.3f} M element/s '
    f'ETA={hours:02d}:{minutes:02d}:{seconds:02d}'
)
print(
    f'  detector={state["current_detector"]} rotation={state["current_rotation"]} '
    f'nonzero={state["nonzero_elements"]} '
    f'raw_sum={state["unwindowed_sum"]:.6e} '
    f'windowed_sum={state["windowed_sum"]:.6e}'
)
if state["status"] in {"complete", "failed"}:
    raise SystemExit(10)
PY
    status=$?
    set -e
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader || true
    if [[ "$status" -eq 10 ]]; then
        exit 0
    fi
    sleep "$INTERVAL_SECONDS"
done
