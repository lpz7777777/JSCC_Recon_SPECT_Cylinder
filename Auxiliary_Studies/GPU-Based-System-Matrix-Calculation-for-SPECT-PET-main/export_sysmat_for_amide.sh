#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATLAB_BIN="${MATLAB_BIN:-matlab}"
TOOL_DIR="$ROOT_DIR/ExportSysMatForAmide"

command -v "$MATLAB_BIN" >/dev/null 2>&1 || {
    printf 'MATLAB executable not found: %s\n' "$MATLAB_BIN" >&2
    exit 1
}

exec "$MATLAB_BIN" -nodesktop -nosplash -r \
    "addpath('$TOOL_DIR'); try, run_export_for_amide; catch exception, fprintf(2, '%s\\n', getReport(exception, 'extended')); exit(1); end; exit(0);"
