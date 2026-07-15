#!/usr/bin/env bash
# Monitor JSCC and EHE scatter jobs without modifying their processes or files.

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCH_MODE=0
INTERVAL=30
SHOW_JSCC=1
SHOW_EHE=1
SHOW_CROSS=1
SHOW_JSCC_DIRECT=1
SHOW_EHE_DIRECT=1

if [[ -n "${FORCE_COLOR:-}" ]] || { [[ -z "${NO_COLOR:-}" ]] && [[ -t 1 ]] && [[ "${TERM:-dumb}" != "dumb" ]]; }; then
    COLOR_ENABLED=1
    COLOR_RESET=$'\033[0m'
    COLOR_BOLD=$'\033[1m'
    COLOR_DIM=$'\033[2m'
    COLOR_RED=$'\033[31m'
    COLOR_GREEN=$'\033[32m'
    COLOR_YELLOW=$'\033[33m'
    COLOR_BLUE=$'\033[34m'
    COLOR_MAGENTA=$'\033[35m'
    COLOR_CYAN=$'\033[36m'
else
    COLOR_ENABLED=0
    COLOR_RESET=""
    COLOR_BOLD=""
    COLOR_DIM=""
    COLOR_RED=""
    COLOR_GREEN=""
    COLOR_YELLOW=""
    COLOR_BLUE=""
    COLOR_MAGENTA=""
    COLOR_CYAN=""
fi

styled() {
    local style=$1
    local text=$2
    if ((COLOR_ENABLED)); then
        printf '%s%s%s' "$style" "$text" "$COLOR_RESET"
    else
        printf '%s' "$text"
    fi
}

integer_style() {
    local value=$1
    local warning_at=$2
    local critical_at=$3
    local value_int=${value%%.*}

    if [[ ! $value_int =~ ^[0-9]+$ ]]; then
        printf '%s' "$COLOR_DIM"
    elif ((value_int >= critical_at)); then
        printf '%s' "$COLOR_BOLD$COLOR_RED"
    elif ((value_int >= warning_at)); then
        printf '%s' "$COLOR_YELLOW"
    else
        printf '%s' "$COLOR_GREEN"
    fi
}

usage() {
    cat <<'EOF'
Usage:
  ./monitor_scatter_progress.sh
  ./monitor_scatter_progress.sh --watch [seconds]
  ./monitor_scatter_progress.sh --ehe-only --watch [seconds]
  ./monitor_scatter_progress.sh --direct-only --watch [seconds]
  ./monitor_scatter_progress.sh --cross-only --watch [seconds]
  ./monitor_scatter_progress.sh --jscc-ehe-cross --watch [seconds]

Options:
  --watch [seconds]  Refresh continuously. Default interval: 30 seconds.
  --interval seconds Set the refresh interval; implies --watch.
  --ehe-only         Show only the three EHE Pb/NaI scatter jobs.
  --jscc-only        Show only the three JSCC scatter jobs.
  --direct-only      Hide both 440-to-218 keV cross-window jobs.
  --cross-only       Show only the JSCC and EHE 440-to-218 keV cross-window jobs.
  --jscc-ehe-cross   Show JSCC 218/440/cross and only the EHE cross-window job.
  -h, --help         Show this help text.

The reported progress is a lower bound. A log line is written when a chunk starts,
so the currently running chunk is not counted as complete until the next line.
Colors are enabled only in an interactive terminal. Set NO_COLOR=1 to disable them,
or FORCE_COLOR=1 to force colors in a non-standard terminal or override NO_COLOR.
EOF
}

format_duration() {
    local seconds=${1:-0}
    local days=$((seconds / 86400))
    local hours=$(((seconds % 86400) / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))

    if ((days > 0)); then
        printf '%dd %02dh %02dm' "$days" "$hours" "$minutes"
    elif ((hours > 0)); then
        printf '%dh %02dm %02ds' "$hours" "$minutes" "$secs"
    else
        printf '%dm %02ds' "$minutes" "$secs"
    fi
}

read_pid() {
    local pid_file=$1
    local pid=""
    local command_line=""

    if [[ -r $pid_file ]]; then
        read -r pid < "$pid_file" || true
    fi

    if [[ $pid =~ ^[0-9]+$ ]]; then
        command_line=$(ps -p "$pid" -o args= 2>/dev/null || true)
        if [[ $command_line == *ScatterGen_CircularHole* ]]; then
            printf '%s' "$pid"
        fi
    fi
}

read_pe_pid() {
    local pid_file=$1
    local pid=""
    local command_line=""

    if [[ -r $pid_file ]]; then
        read -r pid < "$pid_file" || true
    fi

    if [[ $pid =~ ^[0-9]+$ ]]; then
        command_line=$(ps -p "$pid" -o args= 2>/dev/null || true)
        if [[ $command_line == *PEGen_CircularHole* ]]; then
            printf '%s' "$pid"
        fi
    fi
}

read_projection_count() {
    local parameter_file=$1
    LC_ALL=C od -An -N4 -t f4 "$parameter_file" 2>/dev/null | awk '{ printf "%.0f", $1 }'
}

read_collimator_hole_count() {
    local parameter_file=$1
    LC_ALL=C od -An -j40 -N4 -t f4 "$parameter_file" 2>/dev/null | awk '{ printf "%.0f", $1 }'
}

last_chunk_value() {
    local log_file=$1
    local key=$2
    awk -v key="$key" '
        /Crystal chunk scatterStart=/ {
            for (i = 1; i <= NF; ++i) {
                split($i, pair, "=")
                if (pair[1] == key) {
                    value = pair[2]
                }
            }
        }
        END { if (value != "") print value }
    ' "$log_file" 2>/dev/null
}

max_chunk_value() {
    local log_file=$1
    local key=$2
    awk -v key="$key" '
        /Crystal chunk scatterStart=/ {
            for (i = 1; i <= NF; ++i) {
                split($i, pair, "=")
                if (pair[1] == key && pair[2] + 0 > value) {
                    value = pair[2] + 0
                }
            }
        }
        END { if (value > 0) print value }
    ' "$log_file" 2>/dev/null
}

has_completion_marker() {
    local log_file=$1
    awk '/Compton Scatter Sysmat written\./ { found = 1 } END { exit !found }' "$log_file" 2>/dev/null
}

has_crystal_completion_marker() {
    local log_file=$1
    awk '/Time of crystalScatterSysMatCuda function:/ { found = 1 } END { exit !found }' "$log_file" 2>/dev/null
}

scatter_phase() {
    local log_file=$1
    if awk '/Compton Scatter Sysmat written\./ { found = 1 } END { exit !found }' "$log_file" 2>/dev/null; then
        printf 'complete'
    elif awk '/Kernel collimatorScatterSysMatCuda Launched/ { found = 1 } END { exit !found }' "$log_file" 2>/dev/null; then
        printf 'collimator scatter'
    elif awk '/Initializing Collimator GeometryRelationship|^Total bits:/ { found = 1 } END { exit !found }' "$log_file" 2>/dev/null; then
        printf 'collimator geometry'
    else
        printf 'crystal scatter'
    fi
}

last_error() {
    local log_file=$1
    awk '/GPUassert|launch failed|^error$/ { message = $0 } END { print message }' "$log_file" 2>/dev/null
}

gpu_summary() {
    local gpu=$1
    local values
    local sm_util
    local memory_used
    local memory_total
    local power_draw
    local power_limit
    local temperature
    local sm_style
    local temperature_style

    if ! command -v nvidia-smi >/dev/null 2>&1 || [[ ! $gpu =~ ^[0-9]+$ ]]; then
        printf 'GPU information unavailable'
        return
    fi

    values=$(nvidia-smi -i "$gpu" \
        --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1)
    if [[ -z $values ]]; then
        printf 'GPU %s information unavailable' "$gpu"
        return
    fi

    IFS=',' read -r sm_util memory_used memory_total power_draw power_limit temperature <<< "$values"
    sm_util=${sm_util//[[:space:]]/}
    memory_used=${memory_used//[[:space:]]/}
    memory_total=${memory_total//[[:space:]]/}
    power_draw=${power_draw//[[:space:]]/}
    power_limit=${power_limit//[[:space:]]/}
    temperature=${temperature//[[:space:]]/}

    sm_style=$(integer_style "$sm_util" 80 95)
    temperature_style=$(integer_style "$temperature" 80 88)

    styled "$COLOR_BOLD$COLOR_BLUE" "GPU $gpu"
    printf ': '
    styled "$sm_style" "SM ${sm_util}%"
    printf ' | '
    styled "$COLOR_CYAN" "VRAM ${memory_used} / ${memory_total} MiB"
    printf ' | Power %s / %s W | ' "$power_draw" "$power_limit"
    styled "$temperature_style" "Temp ${temperature} C"
}

show_task() {
    local label=$1
    local relative_dir=$2
    local log_name=$3
    local pid_name=$4
    local task_dir="$ROOT_DIR/$relative_dir"
    local log_file="$task_dir/$log_name"
    local pid_file="$task_dir/$pid_name"
    local parameter_file="$task_dir/Params_Detector.dat"
    local now
    local pid
    local projections
    local collimator_holes=0
    local chunk_size
    local last_start
    local total_chunks
    local completed_chunks=0
    local completed_projections=0
    local percent="0.00"
    local elapsed=""
    local eta_seconds=0
    local log_age=""
    local gpu=""
    local status="STOPPED"
    local phase=""
    local error_message=""
    local pe_pid=""
    local pe_log_file="$task_dir/PEGen.log"
    local status_style
    local log_age_style

    printf '\n'
    styled "$COLOR_BOLD$COLOR_CYAN" "$label"
    printf '\n'

    if [[ ! -r $parameter_file ]]; then
        printf '  Status: configuration is missing\n'
        return
    fi

    if [[ ! -r $log_file ]]; then
        pe_pid=$(read_pe_pid "$task_dir/PEGen.pid")
        if [[ -n $pe_pid ]]; then
            elapsed=$(ps -p "$pe_pid" -o etimes= | tr -d ' ')
            gpu=$(tr '\0' '\n' < "/proc/$pe_pid/environ" 2>/dev/null \
                | awk -F= '$1 == "CUDA_VISIBLE_DEVICES" { split($2, ids, ","); print ids[1]; exit }')
            printf '  Status: '
            styled "$COLOR_BOLD$COLOR_GREEN" 'RUNNING'
            printf ' | PID %s | elapsed ' "$pe_pid"
            format_duration "$elapsed"
            printf '\n  Phase: '
            styled "$COLOR_BOLD$COLOR_BLUE" 'PE generation'
            printf '\n'
            if [[ -r $pe_log_file ]]; then
                now=$(date +%s)
                log_age=$((now - $(stat -c %Y "$pe_log_file")))
                printf '  Log update: %s ago | %s\n' "$(format_duration "$log_age")" "$pe_log_file"
            fi
            if [[ -n $gpu ]]; then
                printf '  '
                gpu_summary "$gpu"
                printf '\n'
            fi
        elif [[ -r $pe_log_file ]] \
            && grep -Fq 'Energy-windowed Photon Electric Sysmat Written.' "$pe_log_file"; then
            printf '  Status: PE complete; waiting for scatter launch\n'
        else
            printf '  Status: not started\n'
        fi
        return
    fi

    projections=$(read_projection_count "$parameter_file")
    if [[ -r $task_dir/Params_Collimator.dat ]]; then
        collimator_holes=$(read_collimator_hole_count "$task_dir/Params_Collimator.dat")
    fi
    chunk_size=$(max_chunk_value "$log_file" "scatterCount")
    last_start=$(last_chunk_value "$log_file" "scatterStart")

    if [[ ! $projections =~ ^[0-9]+$ ]] || ((projections == 0)); then
        printf '  Status: cannot read the projection count from Params_Detector.dat\n'
        return
    fi
    if [[ ! $chunk_size =~ ^[0-9]+$ ]] || ((chunk_size == 0)); then
        printf '  Status: waiting for the first Crystal chunk log entry\n'
        return
    fi

    total_chunks=$(((projections + chunk_size - 1) / chunk_size))
    if has_completion_marker "$log_file"; then
        status="COMPLETE"
        completed_chunks=$total_chunks
        completed_projections=$projections
    elif [[ $last_start =~ ^[0-9]+$ ]]; then
        completed_chunks=$((last_start / chunk_size))
        completed_projections=$last_start
        if has_crystal_completion_marker "$log_file"; then
            completed_chunks=$total_chunks
            completed_projections=$projections
        fi
    fi

    percent=$(awk -v done="$completed_projections" -v total="$projections" 'BEGIN { printf "%.2f", 100 * done / total }')
    pid=$(read_pid "$pid_file")

    if [[ -n $pid ]]; then
        status="RUNNING"
        phase=$(scatter_phase "$log_file")
        elapsed=$(ps -p "$pid" -o etimes= | tr -d ' ')
        gpu=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | awk -F= '$1 == "CUDA_VISIBLE_DEVICES" { split($2, ids, ","); print ids[1]; exit }')
    elif [[ $status != "COMPLETE" ]]; then
        error_message=$(last_error "$log_file")
    fi

    now=$(date +%s)
    log_age=$((now - $(stat -c %Y "$log_file")))

    case $status in
        RUNNING) status_style="$COLOR_BOLD$COLOR_GREEN" ;;
        COMPLETE) status_style="$COLOR_BOLD$COLOR_GREEN" ;;
        *) status_style="$COLOR_BOLD$COLOR_RED" ;;
    esac
    printf '  Status: '
    styled "$status_style" "$status"
    if [[ -n $pid ]]; then
        printf ' | PID %s | elapsed ' "$pid"
        format_duration "$elapsed"
    fi
    printf '\n'

    if [[ $status == "RUNNING" ]]; then
        printf '  Phase: '
        styled "$COLOR_BOLD$COLOR_BLUE" "$phase"
        printf '\n'
    fi

    printf '  Crystal phase: %d/%d chunks complete, >= ' "$completed_chunks" "$total_chunks"
    styled "$COLOR_BOLD$COLOR_GREEN" "${percent}%"
    printf ' (%d/%d projections)\n' "$completed_projections" "$projections"
    if [[ $status == "RUNNING" && $last_start =~ ^[0-9]+$ ]]; then
        printf '  '
        styled "$COLOR_YELLOW" "Current chunk: ${last_start} to $((last_start + chunk_size - 1)) (chunk size ${chunk_size})"
        printf '\n'
    fi

    if ((log_age >= 600)); then
        log_age_style="$COLOR_BOLD$COLOR_RED"
    elif ((log_age >= 180)); then
        log_age_style="$COLOR_YELLOW"
    else
        log_age_style="$COLOR_DIM"
    fi
    printf '  Log update: '
    styled "$log_age_style" "$(format_duration "$log_age") ago"
    printf ' | %s\n' "$log_file"

    if [[ $status == "RUNNING" && $phase == "crystal scatter" && $completed_chunks -gt 0 ]]; then
        local seconds_per_chunk=$((elapsed / completed_chunks))
        local remaining_chunks=$((total_chunks - completed_chunks))
        eta_seconds=$((seconds_per_chunk * remaining_chunks))
        printf '  Forecast basis: %d completed chunks, average ' "$completed_chunks"
        format_duration "$seconds_per_chunk"
        printf '/chunk\n'
        printf '  '
        if [[ $collimator_holes =~ ^[0-9]+$ ]] && ((collimator_holes > 0)); then
            styled "$COLOR_MAGENTA" "Crystal-phase forecast: ${remaining_chunks} chunks remaining | predicted remaining: $(format_duration "$eta_seconds")"
        else
            styled "$COLOR_MAGENTA" "Forecast: ${remaining_chunks} chunks remaining | predicted remaining: $(format_duration "$eta_seconds")"
        fi
        printf '\n  '
        if [[ $collimator_holes =~ ^[0-9]+$ ]] && ((collimator_holes > 0)); then
            styled "$COLOR_BOLD$COLOR_MAGENTA" "Predicted crystal-phase finish: $(date -d "@$((now + eta_seconds))" '+%F %R %Z'); full-job ETA unavailable before collimator timing"
        else
            styled "$COLOR_BOLD$COLOR_MAGENTA" "Predicted finish: $(date -d "@$((now + eta_seconds))" '+%F %R %Z')"
        fi
        printf '\n'
    elif [[ $status == "RUNNING" && $phase == "crystal scatter" ]]; then
        printf '  '
        styled "$COLOR_YELLOW" 'Forecast: waiting for the first chunk to finish'
        printf '\n'
    elif [[ $status == "RUNNING" ]]; then
        printf '  '
        styled "$COLOR_YELLOW" 'Forecast: current collimator phase is not chunked; no reliable full-job ETA yet'
        printf '\n'
    fi

    if [[ -n $gpu ]]; then
        printf '  '
        gpu_summary "$gpu"
        printf '\n'
    fi
    if [[ -n $error_message ]]; then
        printf '  '
        styled "$COLOR_BOLD$COLOR_RED" "Last error: ${error_message}"
        printf '\n'
    fi
}

while (($# > 0)); do
    case $1 in
        --watch)
            WATCH_MODE=1
            if [[ ${2:-} =~ ^[0-9]+$ ]]; then
                INTERVAL=$2
                shift
            fi
            ;;
        --interval)
            if [[ ! ${2:-} =~ ^[0-9]+$ ]] || (( $2 == 0 )); then
                printf '%s\n' '--interval requires a positive integer number of seconds' >&2
                exit 2
            fi
            WATCH_MODE=1
            INTERVAL=$2
            shift
            ;;
        --ehe-only)
            SHOW_JSCC=0
            SHOW_EHE=1
            ;;
        --jscc-only)
            SHOW_JSCC=1
            SHOW_EHE=0
            ;;
        --direct-only)
            SHOW_CROSS=0
            ;;
        --cross-only)
            SHOW_JSCC=1
            SHOW_EHE=1
            SHOW_CROSS=1
            SHOW_JSCC_DIRECT=0
            SHOW_EHE_DIRECT=0
            ;;
        --jscc-ehe-cross)
            SHOW_JSCC=1
            SHOW_EHE=1
            SHOW_CROSS=1
            SHOW_JSCC_DIRECT=1
            SHOW_EHE_DIRECT=0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

trap 'printf "\nMonitoring stopped. Scatter jobs were not changed.\n"; exit 0' INT TERM

while :; do
    if ((WATCH_MODE)) && [[ -t 1 ]]; then
        printf '\033[H\033[2J'
    fi

    styled "$COLOR_BOLD$COLOR_CYAN" 'Scatter progress monitor'
    printf ' | %s\n' "$(date '+%F %T %Z')"
    styled "$COLOR_DIM" 'Progress is a lower bound while a chunk is running.'
    printf '\n'
    printf '%s\n' '--------------------------------------------------------------------------------'

    if ((SHOW_JSCC)); then
        if ((SHOW_JSCC_DIRECT)); then
            show_task "JSCC 218 keV" "runs/JSCC_218keV" "ScatterGen_chunked.log" "ScatterGen_chunked.pid"
            show_task "JSCC 440 keV" "runs/JSCC_440keV" "ScatterGen_chunked.log" "ScatterGen_chunked.pid"
        fi
        if ((SHOW_CROSS)); then
            show_task "JSCC 440 keV to 218 keV window" "runs/JSCC_440keV_to_218keVwin" "ScatterGen_cross_chunked.log" "ScatterGen_cross_chunked.pid"
        fi
    fi
    if ((SHOW_EHE)); then
        if ((SHOW_EHE_DIRECT)); then
            show_task "EHE Pb/NaI 218 keV" "runs/EHE_PbNaI_218keV" "ScatterGen.log" "ScatterGen.pid"
            show_task "EHE Pb/NaI 440 keV" "runs/EHE_PbNaI_440keV" "ScatterGen.log" "ScatterGen.pid"
        fi
        if ((SHOW_CROSS)); then
            show_task "EHE Pb/NaI 440 keV to 218 keV window" "runs/EHE_PbNaI_440keV_to_218keVwin" "ScatterGen.log" "ScatterGen.pid"
        fi
    fi

    if ((!WATCH_MODE)); then
        break
    fi
    printf '\nRefreshing every %s seconds. Press Ctrl-C to stop monitoring.\n' "$INTERVAL"
    sleep "$INTERVAL"
done
