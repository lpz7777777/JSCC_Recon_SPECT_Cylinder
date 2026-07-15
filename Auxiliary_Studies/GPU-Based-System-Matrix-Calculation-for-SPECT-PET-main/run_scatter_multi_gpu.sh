#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 RUN_DIR PE_MATRIX GPU_LIST [SCATTER_BINARY]" >&2
    echo "Example: $0 runs/JSCC_218keV PE_SysMat_shift_0.000000_0.000000_0.000000_v3.sysmat 0,1" >&2
    exit 2
fi

RUN_DIR="$(realpath "$1")"
PE_MATRIX="$2"
if [[ "$PE_MATRIX" != /* ]]; then
    PE_MATRIX="$RUN_DIR/$PE_MATRIX"
fi
PE_MATRIX="$(realpath "$PE_MATRIX")"
IFS=',' read -r -a GPUS <<< "$3"
BINARY="${4:-$ROOT_DIR/ScatterGen_RayTracing_CircularHole/ScatterGen_CircularHole_optimized}"
BINARY="$(realpath "$BINARY")"

for parameter in Params_Collimator.dat Params_Detector.dat Params_Image.dat Params_Physics.dat; do
    [[ -f "$RUN_DIR/$parameter" ]] || { echo "Missing $RUN_DIR/$parameter" >&2; exit 1; }
done
[[ -f "$PE_MATRIX" ]] || { echo "Missing $PE_MATRIX" >&2; exit 1; }
[[ -x "$BINARY" ]] || { echo "Not executable: $BINARY" >&2; exit 1; }
(( ${#GPUS[@]} > 0 )) || { echo "GPU_LIST is empty" >&2; exit 1; }

DETECTOR_COUNT="$(od -An -tf4 -N4 "$RUN_DIR/Params_Detector.dat" | awk '{printf "%.0f", $1}')"
COMBINED_FLAG="$(od -An -tf4 -j12 -N4 "$RUN_DIR/Params_Physics.dat" | awk '{printf "%.0f", $1}')"
PARTIAL_ROOT="$(mktemp -d "$RUN_DIR/.scatter_partials.XXXXXX")"
CACHE_FILE="$RUN_DIR/Geometry_CrystalPairMaterialLengths_v1.cache"
CHUNK_SIZE="${SCATTER_CRYSTAL_CHUNK:-64}"
declare -a PIDS=()
declare -a PARTIAL_DIRS=()

echo "Launching ${#GPUS[@]} partitions for $DETECTOR_COUNT scatter crystals"
for index in "${!GPUS[@]}"; do
    start=$((index * DETECTOR_COUNT / ${#GPUS[@]}))
    end=$(((index + 1) * DETECTOR_COUNT / ${#GPUS[@]}))
    directory="$PARTIAL_ROOT/part_${index}_${start}_${end}"
    mkdir -p "$directory"
    for parameter in Params_Collimator.dat Params_Detector.dat Params_Image.dat Params_Physics.dat; do
        ln -s "$RUN_DIR/$parameter" "$directory/$parameter"
    done
    ln -s "$PE_MATRIX" "$directory/PE_input.sysmat"
    include_global=0
    [[ "$index" == 0 ]] && include_global=1
    (
        cd "$directory"
        CUDA_VISIBLE_DEVICES="${GPUS[$index]}" \
        SCATTER_CRYSTAL_CHUNK="$CHUNK_SIZE" \
        SCATTER_CRYSTAL_START="$start" \
        SCATTER_CRYSTAL_END="$end" \
        SCATTER_INCLUDE_GLOBAL_COMPONENTS="$include_global" \
        SCATTER_PAIR_LENGTH_CACHE="$CACHE_FILE" \
        "$BINARY" -PE PE_input.sysmat -cuda 0 > ScatterGen.log 2>&1
    ) &
    PIDS+=("$!")
    PARTIAL_DIRS+=("$directory")
    echo "  GPU ${GPUS[$index]}: A=[$start,$end) PID=${PIDS[$index]} global_components=$include_global"
done

failed=0
for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$index]}"; then
        echo "Partition $index failed; see ${PARTIAL_DIRS[$index]}/ScatterGen.log" >&2
        failed=1
    fi
done
(( failed == 0 )) || exit 1

declare -a PARTIAL_MATRICES=()
matrix_basename=""
for directory in "${PARTIAL_DIRS[@]}"; do
    matrix="$(find "$directory" -maxdepth 1 -type f -name 'Scatter_SysMat_shift_*.sysmat' -print -quit)"
    [[ -n "$matrix" ]] || { echo "Missing partial matrix in $directory" >&2; exit 1; }
    if [[ -z "$matrix_basename" ]]; then
        matrix_basename="$(basename "$matrix")"
    elif [[ "$(basename "$matrix")" != "$matrix_basename" ]]; then
        echo "Partial output names do not match" >&2
        exit 1
    fi
    PARTIAL_MATRICES+=("$matrix")
done

temporary_scatter="$RUN_DIR/.$matrix_basename.tmp.$$"
node "$ROOT_DIR/tools/sum_float32_matrices.js" \
    "$temporary_scatter" "${PARTIAL_MATRICES[@]}"
mv -f "$temporary_scatter" "$RUN_DIR/$matrix_basename"
echo "Merged scatter matrix: $RUN_DIR/$matrix_basename"

windowed_pe="$(find "$RUN_DIR" -maxdepth 1 -type f -name 'PE_Windowed_SysMat_shift_*.sysmat' -print -quit)"
if [[ -n "$windowed_pe" && "$COMBINED_FLAG" == 1 ]]; then
    combined_basename="${matrix_basename/Scatter_SysMat/SysMat_withScatter}"
    temporary_combined="$RUN_DIR/.$combined_basename.tmp.$$"
    node "$ROOT_DIR/tools/sum_float32_matrices.js" \
        "$temporary_combined" "$windowed_pe" "$RUN_DIR/$matrix_basename"
    mv -f "$temporary_combined" "$RUN_DIR/$combined_basename"
    echo "Merged combined matrix: $RUN_DIR/$combined_basename"
fi

echo "Partition logs retained in $PARTIAL_ROOT"
