#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
GEANT4SIM_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
CODE_DIR="${GEANT4SIM_ROOT}/Geant4Code_CntStatOnly"
MACRO_ROOT="${GEANT4SIM_ROOT}/Macro/UniformFovCntStat_CenterPoint"
RUN_ROOT="${1:-${GEANT4SIM_ROOT}/run/UniformFovCntStat_CenterPoint}"

for required in "${CODE_DIR}/build/gamma01" \
    "${CODE_DIR}/CrystalMatrix.txt" "${SCRIPT_DIR}/RunData" \
    "${SCRIPT_DIR}/runevent"; do
    if [ ! -e "${required}" ]; then
        echo "ERROR: missing ${required}" >&2
        exit 2
    fi
done

mkdir -p "${RUN_ROOT}"
for energy in 218 440; do
    destination="${RUN_ROOT}/${energy}keV"
    macro_dir="${MACRO_ROOT}/${energy}keV"
    macro="${macro_dir}/UniformFovCntStat_${energy}keV_CenterPoint_worker.mac"
    manifest="${macro_dir}/uniform_fov_manifest.json"

    if [ -e "${destination}" ]; then
        echo "ERROR: run directory already exists: ${destination}" >&2
        exit 3
    fi
    if [ ! -f "${macro}" ] || [ ! -f "${manifest}" ]; then
        echo "ERROR: missing ${energy} keV macro or manifest" >&2
        exit 2
    fi

    mkdir "${destination}"
    cp "${CODE_DIR}/build/gamma01" "${destination}/gamma01"
    cp "${CODE_DIR}/CrystalMatrix.txt" "${destination}/CrystalMatrix.txt"
    cp "${macro}" "${destination}/uniform_fov_worker.mac"
    cp "${manifest}" "${destination}/uniform_fov_manifest.json"
    cp "${SCRIPT_DIR}/RunData" "${SCRIPT_DIR}/runevent" "${destination}/"
    chmod u+x "${destination}/gamma01" "${destination}/RunData" \
        "${destination}/runevent"
    echo "Prepared ${destination}"
done

echo "Run roots prepared under ${RUN_ROOT}"
echo "Submit separately with:"
echo "  cd ${RUN_ROOT}/218keV && ./RunData |& tee submit.log"
echo "  cd ${RUN_ROOT}/440keV && ./RunData |& tee submit.log"
