#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load compilers/gcc/v12.2.0
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JSCC_RELEASE=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928/experiments/ELLIPSE500x300_H120/generated/jscc_geant4_5e10_10000/transport_releases/dc23f1c14e3e036c JSCC_SIMULATION=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928/experiments/ELLIPSE500x300_H120/generated/jscc_geant4_5e10_10000/source_registry JSCC_OUTPUT=/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928/experiments/ELLIPSE500x300_H120/generated/jscc_geant4_5e10_10000/transport JSCC_LIMIT=32348
srun --chdir=/tmp --kill-on-bad-exit=1 --wait=0 bash -c '
 index=${SLURM_PROCID:-0}
 if [[ "transport" == transport ]]; then printf -v suffix "worker_%04d" "$index"; output="$JSCC_OUTPUT/$suffix"; else output="$JSCC_OUTPUT"; fi
 exec python3 "$JSCC_RELEASE/jscc_5e10_transport.py" worker --release "$JSCC_RELEASE" --simulation "$JSCC_SIMULATION" --output "$output" --index "$index" --limit "$JSCC_LIMIT"
'
