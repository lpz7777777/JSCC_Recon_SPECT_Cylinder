#!/usr/bin/env bash
# Submit only selected indices after smoke validation; resource settings belong to your cluster.
# Example: sbatch --array=0-9 --cpus-per-task=1 geant4_array.sh
set -euo pipefail
: "${JSCC_REPO_ROOT:?Set repository root}"
: "${JSCC_G4_EXECUTABLE:?Set rebuilt gamma01 executable}"
: "${JSCC_CRYSTAL_MATRIX:?Set matching CrystalMatrix.txt}"
cd "$JSCC_REPO_ROOT"
python experiments/FOV120/workflow.py run \
  --manifest "${FOV120_JOB_MANIFEST:-experiments/FOV120/generated/Simulation/jobs.json}" \
  --index "${SLURM_ARRAY_TASK_ID:?Use a SLURM array}" \
  --executable "$JSCC_G4_EXECUTABLE" --crystal "$JSCC_CRYSTAL_MATRIX"
