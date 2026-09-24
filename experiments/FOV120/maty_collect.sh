#!/usr/bin/env bash
# Submit with afterok dependency on the extension array.
#SBATCH --job-name=FOV120_collect
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=experiments/FOV120/generated/collect.%j.out
#SBATCH --error=experiments/FOV120/generated/collect.%j.err
set -euo pipefail
for dataset in calibration_218 calibration_440 sensitivity_440 sensitivity_validation_440; do
    python3 experiments/FOV120/workflow.py collect --dataset "$dataset" --level all
done
echo FOV120_ALL_FOUR_GROUPS_COLLECTED
