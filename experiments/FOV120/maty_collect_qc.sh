#!/usr/bin/env bash
# Submit only with afterok dependencies on BOTH matching phantom arrays.
#SBATCH --job-name=FOV120_QC_collect
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=experiments/FOV120/generated/qc_collect.%j.out
#SBATCH --error=experiments/FOV120/generated/qc_collect.%j.err
set -euo pipefail
level=${FOV120_COUNT_LEVEL:-1e9}
[[ "$level" == 1e9 || "$level" == 1e10 ]] || exit 2
for dataset in Uniform Contrast; do
  python3 experiments/FOV120/workflow.py collect --dataset "$dataset" --level "$level"
done
echo "FOV120_UNIFORM_CONTRAST_${level}_COLLECTED"
