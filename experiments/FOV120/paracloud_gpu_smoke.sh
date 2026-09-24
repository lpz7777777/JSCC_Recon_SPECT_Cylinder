#!/usr/bin/env bash
#SBATCH --job-name=FOV120_NCCL_smoke
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --qos=gpugpu
#SBATCH --time=00:10:00
#SBATCH --output=experiments/FOV120/generated/nccl_smoke.%j.out
#SBATCH --error=experiments/FOV120/generated/nccl_smoke.%j.err
set -eo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate torch
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export JSCC_TEST_BACKEND=nccl
nvidia-smi
python -c 'import torch; print("torch",torch.__version__,"CUDA",torch.version.cuda,"NCCL",torch.cuda.nccl.version()); print([(torch.cuda.get_device_name(i),torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]); assert torch.cuda.device_count()==2'
torchrun --standalone --nproc_per_node=2 distributed/dual_energy_compton_python/validate_synthetic_distributed.py
