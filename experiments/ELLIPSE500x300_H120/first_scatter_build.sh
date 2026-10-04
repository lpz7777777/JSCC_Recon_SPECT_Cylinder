#!/usr/bin/env bash
#SBATCH --job-name=FIRST_SCATTER_V2_BUILD
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
set -eo pipefail
source /etc/profile.d/modules.sh
module load compilers/gcc/v12.2.0
export Geant4_DIR=/apps/soft/geant/geant4-v11.1.0/lib64/cmake/Geant4
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
set -u
cd "${SLURM_SUBMIT_DIR:?}"
g++ -std=c++17 -Wall -Wextra -Werror -Isource/include source/tests/test_first_scatter_contract.cc -o contract_tests
./contract_tests > logs/contract_tests.txt
/apps/tools/cmake/v3.25.2/bin/cmake -S source -B build -DWITH_GEANT4_UIVIS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/apps/compilers/gcc/v12.2.0/bin/gcc -DCMAKE_CXX_COMPILER=/apps/compilers/gcc/v12.2.0/bin/g++
/apps/tools/cmake/v3.25.2/bin/cmake --build build --parallel 2
python3 first_scatter_workflow.py smoke --base "$PWD"
