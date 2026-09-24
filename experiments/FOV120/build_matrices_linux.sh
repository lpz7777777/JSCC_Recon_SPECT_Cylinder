#!/usr/bin/env bash
# Isolated build. CUDA_ARCH=sm_86 for the available RTX A6000; override on other hosts.
set -euo pipefail
cd "$(dirname "$0")/../.."
engine=Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main
nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
arch=${CUDA_ARCH:-sm_86}
"$nvcc" -std=c++17 -O3 -lineinfo -arch="$arch" \
  "$engine/PEGen_RayTracing_CircularHole/PEGen_V4_Production.cu" \
  -o "$engine/PEGen_RayTracing_CircularHole/PEGen_V4_Production"
"$nvcc" -std=c++17 -O3 -lineinfo -arch="$arch" \
  "$engine/ScatterGen_RayTracing_CircularHole/scatter.cu" \
  "$engine/ScatterGen_RayTracing_CircularHole/ScatterGen_CircularHole.cpp" \
  -o "$engine/ScatterGen_RayTracing_CircularHole/ScatterGen_CircularHole_detector_local"
