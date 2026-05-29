# JSCC SPECT Polar-Coordinate Reconstruction

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## English

### 1. Overview

This repository implements a SPECT reconstruction framework for cylindrical
detector geometry in polar coordinates. It supports single-photon projection
reconstruction, Compton list-mode reconstruction, and weighted joint
reconstruction (JSCC-SD).

The codebase has grown around one main goal:

- reconstructing images from simulated or precomputed system models for
  cylindrical SPECT / JSCC detector layouts

The repository currently contains:

- the main reconstruction pipeline (local GPU, distributed GPU, distributed CPU)
- sparse Compton operators for reduced memory usage
- MATLAB tools for image generation, visualization, and metric analysis
- several independent auxiliary studies, grouped under `Auxiliary_Studies/`
- a complete reproduction guide under `Reproduction/`

### 2. Reconstruction Modes

| Mode | Meaning | Main Data Source | Description |
| --- | --- | --- | --- |
| `SC` | Self-Collimation | `CntStat` | Single-photon reconstruction from projection data |
| `SCD` | Self-Collimation Downsampled | `CntStat` | Projection-only reconstruction with downsampled statistics |
| `JSCCD` | Joint SC + Compton D | `List` | Compton-only list-mode reconstruction |
| `JSCCSD` | Joint SC + Compton SD | `CntStat` + `List` | Weighted joint reconstruction using single-photon and Compton channels |

In practice, the main modes most often used are `SC`, `JSCCSD`, and sparse
Compton variants.

### 3. Repository Layout

```
├── Factors/                         # System matrices & geometry (per energy/rotation)
├── Geant4Sim/                       # MATLAB scripts for Geant4 phantom generation
├── CntStat/                         # Generated projection data (per energy/phantom/count)
├── List/                            # Generated list-mode event data
├── img_cartesian/                   # Cartesian-space reference images
├── Figure/                          # Local reconstruction output figures
├── Figure_Dist_JSCCSD/              # Distributed JSCCSD reconstruction output
├── Figure_Dist_SC/                  # Distributed SC reconstruction output
├── Auxiliary_Studies/               # Independent research projects
│   ├── ComptonSystemMatrixPrototype/
│   ├── CRCVAR_SinglePhoton/
│   ├── EventOrderInference_Experiment/
│   ├── FreePath/
│   └── Reference/
├── FreePath/                        # Free-path simulation (legacy location)
├── Reproduction/                    # Step-by-step reproduction guide
│   ├── Step1_GenerateFactors/
│   ├── Step2_GenerateCntStat/
│   ├── Step3_Reconstruction/
│   │   ├── SC_Recon/
│   │   └── JSCCSD_Recon/
│   └── Step4_Visualization/
├── distributed/                     # Distributed reconstruction
│   ├── python/                      #   Python entry points & reconstruction cores
│   ├── scripts/                     #   SLURM submission scripts
│   └── scripts_tansuo1000/          #   Scripts for a specific cluster
├── compton_sparse_ops.py            # Sparse Compton projector implementation
├── sparse_main_utils.py             # Shared path/config/data-loading helpers
├── process_list_plane_strict.py     # Full-resolution Compton list processing
├── process_list_plane_sparse.py     # Sparse Compton list processing
├── main_plane.py                    # Local JSCC reconstruction entry
├── main_plane_sparse.py             # Local sparse-Compton reconstruction entry
├── main_local_cntstat.py            # Local single-photon-only entry
├── main_local_sparse_jsccsd_only.py # Local sparse JSCCSD-only entry
├── recon_osem_plane.py              # Core local OSEM implementation
├── recon_osem_plane_sparse.py       # Local sparse OSEM implementation
├── recon_osem_local_cntstat.py      # Local SC-only OSEM implementation
├── recon_osem_local_sparse_jsccsd_only.py  # Local sparse JSCCSD-only OSEM
├── GenProj_SPECT_PolarCoor.m        # MATLAB: generate projection data
├── GenProj_Hoffman_SPECT_PolarCoor.m
├── get_img_SPECT_PolarCoor.m        # MATLAB: polar→Cartesian image conversion
├── get_img_SC_Dist_PolarCoor.m      # MATLAB: SC Dist image conversion
├── get_img_JSCCSD_Dist_PolarCoor.m  # MATLAB: JSCCSD Dist image conversion
├── CNRCRC_SPECT.m                   # MATLAB: CRC/CNR evaluation
├── CNRCRC_SC_Dist.m                 # MATLAB: SC Dist CRC/CNR
├── CNRCRC_JSCCSD_Dist.m             # MATLAB: JSCCSD Dist CRC/CNR
├── PVR_HotRod_SC_Dist.m             # MATLAB: hot-rod peak-valley ratio
├── downsample_list.m                # MATLAB: list data downsampling
├── Analyze_List_ComptonScatterStats.m  # MATLAB: Compton scatter statistics
└── README.md
```

### 4. Key Files Explained

#### Python — Reconstruction Core

| File | Purpose |
| --- | --- |
| `recon_osem_plane.py` | Local OSEM for all 4 modes (SC, SCD, JSCCD, JSCCSD) |
| `recon_osem_plane_sparse.py` | Local OSEM with sparse Compton operators |
| `recon_osem_local_cntstat.py` | Local OSEM for SC-only reconstruction |
| `recon_osem_local_sparse_jsccsd_only.py` | Local OSEM for sparse JSCCSD-only |
| `compton_sparse_ops.py` | `ComptonSparseProjector` class: coarse↔fine grid conversion, sparse event row packing/unpacking, `materialize_sparse_event_rows_to_fine()` |
| `sparse_main_utils.py` | `Tee`, `build_save_path`, `load_list_csv`, `downsample_projection_and_list`, path resolution helpers |

#### Python — List Processing

| File | Purpose |
| --- | --- |
| `process_list_plane_strict.py` | Full-resolution Compton backprojection: computes dense T matrix from list events |
| `process_list_plane_sparse.py` | Sparse Compton backprojection: computes compressed T matrix using coarse grid |

#### Python — Local Entry Points

| File | Purpose |
| --- | --- |
| `main_plane.py` | Local multi-mode reconstruction (SC + SCD + JSCCD + JSCCSD) |
| `main_plane_sparse.py` | Local sparse reconstruction (all modes with sparse Compton) |
| `main_local_cntstat.py` | Local SC-only from projection data |
| `main_local_sparse_jsccsd_only.py` | Local sparse JSCCSD-only (no SC/JSCCD intermediate outputs) |

#### Python — Distributed GPU (NCCL + CUDA)

| File | Purpose |
| --- | --- |
| `distributed/python/main_dist.py` | Distributed multi-mode entry (all 4 modes) |
| `distributed/python/main_dist_sparse.py` | Distributed sparse multi-mode entry |
| `distributed/python/main_dist_sparse_jsccsd_only.py` | Distributed GPU sparse JSCCSD-only entry |
| `distributed/python/main_dist_cntstat.py` | Distributed SC-only entry |
| `distributed/python/main_dist_tstream.py` | Distributed T-matrix streaming variant |
| `distributed/python/main_dist_tonline_cache.py` | Distributed online T-matrix caching variant |
| `distributed/python/main_dist_{count}.py` | Pre-configured entries for specific count levels |
| `distributed/python/recon_osem_dist.py` | Distributed OSEM core (all 4 modes, NCCL) |
| `distributed/python/recon_osem_dist_sparse.py` | Distributed sparse OSEM core |
| `distributed/python/recon_osem_dist_sparse_jsccsd_only.py` | Distributed GPU sparse JSCCSD-only OSEM core |
| `distributed/python/recon_osem_dist_cntstat.py` | Distributed SC-only OSEM core |
| `distributed/python/recon_osem_dist_tstream.py` | Distributed T-streaming OSEM core |
| `distributed/python/recon_osem_dist_tonline_cache.py` | Distributed online-cache OSEM core |
| `distributed/python/t_shard_dist.py` | T-matrix shard management |
| `distributed/python/t_online_cache_dist.py` | T-matrix online cache management |

#### Python — Distributed CPU (GLOO, no GPU required) ★ NEW

| File | Purpose |
| --- | --- |
| `distributed/python/main_dist_sparse_jsccsd_only_cpu.py` | Distributed CPU sparse JSCCSD-only entry (GLOO backend) |
| `distributed/python/recon_osem_dist_sparse_jsccsd_only_cpu.py` | Distributed CPU sparse JSCCSD-only OSEM core |

These two files mirror the GPU versions but:
- Use `backend="gloo"` instead of `nccl`
- `device = torch.device("cpu")` instead of CUDA
- No `torch.cuda` calls
- Controlled via `OMP_NUM_THREADS` / `--num-threads`
- Designed for clusters with large CPU RAM (e.g., 768 GB/node) but limited GPUs

#### Python — Utilities

| File | Purpose |
| --- | --- |
| `distributed/python/_path_setup.py` | Adds the repository root to `sys.path` for imports |
| `distributed/python/gpu_mem_report.py` | GPU memory usage logging |
| `distributed/python/cpu_mem_report.py` | CPU memory usage logging |

#### SLURM Scripts

| Script | Purpose |
| --- | --- |
| `distributed/scripts/jsccrecon_dist.sh` | GPU distributed multi-mode |
| `distributed/scripts/jsccrecon_dist_sparse.sh` | GPU distributed sparse multi-mode |
| `distributed/scripts/jsccrecon_dist_sparse_jsccsd_only.sh` | GPU distributed sparse JSCCSD-only (targeting `gpu_5090` partition) |
| `distributed/scripts/jsccrecon_dist_sparse_jsccsd_only_cpu.sh` | **CPU distributed sparse JSCCSD-only (targeting `amd_m9_768` partition) ★ NEW** |
| `distributed/scripts/cntstatrecon_dist.sh` | GPU distributed SC-only |
| `distributed/scripts/jsccrecon_dist_tstream.sh` | GPU distributed T-streaming |
| `distributed/scripts/jsccrecon_dist_tonline_cache.sh` | GPU distributed online-cache |
| `distributed/scripts/jsccrecon_dist_{count}.sh` | Pre-configured GPU scripts for specific count levels (2e9, 5e9, 1e10, 2e10, 5e10, 1e11, 1e100) |

#### MATLAB — Data Generation

| File | Purpose |
| --- | --- |
| `GenProj_SPECT_PolarCoor.m` | Generate projection data (CntStat) from system matrix |
| `GenProj_Hoffman_SPECT_PolarCoor.m` | Generate Hoffman phantom projections |
| `Geant4Sim/ContrastPhantom_Rotate_3D.m` | Generate contrast phantom Geant4 input |
| `Geant4Sim/GenPhan_HotRodPhantom_Rotate_3D.m` | Generate hot-rod phantom Geant4 input |
| `Geant4Sim/BrainPhantom_HoffmanMontage_3D.m` | Generate Hoffman brain phantom |
| `Geant4Sim/Cylinder_Phantom_Rotate_3D.m` | Generate cylinder phantom |
| `Geant4Sim/point_array_Rotate_3D.m` | Generate point-source array |
| `Geant4Sim/HoffmanCompressed_Rotate_3D.m` | Generate compressed Hoffman phantom |

#### MATLAB — Visualization & Evaluation

| File | Purpose |
| --- | --- |
| `get_img_SPECT_PolarCoor.m` | Polar→Cartesian image conversion |
| `get_img_SC_Dist_PolarCoor.m` | SC distributed result visualization |
| `get_img_JSCCSD_Dist_PolarCoor.m` | JSCCSD distributed result visualization |
| `CNRCRC_SPECT.m` | CRC/CNR curve evaluation |
| `CNRCRC_SC_Dist.m` | SC distributed CRC/CNR |
| `CNRCRC_JSCCSD_Dist.m` | JSCCSD distributed CRC/CNR |
| `PVR_HotRod_SC_Dist.m` | SC hot-rod peak-valley ratio analysis |
| `PVR_HotRod_JSCCSD_Dist.m` | JSCCSD hot-rod peak-valley ratio analysis |
| `Analyze_List_ComptonScatterStats.m` | Compton scatter event statistics |
| `downsample_list.m` | Downsample list-mode data |

### 5. Reconstruction Pipeline Architecture

```
                ┌─────────────────────────────────────────────┐
                │           Factor Files (Factors/)            │
                │  SysMat_polar, Detector.csv, RotMat,        │
                │  Sensi_s, Sensi_d, coor_polar_full.csv      │
                └──────────────┬──────────────────────────────┘
                               │
                ┌──────────────▼──────────────────────────────┐
                │        Data Files (CntStat/ + List/)         │
                │  Projection data (.csv)  +  List data (.csv)│
                └──────┬──────────────────────────┬───────────┘
                       │                          │
          ┌────────────▼──────────┐   ┌───────────▼────────────┐
          │   Single-Photon Path  │   │   Compton List Path    │
          │   sysmat @ img        │   │   process_list_plane   │
          │   → forward projection│   │   → Compton backproj   │
          │   → EM weight_s       │   │   → sparse T matrix    │
          └────────────┬──────────┘   │   → EM weight_c        │
                       │              └───────────┬────────────┘
                       │                          │
                       └──────────┬───────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │   Joint OSEM Iteration     │
                    │   weight = α·w_s+(2-α)·w_c │
                    │   img = img · weight/s_map  │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │   Output: Image_JSCCSD      │
                    │   (+ intermediate saves)    │
                    └────────────────────────────┘
```

### 6. Distributed Execution Paths

#### GPU Distributed (existing)

```
SLURM (gpu_5090) → srun → torchrun (NCCL) → main_dist_sparse_jsccsd_only.py
                                                    │
                                                    ▼
                                          recon_osem_dist_sparse_jsccsd_only.py
                                          (device=cuda, all-reduce via NCCL)
```

#### CPU Distributed ★ NEW

```
SLURM (amd_m9_768) → srun → torchrun (GLOO) → main_dist_sparse_jsccsd_only_cpu.py
                                                     │
                                                     ▼
                                           recon_osem_dist_sparse_jsccsd_only_cpu.py
                                           (device=cpu, all-reduce via GLOO)
```

CPU distributed resource layout (BSCC-M9 example):

```
Node (256 cores, 768 GB RAM)
├── Rank 0:  16 cores (OMP_NUM_THREADS=16), ~48 GB RAM
├── Rank 1:  16 cores, ~48 GB RAM
├── ...
└── Rank 15: 16 cores, ~48 GB RAM
Total: 16 ranks × 16 threads = 256 cores

4 Nodes × 16 ranks = 64 total distributed processes
```

Memory tuning guide:

| Config | procs/node | threads/proc | RAM/proc | When to use |
| --- | --- | --- | --- | --- |
| `ntasks=8, cpus=32` | 8 | 32 | 96 GB | Large pixel_num (500K+), large T matrix |
| `ntasks=16, cpus=16` | 16 | 16 | 48 GB | Default, moderate data |
| `ntasks=32, cpus=8` | 32 | 8 | 24 GB | Small data, debugging |

### 7. Typical Data Layout

```text
Factors/
├── 511keV_RotateNum20/
│   ├── SysMat_polar          # System matrix (pixel_num × total_bins, float32 binary)
│   ├── SysMat_tmp            # System matrix variant (for CRC-VAR study)
│   ├── Detector.csv          # Detector 3D positions
│   ├── RotMat_full.csv       # Rotation mapping: pixel → rotated pixel
│   ├── RotMatInv_full.csv    # Inverse rotation mapping
│   ├── coor_polar_full.csv   # Polar coordinate grid (r, θ, z)
│   ├── Sensi_s               # Single-photon sensitivity map
│   └── Sensi_d               # Compton sensitivity map
├── 140keV_RotateNum20/
├── 662keV/
└── ... (per energy/rotation config)
```

### 8. High-Level Workflow

1. **Generate factors**: Run MATLAB scripts to create system matrices and geometry files
2. **Generate data**: Use MATLAB/Geant4 to create projection and list data
3. **Reconstruct**: Run local or distributed reconstruction
4. **Visualize**: Convert polar results to Cartesian images
5. **Evaluate**: Compute CRC, CNR, PVR metrics

### 9. Auxiliary Studies

- `Auxiliary_Studies/CRCVAR_SinglePhoton`: CRC-Variance analysis for single-photon reconstruction
- `Auxiliary_Studies/EventOrderInference_Experiment`: Event order inference for Compton events
- `Auxiliary_Studies/FreePath`: Free-path simulation studies
- `Auxiliary_Studies/ComptonSystemMatrixPrototype`: Compton system matrix prototyping
- `Auxiliary_Studies/Reference`: Reference documents and figures

### 10. Reproduction

See `Reproduction/README.md` for a step-by-step guide to reproduce the results.

### 11. Common Failure Modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `master_addr is only used for static rdzv_backend` | Usually just a torchrun warning | Look for the real Python traceback |
| `SIGTERM`, `Socket Timeout` | Cascade after one rank fails | Find the first failing rank's error |
| `can't allocate memory`, exit `-9` | OOM (CPU RAM or GPU VRAM) | Reduce data size or add more nodes |
| `cholesky` not positive-definite | FIM + βR not PD in CRC-VAR | Increase β, reduce grid, or use matrix-free |

### 12. Practical Notes

- For sparse or large-scale Compton workflows, dense direct methods may become impractical
- When investigating distributed failures, the first Python traceback is the most informative
- The CPU distributed path uses the same math as GPU; results should be identical within float32 precision

---

<a id="中文"></a>

## 中文

### 1. 项目概述

本仓库实现了一套面向圆柱面探测器几何的 SPECT 极坐标重建框架，支持：

- **SC**：单光子投影重建
- **SCD**：降采样投影重建
- **JSCCD**：康普顿 List 模式重建
- **JSCCSD**：单光子与康普顿联合重建（主要使用模式）

此外还包含 CRC-VAR、事件顺序推断、自由程模拟等辅助研究代码。

### 2. 项目目录结构

```
├── Factors/                    # 系统矩阵、几何文件（按能量/旋转数组织）
├── Geant4Sim/                  # MATLAB 脚本：生成 Geant4 体模输入
├── CntStat/                    # 生成的投影数据（按能量/体模/计数水平组织）
├── List/                       # 生成的 List 模式事件数据
├── img_cartesian/              # 笛卡尔坐标参考图像
├── Figure/                     # 本地重建输出
├── Figure_Dist_JSCCSD/         # 分布式 JSCCSD 重建输出
├── Figure_Dist_SC/             # 分布式 SC 重建输出
├── Auxiliary_Studies/          # 辅助研究（与主重建独立）
├── FreePath/                   # 自由程模拟（历史位置）
├── Reproduction/               # 完整复现指南
├── distributed/                # 分布式重建
│   ├── python/                 #   Python 入口与重建核心
│   ├── scripts/                #   SLURM 提交脚本
│   └── scripts_tansuo1000/     #   特定集群脚本
├── compton_sparse_ops.py       # 稀疏 Compton 投影器实现
├── sparse_main_utils.py        # 共享工具（路径、数据加载等）
├── process_list_plane_*.py     # List 事件处理
├── main_*.py                   # 各模式的本地重建入口
├── recon_osem_*.py             # 各模式的 OSEM 重建核心
├── *.m                         # MATLAB 数据生成/可视化/评价脚本
└── README.md
```

### 3. 核心文件说明

#### 重建核心

| 文件 | 功能 |
| --- | --- |
| `recon_osem_plane.py` | 本地完整 OSEM（4 种模式） |
| `recon_osem_plane_sparse.py` | 本地稀疏 Compton OSEM |
| `recon_osem_local_cntstat.py` | 本地 SC-only OSEM |
| `recon_osem_local_sparse_jsccsd_only.py` | 本地稀疏 JSCCSD-only OSEM |
| `compton_sparse_ops.py` | 稀疏投影器：粗细网格转换、事件行打包解包、稀疏展开 |

#### List 事件处理

| 文件 | 功能 |
| --- | --- |
| `process_list_plane_strict.py` | 全分辨率 Compton 反投影 |
| `process_list_plane_sparse.py` | 稀疏 Compton 反投影（使用粗网格压缩） |

#### 分布式 GPU（NCCL + CUDA）

| 文件 | 功能 |
| --- | --- |
| `distributed/python/main_dist_sparse_jsccsd_only.py` | GPU 分布式稀疏 JSCCSD-only 入口 |
| `distributed/python/recon_osem_dist_sparse_jsccsd_only.py` | GPU 分布式稀疏 JSCCSD-only OSEM 核心 |

#### 分布式 CPU（GLOO，无需 GPU）★ 新增

| 文件 | 功能 |
| --- | --- |
| `distributed/python/main_dist_sparse_jsccsd_only_cpu.py` | CPU 分布式稀疏 JSCCSD-only 入口 |
| `distributed/python/recon_osem_dist_sparse_jsccsd_only_cpu.py` | CPU 分布式稀疏 JSCCSD-only OSEM 核心 |

这两个文件与 GPU 版本的**计算逻辑完全一致**，区别仅在于：
- `backend="gloo"` 替代 NCCL
- `device=torch.device("cpu")` 替代 CUDA
- 通过 `OMP_NUM_THREADS` 控制 OpenMP 并行度
- 适用于内存大（768 GB/节点）但 GPU 有限的集群

#### SLURM 提交脚本

| 脚本 | 目标分区 | 功能 |
| --- | --- | --- |
| `jsccrecon_dist_sparse_jsccsd_only.sh` | `gpu_5090` | GPU 分布式稀疏 JSCCSD-only |
| `jsccrecon_dist_sparse_jsccsd_only_cpu.sh` ★ | `amd_m9_768` | **CPU 分布式稀疏 JSCCSD-only** |
| `jsccrecon_dist_sparse.sh` | `gpu_5090` | GPU 分布式稀疏多模式 |
| `cntstatrecon_dist.sh` | `gpu_5090` | GPU 分布式 SC-only |
| `jsccrecon_dist_{count}.sh` | `gpu_5090` | 预配置的 GPU 脚本（各计数水平） |

#### MATLAB 脚本

| 文件 | 功能 |
| --- | --- |
| `GenProj_SPECT_PolarCoor.m` | 从系统矩阵生成投影数据 |
| `get_img_SPECT_PolarCoor.m` | 极坐标→笛卡尔图像转换 |
| `CNRCRC_SPECT.m` | CRC/CNR 曲线计算 |
| `PVR_HotRod_SC_Dist.m` | Hot-rod 峰谷比分析 |
| `downsample_list.m` | List 数据降采样 |
| `Analyze_List_ComptonScatterStats.m` | Compton 散射统计 |

### 4. 重建流水线架构

```
Factor 文件 (SysMat, RotMat, Sensi, ...)
         │
         ▼
数据文件 (CntStat/ 投影 + List/ 事件)
    ┌────┴─────┐
    ▼          ▼
单光子路径   康普顿路径
sysmat@img   process_list_plane
→ weight_s   → T矩阵(稀疏)
    └────┬─────┘
         ▼
   联合 OSEM 迭代
   weight = α·w_s + (2-α)·w_c
   img = img · weight / s_map
         │
         ▼
   输出: Image_JSCCSD
```

### 5. 分布式执行路径

#### GPU 分布式（已有）

```
SLURM (gpu_5090) → srun → torchrun (NCCL) → main_dist_sparse_jsccsd_only.py
```

- 每个 rank 对应一个 GPU
- 通过 NCCL 进行 all-reduce 通信
- 适合 GPU 显存充足（≥48 GB）的场景

#### CPU 分布式 ★ 新增

```
SLURM (amd_m9_768) → srun → torchrun (GLOO) → main_dist_sparse_jsccsd_only_cpu.py
```

- 每 256 核节点启动 16 个 rank，每个 rank 16 个 OpenMP 线程
- 通过 GLOO (TCP over IB) 进行 all-reduce 通信
- 每节点 768 GB 内存，每 rank ~48 GB
- 适合数据规模大（pixel_num 达数十万、T 矩阵达 TB 级）、GPU 显存不足的场景

资源布局示例：

```
SLURM 参数:
  -N 4 --ntasks-per-node=16 --cpus-per-task=16 --exclusive

节点 (256核, 768GB):
  ├── Rank 0:  16核, ~48GB
  ├── Rank 1:  16核, ~48GB
  ├── ...
  └── Rank 15: 16核, ~48GB

总计: 4节点 × 16 ranks = 64 分布式进程
```

调整建议：

| 配置 | 进程/节点 | 线程/进程 | 内存/进程 | 适用场景 |
| --- | --- | --- | --- | --- |
| `ntasks=8, cpus=32` | 8 | 32 | 96 GB | 大 pixel_num (500K+), 大 T 矩阵 |
| `ntasks=16, cpus=16` | 16 | 16 | 48 GB | 默认配置，中等数据规模 |
| `ntasks=32, cpus=8` | 32 | 8 | 24 GB | 小数据，调试 |

### 6. 因子文件说明

以 `Factors/511keV_RotateNum20/` 为例：

| 文件 | 说明 |
| --- | --- |
| `SysMat_polar` | 系统矩阵 (pixel_num × total_bins, float32 二进制) |
| `Detector.csv` | 探测器 3D 坐标 |
| `RotMat_full.csv` | 旋转映射：像素→旋转后像素索引 |
| `RotMatInv_full.csv` | 逆旋转映射 |
| `coor_polar_full.csv` | 极坐标网格 (r, θ, z) |
| `Sensi_s` | 单光子灵敏度图 |
| `Sensi_d` | 康普顿灵敏度图 |
| `SysMat_tmp` | 系统矩阵变体（CRC-VAR 研究使用） |

### 7. 典型工作流程

1. **生成因子**：用 MATLAB 脚本创建系统矩阵和几何文件
2. **生成数据**：用 MATLAB/Geant4 创建投影和 List 数据
3. **重建**：运行本地或分布式重建
4. **可视化**：极坐标结果转笛卡尔图像
5. **评价**：计算 CRC、CNR、PVR 等指标
6. **辅助研究**：按需运行 CRC-VAR、事件顺序推断等

详见 `Reproduction/README.md`。

### 8. 常见报错与排查

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `master_addr is only used for static rdzv_backend` | torchrun 警告 | 找后面的 Python traceback |
| `SIGTERM`, `Socket Timeout` | 一个 rank 失败后连带退出 | 找第一个失败的 rank |
| `can't allocate memory`, 退出码 `-9` | 内存不足 | 减少数据量或增加节点 |
| `cholesky` 非正定 | CRC-VAR 中 β 太小 | 增大 β、减小网格、改用 matrix-free |

### 9. 实用建议

- 排查分布式错误时，优先看第一条 Python 异常
- CPU 分布式与 GPU 版本数学逻辑完全一致，float32 精度范围内结果相同
- 大规模数据时优先考虑 CPU 分布式，内存远大于 GPU 显存