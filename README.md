# JSCC SPECT Polar-Coordinate Reconstruction

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## English

> **Current production state (2026-07-20):** the underlying matrix `A` has
> point-source columns per emitted monoenergetic photon. The canonical no-suffix
> JSCC Factors store the density-basis matrix `B=A*diag(DeltaV_mm3)`, so they
> map gamma-photon activity density rather than equal per-sample source weight.
> A physical uniform Geant4 source must therefore be a volume-uniform cylinder,
> not an equal-weight polar point array. Start with `docs/DEVELOPMENT_HANDOFF.md`
> for current Factors, Geant4 semantics, verified findings, and next tasks.

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
├── Geant4Sim/                       # Geant4 source macros plus JSCC and EHE detector simulations
├── CntStat/                         # Generated projection data (per energy/phantom/count)
├── List/                            # Generated list-mode event data
├── GenProj/                         # MATLAB system-matrix forward projection scripts
├── Results/                         # Local generated results (ignored by Git)
│   ├── Reconstruction/              # Current JSCC/EHE/Geant4 reconstructions
│   ├── Analysis/                    # CNR/CRC and calibration comparisons
│   ├── Logs/                        # Archived local run logs
│   └── Legacy/                      # Historical Figure* outputs
├── Auxiliary_Studies/               # Independent research projects
│   ├── ComptonSystemMatrixPrototype/
│   ├── CRCVAR_SinglePhoton/
│   ├── EventOrderInference_Experiment/
│   ├── FreePath/
│   ├── Sensitivity_SPECT_PolarCoor/
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
| `main_local_multi_energy_cntstat.py` | Local 218/440 CntStat-only reconstruction with explicit 440-to-218 cross-talk correction |
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
| `GenProj/GenProj_SPECT_PolarCoor.m` | Generate projection data (CntStat) from system matrix |
| `GenProj/GenProj_Hoffman_SPECT_PolarCoor.m` | Generate Hoffman phantom projections |
| `GenProj/GenProj_ContrastPhantom_DualEnergy_PolarCoor.m` | Generate 218/440 keV dual-energy contrast phantom CntStat from per-energy system matrices |
| `Geant4Sim/ContrastPhantom_Rotate_3D.m` | Generate contrast phantom Geant4 input |
| `Geant4Sim/ContrastPhantom_DualEnergy_Rotate_3D.m` | Generate 225Ac dual-energy contrast phantom Geant4 input and preview |
| `Geant4Sim/Geant4Code_EHE/` | EHE parallel-hole Pb/NaI CntStat-only Geant4 simulation; no Compton classification or List output |
| `Geant4Sim/Geant4Data_ContrastPhantom_DualEnergy_225Ac_manifest.json` | Paths and validation statistics for installed Geant4 JSCC `1e9`/`1e10` data |
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
| `get_img_SC_MultiOutput_PolarCoor.m` | Manifest-driven comparison of local per-energy and joint SC results |
| `compare_CNRCRC_JSCC_EHE_MultiOutput.m` | Compare JSCC/EHE corrected-218 and direct-440 CRC/CNR curves across count levels |
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

### 6. Multi-Energy Reconstruction

Simultaneous multi-energy reconstruction is a **first-class capability** of this
framework — single-energy is simply the special case where `--e0-list` has one
entry. Every entry-point accepts `--e0-list` (one or more energies in MeV),
together with the parallel lists `--ene-threshold-sum-list` and
`--intensity-list`. All energy channels reconstruct the **same shared image**.

#### How it works

1. **Per-energy loading.** Each energy reads its own factor directory
   (`Factors/<1000*e0>keV_RotateNum<R>/`) and builds parallel per-energy lists:
   `sysmat_all`, `proj_all`, `t_local_all`, `sparse_projector_all`,
   `sensi_s_all`, `sensi_d_all`.

2. **Sensitivity maps are summed.** Per-energy sensitivity maps are summed into
   a single map before iteration:
   ```python
   s_map_arg.s = sum(sensi_s_all)
   s_map_arg.d = sum(sensi_d_all) * s_map_d_ratio
   s_map_arg.j = alpha * s + (2 - alpha) * d
   ```

3. **OSEM inner loop zips the energy dimension.** Inside each iteration, for
   every rotation angle the code zips over all energies and accumulates each
   energy's contribution into one `weight_local`, then divides by the merged
   `s_map`:
   ```python
   for rotate_idx in range(rotate_num):
       for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_l, proj_l, rotmat_all, rotmat_inv_all):
           w_s = alpha * get_weight_single(sysmat, proj[:, rotate_idx], img_rotate)
           weight_local += index_select(w_s, rotmat_inv[:, rotate_idx] - 1)
   img = img * weight_local / s_map
   ```
   The Compton branch does the same via
   `zip(..., sysmat_full_all, sparse_projector_all)`.

   Mathematically the update becomes:
   ```
   img = img · Σ_e [α·w_s^(e) + (2-α)·w_c^(e)] / Σ_e [α·s^(e) + (2-α)·d^(e)]
   ```
   `--intensity-list` weights the relative contribution of each energy.

#### Per-mode support

| Mode | Files | Multi-energy | Note |
| --- | --- | --- | --- |
| Local SC-only | `main_local_cntstat.py` | ✅ Yes | `--e0-list`; `sum(sensi_s_all)` |
| Local full (dense) | `main_plane.py` | ✅ Yes | Energy list hard-coded in Python (no argparse), but full multi-energy plumbing + `MultiEnergy_` output path |
| Local sparse multi-mode | `main_plane_sparse.py` | ✅ Yes | `--e0-list` |
| Local sparse JSCCSD-only | `main_local_sparse_jsccsd_only.py` | ✅ Yes | `--e0-list` |
| Dist sparse multi-mode (GPU) | `main_dist_sparse.py` | ✅ Yes | `--e0-list` + NCCL all_reduce |
| Dist sparse JSCCSD-only (GPU) | `main_dist_sparse_jsccsd_only.py` | ✅ Yes | `--e0-list` + NCCL all_reduce |
| Dist SC-only (GPU) | `main_dist_cntstat.py` | ✅ Yes | `--e0-list` + all_reduce |
| Dist sparse JSCCSD-only (CPU) | `main_dist_sparse_jsccsd_only_cpu.py` | ✅ Yes | `--e0-list` + GLOO all_reduce |
| **Dist dense multi-mode (GPU)** | `main_dist.py` | ⚠️ Partial | Reconstruction core supports it (`ene_num`, `sum(sensi_s_all)`, all_reduce), but `e0_list` is **hard-coded** to `[0.511]` and no `--e0-list` argument is exposed |

The only gap is `main_dist.py` (`distributed/python/main_dist.py:49-51`); its
core `recon_osem_dist.py` is already multi-energy-ready. Exposing the energy
list via argparse there would enable it with no algorithm change.

#### Usage

Two-energy simultaneous reconstruction (e.g. 511 keV + 140 keV):

```bash
python main_local_sparse_jsccsd_only.py \
  --e0-list 0.511 0.140 \
  --ene-threshold-sum-list 0.46 0.13 \
  --intensity-list 1.0 1.0 \
  --data-file-name ContrastPhantom_240_30 \
  --count-level 1e9 \
  --jsccsd-iter 5000 --save-iter-step 50
```

Requirements:

- A factor directory per energy under `Factors/` (e.g. `511keV_RotateNum20/`
  and `140keV_RotateNum20/`), each with `SysMat_polar`, `Sensi_s`, `Sensi_d`,
  `RotMat_full.csv`, etc.
- Matching projection and list data per energy under `CntStat/` and `List/`.
- `--e0-list`, `--ene-threshold-sum-list`, `--intensity-list` must all have the
  same length (validated at startup).

The output directory is automatically prefixed with `ME_` (Multi-Energy) and
tagged with all energies, e.g.
`ME_RotNum20_ContrastPhantom_240_30_(511_140)keV_...`.

#### Local CntStat-only cross-talk-aware pipeline

`main_local_multi_energy_cntstat.py` is the local validation entry point for
218/440 keV data. It loads three independently normalized responses:

```text
A218       = A(218-window <- 218-source)
A440       = A(440-window <- 440-source)
C440to218  = C(218-window <- 440-source)
```

For each requested count level it writes:

```text
Image_S_440keV
Image_S_218keV_Contaminated
Image_S_218keV_CrossTalkCorrected
Image_S_(440_218)keV_CrossTalkCorrected
```

The observation model is:

```text
y440 ~ Poisson(A440*x440)
y218 ~ Poisson(A218*x218 + C440to218*x440)
```

The code first reconstructs `x440` from the clean 440 window. It then forward
projects that image through `C440to218` and holds the resulting measured
per-view CntStat fixed as an additive background while reconstructing `x218`.
The background enters the Poisson denominator but not the A218 sensitivity.
This avoids hard subtraction of noisy projections and keeps the correction in
the reconstruction likelihood. `Image_S_218keV_Contaminated` is reconstructed
with A218 alone to expose the uncorrected bias.

The corrected combined result is a pixel-wise post-reconstruction sum:

```text
x_combined = x440_recon + x218_cross_talk_corrected
```

It does not run another OSEM/MLEM reconstruction and does not assume that the
218 and 440 keV activity distributions are identical. This path does not read
Compton `List` data.
Keep `--intensity-list 1.0 1.0` when the input CntStat was generated by the
dual-energy GenProj script, because that source model already includes the
225Ac gamma yields. OSEM uses an exact sensitivity map for each detector-bin
subset; `--osem-subset-num 1` selects MLEM and reuses the full sensitivity map.

```bash
python main_local_multi_energy_cntstat.py \
  --e0-list 0.218 0.440 \
  --intensity-list 1.0 1.0 \
  --data-file-name ContrastPhantom_DualEnergy_10_30_240_30_225Ac \
  --count-levels 1e9 1e10 1e11 \
  --rotate-num 20 --pixel-num-layer 1280 --pixel-num-z 20 \
  --osem-subset-num 1 \
  --single-sc-iter 20000 --single-sc-save-step 50 \
  --device cuda
```

`GenProj/GenProj_ContrastPhantom_DualEnergy_PolarCoor.m` generates all three
default count levels (`1e9`, `1e10`, and `1e11`) in one run using
`y218=A218*x218+C440to218*x440` and `y440=A440*x440`. Each reconstruction
directory also contains `run_manifest.json`, the predicted 440-to-218 CntStat,
residual diagnostics, task definitions, raw binary image shapes, and model
scope.

Runs are resumable by default. Before each task, the entry point validates the
final float32 image and its iteration history (expected byte sizes, finite and
non-negative values, and equality between the final history frame and final
image). A valid task is skipped; a missing or incomplete task is recomputed from
iteration zero. Pass `--overwrite-existing` only when every requested task must
be regenerated.

Display one run from MATLAB with either the run directory or its `Polar`
subdirectory:

```matlab
get_img_SC_MultiOutput_PolarCoor()
get_img_SC_MultiOutput_PolarCoor("Results/Reconstruction/Figure_Local_SC_MultiOutput/<run-folder>")
```

The reader uses `run_manifest.json` rather than parsing the directory name. It
precomputes the polar-to-Cartesian interpolation plan and writes
`Display/mip_comparison.{png,fig}` and
`Display/final_orthogonal.{png,fig}`.

The current dual-energy response inventory is complete for both detector
systems: JSCC has `218keV_RotateNum20`, `440keV_RotateNum20`, and
`440keV_to218win_RotateNum20`; EHE has the same three responses with the
`_SPECTEHENaI` suffix. Direct and cross-window responses share their geometry
within each detector system. JSCC uses 10496 detector bins and EHE uses 2312.

Current contrast-phantom reconstruction campaigns use MLEM (`subset=1`). The
retained/generated JSCC schedules are `1e9:5000`, `1e10:10000`, and
`1e11:20000` iterations with a 50-iteration save interval. EHE uses 100
iterations at all three count levels and now saves every iteration. The
MATLAB entry point `compare_CNRCRC_JSCC_EHE_MultiOutput.m` reads those
count-dependent runs and evaluates the energy-specific hot rods: 218 keV uses
diameters 10/18/26 mm and 440 keV uses 14/22/30 mm. It writes common-scale
CRC (`0..1`) and CNR (`0..15`) plots plus per-iteration CSV/MAT summaries under
`Results/Analysis/CNRCRC_JSCC_vs_EHE/`.

Two Geant4 JSCC datasets (`1e9` and `1e10`) are installed separately from the
GenProj data:

```text
CntStat/218keV_RotateNum20_Geant4JSCC/
CntStat/440keV_RotateNum20_Geant4JSCC/
List/218-440keV_RotateNum20_Geant4JSCC/
```

Use `--cntstat-dir-suffix _Geant4JSCC` with the local CntStat-only entry point.
Do not set `--factor-dir-suffix`: these measurements still use the standard
JSCC Factors. The List files contain mixed 218/440-source events and have no
primary-energy column, so they must not be presented to a per-energy Compton
pipeline as a pure 218 or pure 440 List. Full paths, counts, schema checks, and
the current no-object-material limitation are recorded in
`Geant4Sim/Geant4Data_ContrastPhantom_DualEnergy_225Ac_manifest.json`.
The installed `1e9` and `1e10` files were replaced on 2026-07-16 with output
from the corrected `EventAction`: it scans every crystal for each energy
window and classifies Compton List events independently, so one event may
contribute multiple CntStat bins and may also be present in List. The four
CntStat matrices and all 40 List view files passed the schema and detector-ID
checks recorded in the manifest. They supersede the earlier legacy datasets.

#### Geant4 225Ac dual-gamma proxy

`Geant4Sim/ContrastPhantom_DualEnergy_Rotate_3D.m` generates 20 GPS macros for a
gamma-only proxy of 225Ac imaging. It does **not** instantiate a radioactive
225Ac ion. Instead, the emitted source densities are

```text
q218(r) = Y218 * x_Fr(r) / sum(x_Fr),  Y218 = 0.114
q440(r) = Y440 * x_Bi(r) / sum(x_Bi),  Y440 = 0.261
```

The Fr and Bi distributions are intentionally different: both energies have a
uniform background, while rods 1/3/5 are hot in the Fr/218 map and rods 2/4/6
are hot in the Bi/440 map. This models daughter redistribution after alpha
decay. Each selected rod is six times its own energy's background. Because
`x_Fr` and `x_Bi` have different spatial integrals, each map is normalized
separately before applying the nuclear yield. The whole-run expected source
fractions are therefore exactly 30.4% at 218 keV and 69.6% at 440 keV. GPS
sampling introduces only the expected finite-count statistical fluctuation.

The 20 views share `1e9` primary photons, normally `5e7` per macro. GPS selects
one source per event, so `/run/beamOn` counts selected 218/440 primary photons,
not parent 225Ac decays. `EventAction` applies 13% FWHM at 511 keV with
`1/sqrt(E)` scaling and writes separate `CntStat_218.csv` and
`CntStat_440.csv`; 440-keV photons scattered into the 218-keV window naturally
contribute to the former. The accepted two-crystal events are written to
`List.csv` using dynamic storage.

Current geometry limitation: the GPS cylinders define source positions only.
The water/PMMA contrast-phantom volume in `DetectorConstruction.cc` is disabled,
and the world is effectively vacuum. Object attenuation and object scatter are
therefore not yet modeled; detector and shielding interactions are modeled.

### 7. Multi-Energy Multi-Output Pipeline

A dedicated distributed pipeline that loads all energies once and produces **all
five image types** in a single run, each with its own iteration count, and with
selective per-energy Compton exclusion.

| Type | Mode | Energies | Output |
| --- | --- | --- | --- |
| 1 | single-photon only (`S`) | per energy | one image per energy |
| 2 | Compton only (`D`) | per whitelisted energy | one image per whitelisted energy |
| 3 | single-photon only (`S`) | all energies | one joint image |
| 4 | Compton only (`D`) | all whitelisted energies | one joint image (auto-deduped vs. Type 2 if redundant) |
| 5 | joint (`J`) | S over all + D over all whitelisted | one joint image |

The Compton whitelist `--compton-energies` declares which energies have usable
Compton (List) data. Energies **not** in the whitelist never have their List
data loaded, T-matrix built, or `sysmat_full` replicated on GPU — so Types 2/4/5
simply do not include them. Type 4 is automatically skipped when it would be
identical to a Type 2 task (same mode, same single-energy Compton subset).

#### Files

| File | Purpose |
| --- | --- |
| `distributed/python/multi_energy_tasks.py` | Pure-logic task model: 5 types → de-duplicated `ReconTask` list + naming |
| `distributed/python/recon_osem_dist_multi_energy.py` | Reconstruction core: 3 OSEM modes (`osem_single_dist` / `osem_compton_dist` / `osem_joint_dist`) + per-task driver |
| `distributed/python/main_dist_multi_energy.py` | Entry point: argparse, data loading (single-photon for all, Compton for whitelist only), task scheduling |
| `distributed/scripts/jsccrecon_dist_multi_energy.sh` | SLURM submission script (`gpu_5090`, 4 nodes × 8 GPUs) |
| `distributed/FRBI_COUPLED_RECON_DESIGN.md` | Design note for future 225Ac Fr/Bi two-image coupled reconstruction with 218-window 440-keV cross-talk |

The three OSEM mode functions share the exact math of `recon_osem_dist_sparse_jsccsd_only.py`;
only the branching (S-only / D-only / joint) and per-task scheduling are new.

Current multi-energy outputs are still **single-image** reconstructions: each task
updates one shared image from the selected SPECT/Compton channels. For 225Ac cases
where 221Fr/218 keV and 213Bi/440 keV are not spatially identical, use the planned
two-image coupled model documented in `distributed/FRBI_COUPLED_RECON_DESIGN.md`
instead of interpreting `A_218win<-218 + (Y440/Y218)*A_218win<-440` as a pure Fr
response.

#### Usage

```bash
# 440 keV + 218 keV; 218 keV has NO Compton data (whitelist = 440 only)
sbatch distributed/scripts/jsccrecon_dist_multi_energy.sh \
  --e0-list 0.440 0.218 \
  --ene-threshold-sum-list 0.40 0.18 \
  --intensity-list 1.0 1.0 \
  --compton-energies 0.440 \
  --data-file-name ContrastPhantom_240_30 --count-level 1e9 \
  --single-sc-iter 1000 --single-sc-save-step 50 \
  --single-compton-iter 2000 --single-compton-save-step 50 \
  --joint-sc-iter 2000 --joint-sc-save-step 50 \
  --joint-compton-iter 2000 --joint-compton-save-step 50 \
  --joint-iter 5000 --joint-save-step 50
```

Each type's iteration count is set independently (0 disables that type). This
produces 5 images (Type 4 deduplicated against Type 2's 440-only task):

```
Image_S_440keV               Image_S_218keV             # Type 1
Image_D_440keV                                          # Type 2
Image_S_(440_218)keV                                    # Type 3
Image_J_S(440_218)keV_D440keV                           # Type 5
```

#### Per-type CLI flags

| Flag | Controls |
| --- | --- |
| `--compton-energies` | Compton whitelist (default = all of `--e0-list`) |
| `--single-sc-iter` / `--single-sc-save-step` | Type 1 (per-energy single-photon) |
| `--single-compton-iter` / `--single-compton-save-step` | Type 2 (per-energy Compton) |
| `--joint-sc-iter` / `--joint-sc-save-step` | Type 3 (all-energies single-photon) |
| `--joint-compton-iter` / `--joint-compton-save-step` | Type 4 (all-energies Compton) |
| `--joint-iter` / `--joint-save-step` | Type 5 (all-energies joint) |

### 8. Distributed Execution Paths

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

### 9. Typical Data Layout

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

### 10. High-Level Workflow

1. **Generate factors**: Run MATLAB scripts to create system matrices and geometry files
2. **Generate data**: Use MATLAB/Geant4 to create projection and list data
3. **Reconstruct**: Run local or distributed reconstruction
4. **Visualize**: Convert polar results to Cartesian images
5. **Evaluate**: Compute CRC, CNR, PVR metrics

### 11. Auxiliary Studies

- `Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main`: GPU-based analytical system-matrix engine (photoelectric + single-Compton-scatter). This fork adds Vacuum collimator support, chunked crystal-scatter computation, and degenerate-geometry NaN fixes. Contains three sub-toolkits:
  - `FileGenerater_3D_Unified/` — generates `Params_*.dat` for JSCC (32×64×4) and ConventionalSPECT geometries, multi-energy, with NIST/XCOM material database and 3D visualization
  - `GenFactors/` — converts the engine's Cartesian `.sysmat` to polar-coordinate `Factors/<E>keV_RotateNum<N>/` (scintillator filtering + Cartesian→polar regrid + rotation matrices)
  - `runs/` — computed PE / Scatter / Combined matrices at 218 & 440 keV
- `Auxiliary_Studies/CRCVAR_SinglePhoton`: CRC-Variance analysis for single-photon reconstruction
- `Auxiliary_Studies/EventOrderInference_Experiment`: Event order inference for Compton events
- `Auxiliary_Studies/FreePath`: Free-path simulation studies
- `Auxiliary_Studies/ComptonSystemMatrixPrototype`: Compton system matrix prototyping
- `Auxiliary_Studies/Sensitivity_SPECT_PolarCoor`: computes normalized `Sensi_d` from streamed Compton List files and the current polar-coordinate Factors. It validates the 10496-detector geometry, supports CPU/CUDA execution and resumable checkpoints, and can explicitly install a validated result into a factor directory.
- `Auxiliary_Studies/Reference`: Reference documents and figures

### 12. Reproduction

See `Reproduction/README.md` for a step-by-step guide to reproduce the results.

### 13. Common Failure Modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `master_addr is only used for static rdzv_backend` | Usually just a torchrun warning | Look for the real Python traceback |
| `SIGTERM`, `Socket Timeout` | Cascade after one rank fails | Find the first failing rank's error |
| `can't allocate memory`, exit `-9` | OOM (CPU RAM or GPU VRAM) | Reduce data size or add more nodes |
| `cholesky` not positive-definite | FIM + βR not PD in CRC-VAR | Increase β, reduce grid, or use matrix-free |

### 14. Practical Notes

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
├── Geant4Sim/                  # Geant4 源 macro、JSCC 与 EHE 探测器蒙卡工程
├── CntStat/                    # 生成的投影数据（按能量/体模/计数水平组织）
├── List/                       # 生成的 List 模式事件数据
├── GenProj/                    # MATLAB 系统矩阵前投影脚本
├── Results/                    # 本地生成结果（Git 忽略）
│   ├── Reconstruction/         # 当前重建结果
│   ├── Analysis/               # 定量分析与校正对比
│   ├── Logs/                   # 历史运行日志
│   └── Legacy/                 # 旧 Figure* 结果归档
├── Auxiliary_Studies/          # 辅助研究（与主重建独立）
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
├── *.m                         # MATLAB 可视化/评价脚本
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
| `GenProj/GenProj_SPECT_PolarCoor.m` | 从系统矩阵生成通用单光子投影数据 |
| `GenProj/GenProj_Hoffman_SPECT_PolarCoor.m` | 从系统矩阵生成 Hoffman 脑模体投影数据 |
| `GenProj/GenProj_ContrastPhantom_DualEnergy_PolarCoor.m` | 从 218/440 keV 系统矩阵生成 225Ac 双能量 Contrast Phantom 投影数据 |
| `Geant4Sim/ContrastPhantom_DualEnergy_Rotate_3D.m` | 生成 225Ac 双能量 Contrast Phantom 的 Geant4 macro 与预览 |
| `Geant4Sim/Geant4Code_EHE/` | 与 ConventionalSPECT Params 一致的 EHE 平行孔 Pb/NaI 纯 CntStat Geant4 工程；不输出 List |
| `Geant4Sim/Geant4Data_ContrastPhantom_DualEnergy_225Ac_manifest.json` | 已整理 Geant4 JSCC `1e9`/`1e10` 数据的路径与核验统计 |
| `get_img_SC_MultiOutput_PolarCoor.m` | 按 manifest 读取并比较本地逐能量与联合 SC 结果 |
| `compare_CNRCRC_JSCC_EHE_MultiOutput.m` | 比较 JSCC/EHE 在三种计数水平下校正 218 与直接 440 的 CRC/CNR 曲线 |
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

### 5. 多能量重建

**多能量同时重建**是本框架的一等能力——单能量只是 `--e0-list` 长度为 1 的特例。所有入口都接受 `--e0-list`（一个或多个能量，单位 MeV），以及配套的 `--ene-threshold-sum-list` 和 `--intensity-list`。各能量通道重建**同一张共享图像**。

#### 实现机制

1. **逐能量加载。** 每个能量读取各自的因子目录（`Factors/<1000×e0>keV_RotateNum<R>/`），建立并行的逐能量列表：`sysmat_all`、`proj_all`、`t_local_all`、`sparse_projector_all`、`sensi_s_all`、`sensi_d_all`。

2. **灵敏度图按能量求和。** 各能量灵敏度图在迭代前求和为单一图：
   ```python
   s_map_arg.s = sum(sensi_s_all)
   s_map_arg.d = sum(sensi_d_all) * s_map_d_ratio
   s_map_arg.j = alpha * s + (2 - alpha) * d
   ```

3. **OSEM 内层循环按能量 zip 累加。** 每次迭代中，对每个旋转角，遍历所有能量并把各能量的贡献累加到同一个 `weight_local`，最后除以合并后的 `s_map`：
   ```python
   for rotate_idx in range(rotate_num):
       for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_l, proj_l, rotmat_all, rotmat_inv_all):
           w_s = alpha * get_weight_single(sysmat, proj[:, rotate_idx], img_rotate)
           weight_local += index_select(w_s, rotmat_inv[:, rotate_idx] - 1)
   img = img * weight_local / s_map
   ```
   康普顿路径同理，通过 `zip(..., sysmat_full_all, sparse_projector_all)` 遍历能量。

   数学上，更新公式变为：
   ```
   img = img · Σ_e [α·w_s^(e) + (2-α)·w_c^(e)] / Σ_e [α·s^(e) + (2-α)·d^(e)]
   ```
   `--intensity-list` 用于调节不同能量的相对贡献权重。

#### 各模式支持情况

| 模式 | 文件 | 多能量 | 说明 |
| --- | --- | --- | --- |
| 本地 SC-only | `main_local_cntstat.py` | ✅ 支持 | `--e0-list`；`sum(sensi_s_all)` |
| 本地 CntStat-only 多输出 | `main_local_multi_energy_cntstat.py` | ✅ 支持 | 218/440 双图；显式建模 440→218 串扰并做固定背景 Poisson 校正；不读取 List |
| 本地完整多模式（dense） | `main_plane.py` | ✅ 支持 | 能量 list 硬编码在 Python 中（无 argparse），但多能量管线完整，且有 `MultiEnergy_` 输出路径 |
| 本地稀疏多模式 | `main_plane_sparse.py` | ✅ 支持 | `--e0-list` |
| 本地稀疏 JSCCSD-only | `main_local_sparse_jsccsd_only.py` | ✅ 支持 | `--e0-list` |
| 分布式稀疏多模式 (GPU) | `main_dist_sparse.py` | ✅ 支持 | `--e0-list` + NCCL all_reduce |
| 分布式稀疏 JSCCSD-only (GPU) | `main_dist_sparse_jsccsd_only.py` | ✅ 支持 | `--e0-list` + NCCL all_reduce |
| 分布式 SC-only (GPU) | `main_dist_cntstat.py` | ✅ 支持 | `--e0-list` + all_reduce |
| 分布式稀疏 JSCCSD-only (CPU) | `main_dist_sparse_jsccsd_only_cpu.py` | ✅ 支持 | `--e0-list` + GLOO all_reduce |
| **分布式 dense 多模式 (GPU)** | `main_dist.py` | ⚠️ 部分 | 重建核心支持（`ene_num`、`sum(sensi_s_all)`、all_reduce 齐备），但 `e0_list` 被**硬编码**为 `[0.511]`，未暴露 `--e0-list` 参数 |

唯一的缺口是 `main_dist.py`（`distributed/python/main_dist.py:49-51`）；其重建核心 `recon_osem_dist.py` 已经具备多能量能力，只需把硬编码的能量 list 改成 argparse 参数即可启用，无需改动算法。

#### 使用方法

双能量同时重建示例（如 511 keV + 140 keV）：

```bash
python main_local_sparse_jsccsd_only.py \
  --e0-list 0.511 0.140 \
  --ene-threshold-sum-list 0.46 0.13 \
  --intensity-list 1.0 1.0 \
  --data-file-name ContrastPhantom_240_30 \
  --count-level 1e9 \
  --jsccsd-iter 5000 --save-iter-step 50
```

前提条件：

- `Factors/` 下需有每个能量对应的因子目录（如 `511keV_RotateNum20/` 和 `140keV_RotateNum20/`），各自包含 `SysMat_polar`、`Sensi_s`、`Sensi_d`、`RotMat_full.csv` 等
- `CntStat/` 和 `List/` 下需有对应每个能量的投影与事件数据
- `--e0-list`、`--ene-threshold-sum-list`、`--intensity-list` 三个列表长度必须一致（启动时有校验）

输出目录会自动加 `ME_`（Multi-Energy）前缀并标记全部能量，例如：
`ME_RotNum20_ContrastPhantom_240_30_(511_140)keV_...`

#### 本地 CntStat-only 显式串扰校正链路

`main_local_multi_energy_cntstat.py` 用于 218/440 keV 数据的本地验证。它一次加载
三套按“每个发射光子”归一化的响应：

```text
A218       = A(218能窗 <- 218源)
A440       = A(440能窗 <- 440源)
C440to218  = C(218能窗 <- 440源)
```

针对每个计数水平依次产出：

```text
Image_S_440keV
Image_S_218keV_Contaminated
Image_S_218keV_CrossTalkCorrected
Image_S_(440_218)keV_CrossTalkCorrected
```

观测模型为：

```text
y440 ~ Poisson(A440*x440)
y218 ~ Poisson(A218*x218 + C440to218*x440)
```

程序先用 `A440/y440` 重建 `x440`，再通过 `C440to218` 前投影得到 218 能窗中
由 440 源造成的逐角度预测 CntStat。该预测在 218 重建时作为固定加性背景进入
Poisson 分母，但不加入 `A218` 的 sensitivity。这样不需要对有噪声的投影做硬减法，
串扰校正直接体现在似然模型中。`Image_S_218keV_Contaminated` 仍用 `A218` 单独
重建同一份受污染数据，用于直观看到未校正偏差。

校正后的合并图只做逐像素相加：

```text
x_combined = x440_recon + x218_cross_talk_corrected
```

程序不会为 combined 再运行 OSEM/MLEM，也不假设 218 与 440 keV 空间分布相同。
这条链路不读取康普顿 `List`。使用双能 GenProj 生成的数据时应保持
`--intensity-list 1.0 1.0`，因为观测 CntStat 已包含 225Ac 的 218/440 keV 伽马
产额，重建端不应重复施加分支比。OSEM 的每个 detector-bin 子集使用自己的精确
灵敏度图；`--osem-subset-num 1` 表示 MLEM，并直接复用完整灵敏度图。

```bash
python main_local_multi_energy_cntstat.py \
  --e0-list 0.218 0.440 \
  --intensity-list 1.0 1.0 \
  --data-file-name ContrastPhantom_DualEnergy_10_30_240_30_225Ac \
  --count-levels 1e9 1e10 1e11 \
  --rotate-num 20 --pixel-num-layer 1280 --pixel-num-z 20 \
  --osem-subset-num 1 \
  --single-sc-iter 20000 --single-sc-save-step 50 \
  --device cuda
```

`GenProj/GenProj_ContrastPhantom_DualEnergy_PolarCoor.m` 默认按
`y218=A218*x218+C440to218*x440`、`y440=A440*x440` 一次生成 `1e9`、`1e10`、
`1e11` 三个计数水平。每个重建目录中的 `run_manifest.json` 会记录输入 CntStat、
预测串扰 CntStat、残差诊断、任务定义、二进制图像尺寸以及本次运行的模型范围。

程序默认支持断点续跑。每个任务开始前会验证最终 float32 图像和迭代历史的文件
大小、有限性、非负性，并确认历史最后一帧与最终图逐元素一致；完整任务自动跳过，
缺失或不完整的任务从第 0 次迭代重新计算。只有确实需要全部重算时才传入
`--overwrite-existing`。

MATLAB 展示入口可接收运行目录或其 `Polar` 子目录：

```matlab
get_img_SC_MultiOutput_PolarCoor()
get_img_SC_MultiOutput_PolarCoor("Results/Reconstruction/Figure_Local_SC_MultiOutput/<run-folder>")
```

当前 JSCC 和 EHE 的双能响应均已完整生成：每个系统都包含 218 直接响应、440
直接响应和 `440keV_to218win` 交叉能窗响应；EHE 三套目录使用
`_SPECTEHENaI` 后缀。同一系统内三套响应共享几何，JSCC 有 10496 个 detector
bins，EHE 有 2312 个。

当前 Contrast Phantom 对比采用 MLEM（`subset=1`）。JSCC 按计数水平分别使用
`1e9:5000`、`1e10:10000`、`1e11:20000` 次迭代，每 50 次保存；EHE 三个计数
水平都使用 100 次迭代，并改为每次迭代保存。`compare_CNRCRC_JSCC_EHE_MultiOutput.m`
会自动读取这组配置，分别评价 218 keV 的 10/18/26 mm 热圆柱和 440 keV 的
14/22/30 mm 热圆柱，在 `Results/Analysis/CNRCRC_JSCC_vs_EHE/` 输出统一坐标范围的
CRC（0 到 1）、CNR（0 到 15）曲线以及逐迭代 CSV/MAT 汇总。

目前已有两组由 Geant4 JSCC 工程实际模拟的 `1e9`、`1e10` 数据，和 GenProj
数据分开保存：

```text
CntStat/218keV_RotateNum20_Geant4JSCC/
CntStat/440keV_RotateNum20_Geant4JSCC/
List/218-440keV_RotateNum20_Geant4JSCC/
```

本地 CntStat-only 重建使用 `--cntstat-dir-suffix _Geant4JSCC`，但不要设置
`--factor-dir-suffix`，因为它仍与标准 JSCC Factors 配套。这里的 List 混合了
218/440 初级源事件，而且没有初级能量标签，不能伪装成纯 218 或纯 440 的单能
List 输入。完整路径、计数统计、格式核验及当前未启用实体水/PMMA 体模的限制见
`Geant4Sim/Geant4Data_ContrastPhantom_DualEnergy_225Ac_manifest.json`。
这两组已归档文件标记为 legacy 诊断数据：生成它们时 `EventAction` 仍只检查最高
能量晶体，并把 CntStat 与 List 设为互斥。修复后的代码会遍历全部晶体，且
CntStat 与 List 独立、可同时记录；正式定量使用前必须重新运行 Geant4。

该函数读取 `run_manifest.json`，不解析长目录名；极坐标到笛卡尔坐标的三角剖分和
插值权重只计算一次。结果写入 `Display/mip_comparison.{png,fig}` 与
`Display/final_orthogonal.{png,fig}`。

#### Geant4 225Ac 双伽马代理源

`Geant4Sim/ContrastPhantom_DualEnergy_Rotate_3D.m` 生成 20 个 GPS macro。当前不
直接生成放射性 225Ac 离子，而是按下式生成两条伽马线：

```text
q218(r) = Y218 * x_Fr(r) / sum(x_Fr),  Y218 = 0.114
q440(r) = Y440 * x_Bi(r) / sum(x_Bi),  Y440 = 0.261
```

Fr 与 Bi 的空间分布有意设为不同：两种能量都包含均匀背景，rod 1/3/5 是
Fr/218 分布的热区，rod 2/4/6 是 Bi/440 分布的热区，用来模拟 alpha 衰变后子体
脱离螯合药物并发生空间迁移。每个热柱相对其本能量背景为 6 倍。由于 `x_Fr` 和
`x_Bi` 的空间积分不同，生成端先分别把两张分布归一化，再施加核数据产额。整个
run 的期望初级光子份额因此严格为 218 keV 的 30.4% 和 440 keV 的 69.6%；GPS
随机抽样只带来有限计数下应有的统计涨落。

20 个视角合计 `1e9` 个初级光子，通常每个 macro 为 `5e7`。GPS 每个 event 只
抽取一个 source，因此 `/run/beamOn` 表示选中的 218/440 初级光子总数，不表示
225Ac 母体衰变次数。`EventAction` 采用 511 keV 处 13% FWHM、按 `1/sqrt(E)`
缩放的能量展宽，分别输出 `CntStat_218.csv` 与 `CntStat_440.csv`；440 keV 光子
散射落入 218 能窗时会自然计入前者。满足条件的双晶体事件写入动态增长的
`List.csv`。

当前几何限制必须注意：GPS 圆柱只定义源位置，不会生成实体模体。
`DetectorConstruction.cc` 中的水/PMMA Contrast Phantom 仍处于禁用状态，世界
材料近似真空，因此尚未模拟物体内衰减和物体散射；探测器及屏蔽结构内的相互作用
仍会被模拟。

### 6. 多能量多输出链路

一套专门的分布式链路，**加载一次所有能量数据，单次运行产出全部 5 种图像**，每种图像可独立设置迭代次数，并支持按能量选择性禁用康普顿数据。

| 类型 | 模式 | 能量选择 | 输出 |
| --- | --- | --- | --- |
| 1 | 仅单光子（`S`） | 每个能量各一张 | 每个能量一张图 |
| 2 | 仅康普顿（`D`） | 每个白名单能量各一张 | 每个白名单能量一张图 |
| 3 | 仅单光子（`S`） | 全部能量 | 一张联合图 |
| 4 | 仅康普顿（`D`） | 全部白名单能量 | 一张联合图（与类型 2 重复时自动去重） |
| 5 | 联合（`J`） | S 用全部能量 + D 用全部白名单能量 | 一张联合图 |

`--compton-energies` 声明哪些能量有可用的康普顿（List）数据。**未列入白名单的能量完全不加载 List、不构建 T 矩阵、不在 GPU 上复制 `sysmat_full`**——类型 2/4/5 自然就不包含它们。当类型 4 与某个类型 2 任务完全相同（同模式、同单能量康普顿子集）时，自动跳过。

#### 文件

| 文件 | 功能 |
| --- | --- |
| `distributed/python/multi_energy_tasks.py` | 纯逻辑任务模型：5 种类型 → 去重后的 `ReconTask` 列表 + 命名 |
| `distributed/python/recon_osem_dist_multi_energy.py` | 重建核心：3 个 OSEM 模式（`osem_single_dist` / `osem_compton_dist` / `osem_joint_dist`）+ 任务驱动 |
| `distributed/python/main_dist_multi_energy.py` | 入口：argparse、数据加载（所有能量加载单光子，仅白名单加载康普顿）、任务调度 |
| `distributed/scripts/jsccrecon_dist_multi_energy.sh` | SLURM 提交脚本（`gpu_5090`，4 节点 × 8 GPU） |
| `distributed/FRBI_COUPLED_RECON_DESIGN.md` | 后续 225Ac Fr/Bi 双图耦合重建设计：218 能窗 440 keV 串扰、分支比权重、OSEM 更新式 |

三个 OSEM 模式函数与 `recon_osem_dist_sparse_jsccsd_only.py` 的数学完全一致；仅分支方式（仅 S / 仅 D / 联合）和任务调度是新增的。

当前多能量输出仍然是**单图像**重建：每个任务用选定的 SPECT/Compton 通道更新同一张共享图像。对于 221Fr/218 keV 与 213Bi/440 keV 空间分布不同的 225Ac 场景，不应把 `A_218win<-218 + (Y440/Y218)*A_218win<-440` 解释为纯 Fr 响应；后续应按 `distributed/FRBI_COUPLED_RECON_DESIGN.md` 中的双图耦合模型实现。

#### 使用方法

```bash
# 440 keV + 218 keV；218 keV 无康普顿数据（白名单仅 440）
sbatch distributed/scripts/jsccrecon_dist_multi_energy.sh \
  --e0-list 0.440 0.218 \
  --ene-threshold-sum-list 0.40 0.18 \
  --intensity-list 1.0 1.0 \
  --compton-energies 0.440 \
  --data-file-name ContrastPhantom_240_30 --count-level 1e9 \
  --single-sc-iter 1000 --single-sc-save-step 50 \
  --single-compton-iter 2000 --single-compton-save-step 50 \
  --joint-sc-iter 2000 --joint-sc-save-step 50 \
  --joint-compton-iter 2000 --joint-compton-save-step 50 \
  --joint-iter 5000 --joint-save-step 50
```

每种类型的迭代次数独立设置（设为 0 则跳过该类型）。上例产出 5 张图（类型 4 因与类型 2 的 440 单能量任务重复而被去重）：

```
Image_S_440keV               Image_S_218keV             # 类型 1
Image_D_440keV                                          # 类型 2
Image_S_(440_218)keV                                    # 类型 3
Image_J_S(440_218)keV_D440keV                           # 类型 5
```

#### 各类型 CLI 参数

| 参数 | 控制 |
| --- | --- |
| `--compton-energies` | 康普顿白名单（默认 = `--e0-list` 全部） |
| `--single-sc-iter` / `--single-sc-save-step` | 类型 1（单能量单光子） |
| `--single-compton-iter` / `--single-compton-save-step` | 类型 2（单能量康普顿） |
| `--joint-sc-iter` / `--joint-sc-save-step` | 类型 3（全能量单光子） |
| `--joint-compton-iter` / `--joint-compton-save-step` | 类型 4（全能量康普顿） |
| `--joint-iter` / `--joint-save-step` | 类型 5（全能量联合） |

### 7. 分布式执行路径

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

### 8. 因子文件说明

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

#### 计算 `Sensi_d`

`Auxiliary_Studies/Sensitivity_SPECT_PolarCoor/` 提供了与当前 Factors 目录直接兼容的
`Sensi_d` 计算链路。它流式读取 Compton List，在计算设备上逐批归约事件反投影，
再使用 `RotMat_full.csv` 做旋转平均。当前生产 Factors 为
`B=A*diag(DeltaV_mm3)` 活度浓度基底，其绝对标定满足
`sum(Sensi_d)/Vsource = 有效事件数/该能量实际发射光子数`。默认严格检查当前
几何应包含 10496 个有效探测器。

218 和 440 keV 应分别输入各自的 List 和发射光子数；混合源 List 不能直接作为
单一能量计算。常用总能阈值分别为 0.18 MeV 和 0.40 MeV。完整命令、断点续算、
结果验收和安装到 `Factors/<energy>keV_RotateNum20/Sensi_d` 的方法见
`Auxiliary_Studies/Sensitivity_SPECT_PolarCoor/README.md`。

218/440 两个独立单能均匀源 macro 位于
`Geant4Sim/Macro/SensiD_UniformFullFOV/`；源为 `R=153 mm`、高 `60 mm`、中心
`(0,-245,0) mm`，完整覆盖 25620 个极坐标单元边界。旧的等权极坐标点阵不能
用于当前密度基灵敏度标定。

### 9. 典型工作流程

1. **生成因子**：用 MATLAB 脚本创建系统矩阵和几何文件
2. **生成数据**：用 MATLAB/Geant4 创建投影和 List 数据
3. **重建**：运行本地或分布式重建
4. **可视化**：极坐标结果转笛卡尔图像
5. **评价**：计算 CRC、CNR、PVR 等指标
6. **辅助研究**：按需运行 CRC-VAR、事件顺序推断等

详见 `Reproduction/README.md`。

### 10. 常见报错与排查

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `master_addr is only used for static rdzv_backend` | torchrun 警告 | 找后面的 Python traceback |
| `SIGTERM`, `Socket Timeout` | 一个 rank 失败后连带退出 | 找第一个失败的 rank |
| `can't allocate memory`, 退出码 `-9` | 内存不足 | 减少数据量或增加节点 |
| `cholesky` 非正定 | CRC-VAR 中 β 太小 | 增大 β、减小网格、改用 matrix-free |

### 11. 实用建议

- 排查分布式错误时，优先看第一条 Python 异常
- CPU 分布式与 GPU 版本数学逻辑完全一致，float32 精度范围内结果相同
- 大规模数据时优先考虑 CPU 分布式，内存远大于 GPU 显存
