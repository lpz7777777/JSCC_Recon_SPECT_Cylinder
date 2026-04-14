# JSCC SPECT Reconstruction Project

This repository contains MATLAB and Python code for SPECT reconstruction with:

- `SC` (single-photon / self-collimation style reconstruction)
- `SCD`
- `JSCCD`
- `JSCCSD`
- distributed reconstruction pipelines
- phantom generation, projection simulation, and image-quality evaluation

The codebase is centered on polar-coordinate reconstruction for SPECT geometry, with both local and distributed workflows.

## Main Directories

- `Factors/`
  Precomputed system matrices, rotation maps, polar coordinates, and related factors.
- `CntStat/`
  Simulated count-statistics data.
- `Figure/`
  Reconstruction outputs and post-processing results.
- `distributed/python/`
  Distributed reconstruction entry points and OSEM implementations.
- `Geant4Sim/`
  Geant4 source-macro generation scripts and phantom generation utilities.

## Core MATLAB Scripts

- `GenProj_SPECT_PolarCoor.m`
  Generates projection data from the system matrix in polar coordinates.
- `GenProj_Hoffman_SPECT_PolarCoor.m`
  Generates Hoffman phantom count-statistics data from the system matrix.
- `get_img_SPECT_PolarCoor.m`
  Converts polar reconstruction outputs to Cartesian images and generates SC / JSCCSD orthogonal views and MIP figures.
- `get_img_SC_Dist_PolarCoor.m`
  Converts distributed SC outputs to Cartesian images and generates orthogonal views and MIP figures.
- `CNRCRC_SPECT.m`
  Computes CRC / CNR curves and image comparisons for selected iterations.

## Core Distributed Python Scripts

- `distributed/python/main_dist_*.py`
  Main distributed reconstruction entry points for different count levels or workflows.
- `distributed/python/main_dist_sparse.py`
  Distributed sparse Compton reconstruction pipeline.
- `distributed/python/recon_osem_dist.py`
  Standard distributed OSEM reconstruction for SC / SCD / JSCCD / JSCCSD.
- `distributed/python/recon_osem_dist_sparse.py`
  Sparse distributed OSEM reconstruction for SC / SCD / JSCCD / JSCCSD.
- `distributed/python/recon_osem_dist_cntstat.py`
  Distributed SC-only reconstruction from count-statistics data.

## Recent Updates

### 1. Per-Iteration Random OSEM Bin Subsets

For the distributed `SC`, `SCD`, and the single-photon part of `JSCCSD`, detector-bin subsets are now reshuffled at every outer iteration instead of staying fixed for the whole run.

Affected files:

- `distributed/python/recon_osem_dist.py`
- `distributed/python/recon_osem_dist_sparse.py`
- `distributed/python/recon_osem_dist_cntstat.py`

Notes:

- This only changes the detector-bin subset partition.
- The Compton `t` subsets are unchanged.
- If `iter_arg.seed` is absent, the code falls back to an internal default seed.

### 2. Configurable Transverse MIP Layer Range

`get_img_SPECT_PolarCoor.m` and `get_img_SC_Dist_PolarCoor.m` now support selecting the axial layer range used for transverse MIP.

New user parameters near the top of each script:

```matlab
mipStartLayer = [];
mipEndLayer = [];
```

Behavior:

- `[]` means full axial range.
- Example:

```matlab
mipStartLayer = 6;
mipEndLayer = 15;
```

This computes MIP only from layers `6:15`.

### 3. Hoffman Phantom Transverse Scaling

The Hoffman phantom generation and projection scripts now support transverse scaling while keeping the reconstruction FOV size fixed.

Affected files:

- `Geant4Sim/BrainPhantom_HoffmanRawCompressed_3D.m`
- `GenProj_Hoffman_SPECT_PolarCoor.m`

Key parameter:

```matlab
cfg.transverse_scale = 1.4;
```

or

```matlab
transverse_scale = 1.4;
```

The generated preview/output directory will include the scale tag, for example:

- `HoffmanRawCompressed_300x300x60_XYx1p40`

## Typical Workflow

### A. Generate or Load Factors

Prepare the factor files under `Factors/<energy>keV_RotateNum<rotate_num>/`, such as:

- `SysMat_polar`
- `RotMat.mat` or `RotMat_full.mat`
- `RotMatInv.mat` or `RotMatInv_full.mat`
- `coor_polar.mat`

### B. Generate Projection / Count Data

Examples:

- Contrast phantom / hot-rod phantom:
  `GenProj_SPECT_PolarCoor.m`
- Hoffman phantom:
  `GenProj_Hoffman_SPECT_PolarCoor.m`

### C. Run Reconstruction

Examples:

- Local / non-distributed workflows:
  MATLAB and local Python scripts in the repository root.
- Distributed workflows:
  `distributed/python/main_dist_*.py`
- Sparse distributed workflow:
  `distributed/python/main_dist_sparse.py`

### D. Convert Images and Visualize

- SC distributed results:
  `get_img_SC_Dist_PolarCoor.m`
- SC + JSCCSD comparison:
  `get_img_SPECT_PolarCoor.m`

### E. Evaluate CRC / CNR

- Run `CNRCRC_SPECT.m`

## Image Output Notes

### `get_img_SPECT_PolarCoor.m`

Outputs typically include:

- `show.png`
- `show.fig`
- `mip.png`
- `mip.fig`

### `get_img_SC_Dist_PolarCoor.m`

Outputs typically include:

- `img_show_sc.png`
- `img_show_sc.fig`
- `mip_sc.png`
- `mip_sc.fig`

### `CNRCRC_SPECT.m`

Outputs typically include:

- CRC / CNR curve figures
- best-iteration comparison figures
- transverse MIP comparison figures for selected iterations

## Geant4 Phantom Utilities

The `Geant4Sim/` directory contains scripts for generating voxelized or analytic source macros for Geant4 simulations, including:

- hot-rod phantom source generation
- Hoffman compressed phantom source generation
- point-array and contrast phantom source generation

Several scripts rotate the source over multiple views to emulate SPECT acquisition.

## Notes

- Most distributed scripts assume the same directory structure on all compute nodes.
- Some reconstruction scripts use GPU memory reporting before iteration starts.
- The repeated PyTorch warning
  `No device id is provided via init_process_group or barrier`
  is only a warning unless followed by an actual traceback.

## Suggested Next Documentation Targets

The following parts are still worth documenting in more detail later:

- exact input/output file formats for `Factors/`, `CntStat/`, and `List_*`
- recommended distributed launch commands
- sparse Compton chain assumptions and limits
- phantom naming conventions for `Figure/` and `Geant4Sim/Preview/`
