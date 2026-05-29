# CRCVAR_SinglePhoton

This folder computes CRC-VAR curves from single-photon Cartesian system
matrices stored as `SysMat_tmp` under selected `Factors/<factor_name>`
directories.

The current implementation provides two paths:

- `run_crcvar_single_photon.py`
  matrix-free Fisher-operator method, better suited for larger working grids
- `run_crcvar_single_photon_direct.py`
  dense direct method, intended for smaller interpolated working grids

## Input Assumptions

The scripts assume each selected factor directory contains:

- `SysMat_tmp`

Current default geometry assumptions:

- native image grid: `51 x 51 x 20`
- native voxel size: `6 x 6 x 3 mm`

`SysMat_tmp` is interpreted as a float32 Cartesian system matrix with shape:

```text
51 x 51 x 20 x Ndet
```

Internally it is reshaped into a 2D matrix with:

```text
[num_detectors, num_voxels]
```

## 1. Matrix-Free Script

Script:

- `run_crcvar_single_photon.py`

Main characteristics:

- does not explicitly build a dense global FIM
- uses a matrix-free Fisher operator
- solves the required systems with batched PCG
- evaluates CRC and VAR on a regular point grid, then averages over those
  sampled voxels
- supports GPU well

Typical PowerShell example:

```powershell
conda run --no-capture-output -n pytorch python -u .\Auxiliary_Studies\CRCVAR_SinglePhoton\run_crcvar_single_photon.py `
  --factor-dirs `
  Factors\140keV_RotateNum20 `
  Factors\140keV_RotateNum20_SPECTEHENaI `
  --run-name CRCVAR_140keV_compare `
  --sample-grid-nx 20 `
  --sample-grid-ny 20 `
  --sample-grid-nz 5 `
  --sample-spacing-mm 10 `
  --point-batch-size 8 `
  --device cuda
```

## 2. Direct Dense Script

Script:

- `run_crcvar_single_photon_direct.py`

Main characteristics:

- can first interpolate the native `51 x 51 x 20` system matrix to a smaller
  working grid such as `25 x 25 x 10` or `40 x 40 x 20`
- explicitly builds a dense FIM
- explicitly builds a dense regularizer matrix `R`
- `R` is now vectorized and built only once, then reused across all factors
- uses a center cylindrical ROI average instead of the previous point-grid
  average

Current ROI definition:

- cylindrical ROI centered at the FOV center
- controlled by:
  - `--roi-diameter-mm`
  - `--roi-height-mm`
- if `--roi-height-mm` is omitted, the full working-grid z extent is used

Important numerical note:

- the direct script uses Cholesky factorization of `FIM + beta * R`
- if beta is too small, the system may fail to be strictly positive-definite
- in that case, increase `--beta-level-start`

Typical PowerShell example:

```powershell
conda run --no-capture-output -n pytorch python -u .\Auxiliary_Studies\CRCVAR_SinglePhoton\run_crcvar_single_photon_direct.py `
  --factor-dirs `
  Factors\140keV_RotateNum20 `
  Factors\140keV_RotateNum20_SPECTEHENaI `
  Factors\511keV_RotateNum20 `
  Factors\511keV_RotateNum20_SPECTEHENaI `
  --run-name CRCVAR_direct_full_4factors_25x25x10 `
  --device cuda `
  --dtype float32 `
  --interp-nx 25 `
  --interp-ny 25 `
  --interp-nz 10 `
  --roi-diameter-mm 150 `
  --solve-batch-size 128 `
  --fim-row-chunk 512 `
  --interp-detector-batch-size 256
```

Example with a limited center ROI:

```powershell
conda run --no-capture-output -n pytorch python -u .\Auxiliary_Studies\CRCVAR_SinglePhoton\run_crcvar_single_photon_direct.py `
  --factor-dirs Factors\140keV_RotateNum20 `
  --run-name CRCVAR_direct_roi_example `
  --device cuda `
  --interp-nx 40 `
  --interp-ny 40 `
  --interp-nz 20 `
  --roi-diameter-mm 200 `
  --roi-height-mm 50 `
  --beta-level-start -6
```

## 3. Progress Logging

Both scripts now print progress information during computation.

The direct script reports at least:

- interpolation progress
- dense `R` construction
- dense FIM accumulation
- factor index / total factor count
- beta index / total beta count
- solve-batch progress
- elapsed time and ETA

If you do not see live logs in PowerShell, use:

```powershell
conda run --no-capture-output -n pytorch python -u ...
```

## 4. Output Layout

Results are written to:

```text
Auxiliary_Studies/CRCVAR_SinglePhoton/Result/<run-name>/SinglePhoton
```

Important files:

- `beta_values`
- `CRC_mean_<factor_label>`
- `Var_mean_<factor_label>`
- `summary.json`

`summary.json` records:

- factor labels and source directories
- native grid and working grid
- working voxel sizes
- ROI settings for the direct script
- beta schedule
- device / dtype
- solve configuration

## 5. MATLAB Plotting

Plot script:

- `plot_crcvar_single_photon.m`

Current behavior:

- asks you to select a result folder under `Result/`
- reads `summary.json`, `beta_values`, `CRC_mean_*`, `Var_mean_*`
- plots CRC-VAR curves on log-log axes
- uses line style to distinguish energy
- uses color to distinguish system type:
  - red for labels containing `SPECTEHENaI`
  - green otherwise

## 6. Practical Recommendations

For larger working grids:

- prefer `run_crcvar_single_photon.py`

For smaller interpolated grids where you explicitly want dense direct solves:

- use `run_crcvar_single_photon_direct.py`
- keep the working grid modest
- start beta from a not-too-small level such as `1e-6` if Cholesky fails

The direct method becomes expensive very quickly as the working-grid voxel
count grows, even if GPU memory is still available.

