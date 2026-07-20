# Polar source measure

## Problem

`SysMat_polar(:, j)` is the detector response per photon emitted at polar
sample `j`. It is a point-response column and does not include a polar-cell
Jacobian or voxel volume. The corresponding forward model is therefore

```text
y = A x
x_j = emitted activity integrated over polar cell j
```

It is not correct to interpret `x_j` directly as activity density when the
polar samples represent unequal physical volumes.

The current grid uses radii `0, 6, ..., 150 mm` and changes angular sampling
in steps:

```text
r =   6..36 mm: 20 points/ring
r =  42..72 mm: 40 points/ring
r = 78..108 mm: 60 points/ring
r = 114..150 mm: 80 points/ring
```

For radial bounds `r_-`, `r_+`, angular count `N_theta`, and axial thickness
`Delta z`, one sample represents

```text
Delta V_j = pi * (r_+^2 - r_-^2) / N_theta * Delta z
```

On the current grid, cell volumes inside the 240 x 30 mm contrast phantom
range from `33.9292` to `203.5752 mm3`, a factor of 6. Equal values at every
sample are therefore not a physically uniform activity density.

## Geant4 distinction

Two existing Geant4 studies use different source measures:

- `GenerateUniformFovCntStatMacros.m` creates an equal-weight point array at
  all 25,620 polar samples. This deliberately matches `x_j = constant`; it is
  not a uniform volume cylinder.
- `ContrastPhantom_DualEnergy_Rotate_3D.m` uses GPS `Volume/Cylinder` sources.
  It represents uniform activity per physical volume inside each cylinder.

The equal-point Uniform-FOV experiment remains valid for detector-row response
calibration. It cannot validate the polar Jacobian or the density convention.

## 2026-07-20 audit

Run:

```powershell
C:\ProgramData\anaconda3\envs\pytorch\python.exe `
  GenProj\analyze_polar_source_measure.py
```

Outputs are written to `Results/Analysis/PolarSourceMeasure_20260720/`.
The audit:

1. reconstructs exact polar-cell bounds from `coor_polar_full.csv`;
2. verifies that the cell overlaps sum to the analytic 240 x 30 mm cylinder
   volume with zero numerical closure error;
3. integrates each off-center hot cylinder with deterministic 32 x 32 polar
   subcell quadrature (individual rod-volume errors below 0.19%);
4. forward-projects equal-point and physical-volume source maps with the V4-S
   layer-corrected `A218`, `A440`, and `C440to218` matrices;
5. compares both predictions with the existing Geant4 1e10 CntStat.

Using physical-volume weights changes the normalized full-CntStat shape error
only slightly:

```text
window   equal-point L2   physical-volume L2
218      0.0572051        0.0568716
440      0.0914871        0.0914071
```

Repeating the comparison with uncalibrated `CenterPoint_RawV4S` Factors gives
the same conclusion: `0.0634471 -> 0.0632619` for the 218 window and
`0.0921430 -> 0.0920214` for the 440 window. The conclusion is therefore not
an artifact of the fitted Uniform-FOV layer correction.

The source distributions themselves are materially different. Equal-point
weighting puts about 6.1-6.35% of emitted activity inside `r <= 18 mm`, while
physical-volume weighting puts only 2.69-2.80% there. Equal-point GenProj thus
overweights the central region by about 2.26 times. The full spatial total
variation between the two source distributions is 10.7-11.1%.

This is expected because detector projections are weakly sensitive to local
redistribution near the FOV center. The result means:

- the source-measure mismatch is real and can create a central trend in the
  reconstructed image;
- it is not the main cause of the remaining 5.7%/9.1% global CntStat shape
  residual, which still indicates response-model mismatch;
- post-hoc division of a finite-iteration image by cell volume can overcorrect
  poorly resolved inner rings and is not the preferred implementation.

## Required production convention

Use activity density `rho_j` as the user-facing image and write the forward
model as

```text
x_j = Delta V_j * rho_j
y = A * diag(Delta V) * rho
```

For cylinders and rods that cross cell boundaries, replace `Delta V_j` with
the actual overlap volume. Normalize source yields only after this volume
integration.

For reconstruction, either explicitly use `B = A * diag(Delta V)` or retain
the existing integrated-activity variable `x` while:

- initializing `x` proportional to `Delta V` for a uniform-density prior;
- converting `rho = x / Delta V` before Cartesian interpolation and display;
- recording both conventions in the run manifest.

The initial audit did not alter production Factors or images. The analysis
script remains non-destructive; the separate density-basis production test is
documented below.

## Density-basis Factors and 1e10 reconstruction

The follow-up production test first created separate Factors without modifying
the V4-S-L source matrices:

```text
Factors/218keV_RotateNum20_CenterPoint_PEv4_UniformFOVLayer_PolarVolume
Factors/440keV_RotateNum20_CenterPoint_PEv4_UniformFOVLayer_PolarVolume
Factors/440keV_to218win_RotateNum20_CenterPoint_PEv4_UniformFOVLayer_PolarVolume
```

After validation, these three matrices were promoted to the long-term
no-suffix JSCC directories:

```text
Factors/218keV_RotateNum20
Factors/440keV_RotateNum20
Factors/440keV_to218win_RotateNum20
```

Future generation uses the MATLAB production entry point
`run_gen_jscc_production_factors`; the Python builder below is retained as the
one-time migration/audit implementation.

Generate them with:

```powershell
C:\ProgramData\anaconda3\envs\pytorch\python.exe `
  GenProj\build_polar_volume_weighted_factors.py
```

Every on-disk `(pixel, detector-bin)` row is multiplied by that pixel's full
polar-cell volume in `mm3`. The three transformed matrices preserve the source
matrix byte count and pass rotation-invariance and float32 sum checks. Their
relative transformed-sum errors are `2.24e-9`, `2.32e-9`, and `2.19e-9`.

The Geant4 1e10 CntStat was reconstructed for 2000 MLEM iterations with these
Factors. The output is under
`Results/LocalReconstructionRuns/PEv4_UniformFOVLayer_PolarVolumeDensity_Calibrated/`.
The old and new images were
normalized by their own rod-excluded `r=30..108 mm` background median. Center
ratio means the median background radial profile over `r=0,6,12,18 mm` divided
by that middle-background median:

```text
image                   old center ratio   density-basis ratio   bias reduction
440 keV                     0.699                1.157                47.9%
218 keV corrected           0.419                0.890                81.0%
440 + 218 corrected         0.607                1.010                97.6%
```

The corrected-218 radial CV over `r=0..108 mm` improves from `0.311` to
`0.144`; the corrected sum improves from `0.249` to `0.223`. The 440 radial CV
changes from `0.247` to `0.277` because the density-basis profile is mildly high
in the center and still decreases toward the FOV edge. Thus the polar measure
is a major cause of the center depression, but it does not explain all
position-dependent matrix/Geant4 mismatch.

Full comparison outputs are in
`Results/Analysis/PolarVolumeRecon_20260720/Comparison/`. Reproduce them with:

```powershell
C:\ProgramData\anaconda3\envs\pytorch\python.exe `
  GenProj\compare_polar_volume_reconstruction.py
```
