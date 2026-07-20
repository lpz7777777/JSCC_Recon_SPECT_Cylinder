# JSCC System-Matrix Development History

This directory is the compact, permanent record of analytical system-matrix
changes and their comparisons with Geant4. Large obsolete matrices and Factors
are not retained merely for provenance; their small reports are copied under
`evidence/` before deletion.

## Comparison policy

There are two distinct questions and they must not be mixed:

1. **Physics-model comparison:** compare the raw analytical matrix with Geant4
   using Factors generated with `calibration_profile='none'`. This is the only
   comparison used to decide whether a code change improves the model.
2. **Application calibration:** after selecting a physics model, derive and
   apply a new correction from data independent of the final validation set.
   Report this as a separate calibrated result.

`GenFactors/run_gen_response_factors.m` currently defaults to the historical
`center_point_20260716` profile. Its four layer factors were fitted to the
pre-PE-v4 detector-local matrix:

```text
response     y=30       y=60       y=90       y=120 mm
A218         0.8740232  0.8793926  0.8719679  0.8708826
A440         0.8720313  0.8912363  0.8847924  0.8691287
C440to218    1.1411090  1.1636691  1.2201934  1.2372030
```

Consequently, all historical `CenterPoint*` comparisons listed below contain
the same old center-point calibration unless explicitly marked otherwise.
They are useful for comparing shape changes under a fixed transform, but they
are not raw matrix-vs-Geant4 validation. Never reuse these factors to judge a
new transport implementation.

For an uncalibrated development Factor call, pass the profile explicitly:

```matlab
grid = struct( ...
    'include_center_point', true, ...
    'run_name_suffix', '_pe_v4', ...
    'calibration_profile', 'none');
run_gen_response_factors( ...
    ["JSCC/A218", "JSCC/A440", "JSCC/C440to218"], ...
    'CenterPoint_PEv4_Raw', grid);
```

## Version timeline

| ID | Main model change | Validation state | Main conclusion |
| --- | --- | --- | --- |
| V3-DL | Exact detector-local first-position escape; target-B surface integration | Full Uniform-FOV Geant4, but with old center calibration | Established the pre-PE-v4 baseline. |
| V4-A | PE v4 finite-distance visible-surface model with the first 256 Halton points | Full Uniform-FOV Geant4, same old calibration | Invalid directional artifact: x-z plane explained 85-93% of the v4-v3 detector difference. Deleted after evidence snapshot. |
| V4-S | Four-way reflection-paired symmetric Halton PE v4; GAGG/W densities aligned to 6.60/19.35 g/cm3 | Full Uniform-FOV Geant4, same old calibration | Removed the diagonal artifact. Shape L2 changed little for A218/A440 and improved 2.5% for C440to218 versus V3-DL. |
| V4-S-L | V4-S followed by a layer correction fitted to the same Uniform-FOV data | Same fitted data, not independent validation | Shape L2 became 0.003482/0.005622/0.010619. This measures calibration capacity, not a transport-model improvement. |

## Historical Uniform-FOV metrics

The table below uses the historical calibrated `CenterPoint*` Factors. Values
are shape L2 after total scaling; lower is better.

| Version | A218 | A440 | C440to218 | Calibration/data note |
| --- | ---: | ---: | ---: | --- |
| V3-DL | 0.007150 | 0.010496 | 0.011693 | old center-point layer profile |
| V4-A | 0.008214 | 0.011078 | 0.012002 | same old profile; asymmetric sampling |
| V4-S | 0.007276 | 0.010593 | 0.011404 | same old profile; symmetric sampling |
| V4-S-L | 0.003482 | 0.005622 | 0.010619 | additionally fitted to this Uniform-FOV data |

V4-S reduced V4-A shape L2 by 11.4% for A218, 4.4% for A440, and
5.0% for C440to218. Compared with V3-DL, V4-S was 1.8% worse for A218,
0.9% worse for A440, and 2.5% better for C440to218. These numbers quantify the
sampling correction but do not isolate raw transport accuracy because the old
calibration is present in every Factor set.

## Current V4-S status

V4-S is the active production model. Its symmetric PE-v4 matrices, detector-
local ScatterGen model, and raw center-inclusive Factors are retained. The next
validation must distinguish polar-grid source measure from transport error:
a vector with equal value at every polar sample is not a uniform activity
density unless each sample represents the same physical volume.

The 2026-07-20 source-measure audit confirmed a 6:1 range in represented cell
volume. Physical-volume weighting reduced the Geant4 contrast-phantom CntStat
shape L2 only from 0.057205 to 0.056872 in the 218 window and from 0.091487 to
0.091407 in the 440 window. The mismatch can explain a local center trend in
the reconstructed image, but it does not explain the remaining global detector
shape residual. Keep source discretization and transport accuracy as separate
validation axes. Full details are in `GenProj/POLAR_SOURCE_MEASURE.md` at the
main project root.

The uncalibrated V4-S replay reaches the same result (`0.063447 -> 0.063262`
and `0.092143 -> 0.092021`), so the conclusion does not depend on the
Uniform-FOV layer fit.

The subsequent density-basis reconstruction used
`B=A*diag(DeltaV_mm3)` with the V4-S-L Factors and the existing Geant4 1e10
CntStat for 2000 MLEM iterations. The center-to-middle background ratio changed
from `0.699 -> 1.157` for A440, `0.419 -> 0.890` for corrected A218, and
`0.607 -> 1.010` for their corrected sum. This confirms that source measure was
a major cause of the displayed center depression. The remaining 440 center
overshoot and outer-FOV decline must still be treated as position-dependent
response mismatch, not as a reason to remove the volume basis.

The validated density-basis matrices are now the standard no-suffix JSCC
Factors. Future generation uses `GenFactors/run_gen_jscc_production_factors.m`,
which combines the V4-S `_pe_v4` inputs, center-inclusive grid, validated
absolute layer factors, and `A*diag(DeltaV_mm3)` transform in one staged,
validated operation. The generic generator defaults to no empirical
calibration so raw transport comparisons remain distinguishable from fitted
application Factors.

## Evidence layout

```text
evidence/
  V4A_vs_V3/       compact report before deleting V4-A matrices
  V4A_vs_V4S/
  V4S_vs_V3/
  V4S_layer_fit/
```
