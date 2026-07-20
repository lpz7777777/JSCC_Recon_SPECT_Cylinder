# JSCC Reconstruction Development Handoff

Last consolidated: 2026-07-20.

This is the primary starting document for a new developer or a new
conversation. Large Factors, Geant4 output, List, CntStat, and Results are
ignored by Git, so the required semantics and evidence are recorded here.

## Production baseline

The active JSCC Factors are exactly the three no-suffix directories:

```text
Factors/218keV_RotateNum20/             A218
Factors/440keV_RotateNum20/             A440
Factors/440keV_to218win_RotateNum20/    C440to218
```

They contain 10496 detector bins, 25620 center-inclusive polar samples, and
20 rotations. They use the PE-v4/detector-local matrix model, the current
Uniform-FOV four-layer calibration, and the density-basis transform:

```text
B = A * diag(DeltaV_mm3)
y218 = B218*rho218 + BC440to218*rho440
y440 = B440*rho440
```

The underlying point response `A_E(d,j)` is normalized per emitted
monoenergetic photon and excludes 225Ac gamma yields. The production matrix
`B_E` maps emitted gamma-photon density. Thus `rho218` and `rho440` are gamma
emission-density images, not directly 225Ac Bq/mm3. Converting to an activity
rate requires division by acquisition time and the appropriate gamma yield,
plus an explicit daughter/parent kinetic model. Fr and Bi maps are deliberately
allowed to differ spatially in this project.

The standard producer is:

```matlab
run_gen_jscc_production_factors
```

It explicitly selects the PE-v4 Uniform-FOV layer profile. In contrast, the
generic `run_gen_response_factors` defaults to `calibration_profile='none'`.
Use the generic path for raw matrix-physics comparisons only.

## Findings that govern current work

### Polar source measure

Polar samples represent unequal physical volumes, spanning a 6:1 range. Equal
weight at each sample is not a uniform physical source. The complete polar
support is:

```text
r = 0..153 mm
z = -30..30 mm
V = 4412492.545673008 mm3
```

Changing the forward model from integrated-cell activity to density basis
`B=A*diag(DeltaV)` corrected the main displayed center depression in the
Geant4 1e10, 2000-iteration reconstruction:

```text
background center/middle ratio       old basis      density basis
440                                  0.699          1.157
218 cross-talk corrected             0.419          0.890
440 + corrected 218                  0.607          1.010
```

The corrected-sum center bias fell by 97.6%. This is a source-measure fix. The
remaining 440 center overshoot and outer-FOV decline are position-dependent
matrix/Geant4 mismatch, not a reason to discard the density basis.

### System-matrix model

V4 PE generation uses detector-local geometry, visible detector-face surface
integration, reflected symmetric Halton samples, and Geant4-aligned GAGG/W
densities (6.60/19.35 g/cm3). An asymmetric V4-A variant created a directional
x-z artifact and was deleted. The later V5 shared-first-interaction experiment
was also removed because its cost was not justified by an evident gain.

Intrinsic GAGG response studies exist, but intrinsic containment is not yet
applied in production PE v4. Future physics work should first quantify residual
position dependence with independent data, then consider W/GAGG boundaries,
near-neighbor shadowing, and intrinsic containment.

### Calibration discipline

Use a uniform cylinder to fit absolute detector efficiency separately for:

```text
A218:       218 source -> 218 window
A440:       440 source -> 440 window
C440to218:  440 source -> 218 window
```

Start with four detector-layer factors. Do not fit 10496 independent rows
unless an independently reproducible residual map justifies it. The fitted
uniform-cylinder data are calibration data, not independent validation. Use
the contrast phantom and radial point-source scan to validate afterward.

## Geant4 data semantics

`EventAction` independently scans all broadened crystal deposits for 218/440
CntStat windows and classifies accepted two-crystal Compton List events. A
single event can increment multiple CntStat bins and can also produce one List
row. The mixed 218+440 List has no reliable primary-energy label and must never
be used as a pure 218 or pure 440 Compton input.

`gamma01.cc` now creates worker-distinct random seeds from high-resolution
time, PID, and Slurm identifiers. `JSCC_RANDOM_SEED` can set an explicit replay
seed. Recompile Geant4Code before running new simulations.

For density-basis Compton sensitivity, run the two separate macros:

```text
Geant4Sim/Macro/SensiD_UniformFullFOV/UniformFullFOV_218keV.mac
Geant4Sim/Macro/SensiD_UniformFullFOV/UniformFullFOV_440keV.mac
```

They define a GPS-only, uniform physical volume source at `(0,-245,0) mm`,
with radius 153 mm and full height 60 mm. Do not add water/PMMA unless the
matrix uses the identical material model. Sum all worker beamOn values for
`--source-photons`.

## Reconstruction state

The current local dual-energy CntStat-only entry point is
`main_local_multi_energy_cntstat.py`. It reconstructs 440 first, forms the
predicted C440to218 contribution, holds it as fixed additive Poisson background
for the 218 reconstruction, and writes:

```text
Image_S_440keV
Image_S_218keV_Contaminated
Image_S_218keV_CrossTalkCorrected
Image_S_(440_218)keV_CrossTalkCorrected
```

The combined image is `rho440 + rho218_corrected`: a gamma-channel composite,
not a direct 225Ac activity map. `--osem-subset-num 1` selects MLEM.

For Geant4 CntStat, use `--cntstat-dir-suffix _Geant4JSCC` with the canonical
no-suffix Factors. Do not substitute GenProj CntStat for Geant4 validation;
GenProj is the matrix-closed-loop test, while Geant4 is the transport test.

## Next tasks

1. Compile the updated Geant4Code and generate separate 218 and 440 full-FOV
   List data with the supplied macros.
2. Calculate, inspect, and only then install each density-basis `Sensi_d`.
   It must satisfy:
   `sum(Sensi_d)/Vsource = kept_events/Nprimary`.
3. Run Compton-only reconstruction before enabling weighted joint SC+Compton
   reconstruction. Keep List thresholds, detector IDs, first-hit convention,
   energy resolution, and event ordering identical between sensitivity and
   reconstruction.
4. Validate remaining 440 spatial mismatch on independent contrast and radial
   point-source data. Keep raw physics comparisons separate from calibrations.

## Do-not-mix table

| Item | Correct role | Never use as |
| --- | --- | --- |
| Canonical no-suffix JSCC Factors | Density-basis production reconstruction | Old integrated-cell Factors |
| `CntStat/*_Geant4JSCC` | Geant4 reconstruction input | GenProj replacement |
| GenProj CntStat | Fast matrix closed-loop validation | Geant4 transport validation |
| Mixed 218+440 List | Mixed-event diagnostics | Per-energy Sensi_d input |
| `SensitivityPointArray_*` macros | Historical equal-point study | Current density-basis Sensi_d data |
| `SensiD_UniformFullFOV` macros | Current monoenergetic Sensi_d data | 225Ac branching-ratio source |

## Detailed references

| Subject | Document |
| --- | --- |
| Repository/data conventions | `README.md` |
| Polar-volume derivation and evidence | `GenProj/POLAR_SOURCE_MEASURE.md` |
| Matrix version evidence | `Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main/docs/SystemMatrix_Development_History/README.md` |
| Factor generation | `Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main/GenFactors/README.md` |
| Geant4 conventions | `Geant4Sim/README.md`, `Geant4Sim/Geant4Code/README.md` |
| Compton sensitivity | `Auxiliary_Studies/Sensitivity_SPECT_PolarCoor/README.md` |
| Result layout | `Results/README.md`; current runs are under `Results/LocalReconstructionRuns/` |
