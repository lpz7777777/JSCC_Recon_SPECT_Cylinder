# Local Results Layout

For the interpretation of the calibrated PE-v4 and PolarVolume-density runs and
the current validation status, see
`../docs/DEVELOPMENT_HANDOFF.md`.

`Results/` contains generated reconstruction, analysis, and historical output.
The contents are intentionally ignored by Git; only this guide is tracked.

```text
Results/
|-- Reconstruction/
|   |-- Figure_Local_SC_MultiOutput/          # JSCC GenProj reconstructions
|   |-- Figure_Local_SC_MultiOutput_EHE/      # EHE reconstructions
|   |-- Figure_Local_SC_MultiOutput_Geant4JSCC/
|   `-- Geant4JSCC_FactorCalibration_20260718/
|       |-- Baseline_CenterPoint/
|       |-- FOVLayerTP/
|       |-- FOVDetectorTP/
|       `-- Comparison/
|-- R/
|-- LocalReconstructionRuns/
|   |-- PEv4_UniformFOVLayer_Calibrated/     # calibrated PE-v4 integrated-cell result
|   `-- PEv4_UniformFOVLayer_PolarVolumeDensity_Calibrated/ # canonical density result
|       `-- JSCC_Rotate20_E218_440_Count1e10_MLEM2000_OSEM1_CrossTalkCorrected/
|-- Analysis/
|   |-- CNRCRC_JSCC_vs_EHE/
|   |-- PEV4ReferenceValidation_*/              # Selected-pair PE v4 convergence
|   |-- PEV4GPUValidation_*/                    # GPU/CPU production checks
|   |-- PEV4FullMatrixDirectComparison_*/       # Full raw PE v4/v3 totals
|   |-- UniformFov_PEv3_vs_PEv4/                # Legacy asymmetric-v4 comparison
|   |-- UniformFov_PEv3_vs_PEv4_SymmetricHalton/
|   |-- UniformFov_PEv4_LayerCorrectionValidation/
|   |-- SM_Physics_v4_20260718/                 # Retained V4 symmetry and W/GAGG path audits
|   |-- PolarSourceMeasure_20260720/             # Grid/source-measure and CntStat comparison
|   |-- PolarSourceMeasure_20260720_RawV4S/      # Same comparison with uncalibrated V4-S
|   |-- PolarVolumeRecon_20260720/               # 1e10 old-vs-density reconstruction analysis
|   `-- ReferenceImages/                      # Generated Cartesian references
|-- Logs/
`-- Legacy/
    |-- Figure/
    |-- Figure_Dist_SC/
    `-- Figure_Dist_JSCCSD/
```

Large source data remain under `CntStat/`, `List/`, `Factors/`, and
`Geant4Sim/run/`. System-matrix calculation output remains under the matrix
project's `runs/` directory.

## 2026-09-24: dual-energy 60-mm baseline and FOV120

`Reconstruction/Distributed_JSCC_ComptonValidation_Geant4_1e10_Iter1000_1node8gpu/`
contains the completed six-output 60-mm experiment, its manifest, predicted
440-to-218 counts, saved iterations and radial-bias diagnostics. It is not a
FOV120 result. See [the current inventory](../docs/FOV120_EXPERIMENT_STATUS.md)
for measured background ratios and interpretation limits.

FOV120 source truth, macro validation and smoke outputs are under
`../experiments/FOV120/generated/`; remote workers are cataloged in that
experiment's status documents. Real FOV120 Geant4 Contrast and Uniform 1e9 six-output 10000-iteration
reconstructions have now completed (jobs 1626525 and 1626560). Their local
final arrays, selected-frame figures and axial metrics live under
`../experiments/FOV120/generated/Results/`, not this legacy `Results/` root.
The reports are full-height, unfiltered and uncropped; complete 200-frame
histories remain on scxi717. Both images have severe long-iteration spatial
noise, so completion does not establish useful 120-mm FOV. Keep the old
central-39-mm pictures separate from these full-height comparisons. Large
arrays and generated images are not committed.
