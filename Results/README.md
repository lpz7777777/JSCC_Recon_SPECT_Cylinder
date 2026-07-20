# Local Results Layout

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
|   |-- V4L/                                  # PE v4 absolute Uniform-FOV layer reconstruction
|   `-- V4PV/                                 # V4L with A*diag(polar-cell volume), density image
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
