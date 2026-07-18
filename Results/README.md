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
|-- Analysis/
|   |-- CNRCRC_JSCC_vs_EHE/
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
