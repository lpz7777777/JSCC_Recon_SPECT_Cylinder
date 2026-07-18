# JSCC Uniform-FOV CntStat-Only Study

This project is a diagnostic derivative of `Geant4Code`. It preserves the
production JSCC detector geometry, physics list, energy broadening, energy
windows, and 10496-bin detector order. It deliberately removes Compton `List`
classification and writes only detector-window counts.

The intended experiment uses all 25620 positions in the CenterPoint polar
grid with equal GPS source intensity. Pure 218 keV and pure 440 keV are run as
separate batches. This measures the detector-row response averaged over the
full image grid:

- pure 218: `CntStat_218.csv` is compared with the row sum of `A218`;
- pure 440: `CntStat_440.csv` is compared with the row sum of `A440`;
- pure 440: `CntStat_218.csv` is compared with the row sum of `C440to218`.

Each event samples exactly one GPS source position. The source positions all
have equal intensity, and the macro has one final `/run/beamOn` command.

## Outputs

Each fresh worker directory receives exactly one row in:

```text
CntStat_218.csv   1 x 10496
CntStat_440.csv   1 x 10496
PrimaryCount.csv  1 x 1
```

No `List.csv`, topology output, energy spectrum, or event-type file is
produced. Outputs use append mode, so never rerun in a worker directory that
already contains output CSV files.

## Random seeds

The automatic seed mixes high-resolution time, process ID, `SLURM_JOB_ID`,
`SLURM_ARRAY_JOB_ID`, and `SLURM_ARRAY_TASK_ID`. Every process logs:

```text
Random seed initialization: seed=..., pid=..., SLURM_JOB_ID=...
```

An explicit positive seed in `1..2147483562` can be supplied through
`JSCC_RANDOM_SEED`. Do not set the same explicit value for parallel workers.

## Linux build

```bash
mkdir build
cd build
module load compilers/gcc/v12.2.0
export Geant4_11_1_path=/apps/soft/geant/geant4-v11.1.0
export Geant4_DIR=$Geant4_11_1_path/lib64/Geant4-11.1.0
source $Geant4_11_1_path/bin/geant4.sh
module load tools/cmake/v3.25.2
cmake -DWITH_GEANT4_UIVIS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_COMPILER=g++ ..
make -j"$(nproc)"
```

Before production, confirm the new seed code is in the executable:

```bash
strings gamma01 | grep "Random seed initialization"
```

Use the scripts under `../UniformFovCntStat_Run` to prepare and submit the two
100-worker batches. Use `../bingxing_UniformFovCntStat.m` to merge results.
