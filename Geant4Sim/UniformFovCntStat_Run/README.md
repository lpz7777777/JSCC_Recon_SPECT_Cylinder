# Uniform-FOV Run Scripts

These scripts intentionally follow the old `run_old/RunData` workflow:

1. prepare one clean run root for 218 keV and one for 440 keV;
2. enter each energy directory and run `./RunData`;
3. `RunData` creates numeric worker directories `1..100` and calls `sbatch`;
4. collect the completed directories with `bingxing_UniformFovCntStat.m`.

The two energy jobs are independent and may be submitted at the same time.
Do not reuse an existing energy directory because Geant4 appends CSV output.

Expected server layout:

```text
Gean4Sim_CntStatResponseStudy/
|-- Geant4Code_CntStatOnly/
|-- Macro/UniformFovCntStat_CenterPoint/
|-- UniformFovCntStat_Run/
`-- run/
```

After compiling `Geant4Code_CntStatOnly/build/gamma01`, prepare both energy
roots and submit them independently:

```bash
ROOT=/path/to/Gean4Sim_CntStatResponseStudy
cd "$ROOT"
./UniformFovCntStat_Run/prepare_uniform_fov_runs.sh

cd "$ROOT/run/UniformFovCntStat_CenterPoint/218keV"
./RunData |& tee submit.log

cd "$ROOT/run/UniformFovCntStat_CenterPoint/440keV"
./RunData |& tee submit.log
```

Each energy directory then has the familiar numeric `1..100` workers. After
copying both completed energy directories back to the local repository, merge
them in MATLAB:

```matlab
repo = 'F:\path\to\repository';
addpath(fullfile(repo, 'Geant4Sim'));
runRoot = fullfile(repo, 'Geant4Sim', 'run', ...
    'UniformFovCntStat_CenterPoint');
outputRoot = fullfile(repo, 'Geant4Sim', 'run', ...
    'merged_UniformFovCntStat');
summaries = bingxing_UniformFovCntStat(runRoot, outputRoot, 1:100);
```

The merge validates one row, 10496 bins, nonnegative integer counts, and one
positive primary count for every worker.

On Linux, normalize transferred script line endings before use:

```bash
sed -i 's/\r$//' RunData runevent prepare_uniform_fov_runs.sh
chmod u+x RunData runevent prepare_uniform_fov_runs.sh
```
