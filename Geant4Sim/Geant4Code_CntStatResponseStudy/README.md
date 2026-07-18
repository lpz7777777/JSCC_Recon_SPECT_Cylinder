# JSCC mixed-energy CntStat response study

## Purpose

This directory is a diagnostic copy of `../Geant4Code`. It is intended to
measure the 218-keV-window response from known 218 and 440 keV primaries in
one mixed run. It keeps the production JSCC geometry, 10496 detector-bin
ordering, energy broadening, total CntStat outputs, and Compton `List.csv`.
The source-separated files are research instrumentation and are not intended
for later high-statistics production runs.

The diagnostic answers a specific question: whether the measured mixed-run
218-keV projection can be decomposed as the direct 218 response plus the
440-to-218 cross-window response under exactly the same Geant4 physics and
geometry.

## Step energy-deposit attribution

`G4Step::GetTotalEnergyDeposit()` is the local energy deposited during the
whole step. It is not an energy value sampled specifically at the start or
end point. The step belongs to the pre-step physical volume, so detector copy
number lookup uses `GetPreStepPoint()->GetTouchableHandle()`. At a geometry
boundary, the post-step touchable may already identify the next volume and
can assign a deposit to the wrong detector bin.

## Event and primary classification

At `BeginOfEventAction`, the code reads the first primary particle from the
first primary vertex and classifies its kinetic energy as 218 keV, 440 keV,
or other. The matching tolerance is 1 keV. Exactly one primary counter is
incremented per Geant4 event.

At event end, deposited energy is broadened independently in every crystal,
then all 10496 crystals are scanned. A single event can therefore increment
more than one detector bin. CntStat classification and the retained Compton
List classification are independent.

The source-separated rules are:

```text
218 primary + crystal in 218 window -> CntStat218_From218
440 primary + crystal in 218 window -> CntStat218_From440
440 primary + crystal in 440 window -> CntStat440_From440
```

## Outputs

Every `/run/beamOn` appends one row to each CntStat/count file in the current
working directory. Existing files are not cleared automatically.

```text
CntStat_218.csv             total 218-window counts, 10496 columns
CntStat_440.csv             total 440-window counts, 10496 columns
CntStat218_From218.csv      218-window counts caused by 218 primaries
CntStat218_From440.csv      218-window counts caused by 440 primaries
CntStat440_From440.csv      440-window counts caused by 440 primaries
PrimaryCount218.csv         number of 218 primary events, one integer
PrimaryCount440.csv         number of 440 primary events, one integer
PrimaryCountOther.csv       unrecognized primary events, normally zero
List.csv                    retained production Compton diagnostic
```

The response-study build also writes topology-separated 440-to-218 counts:

```text
CntStat218_From440_FirstCrystal.csv
CntStat218_From440_OtherCrystal.csv
CntStat218_From440_Hit1.csv
CntStat218_From440_Hit2.csv
CntStat218_From440_Hit3Plus.csv
CntStat218_From440_FirstCrystal_Compton0.csv
CntStat218_From440_FirstCrystal_Compton1.csv
CntStat218_From440_FirstCrystal_Compton2Plus.csv
```

`FirstCrystal` is the first scintillator in which the primary track deposits
energy; `OtherCrystal` contains all other accepted bins. `Hit1/2/3Plus` uses
the number of crystals with more than 1 keV unbroadened deposit in the event.
The Compton categories count primary-track Compton steps in the first crystal.
They are diagnostic labels, not mutually exclusive Geant4 process filters for
the rest of the event. The following identities must close bin by bin:

```text
From440 = FirstCrystal + OtherCrystal
From440 = Hit1 + Hit2 + Hit3Plus
FirstCrystal = FirstCrystal_Compton0
             + FirstCrystal_Compton1
             + FirstCrystal_Compton2Plus
```

For a valid 218/440-only run, bin by bin and in total:

```text
CntStat_218 = CntStat218_From218 + CntStat218_From440
CntStat_440 ~= CntStat440_From440
PrimaryCount218 + PrimaryCount440 = number of events
PrimaryCountOther = 0
```

The second relation uses `~=` because this study does not request a separate
`CntStat440_From218` file. A 218 keV primary cannot physically deposit 440
keV, apart from a vanishing numerical tail introduced by the Gaussian energy
broadening model.

When several independent worker processes simulate one view, give each
worker its own output directory. Sum corresponding CntStat rows and primary
counts after all workers finish; never let workers append concurrently to the
same CSV files.

Merge numeric worker directories with the project-level MATLAB collector:

```matlab
summary = bingxing_CntStatResponseStudy_Topology( ...
    '/path/to/worker_root', 1:100);
```

The default output is `worker_root/merged_CntStatResponseStudy/`. The collector
requires every existing worker directory to contain all standard and topology
CSV files, validates 10496 detector bins, sums every appended row, checks all
three topology identities and the mixed 218 identity per worker and globally,
and streams `List.csv` without loading all List rows into memory. Missing
numeric directories are reported but skipped; an existing incomplete worker
directory is a hard error.

## Center-point example

`center_point_mixed.mac` places both source components at the current FOV
center `(0, -245, 0) mm`, emits isotropically, and uses whole-run expected
weights `218:440 = 0.114:0.261`. Its default run has one million events. Edit
only `/run/beamOn` to change statistics.

Build and run on a configured Geant4 server:

```bash
mkdir build
cd build
cmake -DWITH_GEANT4_UIVIS=OFF ..
cmake --build . --config Release -j
./gamma01 center_point_mixed.mac
```

For the topology study, prefer the 440-only macro so every event contributes
to the response being diagnosed:

```bash
./gamma01 center_point_440_only.mac
```

Its default is `1e8` events, which is sufficient for total and layer topology
fractions. Use `1e9` when detector-bin spatial comparisons are required. Run
in a clean output directory because all CSV writers append rather than
truncate existing files.

For multi-configuration generators, the executable may be under the selected
configuration directory. Run from the build directory so `CrystalMatrix.txt`
and the copied macro are available.

## Radial point-response scan for the central reconstruction artifact

`../GenerateRadialPointResponseMacros.m` generates the next diagnostic
experiment. It does not change the C++ code: this response-study executable
already separates the required pure-primary channels. The generated macros use
the same source-plane convention as the contrast-phantom macro:

```text
Factor coordinate (x, y, z) mm -> GPS coordinate (x, y - 245, z) mm
```

The default set contains every polar-Factor radius, `r=0,6,...,150 mm`. At
every nonzero radius it samples azimuths `0,90,180,270 deg`; the center is
emitted once. This gives `1 + 25*4 = 101` source positions and 202 pure-energy
response runs. It is a full-FOV **radial** scan, not an all-angle scan of all
1281 polar locations. Pure 218 gives `CntStat218_From218`; pure 440 gives both
`CntStat440_From440` and `CntStat218_From440`. These are exactly the three
responses needed to compare Geant4 with `A218`, `A440`, and `C440to218`.

Generate the macros from the project root:

```matlab
run("Geant4Sim/GenerateRadialPointResponseMacros.m")
```

They are written under `Geant4Sim/Macro/RadialPointResponse_JSCC/`, with a
machine-readable `radial_point_manifest.csv`. The production configuration is
100 workers and `1e6` primary photons per configuration per worker. First
make a small smoke run by changing `cfg.events_per_worker_configuration` to
`1e4`. The merged target is `202 x 100 x 1e6 = 2.02e10` primaries, or `1e8`
per source-energy configuration.

The recommended production route is 100 independent worker directories. Each
worker runs `radial_point_response_full_fov_worker.mac`, which sequentially
runs all 202 configurations. Each `/run/beamOn` appends one row, so every
worker has the same manifest row order. Merge with
`../bingxing_CntStatResponseStudy_Rows.m`; it sums matching rows across
workers while retaining all 202 rows. Do not use the older topology collector
for this scan because it intentionally collapses all rows into one response.

For example, on a Linux server after compiling this project:

```bash
mkdir -p results/radial_point_response
cp build/gamma01 results/radial_point_response/
cp Geant4Code_CntStatResponseStudy/CrystalMatrix.txt results/radial_point_response/
cp Macro/RadialPointResponse_JSCC/radial_point_response_full_fov_worker.mac results/radial_point_response/
cp Macro/RadialPointResponse_JSCC/radial_point_manifest.csv results/radial_point_response/
cd results/radial_point_response
./gamma01 radial_point_response_full_fov_worker.mac
```

Run the worker macro only once per numeric worker directory: rerunning appends
another 202 rows. Retain `PrimaryCount218.csv`, `PrimaryCount440.csv`, and the
source-separated CntStat files. The first quantitative comparison should use
response totals and the four detector-depth sums versus the matching Factor
column. Only then use per-detector residuals, because they require much higher
count statistics.

The 202 individual macros remain available for a later targeted campaign or a
targeted rerun. They are not required for the primary radial scan.

## Full-FOV radial result and random-seed warning (2026-07-17)

The merged 100-worker scan contains 202 valid manifest rows and 10496 detector
bins per row. Every row has exactly `1e8` recognized primaries,
`PrimaryCountOther=0`, and the source-separated CntStat identities close
exactly. The calibrated center-point Factors were compared at physical
`z=0 mm` by averaging their `z=-1.5` and `z=+1.5 mm` columns.

Across `r=0:6:150 mm` and four cardinal azimuths, total Geant4/Factor response
is:

```text
response       mean       minimum      maximum
A218         0.99996      0.98201      1.01496
A440         1.00214      0.97142      1.03522
C440to218    1.00852      0.96479      1.03938
```

There is no center-specific efficiency deficit and no monotonic radial drift.
Four detector-depth sums are also centered near one, with larger fluctuations
for low-count A440 and cross-window rows. Therefore an empirical radial scalar
`D_voxel(r)` is not supported as the explanation for the reconstruction's
central low-value region.

Detector-bin residuals cannot yet be interpreted as matrix shape mismatch.
All three responses exceed the independent-Poisson relative-L2 expectation by
approximately `5.44x`, nearly independent of radius and response. This implies
only `100 / 5.44^2 = 3.37` effective independent workers. The data were
generated by the historical `time(NULL)` seed implementation, so Slurm array
jobs starting in the same second replayed identical random histories. Total
and layer sums remain apparently stable while bin-level noise is not reduced
as expected.

The seed initialization is now fixed. By default `gamma01.cc` mixes a
high-resolution clock, process PID, `SLURM_JOB_ID`, `SLURM_ARRAY_JOB_ID`, and
`SLURM_ARRAY_TASK_ID`, then maps the result into the positive Ranecu seed
range. Different nodes may reuse PIDs, so PID is deliberately not the only
identifier. `JSCC_RANDOM_SEED` can explicitly override the automatic seed for
reproducible runs. Every launch logs the final seed and all identifiers.
Recompile `Geant4Code_CntStatResponseStudy` before rerunning. The existing
merged scan remains valid for total/radial/layer conclusions but not for a
high-precision detector-bin residual decision.

Reproduce the analysis with:

```bash
python Geant4Sim/Geant4Code_CntStatResponseStudy/analyze_radial_point_response.py
```

It writes detailed and radius-averaged CSV files, JSON summary, total-response
curves, layer-response curves, and the bin-level overdispersion plot under
`Geant4Sim/run/merged_RadialPointResponse/radial_analysis/`.

## Deferred system-matrix work

The following changes are recorded for the next phase and are deliberately
not implemented in this diagnostic Geant4 project:

1. Rewrite ScatterGen inter-crystal energy-window integration. Integrate the
   Klein-Nishina weight, angle-dependent scattered energy, energy-window
   acceptance, attenuation, and target-crystal absorption over target solid
   angle. Do not evaluate the energy acceptance only at the target center and
   apply it over a broad angular interval.
2. For nearby crystals, replace the current close-target approximation with
   exact box/ray intersection or target-surface subelement integration. This
   must account for the large angular variation across a nearby target.
3. Stop estimating the target azimuth range with an enclosing sphere. Use the
   actual rectangular crystal geometry or converged surface/solid-angle
   quadrature.
4. Export separate diagnostic response components such as
   `C_local_recoil`, `C_intercrystal`, `C_highZ_to_crystal`, and `C_total`, so
   future Geant4 comparisons can localize disagreement before matrices are
   combined.

The production implementation should be validated first with the center-point
mixed run from this directory, then with off-center points and near/far crystal
pairs before regenerating full Factors.

## 1e9 center-point result

The mixed `center_point_mixed.mac` run completed with:

```text
PrimaryCount218 = 304029286
PrimaryCount440 = 695970714
PrimaryCountOther = 0
actual 440/218 ratio = 2.28915682
```

Both output identities close exactly in every detector bin. Per emitted
primary, the measured Geant4 probabilities are:

```text
CntStat218_From218 = 0.00983544
CntStat218_From440 = 0.00183711
CntStat440_From440 = 0.00257049
```

For comparison with the Cartesian matrices, FOV center is the average of the
two central Z columns at `z=-1.5` and `z=+1.5 mm`; X/Y use their exact center
indices. Detector rows are filtered by `Params_Detector flag==1`, which gives
10496 rows in exactly the Geant4/Factors order.

The matrices generated before the target-surface ScatterGen correction give:

```text
response             matrix / Geant4
218 direct                 1.264
440 direct                 1.299
440 -> 218 cross           1.903
```

The cross-window layer ratios localize the dominant error:

```text
detector y (mm)       30      60      90      120
matrix / Geant4     0.938   0.975   0.978    2.308
```

Thus the first three layers already agree in total cross-window probability,
while the finely divided last layer is overestimated by 2.31 times. This is
the expected signature of the old nearby-target bounding-sphere/theta-range
approximation, not a detector-order or primary-branching mismatch.

Reproduce the report after any new center-point run or regenerated matrix:

```bash
python analyze_center_point_response.py
python analyze_440_to_218_topology.py
```

Outputs are written under `build/analysis/`:

```text
center_point_response_summary.json
center_point_detector_comparison.csv
center_point_response_comparison.png
```

The topology analysis writes `topology_summary.json` and
`topology_component_comparison.png` under the selected build directory's
`topology_analysis/`. It also writes an interpretation report and a
per-detector comparison table:

```text
topology_analysis_report.md
topology_detector_comparison.csv
```

For merged parallel output, run:

```bash
python analyze_440_to_218_topology.py \
  --build-dir build/merged_CntStatResponseStudy
```

The analyzer accepts the trailing comma used by the Geant4/merger scalar
counter files, checks the topology identities on integer counts before
normalization, and handles the 1-based crystal IDs stored in `List.csv`.

## Target-surface validation result

The full matrix generated with exact target-box surface integration passes all
size, finite/nonnegative, component-closure, and combined-matrix checks. Its
440-to-218 center response is:

```text
Geant4 probability per 440 primary    1.83711e-3
new matrix probability                1.43294e-3
matrix / Geant4                       0.779996
```

Layer ratios changed from the old `0.938, 0.975, 0.978, 2.308` to
`0.811, 0.796, 0.752, 0.778`. The old last-layer over-count is removed, but a
roughly common 22% low normalization remains. Far-face `1/2/4` subdivision
tests change the result by less than 1%, and pair traversal excludes source
and target crystals from intermediate attenuation.

The new matrix center response consists of:

```text
C_local_recoil          1.01485e-3    70.82%
C_intercrystal          4.08881e-4    28.53%
C_highZ_to_crystal      9.21098e-6     0.64%
```

After detector-plane smoothing, Geant4 and matrix spatial correlation becomes
high, so the corrected target-surface model captures the broad spatial shape.
The old component fit was not an event-topology match and must not be
interpreted as evidence that both first-order components need a common scale.

The retained `List.csv` cannot separate this residual: its strict two-crystal,
one-first-crystal-Compton, near-full-energy 440 subset accounts for only about
36.4% of all measured 440-to-218 detector-bin counts. Rebuild and rerun this
diagnostic project to generate the topology-separated files, then run:

```bash
python analyze_440_to_218_topology.py
```

Do not apply an empirical `1/0.78` scale to production matrices until the
first-crystal/other-crystal and hit-multiplicity outputs identify how much of
the residual comes from multiple Compton histories. Ideal analytical PE also
does not model Geant4 fluorescence and secondary-particle escape, which is
relevant to the direct 218 and 440 photopeak comparisons.

## New 1e9 topology result (2026-07-16)

The merged 100-worker mixed-source run contains exactly `1e9` primary events:

```text
PrimaryCount218                         304056693
PrimaryCount440                         695943307
PrimaryCountOther                               0
440-to-218 accepted detector-bin counts   1277527
```

All four integer closure checks pass exactly. Per 440 primary:

```text
Geant4 total                            1.835676825e-3
matrix total                            1.432938807e-3
matrix / Geant4                              0.780605

Geant4 Hit1                             6.037661341e-4   32.89%
Geant4 Hit2                             8.779077173e-4   47.82%
Geant4 Hit3Plus                         3.540029734e-4   19.28%
```

The new total ratio reproduces the previous `0.779996` result within `0.08%`.
The `Hit3Plus` response alone is `87.9%` as large as the total matrix deficit.
This is strong evidence that multiple-crystal and higher-order histories are
the dominant missing physics. It is a magnitude comparison rather than an
exclusive accounting identity: a local first-crystal count in a Geant4
three-hit event can still be represented by the analytical local term.

The provisional local comparison is:

```text
C_local_recoil / FirstCrystal_Compton1 = 0.958658
```

Its layer ratios are `0.918, 0.954, 0.911, 0.977`, and its detector-plane
shape correlation after four-pitch box smoothing is `0.967`. This is much
closer than the unpartitioned total comparison and does not support applying a
common `1/0.78` normalization to every first-order component.

There is one important diagnostic limitation. `SteppingAction.cc` currently
sets `Scin_CopyNum` and increments `NumCompt` only in a branch where the
primary-gamma step also has `GetTotalEnergyDeposit() > 0`. A Geant4 Compton
step can transfer energy to a tracked secondary without depositing that
energy locally on the primary step. Consequently `FirstCrystal`,
`OtherCrystal`, the Compton subcategories, and the strict `List.csv` subset
are approximate process labels. The energy-window CntStat totals and
`Hit1/Hit2/Hit3Plus` categories do not have this process-label limitation.
Before using the provisional local/intercrystal ratios to change ScatterGen,
the response-study project should identify primary discrete processes
independently of local energy deposit and then rerun a 440-only point source.
