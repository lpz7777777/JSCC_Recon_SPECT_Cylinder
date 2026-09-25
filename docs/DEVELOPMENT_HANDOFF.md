# JSCC Reconstruction Development Handoff

Current consolidated snapshot: 2026-09-25 22:45 China time. For the latest
measured job state, check Slurm; this is a dated handoff, not a live dashboard.
The detailed evidence and per-job history live in
[FOV120 experiment status](FOV120_EXPERIMENT_STATUS.md). Reproducible commands are in
[the experiment README](../experiments/FOV120/README.md), and safe connections
are documented in [remote compute access](REMOTE_COMPUTE_ACCESS.md).

## What has actually been completed

The 60-mm 1e10 baseline is complete (8 GPUs, 1000 iterations, 1,852,124
accepted Compton events); its radial background bias remains unresolved.
For FOV120, the detector remains four layers/10496 active crystals at y=
200/230/260/290 mm. The physical source cylinder is radius 150 mm, height
120 mm; the computational polar support extends to radius 153 mm. There are
40 z layers (-58.5:3:58.5 mm), 1281 polar points per layer (51240 total),
and 20 rotations. Three 218/440/440-to-218 calibrated density-basis Factors
are complete; raw Factors are preserved. Their shared coordinate/rotation/
volume geometry passed validation. The direct 218 and 440 matrices reproduce
the old central 20 layers byte for byte; the cross response differs only at
about 3.5e-8 relative L2. The four calibration groups each accumulated 1e9
primaries. The new 440 `Sensi_d` has independent absolute closure ratio
1.002033 and spatial CV 0.148924; this is not an all-region acceptance result.

Noiseless and Poisson Contrast GenProj closed loops were carried to 10000
iterations. Noiseless rod recovery improves beyond 1000 steps, showing the
former iteration count was insufficient to judge resolution. The Poisson
10000-step images amplify spatial noise sharply (weighted image errors about
188% for 440 and 210% for corrected 218), even though total density integrals
stay near unity. The full-grid truth fixed-point check reaches ~1e-6 maximum
relative update and therefore establishes numerical consistency, not stability
under noise. The 162-position single-view Geant4 point scan and model comparison
are complete; a 20-view point-imaging/FWHM study is not.

Geant4 Uniform and Contrast 1e9 simulations were collected with 20 views,
200 unique workers/seeds and 1e9 actual primaries per dataset. All 46 deployed
files passed size/SHA checks on the scxi717 project. The full-grid, full-event
4-node x 2-GPU six-output 10-step Contrast job 1626513 passed, with 194835
accepted Compton events and peak reserved memory ~9.97 GiB/GPU. It superseded
pending 1626402 and failed 1626482; the latter revealed a missing remote
`distributed/python` dependency, since fixed and checked at startup.
The 10000-step Contrast job 1626525 (10 minutes) and Uniform job 1626560
(13m26s) both completed with exit 0, 200 saved frames per six-output channel.
Local final images, six selected frames and reports are under
`experiments/FOV120/generated/Results/{Contrast_1e9_1626525,Uniform_1e9_1626560}/`;
complete histories remain on scxi717. Generated files are intentionally
Git-ignored. No smoothing, cropping or truth-fit scaling was used in reports.

For Contrast, weighted relative L2 at 1000→10000 iterations is
0.450→1.884 (440 single), 1.983→7.965 (440 Compton), 0.481→2.145
(440 JSCC), and 0.406→2.136 (218 corrected). For Uniform, the corresponding
figures are 0.198→2.119, 3.177→9.284, 0.411→2.634, and 0.234→2.276.
The 1e9 long-iteration images are spatially blotchy despite near-unit integrated
counts. This does not isolate Poisson noise from response mismatch; high-count
comparison is needed. The nominal 120-mm grid is not yet an accepted useful FOV.

Uniform 1e9 axial analysis uses raw polar density, cell-volume weighting and
r<=135 mm. Direct 440 sensitivity in the bottom/top z layer is ~93.5% of the
central value; Compton sensitivity ~96.8%/~96.4%. At 1000 steps the 440
single-photon center/edge CV is 0.163/0.165, at 10000 it is 2.109/2.104.
The top z=58.5-mm layer has 1.42 mean recovery and 3.42 relative L2 at 10000,
but interior layers also spike. Edge loss exists; it is not the only driver.
See `AxialEdgeReport/` for every layer, region metrics, and the figure.

## Physics and channel semantics that must remain consistent

The actual maty `Geant4Sim/Geant4Code` applies a single Gaussian energy draw
per crystal with deposited energy at event end: 13% relative FWHM at 511 keV,
scaled as 1/sqrt(E). This gives 19.903% at 218 keV and 14.010% at 440 keV;
their +/-half-FWHM windows are 196.305–239.695 and 409.179–470.821 keV.
The broadened values determine CntStat and are written to List. Reconstruction
sets `input_energies_already_smeared=True`, so it does not draw a second noise
sample; the same resolution still sets the Compton cone-angle uncertainty.
Single-photon Factors integrate Gaussian energy-window acceptance analytically;
440-to-218 uses the 440-source forced-218-window Scatter matrix.

`Image_440_SinglePlusCompton` is a true JSCC MLEM update of one 440 image:
backprojections from single and Compton data are combined inside each iteration
and divided by `Sensi_s + Sensi_d`. It is not a sum of separately reconstructed
440 images. Only `Image_440SinglePlus218Single` and
`Image_440SingleComptonPlus218Single` are post-reconstruction sums with the
218 cross-talk-corrected image, and they are gamma-channel composites, not
parent-225Ac activity maps. The 218 update uses a fixed predicted 440-to-218
additive Poisson background. In the two 1e9 reconstructions, accepted Compton
counts are 194835/194261 versus 440 single-photon counts 2040723/2041123:
Compton is ~9.5% of the 440 singles, ~8.7% of those two observation counts
combined. Count fraction is not a JSCC information weight. CntStat and List
may include correlated observations from the same primary.

## Current jobs and remaining acceptance work

On maty, Uniform 1e10 array 15388423 (indices 762–961) and Contrast 1e10
array 15388444 (1162–1361) each comprise 200 workers x 5e7 primaries over
20 views. At the 22:45 snapshot each had 20 running workers; afterok collector
15388466 was pending. Do not treat submission as complete simulation. Once
both collections pass PrimaryCount, seeds, hashes and view coverage, transfer
them to the scxi717 project, measure accepted event count, and preflight GPU
memory before 10000-step six-output high-count reconstruction. This comparison
is needed to distinguish finite-count noise from model discrepancy. Full XCAT
production imaging and old 60-mm data reconstructed on both support grids are
still pending. XCAT's 120-mm crop retains ~81.91% of the full kidney label;
both axial crop boundaries touch kidneys, while the lesion ROI remains inside.
The scientific acceptance must report center/middle/edge metrics, point-source
localization and resolution, background uniformity and lesion/organ recovery.

The rest of this document retains earlier design decisions, commands and
snapshot history. Any older statement that a now-completed job is running or
that no FOV120 image exists is a historical statement; use this section and
Slurm completion records for present status.

## Directory map

```text
repository root
|-- Factors/                         canonical/historical matrix packages (ignored)
|-- CntStat/                         projection data (ignored)
|-- List/                            Compton event data (ignored)
|-- GenProj/                         MATLAB forward projection and source measure
|-- Geant4Sim/                       Monte Carlo macros, C++ codes, run collection
|   |-- Geant4Code/                  JSCC CntStat + List production simulation
|   |-- Geant4Code_EHE/              EHE CntStat-only simulation
|   |-- Geant4Code_CntStatOnly/      CntStat-only diagnostics
|   |-- Geant4Code_CntStatResponseStudy/ independent response diagnostics
|   |-- Geant4Code_GAGGIntrinsicResponse/ single-crystal containment study
|   |-- Macro/                       generated source macros, including SensiD sources
|   `-- run/                         collected Geant4 worker output (ignored)
|-- Auxiliary_Studies/               independent research projects
|   |-- GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main/
|   |                                   PE/Compton CUDA matrix generation and Factors
|   |-- Sensitivity_SPECT_PolarCoor/  Compton Sensi_d calculation
|   |-- ComptonSystemMatrixPrototype/ Compton prototype work
|   |-- EventOrderInference_Experiment/ interaction-order experiments
|   |-- CRCVAR_SinglePhoton/          CRC variance studies
|   |-- FreePath/                     free-path studies
|   `-- Reference/                    reference material and figures
|-- distributed/                      distributed reconstruction
|   |-- dual_energy_compton_python/   current six-output distributed Python
|   `-- dual_energy_compton_slurm/    current smoke/production/monitor scripts
|-- Reconstruction/                   reproduction-oriented reconstruction docs/code
|-- Reproduction/                     compact end-to-end instructions
|-- Results/                          local outputs (ignored)
|   `-- LocalReconstructionRuns/      named PE-v4 local reconstruction runs
|-- docs/                             durable cross-project documentation
`-- tests/                            repository-level tests
```

For any directory not listed as a production entry point, inspect its local
README before using it. Directories such as `tmp/`, `Geant4Sim/run/`,
`Factors/`, `CntStat/`, `List/`, and `Results/` are generated-data locations,
not source-code starting points.

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

The current Compton validation response is intentionally frozen at:

```text
shared K*B density-basis event response
13% FWHM at 511 keV
E1 + E2 >= 350 keV
input List energies already broadened by Geant4
theta_stride = 1, z_stride = 1
Factors/440keV_RotateNum20/Sensi_d
```

For higher count levels, use the isolated multi-node/multi-GPU implementation
in `distributed/dual_energy_compton_python/` and the launchers in
`distributed/dual_energy_compton_slurm/`. The distributed implementation keeps
the local six-output mathematics unchanged: detector bins and List lines are
partitioned, rank-local backprojections are combined with NCCL `all_reduce`,
and only rank 0 writes the final images and histories. The older supplied
4-node x 8-GPU template belongs to the earlier high-count configuration.
FOV120 1e9 actually ran on 4 nodes x 2 GPUs. Recheck the available topology,
accepted events and measured GPU memory before choosing FOV120 1e10 ranks.

## Next tasks

The old 60-mm high-count case and FOV120 1e9 Uniform/Contrast six-output cases
are fulfilled. The active gate is completion and verified collection of the two
FOV120 1e10 Geant4 arrays, followed by high-count reconstruction and full-height
metrics. The independent 20-view point-imaging study, original-data support-grid
regression and XCAT production imaging remain. Preserve the frozen K*B response
and source/volume conventions; quantify edge performance and CntStat/List
correlation before interpreting images as quantitative activity.

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
