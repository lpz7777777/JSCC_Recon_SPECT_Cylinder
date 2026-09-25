# JSCC Reconstruction Development Handoff

Last consolidated: 2026-09-25.

## Current handoff: FOV120 and the completed 60-mm baseline

Read [FOV120 experiment status](FOV120_EXPERIMENT_STATUS.md) for the detailed
code/data inventory, verified results, remote job IDs and remaining acceptance
steps. [Remote compute access](REMOTE_COMPUTE_ACCESS.md) is the reusable safe
connection guide for scxi717, maty and 65114; it contains no credentials.
[Experiment commands](../experiments/FOV120/README.md) cover reproducible runs.

The original 60-mm 1e10/8-GPU/1000-iteration reconstruction is complete, with
1,852,124 accepted Compton events and six final outputs. Radial bias remains;
completed reconstruction is not quantitative acceptance. All FOV120 raw response
matrices and raw polar Factors are now complete and validated. The four 1e8
pilot groups and extension array 15384175 are complete. Collection job 15384216
passed: four full 1e9 datasets, 400 workers and 400 distinct seeds. The archive
was hash-verified locally and on 65114. Production calibrated Factors are now
installed on 65114, raw Factors preserved; layer relative SE is 0.0420–0.21694%.
The new Sensi_d and independent closure completed on 65114 GPU 0 (exit 0):
281816 accepted calibration events, absolute mean 2.818160e-4; independent
closure volume-weighted ratio 1.002033 and CV 0.148924%. Sensi_d and hashed
provenance are installed in production Factors. Contrast 1e9
truth and noiseless/Poisson projections exist; reconstruction is still pending.
See the status page for coefficients, archive hash and artifact locations.

NCCL smoke 1624002 passed on two RTX5090 GPUs, including serial-reference
single/Compton/joint MLEM equivalence. Original job 1623854 failed before
numerical execution and was superseded. Two-node NCCL test 1625684 also passed (36 seconds, one GPU on each of
two distinct nodes); full-grid memory tests and actual FOV120 images remain pending.
Noiseless/Poisson closed loops completed 1000 iterations each (~469 seconds).
Noiseless projection residuals are 0.207%/0.236% (218/440), but volume-weighted
image errors remain 36.6%/42.0%; spatial quality is not accepted. All histories
are verified. The production Factors archive is deployed to scxi717; remote SHA256, full
geometry/matrix scan and Sensi_d provenance verification all passed. A Python
3.9 file_digest incompatibility was fixed with streaming SHA256 and a tamper test. The earlier
65114 SSH failures resolved after download; do not relaunch completed loops. Reconstruction work belongs inside the
user-specified main project, under its `experiments/FOV120_20260924/` workspace.
See the status document for full paths and refresh before resubmitting tasks.

Full-grid noiseless truth fixed-point checks now pass (~1e-6 maximum voxel change)
using `experiments/FOV120/diagnose_closedloop.py`. Fractional-volume rod CRC at
1000 iterations is only 1.6–7.4%; spatial quality remains unresolved. Maty smoke
15385867 precedes dependent single-view point-response array 15385868 (162
workers x 1e7, 40 concurrent). Do not confuse it with 20-view PointImaging.

Latest: point scan 15385868 is fully complete, 162 workers / 162 unique seeds /
1.62e9 photons. Verified full counts and MC/model comparisons are local in
`generated/FullData/point_*all*`; snapshot1 remains historical. A 14-figure full-height HTML report is
`generated/ClosedLoop_VisualReport/index.html`, with CSV metrics and source hashes.
Uniform/Contrast 1e9 jobs 15386226/15386227 are running, 20 workers concurrent
each. Afterok collector 15386284 waits for both arrays to complete successfully.
The maty launcher now maps zero-based array indices via a validated offset
to work around MaxArraySize=1001; physical tasks/seeds were not regenerated.

User requested at least 10000 iterations before judging hot-rod recovery.
Both full closed loops are now running on 65114 GPUs 0/1, driver PIDs
3324077/3324078, under generated/ClosedLoop10000 (old 1000-step runs retained).
Postprocessor 3324594 checks exit records then produces full-height diagnostics
and ClosedLoop10000_VisualReport.tar.gz. Read the status page for logs/markers.
1000-step low CRC is an early-iteration observation, not a final capability limit.
The new corrected 218 channel uses the new 10000-step 440 cross-talk estimate.

This is the primary starting document for a new developer or a new
conversation. Large Factors, Geant4 output, List, CntStat, and Results are
ignored by Git, so the required semantics and evidence are recorded here.

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
and only rank 0 writes the final images and histories. The supplied production
template is 4 nodes x 8 GPUs for the installed Geant4 1e10 data. Always run
`preflight.py`, then the 1-node x 2-GPU smoke job, before the production job.

## Next tasks

The old request to run the 60-mm high-count case is fulfilled. For the active
120-mm extension follow the ordered gates in
[FOV120 experiment status, section 6](FOV120_EXPERIMENT_STATUS.md#6-接续顺序与验收门槛):
finish raw matrices, collect/extend independent calibration and sensitivity,
validate GenProj and Geant4 spatial response, then run 1e9/1e10 six-output
reconstruction with full-height evaluation. Preserve the shared K*B physics
and original source/volume conventions. Investigate existing radial bias and
CntStat/List overlap before interpreting combined gamma images quantitatively.

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
