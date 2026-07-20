# GPU-Based System Matrix Calculation for SPECT/PET

A GPU-accelerated system matrix calculation tool designed for SPECT and PET systems, particularly those with complex geometries. This homemade software has been tested and performs efficiently on my systems.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Calculate Photon-Electric System Matrix](#calculate-photon-electric-system-matrix)
  - [Calculate Primary Compton System Matrix](#calculate-primary-compton-system-matrix)
  - [(Optional) Calculate Inter-Crystal Primary Compton System Matrix](#optional-calculate-inter-crystal-primary-compton-system-matrix)
- [Parameter Files](#parameter-files)
  - [Param_Collimator.dat](#param_collimatordat)
  - [Param_Detector.dat](#param_detectordat)
  - [Param_Image.dat](#param_imagedat)
  - [Param_Physics.dat](#param_physicsdat)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

This project provides a GPU-based framework for calculating system matrices essential for SPECT (Single Photon Emission Computed Tomography) and PET (Positron Emission Tomography) systems. It is optimized for complex geometrical configurations and leverages GPU acceleration to significantly reduce computation time.

## Features

- **GPU Acceleration:** Utilizes CUDA for high-performance matrix calculations.
- **Flexible Geometry Support:** Designed to handle complex system geometries.
- **Modular Structure:** Separate modules for photon-electric and Compton scatter calculations.
- **Configurable Parameters:** Easily adjustable parameter files to define system configurations.

## Prerequisites

- **CUDA Toolkit:** Ensure that the CUDA toolkit is installed and properly configured on your system.
- **Compiler:** A compatible C++ compiler (e.g., `gcc`, `clang`).
- **GPU:** NVIDIA GPU with CUDA support (e.g., RTX 6000 Ada).

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/zxc18772763792/GPU-Based-System-Matrix-Calculation-for-SPECT-PET.git
   cd GPU-Based-System-Matrix-Calculation-for-SPECT-PET
   ```

2. **Prepare Parameter Files:**

   Before compiling, prepare four parameter files (`Param_Collimator.dat`, `Param_Detector.dat`, `Param_Image.dat`, `Param_Physics.dat`) as described in the [Parameter Files](#parameter-files) section.

## Usage

### Calculate Photon-Electric System Matrix

1. **Compile the Photon-Electric Module:**

   Navigate to the `PE_Gen_RayTracing_CircularHole` directory and compile the code.

   ```bash
   cd PE_Gen_RayTracing_CircularHole
   ./bd
   ```

2. **Run the Photon-Electric System Matrix Generator:**

   ```bash
   ./PESysMatGen -cuda 0
   ```

   Replace `0` with the appropriate CUDA device ID if necessary.

### Calculate Primary Compton System Matrix

1. **Compile the Compton Scatter Module:**

   Navigate to the `ScatterGen_RayTracing_CircularHole` directory and compile the code.

   ```bash
   cd ../ScatterGen_RayTracing_CircularHole
   ./bd
   ```

2. **Run the Primary Compton Scatter System Matrix Generator:**

   ```bash
   ./ScatterGen_CircularHole \
     -PE <path_to_PE_SystemMatrix> \
     -GeoCrystal <path_to_CrystalGeometryRelationship> \
     -GeoCollimator <path_to_CollimatorGeometryRelationship> \
     -cuda <cuda_device_id>
   ```

   Replace placeholders with the actual paths and CUDA device ID.

### (Optional) Calculate Inter-Crystal Primary Compton System Matrix

1. **Compile the Inter-Crystal Compton Module:**

   Navigate to the `ScatterGen_Crystal` directory and compile the code.

   ```bash
   cd ../ScatterGen_Crystal
   ./bd
   ```

2. **Run the Inter-Crystal Compton System Matrix Generator:**

   ```bash
   ./ScatterGen_Crystal \
     -PE <path_to_PE_SystemMatrix> \
     -GeoCrystal <path_to_CrystalGeometryRelationship> \
     -cuda <cuda_device_id>
   ```

## Parameter Files

Four parameter files are required to define your system. Each file is a pure `float32` array organized as follows:

### `Param_Collimator.dat`

Defines the collimator configuration.

- **Index 0:** `numCollimatorLayers` — Number of collimator layers.
- **For each Collimator Layer (`id_CollimatorLayer`):**
  - `[id * 10 + 0]:` Number of Collimator Holes.
  - `[id * 10 + 1]:` Width of the Collimator Layer (mm).
  - `[id * 10 + 2]:` Thickness of the Collimator Layer (mm).
  - `[id * 10 + 3]:` Height of the Collimator Layer (mm).
  - `[id * 10 + 4]:` Distance between the 1st and current collimator layer (mm).
  - `[id * 10 + 5]:` Total Attenuation Coefficient.
  - `[id * 10 + 6]:` Photon-Electric (PE) Attenuation Coefficient.
  - `[id * 10 + 7]:` Compton Attenuation Coefficient.
- **For each Hole (`id_Hole`):**
  - `[id_Hole * 9 + 100]:` X-coordinate of Hole Center.
  - `[id_Hole * 9 + 101]:` Y1-coordinate of Hole Center.
  - `[id_Hole * 9 + 102]:` Y2-coordinate of Hole Center.
  - `[id_Hole * 9 + 103]:` Z-coordinate of Hole Center.
  - `[id_Hole * 9 + 104]:` Radius of Hole.
  - `[id_Hole * 9 + 105]:` Total Attenuation Coefficient of Hole.
  - `[id_Hole * 9 + 106]:` PE Attenuation Coefficient of Hole.
  - `[id_Hole * 9 + 107]:` Compton Attenuation Coefficient of Hole.
  - `[id_Hole * 9 + 108]:` Flag.

### `Param_Detector.dat`

Defines the detector configuration.

- **Index 0:** `numDetectorBins` — Number of detector bins.
- **For each Detector (`id_Detector`):**
  - `[id * 12 + 1]:` X-coordinate of Detector Center.
  - `[id * 12 + 2]:` Y-coordinate of Detector Center (set Y of 1st collimator to 0).
  - `[id * 12 + 3]:` Z-coordinate of Detector Center.
  - `[id * 12 + 4]:` Width of Detector (mm).
  - `[id * 12 + 5]:` Thickness of Detector (mm).
  - `[id * 12 + 6]:` Height of Detector (mm).
  - `[id * 12 + 7]:` Total Attenuation Coefficient (excluding Rayleigh scatter).
  - `[id * 12 + 8]:` Photon-Electric (PE) Attenuation Coefficient.
  - `[id * 12 + 9]:` Compton Attenuation Coefficient.
  - `[id * 12 + 10]:` Relative FWHM energy resolution at the target PE energy.
  - `[id * 12 + 11]:` Rotation Angle of Detector (Y-axis) [0, 2π).
  - `[id * 12 + 12]:` Flag.

### `Param_Image.dat`

Defines the image voxel configuration.

- **Index 0:** `numImageVoxelX` — Number of image voxels along the X-axis.
- **Index 1:** `numImageVoxelY` — Number of image voxels along the Y-axis.
- **Index 2:** `numImageVoxelZ` — Number of image voxels along the Z-axis.
- **Index 3:** `widthImageVoxelX` (mm) — Width of each voxel along the X-axis.
- **Index 4:** `widthImageVoxelY` (mm) — Width of each voxel along the Y-axis.
- **Index 5:** `widthImageVoxelZ` (mm) — Width of each voxel along the Z-axis.
- **Index 6:** `numRotation` — Number of rotations.
- **Index 7:** `anglePerRotation` (0~2π) — Angle increment per rotation.
- **Index 8:** `shiftFOVX` (mm) — Shift of the Field of View (FOV) along the X-axis.
- **Index 9:** `shiftFOVY` (mm) — Shift of the FOV along the Y-axis.
- **Index 10:** `shiftFOVZ` (mm) — Shift of the FOV along the Z-axis.
- **Index 11:** `FOV2Collimator0` (mm) — Distance from FOV to Collimator layer 0.

### `Param_Physics.dat`

Defines the physics parameters for the simulation.

- **Index 0:** `flagUsingCompton` — Enable (1) or disable (0) Compton scattering.
- **Index 1:** `flagSavingPESysmat` — Enable (1) or disable (0) saving PE system matrix.
- **Index 2:** `flagSavingComptonSysmat` — Enable (1) or disable (0) saving Compton system matrix.
- **Index 3:** `flagSavingPEComptonSysmat` — Enable (1) or disable (0) saving combined PE and Compton system matrix.
- **Index 4:** `flagUsingSameEnergyWindow` — Use (1) or not (0) the same energy window.
- **Index 5:** `lowerThresholdEnergyWindow` — Lower threshold of the energy window.
- **Index 6:** `upperThresholdEnergyWindow` — Upper threshold of the energy window.
- **Index 7:** `targetPEEnergy` — Target PE energy.
- **Index 8:** `flagCalculateCrystalGeometryRelationship` — Enable (1) or disable (0) calculation of crystal geometry relationship.
- **Index 9:** `flagCalculateCollimatorGeometryRelationship` — Enable (1) or disable (0) calculation of collimator geometry relationship.
- **Index 10:** `flagDetectorRecoilEscapeResponse` — Record the recoil-electron energy left in crystal A when a once-Compton-scattered photon escapes A.
- **Index 11:** `flagSelfComptonPhotoelectricResponse` — Record a full-energy pulse in A when the once-Compton-scattered photon is next photoelectrically absorbed in A.

### Energy Resolution Model for Compton Scatter

`Param_Detector[id_Detector*12+10]` stores the relative FWHM energy resolution at the target photopeak energy `E0 = Param_Physics[7]`.

For a Compton-scattered photon with energy `E'`, the scatter kernels extrapolate the relative FWHM by the scintillation-statistics model:

```text
R(E') = R(E0) * sqrt(E0 / E')
sigma(E') = R(E') * E' / 2.35482
```

This means lower-energy scattered photons have worse relative energy resolution. For example, with `R(440 keV)=0.1401`, a 440 keV photon scattered to 218 keV uses `R(218 keV) ~= 0.199`, consistent with the `1/sqrt(E)` scaling used by `FileGenerater_3D_Unified`.

PEGen writes two matrices with distinct roles:

- `PE_SysMat_*_v3.sysmat` or `PE_SysMat_*_v4.sysmat` is the unwindowed direct
  PE transport matrix. The active V4 ScatterGen receives this matrix and
  converts each PE first-interaction probability to a Compton first-interaction
  probability with `mu_compton/mu_photoelectric` before applying the
  detector-local response lookup.
- `PE_Windowed_SysMat_*_v3.sysmat` multiplies every detector row by the Gaussian
  photopeak acceptance of the configured energy window. `SysMat_withScatter_*`
  is formed from this windowed PE response plus the scatter response.

Scatter attenuation coefficients are interpolated by material and photon
energy from the embedded 1-1000 keV NIST XCOM table in `physics_data/`.

### Detector-Local Scatter Responses

After the first Compton interaction in an active detector crystal A, the
scattered photon has energy `E'`.  For its center-to-boundary path `L_A`, the
implemented mutually exclusive next-step partition is:

```text
P_escape       = exp[-(mu_PE(E') + mu_C(E')) * L_A]
P_second_PE    = (1 - P_escape) * mu_PE(E') / (mu_PE(E') + mu_C(E'))
P_second_C     = (1 - P_escape) * mu_C(E')  / (mu_PE(E') + mu_C(E'))
P_escape + P_second_PE + P_second_C = 1
```

`Physics[10]` adds `P_escape` to crystal A after convolving the recoil deposit
`E0-E'` with A's Gaussian energy response.  This response is integrated over
all escaping directions once per A/image pair; it is not repeated for every
possible destination crystal B.  Thus it includes both A-to-B events and
photons that leave the detector array.

`Physics[11]` adds `P_second_PE` to A at total deposited energy `E0`: the
recoil electron and the following photoelectric absorption belong to one
crystal pulse, so the Gaussian broadening is applied once at `E0`.  The
existing A-to-B response now also includes `P_escape` along the A-to-B
direction before attenuation in intervening materials and absorption in B.
The first-Compton normalization remains the unwindowed PE transport response
times `mu_C(E0)/mu_PE(E0)`.

Both local switches are subordinate to the global `Physics[0]` Compton switch;
when `Physics[0]=0`, neither local response is built or accumulated.

The model follows at most one Compton interaction and then either escape or a
photoelectric second interaction. `P_second_C` is retained in the probability
audit but is not transported further. The detector-local lookup integrates
all illuminated entry faces with projected-area weights. For each entry-face
sample it computes the exact incoming ray-box chord, samples the conditional
first-interaction depth from the truncated exponential distribution, then
computes the exact ray-box escape distance from that position for every
Klein-Nishina outgoing direction. The separate inter-crystal A-to-B source
attenuation still uses its existing center approximation. Only detector
records with `flag=1` produce local pulses; Pb/W shielding records do not.

Detector channels are accumulated independently.  One physical A-to-B event
can therefore contribute one expected pulse to A and another to B.  For EHE,
the current 4 mm detector bins are likewise treated as independent channels;
charge/light sharing, Anger centroiding, and event-level summation in a
continuous NaI crystal are outside this analytical model.

The direction response is precomputed in a bilinearly interpolated lookup
table.  Its convergence controls are:

```text
DETECTOR_LOCAL_SCATTER_ORIENTATION_BINS   default 17
DETECTOR_LOCAL_SCATTER_COSINE_SAMPLES    default 64
DETECTOR_LOCAL_SCATTER_AZIMUTH_SAMPLES   default 64
DETECTOR_LOCAL_SCATTER_POSITION_SAMPLES_PER_AXIS  default 4
```

The last value controls both the two coordinates on each entry face and the
conditional depth quadrature, so each visible face contributes `N^3` physical
first-interaction positions. This lookup is built on the CPU at ScatterGen
startup; it does not enlarge the GPU lookup or the output matrix.

The standard 218 and 440 photopeak cases enable both local responses.  The
standard `440 -> 218 keV window` case enables recoil escape but disables the
same-crystal full-energy response.  Any scatter or combined matrix generated
before these two terms and the A-exit attenuation were introduced must be
recomputed.

For multi-rotation input, ScatterGen consumes the matching PE slice for each
rotation. Builds predating the rotation-offset fix reused rotation 0's PE
slice for every angle, so their multi-angle scatter matrices must also be
recomputed.

The focused validation suite is reproducible with:

```bash
./tests/run_detector_local_scatter_test.sh
SMOKE_TEST_GPU=<GPU_ID> node tests/run_all_switches_validation.js
SMOKE_TEST_GPU=<GPU_ID> node tests/scatter_rotation_pe_offset_test.js
SMOKE_TEST_GPU=<GPU_ID> node tests/physics_smoke_test.js
SMOKE_TEST_GPU=<GPU_ID> SMOKE_TEST_MIXED_JSCC=1 node tests/physics_smoke_test.js
matlab -batch "addpath('tests'); validate_local_scatter_params"
```

The CPU test contains the hand-checkable example `mu_PE=0.3/mm`,
`mu_C=0.2/mm`, `L=4 mm`, for which `P_escape=exp(-2)=0.135335`,
`P_second_PE=0.518799`, and `P_second_C=0.345866`. It also verifies the
440-to-218 recoil overlap and angular-integration convergence. The GPU tests
check independent switches, nonnegative finite matrices, exact
`combined=windowed_PE+scatter`, 9/17/33 orientation-table convergence,
mixed NaI/GAGG/Pb/W identification, global-Compton gating, and per-rotation
PE slice selection. The MATLAB test serializes only into a temporary directory;
it checks a 48-byte physics file and does not modify `runs/`.

`run_all_switches_validation.js` performs one retained 440-to-218-window run
with a one-voxel image, one GAGG record, one W record, and one physical Pb
collimator layer. It enables all applicable physics/save/geometry switches,
surface quadrature, pruning, LUT, both local responses, and all component
outputs. Parameters, PE/Scatter/combined matrices, component matrices, logs,
and `validation_summary.json` are retained under
`run_logs/ScatterSurface_AllSwitches_Validation_<timestamp>/`.

### Inter-Crystal Target-Surface Integration

The production inter-crystal kernel now integrates over the visible surfaces
of the actual rectangular target crystal. The old kernel estimated the target
azimuth range using an enclosing sphere, found a broad theta interval from
the target corners, and multiplied that entire interval by the energy-window
acceptance evaluated toward the target center. That approximation strongly
over-counted nearby `2 x 6 x 2 mm` JSCC back-layer crystals.

For each visible target-face subcell, the new kernel evaluates

```text
dP = KN(theta) / integral_4pi(KN) * dOmega
   * P_window(E'(theta))
   * exp[-mu(E'(theta)) L_exit/intermediate]
   * P_PE,target(E'(theta), exact ray-box chord)
```

where

```text
dOmega = |n dot ray_direction| * dA / distance^2
```

The source-crystal center-to-boundary distance and target-crystal chord are
recomputed for every subcell direction. XCOM coefficients and Gaussian energy
acceptance are also evaluated at that subcell's scattered energy. The
enclosing-sphere phi range is not used by the launched production kernel.
Center-to-center material lengths for intervening third crystals still come
from the validated pair-path cache. This is exact for unobstructed neighboring
pairs and remains a center-ray approximation when a distant pair crosses
other detector material.

Adaptive quadrature controls are environment variables:

```text
SCATTER_TARGET_FACE_SUBDIV              default 1
SCATTER_NEAR_TARGET_FACE_SUBDIV         default 8
SCATTER_NEAR_TARGET_DISTANCE_FACTOR     default 2.0
```

A pair is near when its center distance is no more than the distance factor
times the larger crystal dimension. The default `8 x 8` near-face rule gives
about `0.22%` solid-angle error for immediately adjacent JSCC back-layer
crystals; `4 x 4` gives about `3.3%`. Use `16` as a convergence check before a
final high-cost matrix generation. Values are capped at 16.

The kinematic prefilter now includes both FOV extent and target-box extent and
uses a five-sigma Gaussian support. The surface kernel evaluates the full
Gaussian CDF and has no two-sigma hard cutoff.

Set the following only for diagnostic runs:

```bash
export SCATTER_WRITE_COMPONENTS=1
```

It writes one full matrix for each component:

```text
C_intercrystal.sysmat
C_highZ_to_crystal.sysmat
C_local_recoil.sysmat
C_local_self_photoelectric.sysmat
C_collimator_to_crystal.sysmat
C_total.sysmat
```

`C_intercrystal` uses active scintillator first-interaction records (`flag=1`),
while `C_highZ_to_crystal` uses detector shielding records (`flag>1`). For a
complete V4 run, the elementwise sum of all five physical component matrices
equals `C_total` up to floating-point rounding. Component files must come from
the same executable, parameters, and run as `C_total.sysmat`; never mix
component matrices from different model versions.

All `Scatter_SysMat` and combined matrices generated before this target-surface
change are obsolete, especially `JSCC_440keV_to_218keVwin`. PE matrices may be
reused. The center-ray pair-material cache is obsolete and is ignored.

### Full JSCC Surface Validation (2026-07-15)

The retained full-size validation run is:

```text
runs/JSCC_440keV_to_218keVwin_SurfaceValidation/
```

It uses the restored JSCC 440-to-218 parameters, the validated 440 keV raw PE
matrix, all 12 physics/save/geometry switches, all component outputs, far
`1 x 1` and near `8 x 8` target-face quadrature, structured traversal,
kinematic pruning, and the Compton LUT. Each matrix has shape
`11520 x 52020` and exactly `2,397,081,600` bytes. The full scan passed:

```text
finite values                         yes
nonnegative values                    yes
Scatter == C_total                    exact
sum(five components) == C_total       max abs error 4.55e-13
combined == accepted PE + Scatter     exact
```

The report and visualization are retained under:

```text
run_logs/ScatterSurface_FullValidation_20260715/
```

Reproduce the scan with:

```bash
python tests/validate_full_surface_run.py \
  runs/JSCC_440keV_to_218keVwin_SurfaceValidation \
  --output-dir ../../run_logs/ScatterSurface_FullValidation_20260715
```

The old enclosing-sphere matrix and new target-surface matrix compare with the
`1e9` Geant4 center-point response as follows:

```text
detector y (mm)          30      60      90      120     total
old matrix / Geant4    0.938   0.975   0.978    2.308    1.903
new matrix / Geant4    0.811   0.796   0.752    0.778    0.780
```

The target-surface change removes the anomalous last-layer factor of `2.31`;
all four layers now have a similar normalization. It does not by itself give
absolute Geant4 agreement: the new response is about 22% low. After masked
spatial smoothing over two and four crystal pitches, detector-bin correlation
increases from `0.525` to `0.939` and `0.982`. The earlier unconstrained
component fit that produced scales near `1.28` and `1.31` did not compare
event-topology-equivalent quantities and must not be used to scale either
component.

The full-detector, one-center-voxel convergence run is retained under:

```text
run_logs/ScatterSurface_CenterConvergence_20260715/
```

Increasing far target-face subdivisions from `1` to `2` and `4` changes the
total from `1.43294e-3` to `1.42307e-3` and `1.41923e-3`. Thus far-face
quadrature is converged to about 1% and cannot explain the 22% deficit. The
structured pair traversal explicitly excludes source and target records from
intermediate material lengths, so source/target double attenuation is also
excluded as the cause.

The present analytic response is first-order. Geant4 CntStat includes any
crystal whose total broadened deposit enters the window, including multiple
Compton interactions, one/two/three-plus hit crystals, Rayleigh-redirection
histories, and atomic-deexcitation escape. The response-study Geant4 project
now exports topology-separated counters to quantify those residual terms
before any empirical `1/0.78` matrix scaling is considered.

### 1e9 Geant4 topology diagnosis (2026-07-16)

The completed 100-worker mixed-source run has exact primary accounting:

```text
PrimaryCount218                         304056693
PrimaryCount440                         695943307
PrimaryCountOther                               0
total primary events                    1000000000
440-to-218 accepted detector-bin counts   1277527
```

All integer identities close exactly in every active detector bin:

```text
From440 = FirstCrystal + OtherCrystal
From440 = Hit1 + Hit2 + Hit3Plus
FirstCrystal = Compton0 + Compton1 + Compton2Plus
```

The new run independently reproduces the earlier normalization result:

```text
response                                  per 440 primary
Geant4 440-to-218                         1.835676825e-3
C_total                                   1.432938807e-3
C_total / Geant4                               0.780605
old independent C_total / Geant4               0.779996
```

The difference between the two matrix/Geant4 ratios is only about `0.08%`, so
the approximately 22% deficit is systematic and not residual Monte Carlo
noise. The new per-layer ratios are:

```text
detector y (mm)           30       60       90      120
matrix / Geant4        0.8016   0.7908   0.7587   0.7799
```

The event hit-multiplicity partition is:

```text
category          response / 440 primary    fraction of all accepted bins
Hit1                   6.037661341e-4                    32.89%
Hit2                   8.779077173e-4                    47.82%
Hit3Plus               3.540029734e-4                    19.28%
```

`Hit3Plus` alone is `87.9%` as large as the complete matrix deficit
`4.027380178e-4`. Its fraction grows strongly with detector depth:

```text
detector y (mm)           30       60       90      120
Hit3Plus fraction       2.94%    5.28%    8.78%   25.10%
```

This is strong evidence that multi-crystal and higher-order histories are the
dominant missing physics, especially in the finely segmented last layer. It is
a magnitude comparison, not an exclusive event-by-event closure: an
analytical local-recoil count can correspond to the first accepted bin of a
Geant4 event that later deposits energy in two more crystals.

The provisional topology-matched local comparison is:

```text
C_local_recoil / FirstCrystal_Compton1          0.958658
per-layer ratios                   0.918, 0.954, 0.911, 0.977
four-pitch-smoothed shape correlation            0.967
```

Thus the local first-order term is much closer than the unpartitioned total,
and the data do not support multiplying every first-order component by the
common factor `1/0.780605`. The strict two-crystal `List.csv` target subset is
also not lower than the analytical intercrystal term: provisionally,
`C_intercrystal / ListStrictSecondWindow = 1.222`. This value is not yet a
calibration constant.

The component-level conclusions remain provisional because the Geant4 run was
generated with a `SteppingAction` that sets `FirstCrystal` and counts a primary
Compton process only when that same primary step has
`GetTotalEnergyDeposit()>0`. A discrete Compton step can instead transfer its
energy to a tracked secondary. Therefore `FirstCrystal`, `OtherCrystal`, the
Compton subcategories, and the strict List subset are approximate process
labels. `CntStat218_From440` and `Hit1/Hit2/Hit3Plus` do not have this process
label limitation. Correct the Geant4 classifier and run a 440-only point
source before making component-specific physical normalization changes.

The full machine-readable result, detector table, and figure are under:

```text
Geant4Sim/Geant4Code_CntStatResponseStudy/build/
  merged_CntStatResponseStudy/topology_analysis/
```

### Cost of explicit second-order scatter

Not every second-scatter term has the same cost. A second Compton interaction
inside the same crystal can be incorporated by extending the existing
detector-local lookup with outgoing-energy/deposited-energy state. That adds
lookup-generation work and table dimensions but does not require a full
detector triple, so its runtime cost can remain moderate.

The expensive part for JSCC is explicit cross-crystal second scatter. The
current intercrystal term sums an image-to-first-crystal-to-target-crystal
path. A literal second-order model adds a third crystal and must retain the
energy and direction after the second interaction:

```text
image -> crystal i -> crystal j -> crystal k
```

The naive work changes from detector-pair enumeration to detector-triple
enumeration. With `10496` active crystals (`11520` detector records including
inactive/high-Z records), the conceptual unpruned pair and triple spaces are
about `10496^2 * 52020 = 5.7e12` and `10496^3 * 52020 = 6.0e16`
crystal-path/voxel combinations. Thus an unpruned triple kernel is not
practical and adds approximately another factor of `10496` before geometry and
kinematic pruning. The energy-window acceptance also can no longer be
represented by one scalar. Storing only the accumulated output still needs one
ordinary `11520 x 52020` matrix (`2,397,081,600` bytes), but storing diagnostic
second-order components costs another full matrix per component; the dominant
cost is computation and intermediate state rather than final output size.

A practical physical implementation must be sparse. Restrict the second step
to geometrically reachable near neighbors, discretize the outgoing energy and
direction, precompute crystal-to-crystal transition tables, and accumulate the
result without materializing a detector triple. Such a method is feasible but
will still likely cost several to tens of times one first-order component,
depending on neighbor count and energy/angular bins. It is a separate model
development and validation task, not a small switch in the present kernel.

### Current JSCC Matrix Status and Layer Calibration (2026-07-16)

The three standard JSCC responses have now been regenerated after both major
scatter-model corrections:

1. Inter-crystal scatter integrates the visible surfaces of the real target
   box. The enclosing-sphere azimuth estimate is no longer used by the
   production kernel. Near targets use subdivided surface integration and
   exact source/target ray-box path lengths.
2. Detector-local scatter no longer starts every outgoing photon at the
   crystal center. It integrates projected entrance faces, conditional
   first-interaction depth, Klein-Nishina outgoing direction, and exact
   ray-box escape distance from each sampled interaction position.

The implementation also includes per-rotation PE-slice selection, independent
recoil-escape and same-crystal Compton-to-PE switches, forced-window support,
component closure diagnostics, structured material traversal, kinematic
pruning, LUT validation, and chunked Combined output. Focused CPU/GPU tests
cover probability conservation, arbitrary-position ray-box geometry,
orientation/position convergence, component closure, rotation indexing, and
nonnegative finite output.

The current raw files remain per emitted source photon and are preserved in:

```text
runs/JSCC_218keV/
runs/JSCC_440keV/
runs/JSCC_440keV_to_218keVwin/
```

Their complete streamed scan is:

```text
response                  PE_Windowed        Scatter        Combined/Cross
JSCC/A218                 1179.355342       112.995173       1292.350515
JSCC/A440                  364.881094        42.048601        406.929695
JSCC/C440to218                    n/a        98.930624         98.930624
```

Every file has shape `11520 x 52020`, contains no NaN/Inf or negative value,
and both direct responses satisfy `Combined = PE_Windowed + Scatter` exactly
in float32. The detector-local position model changes the raw Scatter totals
by `-21.22%`, `-23.22%`, and `+5.19%` for A218, A440, and C440to218,
respectively. These signs are physically consistent: distributed positions
reduce same-crystal second-PE containment while increasing recoil escape.

Comparison of the center-interpolated matrix columns with the current `1e9`
Geant4 response study gives:

```text
response             Geant4 / primary       raw matrix       matrix / Geant4
JSCC/A218              9.899278882e-3      1.134109603e-2          1.145649
JSCC/A440              2.584940443e-3      2.956339967e-3          1.143678
JSCC/C440to218         1.835676825e-3      1.506842172e-3          0.820865
```

The position-integrated direct ratios agree with the independent detector-box
prediction to better than about `0.6%`, which validates the Detector-Local
rewrite. The remaining direct excess and cross-window deficit are different
physics effects and must not share one scale.

#### Current Factors calibration

Keep raw `.sysmat` files unchanged. Calibration is applied only while
exporting active scintillator rows to project-root `Factors/`, before
Cartesian-to-polar interpolation. Row scaling commutes with interpolation, so
`SysMat_tmp` and `SysMat_polar` remain consistent. It does not include or alter
the 225Ac branching ratio.

```text
detector y (mm)       active rows      D218       D440       D440to218
30                            512    0.8740232  0.8720313    1.1411090
60                            768    0.8793926  0.8912363    1.1636691
90                           1024    0.8719679  0.8847924    1.2201934
120                          8192    0.8708826  0.8691287    1.2372030
total                       10496
```

`GenFactors/run_gen_response_factors.m` enables these vectors only for
`JSCC/A218`, `JSCC/A440`, and `JSCC/C440to218`. It audits all four detector
depths and active-row counts, aborts on unclassified rows, and records the
calibration name, source, scope, layer positions, scales, and the absence of
branching-ratio weighting in `factor_manifest.json`. EHE Factors do not use
these JSCC-derived corrections.

These are center-point empirical calibrations. They are suitable for the next
reconstruction comparison, but they are not yet a final position-independent
detector model.

### PE v4 Reference and Production Kernel (2026-07-18)

The first PE v4 development stage is implemented without changing or
overwriting the production v3 matrices. The new common state is defined in
`common/first_interaction.h`:

```text
FirstInteractionState
  entry face axis/sign and local entry position
  local first-interaction position
  normalized incoming direction
  target chord and sampled conditional depth
  surface, interaction, PE, and Compton weights
```

Both `common/pe_v4_reference.h` and the production detector-local lookup in
`common/detector_local_scatter.h` now use this state generator. Entry-face
selection, exact ray-box exit distance, first-interaction probability, and
truncated-exponential depth sampling therefore have one implementation. The
detector-local regression test retains its previous values and probability
partition after the refactor.

`PEGen_RayTracing_CircularHole/PEGen_V4_Reference.cpp` computes one selected
detector/voxel pair. For every visible target face it evaluates exact
`dOmega/(4*pi)`, target chord, PE/Compton branching, and attenuation through
every other real detector box. This deliberately costs `O(N_detector)` per
surface sample and is a reference, not the full GPU production implementation.
It supports the current zero-hole/vacuum JSCC collimator and aborts when a
physical collimator hole is present.

Two surface rules are retained:

1. Composite two-point Gauss-Legendre per face cell for smooth analytic tests.
2. Center-symmetric Halton low-discrepancy points for the default JSCC
   reference. Every base point is reflected across both face axes, with a
   center point for odd sample counts. This retains irregular low-discrepancy
   coverage while making the finite-sample surface centroid exactly centered.
   Upstream crystal shadows are discontinuous, so regular midpoint/Gauss grids
   can phase-lock to the 4.2 mm detector lattice.

Focused analytic validation passes:

```text
opaque rectangular solid-angle relative error    4.57e-14
PE + Compton first-interaction closure            1.98e-18
truncated-exponential mean-depth error             1.82e-7 mm
smooth finite-attenuation 64-to-128 change         0.0018%
detector-local probability partition               exact within 2e-12
scatter.cu CUDA 12.8 compile after refactor         pass
```

The retained local multi-pair report is under
`Results/Analysis/PEV4ReferenceValidation_20260718_JSCC218_Halton/`. It selects
the central active row of each detector layer and source voxels near `y=0` and
`y=150 mm`. Halton 32/64 passes the 2% convergence gate for all eight pairs:

```text
layer y (mm)   source y (mm)   fine convergence   v4 / v3
30                  0              0.041%          1.0001
30                150              0.080%          1.0008
60                  0              0.115%          1.9128
60                150              0.564%          0.9420
90                  0              0.228%          1.0413
90                150              0.108%          0.9930
120                 0              0.213%          1.0103
120               150              0.128%          1.0013
```

The approximately `1.91` ratio is a low-probability, strongly shadowed second-
layer element (`PE_v4 ~= 2.2e-7`), not a full-layer normalization. It proves
that v3 can have large source-position-dependent local errors even when total
efficiency is calibrated. At the same detector row and `y=150 mm`, v3 is about
6% high. A detector-row scale cannot correct both conditions.

`PEGen_RayTracing_CircularHole/PEGen_V4_Production.cu` is the production GPU
implementation for complete JSCC matrices. It preserves the reference-model
equations while replacing the reference `O(N_detector)` attenuation scan with
a four-layer, center-indexed `x-z` grid. For every Halton target-face sample it
queries only cells swept by the source-to-entry segment, then evaluates exact
ray-box chords for all candidate GAGG and tungsten records. The grid has no
fixed candidate-array limit and therefore cannot silently truncate a crowded
or grazing path.

Direct PE does not require explicit first-depth quadrature. For each incident
ray the target contribution is evaluated analytically as

```text
dOmega/(4*pi) * upstream_survival
  * mu_PE/mu_total * (1 - exp(-mu_total * target_chord))
```

The production defaults are symmetric Halton `16 x 16` per visible face, four
detector rows per checkpoint, and 32 surface samples per short CUDA launch.
Short launches avoid Windows WDDM timeouts. Detector-row chunks are written to
`.partial` files, checkpointed in atomic JSON plus append-only TSV, and can be
resumed only on a complete row boundary. Both active detector rows and the 1024
tungsten rows are generated: tungsten rows are excluded from Factors but are
required as first-interaction source terms by ScatterGen.

On the local RTX 6000 Ada, a pre-density-alignment benchmark of the complete
`11520 x 52020` matrices took 103 s at 218 keV and 109 s at 440 keV. Those
outputs are retained only under
`runs/Archive_DensityMismatch_6p63_19p30_20260718` and must not be used for
physics comparisons. The timing remains a useful estimate. The retained
eight-pair GPU gate reports:

```text
maximum GPU vs CPU error at identical Halton-16 samples     1.76e-6
maximum production Halton-16 vs CPU Halton-32 difference    1.265%
resume output vs uninterrupted output SHA-256               identical
```

Build and run the complete local JSCC chain with:

```powershell
cd Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main
./PEGen_RayTracing_CircularHole/build_pe_v4_production.ps1
./run_pe_v4_jscc_pipeline.ps1 -CudaId 0 -FaceSubdivisions 16
./monitor_pe_v4_pipeline.ps1
```

The pipeline retains the standard runs and writes `JSCC_218keV_pe_v4`,
`JSCC_440keV_pe_v4`, and `JSCC_440keV_to_218keVwin_pe_v4`. It generates the
two PE matrices first and then runs the existing detector-local ScatterGen for
the three response windows. `GenFactors/run_gen_response_factors.m` accepts
`grid_options.run_name_suffix="_pe_v4"`, so these runs can be exported without
replacing v3 inputs.

#### First full-matrix Uniform-FOV result and quadrature correction

The first density-aligned full matrices generated on 2026-07-18 used the
legacy, unreflected first 256 Halton points. Against the same merged Geant4
Uniform-FOV data, the exported `CenterPoint_PEv4` Factors did not improve the
v3 baseline:

```text
response     v3 total error   legacy-v4 total error   v3 shape L2   legacy-v4 shape L2
A218              0.102%             0.459%              0.007150          0.008214
A440              0.200%             0.236%              0.010496          0.011078
C440to218          1.221%             1.115%              0.011693          0.012002
```

The old 256-point sample centroid was `(u,v)=(0.49805,0.49516)`. Across the
three responses, 85.5%, 91.4%, and 93.1% of the per-detector v4-v3 change was
explained by a linear detector `x-z` plane. Its direction matched the sample
centroid offset. This is a deterministic quadrature artifact, not evidence
against visible-face/ray-box integration. Production and CPU reference sampling
now use four-way reflected Halton groups; regression tests require exact x and
z reflection agreement at the production `16 x 16` level. The legacy matrices
and `CenterPoint_PEv4` Factors must not be promoted as final production data.
Comparison tables and maps are retained under
`Results/Analysis/UniformFov_PEv3_vs_PEv4/`.

After the correction, the CUDA production executable rebuild passed and a
representative 218-keV detector/voxel smoke test gave:

```text
detector row 505, voxel 24709, Halton 16 x 16
GPU PE probability       4.8530046115e-6
CPU PE probability       4.8530046423e-6
relative error           6.34e-9
```

The result is retained under
`Results/Analysis/PEV4SymmetricHaltonSmoke_20260718/`. The pipeline now rejects
completed outputs whose manifest model is not
`PE_v4_visible_surface_symmetric_halton_layer_grid`, and rejects Scatter output
older than its PE input. This prevents a source rebuild from silently mixing
legacy PE and current Scatter matrices.

The same pipeline now regenerates all three JSCC `Params_*.dat` sets with
`FileGenerater_3D_Unified/generate_jscc_218_440_response_params.m` before it
prepares run directories. Params are always recopied with replacement before
the material-density gate. This fixes a failure mode in which newly created
`_pe_v4` runs inherited old `GAGG=6.63/W=19.30` coefficients from a standard
run even though the current material database specifies `6.60/19.35`.

#### Symmetric-Halton full production result

The complete corrected production run finished on 2026-07-18. Matrix scans
confirmed the symmetric model manifest, exact byte counts, finite nonnegative
elements, and exact float32 closure of both combined matrices. Local timings on
the RTX 6000 Ada were:

```text
218 PE                 105.17 s
440 PE                 106.48 s
218 Scatter             19.42 min
440 Scatter             10.32 min
440-to-218 Scatter      20.31 min
```

The standard `CenterPoint_PEv4` Factors now refer to these corrected matrices;
the first matrices and Factors were renamed with
`legacy_asymmetric_halton_20260718` / `LegacyAsymmetricHalton` suffixes.
Against the same Uniform-FOV Geant4 data:

```text
response     v3 total error   symmetric-v4 total error   v3 shape L2   symmetric-v4 shape L2
A218              0.102%                0.535%               0.007150          0.007276
A440              0.200%                0.218%               0.010496          0.010593
C440to218          1.221%                1.117%               0.011693          0.011404
```

Relative to legacy asymmetric PE v4, symmetric sampling reduces shape L2 by
11.4% for A218, 4.4% for A440, and 5.0% for C440to218. The detector `x-z`
linear-gradient explanation drops from 85--93% for legacy-v4-minus-v3 to less
than 0.002% for symmetric-v4-minus-v3. The remaining v4-v3 change is symmetric
and predominantly quadratic/edge-related, while only 2--6% of the actual
Geant4-to-v4 residual is explained by a simple quadratic detector-position
model. The quadrature artifact is therefore fixed. PE v4 gives a modest real
improvement for the cross-window response, but the direct A218/A440 responses
remain statistically close to, and not better than, the calibrated v3 baseline.
Reports and detector maps are under
`Results/Analysis/UniformFov_PEv3_vs_PEv4_SymmetricHalton/` and
`Results/Analysis/UniformFov_PEv4_Legacy_vs_Symmetric/`.

### Development focus after PE v4 Uniform-FOV validation

The absolute Uniform-FOV layer correction applied to symmetric PE v4 gives
zero layer-total error by construction and reduces detector-shape L2 to
`0.003482`, `0.005622`, and `0.010619` for A218, A440, and C440to218. The first
Geant4 1e10 reconstruction displayed a central depression because it treated
integrated activity per nonuniform polar cell as activity density.

The retained V4-S model uses the symmetric PE-v4 response together with the
detector-local, center-path ScatterGen implementation. Before changing
transport again, uniform-background comparisons must use a source vector whose
weights represent physical voxel volume rather than one equal value per polar
sample.

This source-measure issue is now resolved in production. `gen_factors.m`
writes the density-basis matrix `B=A*diag(DeltaV_mm3)`, and
`run_gen_jscc_production_factors.m` installs center-inclusive, calibrated V4-S
matrices into the standard no-suffix Factors directories. In the 2000-iteration
Geant4 1e10 test, the center-to-middle background ratio changed from
`0.699 -> 1.157` for 440, `0.419 -> 0.890` for corrected 218, and
`0.607 -> 1.010` for the corrected sum. The source measure explains the former
central depression; the remaining 440 center overshoot and outer-FOV decline
remain position-dependent response-model questions.

The current JSCC material densities are deliberately aligned with Geant4:
`GAGG=6.60 g/cm^3` and `W=19.35 g/cm^3`. Both
`physics_data/nist_xcom_materials_1_1000keV.csv` and its generated CUDA header
use these values. Before matrix generation, the pipeline runs
`tests/validate_jscc_material_density_alignment.py`, which cross-checks the
three Geant4 material definitions, mass-to-linear XCOM conversion, embedded
CUDA arrays, direct/cross-run `Params_Detector.dat`, and the 440 cross-run
identity.

The production run was stopped when the old `6.63/19.30` density pair was
found. The three `_pe_v4` run directories now contain regenerated symmetric PE
v4/ScatterGen inputs with `6.60/19.35`; legacy asymmetric PE results are kept
in dated archive directories. A current-density second-layer smoke test passes
with GPU/CPU relative errors of `1.88e-6` and `6.18e-7` for source voxels near
`y=0` and `y=150 mm`.

Reproduce the CPU checks on Windows with:

```powershell
./tests/run_pe_v4_reference_test.ps1
```

On Linux, build the reference executable and run the representative JSCC set:

```bash
cd PEGen_RayTracing_CircularHole
g++ -std=c++17 -O3 -I.. -o PEGen_V4_Reference PEGen_V4_Reference.cpp
cd ..
python tests/validate_pe_v4_jscc_reference.py runs/JSCC_218keV \
  --binary PEGen_RayTracing_CircularHole/PEGen_V4_Reference \
  --surface-rule halton --face-levels 32 64 --depth-subdiv 8
```

#### Remaining development directions

The following order is retained for future development. Intrinsic-response
correction is explicitly deferred and is not part of the current v4 work.

1. Retain the validated polar-volume density basis in every production Factor.
2. Revalidate detector/layer efficiency with an independent physical uniform
   cylinder and reserve the contrast phantom for post-calibration validation.
3. Add coherent/Rayleigh coefficients and an unchanged-energy directional
   transition after PE v4 geometry is stable. Current XCOM transport omits this
   branch, which can alter detector-bin shape without a large total-efficiency
   change.
4. Add higher-order 440-to-218 Compton transport as sparse neighbor/energy/
   direction transitions. Never enumerate or materialize detector triples.
5. Add physical-hole attenuation to both v4 paths before using them for EHE.
   The current reference intentionally aborts for nonzero hole counts.
6. Revisit finite-window intrinsic containment only after the geometry and
   V4-S production path pass radial Geant4 validation. Exact 1 eV
   containment factors remain a lower-bound diagnostic and must not be applied.

The center/edge audit finds voxel-sensitivity mirror p95 residuals of
`0.751%/0.885%` (x/z) at 218 keV and `0.410%/0.435%` at 440 keV. Individual
detector mirror-pair row sums are less symmetric because the real geometry is
not mirror-closed: x reflection has 1556 missing positions and 752 GAGG/W
material mismatches; z reflection has 1488 and 668. Center PE sensitivity is
`1.148` (218) and `1.129` (440) times the FOV mean. The former center
depression was a polar-measure display error, not a low PE-v4 center
sensitivity or directional Halton artifact.

#### Priorities for further improvement

##### Post-volume-basis response diagnostic sequence

1. Simulate independent monoenergetic uniform cylinders for A218, A440, and
   C440to218 with known primary counts and the production energy windows.
2. Forward-project the identical physical source with density-basis Factors,
   including partial-cell overlap at the cylinder boundary.
3. Fit only low-dimensional detector-layer factors on that calibration data.
4. Validate on the existing contrast phantom and radial point-source scan; do
   not report the fitted uniform cylinder as independent validation.
5. If the 440 center overshoot or outer decline remains, diagnose a smooth
   source-position term separately from detector-row efficiency before
   changing transport physics.

1. Measure finite-photopeak-window GAGG containment, not only exact `1 eV`
   energy containment. The intrinsic study should retain deposited-energy or
   escaped-energy spectra for first-PE and first-Compton-to-PE histories, then
   convolve them with the same detector resolution and energy window used by
   the matrix code. Directly multiplying by exact-containment probabilities
   over-corrects the direct matrices.
2. Add higher-order scatter for the 440-to-218 response. The remaining
   approximately `18%` center deficit and its depth dependence are consistent
   with multi-crystal/multiple-Compton histories absent from the first-order
   model. Use a sparse near-neighbor transition model with discretized energy
   and direction; do not materialize a detector triple.
3. Validate the four row factors at off-center radial and axial source
   positions. If they vary systematically, replace the center-only correction
   with a regularized low-rank `D_detector * A * D_voxel` model or a small
   interpolated layer/radius/axial table. Avoid noisy per-crystal scaling.
4. Keep Geant4 and analytical definitions synchronized: crystal material and
   density, energy broadening, active-channel mapping, source normalization,
   object material, and event classification must match. Use topology counters
   only after their process labels have been independently validated.
5. Before another full 2.397 GB regeneration, test each new physical term on
   one-voxel/center-column runs with component output and convergence scans.

### Historical Pre-Position Calibration (Superseded)

The following section documents the pre-Detector-Local-position diagnosis.
Its matrix totals and calibration vectors are retained for provenance only.
Do not apply them to the current matrices or Factors; use the current section
above.

The complete post-surface-kernel JSCC matrices generated on 2026-07-16 pass a
full streamed validation. Every matrix has shape `11520 x 52020`, contains no
NaN/Inf or negative value, and the two direct responses satisfy
`SysMat_withScatter = PE_Windowed + Scatter` to float32 rounding. Their full
matrix sums are:

```text
response                  PE_Windowed        Scatter        Combined/Cross
JSCC/A218                 1179.355342       143.428566       1322.783908
JSCC/A440                  364.881094        54.765431        419.646525
JSCC/C440to218                    n/a        94.047874         94.047874
```

Comparison of the two center-Z columns with the latest `1e9` mixed-source
Geant4 run gives:

```text
response             Geant4 / primary       raw matrix       matrix / Geant4
JSCC/A218              9.899278882e-3      1.183390427e-2          1.195431
JSCC/A440              2.584940443e-3      3.159536694e-3          1.222286
JSCC/C440to218         1.835676825e-3      1.432938856e-3          0.780605
```

Thus all three effective reconstruction responses need independent empirical
calibration. The cross-talk factors must not be reused for the direct
responses. Preserve the raw physics matrices and define calibrated responses
with three detector-row diagonal operators:

```text
A218_calibrated       = D218       * (PE218_Windowed + Scatter218)
A440_calibrated       = D440       * (PE440_Windowed + Scatter440)
C440to218_calibrated  = D440to218  * Scatter440to218
```

Do not alter the 218/440 branching ratio. The calibration acts on response per
emitted photon; nuclear emission probabilities are still applied later in
GenProj/reconstruction.

#### Direct 218 and 440 response calibration

The direct PE-only terms are below Geant4 by `3.36%` at 218 keV and `10.69%`
at 440 keV, but adding the present first-order same-energy Scatter terms makes
the Combined matrices too high by `19.54%` and `22.23%`, respectively. The
four-pitch-smoothed detector-plane correlations of the raw Combined matrices
with Geant4 are `0.9965` and `0.9884`, so their broad response shapes are
already accurate and the dominant mismatch is amplitude.

For the current effective reconstruction model, scale the final Combined
response rather than scaling PE and Scatter separately. Holding PE fixed would
require highly layer-dependent Scatter-only factors, especially at 218 keV,
and the current diagnostic run does not provide process-exact direct-response
component labels. Combined scaling is therefore the more stable provisional
calibration while raw PE/Scatter files remain available for physics work.

The center-point layer factors are:

```text
detector layer y (mm)        30         60         90        120
D218                       0.8336583  0.8392622  0.8328231  0.8375539
D440                       0.8095773  0.8280269  0.8227386  0.8168062
```

The corresponding one-scalar alternatives are:

```text
A218_calibrated = 0.8365184 * A218_raw
A440_calibrated = 0.8181391 * A440_raw
```

These global factors already leave the four center-point layer ratios within
`0.45%` at 218 keV and `1.20%` at 440 keV. Use the layer factors when following
the same detector-depth calibration convention as the cross response; use the
global factors for the more conservative single-center preliminary test.

#### Why the raw direct matrices are high

Do not apply the direct-response factors above until the following algorithmic
issue has been tested. The excess is localized to the detector-local
same-crystal Compton-then-photoelectric term, not to the corrected target
surface kernel. The retained same-version center-component run gives:

```text
component fraction of total same-energy Scatter
energy       C_local_self_photoelectric   C_intercrystal   C_highZ_to_crystal
218 keV                   85.65%               13.75%              0.60%
440 keV                   93.71%                4.20%              2.10%
```

Removing only `C_local_self_photoelectric` from the new full direct responses
changes the center comparison to:

```text
energy       raw Combined / Geant4       (PE + nonlocal Scatter) / Geant4
218 keV                 1.19543                            1.00294
440 keV                 1.22229                            0.91740
```

Thus the entire 218 excess and most of the 440 excess are carried by this one
local term. The current implementation computes:

```text
P(first Compton anywhere in crystal)
  * P(second photoelectric before escape | first Compton at crystal center)
```

The first factor comes from the volume-integrated raw PE matrix multiplied by
`mu_Compton(E0)/mu_PE(E0)`, but the second factor calls
`detectorCenterExitDistance` and starts every scattered photon at the geometric
center of the complete crystal. This factorization is not valid because the
second path length depends strongly on the actual first-interaction position.

A fixed-seed independent GAGG box calculation using the same XCOM coefficients
and Klein-Nishina angular distribution compared the current center model with
uniform entrance-face coordinates and the correct conditional exponential
first-interaction depth. For normal incidence:

```text
energy   crystal (mm)   center second-PE   distributed second-PE   center/high
218      3 x 3 x 3            0.36548             0.26593             1.374
218      2 x 6 x 2            0.33004             0.24600             1.342
440      3 x 3 x 3            0.12364             0.08955             1.381
440      2 x 6 x 2            0.11046             0.08170             1.352
```

Therefore the center-start approximation overestimates the local second-PE
branch by about `34-38%`. Replacing only this approximation with the
distributed-position estimate would reduce the predicted Combined/Geant4
ratios from `1.195/1.222` to approximately `1.140/1.137`. It explains a
material part of the excess but not all of it.

The remaining common approximately 14% excess is consistent with another
known analytical assumption: both PEGen and the local self-PE branch treat a
photoelectric interaction as depositing its complete photon energy in the
same crystal and then apply only Gaussian energy broadening. Geant4 option4
tracks GAGG atomic de-excitation, fluorescence photons, photoelectrons, and
other secondaries. In 2-3 mm high-Z GAGG crystals, energy can leave the
interaction crystal or enter a neighboring crystal, moving that crystal out
of the photopeak window. Coherent/Rayleigh scattering, the small density and
Ce-composition difference, and production cuts are additional smaller model
differences; the material and gross geometry themselves match.

The direct-response correction work is now split into these stages:

1. The center-only local lookup has been replaced with an incoming-face,
   conditional first-depth, and outgoing-direction integral using exact
   ray-box escape distance from every sampled first-interaction position.
   Its focused C++ test checks arbitrary-position ray-box geometry, probability
   conservation, and angular/position convergence. For a `3 x 3 x 3 mm` GAGG
   crystal at 440 keV, changing position quadrature from `4^3` to `6^3` changes
   local self-PE by `7.9e-5` and escape probability by `7.8e-4`.
2. The standalone project
   `Geant4Sim/Geant4Code_GAGGIntrinsicResponse` now measures the required
   energy-containment probability for both JSCC GAGG crystal types. It reports
   first-PE, first-Compton-then-second-PE, and first-Compton-then-eventual-PE
   histories separately. Its four `1e7` production macros still need to be run
   in a Geant4 environment before containment factors can be applied.
3. Add the measured component-specific energy-containment probability for
   photoelectric interactions in
   each GAGG crystal type, preferably from a small dedicated Geant4 intrinsic
   response simulation that separates first-PE and Compton-then-PE histories.
4. Regenerate only center-column or `1 x 1 x 1` diagnostic responses with
   component output before rerunning either 2.397 GB direct matrix.

Missing second/third Compton histories cannot explain the present positive
excess; adding them before correcting local containment would increase the
direct matrices further. Keep all calibration factors provisional and leave
the raw full matrices unchanged until this local-response test is complete.

#### 440-to-218 cross-response calibration

The fastest center-point-only check is:

```text
A_cross_calibrated = 1.281057 * A_cross
```

This exactly corrects the total center-point normalization but over-corrects
the already close local first-Compton component. It is acceptable only as a
sensitivity test of reconstruction bias, not as a physical production matrix.
For the cross-talk response actually used by reconstruction, it is
nevertheless a useful first temporary correction: after this scalar is
applied, the four layer matrix/Geant4 ratios become:

```text
detector layer y (mm)       30       60       90      120
scaled matrix / Geant4    1.027    1.013    0.972    0.999
```

Thus the center-point layer error is already within about `2.8%` with one
scalar and no additional runtime cost. Use this global scale first for rapid
reconstruction comparisons; use the layer-specific correction below only
when tighter center-point depth normalization is needed.

For the present JSCC geometry, a better low-cost correction is detector-layer
row scaling of the cross-talk response matrix:

```text
detector layer y (mm)       30        60        90       120
row scale                 1.24749   1.26460   1.31802   1.28228
```

Equivalently, for every matrix row `d` belonging to layer `l`:

```text
A_cross_calibrated[d, :] = layer_scale[l] * A_cross[d, :]
```

Apply these factors only to active scintillator rows (`detector flag == 1`).
This preserves the current within-layer response shape, which agrees well
after detector-plane smoothing, while matching each layer's center-point
Geant4 normalization. It has negligible generation and reconstruction cost
and does not require another system matrix in memory if the four factors are
applied during matrix export/loading.

#### Remote regeneration procedure

Keep matrix generation and empirical calibration as two explicit stages. On
the remote GPU system, first regenerate all raw PE/scatter matrices with the
normal physical kernels and preserve those files unchanged. Apply this
calibration only when preparing the reconstruction response. This keeps the
unmodified first-order matrix available for later physics comparisons.

For the six dual-energy responses handled by `run_gen_response_factors`, the
calibration scope is exactly:

```text
response                         input matrix                         calibrate
JSCC/A218                        SysMat_withScatter                       yes
JSCC/A440                        SysMat_withScatter                       yes
JSCC/C440to218                   Scatter_SysMat                          yes
SPECTEHENaI/A218                 SysMat_withScatter                       no
SPECTEHENaI/A440                 SysMat_withScatter                       no
SPECTEHENaI/C440to218            Scatter_SysMat                           no
```

The present factors are derived only from the JSCC center-point diagnosis and
must not be transferred to EHE. The JSCC forced-window run is expected at:

```text
runs/JSCC_440keV_to_218keVwin/
```

Use only:

```text
Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat
```

Do not use that run's `SysMat_withScatter`: the raw 440 keV PE matrix does not
obey the forced 218 keV window. The layer calibration does not change this
cross-window file-selection rule.

The preferred application point is in `GenFactors/gen_factors.m`, immediately
after `flag==1` filtering and before Cartesian-to-polar interpolation. At that
point the fourth dimension contains the 10496 active scintillator rows in
their original order. Select one scale vector according to the JSCC response;
the equivalent MATLAB operation is:

```matlab
det_scin = det(scin_mask, :);
layer_scale = ones(size(det_scin, 1), 1, 'single');

switch response_name
    case 'JSCC/A218'
        scales = single([0.8336583, 0.8392622, 0.8328231, 0.8375539]);
    case 'JSCC/A440'
        scales = single([0.8095773, 0.8280269, 0.8227386, 0.8168062]);
    case 'JSCC/C440to218'
        scales = single([1.2474935, 1.2645993, 1.3180240, 1.2822758]);
    otherwise
        error('No JSCC calibration is defined for %s.', response_name);
end

layer_y = [30, 60, 90, 120];
for idx = 1:numel(layer_y)
    selected = abs(det_scin(:, 2) - layer_y(idx)) < 1e-4;
    layer_scale(selected) = scales(idx);
end

if any(layer_scale == 1)
    error('JSCC layer calibration did not classify every active row.');
end

SysMat = SysMat .* reshape(layer_scale, 1, 1, 1, []);
```

This block must be enabled only for the three listed JSCC responses; do not
insert it as an unconditional operation in the generic factor converter and
do not apply it to EHE. Detector-row scaling commutes with image-coordinate
interpolation, so applying it before polar interpolation gives the same
calibrated `SysMat_polar` as scaling the finished polar matrix. Applying it
before interpolation is simpler and also keeps `SysMat_tmp` and
`SysMat_polar` mutually consistent.

For the current JSCC `Params_Detector.dat`, the required active-row audit is:

```text
layer y (mm)   active rows      D218       D440       D440to218
30                    512      0.8336583  0.8095773    1.2474935
60                    768      0.8392622  0.8280269    1.2645993
90                   1024      0.8328231  0.8227386    1.3180240
120                  8192      0.8375539  0.8168062    1.2822758
total               10496
```

Abort factor generation if any active row has another y coordinate, if any
expected count differs, or if any `flag!=1` row is selected. Record at least
the following in the generated `factor_manifest.json`:

```text
calibration_name: JSCC_<response>_G4Center_LayerScale_20260716
calibration_scope: detector_rows
calibration_source: 1e9 Geant4 center-point response run
layer_y_mm: [30, 60, 90, 120]
layer_scale: <the response-specific vector above>
branching_ratio_included: false
```

After conversion, verify that `SysMat_tmp` and `SysMat_polar` remain finite and
nonnegative, retain their expected byte counts, and contain exactly 10496
detector rows. Every source `.sysmat` remains per emitted photon. Apply
the `225Ac` 218/440 emission ratio later in GenProj/reconstruction exactly as
before; do not fold the branching ratio into any detector-layer scale.

Do not correct only `C_intercrystal`. Holding `C_local_recoil` and
`C_highZ_to_crystal` fixed would require a total intercrystal scale near
`1.985`, but the required residual is strongly layer dependent and the Geant4
component labels are not yet exact. That correction would have weak physical
meaning and would distort the depth distribution.

All layer factors are calibrated from one FOV-center source. Before using them
as final production Factors, validate at representative off-center radial and
axial locations. If the factors remain stable, retain each four-value row
calibration. If they vary with source position, use a low-rank separable
calibration `D_detector * A * D_voxel` or a small interpolated table of
response/layer/radius/axial factors. Do not use noisy per-crystal ratios
without spatial smoothing and regularization.

Combined output is now written in fixed-size sequential chunks. This avoids a
second full-size host allocation after the scatter matrix has completed and
prevents losing only `SysMat_withScatter` at the end of a long run.

### JSCC 218/440 Response-Matrix Update

For simultaneous 218/440 keV imaging, the 218 keV measurement window should include the cross-talk response from 440 keV photons that Compton scatter or broaden into the 218 keV window:

```text
y_218win = N_decay * (Y_218 * A_218win<-218 * x_Fr
                    + Y_440 * A_218win<-440 * x_Bi) + noise

y_440win = N_decay * (Y_440 * A_440win<-440 * x_Bi) + noise
```

`Y_218` and `Y_440` are the photon emission probabilities per parent `225Ac` decay. The project convention is to use the 218 keV line from `221Fr` and the 440 keV line from `213Bi`. Common literature values are approximately `Y_218 = 0.114` and `Y_440 = 0.259-0.261`; use one consistent nuclear-data table throughout simulation, factor generation, and reconstruction. With the project default `Y_218 = 0.114`, `Y_440 = 0.261`, the relative source strength is:

```text
Y_440 / Y_218 = 0.261 / 0.114 = 2.28947
```

This branching-ratio factor is not part of the raw energy-specific system matrices. The generated matrices are per emitted photon at the specified source energy. If one wants an `225Ac`-equivalent 218-window system matrix normalized to the 218 keV yield, use:

```text
A_225Ac_218win_norm218 =
    A_218win<-218 + (Y_440 / Y_218) * A_218win<-440
```

This is the matrix that a point-source `225Ac` Monte Carlo/system-matrix measurement would estimate in the 218 window after normalizing by the number of emitted 218 keV photons, because the same run emits both 218 and 440 keV photons with their physical branching ratio. If the same measured/simulated `CntStat` already came from such a branching-ratio mixed run, do not multiply that `CntStat` by `Y_440/Y_218` again; the factor is already present in the counts.

If a simplified single-image model assumes `x_Fr = x_Bi = x`, the 218-window effective response can be written as:

```text
A_eff_218win = Y_218 * A_218win<-218 + Y_440 * A_218win<-440
```

For the intended dual-daughter model where `x_Fr` and `x_Bi` may differ spatially, keep the two terms separate in the forward model. The summed `A_225Ac_218win_norm218` form is valid for a co-located `225Ac` point-source matrix or for a simplified single-image model where the 218 and 440 source distributions are assumed identical.

The scatter energy-resolution scaling was fixed on 2026-07-09 in all scatter kernels:

```text
old: R(E') = R(E0) * sqrt(E' / E0)
new: R(E') = R(E0) * sqrt(E0 / E')
```

The new JSCC parameter sets can be regenerated with:

```matlab
cd FileGenerater_3D_Unified
generate_jscc_218_440_response_params
```

This writes parameter files to both `FileGenerater_3D_Unified/output/` and `runs/`:

```text
JSCC_218keV
  218 keV source, automatic 218 keV photopeak window.

JSCC_440keV
  440 keV source, automatic 440 keV photopeak window.

JSCC_440keV_to_218keVwin
  440 keV source, forced 218 keV energy window [196.305380, 239.694620] keV.
  This is A_218win<-440. Use Scatter_SysMat_*.sysmat only.
```

Important workflow notes:

- Recompile the ScatterGen executables before rerunning matrices; existing binaries still contain the old resolution formula.
- Existing `Scatter_SysMat_*.sysmat` and `SysMat_withScatter_*.sysmat` files must be regenerated to reflect the fix.
- For `JSCC_440keV_to_218keVwin`, run ScatterGen with the 440 keV PE matrix as input, then use the generated `Scatter_SysMat_*.sysmat` as the cross-talk matrix.
- The current 440-to-218 parameter set disables combined output, so use
  `Scatter_SysMat_*.sysmat` for that term. If combined output is enabled later,
  the direct 440 keV PE Gaussian tail will now be filtered by the forced 218 keV window.

### Scatter Performance And Multi-GPU Execution

The optimized scatter executable is built separately, so calculations already
running with `ScatterGen_CircularHole` are not modified:

```bash
./ScatterGen_RayTracing_CircularHole/bd
```

The resulting `ScatterGen_CircularHole_optimized` targets A6000 (`sm_86`) and
RTX 4090 (`sm_89`) without `--use_fast_math`. It precomputes energy-only
Klein-Nishina data and A-to-B material path lengths, performs deterministic
per-chunk A reduction, conservatively prunes impossible energy-window pairs,
and uses exact layer-grid traversal for validated axis-aligned JSCC/EHE layouts.
Unsupported geometries automatically fall back to the generic bitmap path.

To calculate one run with several GPUs:

```bash
./run_scatter_multi_gpu.sh \
  runs/JSCC_218keV_pe_v4 \
  PE_SysMat_shift_0.000000_0.000000_0.000000_v4.sysmat \
  0,1,2,3
```

Each process receives a disjoint scatter-crystal A range. Only the first process
adds the detector-local and collimator terms, so those components are not
double-counted. Partial `float32` matrices are merged in GPU-list order. The
launcher shares `Geometry_CrystalPairMaterialLengths_v1.cache` between workers.

Reference fallbacks are available for numerical comparisons:

```bash
SCATTER_COMPTON_INTEGRAND_LUT=0
SCATTER_KINEMATIC_PRUNING=0
SCATTER_STRUCTURED_TRAVERSAL=0
```

See `ScatterGen_RayTracing_CircularHole/ReadMe.txt` for cache format, source
range controls, validation tolerances, and the measured equivalence results.

## Contact

This program has been tested with an advanced SPECT system, and the analytical system matrix matches perfectly with experimental results. Calculating the photon-electric system matrix for a 200x200 2D FOV with ~6000 crystals on an RTX 6000 Ada GPU takes approximately **3 minutes**, while calculating the primary Compton scatter system matrix takes about **60 minutes**.

**Note:** There are some hard-coded elements in the current version. I apologize for any inconvenience and plan to address these issues when time permits. If you have any questions or need assistance, feel free to reach out.

- **Email:** [18772763792@163.com](mailto:18772763792@163.com) or [zhengxc21@mails.tsinghua.edu.cn](mailto:zhengxc21@mails.tsinghua.edu.cn)
- **WeChat:** zxc18772763792

## Acknowledgements

Thank you for using this tool. Contributions and feedback are welcome to help improve its functionality and performance.


## Usage Terms

This software is available **strictly for academic, research, and educational purposes only**. Commercial use is expressly prohibited without prior written permission. 

When using this work in academic publications or research, you must include proper attribution:

> [Xingchun Zheng, etc.], "GPU-Based System Matrix Calculation for SPECT/PET", GitHub repository, 2023. https://github.com/zxc18772763792/GPU-Based-System-Matrix-Calculation-for-SPECT-PET

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).


---
