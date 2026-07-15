# GAGG intrinsic photopeak-containment study

This is a deliberately small Geant4 project for measuring the physical
full-energy containment that is missing from the analytical system-matrix
model. It contains only a vacuum world and one GAGG:Ce crystal. It does not
model the JSCC collimator, shielding, neighboring crystals, optical photons,
or electronic energy-resolution broadening.

The GAGG definition is identical to `Geant4Code`:

- base stoichiometry: `Gd3Al2Ga3O12`;
- base/cerium mass fractions: 99%/1%;
- density: `6.6 g/cm3`;
- physics: `FTFP_BERT` with `G4EmStandardPhysics_option4`;
- fluorescence, Auger emission, and PIXE enabled;
- production cut: `0.1 um`.

## What is measured

Monoenergetic gamma rays illuminate the complete `-Y` crystal face with a
uniform position distribution and normal `+Y` incidence. Energy deposition
from every track is accumulated only while that track is inside the crystal.
The primary gamma process sequence is recorded independently.

An event is physically contained when

```text
same-crystal deposited energy >= primary energy - fullEnergyTolerance
```

The default tolerance is `1 eV`. No Gaussian detector blur is applied. Three
conditional containment probabilities are reported:

1. `first_pe_containment`: the primary gamma's first interaction is PE.
2. `first_compton_second_pe_containment`: the first interaction is Compton and
   the very next primary-gamma interaction is PE. This is the history modeled
   by the current detector-local `Compton -> PE` term.
3. `first_compton_eventual_pe_containment`: the first interaction is Compton
   and the primary gamma eventually undergoes PE after one or more later
   interactions. This is a higher-order diagnostic and must not be applied to
   a first-order matrix term.

Rayleigh-first events are classified as `first_other`. Escaped fluorescence
X-rays, electrons, or scattered photons reduce the containment probability
naturally.

## Cases

The `macros` directory contains `1e7`-event production runs for the two JSCC
GAGG crystal types and both primary energies:

```text
GAGG_3x3x3_218keV.mac
GAGG_3x3x3_440keV.mac
GAGG_2x6x2_218keV.mac
GAGG_2x6x2_440keV.mac
```

Here dimensions are `(width X, thickness Y, height Z)`. The beam travels along
the thickness axis, matching the detector-local coordinate convention. A
`10000`-event `smoke.mac` is included for build validation.

## Build

In a shell where Geant4 has been configured, run:

```bash
cd Geant4Sim/Geant4Code_GAGGIntrinsicResponse
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd build
./gagg_intrinsic smoke.mac 2
```

The smoke run should create `smoke_GAGG_3x3x3_218keV.csv`. Check that:

- `events` is 10000;
- `entered` is 10000;
- all category counts are nonnegative;
- every reported containment probability is between 0 and 1;
- `first_compton_second_pe <= first_compton_eventual_pe <= first_compton`.

### Windows and Visual Studio

Visual Studio is a multi-configuration generator. In the original CMake
layout it placed the executable in `build/Debug` or `build/Release`, while the
macro files were copied to `build`. The current CMake configuration overrides
all runtime output directories so a newly configured build places
`gagg_intrinsic.exe` directly beside the macros in `build`.

Reconfigure once after pulling this change, then build and run from PowerShell:

```powershell
cmake -S . -B build
cmake --build build --config Release
Set-ExecutionPolicy -Scope Process Bypass
.\run_windows.ps1 -Configuration Release -Macro smoke.mac -Threads 2
```

The same script is copied into `build`, so this is also valid:

```powershell
cd build
Set-ExecutionPolicy -Scope Process Bypass
.\run_windows.ps1 -Configuration Release -Macro smoke.mac -Threads 2
```

The script reads `Geant4_DIR` from `CMakeCache.txt`. If the installation has a
`bin/geant4.bat`, the script imports its DLL and physics-dataset environment;
otherwise it adds the corresponding Geant4 `bin` directory to the
process-local `PATH`. It then runs with `build` as the working directory and
verifies that the smoke CSV was created. It does not modify the system-wide
environment.

For an existing build that has not been reconfigured, do not copy the EXE.
Run it with an explicit macro argument from the build directory:

```powershell
cd build
.\Debug\gagg_intrinsic.exe .\smoke.mac 2
```

The executable is a batch program, not a double-click application. Starting it
without `macro.mac` intentionally prints its usage and exits. If Windows
reports `0xC0000135`, a Geant4 DLL could not be found; use the script above or
run from the Geant4 command prompt that was used to configure the project.

## Production runs

The optional second executable argument is the worker-thread count for an MT
Geant4 build. It is ignored by a sequential Geant4 build. On a machine with at
least 32 available CPU cores, the four cases can run concurrently:

```bash
./gagg_intrinsic GAGG_3x3x3_218keV.mac 8 > GAGG_3x3x3_218keV.log 2>&1 &
./gagg_intrinsic GAGG_3x3x3_440keV.mac 8 > GAGG_3x3x3_440keV.log 2>&1 &
./gagg_intrinsic GAGG_2x6x2_218keV.mac 8 > GAGG_2x6x2_218keV.log 2>&1 &
./gagg_intrinsic GAGG_2x6x2_440keV.mac 8 > GAGG_2x6x2_440keV.log 2>&1 &
wait
```

Reduce the command-line thread count or run cases sequentially when the server
has fewer cores. The output filenames are distinct, so concurrent runs
from one build directory do not collide.

Build the compact lookup after all four CSV files exist:

```bash
python ../build_containment_lookup.py lookup_GAGG_*.csv \
  --output gagg_intrinsic_containment_lookup.csv
```

The lookup includes the conditional denominator, contained count, probability,
and binomial standard error for every size/energy/history combination. By
default the builder requires exactly all four production combinations and
checks the event/category count invariants; `--allow-partial` is available only
for smoke/debug data. Preserve
the four raw CSV files alongside the compact lookup.

## How the lookup will be used

Do not use one common empirical scale for all direct-response components.

- PEGen's first-photoelectric response should use `first_pe_containment` for
  the matching crystal size and source energy.
- Detector-local `Compton -> PE` should use
  `first_compton_second_pe_containment` for the matching size and energy.
- `first_compton_eventual_pe_containment` quantifies histories that the current
  first-order local model does not generate; it is not a substitute for an
  explicit higher-order model.

The current analytical lookup still applies the Gaussian energy-window
acceptance separately. Therefore these Geant4 values must represent physical
same-crystal containment before electronic broadening, as this project does.

## Scope and next validation

The production macros measure normal incidence. That is the first lookup
needed for the center-source discrepancy and is expected to capture the main
fluorescence/electron escape effect. If the corrected center response remains
angle dependent, extend this project with oblique incidence and face-projected
sampling before generating another full matrix.

This repository does not currently contain a usable Geant4 installation, so
the C++ target has not been compiled locally. Compile and run `smoke.mac` in the
same Geant4 environment used for the JSCC response study before starting the
four `1e7` cases.
