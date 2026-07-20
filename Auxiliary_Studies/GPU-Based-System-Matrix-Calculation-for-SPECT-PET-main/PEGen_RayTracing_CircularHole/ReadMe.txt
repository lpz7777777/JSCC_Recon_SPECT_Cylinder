Total 4 parameter files: Param_Collimator:Param_Detector:Param_Image:Param_Physics
Param_Collimator:
Param_Collimator[0]:numCollimatorLayers
parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 0]:Number of Collimator Holes in Collimator Layer id_CollimatorLayer
parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 1]:Width  of Collimator Layer id_CollimatorLayer
parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 2]:Thickness  of Collimator Layer id_CollimatorLayer
parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 3]:Height  of Collimator Layer id_CollimatorLayer
parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 4]:Distance between1st collimator layer and collimator layer id_CollimatorLayer
parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 5]:Total Attneuation coefficient of collimator layer id_CollimatorLayer
parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 6]:PE Attneuation coefficient of collimator layer id_CollimatorLayer
parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 7]:Compton Attneuation coefficient of collimator layer id_CollimatorLayer
parameter_Collimator[id_Hole * 9 + 100]:x of hole center
parameter_Collimator[id_Hole * 9 + 101]:y1 of hole center
parameter_Collimator[id_Hole * 9 + 102]:y2 of hole center
parameter_Collimator[id_Hole * 9 + 103]:z of hole center
parameter_Collimator[id_Hole * 9 + 104]:R of hole center
parameter_Collimator[id_Hole * 9 + 105]:Total Attneuation coefficient of hole
parameter_Collimator[id_Hole * 9 + 106]:PE Attneuation coefficient of hole
parameter_Collimator[id_Hole * 9 + 107]:Compton Attneuation coefficient of hole
parameter_Collimator[id_Hole * 9 + 108]:flag

Param_Detector:
Param_Detector[0]:numDetectorBins
Param_Detector[id_Detector*12+1]:x of detector center
Param_Detector[id_Detector*12+2]:y of detector center (set y(1st collimator)=0)
Param_Detector[id_Detector*12+3]:z of detector center
Param_Detector[id_Detector*12+4]:width of detector
Param_Detector[id_Detector*12+5]:thickness of detector
Param_Detector[id_Detector*12+6]:height of detector
Param_Detector[id_Detector*12+7]:total attenuation coefficient of detector (without Rayleigh scatter)
Param_Detector[id_Detector*12+8]:photon-electrical attenuation coefficient of detector
Param_Detector[id_Detector*12+9]:compton attenuation coefficient of detector
Param_Detector[id_Detector*12+10]:energy resolution @ target PE energy
Param_Detector[id_Detector*12+11]:rotation angel of detector (y axis) [0,2pi)
Param_Detector[id_Detector*12+12]:flag

Param_Image:
Param_Image[0]:numImageVoxelX
Param_Image[1]:numImageVoxelY
Param_Image[2]:numImageVoxelZ
Param_Image[3]:widthImageVoxelX(mm)
Param_Image[4]:widthImageVoxelY(mm)
Param_Image[5]:widthImageVoxelZ(mm)
Param_Image[6]:numRotation
Param_Image[7]:angelPerRotation(0~2pi)
Param_Image[8]:shiftFOVX(mm)
Param_Image[9]:shiftFOVY(mm)
Param_Image[10]:shiftFOVZ(mm)
Param_Image[11]:FOV2Collimator0(mm)

Param_Physics:
Param_Physics[0]:flagUsingCompton
Param_Physics[1]:flagSavingPESysmat
Param_Physics[2]:flagSavingComptonSysmat
Param_Physics[3]:flagSaving PE+Compton Sysmat
Param_Physics[4]:flagUsingSameEnegryWindow
Param_Physics[5]:lowerThresholdofEnegryWindow
Param_Physics[6]:upperThresholdofEnegryWindow
Param_Physics[7]:target PE Energy
Param_Physics[8]:flagCalCulateCrystalGeometryRelationShip
Param_Physics[9]:flagCalCulateCollimatorGeometryRelationShip

Outputs:
PE_SysMat_*_v3.sysmat: unwindowed PE transport matrix used as ScatterGen input
PE_Windowed_SysMat_*_v3.sysmat: PE response accepted by the configured Gaussian energy window

PE v4 selected-pair reference model (2026-07-18):

PEGen_V4_Reference.cpp is a CPU reference implementation for one selected
detector/voxel pair. It does not replace PEGen_CircularHole and never writes a
full matrix. It integrates the visible target-box entry faces, exact dOmega,
exact target ray-box chord, the truncated-exponential first-interaction depth,
and exact ray-box attenuation through every other detector record. It currently
supports the JSCC vacuum/no-hole collimator and rejects physical collimator
holes explicitly.

Build on Linux:

  g++ -std=c++17 -O3 -I.. -o PEGen_V4_Reference PEGen_V4_Reference.cpp

Run from a JSCC run directory containing Params_*.dat:

  /path/to/PEGen_V4_Reference \
    --detector 505 --voxel 24709 \
    --surface-rule halton --face-subdiv 64 --depth-subdiv 8 \
    --v3 PE_SysMat_shift_0.000000_0.000000_0.000000_v3.sysmat

Surface rules:

  halton  four-way reflected, center-symmetric two-dimensional low-discrepancy
          points; production default because crystal shadows make the
          integrand discontinuous. Symmetry prevents a detector x-z gradient
          from finite-sample surface-centroid bias.
  gauss   composite 2x2 Gauss-Legendre nodes per face cell; use for smooth
          geometry and analytic regression checks.

Always compare at least two nested face levels. The standard JSCC validation
uses Halton 32/64 and requires <=2% relative change. Strongly shadowed pairs
should also be checked at 128. The output reports PE/Compton closure, mean first
depth, six entry-face PE components, the matching v3 element, and v4/v3.

The PE v4 reference state and quadrature implementation is in:

  common/first_interaction.h
  common/pe_v4_reference.h

common/detector_local_scatter.h uses the same FirstInteractionState data type
for its internal detector-pair integration. In the active V4-S pipeline,
ScatterGen still constructs that state from its own detector-local geometry;
PEGen does not pass a cached first-interaction state into ScatterGen.

PE v4 production model:

PEGen_V4_Production.cu generates complete raw and energy-windowed PE matrices.
It uses the same visible-face and exact target-chord equations as the CPU
reference, but replaces the all-detector attenuation scan with an x-z spatial
grid for each real detector layer. Exact ray-box chords are still evaluated for
every candidate. Both active GAGG and flag-2 tungsten target rows are retained
because ScatterGen uses the tungsten rows as first-interaction source terms.
The N*N face samples are generated as Halton points plus all reflections about
u=0.5 and v=0.5 (and one center point when N is odd), so their surface centroid
is exactly centered for every supported N.

Build on Windows:

  ./build_pe_v4_production.ps1

Build on Linux with ./bd, or directly:

  nvcc -std=c++17 -O3 -lineinfo -arch=sm_89 \
    -o PEGen_V4_Production PEGen_V4_Production.cu

Run from a JSCC directory containing Params_*.dat:

  /path/to/PEGen_V4_Production \
    --cuda 0 --face-subdiv 16 --rows-per-chunk 4 \
    --samples-per-launch 32

Default outputs:

  PE_SysMat_shift_*_v4.sysmat
  PE_Windowed_SysMat_shift_*_v4.sysmat
  PE_v4_progress.json
  PE_v4_progress.tsv
  PE_v4_manifest.json

Use --resume to continue matching .partial files. Resume is accepted only when
both output files exist, have equal sizes, and end on a complete detector-row
boundary. Use monitor_pe_v4.ps1 on Windows or monitor_pe_v4.sh on Linux.

The current production model supports the zero-hole JSCC collimator. It does
not apply intrinsic response. In the active V4 pipeline, ScatterGen consumes
the unwindowed PE-v4 scalar matrix and derives first-Compton probability as
`PE_probability * mu_compton / mu_photoelectric`. Its detector-local lookup
integrates first positions internally for each crystal geometry and incoming
direction. No PEGen first-interaction-state cache is passed into ScatterGen.

JSCC density alignment:

  Geant4 GAGG = 6.60 g/cm^3
  Geant4 W    = 19.35 g/cm^3

Run tests/validate_jscc_material_density_alignment.py before production. It
checks Geant4 source definitions, the generated XCOM CSV and CUDA header, and
all three _pe_v4 Params_Detector.dat files. The top-level production pipeline
runs this validation automatically and rebuilds both PE v4 and ScatterGen from
the current source before starting matrix calculations.
