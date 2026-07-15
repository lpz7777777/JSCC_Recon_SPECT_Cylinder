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
Note: this is relative FWHM at E0=Param_Physics[7]. Scatter photons use R(E')=R(E0)*sqrt(E0/E').
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
Param_Physics[10]:flagDetectorRecoilEscapeResponse
Param_Physics[11]:flagSelfComptonPhotoelectricResponse

Physics data:
Photoelectric and incoherent Compton coefficients for NaI, GAGG, Pb, and W are
linearly interpolated from the embedded 1-1000 keV NIST XCOM table. The input
PE matrix must be the unwindowed PE_SysMat_*_v3.sysmat transport matrix.

Collimator scatter uses physical high-Z volume cells covering the plate minus
the aperture area. Incident and outgoing attenuation through the plate depth
are integrated analytically. Optional convergence controls:
COLLIMATOR_SCATTER_SAMPLES_PER_LAYER and COLLIMATOR_SCATTER_AREA_SUBDIV.

Detector-local scatter after the first Compton interaction in active crystal A:
  P_escape    = exp[-(mu_PE(E')+mu_C(E'))*L_A]
  P_second_PE = (1-P_escape)*mu_PE(E')/(mu_PE(E')+mu_C(E'))
  P_second_C  = (1-P_escape)*mu_C(E') /(mu_PE(E')+mu_C(E'))
These three mutually exclusive probabilities are checked to sum to one.
Physics[10] records E0-E' in A for the escape branch after Gaussian energy
windowing. It integrates all escape directions once per A/image pair and is
not multiplied by the number of possible destination crystals B.
Physics[11] records E0 in A for exactly one Compton followed by PE in A, with
one Gaussian broadening operation at E0. The A-to-B kernel separately records
the terminal PE pulse in B and now includes survival from A's center to its
boundary at E'. A single A-to-B event may therefore make pulses in both A and B.
Both local switches are subordinate to Physics[0]; disabling global Compton
suppresses both terms even if Physics[10:11] are one.

The calculation stops after a second Compton interaction; that branch is
included in the probability partition but is not transported. First-interaction
position and A-exit length use the existing crystal-center approximation.
Only detector flag=1 records produce local pulses. EHE bins are independent
analytical channels; continuous-crystal Anger light sharing and centroiding
are not modeled.

Detector-local lookup convergence controls:
DETECTOR_LOCAL_SCATTER_ORIENTATION_BINS (default 17)
DETECTOR_LOCAL_SCATTER_COSINE_SAMPLES   (default 96)
DETECTOR_LOCAL_SCATTER_AZIMUTH_SAMPLES  (default 96)

ScatterGen remains backward-compatible with 10-float Params_Physics.dat files:
the zero-initialized missing Physics[10:11] values disable both new terms.
All scatter/combined matrices must be regenerated to include these responses
and the new A-exit attenuation.
Multi-rotation ScatterGen runs now use the corresponding PE matrix slice for
each angle; older builds incorrectly reused rotation 0 PE for every angle.

Performance implementation (2026-07):

Build a precise-math binary for the GPUs used on this workstation:

  ./bd

This creates ScatterGen_CircularHole_optimized with native sm_86 (A6000) and
sm_89 (RTX 4090) code. The build deliberately does not use --use_fast_math.
The original ScatterGen_CircularHole filename is not overwritten.

The optimized crystal-scatter path preserves the existing physical model and
center-point approximation while removing repeated work:

  * The fixed Klein-Nishina normalization is evaluated once per run.
  * A 1e-5-radian integrand lookup removes per-sample sin/cos calls while
    preserving the legacy 0.01-radian summation order. A 2048-interval runtime
    validation aborts if the lookup differs beyond its configured tolerance.
  * Every A-to-B pair stores path length through NaI, GAGG, Pb, and W once;
    each voxel then applies the energy-dependent interpolated XCOM coefficient.
  * One (B,voxel) worker accumulates the current A chunk and writes once, rather
    than issuing one atomic output update per A.
  * A conservative FOV/energy-window bound rejects only A-to-B pairs that
    cannot pass the existing two-sigma energy prefilter for any image voxel.
  * Axis-aligned uniform detector layers use exact grid traversal. Geometry
    that does not pass all layer/grid checks automatically uses the generic
    bitmap implementation. Final box intersections use the same expression as
    the generic path.

Default controls and exact-reference fallbacks:

  SCATTER_CRYSTAL_CHUNK=64
  SCATTER_COMPTON_INTEGRAND_LUT=0    use legacy per-thread trigonometric sum
  SCATTER_KINEMATIC_PRUNING=0        disable conservative pair rejection
  SCATTER_STRUCTURED_TRAVERSAL=0     force generic bitmap geometry

An optional material-length cache can be shared by 218, 440, and 440-to-218
runs with identical detector geometry and material labels:

  SCATTER_PAIR_LENGTH_CACHE=/path/Geometry_CrystalPairMaterialLengths_v1.cache

The file contains a geometry/material hash and per-A committed-row flags. For
11520 detector records its fully populated data area is about 1.98 GiB; it is
created as a sparse file and only completed rows are read. Energy-window flags,
attenuation coefficients, and response probabilities are never cached.

One matrix can be partitioned over several GPUs by scatter crystal A:

  ./run_scatter_multi_gpu.sh RUN_DIR PE_MATRIX 0,1,2,3

The launcher sets SCATTER_CRYSTAL_START/SCATTER_CRYSTAL_END for each worker,
computes detector-local and collimator components only in worker zero, shares
the material-length cache, and merges float32 partial matrices in deterministic
GPU-list order. It retains per-partition logs under RUN_DIR/.scatter_partials.*.
For manual partitioning, SCATTER_INCLUDE_GLOBAL_COMPONENTS=1 enables the two
non-partitioned components in exactly one worker.

Validation covers NaI and mixed GAGG/W detectors, direct 218/440 windows,
440-to-218 cross-talk, two-layer synthetic grids, actual 11520-record JSCC
geometry, cache populate/reuse, and source-range partition merging. Structured
and generic paths are element-identical on the actual fixed-A JSCC geometry.
Against the preserved baseline, the largest observed synthetic-grid difference
is 1.82e-12 absolute and 1.90e-7 relative. These are floating-point lookup and
reduction-order effects; no interaction probability or attenuation path is
discarded or approximated beyond the model already documented above.
