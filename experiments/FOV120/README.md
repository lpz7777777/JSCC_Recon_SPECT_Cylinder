# FOV120: 218/440 keV axial extension

Detailed Chinese inventory and progress: [experiment status](../../docs/FOV120_EXPERIMENT_STATUS.md).
Cross-project authentication and resource usage: [safe access](../../docs/REMOTE_COMPUTE_ACCESS.md).

## Latest (2026-09-25)

All matrices and full four 1e9 calibration/sensitivity collections are complete.
Production calibrated Factors are installed on 65114 with raw responses retained.
Full data and calibration report are locally in `generated/FullData/`.
Contrast 1e9 truth and noiseless/Poisson projections have been generated on 65114;
new Sensi_d and independent closure also completed (ratio 1.002033, CV 0.148924%).
Sensi_d plus provenance are installed in production Factors. Closed-loop reconstruction,
scxi717 deployment, full-event/multi-node checks and phantom imaging remain pending.
See the linked status page for verified hashes, coefficients and subsequent updates.

## Historical: calibration extension (2026-09-24 evening)

All three raw matrices and raw polar Factors are complete and validated.
The four pilot groups have been collected with worker/output hash checks,
actual primary-count closure (4e8 total) and 40 globally distinct seeds.
Pilot files are locally in `generated/PilotData/`; a copy on 65114 is used for
model-response assessment. No pilot-calibrated Factors have been installed.

`assess_pilot.py --raw-root <FactorsRaw> --data-root <PilotData> --output <new.json>`
reproduces layer statistics and candidate scale factors. At 1e8 photons, the
largest per-layer Poisson relative SE is 0.6847%. Candidate layer scales are
0.8603–0.8784 (218), 0.8684–0.8905 (440), and 1.1487–1.2519 (440→218).
These are calibration fit estimates, not independent spatial validation.

Extension array **15384175** runs 360 independent workers, 1e7 photons each,
with at most 40 concurrent. Indices are `10-99,110-199,210-299,310-399` from the
original immutable task manifest. Each of the four groups reaches 1e9 photons
when its pilot and extension are combined. Dependent collection job **15384216**
runs `maty_collect.sh` only after the whole extension array succeeds; it verifies
all 400 workers and creates four `_all` collections. If dependency failure occurs,
inspect worker records before retrying; never resubmit successful indices.

The measured pilot worker durations were about 17.5–19 minutes per 1e7 photons.
Nine waves at 40-way concurrency suggest about three hours excluding queueing
and changed node performance. The full-data calibration and Sensi_d are still
pending those collections.

Full central-20-layer regression passed for all three raw responses: the two
direct-channel combined matrices match byte-for-byte; the cross-window scatter
matrix has relative L2 error 3.4740e-8 and max absolute error 9.0949e-13,
below the 1e-5 gate. Cross-window hashes differ, so retain the numerical report
instead of claiming exact identity for all three responses.

This experiment keeps the four-layer detector, PE-v4 response, 20 views and
3-mm axial sampling. It uses 40 axial layers / 51240 polar cells. Physical
sources occupy R<=150 mm; full computational support and Sensi_d calibration
use R=153 mm. The source is a two-gamma proxy in vacuum, not a decay-chain or
patient attenuation simulation. Both new phantom channels use a common activity
scale, with gamma yields 0.114/0.259. Historical 0.261 datasets retain their
original convention.

`config.json` is the experiment specification. Large regenerable artifacts are
under `generated/` (ignored by Git). Original Factors and 60-mm data are not
replaced. Existing XCAT or task manifests cannot be overwritten by these tools.

## Verified locally, 2026-09-24

- Three 51x51x40 matrix parameter sets generated successfully with MATLAB R2022b.
- Analytic 20/40-layer Factors conversion passed: interpolation, exact cell
  volume, detector y translation to 200/230/260/290 mm, rotation inverses.
- XCAT 80x200x200 native crop -> 40x100x100 truth, 22 macros, source-integral
  and all-view geometric checks passed. Maximum macro/truth activity error
  1.67e-7. Full-XCAT kidney-label retention 81.91%; both axial boundaries still
  intersect kidneys; lesion ROI is fully within the axial crop.
- Rebuilt Geant4 11.1.1. Six 10000-photon smoke runs passed: pure 218, pure 440,
  physical uniform, contrast, XCAT views 1 and 11. Outputs include actual
  PrimaryCount and detector/list checksums. These are not production data.
- All 31 Python unit/regression tests and two-process CPU GLOO comparison passed.
  CPU collectives do not replace the cluster NCCL/GPU smoke test.
- XCAT cell-volume quadrature matches the hybrid MC source (3-mm interior,
  1.5-mm boundary). At 16 samples/axis the largest whole-source integral error
  is about 0.0054%.
- On `65114_lipeize`, fresh Linux binaries compiled successfully. The complete
  218-keV PE matrix (4,794,163,200 bytes) finished in 367.94 seconds. A full scan
  found only finite nonnegative values; its central 20 layers have the same
  SHA256 as the entire old 60-mm PE matrix, across all 11520 raw detector rows.
  This validates the unwindowed PE regression, not the pending scatter response.

**Running remotely:** the three production response matrices on the available
RTX A6000 GPU 0, followed automatically by MATLAB raw-Factors conversion and
geometry/numerical checks. Root: `/home/lipeize/JSCC_FOV120_20260924` on SSH alias
`65114_lipeize`. Matrix PID at launch: 2800483; conversion dependency PID: 2802514.
Other GPUs were occupied and were not used. Run
`python experiments/FOV120/remote_status.py` for fresh read-only status.

**Not completed:** high-statistics calibration, new Sensi_d, 1e9/1e10 phantom
simulations and reconstructed images. The supercomputer connection and Geant4
environment have now been verified as described below.
`generated/execution_status.json` records evidence;
`generated/remote_status.json` is a timestamped progress snapshot,
not a completion certificate.

The user identified the Geant4 supercomputer as `maty@192.168.11.1:22`,
with the existing project at
`/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/Geant4Sim`.
Authentication now works through the Windows SSH agent after the user unlocked
the existing private key. The login node is `ibcln01`; the original directory
contains the historical dual-energy 1e9/1e10 runs. This is a different host
from the previously attempted ParaCloud endpoint.
The latest matrix snapshot shows both PE runs complete (367.94/371.96 seconds)
and the 218-keV scatter calculation running, with raw Factors still pending.

The isolated supercomputer deployment is the sibling directory
`JSCC_SPECT/FOV120_20260924`. It contains current sources and the frozen task
manifest; the original `Geant4Code` and `Geant4Sim` are unchanged. Use
`python experiments/FOV120/cluster_status.py` to read progress. Cluster scripts
use `cnmix`, GCC 12.2.0, CMake 3.25.2 and Geant4 11.1.0 (local smoke used 11.1.1).
The first build job 15376310 failed before simulation because CMake selected
system GCC 4.8.5; `maty_build_smoke.sh` now supplies explicit compiler paths.
Replacement build/smoke job 15376312 completed all six 10000-photon checks,
including actual primary-count closure and detector output shape/hash checks.
Pilot array 15377351 was submitted with the indices below (4e8 total photons).
Explicit `--mem=4G` was rejected by this
partition at submission, so scripts follow the existing project's memory defaults.
Python 3.10.4/numpy 1.24.3 are available; SHA256 streaming supports that Python.

After six cluster smoke checks pass, submit the four independent monoenergetic
pilot groups (1e8 photons/group, 40 workers total, at most 20 concurrent):

```bash
sbatch --parsable --array=0-9,100-109,200-209,300-309%20 experiments/FOV120/maty_pilot.sh
```

This command is run from the isolated cluster root. The script verifies smoke
completion and executable/crystal hashes before each worker starts. Do not
resubmit completed worker indices: their output directories are immutable.

## 1. Parameters and raw matrices

### Reconstruction supercomputer deployment

Password authentication has been verified for `scxi717@BSCC-N56R5` through
`ssh.cn-zhongwei-1.paracloud.com:22`, reaching `scxi717@ln01`. The isolated
deployment is `/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/FOV120_20260924`. Existing projects and
the running `JSCC-PoissonAB` job were not changed. Source transfer used SFTP;
credentials stay on the Windows host and are not included in deployment bundles.

Account association reports `GrpTRES=gres/gpu=100`; QoS `gpugpu` reports
`MaxTRESPJ=node=8`. Thus the login banner's generic 16-GPU limit does not describe
this account. The 5090 partition configuration has eight GPUs per node. The
planned four-node/eight-GPU-per-node allocation is within those reported limits,
but depends on availability and the remaining scheduler constraints.

Original smoke job **1623854** failed at startup (`module: command not found`,
exit 127:0); no numerical check ran. Scripts now source `/etc/profile.d/modules.sh`
explicitly before loading modules. Replacement two-GPU NCCL numerical smoke job
**1624002** was submitted from the relocated workspace with a ten-minute limit. This checks single-photon, Compton and joint MLEM against
serial reference calculations; it does not replace full FOV120 data validation
or cross-node communication tests. CPU/GLOO equivalence was rerun and passed
after adding the optional CUDA/NCCL backend.

Use `reconstruction_ssh.py --command '<read-only status command>'` locally for
access with the Windows-encrypted credential. The helper requires Paramiko and
known host-key verification; it never passes the password on a command line.

Run commands from the repository root. Python needs numpy, scipy, matplotlib,
torch, and pytest for tests; MATLAB uses the existing project toolboxes.

```matlab
addpath('Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main/FileGenerater_3D_Unified');
generate_jscc_218_440_response_params('_pe_v4_FOV120', ...
    fullfile(pwd,'experiments','FOV120','config.json'));
```

Build current PE-v4 and detector-local ScatterGen on the compute host using
their existing project build instructions. Do not use the historical v3 PE
binary or the collimator-only scatter binary. Then:

```bash
python experiments/FOV120/run_matrices.py --pe /path/to/PEGen_V4_Production --scatter /path/to/ScatterGen_CircularHole_detector_local --cuda 0
```

The three run directories end in `_pe_v4_FOV120`. Each complete raw matrix is
4,794,163,200 bytes. The cross response reads only Scatter_SysMat; its input PE
is the 440-keV unwindowed v4 response. Keep all Params and PE manifests with
their outputs. Do not regenerate Params over a previously computed run with
different geometry. The runner rejects stale/incomplete outputs.

Run `compare_center_matrices.py --old <60mm-raw> --new <120mm-raw>
--output <report.json>` separately for the direct and cross responses. It
compares new z indices [10,30) against the old 20 layers without calibration.
The default relative L2 tolerance is 1e-5; a failure requires investigation,
not silently loosening the tolerance.

```matlab
addpath('Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main/GenFactors');
run_gen_fov120_factors;
```

This creates `generated/FactorsRaw/{218keV_RotateNum20,440keV_RotateNum20,
440keV_to218win_RotateNum20}` with density weighting but no historical layer
calibration. Each polar matrix is 2,151,260,160 bytes. Factor conversion reads
float32 directly; allow additional host RAM for interpolation/permutation.

## 2. Sources, workers and data collection

```bash
python Geant4Sim/generate_xcat_ac225_psma_abdomen.py --config experiments/FOV120/config.json
python Geant4Sim/validate_xcat_ac225_psma_abdomen.py --directory experiments/FOV120/generated/XCAT
python experiments/FOV120/workflow.py prepare
python experiments/FOV120/workflow.py prepare-points
```

Already generated outputs need not be regenerated. Copy `generated/XCAT` and
the task/macro directories to the compute host; job macro paths are portable
relative paths. XCAT source provenance retains the original absolute input
path and SHA-256. Large XCAT source data are needed only for regeneration.

`generated/Simulation/jobs.json` has 1762 tasks:

| Indices | Dataset |
|---|---|
| 0–99 | pure-218 calibration (first 10 pilot, remaining 90 extension) |
| 100–199 | pure-440 calibration (same split) |
| 200–299 | pure-440 Sensi_d source |
| 300–399 | independent pure-440 Sensi_d validation source |
| 400–561 | single-view point-response scans |
| 562–961 | Uniform: first 200 tasks 1e9, next 200 tasks 1e10 |
| 962–1361 | Contrast: same count split |
| 1362–1761 | XCAT: same count split |

Calibration/sensitivity pilots have 1e8 photons and extensions add 9e8. Points
span z=0,+/-15,+/-30,+/-45,+/-57 mm; r=0,75,135 mm; four noncentral azimuths.
`generated/PointImaging/jobs.json` separately contains 3240 tasks: 162 points,
20 views, 500000 photons/view (1e7 per point). These permit positioning/PSF
reconstruction; the single-view response scans alone do not.

Each task has an independent reproducible seed and worker directory. Run a
short test (separate smoke directory) before production:

```bash
python experiments/FOV120/workflow.py run --index 1362 --smoke --executable /path/to/gamma01 --crystal /path/to/CrystalMatrix.txt
```

The Geant4 environment/DLL search path must already be configured. On Windows,
add the Geant4 bin directory to PATH in the current shell and retain the G4 data
variables. Workers never append to a pre-existing directory. Failed workers
remain available for diagnosis; use a fresh task output location after repair.

For SLURM, set `JSCC_REPO_ROOT`, `JSCC_G4_EXECUTABLE`, `JSCC_CRYSTAL_MATRIX`,
and submit selected indices with `geant4_array.sh`. Start with pilots; do not
submit all count levels automatically. Select the point-imaging manifest using
`FOV120_JOB_MANIFEST`. Match CPU/memory/time/partition options to the host.

After pilot+extension complete:

```bash
python experiments/FOV120/workflow.py collect --dataset calibration_218 --level all
python experiments/FOV120/workflow.py collect --dataset calibration_440 --level all
python experiments/FOV120/calibrate.py
python experiments/FOV120/workflow.py collect --dataset sensitivity_440 --level all
python experiments/FOV120/workflow.py collect --dataset sensitivity_validation_440 --level all
python experiments/FOV120/run_sensitivity.py --device cuda
```

Collection rejects missing/failed/changed workers, duplicate seeds, wrong
detector/executable hashes and smoke photon counts. CntStat is written as
views x detector bins, matching the existing loader. List files are streamed
per view. Calibration fits absolute independent layer scales and requires
<=1% Poisson relative SE per layer; no 10496-row fitting. `FactorsRaw` is kept.

The Sensi_d wrapper uses separate simulation seeds, identical K*B response and
already-smeared energies. It runs the existing absolute normalization check
and an independent uniform-image closure diagnostic, then installs a new
51240-element sensitivity with provenance. Review the spatial closure report:
writing Sensi_d is not a claim that every spatial bias has passed validation.

Collect imaging datasets separately, e.g. `workflow.py collect --dataset XCAT
--level 1e9`. Keep simulation budgets separate from calibration budgets.

## 3. Closed loop and reconstruction

```bash
python experiments/FOV120/imaging.py truth --dataset Uniform --output experiments/FOV120/generated/Truth_Uniform_1e9.npz
python experiments/FOV120/imaging.py genproj --truth experiments/FOV120/generated/Truth_Uniform_1e9.npz --output experiments/FOV120/generated/GenProj
python main_local_multi_energy_cntstat.py --factors-dir experiments/FOV120/generated/Factors --cntstat-dir experiments/FOV120/generated/GenProj/Noiseless --data-file-name Closure --count-levels 1e9 --pixel-num-layer 1281 --pixel-num-z 40 --single-sc-iter 1000 --single-sc-save-step 50 --output-root experiments/FOV120/generated/ClosedLoop/Noiseless
```

Repeat with `GenProj/Poisson` and a separate output directory. GenProj remains
separate from Geant4 validation. For common-region regression, reconstruct the
old ContrastPhantom data twice with original and FOV120 Factors; never relabel
that source as the 120-mm extended phantom.

Before cluster production:

```bash
python experiments/FOV120/validate_cpu_collectives.py
python distributed/dual_energy_compton_python/preflight.py --experiment-config experiments/FOV120/config.json --factors-dir experiments/FOV120/generated/Factors --cntstat-dir experiments/FOV120/generated/CntStat --list-dir experiments/FOV120/generated/List --data-file-name XCAT --count-level 1e9 --world-size 2
torchrun --standalone --nproc_per_node=2 distributed/dual_energy_compton_python/main_dist_dual_energy_compton.py --experiment-config experiments/FOV120/config.json --factors-dir experiments/FOV120/generated/Factors --cntstat-dir experiments/FOV120/generated/CntStat --list-dir experiments/FOV120/generated/List --data-file-name XCAT --count-level 1e9 --iterations 2 --save-step 1 --max-events-per-view 256 --output-dir experiments/FOV120/generated/Results/XCAT_GPU_smoke
```

The short GPU job checks six outputs/collectives, not image quality. The CPU
FileStore helper avoids the Windows torchrun/libuv launcher limitation.

For full production use `reconstruct.sh`: set `FOV120_DATASET`,
`FOV120_COUNT_LEVEL`, `FOV120_ACCEPTED_EVENTS` (estimated from the new pilot),
and `FOV120_GPU_GIB` (actual GPU capacity). The default is 4x8 GPUs, 1000 MLEM,
save every 50 iterations. Preflight reserves 20% device memory plus a 4-GiB
workspace allowance; verify actual peaks in the smoke job. Each rank also
loads about 6 GiB of matrix host storage before copies/framework overhead.
The script inherits the existing cluster's module/partition defaults; adapt
those site settings only, leaving physical-response parameters fixed.

For a collected monoenergetic 20-view point, use `reconstruct_point.py
--factor <energy-Factors> --projection <point-CntStat.csv> --position-mm x y z
--output <fresh-directory>`. Its axial FWHM is reported as censored if either
half-maximum crossing lies outside the grid.

## 4. Full-FOV evaluation and acceptance

```bash
python experiments/FOV120/imaging.py truth --dataset XCAT --photons 10000000000 --quadrature 16 --output experiments/FOV120/generated/Truth_XCAT_1e10.npz
python experiments/FOV120/imaging.py evaluate --result <six-output-result-dir> --truth experiments/FOV120/generated/Truth_XCAT_1e10.npz
```

Truth uses cell-volume quadrature in r²/theta/z; the R=150 boundary intersection
is exact, and organ/rod intersections are numerically integrated. Inspect the
reported integral errors and compare q=8/16 before quantitative interpretation.
XCAT sampling follows the same 3-mm interior/1.5-mm boundary representation as
the emitted cuboids, rather than spreading partial boundary activity over a
complete 3-mm cell.
No branch ratio is applied to Factors. `rho` is emitted photon density for the
whole acquisition; the projector divides by 20 for each view.

Evaluation reads every saved frame and outputs full-height orthogonal planes,
coronal MIPs, iteration galleries, central/middle/edge metrics, background CV,
CNR/CRC, organ volume-weighted integrals, and ROI excess-centroid error. Images
are unfiltered and uncropped, with gray_r and a shared truth intensity range;
raw reconstruction arrays are unchanged. CNR/CRC are null when reference
contrast or background variance makes them undefined.

Infrastructure completion, independent spatial validation and acceptable
effective FOV are distinct outcomes. In particular, current SC and List
channels can overlap statistically, and the inherited K*B response is an
approximation. Report residual bias and edge limitations; do not treat the
sum of gamma channels as a quantitative parent-225Ac activity map.

Tests:

```bash
python -m pytest tests/test_fov120.py tests/test_distributed_dual_energy_compton.py tests/test_cntstat_crosstalk.py tests/test_compton_density_basis_closure.py tests/test_compton_operator_equivalence.py -q
python experiments/FOV120/validate_cpu_collectives.py
```

MATLAB: `addpath('experiments/FOV120'); test_factor_grid;`.
