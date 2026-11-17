# JSCC v5 uncertainty audit and isolated equivalent optimization

Read [REPORT.html](REPORT.html) for the Chinese scientific report, measured tables,
limitations and recommendations. This directory is an isolated local study.
No production source, release, event selection, S, image, checkpoint or job is changed.

## Delivered implementation

- `adapters.py`: bounded fixed-prefix/all-view cache around the SHA-verified frozen
  `ViewResponse`; explicit per-energy/phase ownership; ordered 8×32 transfer grouping,
  two bounded pinned buffers and two CUDA buffers; copy/compute events and lifetime guards.
- `bench.py`: paired baseline/cache/grouped synchronous/grouped prefetch tests.
  Full 132040/78920 production geometry, **manufactured** SC and event matrices/counts.
  The Compton S is the independently calibrated frozen S, never `C.T @ ones`.
  This deliberately is NOT a physically matched toy reconstruction or a quality claim.
- `real_kernel_patch.py`: actual frozen 260 List events, actual detector and 44
  candidate locations around six sampling transitions, unchanged v5 energy factor.
  Matching full B is absent locally, so this is **not complete C=K×B**.
- `component_bench.py`: completed-operation timing (CUDA Events when available),
  staging, H2D calls/bytes, forward/backprojection and total wall time.
- `gpu_stress.py`: bounded CUDA stream/lifetime tests, including non-default compute
  stream, >2 groups, empty/tail blocks, early close and buffer reuse. **Not run here.**
- `verify.py`: original frozen SC saved-callback loop with zero/fixed background,
  original save50 order, paired results, and final immutable-input SHA checks.

## Reproduce locally from the project root

Python requires existing NumPy, SciPy, PyTorch, psutil and matplotlib. No installation
or precision-setting changes are made. Recorded machine: PyTorch 2.14.0+cpu, no CUDA.
The frozen executed code is imported only after SHA verification; similarly named
top-level files are not used as substitutes.

```powershell
python -B -X utf8 docs/research/jscc_uncertainty_optimization_20261115/identity_audit.py
python -B -X utf8 docs/research/jscc_uncertainty_optimization_20261115/extract_point_sources.py
python -B -X utf8 docs/research/jscc_uncertainty_optimization_20261115/uncertainty_audit.py
python -B -X utf8 docs/research/jscc_uncertainty_optimization_20261115/additional_evidence.py
python -B -X utf8 docs/research/jscc_uncertainty_optimization_20261115/bench.py
python -B -X utf8 docs/research/jscc_uncertainty_optimization_20261115/real_kernel_patch.py
python -B -X utf8 docs/research/jscc_uncertainty_optimization_20261115/component_bench.py
python -B -X utf8 docs/research/jscc_uncertainty_optimization_20261115/verify.py
python -B -X utf8 docs/research/jscc_uncertainty_optimization_20261115/build_report.py
python -B -X utf8 docs/research/jscc_uncertainty_optimization_20261115/finalize.py
```

The point-source extraction reads an existing 1.6GB archive sequentially and retains
only 140 small source-coordinate summaries. Every member used is checked against its
delivered SHA. It does not duplicate the archive or perform simulation. If the existing
`point_source_truth.json` is retained and its manifest still matches, extraction need
not be repeated. Research reports/JSON in this directory can be regenerated;
formal inputs cannot. The benchmark uses 100 iterations, save50, 3 paired repeats,
8 CPU threads; a 50-iteration option is provided solely for bounded device checks.

On a separately available CUDA workstation, with the same input tree and an appropriate
PyTorch environment, these are **local commands**, not batch submission scripts:

```powershell
python -B -X utf8 docs/research/jscc_uncertainty_optimization_20261115/gpu_stress.py --device cuda:0
python -B -X utf8 docs/research/jscc_uncertainty_optimization_20261115/real_kernel_patch.py --device cuda
python -B -X utf8 docs/research/jscc_uncertainty_optimization_20261115/bench.py --device cuda --iterations 100 --max-host-mib 768 --max-gpu-mib 2048
python -B -X utf8 docs/research/jscc_uncertainty_optimization_20261115/component_bench.py --device cuda
```

CUDA commands reject absent CUDA or insufficient measured headroom. They do not install
drivers, change TF32, change precision, launch distributed processes, or submit jobs.
Do not infer GPU speed from CPU timings. A future production-sized trial still needs
actual matching B/C slices, deployed-library and GPU numerical regression, full-rank
communication checks, quota/device measurements and demonstrated end-to-end gain.

## Adapter usage in an isolated caller

`BoundedViewResponse` has the same `matrix`, `sensitivity`, `full_rows`, and `geometry`
interface consumed by the frozen solver. Construct it only for `sc_solve` or
`joint_solve_after_compton_prepare`. Close before preparing Compton or changing energy.
Full cache reuses the existing frozen device-cache implementation; partial cache is a
fixed prefix, no new arithmetic. Budget includes a construction temporary and 20% margin.

For existing CPU event blocks, create one `OrderedTransferPool` and pass
`[ViewBlocks(pool, blocks_for_view) for blocks_for_view in blocks_by_view]` to the frozen
`joint_mlem`. The actual `_event_weight` executes unchanged. A yielded block is borrowed
until the iterator advances: enqueue all usage on the captured compute stream; clone if
retention is required. Do not change stream inside an iterator or consume a pool twice.
Close/fully consume the iterator before resetting stats or closing the pool.

No observed-event transpose is used to replace S. No image filtering, response threshold,
event discard, alternative q cut, kernel-width rescaling, FP16/TF32 setting, OSEM or new
arithmetic batch size is introduced. Physics changes require separate matching calibration.

## Evidence interpretation

- `identity.json`, `additional_evidence.json`: fresh source/data SHA checks and actual
  run-manifest/accounting evidence. Historical timing is labelled historical.
- `uncertainty.json`: 10,270 independent energy-selected ideal point events, 44 strata,
  original conditional PIT, signed residual decomposition, derivative check. E1/E2 bin
  coverage is descriptive because conditioning changes the reference distribution.
- `bench_cpu.json`: small manufactured operator experiment; bitwise paired results.
- `real_kernel_patch_cpu.json`: real energy-factor patch only.
- `component_bench_cpu.json`: instrumented CPU component costs, no H2D or MPI.
- `verification.json`: pass/fail checks plus local Python source hashes.
- `delivery_manifest.json`: final small-study artifact hashes and checked report links;
  unrelated working-tree changes are recorded without changing them.

The readout law, position approximation and material residual law are distinct. The
old angular Gaussian/q acceptance is not the production v5 likelihood. No evidence here
licenses narrowing sigma to make NEMA images prettier. No whole-physics validation or
GPU speed claim is implied by CPU equality.
