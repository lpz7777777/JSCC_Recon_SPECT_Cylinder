# Independent EHE actual Geant4 5e10 + original MLEM200

This is the user-requested new actual transport acquisition. It uses no old
observation counts and is separate from the completed actual 5e9 and
matrix-forward-plus-Poisson 5e9 studies.

Latest 2026-10-10 milestone: full actual **5e10** transport now passed local
strict archive/all-worker acceptance in `transport_acceptance.json` (1000
workers,20 views,16,026 fetched archive members, exact source/seed/macros/
geometry/windows/tags/member-SHA closure). Native windows218/440 are
232888/118846; primary tags are [14690373781,35309626219,0]. CPU recovery job
15684979 completed successfully. GPU synchronization and validation/formal
reconstruction acceptance remain pending until their actual authorities exist.
This transport execution acceptance is not physical response calibration.

Current actual registration: recovery CPU job **15684979**, launcher/acceptance
release `26ceb64dbf1bd332`. Original job **15683333** actually FAILED
with exit9 after 1:40:24: the cluster's default srun WaitTime=50 killed the
remaining tasks 50 seconds after the first successful task exited. This was
not the four-hour allocation limit; no OOM evidence is present in accounting
or the retained launch log. The original failed job is not relabelled PASSED.

Thirteen complete original workers passed member-SHA verification and are
reused by their exact receipt/member hashes and original allocation identity.
Only the **987 interrupted workers** run again in a separate transport output
folder. Incomplete original outputs and all original logs/releases remain
unchanged. The original Geant4 binary and original Python worker wrapper,
macros, source directions, 50M events/worker and seeds are unchanged. The new
18-node/1000-rank allocation explicitly uses `srun --wait=0`; 13 ranks verify
reuse and 987 ranks compute. The observed running step exceeded the old
50-second first-exit limit, with all 987 new workers started and no errors.
Every final view still requires 50 accepted workers; the final accepted dose
must be 5e10, never a sum of partial and restarted histories.

2026-10-10 collection status: recovery job15684979 root/batch/extern/step all
actually COMPLETED/0:0 (root 2:21:06, step 2:21:10). All 987 missing workers
finished with receipts, plus the thirteen verified original workers. The
remote collection/archive finished after the local 600-second SSH command
expired. The finished 58,948,637-byte archive and 16,026-member receipt are
preserved; no simulation or collection is repeated. Its receipt is named
`transport_counts.tar.json` by the archive producer's `with_suffix` operation.
The continuation now recognizes a completed archive, strictly fetches it and
checks all 1000 workers locally before GPU synchronization and validation.
Remote recorded primaries are [14690373781,35309626219,0] = 5e10; native
windows218/440 are232888/118846. These totals remain distinct from full
local/GPU execution acceptance until those registered authorities exist.

[The angular audit](SOURCE_ANGLE.md) confirms **full 4π isotropic emission**,
one photon/event and a dose-equivalent multiplier of **1**. The macro's
`/xcat/angle` rotates source positions. There is no hemisphere angular
restriction. All 20 deployed macros and the current remote generator/source
and accepted executable SHA are bound in `source_angular_audit.json`.

The source retains the existing H60 3mm 3D cuboids, 20 rotations, gamma yields
0.114/0.259, centerY −345, full 120mm geometry and original EHE materials,
1250 holes and 2312 NaI bins. The original accepted Geant4 executable is
reused unchanged; source macros only change each worker's beamOn from 25M
to 50M. The larger single multi-node job follows the actual partition and
account limits without submitting 1000 separate jobs or modifying other jobs.

The original complete A218/A440/C440to218 matrices are reused by exact SHA.
Unstarted initial GPU release `8dc6d9218cb7c290` is preserved; new release
`80ff01a669cee377` is preserved with an explicit mixed-allocation
receipt verifier. New storage-only release `7ace84c81c85b6a8` retains these
scientific files by exact SHA and bootstraps the accepted archive on node-local
disk within each actual validation/reconstruction/verification allocation. The original reconstruction runner/operator/solver/geometry/
truth bytes are unchanged. Actual
transport acceptance and complete-input validation10 precede the unique
formal200 run. Both solver helpers, all 78920 active cells, all 2312 bins,
20 views, volume/rotation geometry, unit initial density and original MLEM
remain unchanged. There is no regularization or brightness fitting.
440 single200 runs first; 218 single200 uses a fixed additive background
computed from this acquisition's own final440 image. The third output is
the same-iteration gamma-density sum. Two stages save 40 atomic/fsync
checkpoints in total and each of three routes saves 20 frames.

The user explicitly requested continuation with the original response method
despite the earlier cross-window discrepancy. Existing physical HOLD evidence
is preserved and no physical calibration PASS is claimed by this study's
execution/numerical acceptance. There is no physical-gate resubmission or new
response calculation.

The one-time bounded local controller watches the registered transport,
strictly collects all 1000 receipts/source macros/counts, syncs only this new
acquisition, then runs validation10 → independent all-row/all-view acceptance
→ formal200 → independent acceptance → strict fetch. Any failed/partial stage
is retained and stops the controller. The old recurring `compton-v5`
automation stays PAUSED.

After actual accepted results exist, `plot_ehe_5e10.py` produces a 15-route
comparison with actual EHE 5e10, actual EHE 5e9, matrix-Poisson 5e9 and JSCC.
Each EHE keeps 0–200 and JSCC keeps 0–10000; equal columns/iteration numbers
are not equal convergence. All saved-frame 3D CRC/CNR and native120mm density,
noise, peaks, leakage and integral curves accompany axial/coronal/sagittal/
72mm-MIP galleries. Scaling uses actual emitted-source density, fixed 0–10,
crop0, no smoothing, no gain fit and existing 3D sphere ROIs. Different doses,
materials, coverage, model approximations and 440-background budgets are
reported. Numerical and direct visual QA are required for final delivery.

Nineteen local failure-oriented contract checks passed, including full dose/view/
seed partition, rejection of changed/repeated identities, original helper
SHA, macro-only-dose changes and safe archive failure cases. These local
checks do not substitute for actual all-worker or imaging acceptance.

## 2026-10-10 accepted acquisition and GPU input storage continuation

The complete local transport authority is PASSED:1000 workers,20views, actual
50000000000 primaries, windows218/440=232888/118846 and all16026 archive
members. The recovery allocation15684979 exited COMPLETED/0:0 in02:21:06;
the original13 completed workers remain bound to their original receipts.

Shared-disk extraction had written only47 receipts in48 worker directories
after several minutes, with almost no additional CPU time. The owned input
extraction process was explicitly stopped and confirmed fully exited. Its
partial shared directory and logs remain preserved. No simulation, Slurm
compute job, matrix or observation is repeated or canceled. The uploaded
58948637-byte archive retains its accepted SHA, and is not uploaded again.

The new wrapper copies that exact archive to actual Slurm node-local disk,
checks space and local filesystem identity, extracts every approved member,
and calls the unchanged reconstruction or independent verifier. Each target
retains its original complete1000-worker verification. Archive transfer
acceptance alone is not GPU full-worker or reconstruction acceptance.
Independent verification has a separate immutable code release. Each allocated
input receipt must finish successfully and bind job/program/archive/collection
SHA before strict imaging authority is fetched.

Full-input validation10 job1681331 is registered and PENDING(Priority) at this
read-only snapshot. Formal200 has not been submitted. Only actual completed
validation plus strict independent authority permits the unique formal run.
The node-local disk free-space check and extraction throughput still await
actual allocation; no login-host disk reading is claimed as compute-node proof.
The recurring timer remains PAUSED, with a single bounded8-hour controller
and postprocessor continuing this authorized acquisition. Final delivery is
still pending strict final acceptance, comparison, scientific and visual QA.
