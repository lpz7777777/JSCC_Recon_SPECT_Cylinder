# Execution registry and continuation

## Current immutable inputs

- Original CPU worker release `24cef961cf6890a1` remains unchanged.
- Original job15683333 actually FAILED; all original partial outputs stay intact.
- Latest recovery job15684979: 13 exact completed-worker reuse, 987 missing-worker
  calculations, explicit srun --wait=0, same 240-minute cap. Launcher/acceptance
  release `26ceb64dbf1bd332`. Read transport_recovery_job.json and
  transport_recovery_freeze.json; do not submit another recovery.
- GPU original-MLEM release: `80ff01a669cee377`; deployed, input validation
  not yet submitted while actual transport runs.
- Actual accepted Geant4 binary and scientific response release remain the
  same as the completed original study; only dose/new seeds differ.
- All 20 macros and current remote source/executable passed the source-angle
  audit. Full4π emission, multiplier1, 5e10 actual primaries.

Use `python -X utf8 experiments/ELLIPSE500x300_H120/ehe_5e10_workflow.py status`
for read-only scheduler state. Read
`generated/ehe_spect_5e10_200/controller.json` first. If the registered PID is
alive, do not start a parallel advance/fetch/submit controller. Windows PID
liveness is queried through a read-only process handle; no `os.kill(pid,0)`.

The current one-time `watch --hours 8` follows only this study's registered
stages. It does not resume the paused heartbeat or cancel other allocations.
Do not repeat freeze/deploy/submit or replace a partial output. Failed stages
stop the controller for diagnosis with all evidence preserved.

## Gates and outputs

1. Complete recovery root, batch, extern and srun exit; retain original FAILED
   accounting. Exactly 13 original receipt/member SHA bindings plus 987 receipts
   from the completed recovery allocation must cover all 1000 indices once;
   actual primaries/windows/tags/geometry/source commands/seed/binary SHA.
2. Strict archive/file fetch; exact count identity on the GPU host.
3. Full input validation10 using original forward/transpose and history;
   independent complete-row/20-view operator/S/background verification.
4. Strict actual validation authority permits one formal200/save10 run.
5. Actual GPU exit and 20% resource reserve, 40 complete checkpoints, three
   20-frame histories, final/sum/support/finite/nonnegative/background closure;
   independent verification code release and strict SHA fetch.
6. `plot_ehe_5e10.py`, then `verify_ehe_5e10_figures.py`; directly inspect all
   nine figures before recording visual QA and final delivery.

The existing response physical discrepancy is retained as an explicit
limitation under the user's continuation instruction. Do not equate resource,
source-angle, numerical or checkpoint acceptance with physical calibration.
Do not modify energy windows, matrix physics, photon directions or detector
geometry. No efficiency claim comes from angular identity alone.

Only reviewed code, small identity evidence, reports and figures belong in
Git. Matrices, worker observations, source archives, execution binaries,
large histories and credentials remain excluded. Run the existing staged/
outgoing-blob safety checker and bind Git blob SHA to executed code/evidence
before pushing. Final completion remains pending while transport/reconstruction
or scientific/visual QA is incomplete.

## 2026-10-09 launcher failure and bounded recovery

The observed cluster WaitTime=50 and original log "First task exited 50s ago"
close the termination diagnosis. The initial launcher had no --wait override.
transport_stop_acceptance.json and transport_stop_evidence preserve the original
failure, actual 13 successful receipts and SHA. No partial result is accepted.
transport_recovery_running_snapshot.json records 987 starts/13 verified reuse,
no new errors and an actual RUNNING step beyond the old 50-second default.

The one-time controller/postprocessor were restarted after their original PIDs
exited; their original registrations and logs remain in generated storage.
Latest PIDs/source SHA/log names are in one_time_pipeline_registration.json:
watch_recovery_stdout.log/watch_recovery_stderr.log and
postprocess_recovery_stdout.log/postprocess_recovery_stderr.log. The recurring
compton-v5 automation remains paused. A failed/partial recovery stops again;
no automatic overwrite, re-submission or change to physical matrices is allowed.

Mixed-allocation verification binds every reused receipt to original job15683333
and every new receipt to latest job15684979, checks the original full bin/tag/
source/geometry/dose/seed contract and rejects altered or overlapping identity.
The stopped original root never supplies a successful-allocation certificate.
The new independent verification code has a separate release, and the original
unstarted GPU release and its prior deployment/freezing evidence are preserved.
Fifteen local failure-oriented contract checks passed, including five new tests
for changed reused hashes, wrong job identity, overlaps and incomplete partitions;
actual complete transport/reconstruction acceptance remains pending.

## 2026-10-10 completed transport and collection continuation

transport_completion_sacct.json records successful recovery exit and all 987
missing receipts. The thirteen completed original workers remain exactly bound
in transport_stop_acceptance.json; old failed job15683333 is still FAILED.

The original collection SSH call expired at600 seconds. Read-only inspection
found no active collector and a finished archive with matching full archive and
collection SHA. transport_collection_recovery_registration.json binds this
actual evidence. Reuse the existing archive; do not rerun simulations, copy
worker outputs again, overwrite counts or replace immutable CPU/GPU releases.
The producer uses transport_counts.tar.json, while the former local consumer
expected transport_counts.json. The corrected local consumer fetches the actual
producer receipt and verifies its SHA, all source/worker members, original and
recovered allocation identity, windows/tags/seeds/macros/geometry/full5e10 dose.

Latest one-time helper PIDs/source SHA and log names are in
one_time_pipeline_registration.json: watch_collection_resume_stdout.log,
watch_collection_resume_stderr.log and the postprocess_collection_resume logs.
Previous failed helper registrations/logs remain preserved. This is still a
bounded one-time continuation; the recurring compton-v5 timer remains paused.
Formal200, strict final fetch and numerical/visual QA remain pending until
actual registered outputs and acceptance authorities pass.

Local strict transport acceptance is now complete: transport_acceptance.json
binds the actual5e10 primaries, all1000 workers/20views,16,026 archive members,
new seeds/macros, full window/tag/source/geometry/SHA closure and the mixed
allocation proof. Do not collect, fetch or upload the accepted CPU acquisition
again. Continue the latest registered GPU synchronization and imaging stages;
no physical calibration PASS is implied.

## 2026-10-10 shared-input storage bottleneck

input_storage_stop_acceptance.json records the fully exited owned input-only
PID1856510,47 partial receipts and48 worker directories, the unchanged full
archive SHA, and retained shared counts directory. The failed one-time helper
registrations are preserved in generated/shared_input_storage_stop; prior
report registration is one_time_pipeline_before_archive_storage.json.

input_storage_freeze.json registers new immutable7ace84c81c85b6a8. Preserve
80ff01a669cee377 and freeze_before_archive_storage/deployment proof. Its only
input config addition is archive_storage; every original scientific source,
operator, solver, transport verifier, truth and geometry SHA is unchanged.
The bootstrap must execute inside the actual Slurm allocation, select ext4/
xfs/btrfs on a local device with sufficient disk space, copy/check the exact
accepted archive, safely extract all members and call the frozen entry point.
It does not reuse or overwrite the partial shared counts directory. Original
full-worker verification remains necessary within each scientific target.

transport_gpu_archive_acceptance.json is archive transfer identity only;
full GPU worker acceptance is pending actual validation/independent authority.
Each allocated input receipt must be complete, passed, and bound to its exact
job/program/bootstrap/archive/collection identity during strict fetch.
validation_job.json registers1681331. If any stage fails or is partial, retain
all outputs/logs and stop; no automatic resubmission, matrix change or count
replacement. Completed strict validation alone permits formal200.

Nineteen local contract checks actually passed, including shared/RAM scratch
rejection, insufficient-space and alternate-count rejection, changed archive
SHA rejection before science, and exact-byte original-target execution with
cleanup limited to the owned temporary directory. Actual compute-node disk
identity, full1000-worker verification and GPU execution still require their
real receipts; local checks do not replace them.

Latest bounded helper PIDs36496/31452 and source/log SHA registration is in
one_time_pipeline_registration.json. watch_archive_input_stdout/stderr and
postprocess_archive_input_stdout/stderr preserve the current continuation.
Only one controller may advance/fetch/submit. compton-v5 remains PAUSED.
