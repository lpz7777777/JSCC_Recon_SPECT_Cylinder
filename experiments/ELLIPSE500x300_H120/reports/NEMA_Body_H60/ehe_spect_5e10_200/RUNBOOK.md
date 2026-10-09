# Execution registry and continuation

## Current immutable inputs

- CPU transport release: `24cef961cf6890a1`; unique job `15683333`.
- GPU original-MLEM release: `8dc6d9218cb7c290`; deployed, input validation
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

1. Complete root, batch, extern and srun exit plus 1000 worker receipts;
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
