# Independent EHE actual Geant4 5e10 + original MLEM200

This is the user-requested new actual transport acquisition. It uses no old
observation counts and is separate from the completed actual 5e9 and
matrix-forward-plus-Poisson 5e9 studies.

Current actual registration: CPU job **15683333**, release
`24cef961cf6890a1`, one 18-node allocation with 1000 independent single-core
tasks. The actual allocation contains 1008 CPUs; its srun step uses 1000
CPUs. All 20 views have 50 workers × 50 million events, totaling **5e10**.
New seeds are 33100101–33101100. The job was observed RUNNING on
`ibc12b11n[01-18]`. Completion, counts and timing remain pending until actual
exit and all-worker acceptance.

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
GPU reconstruction release `8dc6d9218cb7c290` is frozen and deployed. Actual
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

Ten local failure-oriented contract checks passed, including full dose/view/
seed partition, rejection of changed/repeated identities, original helper
SHA, macro-only-dose changes and safe archive failure cases. These local
checks do not substitute for actual all-worker or imaging acceptance.
