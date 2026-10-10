# Independent EHE actual Geant4 5e10 + original MLEM200

**Delivered 2026-10-10:** actual full4π 5e10 transport, full-input validation10,
sequential original440/218 MLEM200, independent all-row/20-view acceptance,
strict fetch, all saved-frame numeric QA and direct visual QA are complete.
Read [the results and 15-route comparison](RESULTS.md) and
[the final delivery manifest](final_delivery.json).

- Transport15684979: COMPLETED/0:0,1000 independent workers/20views,
  50000000000 actual primaries, seeds33100101–33101100, multiplier1.
  Windows218/440=232888/118846. Original13 complete workers retained;
  only987 missing workers supplemented after diagnosed srun WaitTime failure.
- Validation1681331/independent1681343 and formal1681346/independent1681349
  all COMPLETED/0:0.40 atomic checkpoints, three20-frame histories,
  all78920 cells/2312bins/20views, fixed background from own440200,
  original all-ones MLEM, no regularization or fitted gain.
- Scientific execution release7ace84c81c85b6a8 preserves all original science
  source/solver/operator/matrix/geometry/truth SHA. Node-local ext4 staging
  used the exact accepted archive and did not replace old shared partial data.
- Nine final scientific figures directly inspected. Original galleries remain;
  coronal/sagittal label clipping was fixed in comparison_reviewed, with image
  arrays/planes SHA bound. Complete native120mm metrics and true3D ROIs.
- EHE0–200, JSCC0–10000 keep independent iteration ranges. Actual EHE5e10,
  actual EHE5e9, forward-matrix+Poisson5e9 and six JSCC routes form15 rows.
  Different doses/materials/coverage/background budgets are explicit.

[The source-angle audit](SOURCE_ANGLE.md) proves full4π, one photon/event,
no hemisphere restriction. /xcat/angle rotates source positions.
Original EHE materials,1250 finite holes,2312 NaI bins,H60 truth at3mm,
source centerY−345,120mm/full active geometry are retained unchanged.

The user explicitly continued the original response method despite the
previous physical discrepancy. Old HOLD evidence is preserved; this complete
execution/numerical/result delivery does not claim physical response
calibration or device performance. No response/physics/window/gain was changed.

The bounded controller and postprocessor both completed/exit0; their
registrations/logs are preserved. Do not repeat simulations, response
calculations, validation/formal submission, accepted collection or strict fetch.
The recurring compton-v5 task remains PAUSED. Historical failures and recovery
identity remain documented in [the runbook](RUNBOOK.md).

Only reviewed code,small proofs,CSV metrics,reports and PNGs are committed.
Raw observations,matrix/response blocks,archives,binaries and large histories
remain outside Git. Numeric-stage receipts marked visual-pending are historical;
visual_figure_acceptance.json and final_delivery.json are the final authorities.
