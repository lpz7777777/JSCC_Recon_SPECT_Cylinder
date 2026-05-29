# Event Order Inference Experiment

This folder contains an independent study for inferring the order of two-hit
Compton events when the stored order in `List/*.csv` is treated as unknown
during inference.

The experiment is intentionally separated from the main reconstruction code.
It is used to answer one focused question:

- if the list file only tells us the two hit positions and energies, can we
  infer which interaction happened first?

Current status:

- the hand-crafted baseline has been simplified to one retained method:
  `center_geometry`
- a supervised classifier has been added and currently performs better than
  that baseline on the tested simulated datasets
- both methods support:
  - hard decision
  - reject / drop-low-confidence evaluation
  - soft-confidence output for the supervised model

The repository root still contains a compatibility link named
`EventOrderInference_Experiment`, but the maintained location is now:

- `Auxiliary_Studies/EventOrderInference_Experiment`

## 1. Problem Definition

Only the following events are evaluated:

- exactly two interactions are present
- the two interactions occur across front and rear detector layers
- same-layer events are excluded

Ground truth:

- the stored order in the list file is used only as the evaluation label
- the inference algorithm itself is not allowed to directly use that order

Binary label:

- class 1: `front-first`
- class 0: `rear-first`

So this is a binary classification problem defined only on cross-layer two-hit
events.

## 2. Reconstruction-Matched Filtering

The event filtering is aligned with the actual reconstruction pipeline instead
of an idealized clean-data setting.

The logic is matched to `process_list_plane_strict.py`:

- energy smearing is applied to the stored-order `e1` and `e2`
- thresholds are applied after smearing
- Compton kinematic validity is checked after smearing
- `cpnum1 != cpnum2` is required

For current 511 keV experiments, the default settings are:

- `e0 = 0.511`
- `ene_resolution_662keV = 0.1`
- `ene_resolution = 0.1 * sqrt(0.662 / 0.511) = 0.1138199904`
- `ene_threshold_min = 0.05`
- `ene_threshold_sum = 0.46`
- `ene_threshold_max = 0.3396666667`

This matters because order-inference accuracy is noticeably worse after energy
smearing than on the original exact simulated energies.

## 3. Hand-Crafted Baseline

Script:

- `infer_event_order.py`

Retained method:

- `center_geometry`

Each event is evaluated under two candidate hypotheses:

- the front-layer hit happened first
- the rear-layer hit happened first

Each hypothesis score combines:

- a front-first prior ratio
- a Klein-Nishina term
- a center-based geometric consistency term

Form:

```text
score(front-first) = prior_ratio * KN(E_front) * G_front^geometry_power
score(rear-first)  = KN(E_rear) * G_rear^geometry_power
```

Current tuned defaults:

- `front_prior_ratio = 1.2`
- `geometry_sigma_deg = 20`
- `geometry_power = 1.0`

### Reject Option

For the hand-crafted method:

```text
confidence = |log(score_front) - log(score_rear)|
```

Low-confidence events are rejected first, then retained accuracy is recomputed.

## 4. Supervised Classifier

Script:

- `train_supervised_order_classifier.py`

Current implementation:

- simple MLP classifier
- trained on the same filtered cross-layer events as the hand-crafted baseline
- uses the same energy-smearing logic as the baseline

Current default training setup:

- hidden dims: `128,64`
- dropout: `0.1`
- optimizer: `AdamW`
- learning rate: `1e-3`
- weight decay: `1e-4`
- epochs: `10`
- batch size: `8192`
- device: `cuda` when available

### Features

The current features include:

- front / rear deposited energies
- energy sum and differences
- energy fractions
- front / rear inferred Compton angles
- angle-valid flags
- Klein-Nishina terms
- `center_geometry` terms
- score differences
- detector coordinates
- line-of-response geometry summaries

So the classifier is not replacing the physics cues; it is learning a more
flexible nonlinear combination of them.

## 5. Reported Results

### 5.1 Hand-crafted `center_geometry` under energy smearing

Using the reconstruction-matched smearing and filtering:

- `1e9`:
  - accuracy = `0.78156 ± 0.00059`
  - retained cross-layer event count = `168741 ± 145`
- `2e9`:
  - accuracy = `0.78083 ± 0.00052`
  - retained cross-layer event count = `337236.8 ± 163.3`

Reference result directories:

- `results_center_smear_1e9`
- `results_center_smear_2e9`

Reject sweep on `1e9`:

- reject `0%`: `0.78156`
- reject `5%`: `0.79552`
- reject `10%`: `0.81001`
- reject `20%`: `0.84117`
- reject `30%`: `0.87886`
- reject `40%`: `0.91990`

Reference result directory:

- `results_center_smear_reject_1e9`

### 5.2 Supervised classifier under energy smearing

Main experiment reported here:

- trained on `List_ContrastPhantom_240_30_2e9`
- validated on held-out files from that training directory
- tested on `List_ContrastPhantom_240_30_1e9`
- test set evaluated across 5 independently smeared repeats

Current aggregate test result:

- baseline `center_geometry` accuracy on the same test events:
  - `0.78172 ± 0.00067`
- supervised classifier accuracy:
  - `0.83333 ± 0.00043`
- mean accuracy gain over baseline:
  - `+0.05161 ± 0.00068`

Validation reference:

- baseline `center_geometry` accuracy:
  - `0.78128`
- classifier accuracy:
  - `0.83226`

Reference result directories:

- `supervised_results_main`
- `supervised_results_main_reject_soft`

## 6. Reject Option Results

For the supervised classifier:

```text
confidence = |logit|
```

Test-set retained accuracy:

| Reject Fraction | `center_geometry` | Classifier |
| --- | ---: | ---: |
| 0%  | 0.78172 | 0.83333 |
| 5%  | 0.79572 | 0.84991 |
| 10% | 0.81038 | 0.86662 |
| 20% | 0.84160 | 0.90146 |
| 30% | 0.87910 | 0.93743 |
| 40% | 0.91978 | 0.97038 |

Takeaway:

- the classifier is better than `center_geometry` without rejection
- after rejection, it remains better at every tested reject ratio

## 7. Soft Decision Results

For the supervised classifier:

- hard decision uses `logit >= 0`
- soft decision uses `sigmoid(logit)` as `P(front-first)`

Reported soft-output quality on the test set:

- Brier score:
  - `0.10649 ± 0.00015`
- log loss:
  - `0.31480 ± 0.00044`
- average probability assigned to the true class:
  - `0.78676 ± 0.00021`

Interpretation:

- the output probability carries useful confidence information
- a future reconstruction pipeline could use both order hypotheses with soft
  weights instead of only using hard keep / drop decisions

## 8. Typical Usage

### 8.1 Hand-crafted baseline

```powershell
python .\Auxiliary_Studies\EventOrderInference_Experiment\infer_event_order.py `
  --list-dir .\List\511keV_RotateNum20\List_ContrastPhantom_240_30_1e9 `
  --detector-csv .\Factors\511keV_RotateNum20\Detector.csv `
  --energy-mev 0.511 `
  --front-prior-ratio 1.2 `
  --geometry-sigma-deg 20 `
  --geometry-power 1.0 `
  --ene-resolution-662keV 0.1 `
  --ene-threshold-min 0.05 `
  --ene-threshold-sum 0.46 `
  --smear-repeats 5 `
  --seed 20260331 `
  --output-dir .\Auxiliary_Studies\EventOrderInference_Experiment\results_center_smear_1e9
```

Outputs typically include:

- `summary.json`
- `report.txt`
- `per_file_metrics_seed0.csv`

### 8.2 Supervised classifier

Use the PyTorch conda environment explicitly:

```powershell
conda run -n pytorch python .\Auxiliary_Studies\EventOrderInference_Experiment\train_supervised_order_classifier.py `
  --train-list-dir .\List\511keV_RotateNum20\List_ContrastPhantom_240_30_2e9 `
  --test-list-dir .\List\511keV_RotateNum20\List_ContrastPhantom_240_30_1e9 `
  --detector-csv .\Factors\511keV_RotateNum20\Detector.csv `
  --epochs 10 `
  --train-smear-repeats 3 `
  --test-smear-repeats 5 `
  --batch-size 8192 `
  --output-dir .\Auxiliary_Studies\EventOrderInference_Experiment\supervised_results_main_reject_soft
```

Typical outputs:

- `summary.json`
- `model.pt`

## 9. Visualization

This experiment folder also includes result visualization code so that MATLAB
can read a chosen result folder and directly produce figures.

Typical visual outputs include:

- forward / reverse scatter classification counts
- classification accuracy by category
- reject-ratio versus retained-accuracy curves
- confidence-based removal behavior

Exact plotting entry points may evolve with the experiment, so the practical
rule is:

- select one result folder
- let the MATLAB script read that folder directly
- generate all plots from the stored summary files

## 10. Limitations

- all current conclusions are based on simulated data with known ground truth
- current generalization claims are limited to the tested phantom / count-level
  settings
- same-layer events are still excluded
- classifier probabilities are not yet directly integrated into the main
  reconstruction pipeline
- the current model is still intentionally simple

## 11. Practical Conclusion

At the current stage:

- `center_geometry` is a valid interpretable baseline
- energy smearing significantly hurts event-order inference
- rejecting low-confidence events helps both methods
- the supervised classifier is currently the best-performing approach tried so
  far for:
  - hard accuracy
  - reject-option accuracy
  - useful soft confidence output

If this line of work continues, the most promising next step is not more
retuning of the hand-crafted score, but using classifier confidence directly in
the reconstruction pipeline, ideally in a soft-weighted form.

