# ComptonSystemMatrixPrototype

This folder contains a minimal analytic Compton system-matrix prototype with
one discretized energy variable:

```text
A(voxel, det1, det2, e1_bin)
```

The current goal is not high fidelity. The goal is to build a small, readable,
testable prototype that can be inspected against intuition and compared with
list-mode statistics.

## Current implementation

The prototype script is:

- `prototype_comp_matrix_e1.py`

It now uses `torch`, so the prototype can run on:

- CPU
- GPU via CUDA

## Current scope

The prototype currently:

- uses a very small Cartesian FOV
- selects only a small subset of front-layer and rear-layer detectors
- discretizes only `e1`
- includes both `front -> rear` and `rear -> front` ordered two-hit channels
- splits both `det1` and `det2` into finite sub-cells and sums their probability contributions
- uses NaI attenuation interpolation from `Auxiliary_Studies/FreePath/data.txt`
- writes a small dense prototype tensor for inspection
- writes an `x-z` response slice plot parallel to the selected detector surface

## Current approximation level

For a given `(voxel, det1, det2, e1_bin)`, the current weight is still a rough
probability approximation. It combines:

- source-to-`det1` sub-cell projected-area probability
- attenuation by other detector crystals along `voxel -> det1`
- survival to the selected depth interval inside `det1`
- Compton interaction probability in that `det1` depth interval
- normalized Klein-Nishina angular probability density
- `det1`-to-`det2` sub-cell solid-angle probability
- attenuation by other detector crystals along `det1 -> det2`
- survival to the selected depth interval inside `det2`
- second-hit recording probability in that `det2` depth interval
- an `e1` bin mass around the geometry-implied first deposited energy
- summation over all `det1` / `det2` sub-cell combinations

## Energy resolution handling

The current script **does include** a simple energy-resolution model by default.

It uses the same style as the reconstruction scripts:

- reference FWHM at 662 keV: `--ene-resolution-662kev`
- effective resolution:

```text
ene_resolution = ene_resolution_662kev * sqrt(0.662 / e0)
```

- first-deposit sigma:

```text
sigma_e = e1 * ene_resolution / 2.355 * sqrt(e0 / e1)
```

This `sigma_e` is then used to distribute the geometry-implied `e1` into the
discrete `e1` bins.

If you want to inspect the purely geometric version without energy blur, use:

```powershell
--disable-energy-blur
```

## What is still missing

Still not included:

- exact continuous volume integral inside `det1` and `det2`
- same-layer scattering
- exact reconstruction-event filtering logic
- higher-order interactions
- finite detector energy threshold logic at the second interaction
- any material between source, `det1`, and `det2` other than the detector crystals themselves

So this is still a prototype, not a final Compton forward model.

## Typical usage

PowerShell example:

```powershell
python .\Auxiliary_Studies\ComptonSystemMatrixPrototype\prototype_comp_matrix_e1.py `
  --detector-csv .\Factors\511keV_RotateNum20\Detector.csv `
  --device cuda `
  --hemisphere positive `
  --front-det-count 6 `
  --rear-det-count 6 `
  --fov-nx 5 `
  --fov-ny 5 `
  --fov-nz 3 `
  --sx-mm 12 `
  --sy-mm 12 `
  --sz-mm 6 `
  --e1-min-mev 0.0 `
  --e1-bin-count 12 `
  --det-sample-nx 2 `
  --det-sample-ny 2 `
  --det-sample-nz 2 `
  --voxel-chunk-size 128 `
  --rear-chunk-size 2 `
  --output-dir .\Auxiliary_Studies\ComptonSystemMatrixPrototype\Result\prototype_run_01
```

The detector-size handling is currently implemented by uniform sub-cell
integration. For example, `2 x 2 x 2` means each selected crystal is split into
8 sub-cells. The final response is the sum of all sub-cell pair probability
contributions, not their average.

You can also manually specify which `(det1, det2, e1_bin)` should be used for
the slice plot:

```powershell
--slice-front-index 0 --slice-rear-index 0 --slice-e1-bin 3
```

If not specified, the script automatically picks the strongest bin at the
center voxel.

## Output files

The script writes:

- `prototype_response_tensor.npz`
- `summary.json`
- `detector_selection.csv`
- `voxel_table.csv`
- `center_voxel_topbins.csv`
- `xz_slice_response.png`

Important outputs:

- `prototype_response_tensor.npz`
  contains the full small dense prototype tensor
- `center_voxel_topbins.csv`
  tells you which `(det1, det2, e1_bin)` bins dominate for the center voxel
- `xz_slice_response.png`
  shows one `x-z` slice of the FOV response for a selected virtual bin

The `.npz` file also stores:

- `front_interaction_samples_mm`
- `rear_interaction_samples_mm`

This slice is meant to help you visually inspect whether the response begins to
look like a Compton-cone-type structure.

## Notes on the current detector-size model

- Front-layer crystals currently use size `3 x 3 x 3 mm`.
- Rear-layer crystals currently use size `2 x 6 x 2 mm`.
- Sample points are placed at uniform interior sub-cell centers, not on crystal
  boundaries.
- The current output is now much closer to a physical probability than the
  previous center-sampling version, but it is still an approximation because
  each sub-cell uses a center-point geometry approximation.
- The current blocker search first builds a candidate list geometrically and
  then computes exact segment-box intersection lengths only for those candidate
  detector crystals.
- `--voxel-chunk-size` and `--rear-chunk-size` can be used to reduce peak
  memory usage when blocker attenuation is enabled.
