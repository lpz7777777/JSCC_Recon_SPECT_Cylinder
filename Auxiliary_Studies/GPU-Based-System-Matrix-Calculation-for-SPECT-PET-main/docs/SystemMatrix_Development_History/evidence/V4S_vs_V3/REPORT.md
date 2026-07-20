# Uniform-FOV PE v3 vs PE v4 symmetric Halton

Baseline: `CenterPoint` (PE v3)
Candidate: `CenterPoint_PEv4` (PE v4 symmetric Halton)

| Response | Set | G4/matrix total | Absolute total error | Shape L2 | TV |
| --- | --- | ---: | ---: | ---: | ---: |
| A218 | PE v3 | 0.998984 | 0.001016 | 0.007150 | 0.003189 |
| A218 | PE v4 symmetric Halton | 0.994646 | 0.005354 | 0.007276 | 0.003171 |
| A440 | PE v3 | 1.002002 | 0.002002 | 0.010496 | 0.003520 |
| A440 | PE v4 symmetric Halton | 1.002177 | 0.002177 | 0.010593 | 0.003570 |
| C440to218 | PE v3 | 1.012213 | 0.012213 | 0.011693 | 0.004276 |
| C440to218 | PE v4 symmetric Halton | 1.011172 | 0.011172 | 0.011404 | 0.004149 |

## Interpretation

The center-symmetric rule removes the former diagonal detector gradient. The
linear `x-z` plane explains less than 0.002% of the symmetric-v4-minus-v3 change
for every response. A218 and A440 are effectively comparable to calibrated v3
in normalized shape, but do not improve it; A218 also needs a 0.535% global
efficiency reduction if absolute normalization is required. C440to218 improves
both shape L2 (2.5%) and TV (3.0%) relative to v3. This supports the geometric
rewrite for cross-window scatter, but does not yet establish a direct-response
advantage.
