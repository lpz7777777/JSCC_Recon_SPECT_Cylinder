# Uniform-FOV PE v3 vs PE v4

Baseline: `CenterPoint` (PE v3)
Candidate: `CenterPoint_PEv4` (PE v4)

| Response | Set | G4/matrix total | Absolute total error | Shape L2 | TV |
| --- | --- | ---: | ---: | ---: | ---: |
| A218 | PE v3 | 0.998984 | 0.001016 | 0.007150 | 0.003189 |
| A218 | PE v4 | 0.995409 | 0.004591 | 0.008214 | 0.003568 |
| A440 | PE v3 | 1.002002 | 0.002002 | 0.010496 | 0.003520 |
| A440 | PE v4 | 1.002357 | 0.002357 | 0.011078 | 0.003752 |
| C440to218 | PE v3 | 1.012213 | 0.012213 | 0.011693 | 0.004276 |
| C440to218 | PE v4 | 1.011154 | 0.011154 | 0.012002 | 0.004412 |

## Interpretation

This first PE v4 production run does **not** improve the Uniform-FOV match and
must not replace the PE v3 baseline. Total efficiency is nearly unchanged, but
shape L2 increases for all three responses.

The detector-wise PE-v4-minus-PE-v3 change is dominated by an `x-z` plane:

| Response | Plane R2 |
| --- | ---: |
| A218 | 0.8546 |
| A440 | 0.9135 |
| C440to218 | 0.9314 |

The legacy first-256 Halton surface samples have centroid
`(u,v)=(0.49805,0.49516)` instead of `(0.5,0.5)`. The resulting entry-position
offset has the same direction as the detector gradient. The production and CPU
reference implementations now use four-way reflected, center-symmetric Halton
groups. The matrices assessed by this report predate that correction and are
retained only as diagnostic evidence.
