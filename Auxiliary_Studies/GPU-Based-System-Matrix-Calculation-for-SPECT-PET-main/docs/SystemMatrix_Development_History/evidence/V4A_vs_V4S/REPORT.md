# Uniform-FOV PE v4 legacy asymmetric vs PE v4 symmetric Halton

Baseline: `CenterPoint_PEv4_LegacyAsymmetricHalton` (PE v4 legacy asymmetric)
Candidate: `CenterPoint_PEv4` (PE v4 symmetric Halton)

| Response | Set | G4/matrix total | Absolute total error | Shape L2 | TV |
| --- | --- | ---: | ---: | ---: | ---: |
| A218 | PE v4 legacy asymmetric | 0.995409 | 0.004591 | 0.008214 | 0.003568 |
| A218 | PE v4 symmetric Halton | 0.994646 | 0.005354 | 0.007276 | 0.003171 |
| A440 | PE v4 legacy asymmetric | 1.002357 | 0.002357 | 0.011078 | 0.003752 |
| A440 | PE v4 symmetric Halton | 1.002177 | 0.002177 | 0.010593 | 0.003570 |
| C440to218 | PE v4 legacy asymmetric | 1.011154 | 0.011154 | 0.012002 | 0.004412 |
| C440to218 | PE v4 symmetric Halton | 1.011172 | 0.011172 | 0.011404 | 0.004149 |

## Interpretation

Center-symmetric sampling reduces shape L2 by 11.4% (A218), 4.4% (A440), and
5.0% (C440to218), and reduces TV for all three responses. The large diagonal
gradient in the legacy detector maps is absent. The old matrices are retained
for provenance only and must not be used as production PE v4 inputs.
