# Uniform-FOV PE v4 vs PE v4 Uniform-FOV layer corrected

Baseline: `CenterPoint_PEv4` (PE v4)
Candidate: `CenterPoint_PEv4_UniformFOVLayer` (PE v4 Uniform-FOV layer corrected)

| Response | Set | G4/matrix total | Absolute total error | Shape L2 | TV |
| --- | --- | ---: | ---: | ---: | ---: |
| A218 | PE v4 | 0.994646 | 0.005354 | 0.007276 | 0.003171 |
| A218 | PE v4 Uniform-FOV layer corrected | 1.000000 | 0.000000 | 0.003482 | 0.001707 |
| A440 | PE v4 | 1.002177 | 0.002177 | 0.010593 | 0.003570 |
| A440 | PE v4 Uniform-FOV layer corrected | 1.000000 | 0.000000 | 0.005622 | 0.002474 |
| C440to218 | PE v4 | 1.011172 | 0.011172 | 0.011404 | 0.004149 |
| C440to218 | PE v4 Uniform-FOV layer corrected | 1.000000 | 0.000000 | 0.010619 | 0.003956 |
