# NIST XCOM material tables

`download_nist_xcom.py` downloads photoelectric-absorption and incoherent
(Compton) mass interaction coefficients from the official NIST XCOM web
database for every integer energy from 1 through 1000 keV.

Materials and densities used by the system-matrix generators:

| Material | XCOM formula/symbol | Density (g/cm^3) |
|---|---|---:|
| NaI | `NaI` | 3.67 |
| GAGG | `Gd3Al2Ga3O12` | 6.60 |
| Pb | `Pb` | 11.35 |
| W | `W` | 19.35 |

The GAGG and W densities intentionally match
`Geant4Sim/Geant4Code*/src/DetectorConstruction.cc`. Density changes must be
applied to both the CSV used by `material_db.m` and the CUDA header embedded in
ScatterGen.

The generated CSV preserves both the XCOM mass coefficients in `cm^2/g` and
the linear coefficients in `1/mm`. The CUDA header embeds the linear values so
that calculation jobs do not require network access or a run-time data path.

Source: <https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html>

Regenerate both files with:

```bash
python3 physics_data/download_nist_xcom.py
```

To change only densities while retaining the already downloaded NIST mass
coefficients:

```bash
python3 physics_data/download_nist_xcom.py --rebuild-linear-from-existing
```
