#!/usr/bin/env python3
"""Verify JSCC GAGG/W densities and generated coefficients match Geant4."""

from __future__ import annotations

import argparse
import array
import csv
import json
import math
import re
from pathlib import Path


EXPECTED_DENSITIES = {"GAGG": 6.60, "W": 19.35}
ENERGIES = (218, 440)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--run-suffix", default="_pe_v4")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_float32(path: Path) -> list[float]:
    values = array.array("f")
    values.frombytes(path.read_bytes())
    return values.tolist()


def parse_geant4_densities(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    gagg = re.search(
        r"// GAGG--.*?density\s*=\s*([0-9.]+)\s*\*\s*g/cm3",
        text,
        re.DOTALL,
    )
    tungsten = re.search(
        r'density\s*=\s*([0-9.]+)\s*\*\s*g/cm3;\s*\n\s*G4Material\* W',
        text,
    )
    if not gagg or not tungsten:
        raise ValueError(f"Cannot parse GAGG/W densities from {path}")
    return {"GAGG": float(gagg.group(1)), "W": float(tungsten.group(1))}


def read_xcom_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="ascii") as stream:
        reader = csv.DictReader(line for line in stream if not line.startswith("#"))
        return {int(row["energy_keV"]): row for row in reader}


def parse_header_array(text: str, name: str) -> list[float]:
    match = re.search(
        rf"static const float {name}\[[^]]+\]\s*=\s*\{{(.*?)\}};",
        text,
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"Cannot find {name} in generated XCOM header")
    return [
        float(value)
        for value in re.findall(r"([0-9]+\.[0-9]+e[+-][0-9]+)f", match.group(1), re.I)
    ]


def detector_coefficients(path: Path) -> dict[int, tuple[float, float]]:
    values = read_float32(path)
    count = round(values[0])
    records = [values[1 + 12 * index : 1 + 12 * (index + 1)] for index in range(count)]
    result = {}
    for flag in (1, 2):
        selected = [record for record in records if round(record[11]) == flag]
        if not selected:
            raise ValueError(f"No detector records with flag={flag} in {path}")
        unique = {(record[7], record[8]) for record in selected}
        if len(unique) != 1:
            raise ValueError(f"Nonuniform coefficients for flag={flag} in {path}")
        result[flag] = next(iter(unique))
    return result


def require_close(actual: float, expected: float, message: str) -> None:
    if not math.isclose(actual, expected, rel_tol=2e-6, abs_tol=2e-8):
        raise ValueError(f"{message}: actual={actual:.12g}, expected={expected:.12g}")


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    engine = repo / "Auxiliary_Studies" / "GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main"
    geant4_sources = (
        repo / "Geant4Sim" / "Geant4Code" / "src" / "DetectorConstruction.cc",
        repo / "Geant4Sim" / "Geant4Code_CntStatOnly" / "src" / "DetectorConstruction.cc",
        repo / "Geant4Sim" / "Geant4Code_CntStatResponseStudy" / "src" / "DetectorConstruction.cc",
    )
    geant4 = {str(path): parse_geant4_densities(path) for path in geant4_sources}
    for path, densities in geant4.items():
        for material, expected in EXPECTED_DENSITIES.items():
            require_close(densities[material], expected, f"{path} {material} density")

    csv_path = engine / "physics_data" / "nist_xcom_materials_1_1000keV.csv"
    rows = read_xcom_rows(csv_path)
    header_text = (engine / "physics_data" / "nist_xcom_materials_1_1000keV.h").read_text()
    header_pe = parse_header_array(header_text, "kXcomMuPhotoelectric")
    header_compton = parse_header_array(header_text, "kXcomMuCompton")
    if len(header_pe) != 4000 or len(header_compton) != 4000:
        raise ValueError("Generated XCOM header does not contain 4 x 1000 coefficients")

    checks = []
    material_index = {"GAGG": 1, "W": 3}
    for energy in ENERGIES:
        row = rows[energy]
        expected_by_material = {}
        for material, density in EXPECTED_DENSITIES.items():
            mass_pe = float(row[f"{material}_photoelectric_cm2_g"])
            mass_compton = float(row[f"{material}_incoherent_cm2_g"])
            expected_pe = mass_pe * density / 10.0
            expected_compton = mass_compton * density / 10.0
            csv_pe = float(row[f"{material}_mu_photoelectric_per_mm"])
            csv_compton = float(row[f"{material}_mu_compton_per_mm"])
            require_close(csv_pe, expected_pe, f"CSV {material} PE at {energy} keV")
            require_close(
                csv_compton, expected_compton, f"CSV {material} Compton at {energy} keV"
            )
            header_index = material_index[material] * 1000 + energy - 1
            require_close(
                header_pe[header_index], expected_pe, f"header {material} PE at {energy} keV"
            )
            require_close(
                header_compton[header_index],
                expected_compton,
                f"header {material} Compton at {energy} keV",
            )
            expected_by_material[material] = (expected_pe, expected_compton)

        run_name = f"JSCC_{energy}keV{args.run_suffix}"
        params = detector_coefficients(engine / "runs" / run_name / "Params_Detector.dat")
        for flag, material in ((1, "GAGG"), (2, "W")):
            require_close(
                params[flag][0],
                expected_by_material[material][0],
                f"{run_name} flag={flag} PE",
            )
            require_close(
                params[flag][1],
                expected_by_material[material][1],
                f"{run_name} flag={flag} Compton",
            )
        checks.append(
            {
                "energy_keV": energy,
                "run": run_name,
                "GAGG_mu_photoelectric_per_mm": params[1][0],
                "GAGG_mu_compton_per_mm": params[1][1],
                "W_mu_photoelectric_per_mm": params[2][0],
                "W_mu_compton_per_mm": params[2][1],
            }
        )

    cross_name = f"JSCC_440keV_to_218keVwin{args.run_suffix}"
    cross = detector_coefficients(engine / "runs" / cross_name / "Params_Detector.dat")
    standard_440 = detector_coefficients(
        engine / "runs" / f"JSCC_440keV{args.run_suffix}" / "Params_Detector.dat"
    )
    if cross != standard_440:
        raise ValueError("440-to-218 cross run coefficients differ from the 440 run")

    summary = {
        "status": "PASS",
        "expected_density_g_cm3": EXPECTED_DENSITIES,
        "geant4_sources": geant4,
        "coefficient_checks": checks,
        "cross_run_matches_440": True,
    }
    output = args.output or engine / "runs" / "PEV4_material_density_validation.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
