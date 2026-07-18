#!/usr/bin/env python3
"""Estimate how exact GAGG containment changes the center-point response."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ENERGIES = (218, 440)
COMPONENTS = (
    "C_intercrystal.sysmat",
    "C_highZ_to_crystal.sysmat",
    "C_local_recoil.sysmat",
    "C_collimator_to_crystal.sysmat",
)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    repository = project.parents[1]
    matrix_project = (
        repository
        / "Auxiliary_Studies"
        / "GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookup", type=Path, default=project / "build_release" / "gagg_intrinsic_containment_lookup.csv")
    parser.add_argument("--runs-dir", type=Path, default=matrix_project / "runs")
    parser.add_argument(
        "--component-root",
        type=Path,
        default=repository / "run_logs" / "ScatterSurface_DirectCenter_20260715",
    )
    parser.add_argument(
        "--position-bias",
        type=Path,
        default=repository
        / "run_logs"
        / "DetectorLocalPositionBias_20260716"
        / "position_bias_summary.json",
    )
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=repository
        / "run_logs"
        / "ScatterSurface_DirectCenter_20260715"
        / "direct_center_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "build_release" / "containment_impact_summary.json",
    )
    return parser.parse_args()


def read_lookup(path: Path) -> dict[tuple[int, tuple[int, int, int], str], float]:
    result = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (
                int(float(row["energy_keV"])),
                (
                    int(float(row["width_mm"])),
                    int(float(row["thickness_mm"])),
                    int(float(row["height_mm"])),
                ),
                row["category"],
            )
            result[key] = float(row["containment_probability"])
    return result


def read_position_factors(path: Path) -> dict[tuple[int, tuple[int, int, int]], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in payload["results"]:
        key = (
            int(row["energy_keV"]),
            tuple(int(value) for value in row["dimensions_mm"]),
        )
        result[key] = (
            row["distributed_second_photoelectric_probability"]
            / row["center_second_photoelectric_probability"]
        )
    return result


def detector_and_center_pe(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    raw_detector = np.fromfile(run_dir / "Params_Detector.dat", dtype="<f4")
    detector_count = int(raw_detector[0])
    detector = raw_detector[1:].reshape(detector_count, 12)
    image = np.fromfile(run_dir / "Params_Image.dat", dtype="<f4")
    nx, ny, nz = (int(value) for value in image[:3])
    voxel_count = nx * ny * nz
    matrix = np.memmap(
        run_dir / "PE_Windowed_SysMat_shift_0.000000_0.000000_0.000000_v3.sysmat",
        dtype="<f4",
        mode="r",
        shape=(detector_count, voxel_count),
    )
    x_index = nx // 2
    y_index = ny // 2
    z_indices = (nz // 2 - 1, nz // 2)
    columns = [z * nx * ny + y_index * nx + x_index for z in z_indices]
    center = 0.5 * (
        np.asarray(matrix[:, columns[0]], dtype=np.float64)
        + np.asarray(matrix[:, columns[1]], dtype=np.float64)
    )
    del matrix
    return detector, center


def row_coefficients(
    detector: np.ndarray,
    energy: int,
    lookup: dict[tuple[int, tuple[int, int, int], str], float],
    position: dict[tuple[int, tuple[int, int, int]], float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pe = np.ones(len(detector), dtype=np.float64)
    local = np.ones(len(detector), dtype=np.float64)
    position_factor = np.ones(len(detector), dtype=np.float64)
    dimensions = np.rint(detector[:, 3:6]).astype(int)
    active = np.rint(detector[:, 11]).astype(int) == 1
    for size in {(3, 3, 3), (2, 6, 2)}:
        selected = active & np.all(dimensions == size, axis=1)
        if not np.any(selected):
            continue
        pe[selected] = lookup[(energy, size, "first_pe")]
        local[selected] = lookup[(energy, size, "first_compton_second_pe")]
        position_factor[selected] = position[(energy, size)]
    return pe, local, position_factor


def read_component(directory: Path, name: str, detector_count: int) -> np.ndarray:
    values = np.fromfile(directory / name, dtype="<f4").astype(np.float64)
    if len(values) != detector_count:
        raise ValueError(f"{directory / name}: expected {detector_count} values, got {len(values)}")
    return values


def main() -> None:
    args = parse_args()
    lookup = read_lookup(args.lookup)
    position = read_position_factors(args.position_bias)
    reference = json.loads(args.reference_summary.read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "interpretation": (
            "Exact 1 eV containment is a lower-bound correction for a finite photopeak window. "
            "Intercrystal/high-Z terminal PE components are left unchanged."
        ),
        "energies": {},
    }

    for energy in ENERGIES:
        run_dir = args.runs_dir / f"JSCC_{energy}keV"
        component_dir = args.component_root / f"JSCC_{energy}keV_far4_near8"
        detector, pe_windowed = detector_and_center_pe(run_dir)
        detector_count = len(detector)
        active = np.rint(detector[:, 11]).astype(int) == 1
        local_center = read_component(
            component_dir, "C_local_self_photoelectric.sysmat", detector_count
        )
        nonlocal_response = sum(
            (read_component(component_dir, name, detector_count) for name in COMPONENTS),
            start=np.zeros(detector_count, dtype=np.float64),
        )
        pe_factor, local_containment, position_factor = row_coefficients(
            detector, energy, lookup, position
        )

        components = {
            "pe_windowed_raw": float(pe_windowed[active].sum()),
            "local_center_raw": float(local_center[active].sum()),
            "nonlocal_unchanged": float(nonlocal_response[active].sum()),
            "local_position_corrected": float(
                (local_center * position_factor)[active].sum()
            ),
            "local_position_and_exact_containment": float(
                (local_center * position_factor * local_containment)[active].sum()
            ),
            "pe_exact_containment": float((pe_windowed * pe_factor)[active].sum()),
        }
        scenarios = {
            "raw_center_model": (
                components["pe_windowed_raw"]
                + components["local_center_raw"]
                + components["nonlocal_unchanged"]
            ),
            "position_only": (
                components["pe_windowed_raw"]
                + components["local_position_corrected"]
                + components["nonlocal_unchanged"]
            ),
            "position_plus_local_exact_containment": (
                components["pe_windowed_raw"]
                + components["local_position_and_exact_containment"]
                + components["nonlocal_unchanged"]
            ),
            "position_plus_all_exact_containment": (
                components["pe_exact_containment"]
                + components["local_position_and_exact_containment"]
                + components["nonlocal_unchanged"]
            ),
        }
        geant4 = float(reference[str(energy)]["geant4"])
        report["energies"][str(energy)] = {
            "geant4_center_probability": geant4,
            "components": components,
            "scenarios": {
                name: {
                    "probability": value,
                    "matrix_to_geant4": value / geant4,
                }
                for name, value in scenarios.items()
            },
            "reference_raw_combined": float(reference[str(energy)]["combined"]),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
