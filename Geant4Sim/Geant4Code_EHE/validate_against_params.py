#!/usr/bin/env python3
"""Compare the constructed Geant4 EHE geometry with serialized MATLAB Params."""

from __future__ import annotations

import argparse
import csv
import math
import struct
from pathlib import Path
from typing import Iterable


FOV_CENTER_Y_MM = -245.0
TOLERANCE_MM = 1.0e-4


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_params = (
        repo_root
        / "Auxiliary_Studies"
        / "GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main"
        / "FileGenerater_3D_Unified"
        / "output"
        / "EHE_PbNaI_218keV"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Compare EHE_DetectorGeometry.csv and EHE_CollimatorHoles.csv "
            "written by ehe_spect with the final Params_*.dat files."
        )
    )
    parser.add_argument("--params-dir", type=Path, default=default_params)
    parser.add_argument("--geant-output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tolerance-mm", type=float, default=TOLERANCE_MM)
    return parser.parse_args()


def read_float32(path: Path) -> list[float]:
    payload = path.read_bytes()
    if len(payload) % 4:
        raise ValueError(f"{path} length is not divisible by four bytes")
    return list(struct.unpack(f"<{len(payload) // 4}f", payload))


def rows(values: list[float], offset: int, width: int, count: int) -> list[list[float]]:
    expected = offset + width * count
    if len(values) != expected:
        raise ValueError(f"expected {expected} floats, found {len(values)}")
    return [values[offset + i * width : offset + (i + 1) * width] for i in range(count)]


def read_csv_numbers(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="ascii") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def maximum_error(errors: Iterable[float]) -> float:
    return max((abs(value) for value in errors), default=0.0)


def compare(params_dir: Path, geant_dir: Path, tolerance: float) -> dict[str, object]:
    detector_raw = read_float32(params_dir / "Params_Detector.dat")
    collimator_raw = read_float32(params_dir / "Params_Collimator.dat")
    image = read_float32(params_dir / "Params_Image.dat")

    detector_count = round(detector_raw[0])
    hole_count = round(collimator_raw[10])
    params_detectors = rows(detector_raw, 1, 12, detector_count)
    params_holes = rows(collimator_raw, 100, 9, hole_count)
    geant_detectors = read_csv_numbers(geant_dir / "EHE_DetectorGeometry.csv")
    geant_holes = read_csv_numbers(geant_dir / "EHE_CollimatorHoles.csv")

    if detector_count != 2312 or len(geant_detectors) != detector_count:
        raise ValueError(
            f"detector count mismatch: Params={detector_count}, Geant4={len(geant_detectors)}"
        )
    if hole_count != 1250 or len(geant_holes) != hole_count:
        raise ValueError(f"hole count mismatch: Params={hole_count}, Geant4={len(geant_holes)}")

    reference_shift_y = FOV_CENTER_Y_MM + image[11]
    detector_errors: list[float] = []
    for index, (params_row, geant_row) in enumerate(zip(params_detectors, geant_detectors), 1):
        if round(geant_row["id"]) != index:
            raise ValueError(f"Geant4 detector ID mismatch at row {index}")
        expected = [
            params_row[0],
            params_row[1] + reference_shift_y,
            params_row[2],
            params_row[3],
            params_row[4],
            params_row[5],
        ]
        actual = [
            geant_row["x_mm"],
            geant_row["y_mm"],
            geant_row["z_mm"],
            geant_row["size_x_mm"],
            geant_row["size_y_mm"],
            geant_row["size_z_mm"],
        ]
        detector_errors.extend(a - b for a, b in zip(actual, expected))

    hole_errors: list[float] = []
    nearest = [math.inf] * hole_count
    for index, (params_row, geant_row) in enumerate(zip(params_holes, geant_holes), 1):
        if round(geant_row["id"]) != index:
            raise ValueError(f"Geant4 hole ID mismatch at row {index}")
        expected = [
            params_row[0],
            params_row[1] + reference_shift_y,
            params_row[2] + reference_shift_y,
            params_row[3],
            params_row[4],
        ]
        actual = [
            geant_row["x_mm"],
            geant_row["y1_mm"],
            geant_row["y2_mm"],
            geant_row["z_mm"],
            geant_row["radius_mm"],
        ]
        hole_errors.extend(a - b for a, b in zip(actual, expected))

    for i, first in enumerate(geant_holes):
        for j in range(i + 1, hole_count):
            second = geant_holes[j]
            distance = math.hypot(first["x_mm"] - second["x_mm"], first["z_mm"] - second["z_mm"])
            nearest[i] = min(nearest[i], distance)
            nearest[j] = min(nearest[j], distance)

    max_detector_error = maximum_error(detector_errors)
    max_hole_error = maximum_error(hole_errors)
    if max_detector_error > tolerance or max_hole_error > tolerance:
        raise ValueError(
            f"coordinate mismatch exceeds {tolerance:g} mm: "
            f"detector={max_detector_error:.9g}, holes={max_hole_error:.9g}"
        )

    return {
        "status": "PASS",
        "detector_count": detector_count,
        "hole_count": hole_count,
        "max_detector_error_mm": max_detector_error,
        "max_hole_error_mm": max_hole_error,
        "nearest_pitch_min_mm": min(nearest),
        "nearest_pitch_max_mm": max(nearest),
        "minimum_septum_mm": min(nearest) - 2.0 * geant_holes[0]["radius_mm"],
        "reference_shift_y_mm": reference_shift_y,
        "params_detectors": params_detectors,
        "params_holes": params_holes,
        "geant_detectors": geant_detectors,
        "geant_holes": geant_holes,
    }


def write_report(path: Path, result: dict[str, object]) -> None:
    keys = [
        "status",
        "detector_count",
        "hole_count",
        "max_detector_error_mm",
        "max_hole_error_mm",
        "nearest_pitch_min_mm",
        "nearest_pitch_max_mm",
        "minimum_septum_mm",
        "reference_shift_y_mm",
    ]
    with path.open("w", encoding="ascii") as stream:
        stream.write("Geant4 EHE geometry versus MATLAB Params\n")
        for key in keys:
            stream.write(f"{key} = {result[key]}\n")


def write_plot(path: Path, result: dict[str, object]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    holes = result["geant_holes"]
    detectors = result["geant_detectors"]
    params_holes = result["params_holes"]
    shift_y = float(result["reference_shift_y_mm"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter([h["x_mm"] for h in holes], [h["z_mm"] for h in holes], s=5, label="Geant4")
    ax.scatter([h[0] for h in params_holes], [h[3] for h in params_holes], s=2, label="Params")
    ax.set_title("1250 collimator-hole centers")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.scatter([d["x_mm"] for d in detectors], [d["z_mm"] for d in detectors], s=4)
    ax.set_title("2312 NaI detector-bin centers")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    center_ids = sorted(range(len(holes)), key=lambda i: holes[i]["x_mm"] ** 2 + holes[i]["z_mm"] ** 2)[:9]
    for i in center_ids:
        circle = plt.Circle((holes[i]["x_mm"], holes[i]["z_mm"]), holes[i]["radius_mm"], fill=False)
        ax.add_patch(circle)
    ax.autoscale()
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Central triangular-lattice unit cells")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")

    ax = axes[1, 1]
    col_front = params_holes[0][1] + shift_y
    col_back = params_holes[0][2] + shift_y
    det_front = min(d["y_mm"] - d["size_y_mm"] / 2.0 for d in detectors)
    det_back = max(d["y_mm"] + d["size_y_mm"] / 2.0 for d in detectors)
    ax.axvspan(FOV_CENTER_Y_MM, col_front, color="#8bcf9b", alpha=0.35, label="FOV-to-collimator")
    ax.axvspan(col_front, col_back, color="#55565c", alpha=0.8, label="Pb")
    ax.axvspan(det_front, det_back, color="#e9ad23", alpha=0.9, label="NaI")
    ax.set_xlim(FOV_CENTER_Y_MM, det_back + 15.0)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Geant4 global Y (mm)")
    ax.set_title("Side profile")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower left")

    fig.suptitle("EHE Geant4 Geometry vs MATLAB Params", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    params_dir = args.params_dir.resolve()
    geant_dir = args.geant_output_dir.resolve()
    output_dir = (args.output_dir or geant_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = compare(params_dir, geant_dir, args.tolerance_mm)
    report_path = output_dir / "EHE_GeometryComparison.txt"
    plot_path = output_dir / "EHE_GeometryComparison.png"
    write_report(report_path, result)
    plotted = write_plot(plot_path, result)

    print("EHE Geant4 geometry versus Params: PASS")
    print(f"  detectors: {result['detector_count']}")
    print(f"  holes: {result['hole_count']}")
    print(f"  max detector error: {result['max_detector_error_mm']:.9g} mm")
    print(f"  max hole error: {result['max_hole_error_mm']:.9g} mm")
    print(f"  report: {report_path}")
    if plotted:
        print(f"  plot: {plot_path}")
    else:
        print("  plot skipped: matplotlib is not installed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
