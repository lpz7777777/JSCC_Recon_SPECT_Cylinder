#!/usr/bin/env python3
"""Run converged PE v4 reference checks on representative JSCC pairs."""

from __future__ import annotations

import argparse
import array
import csv
import json
import math
import subprocess
import sys
from pathlib import Path


def read_float32(path: Path) -> list[float]:
    values = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def detector_records(values: list[float]) -> list[list[float]]:
    count = round(values[0])
    required = 1 + 12 * count
    if len(values) < required:
        raise ValueError(f"Params_Detector.dat has {len(values)} values; need {required}")
    return [values[1 + 12 * index : 1 + 12 * (index + 1)] for index in range(count)]


def nearest_active_detector(records: list[list[float]], layer_y: float) -> int:
    candidates = [
        (index, record)
        for index, record in enumerate(records)
        if round(record[11]) == 1 and abs(record[1] - layer_y) < 1e-4
    ]
    if not candidates:
        raise ValueError(f"No active detector found at local y={layer_y} mm")
    return min(candidates, key=lambda item: abs(item[1][0]) + abs(item[1][2]))[0]


def nearest_voxel(image: list[float], target_x: float, target_y: float, target_z: float) -> int:
    count_x, count_y, count_z = (round(value) for value in image[:3])
    axes = []
    for count, width, shift, target in zip(
        (count_x, count_y, count_z), image[3:6], image[8:11],
        (target_x, target_y, target_z),
    ):
        coordinates = [(index - count / 2 + 0.5) * width + shift for index in range(count)]
        axes.append(min(range(count), key=lambda index: abs(coordinates[index] - target)))
    index_x, index_y, index_z = axes
    return (index_z * count_y + index_y) * count_x + index_x


def read_single_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(rows)}")
    return rows[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="JSCC run directory containing Params_*.dat and raw v3 PE")
    parser.add_argument("--binary", type=Path, required=True, help="PEGen_V4_Reference executable")
    parser.add_argument("--output-dir", type=Path, help="Default: <run_dir>/pe_v4_reference_validation")
    parser.add_argument("--face-levels", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--depth-subdiv", type=int, default=8)
    parser.add_argument("--surface-rule", choices=["halton", "gauss"], default="halton")
    parser.add_argument("--layer-y", type=float, nargs="+", default=[30, 60, 90, 120])
    parser.add_argument("--source-y", type=float, nargs="+", default=[0, 150])
    parser.add_argument("--maximum-relative-convergence", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    binary = args.binary.resolve()
    output_dir = (args.output_dir or run_dir / "pe_v4_reference_validation").resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    if len(args.face_levels) < 2 or sorted(set(args.face_levels)) != args.face_levels:
        raise ValueError("--face-levels must contain at least two strictly increasing values")
    if any(level < 1 for level in args.face_levels) or args.depth_subdiv < 1:
        raise ValueError("Subdivision values must be positive")

    detector_values = read_float32(run_dir / "Params_Detector.dat")
    image = read_float32(run_dir / "Params_Image.dat")
    image.extend([0.0] * max(0, 12 - len(image)))
    records = detector_records(detector_values)
    detector_count = len(records)
    voxel_count = round(image[0]) * round(image[1]) * round(image[2])
    matrix_path = run_dir / "PE_SysMat_shift_0.000000_0.000000_0.000000_v3.sysmat"
    expected_matrix_bytes = detector_count * voxel_count * round(image[6]) * 4
    if matrix_path.stat().st_size != expected_matrix_bytes:
        raise ValueError(
            f"Unexpected v3 matrix size {matrix_path.stat().st_size}; expected {expected_matrix_bytes}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    details: list[dict[str, object]] = []
    pair_summaries: list[dict[str, object]] = []
    failed = False
    for layer_y in args.layer_y:
        detector_index = nearest_active_detector(records, layer_y)
        for source_y in args.source_y:
            voxel_index = nearest_voxel(image, 0.0, source_y, 0.0)
            level_rows = []
            for level in args.face_levels:
                csv_path = output_dir / (
                    f"layer_{layer_y:g}_source_y_{source_y:g}_{args.surface_rule}_face_{level}.csv"
                )
                command = [
                    str(binary),
                    "--detector", str(detector_index),
                    "--voxel", str(voxel_index),
                    "--face-subdiv", str(level),
                    "--depth-subdiv", str(args.depth_subdiv),
                    "--surface-rule", args.surface_rule,
                    "--v3", str(matrix_path),
                    "--output", str(csv_path),
                ]
                completed = subprocess.run(
                    command,
                    cwd=run_dir,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Reference failed for layer={layer_y}, source_y={source_y}, "
                        f"face={level}:\n{completed.stdout}"
                    )
                row = read_single_csv(csv_path)
                numeric = {
                    "layer_y_mm": layer_y,
                    "source_y_mm": source_y,
                    "surface_rule": row["surface_rule"],
                    "detector_index": detector_index,
                    "voxel_index": voxel_index,
                    "face_subdivisions": level,
                    "photoelectric_probability": float(row["photoelectric_probability"]),
                    "compton_probability": float(row["compton_probability"]),
                    "first_interaction_probability": float(row["first_interaction_probability"]),
                    "closure_error": float(row["closure_error"]),
                    "mean_depth_mm": float(row["mean_depth_mm"]),
                    "v3_photoelectric_probability": float(row["v3_photoelectric_probability"]),
                    "v4_over_v3": float(row["v4_over_v3"]),
                }
                details.append(numeric)
                level_rows.append(numeric)

            coarse = level_rows[-2]["photoelectric_probability"]
            fine = level_rows[-1]["photoelectric_probability"]
            relative_convergence = abs(float(fine) - float(coarse)) / max(abs(float(fine)), 1e-30)
            maximum_closure = max(abs(float(row["closure_error"])) for row in level_rows)
            pair_passed = (
                relative_convergence <= args.maximum_relative_convergence
                and maximum_closure <= 1e-12
                and math.isfinite(float(level_rows[-1]["v4_over_v3"]))
            )
            failed = failed or not pair_passed
            pair_summaries.append({
                "layer_y_mm": layer_y,
                "source_y_mm": source_y,
                "detector_index": detector_index,
                "voxel_index": voxel_index,
                "coarse_face_subdivisions": level_rows[-2]["face_subdivisions"],
                "fine_face_subdivisions": level_rows[-1]["face_subdivisions"],
                "relative_convergence": relative_convergence,
                "fine_v4_over_v3": level_rows[-1]["v4_over_v3"],
                "maximum_closure_error": maximum_closure,
                "passed": pair_passed,
            })

    with (output_dir / "pe_v4_reference_details.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    summary = {
        "status": "FAIL" if failed else "PASS",
        "run_directory": str(run_dir),
        "binary": str(binary),
        "face_levels": args.face_levels,
        "depth_subdivisions": args.depth_subdiv,
        "surface_rule": args.surface_rule,
        "maximum_relative_convergence": args.maximum_relative_convergence,
        "pairs": pair_summaries,
    }
    (output_dir / "pe_v4_reference_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = [
        "# PE v4 JSCC Reference Validation",
        "",
        f"Status: **{summary['status']}**",
        f"Surface rule: `{args.surface_rule}`",
        "",
        "| Layer y (mm) | Source y (mm) | Detector | Voxel | Convergence | Fine v4/v3 | Pass |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for item in pair_summaries:
        report.append(
            f"| {item['layer_y_mm']:g} | {item['source_y_mm']:g} | "
            f"{item['detector_index']} | {item['voxel_index']} | "
            f"{item['relative_convergence']:.6g} | {item['fine_v4_over_v3']:.6g} | "
            f"{'yes' if item['passed'] else 'no'} |"
        )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nOutput: {output_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
