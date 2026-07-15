#!/usr/bin/env python3
"""Validate a full ScatterGen target-surface run and compare its center column."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCATTER_NAME = "Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat"
COMBINED_NAME = "SysMat_withScatter_shift_0.000000_0.000000_0.000000.sysmat"
COMPONENT_NAMES = (
    "C_intercrystal.sysmat",
    "C_highZ_to_crystal.sysmat",
    "C_local_recoil.sysmat",
    "C_local_self_photoelectric.sysmat",
    "C_collimator_to_crystal.sysmat",
)
TOTAL_NAME = "C_total.sysmat"


def native_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def read_params(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detector_raw = np.fromfile(run_dir / "Params_Detector.dat", dtype="<f4")
    detector_count = int(detector_raw[0])
    expected_detector_values = 1 + 12 * detector_count
    if detector_raw.size != expected_detector_values:
        raise ValueError(
            f"Params_Detector.dat has {detector_raw.size} floats; "
            f"expected {expected_detector_values}."
        )
    detector = detector_raw[1:].reshape(detector_count, 12)

    image = np.fromfile(run_dir / "Params_Image.dat", dtype="<f4")
    if image.size < 12:
        raise ValueError("Params_Image.dat must contain at least 12 floats.")
    physics = np.fromfile(run_dir / "Params_Physics.dat", dtype="<f4")
    if physics.size < 12:
        raise ValueError("Params_Physics.dat must contain all 12 physics values.")
    return detector, image, physics


def matrix_shape(detector: np.ndarray, image: np.ndarray) -> tuple[int, int]:
    image_shape = tuple(int(value) for value in image[:3])
    rotations = int(math.floor(float(image[6]) + 0.001))
    voxel_count = math.prod(image_shape) * rotations
    return len(detector), voxel_count


def matrix_map(path: Path, shape: tuple[int, int]) -> np.memmap:
    expected_bytes = math.prod(shape) * np.dtype("<f4").itemsize
    path_native = native_path(path)
    if not os.path.isfile(path_native):
        raise FileNotFoundError(path)
    actual_bytes = os.stat(path_native).st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{path.name}: expected {expected_bytes} bytes, found {actual_bytes}."
        )
    return np.memmap(
        path_native, dtype="<f4", mode="r", shape=shape, order="C"
    )


def photopeak_acceptance(detector: np.ndarray, physics: np.ndarray) -> np.ndarray:
    source_energy = float(physics[7])
    if int(math.floor(float(physics[4]) + 0.5)) > 0:
        lower = float(physics[5])
        upper = float(physics[6])
    else:
        lower = None
        upper = None

    values = np.empty(len(detector), dtype=np.float32)
    for index, relative_fwhm_raw in enumerate(detector[:, 9]):
        relative_fwhm = float(relative_fwhm_raw)
        row_lower = lower
        row_upper = upper
        if row_lower is None or row_upper is None:
            row_lower = (1.0 - relative_fwhm / 2.0) * source_energy
            row_upper = (1.0 + relative_fwhm / 2.0) * source_energy
        if relative_fwhm <= 0.0:
            probability = float(row_lower <= source_energy <= row_upper)
        else:
            sigma = relative_fwhm * source_energy / 2.35482
            denominator = sigma * math.sqrt(2.0)
            z_lower = (row_lower - source_energy) / denominator
            z_upper = (row_upper - source_energy) / denominator
            probability = 0.5 * (math.erf(z_upper) - math.erf(z_lower))
            probability = min(1.0, max(0.0, probability))
        values[index] = np.float32(probability)
    return values


def initialize_stats() -> dict[str, float | int | None]:
    return {
        "minimum": None,
        "maximum": None,
        "sum": 0.0,
        "nonzero": 0,
        "invalid": 0,
        "negative": 0,
    }


def update_stats(stats: dict[str, float | int | None], values: np.ndarray) -> None:
    finite = np.isfinite(values)
    stats["invalid"] = int(stats["invalid"]) + int((~finite).sum())
    finite_values = values[finite]
    if finite_values.size == 0:
        return
    current_minimum = float(finite_values.min())
    current_maximum = float(finite_values.max())
    stats["minimum"] = (
        current_minimum
        if stats["minimum"] is None
        else min(float(stats["minimum"]), current_minimum)
    )
    stats["maximum"] = (
        current_maximum
        if stats["maximum"] is None
        else max(float(stats["maximum"]), current_maximum)
    )
    stats["sum"] = float(stats["sum"]) + float(
        finite_values.sum(dtype=np.float64)
    )
    stats["nonzero"] = int(stats["nonzero"]) + int(
        np.count_nonzero(finite_values)
    )
    stats["negative"] = int(stats["negative"]) + int(
        np.count_nonzero(finite_values < 0.0)
    )


def validate_matrices(
    run_dir: Path,
    detector: np.ndarray,
    image: np.ndarray,
    physics: np.ndarray,
    pe_path: Path,
    chunk_rows: int,
) -> dict[str, object]:
    shape = matrix_shape(detector, image)
    paths = {
        SCATTER_NAME: run_dir / SCATTER_NAME,
        COMBINED_NAME: run_dir / COMBINED_NAME,
        TOTAL_NAME: run_dir / TOTAL_NAME,
        **{name: run_dir / name for name in COMPONENT_NAMES},
    }
    matrices = {name: matrix_map(path, shape) for name, path in paths.items()}
    pe = matrix_map(pe_path, shape)
    acceptance = photopeak_acceptance(detector, physics)
    stats = {name: initialize_stats() for name in matrices}

    maximum_scatter_total_error = 0.0
    maximum_component_closure_error = 0.0
    maximum_combined_closure_error = 0.0
    maximum_component_relative_error = 0.0
    maximum_combined_relative_error = 0.0

    for start in range(0, shape[0], chunk_rows):
        end = min(shape[0], start + chunk_rows)
        chunks = {
            name: np.asarray(matrix[start:end, :], dtype=np.float32)
            for name, matrix in matrices.items()
        }
        for name, values in chunks.items():
            update_stats(stats[name], values)

        scatter = chunks[SCATTER_NAME]
        total = chunks[TOTAL_NAME]
        component_sum = np.zeros_like(scatter)
        for name in COMPONENT_NAMES:
            component_sum += chunks[name]
        expected_combined = (
            np.asarray(pe[start:end, :], dtype=np.float32)
            * acceptance[start:end, np.newaxis]
            + scatter
        )

        scatter_total_error = np.abs(scatter.astype(np.float64) - total)
        component_error = np.abs(total.astype(np.float64) - component_sum)
        combined_error = np.abs(
            chunks[COMBINED_NAME].astype(np.float64) - expected_combined
        )
        maximum_scatter_total_error = max(
            maximum_scatter_total_error, float(scatter_total_error.max())
        )
        maximum_component_closure_error = max(
            maximum_component_closure_error, float(component_error.max())
        )
        maximum_combined_closure_error = max(
            maximum_combined_closure_error, float(combined_error.max())
        )
        maximum_component_relative_error = max(
            maximum_component_relative_error,
            float((component_error / np.maximum(np.abs(total), 1e-30)).max()),
        )
        maximum_combined_relative_error = max(
            maximum_combined_relative_error,
            float(
                (
                    combined_error
                    / np.maximum(np.abs(expected_combined.astype(np.float64)), 1e-30)
                ).max()
            ),
        )

    invalid_or_negative = {
        name: {
            "invalid": int(values["invalid"]),
            "negative": int(values["negative"]),
        }
        for name, values in stats.items()
        if int(values["invalid"]) or int(values["negative"])
    }
    tolerances = {
        "scatter_total_absolute": 0.0,
        "component_absolute": 2e-7,
        "combined_absolute": 2e-7,
    }
    checks = {
        "all_files_exact_size": True,
        "all_values_finite_and_nonnegative": not invalid_or_negative,
        "scatter_equals_total": maximum_scatter_total_error
        <= tolerances["scatter_total_absolute"],
        "components_close_to_total": maximum_component_closure_error
        <= tolerances["component_absolute"],
        "combined_closes": maximum_combined_closure_error
        <= tolerances["combined_absolute"],
    }
    result: dict[str, object] = {
        "shape": list(shape),
        "expected_bytes_per_matrix": math.prod(shape) * 4,
        "matrix_stats": stats,
        "closure": {
            "maximum_scatter_total_absolute_error": maximum_scatter_total_error,
            "maximum_component_absolute_error": maximum_component_closure_error,
            "maximum_component_relative_error": maximum_component_relative_error,
            "maximum_combined_absolute_error": maximum_combined_closure_error,
            "maximum_combined_relative_error": maximum_combined_relative_error,
        },
        "tolerances": tolerances,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    return result


def build_combined_matrix(
    run_dir: Path,
    detector: np.ndarray,
    image: np.ndarray,
    physics: np.ndarray,
    pe_path: Path,
    chunk_rows: int,
) -> Path:
    shape = matrix_shape(detector, image)
    pe = matrix_map(pe_path, shape)
    scatter = matrix_map(run_dir / SCATTER_NAME, shape)
    acceptance = photopeak_acceptance(detector, physics)
    destination = run_dir / COMBINED_NAME
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if os.path.exists(native_path(temporary)):
        os.unlink(native_path(temporary))
    combined = np.memmap(
        native_path(temporary), dtype="<f4", mode="w+", shape=shape, order="C"
    )
    for start in range(0, shape[0], chunk_rows):
        end = min(shape[0], start + chunk_rows)
        combined[start:end, :] = (
            np.asarray(pe[start:end, :], dtype=np.float32)
            * acceptance[start:end, np.newaxis]
            + np.asarray(scatter[start:end, :], dtype=np.float32)
        )
    combined.flush()
    del combined
    os.replace(native_path(temporary), native_path(destination))
    return destination


def read_geant4_cross(build_dir: Path) -> tuple[np.ndarray, int]:
    primary_count = int(np.loadtxt(build_dir / "PrimaryCount440.csv"))
    counts = np.loadtxt(
        build_dir / "CntStat218_From440.csv", delimiter=",", dtype=np.float64
    )
    counts = np.ravel(counts)
    return counts / primary_count, primary_count


def extract_center_column(
    matrix_path: Path,
    shape: tuple[int, int],
    image: np.ndarray,
) -> np.ndarray:
    nx, ny, nz = (int(value) for value in image[:3])
    if nx % 2 != 1 or ny % 2 != 1 or nz % 2 != 0:
        raise ValueError("Center interpolation requires odd X/Y and even Z.")
    lower_z = nz // 2 - 1
    upper_z = nz // 2
    columns = [
        z * nx * ny + (ny // 2) * nx + nx // 2
        for z in (lower_z, upper_z)
    ]
    matrix = matrix_map(matrix_path, shape)
    return 0.5 * (
        np.asarray(matrix[:, columns[0]], dtype=np.float64)
        + np.asarray(matrix[:, columns[1]], dtype=np.float64)
    )


def compare_center_response(
    run_dir: Path,
    detector: np.ndarray,
    image: np.ndarray,
    build_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    active = detector[:, 11] == 1
    active_detector = detector[active]
    geant4, primary_count = read_geant4_cross(build_dir)
    if geant4.size != int(active.sum()):
        raise ValueError(
            f"Geant4 has {geant4.size} detector bins; expected {int(active.sum())}."
        )
    matrix = extract_center_column(
        run_dir / SCATTER_NAME, matrix_shape(detector, image), image
    )[active]
    geant4_total = float(geant4.sum())
    matrix_total = float(matrix.sum())
    layers = np.unique(active_detector[:, 1])

    rows = []
    layer_summary = {}
    for layer in layers:
        selection = active_detector[:, 1] == layer
        geant4_layer = float(geant4[selection].sum())
        matrix_layer = float(matrix[selection].sum())
        layer_summary[f"{layer:g}"] = {
            "geant4": geant4_layer,
            "matrix": matrix_layer,
            "matrix_to_geant4_ratio": matrix_layer / geant4_layer,
        }
    for index in range(len(active_detector)):
        rows.append(
            [
                index + 1,
                *[float(value) for value in active_detector[index, :3]],
                float(geant4[index]),
                float(matrix[index]),
            ]
        )
    with (output_dir / "center_point_cross_response.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "active_detector_index",
                "x_mm",
                "y_mm",
                "z_mm",
                "geant4_probability_per_primary",
                "matrix_probability_per_primary",
            ]
        )
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    positions = np.arange(len(layers))
    width = 0.36
    geant4_layers = np.array(
        [layer_summary[f"{layer:g}"]["geant4"] for layer in layers]
    )
    matrix_layers = np.array(
        [layer_summary[f"{layer:g}"]["matrix"] for layer in layers]
    )
    axes[0].bar(positions - width / 2, geant4_layers, width, label="Geant4")
    axes[0].bar(positions + width / 2, matrix_layers, width, label="Matrix")
    axes[0].set_xticks(positions, [f"{value:g}" for value in layers])
    axes[0].set_xlabel("Detector layer y (mm)")
    axes[0].set_ylabel("Counts per 440 keV primary")
    axes[0].set_title("440 keV source in 218 keV window")
    axes[0].legend()

    axes[1].plot(
        positions,
        matrix_layers / geant4_layers,
        marker="o",
        color="tab:red",
    )
    axes[1].axhline(1.0, color="black", linewidth=1)
    axes[1].set_xticks(positions, [f"{value:g}" for value in layers])
    axes[1].set_xlabel("Detector layer y (mm)")
    axes[1].set_ylabel("Matrix / Geant4")
    axes[1].set_title("Layer response ratio")

    last_layer = active_detector[:, 1] == layers[-1]
    scatter = axes[2].scatter(
        active_detector[last_layer, 0],
        active_detector[last_layer, 2],
        c=matrix[last_layer] / np.maximum(geant4[last_layer], 1e-30),
        s=9,
        cmap="coolwarm",
        vmin=0.0,
        vmax=2.0,
    )
    axes[2].set_xlabel("x (mm)")
    axes[2].set_ylabel("z (mm)")
    axes[2].set_title(f"Last-layer bin ratio, y={layers[-1]:g} mm")
    figure.colorbar(scatter, ax=axes[2], label="Matrix / Geant4")
    figure.savefig(output_dir / "center_point_cross_response.png", dpi=180)
    plt.close(figure)

    return {
        "geant4_primary_count_440": primary_count,
        "geant4_probability_per_primary": geant4_total,
        "matrix_probability_per_primary": matrix_total,
        "matrix_to_geant4_ratio": matrix_total / geant4_total,
        "detector_bin_correlation": float(np.corrcoef(geant4, matrix)[0, 1]),
        "normalized_l1_shape_distance": float(
            np.abs(geant4 / geant4_total - matrix / matrix_total).sum()
        ),
        "by_layer": layer_summary,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repository = script_dir.parent
    project_root = repository.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pe", type=Path, default=None)
    parser.add_argument(
        "--geant4-build",
        type=Path,
        default=project_root
        / "Geant4Sim"
        / "Geant4Code_CntStatResponseStudy"
        / "build",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--chunk-rows", type=int, default=16)
    parser.add_argument(
        "--build-missing-combined",
        action="store_true",
        help="Build a missing combined matrix from PE and Scatter before validation.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "validation_report").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pe_path = (args.pe or run_dir / "PE_SysMat_shift_0.000000_0.000000_0.000000_v3.sysmat").resolve()
    detector, image, physics = read_params(run_dir)

    combined_path = run_dir / COMBINED_NAME
    if (
        not os.path.isfile(native_path(combined_path))
        and args.build_missing_combined
    ):
        build_combined_matrix(
            run_dir,
            detector,
            image,
            physics,
            pe_path,
            args.chunk_rows,
        )

    summary = {
        "run_dir": str(run_dir),
        "pe_path": str(pe_path),
        "physics": [float(value) for value in physics[:12]],
        "full_matrix_validation": validate_matrices(
            run_dir,
            detector,
            image,
            physics,
            pe_path,
            args.chunk_rows,
        ),
        "center_point_440_to_218": compare_center_response(
            run_dir,
            detector,
            image,
            args.geant4_build.resolve(),
            output_dir,
        ),
    }
    summary["status"] = summary["full_matrix_validation"]["status"]
    report_path = output_dir / "validation_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
