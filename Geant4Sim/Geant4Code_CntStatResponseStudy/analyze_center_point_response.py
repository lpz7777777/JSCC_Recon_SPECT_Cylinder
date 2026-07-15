#!/usr/bin/env python3
"""Compare a mixed center-point Geant4 run with Cartesian matrix columns."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MATRIX_FILENAMES = {
    "218_pe": (
        "JSCC_218keV",
        "PE_Windowed_SysMat_shift_0.000000_0.000000_0.000000_v3.sysmat",
    ),
    "218_scatter": (
        "JSCC_218keV",
        "Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat",
    ),
    "218_direct": (
        "JSCC_218keV",
        "SysMat_withScatter_shift_0.000000_0.000000_0.000000.sysmat",
    ),
    "440_pe": (
        "JSCC_440keV",
        "PE_Windowed_SysMat_shift_0.000000_0.000000_0.000000_v3.sysmat",
    ),
    "440_scatter": (
        "JSCC_440keV",
        "Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat",
    ),
    "440_direct": (
        "JSCC_440keV",
        "SysMat_withScatter_shift_0.000000_0.000000_0.000000.sysmat",
    ),
    "cross": (
        "JSCC_440keV_to_218keVwin",
        "Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat",
    ),
}


def read_one_row(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if values.shape != (1, 10496):
        raise ValueError(f"Expected one 10496-bin row in {path}, got {values.shape}.")
    return values[0]


def extract_center_column(
    matrix_path: Path,
    detector_count: int,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    nx, ny, nz = image_shape
    voxel_count = nx * ny * nz
    matrix = np.memmap(
        matrix_path,
        dtype="<f4",
        mode="r",
        shape=(detector_count, voxel_count),
        order="C",
    )
    if nx % 2 != 1 or ny % 2 != 1 or nz % 2 != 0:
        raise ValueError("This center-point interpolation expects odd X/Y and even Z.")
    x_index = nx // 2
    y_index = ny // 2
    lower_z = nz // 2 - 1
    upper_z = nz // 2
    columns = [
        z_index * nx * ny + y_index * nx + x_index
        for z_index in (lower_z, upper_z)
    ]
    result = 0.5 * (
        np.asarray(matrix[:, columns[0]], dtype=np.float64)
        + np.asarray(matrix[:, columns[1]], dtype=np.float64)
    )
    del matrix
    return result


def response_metrics(geant4: np.ndarray, matrix: np.ndarray) -> dict[str, float]:
    geant4_total = float(geant4.sum())
    matrix_total = float(matrix.sum())
    return {
        "geant4_probability_per_primary": geant4_total,
        "matrix_probability_per_primary": matrix_total,
        "matrix_to_geant4_ratio": matrix_total / geant4_total,
        "detector_bin_correlation": float(np.corrcoef(geant4, matrix)[0, 1]),
        "normalized_l1_shape_distance": float(
            np.abs(geant4 / geant4_total - matrix / matrix_total).sum()
        ),
    }


def write_comparison_csv(
    path: Path,
    detector: np.ndarray,
    geant4: dict[str, np.ndarray],
    matrix: dict[str, np.ndarray],
) -> None:
    header = [
        "detector_index",
        "x_mm",
        "y_mm",
        "z_mm",
        "g4_218from218",
        "matrix_218direct",
        "g4_440from440",
        "matrix_440direct",
        "g4_218from440",
        "matrix_440to218",
    ]
    columns = np.column_stack(
        [
            np.arange(1, len(detector) + 1),
            detector[:, :3],
            geant4["218_direct"],
            matrix["218_direct"],
            geant4["440_direct"],
            matrix["440_direct"],
            geant4["cross"],
            matrix["cross"],
        ]
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(columns)


def make_figure(
    path: Path,
    detector: np.ndarray,
    geant4: dict[str, np.ndarray],
    matrix: dict[str, np.ndarray],
) -> None:
    layers = np.unique(detector[:, 1])
    labels = ["218 from 218", "440 from 440", "218 from 440"]
    keys = ["218_direct", "440_direct", "cross"]
    figure = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 4)

    axis = figure.add_subplot(grid[0, :2])
    positions = np.arange(len(keys))
    width = 0.36
    axis.bar(
        positions - width / 2,
        [geant4[key].sum() for key in keys],
        width,
        label="Geant4",
    )
    axis.bar(
        positions + width / 2,
        [matrix[key].sum() for key in keys],
        width,
        label="Current matrix",
    )
    axis.set_yscale("log")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Counts per emitted primary")
    axis.set_title("Center-point total response")
    axis.legend()

    axis = figure.add_subplot(grid[0, 2:])
    cross_g4 = np.array([geant4["cross"][detector[:, 1] == y].sum() for y in layers])
    cross_matrix = np.array([matrix["cross"][detector[:, 1] == y].sum() for y in layers])
    layer_positions = np.arange(len(layers))
    axis.bar(layer_positions - width / 2, cross_g4 / cross_g4.sum(), width, label="Geant4")
    axis.bar(
        layer_positions + width / 2,
        cross_matrix / cross_matrix.sum(),
        width,
        label="Current matrix",
    )
    axis.set_xticks(layer_positions, [f"{value:g}" for value in layers])
    axis.set_xlabel("Detector layer y (mm)")
    axis.set_ylabel("Fraction of 440-to-218 response")
    axis.set_title("Cross-window depth distribution")
    axis.legend()

    last_layer = detector[:, 1] == layers[-1]
    x_values = np.unique(detector[last_layer, 0])
    z_values = np.unique(detector[last_layer, 2])

    def image(values: np.ndarray) -> np.ndarray:
        output = np.full((len(z_values), len(x_values)), np.nan)
        x_index = np.searchsorted(x_values, detector[last_layer, 0])
        z_index = np.searchsorted(z_values, detector[last_layer, 2])
        output[z_index, x_index] = values[last_layer]
        return output

    maps = [
        (image(geant4["cross"]), "Geant4 440 to 218"),
        (image(matrix["cross"]), "Current matrix 440 to 218"),
    ]
    maximum = max(np.nanmax(item[0]) for item in maps)
    for column, (values, title) in enumerate(maps):
        axis = figure.add_subplot(grid[1, column * 2 : column * 2 + 2])
        plotted = axis.imshow(
            values,
            origin="lower",
            extent=[x_values[0], x_values[-1], z_values[0], z_values[-1]],
            aspect="auto",
            vmin=0.0,
            vmax=maximum,
            cmap="magma",
        )
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("z (mm)")
        axis.set_title(f"{title}, y={layers[-1]:g} mm")
        figure.colorbar(plotted, ax=axis, label="Counts per primary")

    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=script_dir / "build")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=project_root
        / "Auxiliary_Studies"
        / "GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main"
        / "runs",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.build_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_218 = int(np.loadtxt(args.build_dir / "PrimaryCount218.csv"))
    primary_440 = int(np.loadtxt(args.build_dir / "PrimaryCount440.csv"))
    primary_other = int(np.loadtxt(args.build_dir / "PrimaryCountOther.csv"))
    geant4 = {
        "218_direct": read_one_row(args.build_dir / "CntStat218_From218.csv")
        / primary_218,
        "440_direct": read_one_row(args.build_dir / "CntStat440_From440.csv")
        / primary_440,
        "cross": read_one_row(args.build_dir / "CntStat218_From440.csv")
        / primary_440,
    }

    detector_raw = np.fromfile(
        args.runs_dir / "JSCC_218keV" / "Params_Detector.dat", dtype="<f4"
    )
    detector = detector_raw[1:].reshape(-1, 12)
    active = detector[:, 11] == 1
    image = np.fromfile(
        args.runs_dir / "JSCC_218keV" / "Params_Image.dat", dtype="<f4"
    )
    image_shape = tuple(int(value) for value in image[:3])

    matrix = {}
    all_matrix_components = {}
    for key, (run_name, filename) in MATRIX_FILENAMES.items():
        values = extract_center_column(
            args.runs_dir / run_name / filename,
            len(detector),
            image_shape,
        )[active]
        all_matrix_components[key] = values
        if key in geant4:
            matrix[key] = values

    closure_218 = read_one_row(args.build_dir / "CntStat_218.csv") - (
        read_one_row(args.build_dir / "CntStat218_From218.csv")
        + read_one_row(args.build_dir / "CntStat218_From440.csv")
    )
    closure_440 = read_one_row(args.build_dir / "CntStat_440.csv") - read_one_row(
        args.build_dir / "CntStat440_From440.csv"
    )
    layers = np.unique(detector[active, 1])
    summary = {
        "primary_counts": {
            "218": primary_218,
            "440": primary_440,
            "other": primary_other,
            "ratio_440_to_218": primary_440 / primary_218,
        },
        "closure": {
            "218_max_abs_bin_difference": float(np.abs(closure_218).max()),
            "440_max_abs_bin_difference": float(np.abs(closure_440).max()),
        },
        "response": {
            key: response_metrics(geant4[key], matrix[key]) for key in geant4
        },
        "matrix_component_totals": {
            key: float(values.sum()) for key, values in all_matrix_components.items()
        },
        "cross_by_layer": {
            f"{layer:g}": {
                "geant4": float(geant4["cross"][detector[active, 1] == layer].sum()),
                "matrix": float(matrix["cross"][detector[active, 1] == layer].sum()),
            }
            for layer in layers
        },
    }
    for values in summary["cross_by_layer"].values():
        values["matrix_to_geant4_ratio"] = values["matrix"] / values["geant4"]

    (output_dir / "center_point_response_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_comparison_csv(
        output_dir / "center_point_detector_comparison.csv",
        detector[active],
        geant4,
        matrix,
    )
    make_figure(
        output_dir / "center_point_response_comparison.png",
        detector[active],
        geant4,
        matrix,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
