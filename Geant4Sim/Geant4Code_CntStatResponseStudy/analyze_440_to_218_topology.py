#!/usr/bin/env python3
"""Compare topology-separated Geant4 440-to-218 counts with matrix components."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TOPOLOGY_FILES = {
    "total": "CntStat218_From440.csv",
    "first_crystal": "CntStat218_From440_FirstCrystal.csv",
    "other_crystal": "CntStat218_From440_OtherCrystal.csv",
    "hit_1": "CntStat218_From440_Hit1.csv",
    "hit_2": "CntStat218_From440_Hit2.csv",
    "hit_3plus": "CntStat218_From440_Hit3Plus.csv",
    "first_compton_0": "CntStat218_From440_FirstCrystal_Compton0.csv",
    "first_compton_1": "CntStat218_From440_FirstCrystal_Compton1.csv",
    "first_compton_2plus": "CntStat218_From440_FirstCrystal_Compton2Plus.csv",
}

MATRIX_FILES = {
    "local_recoil": "C_local_recoil.sysmat",
    "intercrystal": "C_intercrystal.sysmat",
    "highz_to_crystal": "C_highZ_to_crystal.sysmat",
    "total": "C_total.sysmat",
}

WINDOW_218_LOWER_KEV = 196.3053741455078
WINDOW_218_UPPER_KEV = 239.6946258544922
SOURCE_CLASS_MIDPOINT_KEV = 0.5 * (218.0 + 440.0)
SMOOTHING_RADII = (0, 1, 2, 4, 8)


def read_csv_values(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8").strip().rstrip(",")
    if not text:
        return np.empty(0, dtype=np.float64)
    return np.fromstring(text, sep=",", dtype=np.float64)


def read_count_row(path: Path, expected_count: int) -> np.ndarray:
    values = read_csv_values(path)
    if values.size != expected_count:
        raise ValueError(
            f"{path.name}: expected {expected_count} bins, found {values.size}."
        )
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0)
        or np.any(values != np.rint(values))
    ):
        raise ValueError(f"{path.name}: counts must be finite nonnegative integers.")
    return values.astype(np.int64)


def read_scalar_count(path: Path) -> int:
    values = read_csv_values(path)
    if (
        values.size != 1
        or not np.isfinite(values[0])
        or values[0] < 0
        or values[0] != round(float(values[0]))
    ):
        raise ValueError(f"{path.name}: expected one finite nonnegative integer.")
    return int(round(float(values[0])))


def center_column(
    path: Path,
    detector_count: int,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    nx, ny, nz = image_shape
    voxel_count = nx * ny * nz
    expected_bytes = detector_count * voxel_count * np.dtype("<f4").itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{path.name}: expected {expected_bytes} bytes, found {path.stat().st_size}."
        )
    matrix = np.memmap(
        path,
        dtype="<f4",
        mode="r",
        shape=(detector_count, voxel_count),
        order="C",
    )
    columns = [
        z * nx * ny + (ny // 2) * nx + nx // 2
        for z in (nz // 2 - 1, nz // 2)
    ]
    result = 0.5 * (
        np.asarray(matrix[:, columns[0]], dtype=np.float64)
        + np.asarray(matrix[:, columns[1]], dtype=np.float64)
    )
    del matrix
    return result


def grouped_summary(values: np.ndarray, detector: np.ndarray) -> dict[str, object]:
    return {
        "total": float(values.sum()),
        "by_layer": {
            f"{layer:g}": float(values[detector[:, 1] == layer].sum())
            for layer in np.unique(detector[:, 1])
        },
    }


def box_sum(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.copy()
    kernel = 2 * radius + 1
    padded = np.pad(values, radius, mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = np.cumsum(np.cumsum(integral, axis=0), axis=1)
    height, width = values.shape
    return (
        integral[kernel : kernel + height, kernel : kernel + width]
        - integral[:height, kernel : kernel + width]
        - integral[kernel : kernel + height, :width]
        + integral[:height, :width]
    )


def smooth_detector_planes(
    values: np.ndarray,
    detector: np.ndarray,
    radius: int,
) -> np.ndarray:
    if radius <= 0:
        return values.copy()
    output = np.zeros_like(values, dtype=np.float64)
    for layer in np.unique(detector[:, 1]):
        indices = np.flatnonzero(detector[:, 1] == layer)
        x_values = np.unique(detector[indices, 0])
        z_values = np.unique(detector[indices, 2])
        x_index = np.searchsorted(x_values, detector[indices, 0])
        z_index = np.searchsorted(z_values, detector[indices, 2])
        grid = np.zeros((len(z_values), len(x_values)), dtype=np.float64)
        mask = np.zeros_like(grid)
        grid[z_index, x_index] = values[indices]
        mask[z_index, x_index] = 1.0
        numerator = box_sum(grid, radius)
        denominator = box_sum(mask, radius)
        output[indices] = numerator[z_index, x_index] / np.maximum(
            denominator[z_index, x_index], 1.0
        )
    return output


def shape_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference_total = float(reference.sum())
    candidate_total = float(candidate.sum())
    if reference_total <= 0 or candidate_total <= 0:
        return {"correlation": float("nan"), "normalized_l1": float("nan")}
    return {
        "correlation": float(np.corrcoef(reference, candidate)[0, 1]),
        "normalized_l1": float(
            np.abs(reference / reference_total - candidate / candidate_total).sum()
        ),
    }


def comparison_summary(
    matrix: np.ndarray,
    geant4: np.ndarray,
    detector: np.ndarray,
) -> dict[str, object]:
    matrix_total = float(matrix.sum())
    geant4_total = float(geant4.sum())
    scale_denominator = float(np.dot(matrix, matrix))
    result: dict[str, object] = {
        "matrix_total": matrix_total,
        "geant4_total": geant4_total,
        "matrix_to_geant4": matrix_total / geant4_total,
        "least_squares_matrix_scale": float(np.dot(matrix, geant4))
        / scale_denominator,
        "shape_by_smoothing_radius_pitches": {},
        "by_layer": {},
    }
    for radius in SMOOTHING_RADII:
        matrix_smoothed = smooth_detector_planes(matrix, detector, radius)
        geant4_smoothed = smooth_detector_planes(geant4, detector, radius)
        result["shape_by_smoothing_radius_pitches"][str(radius)] = shape_metrics(
            matrix_smoothed, geant4_smoothed
        )
    for layer in np.unique(detector[:, 1]):
        selected = detector[:, 1] == layer
        matrix_layer = float(matrix[selected].sum())
        geant4_layer = float(geant4[selected].sum())
        result["by_layer"][f"{layer:g}"] = {
            "matrix": matrix_layer,
            "geant4": geant4_layer,
            "matrix_to_geant4": matrix_layer / geant4_layer,
        }
    return result


def analyze_list(
    path: Path,
    detector_count: int,
    primary_count_440: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    rows = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if rows.shape[1] != 5 or not np.all(np.isfinite(rows)):
        raise ValueError(f"{path.name}: expected five finite columns.")

    first_index = rows[:, 0].astype(np.int64) - 1
    second_index = rows[:, 2].astype(np.int64) - 1
    if (
        np.any(rows[:, 0] != first_index + 1)
        or np.any(rows[:, 2] != second_index + 1)
        or np.any(first_index < 0)
        or np.any(first_index >= detector_count)
        or np.any(second_index < 0)
        or np.any(second_index >= detector_count)
    ):
        raise ValueError(f"{path.name}: crystal IDs must be 1-based integers in range.")

    first_energy_kev = rows[:, 1] * 1000.0
    second_energy_kev = rows[:, 3] * 1000.0
    summed_energy_kev = first_energy_kev + second_energy_kev
    inferred_440 = summed_energy_kev >= SOURCE_CLASS_MIDPOINT_KEV
    first_in_window = inferred_440 & (
        (first_energy_kev >= WINDOW_218_LOWER_KEV)
        & (first_energy_kev <= WINDOW_218_UPPER_KEV)
    )
    second_in_window = inferred_440 & (
        (second_energy_kev >= WINDOW_218_LOWER_KEV)
        & (second_energy_kev <= WINDOW_218_UPPER_KEV)
    )

    first_vector = np.bincount(
        first_index[first_in_window], minlength=detector_count
    ).astype(np.float64) / primary_count_440
    second_vector = np.bincount(
        second_index[second_in_window], minlength=detector_count
    ).astype(np.float64) / primary_count_440
    percentiles = (0.0, 1.0, 25.0, 50.0, 75.0, 99.0, 100.0)
    summary = {
        "row_count": int(len(rows)),
        "crystal_id_base_in_file": 1,
        "energy_columns_unit_in_file": "MeV",
        "primary_energy_classification": (
            "inferred 440 when E1+E2 >= 329 keV; List.csv has no primary tag"
        ),
        "inferred_218_rows": int(np.count_nonzero(~inferred_440)),
        "inferred_440_rows": int(np.count_nonzero(inferred_440)),
        "inferred_440_first_bin_counts_in_218_window": int(
            np.count_nonzero(first_in_window)
        ),
        "inferred_440_second_bin_counts_in_218_window": int(
            np.count_nonzero(second_in_window)
        ),
        "inferred_440_both_bins_in_218_window_events": int(
            np.count_nonzero(first_in_window & second_in_window)
        ),
        "summed_energy_kev_percentiles": {
            f"{value:g}": float(result)
            for value, result in zip(
                percentiles, np.percentile(summed_energy_kev, percentiles)
            )
        },
    }
    return summary, {
        "strict_first_window": first_vector,
        "strict_second_window": second_vector,
    }


def write_detector_csv(
    path: Path,
    detector: np.ndarray,
    geant4: dict[str, np.ndarray],
    matrix: dict[str, np.ndarray],
    list_vectors: dict[str, np.ndarray],
) -> None:
    names = [
        "detector_id_1based",
        "x_mm",
        "y_mm",
        "z_mm",
        *[f"g4_{name}" for name in geant4],
        *[f"matrix_{name}" for name in matrix],
        *[f"list_{name}" for name in list_vectors],
    ]
    columns = [
        np.arange(1, len(detector) + 1),
        detector[:, 0],
        detector[:, 1],
        detector[:, 2],
        *geant4.values(),
        *matrix.values(),
        *list_vectors.values(),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerows(np.column_stack(columns))


def make_figure(
    path: Path,
    geant4: dict[str, np.ndarray],
    matrix: dict[str, np.ndarray],
    detector: np.ndarray,
    list_vectors: dict[str, np.ndarray],
) -> None:
    layers = np.unique(detector[:, 1])
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    def totals(names: tuple[str, ...]) -> list[float]:
        return [float(geant4[name].sum()) for name in names]

    names = ("first_crystal", "other_crystal")
    axes[0, 0].bar(("First", "Other"), totals(names), color=("tab:blue", "tab:orange"))
    axes[0, 0].set_title("Accepted-bin location")
    axes[0, 0].set_ylabel("Counts per 440 primary")

    names = ("hit_1", "hit_2", "hit_3plus")
    axes[0, 1].bar(
        ("1 crystal", "2 crystals", "3+ crystals"),
        totals(names),
        color=("tab:green", "tab:purple", "tab:red"),
    )
    axes[0, 1].set_title("Event hit multiplicity")
    axes[0, 1].set_ylabel("Counts per 440 primary")

    names = ("first_compton_0", "first_compton_1", "first_compton_2plus")
    axes[0, 2].bar(
        ("0", "1", "2+"),
        totals(names),
        color=("tab:gray", "tab:cyan", "tab:pink"),
    )
    axes[0, 2].set_title("Primary Compton labels in first crystal")
    axes[0, 2].set_xlabel("Labeled Compton steps")
    axes[0, 2].set_ylabel("Counts per 440 primary")

    positions = np.arange(len(layers))
    width = 0.26

    hit_12 = geant4["hit_1"] + geant4["hit_2"]
    total_series = (
        (geant4["total"], "Geant4 all", "tab:blue"),
        (hit_12, "Geant4 hit 1+2", "tab:green"),
        (matrix["total"], "Matrix total", "tab:orange"),
    )
    for index, (values, label, color) in enumerate(total_series):
        layer_values = [float(values[detector[:, 1] == layer].sum()) for layer in layers]
        axes[1, 0].bar(
            positions + (index - 1) * width, layer_values, width, label=label, color=color
        )
    axes[1, 0].set_title("Total response by layer")
    axes[1, 0].set_ylabel("Counts per 440 primary")
    axes[1, 0].legend()

    local_series = (
        (geant4["first_compton_1"], "Geant4 first, label=1", "tab:blue"),
        (matrix["local_recoil"], "Matrix local recoil", "tab:orange"),
    )
    for index, (values, label, color) in enumerate(local_series):
        layer_values = [float(values[detector[:, 1] == layer].sum()) for layer in layers]
        axes[1, 1].bar(
            positions + (index - 0.5) * width, layer_values, width, label=label, color=color
        )
    axes[1, 1].set_title("Local first-Compton proxy")
    axes[1, 1].set_xlabel("Detector layer y (mm)")
    axes[1, 1].set_ylabel("Counts per 440 primary")
    axes[1, 1].legend()

    if "strict_second_window" in list_vectors:
        inter_reference = list_vectors["strict_second_window"]
        inter_label = "List strict second crystal"
    else:
        inter_reference = geant4["other_crystal"]
        inter_label = "Geant4 other crystal"
    inter_series = (
        (inter_reference, inter_label, "tab:blue"),
        (matrix["intercrystal"], "Matrix intercrystal", "tab:orange"),
    )
    for index, (values, label, color) in enumerate(inter_series):
        layer_values = [float(values[detector[:, 1] == layer].sum()) for layer in layers]
        axes[1, 2].bar(
            positions + (index - 0.5) * width, layer_values, width, label=label, color=color
        )
    axes[1, 2].set_title("Intercrystal first-order proxy")
    axes[1, 2].set_xlabel("Detector layer y (mm)")
    axes[1, 2].set_ylabel("Counts per 440 primary")
    axes[1, 2].legend()

    for axis in axes[1, :]:
        axis.set_xticks(positions, [f"{layer:g}" for layer in layers])
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(path: Path, summary: dict[str, object]) -> None:
    geant4 = summary["geant4"]
    comparisons = summary["comparisons"]
    total = geant4["total"]["total"]
    hit_3plus = geant4["hit_3plus"]["total"]
    deficit = total - summary["matrix"]["total"]["total"]
    lines = [
        "# 440-to-218 Topology Analysis",
        "",
        f"- 440 primary count: {summary['primary_count_440']}",
        f"- Geant4 total response: {total:.9e} per 440 primary",
        f"- Matrix total response: {summary['matrix']['total']['total']:.9e}",
        f"- Matrix / Geant4: {comparisons['matrix_total_vs_geant4_total']['matrix_to_geant4']:.6f}",
        f"- Geant4 3+ crystal response: {hit_3plus:.9e} ({hit_3plus / total:.2%} of total)",
        f"- Matrix deficit: {deficit:.9e}; 3+ crystal magnitude / deficit: {hit_3plus / deficit:.2%}",
        f"- Local recoil / first-crystal Compton-label-1: {comparisons['local_recoil_vs_first_compton_1']['matrix_to_geant4']:.6f}",
        "",
        "## Interpretation",
        "",
        "The magnitude of the 3+ crystal category explains most of the total matrix deficit,",
        "which strongly points to multi-crystal and higher-order histories missing from the",
        "first-order analytical matrix. This is a magnitude comparison, not an exclusive",
        "event-by-event closure, because the analytical local term can still represent a bin",
        "that belongs to a multi-crystal Geant4 event.",
        "",
        "The current EventAction topology labels are approximate. SteppingAction counts a",
        "primary Compton step and selects the first scintillator only inside a branch requiring",
        "GetTotalEnergyDeposit() > 0. A discrete Compton interaction can transfer energy to a",
        "tracked secondary while depositing little or no energy on the primary-gamma step.",
        "Therefore FirstCrystal, OtherCrystal, and the Compton subcategories must not yet be",
        "treated as exact physical process partitions. Hit1/Hit2/Hit3Plus and the detector-bin",
        "CntStat totals do not have this process-label limitation.",
        "",
        "List.csv stores 1-based crystal IDs and does not store the primary energy. This report",
        "infers a 440 primary for strict List comparisons when E1+E2 >= 329 keV.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    default_matrix_run = (
        project_root
        / "Auxiliary_Studies"
        / "GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main"
        / "runs"
        / "JSCC_440keV_to_218keVwin_SurfaceValidation"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=script_dir / "build")
    parser.add_argument("--matrix-run", type=Path, default=default_matrix_run)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-list",
        action="store_true",
        help="Skip the optional List.csv analysis to reduce memory use.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.build_dir / "topology_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    detector_raw = np.fromfile(args.matrix_run / "Params_Detector.dat", dtype="<f4")
    detector_count = int(detector_raw[0])
    detector_all = detector_raw[1:].reshape(detector_count, 12)
    active = detector_all[:, 11] == 1
    detector = detector_all[active]
    image = np.fromfile(args.matrix_run / "Params_Image.dat", dtype="<f4")
    image_shape = tuple(int(value) for value in image[:3])

    primary_count = read_scalar_count(args.build_dir / "PrimaryCount440.csv")
    if primary_count <= 0:
        raise ValueError("PrimaryCount440.csv must be positive.")
    geant4_counts = {
        name: read_count_row(args.build_dir / filename, len(detector))
        for name, filename in TOPOLOGY_FILES.items()
    }
    geant4 = {
        name: values.astype(np.float64) / primary_count
        for name, values in geant4_counts.items()
    }
    matrix = {
        name: center_column(args.matrix_run / filename, detector_count, image_shape)[
            active
        ]
        for name, filename in MATRIX_FILES.items()
    }

    closure = {
        "first_plus_other_max_abs_bin_count": int(
            np.abs(
                geant4_counts["total"]
                - geant4_counts["first_crystal"]
                - geant4_counts["other_crystal"]
            ).max()
        ),
        "hit_multiplicity_max_abs_bin_count": int(
            np.abs(
                geant4_counts["total"]
                - geant4_counts["hit_1"]
                - geant4_counts["hit_2"]
                - geant4_counts["hit_3plus"]
            ).max()
        ),
        "first_compton_max_abs_bin_count": int(
            np.abs(
                geant4_counts["first_crystal"]
                - geant4_counts["first_compton_0"]
                - geant4_counts["first_compton_1"]
                - geant4_counts["first_compton_2plus"]
            ).max()
        ),
    }

    list_summary: dict[str, object] | None = None
    list_vectors: dict[str, np.ndarray] = {}
    list_path = args.build_dir / "List.csv"
    if not args.skip_list and list_path.exists():
        list_summary, list_vectors = analyze_list(
            list_path, len(detector), primary_count
        )

    hit_12 = geant4["hit_1"] + geant4["hit_2"]
    comparisons = {
        "matrix_total_vs_geant4_total": comparison_summary(
            matrix["total"], geant4["total"], detector
        ),
        "matrix_total_vs_geant4_hit1_plus_hit2": comparison_summary(
            matrix["total"], hit_12, detector
        ),
        "local_recoil_vs_first_compton_1": comparison_summary(
            matrix["local_recoil"], geant4["first_compton_1"], detector
        ),
        "matrix_nonlocal_vs_other_crystal": comparison_summary(
            matrix["intercrystal"] + matrix["highz_to_crystal"],
            geant4["other_crystal"],
            detector,
        ),
    }
    if "strict_second_window" in list_vectors:
        comparisons["intercrystal_vs_list_strict_second_window"] = comparison_summary(
            matrix["intercrystal"], list_vectors["strict_second_window"], detector
        )

    total_response = float(geant4["total"].sum())
    fractions = {
        name: float(geant4[name].sum()) / total_response
        for name in (
            "first_crystal",
            "other_crystal",
            "hit_1",
            "hit_2",
            "hit_3plus",
        )
    }
    first_response = float(geant4["first_crystal"].sum())
    fractions.update(
        {
            f"{name}_of_first_crystal": float(geant4[name].sum()) / first_response
            for name in (
                "first_compton_0",
                "first_compton_1",
                "first_compton_2plus",
            )
        }
    )

    summary: dict[str, object] = {
        "primary_count_440": primary_count,
        "closure": closure,
        "fractions": fractions,
        "geant4": {
            name: grouped_summary(values, detector) for name, values in geant4.items()
        },
        "matrix": {
            name: grouped_summary(values, detector) for name, values in matrix.items()
        },
        "comparisons": comparisons,
        "list": list_summary,
        "classification_caveat": (
            "Current SteppingAction labels first-crystal and primary Compton only "
            "when the primary step has GetTotalEnergyDeposit()>0; these labels are "
            "not exact process-history partitions."
        ),
    }
    summary["status"] = "PASS" if max(closure.values()) == 0 else "FAIL"

    (output_dir / "topology_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_detector_csv(
        output_dir / "topology_detector_comparison.csv",
        detector,
        geant4,
        matrix,
        list_vectors,
    )
    make_figure(
        output_dir / "topology_component_comparison.png",
        geant4,
        matrix,
        detector,
        list_vectors,
    )
    write_report(output_dir / "topology_analysis_report.md", summary)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
