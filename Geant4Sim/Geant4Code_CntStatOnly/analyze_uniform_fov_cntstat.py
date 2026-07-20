#!/usr/bin/env python3
"""Compare uniform-FOV Geant4 CntStat with full Factor forward projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


POINT_COUNT = 25620
DETECTOR_COUNT = 10496
LAYERS_MM = (30, 60, 90, 120)
RESPONSES = (
    ("A218", 218, "CntStat_218.csv", "218keV_RotateNum20"),
    ("A440", 440, "CntStat_440.csv", "440keV_RotateNum20"),
    ("C440to218", 440, "CntStat_218.csv", "440keV_to218win_RotateNum20"),
)


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    repo = script.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merged-root",
        type=Path,
        default=repo / "Geant4Sim" / "run" / "merged_UniformFovCntStat",
        help="Directory containing 218keV and 440keV merged subdirectories.",
    )
    parser.add_argument("--factors-dir", type=Path, default=repo / "Factors")
    parser.add_argument("--factor-suffix", default="CenterPoint")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--minimum-expected-count",
        type=float,
        default=100.0,
        help="Minimum matrix-predicted count for a reported row factor.",
    )
    return parser.parse_args()


def factor_directory_name(prefix: str, suffix: str) -> str:
    suffix = suffix.strip("_")
    return f"{prefix}_{suffix}" if suffix else prefix


def read_count_row(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64).reshape(-1)
    if values.shape != (DETECTOR_COUNT,):
        raise ValueError(f"{path} has shape {values.shape}; expected {(DETECTOR_COUNT,)}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{path} contains invalid counts")
    return values


def read_primary(path: Path) -> float:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64).reshape(-1)
    if values.shape != (1,) or not np.isfinite(values[0]) or values[0] <= 0:
        raise ValueError(f"{path} must contain one positive primary count")
    return float(values[0])


def mean_factor_response(path: Path) -> np.ndarray:
    matrix = np.memmap(
        path,
        dtype=np.float32,
        mode="r",
        shape=(POINT_COUNT, DETECTOR_COUNT),
    )
    total = np.zeros(DETECTOR_COUNT, dtype=np.float64)
    for start in range(0, POINT_COUNT, 128):
        total += matrix[start : start + 128].sum(axis=0, dtype=np.float64)
    return total / POINT_COUNT


def shape_metrics(observed: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    mask = expected > 1e-12
    observed = observed[mask]
    expected = expected[mask]
    scaled = expected * (observed.sum() / expected.sum())
    relative_l2 = np.linalg.norm(observed - scaled) / np.linalg.norm(scaled)
    poisson_l2 = np.sqrt(scaled.sum()) / np.linalg.norm(scaled)
    total_variation = 0.5 * np.abs(
        observed / observed.sum() - expected / expected.sum()
    ).sum()
    return {
        "relative_l2": float(relative_l2),
        "relative_l2_over_poisson": float(relative_l2 / poisson_l2),
        "total_variation": float(total_variation),
    }


def save_detector_map(
    table: pd.DataFrame, response: str, output_dir: Path
) -> None:
    values = table.loc[table.valid_for_correction, "total_preserving_factor"]
    lower, upper = np.nanpercentile(values, [1, 99])
    lower = min(float(lower), 0.999)
    upper = max(float(upper), 1.001)
    norm = TwoSlopeNorm(vmin=lower, vcenter=1.0, vmax=upper)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True)
    image = None
    for axis, layer in zip(axes, LAYERS_MM):
        group = table[table.detector_y_mm == layer]
        image = axis.scatter(
            group.detector_x_mm,
            group.detector_z_mm,
            c=group.total_preserving_factor,
            s=8 if layer == 120 else 18,
            marker="s",
            linewidths=0,
            cmap="coolwarm",
            norm=norm,
        )
        axis.set_title(f"y={layer} mm")
        axis.set_aspect("equal")
        axis.set_xlabel("x (mm)")
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("z (mm)")
    fig.colorbar(image, ax=axes, label="Total-preserving detector-row factor")
    fig.suptitle(response)
    fig.savefig(output_dir / f"detector_row_correction_{response}.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    merged_root = args.merged_root.resolve()
    output_dir = (args.output_dir or merged_root / "uniform_fov_analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector_path = (
        args.factors_dir
        / factor_directory_name(RESPONSES[0][3], args.factor_suffix)
        / "Detector.csv"
    )
    detectors = pd.read_csv(detector_path)
    if len(detectors) != DETECTOR_COUNT:
        raise ValueError(f"Unexpected detector count in {detector_path}")

    summary: dict[str, dict] = {"responses": {}}
    all_tables: list[pd.DataFrame] = []
    for response, source_energy, count_name, factor_prefix in RESPONSES:
        factor_name = factor_directory_name(factor_prefix, args.factor_suffix)
        energy_dir = merged_root / f"{source_energy}keV"
        observed = read_count_row(energy_dir / count_name)
        primary = read_primary(energy_dir / "PrimaryCount.csv")
        matrix_probability = mean_factor_response(
            args.factors_dir / factor_name / "SysMat_polar"
        )
        expected = primary * matrix_probability
        valid = expected >= args.minimum_expected_count

        raw_factor = np.full(DETECTOR_COUNT, np.nan)
        total_preserving = np.full(DETECTOR_COUNT, np.nan)
        positive = matrix_probability > 0
        raw_factor[positive] = observed[positive] / expected[positive]
        total_preserving[positive] = (
            observed[positive] / observed.sum()
        ) / (matrix_probability[positive] / matrix_probability.sum())

        table = pd.DataFrame(
            {
                "response": response,
                "detector_index": detectors["index"].to_numpy(),
                "detector_x_mm": detectors["x"].to_numpy(),
                "detector_y_mm": detectors["y"].to_numpy(),
                "detector_z_mm": detectors["z"].to_numpy(),
                "observed_count": observed,
                "matrix_expected_count": expected,
                "raw_geant4_over_matrix": raw_factor,
                "total_preserving_factor": total_preserving,
                "valid_for_correction": valid,
            }
        )
        table.to_csv(output_dir / f"detector_row_correction_{response}.csv", index=False)
        all_tables.append(table)
        save_detector_map(table, response, output_dir)

        valid_factor = total_preserving[valid]
        response_summary = {
            "source_energy_keV": source_energy,
            "primary_count": primary,
            "observed_detector_counts": float(observed.sum()),
            "matrix_expected_detector_counts": float(expected.sum()),
            "total_geant4_over_matrix": float(observed.sum() / expected.sum()),
            "valid_detector_bins": int(valid.sum()),
            "total_preserving_factor_p01": float(np.percentile(valid_factor, 1)),
            "total_preserving_factor_median": float(np.median(valid_factor)),
            "total_preserving_factor_p99": float(np.percentile(valid_factor, 99)),
        }
        response_summary.update(shape_metrics(observed, expected))
        summary["responses"][response] = response_summary

    pd.concat(all_tables, ignore_index=True).to_csv(
        output_dir / "detector_row_corrections_all.csv", index=False
    )
    (output_dir / "uniform_fov_response_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
