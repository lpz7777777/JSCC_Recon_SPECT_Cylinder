#!/usr/bin/env python3
"""Compare merged full-FOV radial Geant4 responses with calibrated Factors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESPONSES = (
    ("A218", 218, "CntStat218_From218.csv", "218keV_RotateNum20_CenterPoint"),
    ("A440", 440, "CntStat440_From440.csv", "440keV_RotateNum20_CenterPoint"),
    ("C440to218", 440, "CntStat218_From440.csv", "440keV_to218win_RotateNum20_CenterPoint"),
)
LAYERS_MM = (30, 60, 90, 120)


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    repo = script.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=repo / "Geant4Sim" / "run" / "merged_RadialPointResponse",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo / "Geant4Sim" / "Macro" / "RadialPointResponse_JSCC"
        / "radial_point_manifest.csv",
    )
    parser.add_argument("--factors-dir", type=Path, default=repo / "Factors")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_counts(path: Path, rows: int, columns: int) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if columns == 1:
        values = values.reshape(-1, 1)
    if values.shape != (rows, columns):
        raise ValueError(f"{path} has shape {values.shape}; expected {(rows, columns)}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{path} contains invalid counts")
    return values


def point_index(coordinates: np.ndarray, x_mm: float, y_mm: float) -> int:
    distance = np.hypot(coordinates[:, 0] - x_mm, coordinates[:, 1] - y_mm)
    index = int(np.argmin(distance))
    if distance[index] > 1e-4:
        raise ValueError(f"No polar point matches ({x_mm}, {y_mm}) mm")
    return index


def poisson_shape_metrics(observed: np.ndarray, matrix_probability: np.ndarray) -> dict:
    expected = matrix_probability.copy()
    expected *= observed.sum() / expected.sum()
    mask = expected > 1e-12
    y = observed[mask]
    mu = expected[mask]
    positive = y > 0
    deviance_terms = mu.copy()
    deviance_terms[positive] = (
        y[positive] * np.log(y[positive] / mu[positive])
        - (y[positive] - mu[positive])
    )
    deviance_per_bin = 2.0 * deviance_terms.sum() / mask.sum()
    relative_l2 = np.linalg.norm(observed - expected) / np.linalg.norm(expected)
    poisson_l2 = np.sqrt(expected.sum()) / np.linalg.norm(expected)
    tv = 0.5 * np.abs(observed / observed.sum() - expected / expected.sum()).sum()
    poisson_tv = 0.5 * np.sqrt(2.0 * expected / np.pi).sum() / expected.sum()
    return {
        "deviance_per_positive_matrix_bin": deviance_per_bin,
        "relative_l2": relative_l2,
        "relative_l2_over_poisson": relative_l2 / poisson_l2,
        "total_variation": tv,
        "total_variation_over_poisson": tv / poisson_tv,
    }


def main() -> None:
    args = parse_args()
    merged_dir = args.merged_dir.resolve()
    manifest = pd.read_csv(args.manifest.resolve())
    output_dir = (args.output_dir or merged_dir / "radial_analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(manifest) != 202:
        raise ValueError(f"Expected 202 manifest rows, found {len(manifest)}")

    factor_ref = args.factors_dir / "218keV_RotateNum20_CenterPoint"
    coordinates = np.loadtxt(factor_ref / "coor_polar.csv", delimiter=",")
    detectors = pd.read_csv(factor_ref / "Detector.csv")
    if len(coordinates) != 1281 or len(detectors) != 10496:
        raise ValueError("Unexpected center-point Factor geometry")
    detector_layers = detectors["y"].to_numpy()

    primary_218 = read_counts(merged_dir / "PrimaryCount218.csv", 202, 1)[:, 0]
    primary_440 = read_counts(merged_dir / "PrimaryCount440.csv", 202, 1)[:, 0]
    primary_other = read_counts(merged_dir / "PrimaryCountOther.csv", 202, 1)[:, 0]
    if np.any(primary_other != 0) or np.any(primary_218 + primary_440 != 1e8):
        raise ValueError("Primary-count validation failed")

    records: list[dict] = []
    points_per_layer = len(coordinates)
    z_minus_index = 9
    z_plus_index = 10

    for response, energy, count_file, factor_name in RESPONSES:
        counts = read_counts(merged_dir / count_file, 202, len(detectors))
        primary = primary_218 if energy == 218 else primary_440
        matrix = np.memmap(
            args.factors_dir / factor_name / "SysMat_polar",
            dtype=np.float32,
            mode="r",
            shape=(points_per_layer * 20, len(detectors)),
        )
        for row_index, row in manifest[manifest["energy_keV"] == energy].iterrows():
            polar_index = point_index(
                coordinates, float(row.factor_x_mm), float(row.factor_y_mm)
            )
            matrix_probability = 0.5 * (
                matrix[z_minus_index * points_per_layer + polar_index].astype(np.float64)
                + matrix[z_plus_index * points_per_layer + polar_index].astype(np.float64)
            )
            observed = counts[row_index]
            geant_probability = observed / primary[row_index]
            record = {
                "manifest_row": int(row_index + 1),
                "label": row.label,
                "response": response,
                "energy_keV": energy,
                "radius_mm": float(row.radius_mm),
                "angle_deg": float(row.angle_deg),
                "primary_count": float(primary[row_index]),
                "accepted_detector_counts": float(observed.sum()),
                "geant4_probability": float(geant_probability.sum()),
                "matrix_probability": float(matrix_probability.sum()),
                "geant4_over_matrix": float(
                    geant_probability.sum() / matrix_probability.sum()
                ),
            }
            for layer in LAYERS_MM:
                layer_mask = detector_layers == layer
                record[f"geant4_over_matrix_layer_{layer}"] = float(
                    geant_probability[layer_mask].sum()
                    / matrix_probability[layer_mask].sum()
                )
            record.update(poisson_shape_metrics(observed, primary[row_index] * matrix_probability))
            records.append(record)

    details = pd.DataFrame.from_records(records)
    details.to_csv(output_dir / "radial_response_details.csv", index=False)

    numeric_columns = [
        "geant4_over_matrix",
        *(f"geant4_over_matrix_layer_{layer}" for layer in LAYERS_MM),
        "deviance_per_positive_matrix_bin",
        "relative_l2_over_poisson",
        "total_variation_over_poisson",
    ]
    radial = (
        details.groupby(["response", "radius_mm"])[numeric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    radial.columns = [
        "_".join(str(part) for part in column if part) if isinstance(column, tuple) else column
        for column in radial.columns
    ]
    radial.to_csv(output_dir / "radial_response_by_radius.csv", index=False)

    summary = {"responses": {}, "interpretation": {}}
    for response, group in details.groupby("response"):
        scale = group["geant4_over_matrix"]
        noise_ratio = group["relative_l2_over_poisson"]
        radial_group = group.groupby("radius_mm")
        radial_scale = radial_group["geant4_over_matrix"].mean()
        radial_noise = radial_group["relative_l2_over_poisson"].mean()
        radial_slope, _ = np.polyfit(
            radial_scale.index.to_numpy(dtype=float), radial_scale.to_numpy(), 1
        )
        summary["responses"][response] = {
            "geant4_over_matrix_mean": float(scale.mean()),
            "geant4_over_matrix_min": float(scale.min()),
            "geant4_over_matrix_max": float(scale.max()),
            "geant4_over_matrix_center": float(radial_scale.loc[0.0]),
            "geant4_over_matrix_edge_mean": float(radial_scale.loc[150.0]),
            "radial_slope_per_100mm": float(radial_slope * 100.0),
            "radial_correlation": float(
                np.corrcoef(radial_scale.index.to_numpy(dtype=float), radial_scale)[0, 1]
            ),
            "relative_l2_over_poisson_median": float(noise_ratio.median()),
            "relative_l2_over_poisson_radial_max": float(radial_noise.max()),
            "deviance_per_bin_median": float(
                group["deviance_per_positive_matrix_bin"].median()
            ),
            "inferred_effective_independent_worker_count": float(
                100.0 / noise_ratio.median() ** 2
            ),
        }
    noise_medians = {
        response: values["relative_l2_over_poisson_median"]
        for response, values in summary["responses"].items()
    }
    maximum_noise_ratio = max(noise_medians.values())
    maximum_radial_noise_ratio = max(
        values["relative_l2_over_poisson_radial_max"]
        for values in summary["responses"].values()
    )
    if maximum_radial_noise_ratio <= 1.15:
        detector_bin_interpretation = (
            "Detector-bin residuals are consistent with independent-Poisson "
            f"statistics: the largest median L2/Poisson ratio is "
            f"{maximum_noise_ratio:.3f}."
        )
    else:
        detector_bin_interpretation = (
            "The global seed-correlation artifact is absent, with a largest median "
            f"L2/Poisson ratio of {maximum_noise_ratio:.3f}, but systematic "
            "detector-bin shape mismatch remains at some radii: the largest "
            f"radial-mean ratio is {maximum_radial_noise_ratio:.3f}."
        )
    radial_trends = {
        response: values["radial_slope_per_100mm"]
        for response, values in summary["responses"].items()
        if abs(values["radial_slope_per_100mm"]) >= 0.002
    }
    if radial_trends:
        radial_interpretation = (
            "Material radial total-response trends remain for "
            + ", ".join(
                f"{response} ({slope:+.4f} per 100 mm)"
                for response, slope in radial_trends.items()
            )
            + "."
        )
    else:
        radial_interpretation = (
            "No material monotonic radial total-efficiency mismatch was found."
        )
    summary["interpretation"] = {
        "radial_scalar_mismatch": radial_interpretation,
        "detector_bin_shape": detector_bin_interpretation,
    }
    (output_dir / "radial_response_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    colors = {"A218": "#237a57", "A440": "#2667a5", "C440to218": "#b54a35"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for axis, (response, _, _, _) in zip(axes, RESPONSES):
        group = details[details.response == response]
        for angle, angle_group in group.groupby("angle_deg"):
            if angle == 0 and len(angle_group) == 26:
                label = "0 deg"
            else:
                label = f"{angle:g} deg"
            axis.plot(
                angle_group.radius_mm,
                angle_group.geant4_over_matrix,
                marker="o",
                markersize=2.8,
                linewidth=0.8,
                alpha=0.55,
                label=label,
            )
        means = group.groupby("radius_mm").geant4_over_matrix.mean()
        axis.plot(means.index, means.values, color=colors[response], linewidth=2.2, label="mean")
        axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
        axis.set_title(response)
        axis.set_xlabel("Radius (mm)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Geant4 / calibrated Factor total response")
    axes[-1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "radial_total_response.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for axis, (response, _, _, _) in zip(axes, RESPONSES):
        group = details[details.response == response].groupby("radius_mm")
        for layer in LAYERS_MM:
            column = f"geant4_over_matrix_layer_{layer}"
            axis.plot(group[column].mean(), marker="o", markersize=3, label=f"y={layer} mm")
        axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
        axis.set_ylabel(f"{response}\nGeant4 / Factor")
        axis.grid(alpha=0.2)
        axis.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("Radius (mm)")
    fig.tight_layout()
    fig.savefig(output_dir / "radial_layer_response.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for response, _, _, _ in RESPONSES:
        group = details[details.response == response].groupby("radius_mm")
        axis.plot(
            group["relative_l2_over_poisson"].mean(),
            marker="o",
            markersize=3,
            label=response,
            color=colors[response],
        )
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--", label="independent Poisson")
    axis.set_xlabel("Radius (mm)")
    axis.set_ylabel("Detector-bin relative L2 / Poisson expectation")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "radial_shape_overdispersion.png", dpi=220)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
