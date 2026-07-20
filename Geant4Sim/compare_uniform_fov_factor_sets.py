#!/usr/bin/env python3
"""Compare two JSCC Factor sets against the same uniform-FOV Geant4 data."""

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
RESPONSE_FILES = {
    "A218": (218, "CntStat_218.csv", "218keV_RotateNum20"),
    "A440": (440, "CntStat_440.csv", "440keV_RotateNum20"),
    "C440to218": (440, "CntStat_218.csv", "440keV_to218win_RotateNum20"),
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-root",
        type=Path,
        default=repo / "Geant4Sim" / "run" / "merged_UniformFovCntStat",
    )
    parser.add_argument("--factors-dir", type=Path, default=repo / "Factors")
    parser.add_argument("--baseline-suffix", default="CenterPoint")
    parser.add_argument("--candidate-suffix", default="CenterPoint_PEv4")
    parser.add_argument("--baseline-label", default="PE v3")
    parser.add_argument("--candidate-label", default="PE v4")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "Results" / "Analysis" / "UniformFov_PEv3_vs_PEv4",
    )
    parser.add_argument("--chunk-rows", type=int, default=128)
    return parser.parse_args()


def factor_name(prefix: str, suffix: str) -> str:
    suffix = suffix.strip("_")
    return f"{prefix}_{suffix}" if suffix else prefix


def read_vector(path: Path, expected_size: int) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64).reshape(-1)
    if values.shape != (expected_size,):
        raise ValueError(f"{path} has shape {values.shape}; expected {(expected_size,)}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{path} contains invalid values")
    return values


def mean_factor_response(path: Path, chunk_rows: int) -> np.ndarray:
    expected_bytes = POINT_COUNT * DETECTOR_COUNT * np.dtype(np.float32).itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Unexpected matrix size: {path}")
    matrix = np.memmap(
        path, dtype=np.float32, mode="r", shape=(POINT_COUNT, DETECTOR_COUNT)
    )
    total = np.zeros(DETECTOR_COUNT, dtype=np.float64)
    for start in range(0, POINT_COUNT, chunk_rows):
        stop = min(POINT_COUNT, start + chunk_rows)
        total += matrix[start:stop].sum(axis=0, dtype=np.float64)
    return total / POINT_COUNT


def response_metrics(observed: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    if observed.sum() <= 0 or expected.sum() <= 0:
        raise ValueError("Observed and expected totals must be positive")
    total_ratio = observed.sum() / expected.sum()
    scaled = expected * total_ratio
    residual = observed - scaled
    relative_l2 = np.linalg.norm(residual) / np.linalg.norm(scaled)
    poisson_l2 = np.sqrt(scaled.sum()) / np.linalg.norm(scaled)
    observed_shape = observed / observed.sum()
    expected_shape = expected / expected.sum()
    return {
        "observed_total": float(observed.sum()),
        "expected_total": float(expected.sum()),
        "geant4_over_matrix_total": float(total_ratio),
        "absolute_total_relative_error": float(abs(total_ratio - 1.0)),
        "relative_l2_after_total_scaling": float(relative_l2),
        "relative_l2_over_poisson": float(relative_l2 / poisson_l2),
        "total_variation": float(0.5 * np.abs(observed_shape - expected_shape).sum()),
        "shape_correlation": float(np.corrcoef(observed_shape, expected_shape)[0, 1]),
    }


def plot_detector_comparison(
    table: pd.DataFrame,
    response: str,
    baseline_label: str,
    candidate_label: str,
    output_dir: Path,
) -> None:
    columns = (
        ("baseline_shape_ratio", baseline_label),
        ("candidate_shape_ratio", candidate_label),
        ("shape_ratio_change", f"{candidate_label} - {baseline_label}"),
    )
    fig, axes = plt.subplots(
        len(LAYERS_MM), len(columns), figsize=(13.5, 14.5), constrained_layout=True
    )
    ratio_norm = TwoSlopeNorm(vmin=0.95, vcenter=1.0, vmax=1.05)
    change_norm = TwoSlopeNorm(vmin=-0.03, vcenter=0.0, vmax=0.03)
    images = [None, None, None]
    for row, layer in enumerate(LAYERS_MM):
        group = table[table.detector_y_mm == layer]
        for column, (field, title) in enumerate(columns):
            norm = change_norm if field == "shape_ratio_change" else ratio_norm
            cmap = "PuOr" if field == "shape_ratio_change" else "coolwarm"
            images[column] = axes[row, column].scatter(
                group.detector_x_mm,
                group.detector_z_mm,
                c=group[field],
                s=7 if layer == 120 else 16,
                marker="s",
                linewidths=0,
                cmap=cmap,
                norm=norm,
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].grid(alpha=0.12)
            if row == 0:
                axes[row, column].set_title(title)
            if column == 0:
                axes[row, column].set_ylabel(f"y={layer} mm\nz (mm)")
            if row == len(LAYERS_MM) - 1:
                axes[row, column].set_xlabel("x (mm)")
    for column, image in enumerate(images):
        label = "Geant4 / matrix after total normalization"
        if column == 2:
            label = "Change in normalized Geant4 / matrix ratio"
        fig.colorbar(image, ax=axes[:, column], shrink=0.72, label=label)
    fig.suptitle(f"Uniform-FOV detector response: {response}", fontsize=15)
    fig.savefig(output_dir / f"detector_comparison_{response}.png", dpi=220)
    plt.close(fig)


def plot_metric_summary(
    metrics: pd.DataFrame,
    baseline_label: str,
    candidate_label: str,
    output_dir: Path,
) -> None:
    fields = (
        ("absolute_total_relative_error", "Absolute total-efficiency error"),
        ("relative_l2_after_total_scaling", "Shape relative L2"),
        ("total_variation", "Total variation"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    responses = list(RESPONSE_FILES)
    x = np.arange(len(responses))
    width = 0.36
    colors = ("#4c78a8", "#e45756")
    for axis, (field, title) in zip(axes, fields):
        baseline = [
            metrics.loc[(response, "baseline"), field] for response in responses
        ]
        candidate = [
            metrics.loc[(response, "candidate"), field] for response in responses
        ]
        axis.bar(x - width / 2, baseline, width, label=baseline_label, color=colors[0])
        axis.bar(x + width / 2, candidate, width, label=candidate_label, color=colors[1])
        axis.set_xticks(x, responses)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.savefig(output_dir / "metric_summary.png", dpi=220)
    plt.close(fig)


def plot_layer_ratios(
    layer_table: pd.DataFrame,
    baseline_label: str,
    candidate_label: str,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for axis, response in zip(axes, RESPONSE_FILES):
        group = layer_table[layer_table.response == response]
        baseline = group[group.factor_set == "baseline"]
        candidate = group[group.factor_set == "candidate"]
        axis.plot(
            baseline.layer_y_mm,
            baseline.geant4_over_matrix,
            "o-",
            color="#4c78a8",
            label=baseline_label,
        )
        axis.plot(
            candidate.layer_y_mm,
            candidate.geant4_over_matrix,
            "s--",
            color="#e45756",
            label=candidate_label,
        )
        axis.axhline(1.0, color="#333333", linewidth=1)
        axis.set_xticks(LAYERS_MM)
        axis.set_title(response)
        axis.set_xlabel("Detector layer y (mm)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Geant4 / matrix")
    axes[0].legend(frameon=False)
    fig.savefig(output_dir / "layer_efficiency_ratios.png", dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.chunk_rows < 1:
        raise ValueError("--chunk-rows must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    factors_dir = args.factors_dir.resolve()
    merged_root = args.merged_root.resolve()

    detector_path = factors_dir / factor_name(
        RESPONSE_FILES["A218"][2], args.baseline_suffix
    ) / "Detector.csv"
    detector = pd.read_csv(detector_path)
    if len(detector) != DETECTOR_COUNT:
        raise ValueError(f"Unexpected detector count in {detector_path}")

    metric_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "baseline_suffix": args.baseline_suffix,
        "candidate_suffix": args.candidate_suffix,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "responses": {},
    }
    for response, (energy, count_file, prefix) in RESPONSE_FILES.items():
        observed = read_vector(merged_root / f"{energy}keV" / count_file, DETECTOR_COUNT)
        primary = read_vector(merged_root / f"{energy}keV" / "PrimaryCount.csv", 1)[0]
        factor_paths = {
            "baseline": factors_dir / factor_name(prefix, args.baseline_suffix),
            "candidate": factors_dir / factor_name(prefix, args.candidate_suffix),
        }
        expected_sets: dict[str, np.ndarray] = {}
        response_summary: dict[str, object] = {}
        for set_name, factor_dir in factor_paths.items():
            matrix_probability = mean_factor_response(
                factor_dir / "SysMat_polar", args.chunk_rows
            )
            expected = primary * matrix_probability
            expected_sets[set_name] = expected
            metrics = response_metrics(observed, expected)
            metric_rows.append(
                {"response": response, "factor_set": set_name, **metrics}
            )
            response_summary[set_name] = metrics
            for layer in LAYERS_MM:
                mask = detector.y.to_numpy() == layer
                layer_rows.append(
                    {
                        "response": response,
                        "factor_set": set_name,
                        "layer_y_mm": layer,
                        "detector_bins": int(mask.sum()),
                        "observed_count": float(observed[mask].sum()),
                        "matrix_expected_count": float(expected[mask].sum()),
                        "geant4_over_matrix": float(
                            observed[mask].sum() / expected[mask].sum()
                        ),
                    }
                )

        improvements = {}
        for field in (
            "absolute_total_relative_error",
            "relative_l2_after_total_scaling",
            "total_variation",
        ):
            baseline_value = response_summary["baseline"][field]
            candidate_value = response_summary["candidate"][field]
            improvements[f"{field}_relative_reduction"] = (
                (baseline_value - candidate_value) / baseline_value
                if baseline_value > 0
                else 0.0
            )
        response_summary["improvement"] = improvements
        summary["responses"][response] = response_summary

        baseline_scaled = expected_sets["baseline"] * (
            observed.sum() / expected_sets["baseline"].sum()
        )
        candidate_scaled = expected_sets["candidate"] * (
            observed.sum() / expected_sets["candidate"].sum()
        )
        table = pd.DataFrame(
            {
                "detector_index": detector["index"].to_numpy(),
                "detector_x_mm": detector.x.to_numpy(),
                "detector_y_mm": detector.y.to_numpy(),
                "detector_z_mm": detector.z.to_numpy(),
                "observed_count": observed,
                "baseline_expected_count": expected_sets["baseline"],
                "candidate_expected_count": expected_sets["candidate"],
                "baseline_shape_ratio": observed / baseline_scaled,
                "candidate_shape_ratio": observed / candidate_scaled,
            }
        )
        table["shape_ratio_change"] = (
            table.candidate_shape_ratio - table.baseline_shape_ratio
        )
        table.to_csv(output_dir / f"detector_comparison_{response}.csv", index=False)
        plot_detector_comparison(
            table, response, args.baseline_label, args.candidate_label, output_dir
        )

    metrics = pd.DataFrame(metric_rows).set_index(["response", "factor_set"])
    layers = pd.DataFrame(layer_rows)
    metrics.reset_index().to_csv(output_dir / "response_metrics.csv", index=False)
    layers.to_csv(output_dir / "layer_metrics.csv", index=False)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_metric_summary(metrics, args.baseline_label, args.candidate_label, output_dir)
    plot_layer_ratios(layers, args.baseline_label, args.candidate_label, output_dir)

    report = [
        f"# Uniform-FOV {args.baseline_label} vs {args.candidate_label}",
        "",
        f"Baseline: `{args.baseline_suffix}` ({args.baseline_label})",
        f"Candidate: `{args.candidate_suffix}` ({args.candidate_label})",
        "",
        "| Response | Set | G4/matrix total | Absolute total error | Shape L2 | TV |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for response in RESPONSE_FILES:
        for set_name, label in (
            ("baseline", args.baseline_label),
            ("candidate", args.candidate_label),
        ):
            row = metrics.loc[(response, set_name)]
            report.append(
                f"| {response} | {label} | {row.geant4_over_matrix_total:.6f} | "
                f"{row.absolute_total_relative_error:.6f} | "
                f"{row.relative_l2_after_total_scaling:.6f} | "
                f"{row.total_variation:.6f} |"
            )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved comparison to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
