#!/usr/bin/env python3
"""Compare baseline, layer-corrected, and detector-corrected reconstructions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay


TASKS = (
    ("440 keV", "Image_S_440keV"),
    ("218 keV + contamination", "Image_S_218keV_Contaminated"),
    ("218 keV corrected", "Image_S_218keV_CrossTalkCorrected"),
    ("440 + 218 corrected", "Image_S_(440_218)keV_CrossTalkCorrected"),
)
VARIANTS = ("Baseline", "FOV layer", "FOV detector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--layer", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_image(path: Path, point_count: int) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % point_count != 0:
        raise ValueError(f"Unexpected image size: {path}")
    return values.reshape(-1, point_count)


def cartesian_mip(image: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    centers = np.arange(-148.5, 150.0, 3.0)
    grid_x, grid_y = np.meshgrid(centers, centers)
    triangulation = Delaunay(coordinates[:, :2])
    volume = np.empty((image.shape[0], len(centers), len(centers)), dtype=np.float32)
    for z_index, layer in enumerate(image):
        interpolator = LinearNDInterpolator(triangulation, layer, fill_value=0.0)
        volume[z_index] = interpolator(grid_x, grid_y)
    return volume.max(axis=0)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates = np.loadtxt(args.coordinates, delimiter=",", dtype=np.float64)
    point_count = len(coordinates)
    radii = np.hypot(coordinates[:, 0], coordinates[:, 1])
    core = radii <= 6.0 + 1e-6
    reference = (radii >= 12.0 - 1e-6) & (radii <= 30.0 + 1e-6)
    run_dirs = [args.baseline.resolve(), args.layer.resolve(), args.detector.resolve()]

    images: dict[str, list[np.ndarray]] = {}
    mips: dict[str, list[np.ndarray]] = {}
    metrics: list[dict] = []
    for task_label, filename in TASKS:
        task_images = [read_image(run / filename, point_count) for run in run_dirs]
        task_mips = [cartesian_mip(image, coordinates) for image in task_images]
        images[filename] = task_images
        mips[filename] = task_mips
        baseline = task_images[0].astype(np.float64)
        for variant, image in zip(VARIANTS, task_images):
            values = image.astype(np.float64)
            metrics.append(
                {
                    "task": task_label,
                    "variant": variant,
                    "sum": float(values.sum()),
                    "relative_sum_to_baseline": float(values.sum() / baseline.sum()),
                    "relative_l2_to_baseline": float(
                        np.linalg.norm(values - baseline) / np.linalg.norm(baseline)
                    ),
                    "core_mean": float(values[:, core].mean()),
                    "reference_annulus_mean": float(values[:, reference].mean()),
                    "core_to_reference_ratio": float(
                        values[:, core].mean() / values[:, reference].mean()
                    ),
                }
            )

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(output_dir / "reconstruction_comparison_metrics.csv", index=False)

    fig, axes = plt.subplots(len(TASKS), len(VARIANTS), figsize=(14, 17), constrained_layout=True)
    for row, (task_label, filename) in enumerate(TASKS):
        row_mips = mips[filename]
        display_max = np.percentile(np.concatenate([m.ravel() for m in row_mips]), 99.5)
        for column, (variant, mip) in enumerate(zip(VARIANTS, row_mips)):
            axes[row, column].imshow(
                mip,
                cmap="gray_r",
                origin="lower",
                extent=(-150, 150, -150, 150),
                vmin=0,
                vmax=display_max,
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].set_title(variant if row == 0 else "")
            if column == 0:
                axes[row, column].set_ylabel(task_label)
            if row == len(TASKS) - 1:
                axes[row, column].set_xlabel("x (mm)")
    fig.suptitle("1e10 Geant4 CntStat | 2000-iteration MLEM | common row scales")
    fig.savefig(output_dir / "final_mip_factor_comparison.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(len(TASKS), 2, figsize=(10, 17), constrained_layout=True)
    for row, (task_label, filename) in enumerate(TASKS):
        baseline = mips[filename][0]
        scale = np.percentile(baseline, 99.5)
        differences = [
            (mips[filename][1] - baseline) / scale,
            (mips[filename][2] - baseline) / scale,
        ]
        limit = max(0.001, np.percentile(np.abs(np.concatenate([d.ravel() for d in differences])), 99))
        for column, (label, difference) in enumerate(
            zip(("FOV layer - baseline", "FOV detector - baseline"), differences)
        ):
            image = axes[row, column].imshow(
                difference * 100.0,
                cmap="coolwarm",
                origin="lower",
                extent=(-150, 150, -150, 150),
                vmin=-limit * 100.0,
                vmax=limit * 100.0,
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].set_title(label if row == 0 else "")
            if column == 0:
                axes[row, column].set_ylabel(task_label)
            fig.colorbar(image, ax=axes[row, column], label="Difference / baseline p99.5 (%)")
    fig.suptitle("Factor correction effects on final transverse MIP")
    fig.savefig(output_dir / "final_mip_factor_differences.png", dpi=220)
    plt.close(fig)

    diagnostics = {}
    for variant, run in zip(VARIANTS, run_dirs):
        with (run / "run_manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        diagnostics[variant] = manifest.get("cross_talk_diagnostics", {})
    (output_dir / "cross_talk_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(metrics_frame.to_string(index=False))
    print(f"Saved comparison to {output_dir}")


if __name__ == "__main__":
    main()
