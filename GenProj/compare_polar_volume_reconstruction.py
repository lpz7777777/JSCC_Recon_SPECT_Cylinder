#!/usr/bin/env python3
"""Compare integrated-cell and polar-volume density-basis reconstructions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay


RUN_NAME = "JSCC_Rotate20_E218_440_Count1e10_MLEM2000_OSEM1_CrossTalkCorrected"
TASKS = (
    ("440 keV", "Image_S_440keV"),
    ("218 keV corrected", "Image_S_218keV_CrossTalkCorrected"),
    ("440 + 218 corrected", "Image_S_(440_218)keV_CrossTalkCorrected"),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--old-run",
        type=Path,
        default=root / "Results/LocalReconstructionRuns/PEv4_UniformFOVLayer_Calibrated" / RUN_NAME,
    )
    parser.add_argument(
        "--new-run",
        type=Path,
        default=root / "Results/LocalReconstructionRuns/PEv4_UniformFOVLayer_PolarVolumeDensity_Calibrated" / RUN_NAME,
    )
    parser.add_argument(
        "--factor-dir",
        type=Path,
        default=root / "Factors/218keV_RotateNum20",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "Results/Analysis/PolarVolumeRecon_20260720/Comparison",
    )
    return parser.parse_args()


def polar_dir(run: Path) -> Path:
    candidate = run if (run / "run_manifest.json").is_file() else run / "Polar"
    if not (candidate / "run_manifest.json").is_file():
        raise FileNotFoundError(candidate / "run_manifest.json")
    return candidate


def load_image(directory: Path, name: str, size: int) -> np.ndarray:
    image = np.fromfile(directory / name, dtype="<f4").astype(np.float64)
    if image.size != size or not np.all(np.isfinite(image)) or np.any(image < 0.0):
        raise ValueError(f"Invalid image {directory / name}")
    return image


def rod_exclusion(coordinates: np.ndarray) -> np.ndarray:
    mask = np.ones(coordinates.shape[0], dtype=bool)
    for index, diameter in enumerate(np.arange(10.0, 31.0, 4.0)):
        theta = index * math.pi / 3.0
        cx = 60.0 * math.cos(theta)
        cy = 60.0 * math.sin(theta)
        distance = np.hypot(coordinates[:, 0] - cx, coordinates[:, 1] - cy)
        mask &= distance > 0.5 * diameter + 3.0
    return mask


def ring_profile(values: np.ndarray, radius: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radii = np.unique(radius)
    profile = np.full(radii.size, np.nan)
    for index, value in enumerate(radii):
        selected = mask & (radius == value)
        if np.any(selected):
            profile[index] = np.median(values[selected])
    return radii, profile


def normalize_profile(profile: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, float]:
    reference = np.isfinite(profile) & (radii >= 30.0) & (radii <= 108.0)
    scale = float(np.median(profile[reference]))
    return profile / scale, scale


def profile_metrics(profile: np.ndarray, radii: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(profile) & (radii <= 108.0)
    center = np.isfinite(profile) & (radii <= 18.0)
    middle = np.isfinite(profile) & (radii >= 30.0) & (radii <= 108.0)
    center_ratio = float(np.median(profile[center]) / np.median(profile[middle]))
    return {
        "center_r0_18_to_middle_ratio": center_ratio,
        "absolute_center_bias_from_unity": abs(center_ratio - 1.0),
        "ring_profile_cv_r0_108": float(np.std(profile[valid]) / np.mean(profile[valid])),
        "ring_profile_rmse_from_uniform_r0_108": float(np.sqrt(np.mean((profile[valid] - 1.0) ** 2))),
        "center_point_relative_value": float(profile[np.where(radii == 0.0)[0][0]]),
        "r6_relative_value": float(profile[np.where(radii == 6.0)[0][0]]),
        "r18_relative_value": float(profile[np.where(radii == 18.0)[0][0]]),
    }


def interpolation_plan(xy: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangulation = Delaunay(xy)
    simplex = triangulation.find_simplex(query)
    valid = simplex >= 0
    transform = triangulation.transform[simplex[valid]]
    delta = query[valid] - transform[:, 2]
    bary12 = np.einsum("nij,nj->ni", transform[:, :2], delta)
    weights = np.column_stack((bary12, 1.0 - np.sum(bary12, axis=1)))
    vertices = triangulation.simplices[simplex[valid]]
    return valid, vertices, weights


def make_mip(
    values: np.ndarray,
    coordinates: np.ndarray,
    z_values: np.ndarray,
    grid_axis: np.ndarray,
) -> np.ndarray:
    points_per_layer = coordinates.shape[0] // z_values.size
    xy = coordinates[:points_per_layer, :2]
    grid_x, grid_y = np.meshgrid(grid_axis, grid_axis, indexing="xy")
    query = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    valid, vertices, weights = interpolation_plan(xy, query)
    mip = np.zeros(query.shape[0], dtype=np.float64)
    selected_layers = np.where(np.abs(z_values) <= 13.5)[0]
    for layer in selected_layers:
        layer_values = values[layer * points_per_layer : (layer + 1) * points_per_layer]
        interpolated = np.zeros(query.shape[0], dtype=np.float64)
        interpolated[valid] = np.sum(layer_values[vertices] * weights, axis=1)
        mip = np.maximum(mip, interpolated)
    mip = mip.reshape(grid_x.shape)
    mip[np.hypot(grid_x, grid_y) > 150.0] = np.nan
    return mip


def main() -> None:
    args = parse_args()
    old_dir = polar_dir(args.old_run.resolve())
    new_dir = polar_dir(args.new_run.resolve())
    factor_dir = args.factor_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    coordinates = np.loadtxt(factor_dir / "coor_polar_full.csv", delimiter=",", dtype=np.float64)
    radius = np.round(np.hypot(coordinates[:, 0], coordinates[:, 1]), 8)
    z_value = np.round(coordinates[:, 2], 8)
    z_values = np.unique(z_value)
    size = coordinates.shape[0]
    background_mask = (
        (radius <= 108.0)
        & (np.abs(z_value) <= 13.5)
        & rod_exclusion(coordinates)
    )

    grid_axis = np.arange(-148.5, 150.0, 3.0)
    figure, axes = plt.subplots(3, 3, figsize=(13.5, 12.0), constrained_layout=True)
    summary: dict[str, object] = {
        "old_run": str(old_dir),
        "new_run": str(new_dir),
        "old_variable": "integrated activity per polar cell",
        "new_variable": "activity density per mm3",
        "normalization": "each image/profile divided by its own rod-excluded r=30..108 mm background median",
        "tasks": {},
    }
    csv_rows = []

    for row, (label, file_name) in enumerate(TASKS):
        old_image = load_image(old_dir, file_name, size)
        new_image = load_image(new_dir, file_name, size)
        radii, old_profile = ring_profile(old_image, radius, background_mask)
        _, new_profile = ring_profile(new_image, radius, background_mask)
        old_profile, old_scale = normalize_profile(old_profile, radii)
        new_profile, new_scale = normalize_profile(new_profile, radii)
        old_normalized = old_image / old_scale
        new_normalized = new_image / new_scale
        old_metrics = profile_metrics(old_profile, radii)
        new_metrics = profile_metrics(new_profile, radii)
        summary["tasks"][label] = {
            "old": old_metrics,
            "polar_volume": new_metrics,
            "change": {
                key: new_metrics[key] - old_metrics[key]
                for key in old_metrics
            },
        }

        old_mip = make_mip(old_normalized, coordinates, z_values, grid_axis)
        new_mip = make_mip(new_normalized, coordinates, z_values, grid_axis)
        finite_values = np.concatenate((old_mip[np.isfinite(old_mip)], new_mip[np.isfinite(new_mip)]))
        vmax = float(np.percentile(finite_values, 99.5))
        for column, (mip, title) in enumerate(
            ((old_mip, "Old A: integrated-cell image"), (new_mip, "A diag(DeltaV): density image"))
        ):
            axis = axes[row, column]
            image = axis.imshow(
                mip,
                origin="lower",
                extent=(grid_axis[0] - 1.5, grid_axis[-1] + 1.5, grid_axis[0] - 1.5, grid_axis[-1] + 1.5),
                cmap="gray_r",
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
            )
            axis.set_aspect("equal")
            axis.set(xlabel="x (mm)", ylabel="y (mm)")
            axis.set_title(f"{label}\n{title}")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)

        axis = axes[row, 2]
        axis.plot(radii, old_profile, "o-", linewidth=1.5, markersize=3.5, label="Old displayed x")
        axis.plot(radii, new_profile, "s-", linewidth=1.5, markersize=3.5, label="Polar-volume density")
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.axvspan(0.0, 18.0, color="#d9e8ef", alpha=0.5)
        axis.set(
            title=f"{label}\nRod-excluded background radial median",
            xlabel="Radius (mm)",
            ylabel="Relative background",
            xlim=(0.0, 110.0),
            ylim=(0.0, max(1.5, np.nanmax([old_profile[radii <= 108], new_profile[radii <= 108]]) * 1.08)),
        )
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)

        for radius_value, old_value, new_value in zip(radii, old_profile, new_profile):
            csv_rows.append(
                {
                    "task": label,
                    "radius_mm": radius_value,
                    "old_relative_background": old_value,
                    "polar_volume_relative_background": new_value,
                }
            )

    figure.suptitle("Geant4 1e10, 2000-iteration MLEM: polar-volume basis comparison", fontsize=15)
    figure.savefig(output_dir / "old_vs_polar_volume_reconstruction.png", dpi=220)
    plt.close(figure)

    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    with (output_dir / "radial_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
