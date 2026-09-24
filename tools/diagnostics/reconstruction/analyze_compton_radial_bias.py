#!/usr/bin/env python3
"""Quantify radial background bias in saved JSCC Compton reconstructions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HISTORY_FILES = {
    "single": "Image_440_SinglePhoton_Iter_{iterations}_{frames}",
    "compton": "Image_440_ComptonOnly_Iter_{iterations}_{frames}",
    "joint": "Image_440_SinglePlusCompton_Iter_{iterations}_{frames}",
}
COLORS = {"single": "#2878b5", "compton": "#c4473a", "joint": "#3f8f5f"}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--factor-dir",
        type=Path,
        default=repo / "Factors" / "440keV_RotateNum20",
    )
    parser.add_argument("--phantom-radius-mm", type=float, default=120.0)
    parser.add_argument("--phantom-half-height-mm", type=float, default=15.0)
    parser.add_argument("--inner-radius-mm", type=float, default=30.0)
    parser.add_argument("--outer-radius-min-mm", type=float, default=90.0)
    # Keep one full ring away from the r=120 mm source boundary so radial
    # resolution and partial-volume loss do not hide the peripheral plateau.
    parser.add_argument("--outer-radius-max-mm", type=float, default=108.0)
    return parser.parse_args()


def load_history(path: Path, pixel_count: int, frames: int) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    expected = pixel_count * frames
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} values; expected {expected}")
    return values.reshape(frames, pixel_count).astype(np.float64)


def group_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = values[mask]
    return {
        "median": float(np.median(selected)),
        "mean": float(np.mean(selected)),
        "p10": float(np.percentile(selected, 10)),
        "p90": float(np.percentile(selected, 90)),
        "count": int(selected.size),
    }


def exact_ring_profiles(
    histories: dict[str, np.ndarray],
    radius: np.ndarray,
    axial_mask: np.ndarray,
    max_radius: float,
) -> tuple[list[dict], dict[str, np.ndarray], np.ndarray]:
    ring_values = np.unique(np.round(radius[radius < max_radius - 1.0e-4], 5))
    records: list[dict] = []
    profiles: dict[str, np.ndarray] = {}
    for name, history in histories.items():
        profile = np.empty((history.shape[0], ring_values.size), dtype=np.float64)
        for ring_index, ring_radius in enumerate(ring_values):
            mask = axial_mask & np.isclose(radius, ring_radius, atol=1.0e-4)
            for frame_index, image in enumerate(history):
                stats = group_stats(image, mask)
                profile[frame_index, ring_index] = stats["median"]
                records.append(
                    {
                        "modality": name,
                        "frame_index": frame_index,
                        "radius_mm": float(ring_radius),
                        **stats,
                    }
                )
        profiles[name] = profile
    return records, profiles, ring_values


def load_single_sensitivity(factor_dir: Path, pixel_count: int) -> np.ndarray:
    matrix_path = factor_dir / "SysMat_polar"
    detector_count = matrix_path.stat().st_size // (4 * pixel_count)
    matrix = np.memmap(
        matrix_path,
        dtype=np.float32,
        mode="r",
        shape=(pixel_count, detector_count),
    )
    base = np.sum(matrix, axis=1, dtype=np.float64)
    rotation_inverse = np.loadtxt(
        factor_dir / "RotMatInv_full.csv", delimiter=",", dtype=np.int64
    )
    return np.mean(base[rotation_inverse - 1], axis=1)


def radial_median(values: np.ndarray, radius: np.ndarray, rings: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.median(values[np.isclose(radius, ring, atol=1.0e-4)])
            for ring in rings
        ],
        dtype=np.float64,
    )


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    factor_dir = args.factor_dir.resolve()
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    iterations = int(manifest["iterations"])
    save_step = int(manifest["save_step"])
    frames = iterations // save_step

    coordinates = np.loadtxt(factor_dir / "coor_polar_full.csv", delimiter=",")
    pixel_count = coordinates.shape[0]
    radius = np.hypot(coordinates[:, 0], coordinates[:, 1])
    z_mm = coordinates[:, 2]
    # Cell centers at +/-13.5 mm are wholly inside the 30-mm source height.
    axial_limit = args.phantom_half_height_mm - 1.5
    axial_mask = np.abs(z_mm) <= axial_limit + 1.0e-4
    inner_mask = axial_mask & (radius <= args.inner_radius_mm + 1.0e-4)
    outer_mask = (
        axial_mask
        & (radius >= args.outer_radius_min_mm - 1.0e-4)
        & (radius <= args.outer_radius_max_mm + 1.0e-4)
    )

    histories = {
        name: load_history(
            result_dir / template.format(iterations=iterations, frames=frames),
            pixel_count,
            frames,
        )
        for name, template in HISTORY_FILES.items()
    }
    records, profiles, rings = exact_ring_profiles(
        histories, radius, axial_mask, args.phantom_radius_mm
    )
    for record in records:
        record["iteration"] = (record.pop("frame_index") + 1) * save_step

    csv_path = result_dir / "compton_radial_background_profiles.csv"
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    volumes = np.fromfile(
        factor_dir / "polar_cell_volume_mm3.float64", dtype=np.float64
    )
    sensi_d = np.fromfile(factor_dir / "Sensi_d", dtype=np.float32).astype(np.float64)
    if volumes.size != pixel_count or sensi_d.size != pixel_count:
        raise ValueError("Factor sensitivity or volume size does not match coordinates")
    sensi_s = load_single_sensitivity(factor_dir, pixel_count)
    point_sensitivities = {
        "Sensi_s / DeltaV": sensi_s / volumes,
        "Sensi_d / DeltaV": sensi_d / volumes,
    }

    iterations_saved = np.arange(1, frames + 1) * save_step
    region_metrics: dict[str, list[dict]] = {}
    for name, history in histories.items():
        values = []
        for iteration, image in zip(iterations_saved, history):
            inner = group_stats(image, inner_mask)
            outer = group_stats(image, outer_mask)
            values.append(
                {
                    "iteration": int(iteration),
                    "inner_median": inner["median"],
                    "outer_median": outer["median"],
                    "outer_over_inner": outer["median"] / inner["median"],
                }
            )
        region_metrics[name] = values

    sensitivity_metrics = {}
    for name, values in point_sensitivities.items():
        inner = group_stats(values, radius <= args.inner_radius_mm + 1.0e-4)
        outer = group_stats(
            values,
            (radius >= args.outer_radius_min_mm - 1.0e-4)
            & (radius <= args.outer_radius_max_mm + 1.0e-4),
        )
        sensitivity_metrics[name] = {
            "inner_median": inner["median"],
            "outer_median": outer["median"],
            "outer_over_inner": outer["median"] / inner["median"],
        }

    summary = {
        "result_dir": str(result_dir),
        "factor_dir": str(factor_dir),
        "geometry": {
            "phantom_radius_mm": args.phantom_radius_mm,
            "phantom_half_height_mm": args.phantom_half_height_mm,
            "included_voxel_center_abs_z_max_mm": axial_limit,
            "inner_radius_mm": args.inner_radius_mm,
            "outer_radius_range_mm": [
                args.outer_radius_min_mm,
                args.outer_radius_max_mm,
            ],
        },
        "region_metrics": region_metrics,
        "point_sensitivity_metrics": sensitivity_metrics,
    }
    summary_path = result_dir / "compton_radial_bias_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for name, profile in profiles.items():
        normalizer = np.median(histories[name][-1, inner_mask])
        axes[0].plot(
            rings,
            profile[-1] / normalizer,
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=COLORS[name],
            label=name,
        )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"Final background profile, iteration {iterations}")
    axes[0].set_xlabel("Radius (mm)")
    axes[0].set_ylabel("Ring median / inner-region median")

    for name, values in region_metrics.items():
        axes[1].plot(
            [item["iteration"] for item in values],
            [item["outer_over_inner"] for item in values],
            linewidth=1.8,
            color=COLORS[name],
            label=name,
        )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Radial bias evolution")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Outer / inner background median")

    sensitivity_colors = ["#2878b5", "#c4473a"]
    for (name, values), color in zip(point_sensitivities.items(), sensitivity_colors):
        profile = radial_median(values, radius, rings)
        inner = np.median(values[radius <= args.inner_radius_mm + 1.0e-4])
        axes[2].plot(rings, profile / inner, linewidth=1.8, color=color, label=name)
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Installed point sensitivities")
    axes[2].set_xlabel("Radius (mm)")
    axes[2].set_ylabel("Ring median / inner-region median")

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    figure_path = result_dir / "Compton_radial_bias_diagnostic.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    print(json.dumps({
        "figure": str(figure_path),
        "csv": str(csv_path),
        "summary": str(summary_path),
        "final_outer_over_inner": {
            name: values[-1]["outer_over_inner"]
            for name, values in region_metrics.items()
        },
        "point_sensitivity_outer_over_inner": {
            name: values["outer_over_inner"]
            for name, values in sensitivity_metrics.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
