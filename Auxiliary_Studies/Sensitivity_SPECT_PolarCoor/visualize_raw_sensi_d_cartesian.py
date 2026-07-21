"""Render an installed polar-grid Sensi_d as an unnormalized Cartesian x-y map.

The input values are used exactly as stored in ``Factors/<...>/Sensi_d``.
The only reduction is an arithmetic mean across z layers so a 3-D field can be
shown on a single x-y plane.  In particular, this script does not divide by
polar-cell volume and does not apply max, mean, sum, or median normalization.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def collapse_z_to_xy(coords: np.ndarray, values: np.ndarray, grid_step_mm: float):
    xy = coords[:, :2]
    unique_xy, inverse = np.unique(np.round(xy, decimals=8), axis=0, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    xy_mean = sums / counts

    extent = float(np.max(np.abs(unique_xy)))
    axis = np.arange(-extent, extent + 0.5 * grid_step_mm, grid_step_mm)
    xx, yy = np.meshgrid(axis, axis)
    image = griddata(unique_xy, xy_mean, (xx, yy), method="linear")
    nearest = griddata(unique_xy, xy_mean, (xx, yy), method="nearest")
    support = griddata(unique_xy, np.ones(unique_xy.shape[0]), (xx, yy), method="linear")
    image[np.isnan(image) & (support > 0)] = nearest[np.isnan(image) & (support > 0)]
    image[np.hypot(xx, yy) > extent + grid_step_mm] = np.nan
    return xx, yy, image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-step-mm", type=float, default=2.0)
    args = parser.parse_args()

    factor_dir = args.factor_dir.resolve()
    coords = np.loadtxt(factor_dir / "coor_polar_full.csv", delimiter=",")
    sensi_d = np.fromfile(factor_dir / "Sensi_d", dtype=np.float32).astype(np.float64)
    if coords.shape[0] != sensi_d.size:
        raise ValueError(f"Sensi_d has {sensi_d.size} values; coordinates have {coords.shape[0]} rows")

    xx, yy, image = collapse_z_to_xy(coords, sensi_d, args.grid_step_mm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.4, 6.3), constrained_layout=True)
    rendered = axis.pcolormesh(xx, yy, image, shading="auto", cmap="viridis")
    axis.set_aspect("equal")
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_title("511 keV Sensi_d (Cartesian x-y, z-layer mean; raw stored values)")
    colorbar = figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Sensi_d (stored, unnormalized)")
    figure.savefig(args.output, dpi=200)

    summary = {
        "factor_dir": str(factor_dir),
        "output": str(args.output.resolve()),
        "pixel_count": int(sensi_d.size),
        "display": "Cartesian x-y interpolation after z-layer arithmetic mean; no value normalization or volume conversion",
        "sensi_d_raw_min": float(sensi_d.min()),
        "sensi_d_raw_max": float(sensi_d.max()),
        "sensi_d_raw_mean": float(sensi_d.mean()),
        "sensi_d_raw_median": float(np.median(sensi_d)),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
