"""Compare Compton-list Sensi_d with the 440-keV single-photon sensitivity.

Both maps are kept on the canonical 25620-point polar grid.  Sensi_d is a
density-basis quantity from the uniform cylindrical source experiment; the
single-photon map is calculated with the same rotation averaging convention
used by the local MLEM/OSEM reconstruction.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def load_single_photon(
    sysmat_path,
    rotmat_inv_path,
    pixel_count,
    detector_count,
    rotate_num,
    apply_rotation_average=True,
):
    # MATLAB writes [detector, pixel] arrays in column-major order. The local
    # reconstruction therefore maps the file as [pixel, detector] and
    # transposes it; use the same convention here.
    matrix_on_disk = np.memmap(
        sysmat_path, dtype=np.float32, mode="r", shape=(pixel_count, detector_count)
    )
    inverse = np.loadtxt(rotmat_inv_path, delimiter=",", dtype=np.int64)
    if inverse.shape != (pixel_count, rotate_num):
        raise ValueError(f"Unexpected RotMatInv shape: {inverse.shape}")
    base_sensitivity = matrix_on_disk.sum(axis=1, dtype=np.float64)
    if not apply_rotation_average:
        return base_sensitivity
    sensitivity = np.zeros(pixel_count, dtype=np.float64)
    for rotation in range(rotate_num):
        sensitivity += base_sensitivity[inverse[:, rotation] - 1]
    return sensitivity / rotate_num


def radial_profile(radius, values, bins):
    index = np.digitize(radius, bins) - 1
    centers, medians, means, counts = [], [], [], []
    for i in range(len(bins) - 1):
        selected = values[index == i]
        if selected.size:
            centers.append((bins[i] + bins[i + 1]) / 2)
            medians.append(np.median(selected))
            means.append(np.mean(selected))
            counts.append(selected.size)
    return np.asarray(centers), np.asarray(medians), np.asarray(means), np.asarray(counts)


def stats(name, values):
    return {
        "name": name,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "cv": float(np.std(values) / np.mean(values)),
    }


def collapse_z_and_interpolate(coords, values, grid_step_mm=2.0):
    """Return a continuous Cartesian x-y map after averaging the z layers."""
    xy = coords[:, :2]
    unique_xy, inverse = np.unique(np.round(xy, decimals=8), axis=0, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    collapsed = sums / counts
    extent = float(np.max(np.abs(unique_xy)))
    axis = np.arange(-extent, extent + grid_step_mm * 0.5, grid_step_mm)
    xx, yy = np.meshgrid(axis, axis)
    linear = griddata(unique_xy, collapsed, (xx, yy), method="linear")
    nearest = griddata(unique_xy, collapsed, (xx, yy), method="nearest")
    # Keep the exterior of the sampled FOV blank; nearest only fills holes
    # inside the convex hull caused by the polar sampling pattern.
    hull = griddata(unique_xy, np.ones(unique_xy.shape[0]), (xx, yy), method="linear")
    linear[np.isnan(linear) & (hull > 0)] = nearest[np.isnan(linear) & (hull > 0)]
    radius_grid = np.hypot(xx, yy)
    linear[radius_grid > extent + grid_step_mm] = np.nan
    return xx, yy, linear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-dir", type=Path, default=Path("Factors/440keV_RotateNum20"))
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("Auxiliary_Studies/Sensitivity_SPECT_PolarCoor/Result/440keV_RotateNum20_UniformFullFOV_5e10"),
    )
    parser.add_argument(
        "--single-angle",
        action="store_true",
        help="Display unrotated Sensi_d against the unrotated base single-photon sensitivity.",
    )
    args = parser.parse_args()
    factor = args.factor_dir.resolve()
    result = args.result_dir.resolve()
    result.mkdir(parents=True, exist_ok=True)

    coords = np.loadtxt(factor / "coor_polar_full.csv", delimiter=",")
    sensi_d = np.fromfile(result / "Sensi_d", dtype=np.float32).astype(np.float64)
    pixel_count = coords.shape[0]
    if sensi_d.size != pixel_count:
        raise ValueError(f"Sensi_d has {sensi_d.size} values, expected {pixel_count}")

    manifest = json.loads((factor / "factor_manifest.json").read_text(encoding="utf-8"))
    detector_count = int(manifest["detector_num"])
    rotate_num = int(manifest["rotate_num"])
    sensi_s = load_single_photon(
        factor / "SysMat_polar", factor / "RotMatInv_full.csv",
        pixel_count, detector_count, rotate_num,
        apply_rotation_average=not args.single_angle,
    )

    coords_xy = coords[:, :2]
    radius = np.hypot(coords_xy[:, 0], coords_xy[:, 1])
    # Both sensitivities use the density basis B=A*diag(DeltaV): Sensi_s is the
    # column sum of B, while the uniform-source estimator gives
    # Sensi_d_j=DeltaV_j*epsilon_d_j. Divide both by DeltaV to recover the
    # dimensionless per-emitted-photon detection efficiencies.
    volume_path = factor / "polar_cell_volume_mm3.float64"
    cell_volume = np.fromfile(volume_path, dtype=np.float64)
    if cell_volume.size != pixel_count:
        raise ValueError(f"Volume file has {cell_volume.size} values, expected {pixel_count}")
    sensi_d_display = sensi_d / cell_volume
    sensi_s_display = sensi_s / cell_volume
    ratio_s_over_d = sensi_s_display / np.maximum(sensi_d_display, np.finfo(np.float64).tiny)
    correlation = float(np.corrcoef(sensi_d_display, sensi_s_display)[0, 1])

    bins = np.arange(0.0, max(radius) + 6.001, 6.0)
    radial = {}
    for name, values in (("Sensi_d_display", sensi_d_display), ("Sensi_s_display", sensi_s_display), ("ratio_s_over_d", ratio_s_over_d)):
        centers, medians, means, counts = radial_profile(radius, values, bins)
        radial[name] = (centers, medians, means, counts)
    with (result / "Sensi_d_vs_single_photon_radial.csv").open("w", encoding="ascii") as handle:
        handle.write("radius_mm,Sensi_s_over_DeltaV_median,Sensi_s_over_DeltaV_mean,Sensi_d_over_DeltaV_median,Sensi_d_over_DeltaV_mean,Sensi_s_over_d_median,Sensi_s_over_d_mean,count\n")
        for i, r in enumerate(radial["Sensi_d_display"][0]):
            handle.write(",".join(f"{x:.9g}" for x in (r, radial["Sensi_s_display"][1][i], radial["Sensi_s_display"][2][i], radial["Sensi_d_display"][1][i], radial["Sensi_d_display"][2][i], radial["ratio_s_over_d"][1][i], radial["ratio_s_over_d"][2][i], radial["Sensi_d_display"][3][i])) + "\n")

    summary = {
        "factor_dir": str(factor), "result_dir": str(result),
        "pixel_count": pixel_count, "detector_count": detector_count, "rotate_num": rotate_num,
        "spatial_mode": "single_angle" if args.single_angle else "rotation_average",
        "comparison_note": "Sensi_d is accepted Compton-list sensitivity; Sensi_s is direct 440-window single-photon sensitivity. Absolute ratio is not a pure detector efficiency ratio.",
        "correlation_point_efficiency_maps": correlation,
        "ratio_s_over_d_display": stats("Sensi_s / Sensi_d after DeltaV removal", ratio_s_over_d),
        "Sensi_s_point_efficiency": stats("Sensi_s / DeltaV", sensi_s_display),
        "Sensi_d_point_efficiency": stats("Sensi_d / DeltaV", sensi_d_display),
        "Sensi_d": stats("Sensi_d", sensi_d), "Sensi_s": stats("Sensi_s", sensi_s),
    }
    (result / "Sensi_d_vs_single_photon_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cartesian_maps = [(sensi_s_display, "Sensi_s / DeltaV", "viridis"), (sensi_d_display, "Sensi_d / DeltaV", "viridis"), (ratio_s_over_d, "Sensi_s / Sensi_d", "coolwarm")]
    cartesian = [collapse_z_and_interpolate(coords, values) for values, _, _ in cartesian_maps]
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.8), constrained_layout=True)
    for ax, ((_, title, cmap), (_, _, values)) in zip(axes[:3], zip(cartesian_maps, cartesian)):
        image = ax.pcolormesh(cartesian[0][0], cartesian[0][1], values, shading="auto", cmap=cmap)
        ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_title(title + " (Cartesian x-y, z-mean)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax = axes[3]
    ax.plot(radial["Sensi_s_display"][0], radial["Sensi_s_display"][1], label="Sensi_s / DeltaV", linewidth=2)
    ax.plot(radial["Sensi_d_display"][0], radial["Sensi_d_display"][1], label="Sensi_d / DeltaV", linewidth=2)
    ax.set_xlabel("radius (mm)"); ax.set_ylabel("sensitivity (no median normalization)")
    ax.set_title(f"Radial median (corr={correlation:.4f})"); ax.grid(alpha=0.25); ax.legend(loc="best")
    mode_title = "single angle (no rotation average)" if args.single_angle else "rotation average"
    fig.suptitle(f"440 keV JSCC sensitivity: Compton list vs single-photon response, {mode_title}")
    fig.savefig(result / "Sensi_d_vs_single_photon.png", dpi=180)
    fig.savefig(result / "Sensi_d_vs_single_photon_cartesian_xy.png", dpi=180)
    sensi_d_figure, sensi_d_axis = plt.subplots(figsize=(7.4, 6.3), constrained_layout=True)
    sensi_d_image = sensi_d_axis.pcolormesh(
        cartesian[1][0], cartesian[1][1], cartesian[1][2],
        shading="auto", cmap="viridis",
    )
    sensi_d_axis.set_aspect("equal")
    sensi_d_axis.set_xlabel("x (mm)")
    sensi_d_axis.set_ylabel("y (mm)")
    sensi_d_axis.set_title(f"440 keV Sensi_d / DeltaV, {mode_title}")
    sensi_d_colorbar = sensi_d_figure.colorbar(
        sensi_d_image, ax=sensi_d_axis, fraction=0.046, pad=0.04
    )
    sensi_d_colorbar.set_label("Compton point sensitivity")
    sensi_d_figure.savefig(result / "Sensi_d_point_efficiency_cartesian_xy.png", dpi=200)
    plt.close(sensi_d_figure)
    print(json.dumps(summary, indent=2))
    print(f"Figure: {result / 'Sensi_d_vs_single_photon.png'}")


if __name__ == "__main__":
    main()
