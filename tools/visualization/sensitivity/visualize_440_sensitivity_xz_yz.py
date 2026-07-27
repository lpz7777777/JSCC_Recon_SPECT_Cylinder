"""Plot volume-deweighted 440-keV Sensi_s and Sensi_d x-z/y-z sections."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def load_single_photon(factor, pixel_count, detector_count, rotate_num):
    matrix = np.memmap(
        factor / "SysMat_polar", dtype=np.float32, mode="r",
        shape=(pixel_count, detector_count),
    )
    base = matrix.sum(axis=1, dtype=np.float64)
    inverse = np.loadtxt(factor / "RotMatInv_full.csv", delimiter=",", dtype=np.int64)
    if inverse.shape != (pixel_count, rotate_num):
        raise ValueError(f"Unexpected RotMatInv_full shape: {inverse.shape}")
    result = np.zeros(pixel_count, dtype=np.float64)
    for rotation in range(rotate_num):
        result += base[inverse[:, rotation] - 1]
    return result / rotate_num


def interpolate_section(coordinates, values, axis, fixed_axis):
    """Return a [z, axis] section, with fixed_axis='y0' or 'x0'."""
    x_axis = axis
    z_values = np.unique(coordinates[:, 2])
    section = np.full((z_values.size, axis.size), np.nan, dtype=np.float64)
    for z_index, z_value in enumerate(z_values):
        selected = np.isclose(coordinates[:, 2], z_value)
        xy = coordinates[selected, :2]
        layer_values = values[selected]
        if fixed_axis == "y0":
            points = np.column_stack((xy[:, 0], xy[:, 1]))
            query = np.column_stack((x_axis, np.zeros_like(x_axis)))
        else:
            points = np.column_stack((xy[:, 1], xy[:, 0]))
            query = np.column_stack((x_axis, np.zeros_like(x_axis)))
        linear = griddata(points, layer_values, query, method="linear")
        nearest = griddata(points, layer_values, query, method="nearest")
        linear[np.isnan(linear)] = nearest[np.isnan(linear)]
        radius = np.abs(x_axis)
        linear[radius > np.max(np.hypot(xy[:, 0], xy[:, 1])) + 2.0] = np.nan
        section[z_index] = linear
    return z_values, section


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-dir", type=Path, default=Path("Factors/440keV_RotateNum20"))
    parser.add_argument("--sensi-d-path", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("Auxiliary_Studies/Sensitivity_SPECT_PolarCoor/Result/440keV_RotateNum20_XZ_YZ_Sections"),
    )
    parser.add_argument("--grid-step-mm", type=float, default=2.0)
    args = parser.parse_args()
    factor = args.factor_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    coordinates = np.loadtxt(factor / "coor_polar_full.csv", delimiter=",", dtype=np.float64)
    pixel_count = coordinates.shape[0]
    with (factor / "factor_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    detector_count = int(manifest["detector_num"])
    rotate_num = int(manifest["rotate_num"])
    sensi_s = load_single_photon(factor, pixel_count, detector_count, rotate_num)
    sensi_d_path = (args.sensi_d_path or factor / "Sensi_d").resolve()
    sensi_d = np.fromfile(sensi_d_path, dtype=np.float32).astype(np.float64)
    if sensi_d.size != pixel_count:
        raise ValueError(f"Sensi_d has {sensi_d.size} values, expected {pixel_count}")
    volumes = np.fromfile(factor / "polar_cell_volume_mm3.float64", dtype=np.float64)
    if volumes.size != pixel_count:
        raise ValueError(f"DeltaV has {volumes.size} values, expected {pixel_count}")

    # SysMat_polar and Sensi_d are density-basis quantities. The displayed maps
    # are point efficiencies, so remove exactly one polar-cell volume factor.
    sensi_s_efficiency = sensi_s / volumes
    sensi_d_efficiency = sensi_d / volumes
    radius = np.max(np.abs(coordinates[:, :2])) + args.grid_step_mm
    axis = np.arange(-radius, radius + args.grid_step_mm * 0.5, args.grid_step_mm)
    z_axis, s_xz = interpolate_section(coordinates, sensi_s_efficiency, axis, "y0")
    _, d_xz = interpolate_section(coordinates, sensi_d_efficiency, axis, "y0")
    _, s_yz = interpolate_section(coordinates, sensi_s_efficiency, axis, "x0")
    _, d_yz = interpolate_section(coordinates, sensi_d_efficiency, axis, "x0")

    all_values = np.concatenate([
        s_xz[np.isfinite(s_xz)], d_xz[np.isfinite(d_xz)],
        s_yz[np.isfinite(s_yz)], d_yz[np.isfinite(d_yz)],
    ])
    vmin = float(np.nanmin(all_values))
    vmax = float(np.nanmax(all_values))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    panel_data = [
        (s_xz, "Sensi_s / DeltaV: coronal (y=0)"),
        (d_xz, "Sensi_d / DeltaV: coronal (y=0)"),
        (s_yz, "Sensi_s / DeltaV: sagittal (x=0)"),
        (d_yz, "Sensi_d / DeltaV: sagittal (x=0)"),
    ]
    for ax, (image, title) in zip(axes.flat, panel_data):
        shown = ax.imshow(
            image,
            origin="lower",
            extent=(axis[0], axis[-1], z_axis[0], z_axis[-1]),
            aspect="equal",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("x or y (mm)")
        ax.set_ylabel("z (mm)")
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04, label="point efficiency")
    fig.suptitle("440 keV JSCC sensitivity sections, volume deweighted")
    figure_path = output / "Sensi_s_Sensi_d_xz_yz_volume_deweighted.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    independent_figure, independent_axes = plt.subplots(
        2, 2, figsize=(13, 9), constrained_layout=True
    )
    for ax, (image, title) in zip(independent_axes.flat, panel_data):
        valid = image[np.isfinite(image)]
        local_min = float(np.quantile(valid, 0.001))
        local_max = float(np.quantile(valid, 0.999))
        shown = ax.pcolormesh(
            axis, z_axis, image, shading="auto", cmap="viridis",
            vmin=local_min, vmax=local_max,
        )
        ax.set_title(title + " (independent scale)")
        ax.set_xlabel("x or y (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_aspect("equal")
        independent_figure.colorbar(
            shown, ax=ax, fraction=0.046, pad=0.04, label="point efficiency"
        )
    independent_figure.suptitle("440 keV JSCC sensitivity sections, volume deweighted")
    independent_figure_path = output / "Sensi_s_Sensi_d_xz_yz_volume_deweighted_independent_scales.png"
    independent_figure.savefig(independent_figure_path, dpi=200)
    plt.close(independent_figure)

    np.savez_compressed(
        output / "Sensi_s_Sensi_d_xz_yz_volume_deweighted.npz",
        axis_mm=axis, z_mm=z_axis, sensi_s_xz=s_xz, sensi_d_xz=d_xz,
        sensi_s_yz=s_yz, sensi_d_yz=d_yz,
    )
    summary = {
        "factor_dir": str(factor),
        "sensi_d_path": str(sensi_d_path),
        "pixel_count": pixel_count,
        "detector_count": detector_count,
        "rotate_num": rotate_num,
        "volume_deweighting": "Sensi_s / DeltaV and Sensi_d / DeltaV",
        "section_planes": {"coronal": "y=0", "sagittal": "x=0"},
        "grid_step_mm": args.grid_step_mm,
        "axis_range_mm": [float(axis[0]), float(axis[-1])],
        "z_range_mm": [float(z_axis[0]), float(z_axis[-1])],
        "common_color_limits": [vmin, vmax],
        "figure": str(figure_path),
        "independent_scale_figure": str(independent_figure_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
