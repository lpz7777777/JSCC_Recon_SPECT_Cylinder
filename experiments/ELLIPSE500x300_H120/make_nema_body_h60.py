"""Create and preview a centered, 60-mm-high NEMA-like body source geometry.

This only creates phantom truth and figures. It never launches Geant4 or reconstruction.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "nema_body_h60_config.json"
EXPERIMENT = HERE / "config.json"
GENERATED = HERE / "generated/NEMA_Body_H60"
REPORTS = HERE / "reports/NEMA_Body_H60"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def body_boundary(config):
    half_x = config["body_width_mm"] / 2
    half_y = config["body_ap_mm"] / 2
    knot = np.asarray(config["body_lower_contour_knots_mm"], dtype=np.float64)
    if (knot.shape != (6, 2) or knot[0, 0] != 0 or knot[-1, 0] != half_x or
        knot[0, 1] != -half_y or knot[-1, 1] != 0):
        raise ValueError("Body contour endpoints do not match dimensions")
    lower = PchipInterpolator(knot[:, 0], knot[:, 1])

    def contains(x, y):
        x, y = np.broadcast_arrays(x, y)
        within_x = np.abs(x) <= half_x
        top = half_y * np.sqrt(np.clip(1 - (x / half_x) ** 2, 0, 1))
        bottom = lower(np.minimum(np.abs(x), half_x))
        return within_x & (y <= top) & (y >= bottom)

    upper_x = np.linspace(-half_x, half_x, 600)
    upper_y = half_y * np.sqrt(np.maximum(0, 1 - (upper_x / half_x) ** 2))
    lower_x = upper_x[::-1]
    lower_y = lower(np.abs(lower_x))
    outline = np.column_stack((np.r_[upper_x, lower_x],
                               np.r_[upper_y, lower_y]))
    return contains, outline


def partial_xy(x, y, spacing, samples, predicate):
    xx, yy = np.meshgrid(x, y, indexing="xy")
    fraction = np.zeros(xx.shape, dtype=np.float64)
    offsets = (np.arange(samples) + .5) / samples * spacing - spacing / 2
    for dx in offsets:
        for dy in offsets:
            fraction += predicate(xx + dx, yy + dy)
    return (fraction / samples ** 2).astype(np.float32)


def partial_sphere(x, y, z, spacing, samples, center, radius):
    cx, cy, cz = center
    out = np.zeros((len(z), len(y), len(x)), dtype=np.float32)
    ix = np.flatnonzero(abs(x - cx) <= radius + spacing / 2)
    iy = np.flatnonzero(abs(y - cy) <= radius + spacing / 2)
    iz = np.flatnonzero(abs(z - cz) <= radius + spacing / 2)
    if not (len(ix) and len(iy) and len(iz)):
        raise ValueError("Sphere misses Cartesian truth grid")
    xx = x[ix][None, None, :]
    yy = y[iy][None, :, None]
    zz = z[iz][:, None, None]
    fraction = np.zeros((len(iz), len(iy), len(ix)), dtype=np.float64)
    offsets = (np.arange(samples) + .5) / samples * spacing - spacing / 2
    for dx in offsets:
        sx = (xx + dx - cx) ** 2
        for dy in offsets:
            sxy = sx + (yy + dy - cy) ** 2
            for dz in offsets:
                fraction += sxy + (zz + dz - cz) ** 2 <= radius ** 2
    out[np.ix_(iz, iy, ix)] = (fraction / samples ** 3).astype(np.float32)
    return out


def make_truth(config, experiment):
    if (experiment["experiment_id"] != config["experiment"] or
        experiment["physical_shape"] != "ellipse_cylinder" or
        experiment["semi_axes_mm"] != [250.0, 150.0] or
        experiment["height_mm"] != 120.0 or
        config["body_height_mm"] != 60.0 or
        config["truth_spacing_mm"] != experiment["truth_spacing_mm"]):
        raise ValueError("NEMA preview requires the approved ellipse FOV geometry")
    spacing = config["truth_spacing_mm"]
    samples = config["subvoxel_samples_per_axis"]
    if samples < 4:
        raise ValueError("Small sphere truth requires subvoxel sampling")
    x = np.arange(-250.5, 250.6, spacing)
    y = np.arange(-148.5, 148.6, spacing)
    z = np.arange(-58.5, 58.6, spacing)
    if (len(x), len(y), len(z)) != (168, 100, 40):
        raise ValueError("Unexpected full ellipse truth canvas")
    contains, outline = body_boundary(config)
    body_xy = partial_xy(x, y, spacing, samples, contains)
    lung_radius = config["lung_outer_diameter_mm"] / 2
    lung_xy = partial_xy(x, y, spacing, samples,
        lambda xx, yy: xx ** 2 + yy ** 2 <= lung_radius ** 2)
    height_mask = (abs(z) < config["body_height_mm"] / 2).astype(np.float32)
    body = height_mask[:, None, None] * body_xy[None, :, :]
    lung = height_mask[:, None, None] * lung_xy[None, :, :]
    spheres = {}
    centers = []
    for item in config["spheres"]:
        diameter = item["diameter_mm"]
        theta = math.radians(item["angle_deg"])
        radius_from_center = config["sphere_center_ring_radius_mm"]
        center = (radius_from_center * math.cos(theta),
                  radius_from_center * math.sin(theta),
                  config["sphere_equator_z_mm"])
        centers.append({**item, "center_mm": [float(v) for v in center]})
        spheres[int(diameter)] = partial_sphere(x, y, z, spacing, samples,
                                                center, diameter / 2)
    all_sphere = sum(spheres.values())
    if np.any(all_sphere > 1.00001):
        raise ValueError("Sphere masks overlap")
    if np.any(lung + all_sphere > body + .025):
        raise ValueError("Insert or sphere extends beyond the body")
    emission = body - lung - all_sphere
    for item in config["spheres"]:
        value = config["preview_relative_activity"]["hot_spheres" if item["fill"] == "hot" else "cold_spheres"]
        emission += value * spheres[int(item["diameter_mm"])]
    emission = np.maximum(emission, 0).astype(np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    # The entire 290 x 220 mm bounding rectangle fits the physical ellipse.
    corner_radius_sq = (config["body_width_mm"] / 2 / 250) ** 2 + (config["body_ap_mm"] / 2 / 150) ** 2
    if corner_radius_sq >= 1 or np.any((body > 0) &
        (((xx / 250) ** 2 + (yy / 150) ** 2)[None, :, :] > 1)):
        raise ValueError("NEMA body extends beyond physical ellipse")
    return x, y, z, body, lung, spheres, emission, centers, outline


def plot_layout(config, outline, centers, output):
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    ax.add_patch(Ellipse((0, 0), 500, 300, fill=False, edgecolor="#2f72b7",
                         linestyle="--", linewidth=1.7, label="ELLIPSE500×300 FOV"))
    ax.fill(outline[:, 0], outline[:, 1], color="#d9dcdf", ec="#59636d",
            linewidth=2, label="NEMA-like body, 290×220 mm")
    ax.add_patch(Circle((0, 0), config["lung_outer_diameter_mm"] / 2,
                        fc="white", ec="#59636d", linewidth=1.4))
    ax.text(0, 0, "lung\nØ51", ha="center", va="center", fontsize=10)
    for sphere in centers:
        cx, cy, _ = sphere["center_mm"]
        hot = sphere["fill"] == "hot"
        ax.add_patch(Circle((cx, cy), sphere["diameter_mm"] / 2,
                            fc="#c84b40" if hot else "#4384b5", ec="black", lw=.7))
        label_offset = {10: (22, 15), 13: (12, 20), 17: (-27, 20),
                        22: (-34, 13), 28: (-23, -23), 37: (23, -23)}
        dx, dy = label_offset[int(sphere["diameter_mm"])]
        ax.annotate(f"Ø{int(sphere['diameter_mm'])}", (cx, cy),
                    xytext=(cx + dx, cy + dy),
                    fontsize=10, weight="bold",
                    arrowprops={"arrowstyle": "-", "lw": .7, "color": "#333333"})
    ax.annotate("", (-145, -134), (145, -134), arrowprops={"arrowstyle": "|-|", "lw": 1.2})
    ax.text(0, -137, "290 mm", ha="center", va="top", fontsize=10)
    ax.annotate("", (165, -110), (165, 110), arrowprops={"arrowstyle": "|-|", "lw": 1.2})
    ax.text(169, 0, "220 mm", rotation=90, va="center", fontsize=10)
    ax.plot([], [], "o", color="#c84b40", label="hot spheres (4:1 preview)")
    ax.plot([], [], "o", color="#4384b5", label="cold spheres (0:1 preview)")
    ax.set(xlim=(-270, 270), ylim=(-160, 160), xlabel="object x (mm)",
           ylabel="object y (mm)", title="Centered NEMA-like body phantom in ellipse FOV")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_slices(x, y, z, emission, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    k = int(np.argmin(abs(z)))
    jy = int(np.argmin(abs(y - 49.5)))
    ix = int(np.argmin(abs(x - 28.5)))
    panels = ((emission[k], (x[0]-1.5, x[-1]+1.5, y[0]-1.5, y[-1]+1.5),
               f"axial z={z[k]:+.1f} mm", "x (mm)", "y (mm)"),
              (emission[:, jy, :], (x[0]-1.5, x[-1]+1.5, z[0]-1.5, z[-1]+1.5),
               f"coronal y={y[jy]:+.1f} mm", "x (mm)", "z (mm)"),
              (emission[:, :, ix], (y[0]-1.5, y[-1]+1.5, z[0]-1.5, z[-1]+1.5),
               f"sagittal x={x[ix]:+.1f} mm", "y (mm)", "z (mm)"))
    for ax, (image, extent, title, xlabel, ylabel) in zip(axes, panels):
        view = ax.imshow(image, origin="lower", extent=extent, cmap="viridis",
                         vmin=0, vmax=4, interpolation="nearest", aspect="equal")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axes[0].set(xlim=(-165, 165), ylim=(-120, 120))
    axes[1].set(xlim=(-165, 165), ylim=(-36, 36))
    axes[2].set(xlim=(-120, 120), ylim=(-36, 36))
    fig.colorbar(view, ax=axes, shrink=.8, label="relative activity (preview only)")
    fig.suptitle("3-mm fractional-volume truth; background 1, four hot spheres 4, two cold spheres 0")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_3d(outline, centers, output):
    fig = plt.figure(figsize=(9, 7), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    zlow, zhigh = -30, 30
    boundary = outline[::12]
    for level in (zlow, zhigh):
        ax.plot(boundary[:, 0], boundary[:, 1], level, color="#57616c", lw=1.1)
    for index in range(0, len(boundary), 8):
        bx, by = boundary[index]
        ax.plot([bx, bx], [by, by], [zlow, zhigh], color="#9ca4ab", alpha=.55, lw=.6)
    u = np.linspace(0, 2*np.pi, 22)
    v = np.linspace(0, np.pi, 12)
    for sphere in centers:
        cx, cy, cz = sphere["center_mm"]
        r = sphere["diameter_mm"] / 2
        sx = cx + r * np.outer(np.cos(u), np.sin(v))
        sy = cy + r * np.outer(np.sin(u), np.sin(v))
        sz = cz + r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(sx, sy, sz, color="#c84b40" if sphere["fill"] == "hot" else "#4384b5",
                        alpha=.85, linewidth=0, shade=True)
    t = np.linspace(0, 2*np.pi, 90)
    for level in (zlow, zhigh):
        ax.plot(25.5*np.cos(t), 25.5*np.sin(t), level, color="#378f9a", lw=1.6)
    ax.set(xlim=(-160, 160), ylim=(-120, 120), zlim=(-40, 40),
           xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)",
           title="60-mm axial body, central lung insert and six spherical compartments")
    ax.view_init(elev=22, azim=-65)
    ax.set_box_aspect((320, 240, 80))
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main():
    config = json.loads(CONFIG.read_text())
    experiment = json.loads(EXPERIMENT.read_text())
    x, y, z, body, lung, spheres, emission, centers, outline = make_truth(config, experiment)
    GENERATED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    truth_path = GENERATED / "truth_3mm.npz"
    np.savez_compressed(truth_path, x_mm=x, y_mm=y, z_mm=z,
                        body_fraction_zyx=body, lung_fraction_zyx=lung,
                        relative_activity_zyx=emission,
                        **{f"sphere_{diameter}_fraction_zyx": array
                           for diameter, array in spheres.items()})
    plot_layout(config, outline, centers, REPORTS / "layout_in_ellipse.png")
    plot_slices(x, y, z, emission, REPORTS / "activity_slices.png")
    plot_3d(outline, centers, REPORTS / "geometry_3d.png")
    spacing = config["truth_spacing_mm"]
    volume_voxel = spacing ** 3
    sphere_metrics = []
    for item in centers:
        diameter = int(item["diameter_mm"])
        sampled = float(spheres[diameter].sum(dtype=np.float64) * volume_voxel)
        analytic = 4 / 3 * math.pi * (diameter / 2) ** 3
        relative_error = sampled / analytic - 1
        if abs(relative_error) > .01:
            raise ValueError(f"Sphere volume error exceeds 1%: Ø{diameter}, {relative_error:+.2%}")
        sphere_metrics.append({**item, "sampled_volume_mm3": sampled,
                               "analytic_volume_mm3": analytic,
                               "relative_volume_error": relative_error})
    report = {"experiment": config["experiment"], "phantom": config["name"],
              "status": "geometry_and_truth_preview_only_no_transport_no_reconstruction",
              "shape_zyx": list(body.shape), "spacing_mm": spacing,
              "body_dimensions_mm": [config["body_width_mm"], config["body_ap_mm"], config["body_height_mm"]],
              "physical_fov_mm": [500, 300, 120],
              "fov_center_margin_mm": [105, 40, 30],
              "body_volume_mm3": float(body.sum(dtype=np.float64) * volume_voxel),
              "lung_outer_volume_mm3": float(lung.sum(dtype=np.float64) * volume_voxel),
              "spheres": sphere_metrics,
              "relative_activity_integral_mm3": float(emission.sum(dtype=np.float64) * volume_voxel),
              "truth_sha256": sha256(truth_path),
              "configuration_sha256": sha256(CONFIG),
              "preview_sha256": {name: sha256(REPORTS / name)
                                  for name in ("layout_in_ellipse.png", "activity_slices.png", "geometry_3d.png")},
              "material_attenuation": False,
              "standard_compliance": "adapted research phantom: 60-mm length differs from NEMA NU 2 body minimum 180 mm"}
    (REPORTS / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"truth": str(truth_path), "report": str(REPORTS),
                      "shape": report["shape_zyx"], "max_sphere_volume_error":
                      max(abs(row["relative_volume_error"]) for row in sphere_metrics)}, indent=2))


if __name__ == "__main__":
    main()
