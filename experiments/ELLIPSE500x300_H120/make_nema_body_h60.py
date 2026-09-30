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
    """Analytic NEMA Figure 7-1 cross-section, with a 3-mm concentric wall.

    The upper half is a semicircle. Each lower quarter is a circle centred
    70 mm to the side, tangent to a straight base. Its center-line tangency
    with the upper semicircle follows from R_top - R_corner = 70 mm.
    """
    arcs = config["body_arc_geometry_mm"]
    upper = float(arcs["top_inner_radius"])
    lower = float(arcs["bottom_inner_radius"])
    offset = float(arcs["bottom_arc_center_offset_x"])
    cy = float(arcs["circle_center_y"])
    wall = float(arcs["wall_thickness"])
    if not np.isclose(upper - lower, offset):
        raise ValueError("Upper and lower inner arcs are not tangent")
    if not np.isclose(2 * (upper + wall), config["body_outer_width_mm"]):
        raise ValueError("Outer width disagrees with the circular arcs")
    if not np.isclose(upper + lower + 2 * wall, config["body_outer_ap_mm"]):
        raise ValueError("Outer AP dimension disagrees with the circular arcs")
    if not np.isclose(cy + (upper - lower) / 2, 0):
        raise ValueError("Outer bounding box must be centered in the FOV")

    def shape(radius_top, radius_bottom):
        def contains(x, y):
            x, y = np.broadcast_arrays(x, y)
            dy = y - cy
            upper_half = (dy >= 0) & (x*x + dy*dy <= radius_top*radius_top)
            lower_half = ((dy < 0) & (dy >= -radius_bottom) &
                          (np.abs(x) <= offset + np.sqrt(np.maximum(
                              radius_bottom*radius_bottom - dy*dy, 0))))
            return upper_half | lower_half

        theta_top = np.linspace(0, np.pi, 500)
        theta_right = np.linspace(-np.pi/2, 0, 180)
        theta_left = np.linspace(-np.pi, -np.pi/2, 180)
        top = np.column_stack((radius_top*np.cos(theta_top),
                               cy + radius_top*np.sin(theta_top)))
        left = np.column_stack((-offset + radius_bottom*np.cos(theta_left),
                                cy + radius_bottom*np.sin(theta_left)))
        base = np.column_stack((np.linspace(-offset, offset, 160),
                                np.full(160, cy-radius_bottom)))
        right = np.column_stack((offset + radius_bottom*np.cos(theta_right),
                                 cy + radius_bottom*np.sin(theta_right)))
        return contains, np.vstack((top, left, base, right))

    inner_contains, inner_outline = shape(upper, lower)
    outer_contains, outer_outline = shape(upper+wall, lower+wall)
    return inner_contains, inner_outline, outer_contains, outer_outline


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
    inner_contains, inner_outline, outer_contains, outer_outline = body_boundary(config)
    body_xy = partial_xy(x, y, spacing, samples, inner_contains)
    outer_xy = partial_xy(x, y, spacing, samples, outer_contains)
    lung_radius = config["lung_outer_diameter_mm"] / 2
    lung_xy = partial_xy(x, y, spacing, samples,
        lambda xx, yy: xx ** 2 + yy ** 2 <= lung_radius ** 2)
    height_mask = (abs(z) < config["body_height_mm"] / 2).astype(np.float32)
    body = height_mask[:, None, None] * body_xy[None, :, :]
    wall = height_mask[:, None, None] * np.maximum(outer_xy-body_xy, 0)[None, :, :]
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
    background = np.maximum(body - lung - all_sphere, 0)
    concentrations = config["relative_activity_concentration"]
    if (concentrations["hot_sphere_to_own_background_ratio"] != 10 or
        concentrations["opposite_energy_in_sphere"] != 0 or
        concentrations["background_218"] <= 0 or
        concentrations["background_440"] <= 0):
        raise ValueError("Unexpected dual-energy NEMA source prescription")
    activity = {}
    for energy in (218, 440):
        bkg = concentrations[f"background_{energy}"]
        current = bkg * background.copy()
        for item in config["spheres"]:
            if item["hot_energy_keV"] == energy:
                current += 10 * bkg * spheres[int(item["diameter_mm"])]
        activity[energy] = current.astype(np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    # The entire 300 x 230 mm outer bounding rectangle fits the ellipse.
    corner_radius_sq = (config["body_outer_width_mm"] / 2 / 250) ** 2 + (config["body_outer_ap_mm"] / 2 / 150) ** 2
    if corner_radius_sq >= 1 or np.any(((body + wall) > 0) &
        (((xx / 250) ** 2 + (yy / 150) ** 2)[None, :, :] > 1)):
        raise ValueError("NEMA body extends beyond physical ellipse")
    return x, y, z, body, wall, lung, spheres, activity, centers, inner_outline, outer_outline


def plot_layout(config, inner_outline, outer_outline, centers, output):
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    ax.add_patch(Ellipse((0, 0), 500, 300, fill=False, edgecolor="#2f72b7",
                         linestyle="--", linewidth=1.7, label="ELLIPSE500×300 FOV"))
    ax.fill(outer_outline[:, 0], outer_outline[:, 1], color="#9aa4ad",
            ec="#4a5560", linewidth=1.4, label="outer shell, 300×230 mm")
    ax.fill(inner_outline[:, 0], inner_outline[:, 1], color="#d9dcdf",
            ec="#69747e", linewidth=1.1, label="water cavity, approx. 294×224 mm")
    ax.add_patch(Circle((0, 0), config["lung_outer_diameter_mm"] / 2,
                        fc="white", ec="#59636d", linewidth=1.4))
    ax.text(0, 0, "lung\nØ51", ha="center", va="center", fontsize=10)
    for sphere in centers:
        cx, cy, _ = sphere["center_mm"]
        is_218 = sphere["hot_energy_keV"] == 218
        ax.add_patch(Circle((cx, cy), sphere["diameter_mm"] / 2,
                            fc="#c84b40" if is_218 else "#4384b5", ec="black", lw=.7))
        label_offset = {10: (22, 15), 13: (12, 20), 17: (-27, 20),
                        22: (-34, 13), 28: (-23, -23), 37: (23, -23)}
        dx, dy = label_offset[int(sphere["diameter_mm"])]
        ax.annotate(f"Ø{int(sphere['diameter_mm'])}", (cx, cy),
                    xytext=(cx + dx, cy + dy),
                    fontsize=10, weight="bold",
                    arrowprops={"arrowstyle": "-", "lw": .7, "color": "#333333"})
    ax.annotate("", (-150, -138), (150, -138), arrowprops={"arrowstyle": "|-|", "lw": 1.2})
    ax.text(0, -141, "300 mm outer", ha="center", va="top", fontsize=10)
    ax.annotate("", (167, -115), (167, 115), arrowprops={"arrowstyle": "|-|", "lw": 1.2})
    ax.text(171, 0, "230 mm outer", rotation=90, va="center", fontsize=10)
    ax.plot([], [], "o", color="#c84b40", label="218 keV spheres (10:1)")
    ax.plot([], [], "o", color="#4384b5", label="440 keV spheres (10:1)")
    ax.set(xlim=(-270, 270), ylim=(-160, 160), xlabel="object x (mm)",
           ylabel="object y (mm)", title="NEMA circular-arc body in centered ellipse FOV")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_slices(x, y, z, activity, output):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    k = int(np.argmin(abs(z)))
    jy = int(np.argmin(abs(y - 49.5)))
    ix = int(np.argmin(abs(x - 28.5)))
    for row, energy in enumerate((218, 440)):
        source = activity[energy]
        panels = ((source[k], (x[0]-1.5, x[-1]+1.5, y[0]-1.5, y[-1]+1.5),
                   f"{energy} keV axial z={z[k]:+.1f} mm", "x (mm)", "y (mm)"),
                  (source[:, jy, :], (x[0]-1.5, x[-1]+1.5, z[0]-1.5, z[-1]+1.5),
                   f"{energy} keV coronal y={y[jy]:+.1f} mm", "x (mm)", "z (mm)"),
                  (source[:, :, ix], (y[0]-1.5, y[-1]+1.5, z[0]-1.5, z[-1]+1.5),
                   f"{energy} keV sagittal x={x[ix]:+.1f} mm", "y (mm)", "z (mm)"))
        for ax, (image, extent, title, xlabel, ylabel) in zip(axes[row], panels):
            view = ax.imshow(image, origin="lower", extent=extent, cmap="viridis",
                             vmin=0, vmax=10, interpolation="nearest", aspect="equal")
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axes[row, 0].set(xlim=(-165, 165), ylim=(-120, 120))
        axes[row, 1].set(xlim=(-165, 165), ylim=(-36, 36))
        axes[row, 2].set(xlim=(-120, 120), ylim=(-36, 36))
    fig.colorbar(view, ax=axes, shrink=.8, label="relative activity concentration")
    fig.suptitle("3-mm fractional-volume truth; each energy: background 1, its three hot spheres 10")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_axial_comparison(x, y, z, activity, output):
    k = int(np.argmin(abs(z)))
    images = (activity[218][k], activity[440][k],
              activity[218][k] + activity[440][k])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
    extent = (x[0]-1.5, x[-1]+1.5, y[0]-1.5, y[-1]+1.5)
    for ax, image, label in zip(axes, images,
                                ("218 keV: Ø10,17,28 hot",
                                 "440 keV: Ø13,22,37 hot",
                                 "both energies: all six hot")):
        plot = ax.imshow(image, origin="lower", extent=extent, cmap="viridis",
                         vmin=0, vmax=10, interpolation="nearest", aspect="equal")
        ax.set(xlim=(-160, 160), ylim=(-120, 120), title=label,
               xlabel="object x (mm)", ylabel="object y (mm)")
    fig.colorbar(plot, ax=axes, shrink=.7, label="relative activity concentration")
    fig.suptitle(f"Axial truth at z={z[k]:+.1f} mm; each channel 1:10, summed image background 2 and spheres 10")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_3d(outer_outline, centers, output):
    fig = plt.figure(figsize=(9, 7), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    zlow, zhigh = -30, 30
    boundary = outer_outline[::12]
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
        ax.plot_surface(sx, sy, sz, color="#c84b40" if sphere["hot_energy_keV"] == 218 else "#4384b5",
                        alpha=.85, linewidth=0, shade=True)
    t = np.linspace(0, 2*np.pi, 90)
    for level in (zlow, zhigh):
        ax.plot(25.5*np.cos(t), 25.5*np.sin(t), level, color="#378f9a", lw=1.6)
    ax.set(xlim=(-160, 160), ylim=(-120, 120), zlim=(-40, 40),
           xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)",
           title="60-mm NEMA body: three 218-keV and three 440-keV hot spheres")
    ax.view_init(elev=22, azim=-65)
    ax.set_box_aspect((320, 240, 80))
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main():
    config = json.loads(CONFIG.read_text())
    experiment = json.loads(EXPERIMENT.read_text())
    x, y, z, body, wall, lung, spheres, activity, centers, inner_outline, outer_outline = make_truth(config, experiment)
    GENERATED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    truth_path = GENERATED / "truth_3mm.npz"
    np.savez_compressed(truth_path, x_mm=x, y_mm=y, z_mm=z,
                        body_fraction_zyx=body, wall_fraction_zyx=wall,
                        lung_fraction_zyx=lung,
                        activity_218_zyx=activity[218],
                        activity_440_zyx=activity[440],
                        **{f"sphere_{diameter}_fraction_zyx": array
                           for diameter, array in spheres.items()})
    plot_layout(config, inner_outline, outer_outline, centers, REPORTS / "layout_in_ellipse.png")
    plot_slices(x, y, z, activity, REPORTS / "activity_slices.png")
    plot_axial_comparison(x, y, z, activity, REPORTS / "energy_axial_comparison.png")
    plot_3d(outer_outline, centers, REPORTS / "geometry_3d.png")
    spacing = config["truth_spacing_mm"]
    volume_voxel = spacing ** 3
    arcs = config["body_arc_geometry_mm"]
    offset = arcs["bottom_arc_center_offset_x"]
    inner_top = arcs["top_inner_radius"]
    inner_corner = arcs["bottom_inner_radius"]
    wall_mm = arcs["wall_thickness"]
    analytic_inner = (.5 * math.pi * (inner_top**2 + inner_corner**2)
                      + 2 * offset * inner_corner) * config["body_height_mm"]
    analytic_outer = (.5 * math.pi * ((inner_top+wall_mm)**2 + (inner_corner+wall_mm)**2)
                      + 2 * offset * (inner_corner+wall_mm)) * config["body_height_mm"]
    sampled_inner = float(body.sum(dtype=np.float64) * volume_voxel)
    sampled_wall = float(wall.sum(dtype=np.float64) * volume_voxel)
    if (abs(sampled_inner / analytic_inner - 1) > .001 or
        abs(sampled_wall / (analytic_outer-analytic_inner) - 1) > .001):
        raise ValueError("NEMA circular-arc volume differs from analytic geometry by >0.1%")
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
              "body_outer_dimensions_mm": [config["body_outer_width_mm"], config["body_outer_ap_mm"], config["body_height_mm"]],
              "body_arc_geometry_mm": config["body_arc_geometry_mm"],
              "physical_fov_mm": [500, 300, 120],
              "fov_outer_extent_margin_mm": [100, 35, 30],
              "body_volume_mm3": sampled_inner,
              "body_analytic_volume_mm3": analytic_inner,
              "body_relative_volume_error": sampled_inner / analytic_inner - 1,
              "wall_volume_mm3": sampled_wall,
              "wall_analytic_volume_mm3": analytic_outer-analytic_inner,
              "wall_relative_volume_error": sampled_wall / (analytic_outer-analytic_inner) - 1,
              "lung_outer_volume_mm3": float(lung.sum(dtype=np.float64) * volume_voxel),
              "spheres": sphere_metrics,
              "relative_activity_concentration": config["relative_activity_concentration"],
              "hot_sphere_diameters_by_energy_keV": {
                  "218": [10, 17, 28], "440": [13, 22, 37]},
              "relative_activity_integral_mm3": {
                  str(energy): float(array.sum(dtype=np.float64) * volume_voxel)
                  for energy, array in activity.items()},
              "truth_sha256": sha256(truth_path),
              "configuration_sha256": sha256(CONFIG),
              "preview_sha256": {name: sha256(REPORTS / name)
                                  for name in ("layout_in_ellipse.png", "activity_slices.png",
                                               "energy_axial_comparison.png", "geometry_3d.png")},
              "material_attenuation": False,
              "standard_compliance": "cross-section from NEMA NU 2-2007 Fig 7-1; axial length and dual-energy filling are research adaptations"}
    (REPORTS / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"truth": str(truth_path), "report": str(REPORTS),
                      "shape": report["shape_zyx"], "max_sphere_volume_error":
                      max(abs(row["relative_volume_error"]) for row in sphere_metrics)}, indent=2))


if __name__ == "__main__":
    main()
