"""Compare verified NEMA dual-energy reconstruction with its 3-mm source truth.

The output reports relative spatial contrast, not absolute 225Ac activity.
No smoothing is applied. Polar reconstructions are linearly interpolated in XY
onto the saved truth grid at matching z centers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"
REPORTS = HERE / "reports/NEMA_Body_H60"
CHANNELS = {
    218: "218_SinglePhoton_CrossTalkCorrected",
    440: "440_SinglePlusCompton",
}
HOT = {218: (10, 17, 28), 440: (13, 22, 37)}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def xy_interpolator(polar_xy, x, y):
    """Return a callable that interpolates [z, polar_point] arrays to [z,y,x]."""
    tri = Delaunay(polar_xy)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    query = np.column_stack((xx.ravel(), yy.ravel()))
    simplex = tri.find_simplex(query)
    valid = simplex >= 0
    indices = np.zeros((len(query), 3), dtype=np.int64)
    weights = np.zeros((len(query), 3), dtype=np.float64)
    indices[valid] = tri.simplices[simplex[valid]]
    transform = tri.transform[simplex[valid]]
    bary = np.einsum("nij,nj->ni", transform[:, :2, :],
                    query[valid]-transform[:, 2, :])
    weights[valid, :2] = bary
    weights[valid, 2] = 1-bary.sum(axis=1)

    def apply(values):
        if values.ndim != 2 or values.shape[1] != len(polar_xy):
            raise ValueError("Wrong polar image shape")
        sampled = np.einsum("zqc,qc->zq", values[:, indices], weights,
                            optimize=True)
        return sampled.reshape((values.shape[0], len(y), len(x))).astype(np.float32)

    return apply


def weighted_mean(values, weight):
    denominator = float(weight.sum(dtype=np.float64))
    if denominator <= 0:
        raise ValueError("Empty NEMA ROI")
    return float(np.sum(values*weight, dtype=np.float64)/denominator)


def analyze(result_name):
    if "/" in result_name or "\\" in result_name:
        raise ValueError("Result name must be one directory")
    integrity_path = HERE / "reports" / f"{result_name}_integrity.json"
    integrity = json.loads(integrity_path.read_text())
    if integrity["dataset"] != "NEMA_Body_H60" or integrity["job_result"] != result_name:
        raise ValueError("A verified NEMA formal result is required")
    truth_path = GENERATED / "NEMA_Body_H60/truth_3mm.npz"
    manifest = json.loads((REPORTS / "manifest.json").read_text())
    if digest(truth_path) != manifest["truth_sha256"]:
        raise ValueError("NEMA source truth hash changed")
    with np.load(truth_path) as truth_file:
        x, y, z = (truth_file[f"{axis}_mm"].copy() for axis in "xyz")
        truth = {energy: truth_file[f"activity_{energy}_zyx"].copy()
                 for energy in (218, 440)}
        body = truth_file["body_fraction_zyx"].copy()
        lung = truth_file["lung_fraction_zyx"].copy()
        spheres = {diameter: truth_file[f"sphere_{diameter}_fraction_zyx"].copy()
                   for diameter in (10, 13, 17, 22, 28, 37)}
    with np.load(GENERATED / "Geometry/geometry.npz") as geometry:
        coords = geometry["coordinates_mm"]
    if coords.shape != (132040, 3) or len(z) != 40:
        raise ValueError("NEMA and polar grid mismatch")
    polar_z = coords[::3301, 2]
    if not np.allclose(polar_z, z, atol=1e-6):
        raise ValueError("NEMA and polar z centers differ")
    interpolate = xy_interpolator(coords[:3301, :2], x, y)
    all_spheres = sum(spheres.values())
    background_fraction = np.maximum(body-lung-all_spheres, 0)
    pure_background = (background_fraction >= .99) & (np.abs(z[:, None, None]) <= 25.5)
    report = {"result": result_name, "truth_sha256": digest(truth_path),
              "integrity_sha256": digest(integrity_path),
              "method": "3-mm XY barycentric interpolation, matching z centers, no smoothing; CRC=(sphere/background-1)/9; relative channel scaling by pure-background mean",
              "channels": {}, "spheres": []}
    images = {}
    backgrounds = {}
    for energy, channel in CHANNELS.items():
        path = GENERATED / "RemoteResults" / result_name / f"Image_{channel}_full.float32"
        expected = next(row["sha256"]["full"] for row in integrity["outputs"]
                        if row["channel"] == channel)
        if digest(path) != expected:
            raise ValueError(f"Reconstruction SHA-256 mismatch: {channel}")
        polar = np.memmap(path, mode="r", dtype="<f4", shape=(40, 3301))
        reconstructed = np.maximum(interpolate(polar), 0)
        if not np.isfinite(reconstructed).all():
            raise ValueError(f"Nonfinite NEMA image: {channel}")
        mean = float(reconstructed[pure_background].mean())
        if mean <= 0:
            raise ValueError(f"Empty background reconstruction: {channel}")
        images[energy] = reconstructed / mean
        backgrounds[energy] = mean
        report["channels"][str(energy)] = {"channel": channel,
            "background_mean_original_units": mean,
            "background_cv": float(reconstructed[pure_background].std()/mean),
            "image_sha256": expected}
        xx, yy = np.meshgrid(x, y, indexing="xy")
        for diameter in HOT[energy]:
            sphere = spheres[diameter]
            center = next(row["center_mm"] for row in manifest["spheres"]
                          if int(row["diameter_mm"]) == diameter)
            cx, cy, cz = center
            radius = diameter/2
            local_xy = (xx-cx)**2+(yy-cy)**2 <= (radius+25)**2
            local_z = np.abs(z-cz) <= radius+3
            local = local_z[:, None, None] & local_xy[None, :, :]
            bg_mask = local & (background_fraction >= .99)
            bg_values = reconstructed[bg_mask]
            if bg_values.size < 50 or bg_values.mean() <= 0:
                raise ValueError(f"Insufficient local NEMA background: {energy}/{diameter}")
            hot_mean = weighted_mean(reconstructed, sphere)
            background_mean = float(bg_values.mean())
            background_std = float(bg_values.std())
            sampled_truth_hot = weighted_mean(truth[energy], sphere)
            report["spheres"].append({"energy_keV": energy,
                "diameter_mm": diameter, "center_mm": center,
                "sphere_effective_voxels": float(sphere.sum()),
                "background_voxels": int(bg_values.size),
                "hot_mean": hot_mean, "local_background_mean": background_mean,
                "local_background_cv": background_std/background_mean,
                "crc": (hot_mean/background_mean-1)/9,
                "cnr": (hot_mean-background_mean)/background_std
                       if background_std > 0 else None,
                "truth_sampled_crc": (sampled_truth_hot-1)/9})
    REPORTS.mkdir(exist_ok=True)
    report_path = REPORTS / f"{result_name}_spatial.json"
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    index = int(np.argmin(np.abs(z)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    extent = (x[0]-1.5, x[-1]+1.5, y[0]-1.5, y[-1]+1.5)
    for row, energy in enumerate((218, 440)):
        for col, (image, title) in enumerate(((truth[energy][index], "source truth"),
                                               (images[energy][index], "reconstruction / background"))):
            ax = axes[row, col]
            im = ax.imshow(image, extent=extent, origin="lower", vmin=0, vmax=10,
                           cmap="gray_r", interpolation="nearest")
            ax.set(xlim=(-160, 160), ylim=(-120, 120), aspect="equal",
                   title=f"{energy} keV {title}", xlabel="object x (mm)", ylabel="object y (mm)")
    fig.colorbar(im, ax=axes, shrink=.7, label="relative concentration / recovery")
    fig.suptitle(f"NEMA H60, z={z[index]:+.1f} mm; 1e9 primaries, 10000 MLEM iterations")
    figure_path = REPORTS / f"{result_name}_truth_vs_recon.png"
    fig.savefig(figure_path, dpi=175)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, energy in zip(axes, (218, 440)):
        rows = [row for row in report["spheres"] if row["energy_keV"] == energy]
        ax.plot([row["diameter_mm"] for row in rows], [row["crc"] for row in rows],
                "o-", label="reconstruction")
        ax.plot([row["diameter_mm"] for row in rows],
                [row["truth_sampled_crc"] for row in rows], "s--",
                label="sampled truth")
        ax.axhline(1, color="black", linewidth=.8, linestyle=":")
        ax.set(title=f"{energy} keV hot spheres", xlabel="sphere diameter (mm)",
               ylabel="CRC", ylim=(-.3, 1.3))
        ax.legend(fontsize=8)
    curve_path = REPORTS / f"{result_name}_crc.png"
    fig.savefig(curve_path, dpi=175)
    plt.close(fig)
    return report_path, figure_path, curve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_name")
    args = parser.parse_args()
    print(*(str(path) for path in analyze(args.result_name)), sep="\n")


if __name__ == "__main__":
    main()
