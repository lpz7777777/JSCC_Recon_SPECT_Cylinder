"""Independent local-background hot-rod metrics for a verified formal result."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

HERE = Path(__file__).resolve().parent
ROOT = HERE / "generated"
CHANNELS = {218: "218_SinglePhoton_CrossTalkCorrected", 440: "440_SinglePlusCompton"}
GROUPS = ("center", "long_plus", "long_minus", "short_plus", "short_minus")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_name")
    parser.add_argument("--iteration", type=int, default=10000)
    args = parser.parse_args()
    integrity = json.loads((HERE / "reports" / f"{args.result_name}_integrity.json").read_text())
    if integrity["dataset"] != "EllipseContrast" or integrity["job_result"] != args.result_name:
        raise ValueError("A verified EllipseContrast formal result is required")
    if args.iteration not in (50, 500, 1000, 3000, 10000):
        raise ValueError("Selected history frame not available")
    truth_path = ROOT / "Simulation/Contrast_truth.json"
    truth = json.loads(truth_path.read_text())
    rods = truth["rods"]
    with np.load(ROOT / "Geometry/geometry.npz") as geometry:
        polar_coordinates = geometry["coordinates_mm"]
    xy = polar_coordinates[:3301, :2]
    z = polar_coordinates[::3301, 2]
    if len(z) != 40 or len(rods) != 30:
        raise ValueError("Unexpected geometry or rod count")
    x = np.arange(-249., 250., 3.)
    y = np.arange(-150., 151., 3.)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    query = np.column_stack((xx.ravel(), yy.ravel()))
    ellipse = (xx / 250.) ** 2 + (yy / 150.) ** 2 <= 1
    tri = Delaunay(xy)
    report = {"result": args.result_name, "iteration": args.iteration,
              "truth_sha256": digest(truth_path),
              "method": "3-mm XY linear interpolation; local background within 55 mm of each group center; no smoothing; CRC=(rod/background-1)/5",
              "rows": []}
    for energy, channel in CHANNELS.items():
        if args.iteration == 10000:
            image_path = ROOT / "RemoteResults" / args.result_name / f"Image_{channel}_full.float32"
            expected = next(item["sha256"]["full"] for item in integrity["outputs"]
                            if item["channel"] == channel)
            if digest(image_path) != expected:
                raise ValueError(f"Image checksum mismatch: {channel}")
        else:
            image_path = ROOT / "HistorySelected" / args.result_name / f"Image_{channel}_iter{args.iteration:05d}_full.float32"
            source = json.loads((image_path.parent / "source_manifest.json").read_text())
            expected = next(item["sha256"]["history"] for item in integrity["outputs"]
                            if item["channel"] == channel)
            if source["source_history_sha256"][channel] != expected:
                raise ValueError(f"History source mismatch: {channel}")
        polar = np.memmap(image_path, dtype="<f4", mode="r", shape=(40, 3301))
        for group in GROUPS:
            same_group = [rod for rod in rods if rod["group"] == group]
            cx = float(np.mean([rod["center_mm"][0] for rod in same_group]))
            cy = float(np.mean([rod["center_mm"][1] for rod in same_group]))
            cz = same_group[0]["center_mm"][2]
            slices = np.where(np.abs(z - cz) <= 13.5)[0]
            slab = np.stack([LinearNDInterpolator(tri, polar[k], fill_value=0)(query)
                             .reshape(yy.shape) for k in slices])
            slab = np.maximum(slab, 0)
            sampled_truth = np.ones_like(xy[:, 0])
            for rod in same_group:
                if rod["energy"] != energy:
                    continue
                rx, ry, _ = rod["center_mm"]
                sampled_truth[((xy[:, 0] - rx) ** 2 + (xy[:, 1] - ry) ** 2)
                              <= rod["radius_mm"] ** 2] += rod["excess_activity"]
            synthetic = LinearNDInterpolator(tri, sampled_truth, fill_value=0)(query).reshape(yy.shape)
            local = ellipse & ((xx - cx) ** 2 + (yy - cy) ** 2 <= 55 ** 2)
            for rod in same_group:
                rx, ry, _ = rod["center_mm"]
                local &= (xx - rx) ** 2 + (yy - ry) ** 2 > (rod["radius_mm"] + 3) ** 2
            background = slab[:, local].ravel()
            if background.size < 100 or background.mean() <= 0:
                raise ValueError(f"Insufficient local background: {energy}/{group}")
            bg_mean = float(background.mean())
            bg_std = float(background.std())
            synthetic_bg = float(synthetic[local].mean())
            for rod in same_group:
                if rod["energy"] != energy:
                    continue
                rx, ry, _ = rod["center_mm"]
                mask = (xx - rx) ** 2 + (yy - ry) ** 2 <= rod["radius_mm"] ** 2
                values = slab[:, mask]
                if values.size < 20:
                    raise ValueError(f"Too few rod samples: {energy}/{group}/{rod['radius_mm']}")
                rod_mean = float(values.mean())
                synthetic_rod = float(synthetic[mask].mean())
                report["rows"].append({"group": group, "energy_keV": energy,
                    "diameter_mm": 2 * rod["radius_mm"], "rod_voxels": int(values.size),
                    "background_voxels": int(background.size),
                    "rod_mean": rod_mean, "background_mean": bg_mean,
                    "background_cv": bg_std / bg_mean,
                    "crc": (rod_mean / bg_mean - 1) / rod["excess_activity"],
                    "sampled_truth_crc": (synthetic_rod / synthetic_bg - 1) /
                    rod["excess_activity"],
                    "cnr": (rod_mean - bg_mean) / bg_std if bg_std > 0 else None})
    suffix = "contrast_metrics" if args.iteration == 10000 else f"contrast_iter{args.iteration:05d}"
    out = HERE / "reports" / f"{args.result_name}_{suffix}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, energy in zip(axes, CHANNELS):
        for group in GROUPS:
            rows = [row for row in report["rows"] if row["energy_keV"] == energy and row["group"] == group]
            ax.plot([row["diameter_mm"] for row in rows], [row["crc"] for row in rows],
                    marker="o", label=group)
        ax.axhline(1, color="black", linestyle="--", linewidth=.8)
        ax.set(title=f"{energy} keV hot rods", xlabel="diameter (mm)", ylabel="CRC")
        ax.legend(fontsize=7)
    fig.suptitle(f"EllipseContrast 10⁹ primaries, {args.iteration} iterations; raw local contrast")
    figure = out.with_suffix(".png")
    fig.savefig(figure, dpi=170)
    print(out)
    print(figure)
    for group in GROUPS:
        selected = [row for row in report["rows"] if row["group"] == group]
        print(group, "median CRC", round(float(np.median([row["crc"] for row in selected])), 3),
              "positive rods", sum(row["crc"] > 0 for row in selected), "/", len(selected))


if __name__ == "__main__":
    main()
