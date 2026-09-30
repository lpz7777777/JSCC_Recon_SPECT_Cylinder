"""Zoomed, unsmoothed hot-rod iteration galleries with true rod outlines."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

HERE = Path(__file__).resolve().parent
ROOT = HERE / "generated"
RESULT = "EllipseContrast_1e9_1641013"
ITERATIONS = (50, 500, 1000, 3000, 5000, 10000)
CHANNELS = ((218, "218_SinglePhoton_CrossTalkCorrected", "218 corrected"),
            (440, "440_SinglePlusCompton", "440 JSCC"))
GROUPS = ("center", "long_plus", "long_minus", "short_plus", "short_minus")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    truth_file = ROOT / "Simulation/Contrast_truth.json"
    rods = json.loads(truth_file.read_text())["rods"]
    source = json.loads((ROOT / "GalleryFrames" / RESULT /
                         "selected_frames_manifest.json").read_text())
    if source["result"] != RESULT or tuple(source["iterations"]) != ITERATIONS:
        raise ValueError("Verified selected frames required")
    with np.load(ROOT / "Geometry/geometry.npz") as data:
        coordinates = data["coordinates_mm"]
    z_axis = coordinates[::3301, 2]
    x = np.arange(-250.5, 250.6, 3.)
    y = np.arange(-148.5, 148.6, 3.)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    query = np.column_stack((xx.ravel(), yy.ravel()))
    tri = Delaunay(coordinates[:3301, :2])
    extent = (x[0] - 1.5, x[-1] + 1.5, y[0] - 1.5, y[-1] + 1.5)
    output_dir = HERE / "reports/iteration_galleries"
    for group in GROUPS:
        group_rods = [rod for rod in rods if rod["group"] == group]
        cx = float(np.mean([rod["center_mm"][0] for rod in group_rods]))
        cy = float(np.mean([rod["center_mm"][1] for rod in group_rods]))
        cz = group_rods[0]["center_mm"][2]
        index = int(np.argmin(abs(z_axis - cz)))
        z = float(z_axis[index])
        fig, axes = plt.subplots(2, 7, figsize=(19, 6), layout="constrained")
        report = {"group": group, "z_mm": z, "truth_sha256": digest(truth_file),
                  "iterations": ITERATIONS, "display": "gray_r; no smoothing; per-panel p99.5; red true rod outlines",
                  "channels": {}}
        for row, (energy, channel, short) in enumerate(CHANNELS):
            truth = np.ones(xx.shape, dtype=np.float64)
            truth[(xx / 250) ** 2 + (yy / 150) ** 2 > 1] = 0
            true_rods = [rod for rod in group_rods if rod["energy"] == energy]
            for rod in true_rods:
                rx, ry, _ = rod["center_mm"]
                truth[(xx - rx) ** 2 + (yy - ry) ** 2 <= rod["radius_mm"] ** 2] += 5
            report["channels"][channel] = []
            for col in range(7):
                ax = axes[row, col]
                if col == 0:
                    data = truth
                    title = "truth"
                else:
                    iteration = ITERATIONS[col - 1]
                    file = (ROOT / "GalleryFrames" / RESULT /
                            f"Image_{channel}_iter{iteration:05d}_full.float32")
                    expected = source["channels"][channel]["frames_sha256"][str(iteration)]
                    if file.stat().st_size != 132040 * 4 or digest(file) != expected:
                        raise ValueError(f"Selected frame mismatch: {file}")
                    polar = np.memmap(file, mode="r", dtype="<f4", shape=(40, 3301))[index]
                    data = LinearNDInterpolator(tri, polar, fill_value=0)(query).reshape(xx.shape)
                    data = np.maximum(data, 0)
                    data[(xx / 250) ** 2 + (yy / 150) ** 2 > 1] = 0
                    title = str(iteration)
                    report["channels"][channel].append({"iteration": iteration,
                        "frame_sha256": expected, "slice_sum": float(data.sum(dtype=np.float64))})
                selected = data[(xx >= cx - 55) & (xx <= cx + 55) &
                                (yy >= cy - 55) & (yy <= cy + 55)]
                positive = selected[selected > 0]
                vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.
                ax.imshow(data, cmap="gray_r", origin="lower", extent=extent,
                          vmin=0, vmax=max(vmax, 1e-20), interpolation="nearest")
                for rod in true_rods:
                    rx, ry, _ = rod["center_mm"]
                    ax.add_patch(Circle((rx, ry), rod["radius_mm"], fill=False,
                                        edgecolor="red", linewidth=.8))
                ax.set_xlim(cx - 55, cx + 55)
                ax.set_ylim(cy - 55, cy + 55)
                ax.set_aspect("equal")
                ax.set_title(title, fontsize=10)
                ax.tick_params(labelsize=7)
                if col == 0:
                    ax.set_ylabel(f"{short}\ny (mm)")
                if row == 1:
                    ax.set_xlabel("x (mm)", fontsize=8)
        fig.suptitle(f"EllipseContrast {group}, z={z:+.1f} mm, 10⁹ primaries\n"
                     "True rods outlined in red; each panel scaled to its own p99.5",
                     fontsize=13)
        output = output_dir / f"{RESULT}_zoom_{group}.png"
        fig.savefig(output, dpi=160)
        plt.close(fig)
        output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(output, flush=True)


if __name__ == "__main__":
    main()
