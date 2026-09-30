"""Full-FOV six-channel truth-versus-iteration galleries for four 1e9 runs."""
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
RESULTS = ("CircleNewDist_1e9_1640673", "EllipseUniform_1e9_1640929",
           "EllipseContrast_1e9_1641013", "XCAT_1e9_1641014")
ITERATIONS = (50, 500, 1000, 3000, 5000, 10000)
CHANNELS = (
    ("440_SinglePhoton", "440 single", "bi"),
    ("440_ComptonOnly", "440 Compton", "bi"),
    ("440_SinglePlusCompton", "440 JSCC", "bi"),
    ("218_SinglePhoton_CrossTalkCorrected", "218 corrected", "fr"),
    ("440SinglePlus218Single", "440 single + 218", "sum"),
    ("440SingleComptonPlus218Single", "440 JSCC + 218", "sum"),
)
SLICES = {"CircleNewDist": ((20, "center"), (0, "lower_edge"), (39, "upper_edge")),
          "EllipseUniform": ((20, "center"), (0, "lower_edge"), (39, "upper_edge")),
          "EllipseContrast": ((20, "center"), (4, "lower_rods"), (35, "upper_rods")),
          "XCAT": ((20, "center"), (4, "lower"), (35, "upper"))}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def truth_slice(dataset, index, x, y, z):
    xx, yy = np.meshgrid(x, y, indexing="xy")
    if dataset == "CircleNewDist":
        source = ((xx ** 2 + yy ** 2) <= 150 ** 2).astype(np.float64)
        return {"fr": source, "bi": source, "sum": 2 * source}
    ellipse = ((xx / 250) ** 2 + (yy / 150) ** 2 <= 1).astype(np.float64)
    if dataset == "EllipseUniform":
        return {"fr": ellipse, "bi": ellipse, "sum": 2 * ellipse}
    if dataset == "EllipseContrast":
        rods = json.loads((ROOT / "Simulation/Contrast_truth.json").read_text())["rods"]
        output = {"fr": ellipse.copy(), "bi": ellipse.copy()}
        for rod in rods:
            rx, ry, rz = rod["center_mm"]
            if abs(z - rz) >= rod["height_mm"] / 2:
                continue
            mask = (xx - rx) ** 2 + (yy - ry) ** 2 <= rod["radius_mm"] ** 2
            output["fr" if rod["energy"] == 218 else "bi"][mask] += rod["excess_activity"]
        output["sum"] = output["fr"] + output["bi"]
        return output
    if dataset == "XCAT":
        with np.load(ROOT / "XCAT_1e9_fullx/truth_3mm.npz") as data:
            if (not np.array_equal(data["x_mm"], x) or
                not np.array_equal(data["y_mm"], y) or
                abs(data["z_mm"][index] - z) > 1e-8):
                raise ValueError("XCAT truth coordinate mismatch")
            fr = np.asarray(data["fr_zyx"][index], dtype=np.float64)
            bi = np.asarray(data["bi_zyx"][index], dtype=np.float64)
        return {"fr": fr, "bi": bi, "sum": fr + bi}
    raise ValueError(dataset)


def main():
    output_dir = HERE / "reports/iteration_galleries"
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / "Geometry/geometry.npz") as geometry:
        coordinates = geometry["coordinates_mm"]
    z_axis = coordinates[::3301, 2]
    x = np.arange(-250.5, 250.6, 3.)
    y = np.arange(-148.5, 148.6, 3.)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    query = np.column_stack((xx.ravel(), yy.ravel()))
    tri = Delaunay(coordinates[:3301, :2])
    extent = (x[0] - 1.5, x[-1] + 1.5, y[0] - 1.5, y[-1] + 1.5)
    all_reports = []
    for result in RESULTS:
        integrity = json.loads((HERE / "reports" / f"{result}_integrity.json").read_text())
        dataset = integrity["dataset"]
        source_manifest = json.loads((ROOT / "GalleryFrames" / result /
                                      "selected_frames_manifest.json").read_text())
        if (source_manifest["result"] != result or
            tuple(source_manifest["iterations"]) != ITERATIONS or
            set(source_manifest["channels"]) != {row[0] for row in CHANNELS}):
            raise ValueError(f"Incomplete selected frames: {result}")
        for index, label in SLICES[dataset]:
            z = float(z_axis[index])
            truth = truth_slice(dataset, index, x, y, z)
            figure, axes = plt.subplots(len(CHANNELS), len(ITERATIONS) + 1,
                                        figsize=(21, 15), layout="constrained")
            metadata = {"result": result, "dataset": dataset, "z_mm": z,
                        "slice_label": label, "iterations": ITERATIONS,
                        "display": "gray_r; full FOV; no smoothing; independent panel p99.5 scaling",
                        "truth_source": "Geant4 analytic source geometry" if dataset != "XCAT" else
                        "generated/XCAT_1e9_fullx/truth_3mm.npz",
                        "channels": {}}
            for row, (channel, short, truth_key) in enumerate(CHANNELS):
                reference = truth[truth_key]
                truth_max = float(np.percentile(reference[reference > 0], 99.5)) if np.any(reference > 0) else 1.
                axes[row, 0].imshow(reference, cmap="gray_r", origin="lower",
                                    extent=extent, vmin=0, vmax=truth_max,
                                    interpolation="nearest", aspect="equal")
                axes[row, 0].set_ylabel(f"{short}\ny (mm)", fontsize=9)
                if row == 0:
                    axes[row, 0].set_title("truth", fontsize=10)
                metadata["channels"][channel] = []
                for col, iteration in enumerate(ITERATIONS, start=1):
                    file = (ROOT / "GalleryFrames" / result /
                            f"Image_{channel}_iter{iteration:05d}_full.float32")
                    expected = source_manifest["channels"][channel]["frames_sha256"][str(iteration)]
                    if file.stat().st_size != 132040 * 4 or digest(file) != expected:
                        raise ValueError(f"Selected frame mismatch: {file}")
                    polar = np.memmap(file, mode="r", dtype="<f4", shape=(40, 3301))[index]
                    cart = LinearNDInterpolator(tri, polar, fill_value=0)(query).reshape(yy.shape)
                    cart = np.maximum(cart, 0)
                    if dataset == "CircleNewDist":
                        cart[xx ** 2 + yy ** 2 > 150 ** 2] = 0
                    else:
                        cart[(xx / 250) ** 2 + (yy / 150) ** 2 > 1] = 0
                    positive = cart[cart > 0]
                    vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.
                    axes[row, col].imshow(cart, cmap="gray_r", origin="lower",
                                          extent=extent, vmin=0, vmax=max(vmax, 1e-20),
                                          interpolation="nearest", aspect="equal")
                    if row == 0:
                        axes[row, col].set_title(str(iteration), fontsize=10)
                    metadata["channels"][channel].append({"iteration": iteration,
                        "frame_sha256": expected, "panel_p99_5": vmax,
                        "slice_sum": float(cart.sum(dtype=np.float64)),
                        "slice_max": float(cart.max())})
                for ax in axes[row]:
                    ax.set_xticks((-200, 0, 200))
                    ax.set_yticks((-100, 0, 100))
                    ax.tick_params(labelsize=7)
            for ax in axes[-1]:
                ax.set_xlabel("x (mm)", fontsize=8)
            figure.suptitle(f"{dataset}: z={z:+.1f} mm, 10⁹ primaries, six channels\n"
                             "Each panel clipped at its own p99.5; raw reconstruction, no smoothing",
                             fontsize=14)
            output = output_dir / f"{result}_{label}.png"
            figure.savefig(output, dpi=130)
            plt.close(figure)
            metadata_path = output.with_suffix(".json")
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            all_reports.append({"dataset": dataset, "result": result, "slice": label,
                                "z_mm": z, "figure": output.name,
                                "metadata": metadata_path.name})
            print(output, flush=True)
    (output_dir / "index.json").write_text(json.dumps(all_reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
