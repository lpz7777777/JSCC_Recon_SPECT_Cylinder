"""Render selected iteration snapshots for the 1e9 Compton validation."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def mip(coordinates, values, axis):
    xx, yy = np.meshgrid(axis, axis)
    planes = []
    for z in np.unique(coordinates[:, 2]):
        selected = np.isclose(coordinates[:, 2], z)
        planes.append(griddata(coordinates[selected, :2], values[selected], (xx, yy), method="linear"))
    stack = np.stack(planes)
    valid = np.isfinite(stack)
    result = np.max(np.where(valid, stack, -np.inf), axis=0)
    result[~np.any(valid, axis=0)] = np.nan
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("Results/Reconstruction/JSCC_ComptonValidation_Geant4_1e9_Iter1000"))
    parser.add_argument("--factor-dir", type=Path, default=Path("Factors/440keV_RotateNum20"))
    args = parser.parse_args()
    result = args.result_dir.resolve()
    factor = args.factor_dir.resolve()
    coordinates = np.loadtxt(factor / "coor_polar_full.csv", delimiter=",")
    pixel_count = coordinates.shape[0]
    history_count = 20
    iterations = np.arange(50, 1001, 50)
    selected_ids = [0, 1, 3, 7, 11, 15, 19]
    selected_iterations = iterations[selected_ids]
    axis = np.arange(-150.0, 150.01, 3.0)
    modes = [("440_ComptonOnly", "440 Compton-only"), ("440_SinglePlusCompton", "440 single + Compton")]
    metrics = {}
    figure, axes = plt.subplots(2, len(selected_ids), figsize=(3.0 * len(selected_ids), 6), constrained_layout=True)
    for row, (name, label) in enumerate(modes):
        path = result / f"Image_{name}_Iter_1000_{history_count}"
        history = np.fromfile(path, dtype=np.float32).reshape(history_count, pixel_count)
        metrics[name] = []
        for index, iteration in enumerate(iterations):
            values = history[index].astype(np.float64)
            metrics[name].append({
                "iteration": int(iteration), "min": float(values.min()), "max": float(values.max()),
                "mean": float(values.mean()), "sum": float(values.sum()),
                "cv": float(values.std() / max(values.mean(), 1e-30)),
                "p99": float(np.quantile(values, 0.99)),
            })
        vmax = max(metrics[name][idx]["p99"] for idx in selected_ids)
        for column, history_id in enumerate(selected_ids):
            values = history[history_id].astype(np.float64)
            image = mip(coordinates, values, axis)
            ax = axes[row, column]
            shown = ax.imshow(image, origin="lower", extent=(-150, 150, -150, 150), cmap="gray", vmin=0, vmax=vmax)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            item = metrics[name][history_id]
            ax.set_title(f"{selected_iterations[column]}\nmax {item['max']:.0f}", fontsize=9)
            if column == 0:
                ax.set_ylabel(label, fontsize=10)
        figure.colorbar(shown, ax=axes[row, :], fraction=0.017, pad=0.01, label="activity density")
    figure.suptitle("1e9 Geant4: Compton reconstruction evolution (shared p99 scale per row)")
    output = result / "Compton_iteration_evolution_50_to_1000.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    summary = {"selected_iterations": selected_iterations.tolist(), "metrics": metrics, "figure": str(output)}
    (result / "compton_iteration_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
