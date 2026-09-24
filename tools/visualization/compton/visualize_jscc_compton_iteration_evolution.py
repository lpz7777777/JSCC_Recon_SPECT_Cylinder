"""Render selected iteration snapshots for the 1e9 Compton validation."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def cartesian_stack(coordinates, values, axis):
    xx, yy = np.meshgrid(axis, axis)
    planes = []
    for z in np.unique(coordinates[:, 2]):
        selected = np.isclose(coordinates[:, 2], z)
        planes.append(griddata(coordinates[selected, :2], values[selected], (xx, yy), method="linear"))
    return np.stack(planes)


def project(coordinates, values, axis, mode):
    stack = cartesian_stack(coordinates, values, axis)
    valid = np.isfinite(stack)
    if mode == "mip":
        # Boundary detector layers can contain isolated numerical/physical
        # outliers. Keep them in z-mean views, but exclude the outer two
        # layers at each end from every MIP.
        mip_stack = stack[2:-2]
        mip_valid = np.isfinite(mip_stack)
        result = np.max(np.where(mip_valid, mip_stack, -np.inf), axis=0)
    elif mode == "z_mean":
        result = np.nanmean(stack, axis=0)
    elif mode == "center_mean":
        z_values = np.unique(coordinates[:, 2])
        center_ids = np.argsort(np.abs(z_values))[:2]
        result = np.nanmean(stack[center_ids], axis=0)
    else:
        raise ValueError(f"Unknown projection mode: {mode}")
    result[~np.any(valid, axis=0)] = np.nan
    return result


def mip(coordinates, values, axis):
    return project(coordinates, values, axis, "mip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("Results/Reconstruction/JSCC_ComptonValidation_Geant4_1e9_Iter1000"))
    parser.add_argument("--factor-dir", type=Path, default=Path("Factors/440keV_RotateNum20"))
    args = parser.parse_args()
    result = args.result_dir.resolve()
    factor = args.factor_dir.resolve()
    manifest_path = result / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    count_level = str(manifest.get("count_level", "unknown"))
    iterations_count = int(manifest.get("iterations", 1000))
    save_step = int(manifest.get("save_step", 50))
    coordinates = np.loadtxt(factor / "coor_polar_full.csv", delimiter=",")
    pixel_count = coordinates.shape[0]
    history_count = iterations_count // save_step
    iterations = np.arange(save_step, iterations_count + 1, save_step)
    selected_ids = [0, 1, 3, 7, 11, 15, 19]
    selected_iterations = iterations[selected_ids]
    axis = np.arange(-150.0, 150.01, 3.0)
    modes = [("440_ComptonOnly", "440 Compton-only"), ("440_SinglePlusCompton", "440 single + Compton")]
    metrics = {}
    figure, axes = plt.subplots(2, len(selected_ids), figsize=(3.0 * len(selected_ids), 6), constrained_layout=True)
    for row, (name, label) in enumerate(modes):
        path = result / f"Image_{name}_Iter_{iterations_count}_{history_count}"
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
    figure.suptitle(f"{count_level} Geant4: Compton reconstruction evolution (shared p99 scale per row)")
    output = result / "Compton_iteration_evolution_50_to_1000.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)

    diagnostic_rows = [
        ("440_ComptonOnly", "mip", "Compton-only MIP"),
        ("440_ComptonOnly", "z_mean", "Compton-only z mean"),
        ("440_ComptonOnly", "center_mean", "Compton-only center two slices"),
        ("440_SinglePlusCompton", "z_mean", "Single + Compton z mean"),
    ]
    diagnostic, diagnostic_axes = plt.subplots(
        len(diagnostic_rows), len(selected_ids),
        figsize=(3.0 * len(selected_ids), 3.0 * len(diagnostic_rows)),
        constrained_layout=True,
    )
    history_cache = {
        name: np.fromfile(
            result / f"Image_{name}_Iter_{iterations_count}_{history_count}", dtype=np.float32
        ).reshape(history_count, pixel_count)
        for name in ("440_SinglePhoton", "440_ComptonOnly", "440_SinglePlusCompton")
    }
    for row, (name, projection_mode, label) in enumerate(diagnostic_rows):
        projected = [
            project(coordinates, history_cache[name][history_id], axis, projection_mode)
            for history_id in selected_ids
        ]
        scale_values = np.concatenate([item[np.isfinite(item)] for item in projected])
        vmax = float(np.quantile(scale_values, 0.99))
        for column, (history_id, image) in enumerate(zip(selected_ids, projected)):
            ax = diagnostic_axes[row, column]
            shown = ax.imshow(
                image, origin="lower", extent=(-150, 150, -150, 150),
                cmap="gray", vmin=0, vmax=max(vmax, 1e-12),
            )
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(str(selected_iterations[column]), fontsize=9)
            if column == 0:
                ax.set_ylabel(label, fontsize=10)
        diagnostic.colorbar(
            shown, ax=diagnostic_axes[row, :], fraction=0.017, pad=0.01,
            label="activity density",
        )
    diagnostic.suptitle(f"{count_level} Geant4: projection-dependent Compton evolution")
    diagnostic_output = result / "Compton_iteration_evolution_MIP_zmean_center.png"
    diagnostic.savefig(diagnostic_output, dpi=180)
    plt.close(diagnostic)

    all_ids = list(range(history_count))
    full_rows = [
        ("440_SinglePhoton", "center_mean", "440 single-photon center two slices"),
        ("440_ComptonOnly", "mip", "Compton-only MIP"),
        ("440_ComptonOnly", "center_mean", "Compton-only center two slices"),
        ("440_SinglePlusCompton", "center_mean", "Single + Compton center two slices"),
    ]
    full_figure, full_axes = plt.subplots(
        len(full_rows), history_count,
        figsize=(2.15 * history_count, 8.0), constrained_layout=True,
    )
    for row, (name, projection_mode, label) in enumerate(full_rows):
        projected = [
            project(coordinates, history_cache[name][history_id], axis, projection_mode)
            for history_id in all_ids
        ]
        scale_values = np.concatenate([item[np.isfinite(item)] for item in projected])
        vmax = float(np.quantile(scale_values, 0.99))
        for column, (history_id, image) in enumerate(zip(all_ids, projected)):
            ax = full_axes[row, column]
            shown = ax.imshow(
                image, origin="lower", extent=(-150, 150, -150, 150),
                cmap="gray", vmin=0, vmax=max(vmax, 1e-12),
            )
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(str(int(iterations[history_id])), fontsize=8)
            if column == 0:
                ax.set_ylabel(label, fontsize=9)
        full_figure.colorbar(
            shown, ax=full_axes[row, :], fraction=0.008, pad=0.01,
            label="activity density",
        )
    full_figure.suptitle(f"{count_level} Geant4: every saved Compton iteration")
    full_output = result / "Compton_iteration_evolution_all_saved_frames.png"
    full_figure.savefig(full_output, dpi=180)
    plt.close(full_figure)

    summary = {
        "count_level": count_level,
        "iterations": iterations_count,
        "save_step": save_step,
        "selected_iterations": selected_iterations.tolist(),
        "metrics": metrics,
        "figure": str(output),
        "projection_diagnostic_figure": str(diagnostic_output),
        "all_saved_frames_figure": str(full_output),
    }
    (result / "compton_iteration_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
