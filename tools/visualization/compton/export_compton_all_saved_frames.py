"""Export the all-saved-frame Compton gallery with a selectable grayscale map."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualize_jscc_compton_iteration_evolution import project


ROWS = [
    ("440_SinglePhoton", "center_mean", "440 single-photon center two slices"),
    ("440_ComptonOnly", "mip", "Compton-only MIP"),
    ("440_ComptonOnly", "center_mean", "Compton-only center two slices"),
    ("440_SinglePlusCompton", "center_mean", "Single + Compton center two slices"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cmap", choices=("gray", "gray_r"), default="gray_r")
    args = parser.parse_args()

    result = args.result_dir.resolve()
    factor = args.factor_dir.resolve()
    manifest = json.loads((result / "run_manifest.json").read_text(encoding="utf-8"))
    iterations_count = int(manifest["iterations"])
    save_step = int(manifest["save_step"])
    frame_count = iterations_count // save_step
    coordinates = np.loadtxt(factor / "coor_polar_full.csv", delimiter=",")
    pixel_count = coordinates.shape[0]
    axis = np.arange(-150.0, 150.01, 3.0)

    histories = {}
    for name, _, _ in ROWS:
        if name in histories:
            continue
        values = np.fromfile(
            result / f"Image_{name}_Iter_{iterations_count}_{frame_count}", dtype=np.float32
        )
        if values.size != frame_count * pixel_count or not np.isfinite(values).all():
            raise ValueError(f"Invalid iteration history for {name}")
        histories[name] = values.reshape(frame_count, pixel_count)

    figure, axes = plt.subplots(
        len(ROWS), frame_count, figsize=(2.15 * frame_count, 8.0), constrained_layout=True,
    )
    for row, (name, projection_mode, label) in enumerate(ROWS):
        maps = [project(coordinates, history, axis, projection_mode) for history in histories[name]]
        finite = np.concatenate([item[np.isfinite(item)] for item in maps])
        # A common row scale must retain the true brightest voxel rather than
        # clipping it at a high percentile; this preserves the full dynamic range.
        vmax = max(float(np.max(finite)), 1e-12)
        for column, image in enumerate(maps):
            shown = axes[row, column].imshow(
                image, origin="lower", extent=(-150, 150, -150, 150),
                cmap=args.cmap, vmin=0, vmax=vmax,
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].set_title(str((column + 1) * save_step), fontsize=8)
            if column == 0:
                axes[row, column].set_ylabel(label, fontsize=9)
        figure.colorbar(shown, ax=axes[row, :], fraction=0.008, pad=0.01, label="activity density")

    figure.suptitle(f"{manifest['count_level']} Geant4: every saved Compton iteration")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
