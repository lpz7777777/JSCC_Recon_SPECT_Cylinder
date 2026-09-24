"""Export five Compton-validation histories as center-39-mm z-MIP galleries."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay


BASE_FILES = {
    "218_corrected_single": "218_SinglePhoton_CrossTalkCorrected",
    "440_single": "440_SinglePhoton",
    "440_single_compton": "440_SinglePlusCompton",
}
ROWS = [
    ("218_corrected_single", "Corrected 218 single-photon"),
    ("440_single", "440 single-photon"),
    ("440_single_compton", "440 single-photon + Compton"),
    ("440_single_plus_218_corrected", "440 single-photon + corrected 218"),
    (
        "440_single_compton_plus_218_corrected",
        "440 single-photon + Compton + corrected 218",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_linear_interpolator(xy: np.ndarray, axis: np.ndarray):
    """Cache a linear triangular interpolation map for all z layers."""
    xx, yy = np.meshgrid(axis, axis)
    queries = np.column_stack((xx.ravel(), yy.ravel()))
    triangulation = Delaunay(xy)
    simplex = triangulation.find_simplex(queries)
    valid = simplex >= 0
    transforms = triangulation.transform[simplex[valid], :2]
    offsets = triangulation.transform[simplex[valid], 2]
    barycentric = np.einsum("nij,nj->ni", transforms, queries[valid] - offsets)
    weights = np.column_stack((barycentric, 1.0 - barycentric.sum(axis=1)))
    return xx.shape, valid, triangulation.simplices[simplex[valid]], weights


def interpolate_plane(values, shape, valid, vertices, weights):
    image = np.full(np.prod(shape), np.nan, dtype=np.float64)
    image[valid] = np.sum(values[vertices] * weights, axis=1)
    return image.reshape(shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = args.result_dir.resolve()
    factor = args.factor_dir.resolve()
    manifest = json.loads((result / "run_manifest.json").read_text(encoding="utf-8"))
    iterations = int(manifest["iterations"])
    save_step = int(manifest["save_step"])
    frames = iterations // save_step
    coordinates = np.loadtxt(factor / "coor_polar_full.csv", delimiter=",")
    z_values = np.unique(coordinates[:, 2])
    z_selected = z_values[np.abs(z_values) <= 19.5 + 1.0e-9]
    selected_indices = [np.flatnonzero(np.isclose(coordinates[:, 2], z)) for z in z_selected]
    xy = coordinates[selected_indices[0], :2]
    if any(not np.array_equal(coordinates[index, :2], xy) for index in selected_indices[1:]):
        raise ValueError("x/y coordinate ordering differs between selected z layers")
    pixel_count = coordinates.shape[0]
    axis = np.arange(-150.0, 150.01, 3.0)
    map_shape, valid, vertices, weights = prepare_linear_interpolator(xy, axis)

    histories = {}
    input_hashes = {}
    for key, image_name in BASE_FILES.items():
        path = result / f"Image_{image_name}_Iter_{iterations}_{frames}"
        values = np.fromfile(path, dtype=np.float32)
        if values.size != frames * pixel_count or not np.isfinite(values).all():
            raise ValueError(f"Invalid iteration history: {path}")
        histories[key] = values.reshape(frames, pixel_count).astype(np.float64)
        input_hashes[path.name] = sha256(path)
    histories["440_single_plus_218_corrected"] = histories["440_single"] + histories["218_corrected_single"]
    histories["440_single_compton_plus_218_corrected"] = (
        histories["440_single_compton"] + histories["218_corrected_single"]
    )

    projected = {}
    for key, _ in ROWS:
        maps = []
        for frame in histories[key]:
            stack = np.stack(
                [interpolate_plane(frame[index], map_shape, valid, vertices, weights) for index in selected_indices]
            )
            finite = np.isfinite(stack)
            mip = np.max(np.where(finite, stack, -np.inf), axis=0)
            mip[~np.any(finite, axis=0)] = np.nan
            maps.append(mip)
        projected[key] = maps

    figure, axes = plt.subplots(
        len(ROWS), frames, figsize=(2.15 * frames, 10.0), constrained_layout=True,
    )
    row_vmax = {}
    for row, (key, label) in enumerate(ROWS):
        finite_values = np.concatenate([image[np.isfinite(image)] for image in projected[key]])
        vmax = max(float(np.max(finite_values)), 1.0e-12)
        row_vmax[key] = vmax
        for column, image in enumerate(projected[key]):
            shown = axes[row, column].imshow(
                image, origin="lower", extent=(-150, 150, -150, 150),
                cmap="gray_r", vmin=0, vmax=vmax,
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].set_title(str((column + 1) * save_step), fontsize=8)
            if column == 0:
                axes[row, column].set_ylabel(label, fontsize=9)
        figure.colorbar(shown, ax=axes[row, :], fraction=0.008, pad=0.01, label="activity density")
    figure.suptitle(
        f"{manifest['count_level']} Geant4: center-39-mm z-MIP at every saved iteration"
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

    report = {
        "output": str(output),
        "colormap": "gray_r (low=white, high=black)",
        "projection": "MIP over z-layer centers from -19.5 to +19.5 mm",
        "z_layer_centers_mm": z_selected.tolist(),
        "iterations": [(index + 1) * save_step for index in range(frames)],
        "row_vmax_actual_projected_voxel": row_vmax,
        "combined_rows": {
            "440_single_plus_218_corrected": "440_single + 218_corrected_single at each iteration",
            "440_single_compton_plus_218_corrected": "440_single_compton + 218_corrected_single at each iteration",
        },
        "input_sha256": input_hashes,
    }
    args.report.resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
