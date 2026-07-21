"""Visualize the six 1e9 Geant4 JSCC validation reconstructions."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


OUTPUTS = [
    ("440_SinglePhoton", "440 single-photon"),
    ("440_ComptonOnly", "440 Compton-only"),
    ("440_SinglePlusCompton", "440 single + Compton"),
    ("218_SinglePhoton_CrossTalkCorrected", "218 corrected single-photon"),
    ("440SinglePlus218Single", "440 single + 218 single"),
    ("440SingleComptonPlus218Single", "440 single + Compton + 218 single"),
]


def polar_to_cartesian_mip(coordinates, values, axis):
    xx, yy = np.meshgrid(axis, axis)
    planes = []
    for z in np.unique(coordinates[:, 2]):
        selected = np.isclose(coordinates[:, 2], z)
        plane = griddata(
            coordinates[selected, :2], values[selected], (xx, yy), method="linear"
        )
        planes.append(plane)
    stack = np.stack(planes)
    valid = np.isfinite(stack)
    mip = np.max(np.where(valid, stack, -np.inf), axis=0)
    mip[~np.any(valid, axis=0)] = np.nan
    return xx, yy, mip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir", type=Path,
        default=Path("Results/Reconstruction/JSCC_ComptonValidation_Geant4_1e9_Iter1000"),
    )
    parser.add_argument("--factor-dir", type=Path, default=Path("Factors/440keV_RotateNum20"))
    args = parser.parse_args()
    result = args.result_dir.resolve()
    factor = args.factor_dir.resolve()
    coordinates = np.loadtxt(factor / "coor_polar_full.csv", delimiter=",")
    pixel_count = coordinates.shape[0]
    axis = np.arange(-150.0, 150.01, 3.0)

    images = {}
    maps = {}
    for key, _ in OUTPUTS:
        image = np.fromfile(result / f"Image_{key}", dtype=np.float32).astype(np.float64)
        if image.size != pixel_count or not np.isfinite(image).all() or np.any(image < 0):
            raise ValueError(f"Invalid image: {key}")
        images[key] = image
        maps[key] = polar_to_cartesian_mip(coordinates, image, axis)[2]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, OUTPUTS):
        values = maps[key]
        vmax = float(np.nanquantile(values, 0.995))
        plotted = ax.imshow(
            values, origin="lower", extent=(-150, 150, -150, 150),
            cmap="gray", vmin=0, vmax=max(vmax, 1e-12),
        )
        ax.set_title(title); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        ax.set_aspect("equal"); fig.colorbar(plotted, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("JSCC Geant4 contrast phantom, 1e9, 1000 MLEM iterations")
    figure_path = result / "JSCC_ComptonValidation_1e9_Iter1000.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    sum1 = images["440_SinglePhoton"] + images["218_SinglePhoton_CrossTalkCorrected"]
    sum2 = images["440_SinglePlusCompton"] + images["218_SinglePhoton_CrossTalkCorrected"]
    summary = {
        "pixel_count": pixel_count,
        "images": {
            key: {
                "min": float(value.min()), "max": float(value.max()),
                "mean": float(value.mean()), "sum": float(value.sum()),
                "cv": float(value.std() / max(value.mean(), 1e-30)),
            }
            for key, value in images.items()
        },
        "additivity_max_abs_error": {
            "440_single_plus_218": float(np.max(np.abs(sum1 - images["440SinglePlus218Single"]))),
            "440_joint_plus_218": float(np.max(np.abs(sum2 - images["440SingleComptonPlus218Single"]))),
        },
        "correlations": {
            "440_single_vs_compton": float(np.corrcoef(images["440_SinglePhoton"], images["440_ComptonOnly"])[0, 1]),
            "440_single_vs_joint": float(np.corrcoef(images["440_SinglePhoton"], images["440_SinglePlusCompton"])[0, 1]),
        },
        "figure": str(figure_path),
    }
    (result / "visualization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
