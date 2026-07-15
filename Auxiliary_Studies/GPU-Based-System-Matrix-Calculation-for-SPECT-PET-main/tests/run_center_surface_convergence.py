#!/usr/bin/env python3
"""Run full-detector, one-center-voxel target-surface convergence tests."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCATTER_NAME = "Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat"
COMPONENT_NAMES = (
    "C_intercrystal.sysmat",
    "C_highZ_to_crystal.sysmat",
    "C_local_recoil.sysmat",
    "C_local_self_photoelectric.sysmat",
    "C_collimator_to_crystal.sysmat",
    "C_total.sysmat",
)


def prepare_center_input(base_run: Path, output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    output_dir.mkdir(parents=True)
    for name in ("Params_Collimator.dat", "Params_Detector.dat", "Params_Physics.dat"):
        shutil.copy2(base_run / name, output_dir / name)

    detector_raw = np.fromfile(base_run / "Params_Detector.dat", dtype="<f4")
    detector_count = int(detector_raw[0])
    detector = detector_raw[1:].reshape(detector_count, 12)
    image = np.fromfile(base_run / "Params_Image.dat", dtype="<f4")
    nx, ny, nz = (int(value) for value in image[:3])
    if nx % 2 != 1 or ny % 2 != 1 or nz % 2 != 0:
        raise ValueError("Base image must have odd X/Y and even Z.")

    center_image = image.copy()
    center_image[:3] = 1.0
    center_image[6] = 1.0
    center_image[8:11] = 0.0
    center_image.tofile(output_dir / "Params_Image.dat")

    voxel_count = nx * ny * nz
    pe = np.memmap(
        base_run / "PE_SysMat_shift_0.000000_0.000000_0.000000_v3.sysmat",
        dtype="<f4",
        mode="r",
        shape=(detector_count, voxel_count),
        order="C",
    )
    columns = [
        z * nx * ny + (ny // 2) * nx + nx // 2
        for z in (nz // 2 - 1, nz // 2)
    ]
    center_pe = 0.5 * (
        np.asarray(pe[:, columns[0]], dtype=np.float32)
        + np.asarray(pe[:, columns[1]], dtype=np.float32)
    )
    center_pe.astype("<f4").tofile(
        output_dir / "PE_SysMat_shift_0.000000_0.000000_0.000000_v3.sysmat"
    )
    return detector, center_pe


def run_scatter(
    executable: Path,
    run_dir: Path,
    far_subdivisions: int,
    near_subdivisions: int,
    gpu: int,
) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "SCATTER_WRITE_COMPONENTS": "1",
            "SCATTER_STRUCTURED_TRAVERSAL": "1",
            "SCATTER_KINEMATIC_PRUNING": "1",
            "SCATTER_COMPTON_INTEGRAND_LUT": "1",
            "SCATTER_TARGET_FACE_SUBDIV": str(far_subdivisions),
            "SCATTER_NEAR_TARGET_FACE_SUBDIV": str(near_subdivisions),
            "SCATTER_NEAR_TARGET_DISTANCE_FACTOR": "2.0",
            "SCATTER_CRYSTAL_CHUNK": "16",
            "DETECTOR_LOCAL_SCATTER_ORIENTATION_BINS": "17",
            "DETECTOR_LOCAL_SCATTER_COSINE_SAMPLES": "96",
            "DETECTOR_LOCAL_SCATTER_AZIMUTH_SAMPLES": "96",
        }
    )
    process = subprocess.run(
        [
            str(executable),
            "-PE",
            "PE_SysMat_shift_0.000000_0.000000_0.000000_v3.sysmat",
            "-cuda",
            str(gpu),
        ],
        cwd=run_dir,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    (run_dir / "scattergen.log").write_text(
        process.stdout + process.stderr, encoding="utf-8"
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"ScatterGen failed with code {process.returncode}; see {run_dir / 'scattergen.log'}."
        )


def read_response(run_dir: Path, detector: np.ndarray) -> dict[str, object]:
    active = detector[:, 11] == 1
    active_detector = detector[active]
    layers = np.unique(active_detector[:, 1])
    result = {}
    for name in (SCATTER_NAME, *COMPONENT_NAMES):
        values = np.fromfile(run_dir / name, dtype="<f4")
        if values.size != len(detector):
            raise ValueError(f"Unexpected element count in {run_dir / name}.")
        values = values[active].astype(np.float64)
        result[name] = {
            "total": float(values.sum()),
            "by_layer": {
                f"{layer:g}": float(values[active_detector[:, 1] == layer].sum())
                for layer in layers
            },
        }
    return result


def read_geant4(build_dir: Path, detector: np.ndarray) -> dict[str, object]:
    primary_count = int(np.loadtxt(build_dir / "PrimaryCount440.csv"))
    counts = np.ravel(
        np.loadtxt(build_dir / "CntStat218_From440.csv", delimiter=",", dtype=np.float64)
    )
    active_detector = detector[detector[:, 11] == 1]
    if counts.size != len(active_detector):
        raise ValueError("Geant4 detector count does not match active matrix rows.")
    layers = np.unique(active_detector[:, 1])
    probability = counts / primary_count
    return {
        "primary_count": primary_count,
        "total": float(probability.sum()),
        "by_layer": {
            f"{layer:g}": float(probability[active_detector[:, 1] == layer].sum())
            for layer in layers
        },
    }


def write_reports(output_dir: Path, summary: dict[str, object]) -> None:
    (output_dir / "convergence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    layers = list(summary["geant4"]["by_layer"])
    with (output_dir / "convergence_by_layer.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["far_subdivisions", "near_subdivisions", "layer_y_mm", "matrix", "geant4", "ratio"]
        )
        for run in summary["runs"]:
            values = run["response"][SCATTER_NAME]["by_layer"]
            for layer in layers:
                geant4 = summary["geant4"]["by_layer"][layer]
                writer.writerow(
                    [
                        run["far_subdivisions"],
                        run["near_subdivisions"],
                        layer,
                        values[layer],
                        geant4,
                        values[layer] / geant4,
                    ]
                )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for layer in layers:
        far = [run["far_subdivisions"] for run in summary["runs"]]
        response = [
            run["response"][SCATTER_NAME]["by_layer"][layer]
            / summary["geant4"]["by_layer"][layer]
            for run in summary["runs"]
        ]
        axes[0].plot(far, response, marker="o", label=f"y={layer} mm")
    axes[0].axhline(1.0, color="black", linewidth=1)
    axes[0].set_xlabel("Far target face subdivisions per axis")
    axes[0].set_ylabel("Matrix / Geant4")
    axes[0].set_title("Layer response convergence")
    axes[0].legend()

    component_keys = (
        "C_intercrystal.sysmat",
        "C_highZ_to_crystal.sysmat",
        "C_local_recoil.sysmat",
    )
    labels = ("Intercrystal", "High-Z", "Local recoil")
    x = np.arange(len(summary["runs"]))
    bottom = np.zeros(len(summary["runs"]))
    for key, label in zip(component_keys, labels):
        values = np.array(
            [run["response"][key]["total"] for run in summary["runs"]]
        )
        axes[1].bar(x, values, bottom=bottom, label=label)
        bottom += values
    axes[1].axhline(summary["geant4"]["total"], color="black", label="Geant4")
    axes[1].set_xticks(
        x, [str(run["far_subdivisions"]) for run in summary["runs"]]
    )
    axes[1].set_xlabel("Far target face subdivisions per axis")
    axes[1].set_ylabel("Counts per 440 keV primary")
    axes[1].set_title("Center response components")
    axes[1].legend()
    figure.savefig(output_dir / "center_surface_convergence.png", dpi=180)
    plt.close(figure)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repository = script_dir.parent
    project_root = repository.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument(
        "--base-run",
        type=Path,
        default=repository / "runs" / "JSCC_440keV_to_218keVwin_SurfaceValidation",
    )
    parser.add_argument(
        "--geant4-build",
        type=Path,
        default=project_root / "Geant4Sim" / "Geant4Code_CntStatResponseStudy" / "build",
    )
    parser.add_argument("--far", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--near", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        project_root / "run_logs" / f"ScatterSurface_CenterConvergence_{timestamp}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    runs = []
    detector = None
    for far_subdivisions in args.far:
        run_dir = output_dir / f"far_{far_subdivisions}_near_{args.near}"
        detector, _ = prepare_center_input(args.base_run.resolve(), run_dir)
        run_scatter(
            args.executable.resolve(),
            run_dir,
            far_subdivisions,
            args.near,
            args.gpu,
        )
        runs.append(
            {
                "far_subdivisions": far_subdivisions,
                "near_subdivisions": args.near,
                "run_dir": str(run_dir),
                "response": read_response(run_dir, detector),
            }
        )

    if detector is None:
        raise ValueError("At least one far subdivision value is required.")
    summary = {
        "base_run": str(args.base_run.resolve()),
        "executable": str(args.executable.resolve()),
        "geant4": read_geant4(args.geant4_build.resolve(), detector),
        "runs": runs,
    }
    write_reports(output_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
