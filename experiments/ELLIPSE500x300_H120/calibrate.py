"""Fit separate four-layer absolute responses using pure uniform circle data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from validate_factors import NAMES, validate


def layer_scales(matrix, volumes, detector, observed, photons, positions):
    if photons <= 0 or volumes.sum() <= 0:
        raise ValueError("Invalid calibration source normalization")
    predicted = np.zeros(len(detector), dtype=np.float64)
    for begin in range(0, len(volumes), 256):
        predicted += matrix[begin:begin+256].sum(axis=0, dtype=np.float64)
    predicted /= volumes.sum()
    scale = np.empty(len(detector), dtype=np.float64)
    metrics = []
    for position in positions:
        selected = np.isclose(np.abs(detector[:, 2]), position)
        count = int(observed[selected].sum())
        model = float(predicted[selected].sum())
        if not selected.any() or count <= 0 or model <= 0:
            raise ValueError(f"No usable calibration response at {position} mm")
        scale[selected] = count / photons / model
        metrics.append({"layer_mm": position, "counts": count,
                        "model_efficiency": model, "observed_efficiency": count/photons,
                        "scale": count/photons/model, "poisson_relative_se": count**-.5})
    if not np.all(np.isfinite(scale)):
        raise ValueError("Incomplete layer calibration")
    return scale, metrics


def calibrate(raw_root, data_root, destination, level="all", max_relative_se=.01):
    validate(raw_root)
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.with_name(destination.name + ".building")
    staging.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((HERE / "config.json").read_text())
    reports = {}
    for response, name in NAMES.items():
        energy = 218 if response == "A218" else 440
        window = 440 if response == "A440" else 218
        source = raw_root / name
        metadata = json.loads((data_root / "collections" /
                               f"calibration_{energy}_{level}.json").read_text())
        primary = metadata["primary_counts"]
        if primary[2] or primary[0 if energy == 440 else 1]:
            raise ValueError("Calibration must use pure emission")
        photons = sum(primary)
        observed = np.loadtxt(data_root / "CntStat" /
            f"{window}keV_RotateNum20_Geant4JSCC" /
            f"CntStat_calibration_{energy}_{level}.csv", delimiter=",").reshape(-1)
        volumes = np.fromfile(source / "polar_cell_volume_mm3.float64", dtype="<f8")
        detector = np.loadtxt(source / "Detector.csv", delimiter=",", skiprows=1)
        matrix = np.memmap(source / "SysMat_polar", dtype="<f4", mode="r",
                           shape=(len(volumes), cfg["detector_count"]))
        scales, metrics = layer_scales(matrix, volumes, detector, observed,
                                       photons, cfg["detector_y_mm"])
        if any(item["poisson_relative_se"] > max_relative_se for item in metrics):
            raise ValueError(f"Insufficient layer statistics: {response}")
        target = staging / name
        target.mkdir()
        for path in source.iterdir():
            if path.is_file() and path.name not in ("SysMat_polar", "Sensi_d", "factor_manifest.json"):
                shutil.copy2(path, target / path.name)
        with (target / "SysMat_polar").open("wb") as sink:
            for begin in range(0, len(volumes), 256):
                (matrix[begin:begin+256] * scales).astype("<f4").tofile(sink)
        factor_manifest = json.loads((source / "factor_manifest.json").read_text())
        factor_manifest["calibration"] = {"enabled": True,
             "name": "ELLIPSE500x300_H120_uniform_circle_four_layer",
             "source_shape": "circle_R255_H120", "source_primary_counts": primary,
             "workers": metadata["worker_indices"], "seeds": metadata["seeds"],
             "metrics": metrics, "includes_gamma_yield": False}
        (target / "factor_manifest.json").write_text(json.dumps(factor_manifest, indent=2)+"\n")
        reports[response] = metrics
        del matrix
    validate(staging)
    (staging / "calibration_report.json").write_text(json.dumps(reports, indent=2)+"\n")
    staging.rename(destination)
    return reports


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--level", default="all")
    a = p.parse_args()
    print(json.dumps(calibrate(a.raw_root, a.data_root, a.destination, a.level), indent=2))
