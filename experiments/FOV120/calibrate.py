"""Fit three independent four-layer absolute responses from FOV120 uniform MC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from detector_csv import load_detector_coordinates
from fov_config import load_config, validate_factor_geometry

NAMES = {"A218": "218keV_RotateNum20", "A440": "440keV_RotateNum20",
         "C440to218": "440keV_to218win_RotateNum20"}


def layer_scales(matrix, volumes, detector, observed, primary_count):
    if primary_count <= 0:
        raise ValueError("Actual monoenergetic primary count must be positive")
    predicted = np.zeros(len(detector), dtype=np.float64)
    for start in range(0, len(volumes), 1024):
        predicted += matrix[start:start+1024].sum(axis=0, dtype=np.float64) / volumes.sum()
    scale = np.empty(len(detector), dtype=np.float64)
    metrics = []
    for y in (200,230,260,290):
        selected = np.isclose(detector[:,1], y)
        model = float(predicted[selected].sum())
        counts = int(observed[selected].sum())
        if model <= 0 or counts <= 0 or not np.any(selected):
            raise ValueError(f"No usable calibration response in layer {y}")
        value = counts / primary_count / model
        scale[selected] = value
        metrics.append({"y_mm": y, "counts": counts, "model_efficiency": model,
                        "observed_efficiency": counts/primary_count, "scale": value,
                        "poisson_relative_se": counts**-.5})
    if not np.all(np.isin(np.round(detector[:,1], 3), (200,230,260,290))):
        raise ValueError("Unexpected detector layers")
    return scale, metrics


def calibrate(raw_root, data_root, destination, level="all", max_relative_se=.01):
    cfg = load_config()
    validate_factor_geometry({k:raw_root/v for k,v in NAMES.items()}, cfg, scan_matrix=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace Factors: {destination}")
    staging = destination.with_name(destination.name + ".building")
    staging.mkdir(parents=True, exist_ok=False)
    reports = {}
    for response, folder in NAMES.items():
        energy = 218 if response == "A218" else 440
        window = 440 if response == "A440" else 218
        dataset = f"calibration_{energy}"
        metadata = json.loads((data_root/"collections"/f"{dataset}_{level}.json").read_text())
        primary = metadata["primary_counts"]
        if primary[2] or primary[0 if energy == 440 else 1]:
            raise ValueError("Calibration input must be monoenergetic")
        observed = np.loadtxt(data_root/"CntStat"/f"{window}keV_RotateNum20_Geant4JSCC"/
                              f"CntStat_{dataset}_{level}.csv",delimiter=",").reshape(-1)
        source = raw_root/folder
        manifest = json.loads((source/"factor_manifest.json").read_text())
        if manifest.get("calibration",{}).get("enabled") or not manifest.get("maps_activity_density"):
            raise ValueError("Expected uncalibrated density-basis Factors")
        volumes = np.fromfile(source/"polar_cell_volume_mm3.float64",dtype="<f8")
        detector = load_detector_coordinates(source/"Detector.csv", cfg["detector_count"])
        # MATLAB on-disk layout: each pixel stores all detector rows contiguously.
        matrix = np.memmap(source/"SysMat_polar",mode="r",dtype="<f4",shape=(len(volumes),len(detector)))
        scales, metrics = layer_scales(matrix,volumes,detector,observed,sum(primary))
        if any(m["poisson_relative_se"] > max_relative_se for m in metrics):
            raise ValueError(f"Insufficient layer statistics for {response}; add independent calibration workers")
        target = staging/folder
        target.mkdir()
        for path in source.iterdir():
            if path.is_file() and path.name not in ("SysMat_polar","Sensi_d","SysMat_tmp","factor_manifest.json"):
                shutil.copy2(path,target/path.name)
        with (target/"SysMat_polar").open("wb") as stream:
            for start in range(0,len(volumes),1024):
                (matrix[start:start+1024]*scales).astype("<f4").tofile(stream)
        del matrix
        manifest["calibration"] = {"enabled": True, "name": "FOV120_uniform_four_layer",
                                  "parent": str(source), "primary_counts": primary,
                                  "workers": metadata["worker_indices"], "seeds": metadata["seeds"],
                                  "metrics": metrics, "includes_gamma_yield": False}
        (target/"factor_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        reports[response] = metrics
    validate_factor_geometry({k:staging/v for k,v in NAMES.items()},cfg,scan_matrix=True)
    (staging/"calibration_report.json").write_text(json.dumps(reports,indent=2)+"\n")
    staging.rename(destination)
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    generated = Path(__file__).resolve().parent/"generated"
    parser.add_argument("--raw-root",type=Path,default=generated/"FactorsRaw")
    parser.add_argument("--data-root",type=Path,default=generated)
    parser.add_argument("--destination",type=Path,default=generated/"Factors")
    parser.add_argument("--level",default="all")
    parser.add_argument("--max-relative-se",type=float,default=.01)
    args = parser.parse_args()
    print(json.dumps(calibrate(args.raw_root,args.data_root,args.destination,args.level,args.max_relative_se),indent=2))
