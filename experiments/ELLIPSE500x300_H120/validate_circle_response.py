"""Compare calibrated Factors to independent R150, H120 circle Geant4 data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from validate_factors import NAMES


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_fraction(coordinates: np.ndarray, radius: float = 150.) -> np.ndarray:
    r = np.hypot(coordinates[:, 0], coordinates[:, 1])
    nearest = np.rint(r / 6.) * 6.
    if np.max(np.abs(r - nearest)) > 1e-5:
        raise ValueError("Unexpected radial polar-grid coordinate")
    inner = np.maximum(nearest - 3., 0.)
    outer = nearest + 3.
    return np.clip((radius**2 - inner**2) / (outer**2 - inner**2), 0., 1.)


def predict(factor_root: Path, response: str, fraction: np.ndarray,
            source_volume: float, photons: int) -> tuple[np.ndarray, np.ndarray]:
    folder = factor_root / NAMES[response]
    manifest = json.loads((folder / "factor_manifest.json").read_text())
    if not manifest.get("calibration", {}).get("enabled"):
        raise ValueError(f"Uncalibrated Factor: {response}")
    detector = np.loadtxt(folder / "Detector.csv", delimiter=",", skiprows=1)
    matrix = np.memmap(folder / "SysMat_polar", dtype="<f4", mode="r",
                       shape=(len(fraction), len(detector)))
    count = np.zeros(len(detector), np.float64)
    for start in range(0, len(fraction), 256):
        stop = min(start + 256, len(fraction))
        count += (matrix[start:stop] * fraction[start:stop, None]).sum(axis=0, dtype=np.float64)
    count *= photons / source_volume
    return count, detector


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--factors", type=Path, required=True)
    p.add_argument("--counts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    metadata = json.loads((a.counts / "CircleNewDist_1e9.json").read_text())
    photons = metadata["primary_counts"]
    if sum(photons) != 1_000_000_000 or len(metadata["views"]) != 20:
        raise ValueError("Independent circle source count or view mismatch")
    folder = a.factors / NAMES["A218"]
    coordinates = np.loadtxt(folder / "coor_polar_full.csv", delimiter=",")
    volumes = np.fromfile(folder / "polar_cell_volume_mm3.float64", dtype="<f8")
    fraction = source_fraction(coordinates)
    source_volume = float(np.dot(volumes, fraction))
    analytic = math.pi * 150.**2 * 120.
    if abs(source_volume / analytic - 1.) > 1e-5:
        raise ValueError("R150 source support volume does not close")
    predictions = {}
    layer = None
    for response, source_count in (("A218", photons[0]), ("A440", photons[1]),
                                   ("C440to218", photons[1])):
        predictions[response], detector = predict(a.factors, response, fraction,
                                                   source_volume, source_count)
        if layer is None:
            layer = np.abs(detector[:, 2])
        elif not np.array_equal(layer, np.abs(detector[:, 2])):
            raise ValueError("Detector order differs between responses")
    report = {"experiment": "ELLIPSE500x300_H120", "dataset": "CircleNewDist_1e9",
              "independent_of_calibration_source": True,
              "source_volume_mm3": source_volume,
              "source_volume_relative_error": source_volume / analytic - 1.,
              "primary_counts": photons, "windows": {},
              "input_sha256": {"collection": digest(a.counts / "CircleNewDist_1e9.json")}}
    for energy, components in ((218, ("A218", "C440to218")), (440, ("A440",))):
        path = a.counts / str(energy) / "CntStat_CircleNewDist_1e9.csv"
        observed = np.loadtxt(path, delimiter=",", dtype=np.int64)
        if observed.shape != (20, len(layer)) or np.any(observed < 0):
            raise ValueError(f"Unexpected CntStat: {path}")
        observed = observed.sum(axis=0)
        predicted = sum(predictions[name] for name in components)
        rows = []
        for position in (300., 330., 360., 390.):
            selected = np.isclose(layer, position)
            o = int(observed[selected].sum())
            m = float(predicted[selected].sum())
            rows.append({"layer_mm": position, "observed": o, "predicted": m,
                         "predicted_over_observed": m / o,
                         "relative_difference": m / o - 1.,
                         "observed_poisson_relative_se": o**-.5})
        report["windows"][str(energy)] = rows
        report["input_sha256"][f"CntStat_{energy}"] = digest(path)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({e: [round(x["relative_difference"], 4) for x in v]
                      for e, v in report["windows"].items()}))


if __name__ == "__main__":
    main()
