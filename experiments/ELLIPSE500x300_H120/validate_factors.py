"""Validate isolated 132040-column circular Factors before calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geometry import generate

NAMES = {"A218": "218keV_RotateNum20", "A440": "440keV_RotateNum20",
         "C440to218": "440keV_to218win_RotateNum20"}


def validate(root: Path, full_scan=False):
    cfg = json.loads((HERE / "config.json").read_text())
    coords, volumes, fractions, active, rot, inverse = generate(cfg)
    expected_size = len(coords) * cfg["detector_count"] * 4
    signatures = []
    report = {}
    for response, name in NAMES.items():
        folder = root / name
        manifest = json.loads((folder / "factor_manifest.json").read_text())
        if manifest["pixel_num"] != len(coords) or manifest["detector_num"] != cfg["detector_count"]:
            raise ValueError(f"Factor shape mismatch: {response}")
        if manifest["ellipse_fraction_applied"] or not manifest["maps_activity_density"]:
            raise ValueError(f"Wrong response basis: {response}")
        if (folder / "SysMat_polar").stat().st_size != expected_size:
            raise ValueError(f"Raw factor byte count mismatch: {response}")
        if not np.allclose(np.loadtxt(folder / "coor_polar_full.csv", delimiter=","), coords, atol=1e-8):
            raise ValueError(f"Coordinate mismatch: {response}")
        if not np.array_equal(np.loadtxt(folder / "RotMat_full.csv", delimiter=",", dtype=np.int32), rot + 1):
            raise ValueError(f"Rotation mismatch: {response}")
        if not np.array_equal(np.loadtxt(folder / "RotMatInv_full.csv", delimiter=",", dtype=np.int32), inverse + 1):
            raise ValueError(f"Inverse rotation mismatch: {response}")
        for filename, expected, dtype in (("polar_cell_volume_mm3.float64", volumes, "<f8"),
                                          ("ellipse_fraction.float64", fractions, "<f8"),
                                          ("ellipse_active_indices.int32", active, "<i4")):
            got = np.fromfile(folder / filename, dtype=dtype)
            if not np.array_equal(got, expected):
                raise ValueError(f"Geometry payload mismatch: {response}/{filename}")
        detector = np.loadtxt(folder / "Detector.csv", delimiter=",", skiprows=1)
        if detector.shape != (cfg["detector_count"], 4):
            raise ValueError(f"Detector table shape mismatch: {response}")
        if not np.array_equal(detector[:, 0], np.arange(1, cfg["detector_count"]+1)):
            raise ValueError(f"Detector numbering mismatch: {response}")
        if not np.array_equal(np.unique(np.abs(detector[:, 2])), cfg["detector_y_mm"]):
            raise ValueError(f"Detector distance mismatch: {response}")
        signatures.append(detector.tobytes())
        matrix = np.memmap(folder / "SysMat_polar", dtype="<f4", mode="r",
                           shape=(len(coords), cfg["detector_count"]))
        if full_scan:
            for begin in range(0, len(coords), 128):
                slab = matrix[begin:begin+128]
                if not np.isfinite(slab).all() or np.any(slab < 0):
                    raise ValueError(f"Invalid matrix values: {response}, slab {begin}")
        else:
            for begin in (0, len(coords)//2, len(coords)-128):
                slab = matrix[begin:begin+128]
                if not np.isfinite(slab).all() or np.any(slab < 0):
                    raise ValueError(f"Invalid sampled matrix values: {response}, slab {begin}")
        report[response] = {"shape": [len(coords), cfg["detector_count"]],
                            "bytes": expected_size, "full_scan": full_scan,
                            "detector_layers_mm": cfg["detector_y_mm"]}
        del matrix
    if not (signatures[0] == signatures[1] == signatures[2]):
        raise ValueError("Three response detector tables differ")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--full-scan", action="store_true")
    a = p.parse_args()
    print(json.dumps(validate(a.root, a.full_scan), indent=2))
