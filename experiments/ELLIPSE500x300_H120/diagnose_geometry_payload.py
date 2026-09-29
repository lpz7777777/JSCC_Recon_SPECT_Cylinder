"""Quantify generated versus saved ellipse geometry payload differences."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geometry import generate


def main(root):
    cfg = json.loads((HERE / "config.json").read_text())
    coords, volumes, fractions, active, rotation, inverse = generate(cfg)
    factor = root / "218keV_RotateNum20"
    for filename, expected, dtype in (
        ("polar_cell_volume_mm3.float64", volumes, "<f8"),
        ("ellipse_fraction.float64", fractions, "<f8"),
        ("ellipse_active_indices.int32", active, "<i4")):
        path = factor / filename
        observed = np.fromfile(path, dtype=dtype)
        if len(observed) != len(expected):
            raise ValueError(f"Size mismatch {filename}: {len(observed)}, {len(expected)}")
        delta = np.abs(observed - expected)
        mismatches = np.flatnonzero(delta != 0)
        print(json.dumps({"filename": filename, "saved_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "dtype": str(expected.dtype), "length": len(expected),
                          "bitwise_mismatch_count": len(mismatches),
                          "max_abs_difference": float(delta.max(initial=0)),
                          "mean_abs_difference": float(delta.mean()),
                          "example": [{"index": int(i), "saved": float(observed[i]),
                                       "generated": float(expected[i])}
                                      for i in mismatches[:5]]}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
