"""Strict Detector.csv loading shared by local validation utilities."""

from pathlib import Path

import numpy as np


def load_detector_coordinates(path, expected_count=None):
    """Load [id,x,y,z] or [x,y,z] CSV data with an optional header."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        first_fields = handle.readline().strip().split(",")
    try:
        for field in first_fields:
            float(field)
        skiprows = 0
    except ValueError:
        skiprows = 1

    values = np.loadtxt(
        path,
        delimiter=",",
        dtype=np.float32,
        ndmin=2,
        skiprows=skiprows,
    )
    if values.shape[1] == 4:
        detector_ids = values[:, 0]
        rounded_ids = np.rint(detector_ids).astype(np.int64)
        expected_ids = np.arange(1, values.shape[0] + 1, dtype=np.int64)
        if not np.allclose(detector_ids, rounded_ids) or not np.array_equal(
            rounded_ids, expected_ids
        ):
            raise ValueError(
                f"Detector IDs in {path} must be consecutive, one-based, and match row order."
            )
        coordinates = values[:, 1:4]
    elif values.shape[1] == 3:
        coordinates = values
    else:
        raise ValueError(
            f"Detector CSV must contain [id,x,y,z] or [x,y,z], got {values.shape}."
        )
    if expected_count is not None and coordinates.shape[0] != expected_count:
        raise ValueError(
            f"Detector count mismatch in {path}: expected {expected_count}, "
            f"found {coordinates.shape[0]}."
        )
    if not np.isfinite(coordinates).all():
        raise ValueError(f"Detector CSV contains non-finite coordinates: {path}")
    return np.ascontiguousarray(coordinates, dtype=np.float32)
