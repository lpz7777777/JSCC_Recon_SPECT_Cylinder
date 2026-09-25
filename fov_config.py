"""Shared experiment configuration and strict Factors geometry validation."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "experiments/FOV120/config.json"


def validate_sensitivity_provenance(directory, verify_matrix=False):
    directory = Path(directory)
    record = json.loads((directory / "Sensi_d_provenance.json").read_text())
    if (record["operator"], record["resolution_fwhm"], record["reference_keV"],
            record["sum_threshold_MeV"], record["input_already_smeared"]) != ("K*B", .13, 511, .350, True):
        raise ValueError("Sensitivity physics differs from frozen reconstruction response")
    required = {"Sensi_d", "factor_manifest.json", "coor_polar_full.csv", "Detector.csv", "polar_cell_volume_mm3.float64", "SysMat_polar"}
    if set(record["hashes"]) != required:
        raise ValueError("Incomplete Sensi_d provenance")
    for name, checksum in record["hashes"].items():
        if name == "SysMat_polar" and not verify_matrix:
            continue  # preflight hashes it once, rather than once per GPU rank
        with (directory / name).open("rb") as stream:
            # Reconstruction cluster uses Python 3.9; file_digest requires 3.11.
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != checksum:
                raise ValueError(f"Stale Sensi_d provenance: {name}")


def load_config(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    layers = cfg["height_mm"] / cfg["z_spacing_mm"]
    if layers <= 0 or not float(layers).is_integer():
        raise ValueError("height must be a positive integer multiple of z spacing")
    cfg["z_layers"] = int(layers)
    cfg["pixel_count"] = int(layers) * cfg["points_per_layer"]
    cfg["z_centers_mm"] = ((np.arange(int(layers)) + .5) * cfg["z_spacing_mm"]
                            - cfg["height_mm"] / 2)
    return cfg


def factor_geometry(directory):
    directory = Path(directory)
    coordinates = np.loadtxt(directory / "coor_polar_full.csv", delimiter=",", ndmin=2)
    if coordinates.shape[1] != 3 or not np.isfinite(coordinates).all():
        raise ValueError(f"Invalid coordinates: {directory}")
    z, counts = np.unique(coordinates[:, 2], return_counts=True)
    if len(set(counts)) != 1 or len(z) < 2 or not np.allclose(np.diff(z), np.diff(z)[0]):
        raise ValueError("Expected equal polar layers with uniform z spacing")
    if not np.array_equal(coordinates[:, 2], np.repeat(z, counts[0])):
        raise ValueError("Polar coordinates must be stored in contiguous ascending z layers")
    planes=coordinates[:, :2].reshape(len(z), counts[0], 2)
    if not np.allclose(planes, planes[0], rtol=0, atol=1e-8):
        raise ValueError("Every axial layer must use the same ordered polar samples")
    return coordinates, int(counts[0]), len(z)


def validate_factor_geometry(directories, cfg=None, scan_matrix=False):
    """Reject mixed grids before allocating any full system matrix."""
    from detector_csv import load_detector_coordinates
    reference = None
    report = {}
    for label, path in directories.items():
        path = Path(path)
        coords, per_layer, nz = factor_geometry(path)
        det = load_detector_coordinates(path / "Detector.csv")
        volume = np.fromfile(path / "polar_cell_volume_mm3.float64", dtype="<f8")
        rot = np.loadtxt(path / "RotMat_full.csv", delimiter=",", dtype=np.int64, ndmin=2)
        inv = np.loadtxt(path / "RotMatInv_full.csv", delimiter=",", dtype=np.int64, ndmin=2)
        n = len(coords)
        if volume.shape != (n,) or not np.isfinite(volume).all() or np.any(volume <= 0):
            raise ValueError(f"Invalid cell volumes: {path}")
        if rot.shape != inv.shape or rot.shape[0] != n:
            raise ValueError(f"Invalid rotation shapes: {path}")
        for v in range(rot.shape[1]):
            if not np.array_equal(np.sort(rot[:, v]), np.arange(1, n + 1)):
                raise ValueError(f"Invalid rotation permutation: {path}")
            if np.any(inv[:, v] < 1) or np.any(inv[:, v] > n) or not np.array_equal(
                    rot[inv[:, v] - 1, v], np.arange(1, n + 1)):
                raise ValueError(f"Rotation inverse mismatch: {path}")
            if not np.allclose(volume[rot[:, v] - 1], volume, rtol=1e-12):
                raise ValueError(f"Rotation changes cell volumes: {path}")
            if not np.array_equal(coords[rot[:, v]-1, 2], coords[:, 2]):
                raise ValueError(f"In-plane rotation changes axial positions: {path}")
        expected = n * len(det) * 4
        if (path / "SysMat_polar").stat().st_size != expected:
            raise ValueError(f"Matrix byte count mismatch: {path}")
        if cfg:
            if (n, per_layer, nz, len(det), rot.shape[1]) != (
                    cfg["pixel_count"], cfg["points_per_layer"], cfg["z_layers"],
                    cfg["detector_count"], cfg["rotate_num"]):
                raise ValueError(f"Geometry does not match experiment: {path}")
            if not np.allclose(np.unique(coords[:, 2]), cfg["z_centers_mm"]):
                raise ValueError(f"Axial extent mismatch: {path}")
            radii=np.unique(np.round(np.linalg.norm(coords[:, :2], axis=1), 5))
            if not np.array_equal(radii, np.arange(0, cfg["physical_radius_mm"]+1, 6)):
                raise ValueError(f"Radial grid mismatch: {path}")
            if not np.allclose(np.unique(np.abs(det[:, 1])), cfg["detector_y_mm"]):
                raise ValueError(f"Detector distance mismatch: {path}")
            expected_volume = np.pi * cfg["support_radius_mm"] ** 2 * cfg["height_mm"]
            if not np.isclose(volume.sum(), expected_volume, rtol=1e-10):
                raise ValueError(f"Support volume mismatch: {path}")
        current = (coords, det, volume, rot, inv)
        if reference is not None and any(not np.array_equal(a, b) for a, b in zip(reference, current)):
            raise ValueError(f"Response geometry mismatch: {path}")
        reference = current
        if scan_matrix:
            data = np.memmap(path / "SysMat_polar", dtype="<f4", mode="r")
            for start in range(0, data.size, 4_000_000):
                block = data[start:start + 4_000_000]
                if not np.isfinite(block).all() or np.any(block < 0):
                    raise ValueError(f"Non-finite/negative response: {path}")
            del data
        report[label] = {"pixel_count": n, "z_layers": nz, "matrix_bytes": expected,
                         "volume_mm3": float(volume.sum())}
    return report
