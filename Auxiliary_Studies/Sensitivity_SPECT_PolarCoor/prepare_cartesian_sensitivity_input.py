"""Prepare a cylindrical Cartesian point-response matrix for Sensi_d estimation."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-factor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radius-mm", type=float, default=153.0)
    parser.add_argument("--detector-count", type=int, default=10496)
    parser.add_argument("--detector-chunk", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source_factor_dir.resolve()
    output = args.output_dir.resolve()
    matrix_path = source / "SysMat_tmp"
    detector_path = source / "Detector.csv"
    if not matrix_path.is_file() or not detector_path.is_file():
        raise FileNotFoundError("Source Factors must contain SysMat_tmp and Detector.csv")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    x_axis = np.arange(-150.0, 150.0 + 0.1, 6.0)
    y_axis = np.arange(-150.0, 150.0 + 0.1, 6.0)
    z_axis = np.arange(-28.5, 28.5 + 0.1, 3.0)
    coordinates = np.asarray(
        [(x, y, z) for z in z_axis for y in y_axis for x in x_axis],
        dtype=np.float32,
    )
    active = np.hypot(coordinates[:, 0], coordinates[:, 1]) <= args.radius_mm + 1e-6
    active_indices = np.flatnonzero(active)
    active_coordinates = coordinates[active]
    full_pixels = coordinates.shape[0]
    active_pixels = active_coordinates.shape[0]
    expected_bytes = args.detector_count * full_pixels * np.dtype(np.float32).itemsize
    if matrix_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"SysMat_tmp has {matrix_path.stat().st_size} bytes; expected {expected_bytes}"
        )

    # SysMat_tmp is MATLAB [x,y,z,detector]: each detector owns one contiguous
    # Cartesian volume. The sensitivity pipeline expects detector values to be
    # contiguous for each voxel, so transpose in bounded detector chunks.
    source_matrix = np.memmap(
        matrix_path, dtype=np.float32, mode="r", shape=(args.detector_count, full_pixels)
    )
    destination_path = output / "SysMat_cartesian_cylindrical"
    destination = np.memmap(
        destination_path,
        dtype=np.float32,
        mode="w+",
        shape=(active_pixels, args.detector_count),
    )
    for start in range(0, args.detector_count, args.detector_chunk):
        stop = min(start + args.detector_chunk, args.detector_count)
        destination[:, start:stop] = source_matrix[start:stop, active_indices].T
        print(f"detectors {stop}/{args.detector_count}")
    destination.flush()
    del destination
    del source_matrix

    np.savetxt(output / "coor_cartesian_cylindrical.csv", active_coordinates, delimiter=",")
    shutil.copy2(detector_path, output / "Detector.csv")
    manifest = {
        "format_version": 1,
        "purpose": "Cartesian point-response input for Compton Sensi_d",
        "maps_activity_density": False,
        "matrix": "SysMat_cartesian_cylindrical",
        "source_matrix": str(matrix_path),
        "matrix_layout": "on disk [cartesian_voxel, detector], loaded as [detector, voxel]",
        "detector_count": args.detector_count,
        "full_cartesian_pixel_count": full_pixels,
        "active_cartesian_pixel_count": active_pixels,
        "active_radius_mm": args.radius_mm,
        "x_axis_mm": x_axis.tolist(),
        "y_axis_mm": y_axis.tolist(),
        "z_axis_mm": z_axis.tolist(),
        "normalization": "point response per emitted monoenergetic photon; no DeltaV weighting",
    }
    (output / "factor_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared Cartesian input: {output}")


if __name__ == "__main__":
    main()
