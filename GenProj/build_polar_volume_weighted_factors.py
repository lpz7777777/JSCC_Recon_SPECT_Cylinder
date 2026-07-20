#!/usr/bin/env python3
"""Create density-basis Factors by applying polar-cell volumes to A columns.

The input matrix maps integrated activity per polar cell to detector counts:

    y = A x

The output matrix maps physical activity density to detector counts:

    y = (A diag(DeltaV)) rho

`SysMat_polar` is stored as contiguous (pixel, detector-bin) float32 rows, so
right column scaling in the mathematical detector-by-pixel matrix is a row
scaling operation in the on-disk array.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SOURCE_SUFFIX = "CenterPoint_PEv4_UniformFOVLayer"
OUTPUT_SUFFIX = SOURCE_SUFFIX + "_PolarVolume"
FACTOR_STEMS = (
    "218keV_RotateNum20",
    "440keV_RotateNum20",
    "440keV_to218win_RotateNum20",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--source-suffix", default=SOURCE_SUFFIX)
    parser.add_argument("--output-suffix", default=OUTPUT_SUFFIX)
    parser.add_argument("--rows-per-chunk", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def midpoint_bounds(values: np.ndarray, lower_limit: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    midpoint = 0.5 * (values[:-1] + values[1:])
    lower = np.empty_like(values)
    upper = np.empty_like(values)
    lower[1:] = midpoint
    upper[:-1] = midpoint
    lower[0] = values[0] - 0.5 * (values[1] - values[0])
    upper[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    if lower_limit is not None:
        lower[0] = lower_limit
    return lower, upper


def build_volume_vector(coordinates: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    radius = np.round(np.hypot(coordinates[:, 0], coordinates[:, 1]), 8)
    z_value = np.round(coordinates[:, 2], 8)
    radii = np.unique(radius)
    z_values = np.unique(z_value)
    if radii[0] != 0.0 or radii.size < 2 or z_values.size < 2:
        raise ValueError("Expected a center-inclusive multi-layer polar grid.")

    radial_inner, radial_outer = midpoint_bounds(radii, lower_limit=0.0)
    z_lower, z_upper = midpoint_bounds(z_values)
    ring_count = np.array(
        [np.count_nonzero(radius == value) // z_values.size for value in radii],
        dtype=np.int64,
    )
    if ring_count[0] != 1:
        raise ValueError(f"Expected one center point per layer; got {ring_count[0]}.")

    radial_index = np.searchsorted(radii, radius)
    z_index = np.searchsorted(z_values, z_value)
    area_by_ring = math.pi * (radial_outer**2 - radial_inner**2) / ring_count
    thickness_by_layer = z_upper - z_lower
    volume = area_by_ring[radial_index] * thickness_by_layer[z_index]
    if not np.all(np.isfinite(volume)) or np.any(volume <= 0.0):
        raise ValueError("Polar-cell volume vector is nonpositive or nonfinite.")

    expected_domain_volume = math.pi * radial_outer[-1] ** 2 * (
        z_upper[-1] - z_lower[0]
    )
    represented_domain_volume = float(np.sum(volume))
    closure = represented_domain_volume / expected_domain_volume - 1.0
    if abs(closure) > 1.0e-12:
        raise ValueError(f"Polar domain volume closure failed: {closure:.6e}")

    metadata: dict[str, object] = {
        "method": "midpoint radial/axial bounds and equal angular sectors per ring",
        "units": "mm3",
        "coordinate_count": int(coordinates.shape[0]),
        "radii_mm": radii.tolist(),
        "points_per_ring": ring_count.tolist(),
        "z_values_mm": z_values.tolist(),
        "radial_outer_domain_mm": float(radial_outer[-1]),
        "axial_lower_domain_mm": float(z_lower[0]),
        "axial_upper_domain_mm": float(z_upper[-1]),
        "minimum_mm3": float(np.min(volume)),
        "maximum_mm3": float(np.max(volume)),
        "mean_mm3": float(np.mean(volume)),
        "median_mm3": float(np.median(volume)),
        "sum_mm3": represented_domain_volume,
        "analytic_domain_volume_mm3": expected_domain_volume,
        "relative_volume_closure_error": closure,
        "sha256_float64": hashlib.sha256(volume.astype("<f8").tobytes()).hexdigest(),
    }
    return volume, metadata


def validate_rotation_invariance(factor_dir: Path, volume: np.ndarray) -> float:
    rotmat = np.loadtxt(factor_dir / "RotMat_full.csv", delimiter=",", dtype=np.int64)
    if rotmat.shape[0] != volume.size or rotmat.min() < 1 or rotmat.max() > volume.size:
        raise ValueError(f"Invalid RotMat_full.csv in {factor_dir}")
    maximum_error = float(np.max(np.abs(volume[rotmat - 1] - volume[:, None])))
    if maximum_error > 1.0e-10:
        raise ValueError(
            f"Polar-cell volume is not rotation invariant; max error={maximum_error:.6e} mm3."
        )
    return maximum_error


def copy_factor_metadata(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    for source_path in source_dir.iterdir():
        if source_path.name in {"SysMat_polar", "factor_manifest.json"}:
            continue
        if source_path.is_file():
            shutil.copy2(source_path, output_dir / source_path.name)


def scale_matrix(
    source_path: Path,
    output_path: Path,
    volume: np.ndarray,
    rows_per_chunk: int,
) -> dict[str, object]:
    float_size = np.dtype("<f4").itemsize
    element_count = source_path.stat().st_size // float_size
    if source_path.stat().st_size % float_size or element_count % volume.size:
        raise ValueError(f"Matrix size is incompatible with {volume.size} pixels: {source_path}")
    detector_count = element_count // volume.size
    temporary_path = output_path.with_name(output_path.name + ".partial")
    source = None
    output = None
    source_sum = 0.0
    expected_sum = 0.0
    output_sum = 0.0
    nonzero_count = 0
    source_hash = hashlib.sha256()
    output_hash = hashlib.sha256()
    volume32 = volume.astype(np.float32)
    try:
        source = np.memmap(
            source_path, dtype="<f4", mode="r", shape=(volume.size, detector_count)
        )
        output = np.memmap(
            temporary_path,
            dtype="<f4",
            mode="w+",
            shape=(volume.size, detector_count),
        )
        chunk_count = math.ceil(volume.size / rows_per_chunk)
        for chunk_index, first in enumerate(range(0, volume.size, rows_per_chunk)):
            last = min(first + rows_per_chunk, volume.size)
            source_chunk = np.asarray(source[first:last], dtype=np.float32)
            output_chunk = source_chunk * volume32[first:last, None]
            if not np.all(np.isfinite(output_chunk)) or np.any(output_chunk < 0.0):
                raise ValueError(f"Invalid scaled values in pixel rows {first}:{last}.")
            output[first:last] = output_chunk
            source_sum += float(np.sum(source_chunk, dtype=np.float64))
            expected_sum += float(
                np.sum(source_chunk.astype(np.float64) * volume[first:last, None])
            )
            output_sum += float(np.sum(output_chunk, dtype=np.float64))
            nonzero_count += int(np.count_nonzero(output_chunk))
            source_hash.update(source_chunk.tobytes(order="C"))
            output_hash.update(output_chunk.tobytes(order="C"))
            if chunk_index % 16 == 0 or chunk_index + 1 == chunk_count:
                print(
                    f"  rows {last:5d}/{volume.size} "
                    f"({100.0 * last / volume.size:6.2f}%)",
                    flush=True,
                )
        output.flush()
    finally:
        if output is not None:
            del output
        if source is not None:
            del source
    os.replace(temporary_path, output_path)
    relative_sum_error = output_sum / expected_sum - 1.0
    if output_path.stat().st_size != source_path.stat().st_size:
        raise ValueError("Scaled matrix byte size changed unexpectedly.")
    if abs(relative_sum_error) > 2.0e-7:
        raise ValueError(f"Scaled matrix sum check failed: {relative_sum_error:.6e}")

    return {
        "pixel_num": int(volume.size),
        "detector_num": int(detector_count),
        "byte_count": int(output_path.stat().st_size),
        "source_sum": source_sum,
        "expected_scaled_sum_float64": expected_sum,
        "actual_scaled_sum_float32": output_sum,
        "relative_sum_error": relative_sum_error,
        "nonzero_count": nonzero_count,
        "source_sha256": source_hash.hexdigest(),
        "output_sha256": output_hash.hexdigest(),
    }


def write_volume_files(output_dir: Path, volume: np.ndarray, metadata: dict[str, object]) -> None:
    np.savetxt(
        output_dir / "polar_cell_volume_mm3.csv",
        volume,
        delimiter=",",
        fmt="%.12g",
        header="volume_mm3",
        comments="",
    )
    volume.astype("<f8").tofile(output_dir / "polar_cell_volume_mm3.float64")
    with (output_dir / "polar_cell_volume_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)


def build_one(
    source_dir: Path,
    output_dir: Path,
    output_suffix: str,
    volume: np.ndarray,
    volume_metadata: dict[str, object],
    rows_per_chunk: int,
) -> None:
    print(f"\nBuilding {output_dir.name}")
    copy_factor_metadata(source_dir, output_dir)
    write_volume_files(output_dir, volume, volume_metadata)
    rotation_error = validate_rotation_invariance(source_dir, volume)
    matrix_metrics = scale_matrix(
        source_dir / "SysMat_polar",
        output_dir / "SysMat_polar",
        volume,
        rows_per_chunk,
    )

    with (source_dir / "factor_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["grid"]["output_suffix"] = output_suffix
    manifest["matrix_kind_before_density_basis"] = manifest.get("matrix_kind")
    manifest["matrix_kind"] = str(manifest.get("matrix_kind", "response")) + "_density_basis"
    manifest["per_emitted_source_photon"] = False
    manifest["underlying_point_response_normalization"] = (
        "per emitted monoenergetic source photon"
    )
    manifest["maps_activity_density"] = True
    manifest["activity_density_units"] = "emitted photons per mm3"
    manifest["parent_factor_dir_before_density_basis"] = str(source_dir.resolve())
    manifest["polar_volume_weighting"] = {
        "enabled": True,
        "forward_model": "y = A * diag(DeltaV_mm3) * rho",
        "transform": "each on-disk pixel row of SysMat_polar multiplied by full polar-cell volume",
        "volume_file": "polar_cell_volume_mm3.float64",
        "volume_csv": "polar_cell_volume_mm3.csv",
        "volume_metadata": volume_metadata,
        "rotation_invariance_max_abs_error_mm3": rotation_error,
        "matrix_metrics": matrix_metrics,
    }
    manifest["generated_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with (output_dir / "factor_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
    print(
        f"  complete: detectors={matrix_metrics['detector_num']} "
        f"sum_error={matrix_metrics['relative_sum_error']:.3e}"
    )


def main() -> None:
    args = parse_args()
    if args.rows_per_chunk < 1:
        raise ValueError("--rows-per-chunk must be positive.")
    repo_root = args.repo_root.resolve()
    factors_root = repo_root / "Factors"
    source_dirs = [factors_root / f"{stem}_{args.source_suffix}" for stem in FACTOR_STEMS]
    output_dirs = [factors_root / f"{stem}_{args.output_suffix}" for stem in FACTOR_STEMS]
    for source_dir in source_dirs:
        if not (source_dir / "SysMat_polar").is_file():
            raise FileNotFoundError(source_dir / "SysMat_polar")
    if args.overwrite:
        for output_dir in output_dirs:
            if output_dir.exists():
                shutil.rmtree(output_dir)
    existing = [str(path) for path in output_dirs if path.exists()]
    if existing:
        raise FileExistsError("Output directories already exist:\n" + "\n".join(existing))

    coordinates = np.loadtxt(
        source_dirs[0] / "coor_polar_full.csv", delimiter=",", dtype=np.float64
    )
    volume, volume_metadata = build_volume_vector(coordinates)
    for source_dir in source_dirs[1:]:
        other = np.loadtxt(source_dir / "coor_polar_full.csv", delimiter=",", dtype=np.float64)
        if not np.array_equal(coordinates, other):
            raise ValueError(f"Coordinate mismatch: {source_dir}")

    print(json.dumps(volume_metadata, indent=2))
    for source_dir, output_dir in zip(source_dirs, output_dirs):
        try:
            build_one(
                source_dir,
                output_dir,
                args.output_suffix,
                volume,
                volume_metadata,
                args.rows_per_chunk,
            )
        except Exception:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            raise


if __name__ == "__main__":
    main()
