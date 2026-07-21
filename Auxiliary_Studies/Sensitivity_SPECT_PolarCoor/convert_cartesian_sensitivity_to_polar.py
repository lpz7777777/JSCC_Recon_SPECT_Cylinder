"""Convert Cartesian point efficiency to density-basis polar Sensi_d."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata


def rotate_average(values, rotation_path, rotate_num):
    rotation = np.loadtxt(rotation_path, delimiter=",", dtype=np.int64)
    if rotation.shape[0] != values.size or rotation.shape[1] < rotate_num:
        raise ValueError(f"Unexpected rotation shape {rotation.shape}")
    result = np.zeros_like(values, dtype=np.float64)
    for index in range(rotate_num):
        result += values[rotation[:, index] - 1]
    return result / rotate_num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cartesian-result-dir", type=Path, required=True)
    parser.add_argument("--cartesian-input-dir", type=Path, required=True)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rotate-num", type=int, default=20)
    parser.add_argument("--install-to-factor-dir", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cart_result = args.cartesian_result_dir.resolve()
    cart_input = args.cartesian_input_dir.resolve()
    factor = args.factor_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    coordinates = np.loadtxt(cart_input / "coor_cartesian_cylindrical.csv", delimiter=",")
    epsilon_cart = np.fromfile(cart_result / "Sensi_d", dtype=np.float32).astype(np.float64)
    if epsilon_cart.size != coordinates.shape[0]:
        raise ValueError("Cartesian sensitivity and coordinate counts differ")
    run_metadata = json.loads((cart_result / "run_metadata.json").read_text(encoding="utf-8"))
    target_efficiency = float(run_metadata["normalization"]["accepted_events_per_photon"])

    polar_coordinates = np.loadtxt(factor / "coor_polar_full.csv", delimiter=",")
    epsilon_polar = np.empty(polar_coordinates.shape[0], dtype=np.float64)
    for z in np.unique(polar_coordinates[:, 2]):
        source_mask = np.isclose(coordinates[:, 2], z)
        target_mask = np.isclose(polar_coordinates[:, 2], z)
        epsilon_polar[target_mask] = griddata(
            coordinates[source_mask, :2],
            epsilon_cart[source_mask],
            polar_coordinates[target_mask, :2],
            method="linear",
        )
    if not np.isfinite(epsilon_polar).all() or np.any(epsilon_polar < 0):
        raise ValueError("Cartesian-to-polar interpolation produced invalid efficiencies")

    epsilon_polar = rotate_average(
        epsilon_polar, factor / "RotMat_full.csv", args.rotate_num
    )
    volumes = np.fromfile(factor / "polar_cell_volume_mm3.float64", dtype=np.float64)
    source_volume = float(volumes.sum(dtype=np.float64))
    efficiency_before = float(np.sum(epsilon_polar * volumes) / source_volume)
    absolute_scale = target_efficiency / efficiency_before
    epsilon_polar *= absolute_scale
    sensi_d = epsilon_polar * volumes
    efficiency_after = float(np.sum(sensi_d) / source_volume)
    relative_error = abs(efficiency_after / target_efficiency - 1.0)
    if relative_error > 5e-7:
        raise RuntimeError(f"Efficiency closure failed: {relative_error:.3e}")

    epsilon_polar.astype(np.float32).tofile(output / "Sensi_d_point_efficiency")
    sensi_d.astype(np.float32).tofile(output / "Sensi_d")
    metadata = {
        "method": "Cartesian point response -> polar interpolation -> DeltaV density basis",
        "cartesian_result_dir": str(cart_result),
        "cartesian_input_dir": str(cart_input),
        "factor_dir": str(factor),
        "target_accepted_events_per_photon": target_efficiency,
        "polar_volume_weighted_efficiency_before_scale": efficiency_before,
        "cartesian_to_polar_absolute_scale": absolute_scale,
        "polar_volume_weighted_efficiency_after_scale": efficiency_after,
        "relative_efficiency_closure_error": relative_error,
        "equation": "Sensi_d_polar[j] = epsilon_d_polar[j] * DeltaV_mm3[j]",
    }
    (output / "cartesian_to_polar_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if args.install_to_factor_dir:
        destination = factor / "Sensi_d"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {destination}")
        temporary = factor / ".Sensi_d.cartesian.tmp"
        shutil.copyfile(output / "Sensi_d", temporary)
        temporary.replace(destination)
        metadata["installed_sensitivity"] = str(destination)
        (output / "cartesian_to_polar_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
