"""Independent uniform-FOV closure test for the density-basis Compton MLEM operator.

Half A of a uniform List creates Sensi_d.  This program reads disjoint Half B,
performs one streaming MLEM update from a uniform activity-density image, and
checks that the update remains uniform.  It never materializes all List rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


TOOL_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TOOL_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from compton_event_response import build_detector_position_variance  # noqa: E402
from spect_sensitivity import ComptonPhysicsConfig  # noqa: E402
from spect_sensitivity.io import (  # noqa: E402
    count_event_rows,
    expand_compton_paths,
    iter_event_batches,
    load_system_matrix,
    resolve_dataset,
)
from spect_sensitivity.kernel import accumulate_event_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--compton-list", type=Path, nargs="+", required=True)
    parser.add_argument("--sensi-d", type=Path, required=True)
    parser.add_argument("--source-photons", type=float, required=True)
    parser.add_argument("--event-start-fraction", type=float, required=True)
    parser.add_argument("--event-fraction", type=float, required=True)
    parser.add_argument("--energy-mev", type=float, default=0.440)
    parser.add_argument("--rotate-num", type=int, default=20)
    parser.add_argument("--energy-resolution-fwhm", type=float, default=0.13)
    parser.add_argument("--energy-resolution-reference-kev", type=float, default=511.0)
    parser.add_argument("--energy-threshold-sum-mev", type=float, default=0.350)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rotation_average(values: np.ndarray, rotation: np.ndarray, rotate_num: int) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    for rotate_index in range(rotate_num):
        result += values[rotation[:, rotate_index] - 1]
    return result / rotate_num


def write_figure(coordinates: np.ndarray, ratio: np.ndarray, output: Path) -> None:
    z_values = np.unique(coordinates[:, 2])
    center_z = z_values[np.argmin(np.abs(z_values))]
    z_mask = np.isclose(coordinates[:, 2], center_z)
    xz_mask = np.isclose(coordinates[:, 1], 0.0)
    if not np.any(xz_mask):
        nearest_y = np.unique(coordinates[:, 1])[np.argmin(np.abs(np.unique(coordinates[:, 1])))]
        xz_mask = np.isclose(coordinates[:, 1], nearest_y)

    radius = np.linalg.norm(coordinates[:, :2], axis=1)
    radial_values = np.unique(np.round(radius, 5))
    radial_mean = np.asarray([np.mean(ratio[np.isclose(radius, value, atol=1e-4)]) for value in radial_values])
    lo, hi = np.quantile(ratio, [0.01, 0.99])
    lo = min(lo, 0.98)
    hi = max(hi, 1.02)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=170)
    scatter = axes[0].scatter(
        coordinates[z_mask, 0], coordinates[z_mask, 1], c=ratio[z_mask], s=8,
        cmap="coolwarm", vmin=lo, vmax=hi, linewidths=0,
    )
    axes[0].set_aspect("equal")
    axes[0].set_title(f"Closure ratio at z={center_z:g} mm")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    fig.colorbar(scatter, ax=axes[0], label="one-MLEM-update / uniform input")

    scatter = axes[1].scatter(
        coordinates[xz_mask, 0], coordinates[xz_mask, 2], c=ratio[xz_mask], s=8,
        cmap="coolwarm", vmin=lo, vmax=hi, linewidths=0,
    )
    axes[1].set_aspect("auto")
    axes[1].set_title("Nearest y=0 x-z section")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("z (mm)")
    fig.colorbar(scatter, ax=axes[1], label="one-MLEM-update / uniform input")

    axes[2].plot(radial_values, radial_mean, color="#176b87", linewidth=1.8)
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_ylim(lo, hi)
    axes[2].set_title("Radial mean closure ratio")
    axes[2].set_xlabel("radius (mm)")
    axes[2].set_ylabel("ratio")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0 <= args.event_start_fraction < 1 or not 0 < args.event_fraction <= 1:
        raise ValueError("Invalid List interval fractions.")
    if args.event_start_fraction + args.event_fraction > 1.0 + 1e-12:
        raise ValueError("The selected List interval exceeds the input.")

    factor = args.factor_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        torch.cuda.set_device(device)

    paths = expand_compton_paths(tuple(path.resolve() for path in args.compton_list))
    dataset = resolve_dataset(
        factor_dir=factor,
        compton_paths=paths,
        system_matrix_path=factor / "SysMat_polar",
        detector_path=factor / "Detector.csv",
        coordinate_path=factor / "coor_polar_full.csv",
        rotation_path=factor / "RotMat_full.csv",
        rotate_num=args.rotate_num,
        expected_detector_count=10496,
        apply_rotation_average=True,
    )
    volumes = np.fromfile(factor / "polar_cell_volume_mm3.float64", dtype=np.float64)
    if volumes.size != dataset.pixel_count:
        raise ValueError("Polar volume file does not match the factor grid.")
    source_volume = float(volumes.sum(dtype=np.float64))
    sensi_d = np.fromfile(args.sensi_d.resolve(), dtype=np.float32).astype(np.float64)
    if sensi_d.size != dataset.pixel_count or np.any(sensi_d <= 0):
        raise ValueError("Sensi_d has invalid dimensions or non-positive values.")

    total_rows = sum(count_event_rows(paths))
    start_row = int(total_rows * args.event_start_fraction)
    selected_rows = int(total_rows * args.event_fraction)
    represented_photons = args.source_photons * selected_rows / total_rows
    if represented_photons <= 0 or start_row + selected_rows > total_rows:
        raise ValueError("The requested List interval represents zero or too many photons.")

    resolution = args.energy_resolution_fwhm * (args.energy_resolution_reference_kev / 1000.0 / args.energy_mev) ** 0.5
    physics = ComptonPhysicsConfig(
        energy_mev=args.energy_mev,
        energy_resolution_662kev=args.energy_resolution_fwhm,
        energy_resolution_reference_kev=args.energy_resolution_reference_kev,
        energy_threshold_sum_mev=args.energy_threshold_sum_mev,
    )
    detector = torch.from_numpy(dataset.detector_coordinates).to(device)
    coordinates = torch.from_numpy(dataset.voxel_coordinates).to(device)
    system_matrix = load_system_matrix(dataset, device)
    sigma1 = build_detector_position_variance(detector, physics.delta_r1_mm)
    sigma2 = build_detector_position_variance(detector, physics.delta_r2_mm)
    accumulator = torch.zeros(dataset.pixel_count, dtype=torch.float64, device=device)
    diagnostics: dict[str, int] = {}
    generator = torch.Generator(device=device)
    generator.manual_seed(20260727)

    for index, batch in enumerate(
        iter_event_batches(paths, args.batch_size, selected_rows, initial_skip_rows=start_row), start=1
    ):
        event_sum, batch_diagnostics = accumulate_event_batch(
            batch.values.to(device, non_blocking=True), physics, detector, sigma1, sigma2,
            coordinates, system_matrix, generator, input_energies_already_smeared=True,
        )
        accumulator += event_sum.to(torch.float64)
        for name, value in batch_diagnostics.to_dict().items():
            diagnostics[name] = diagnostics.get(name, 0) + int(value)
        if index % 1000 == 0 or index * args.batch_size >= selected_rows:
            print(f"Closure batches={index} rows~{min(index * args.batch_size, selected_rows)}/{selected_rows} kept={diagnostics.get('kept_events', 0)}")

    raw_sensitivity = accumulator.detach().cpu().numpy() * source_volume / represented_photons
    test_sensitivity = rotation_average(raw_sensitivity, dataset.rotation_matrix, args.rotate_num)
    ratio = test_sensitivity / sensi_d
    updated_density = represented_photons / source_volume * ratio
    volume_mean = float(np.sum(ratio * volumes, dtype=np.float64) / source_volume)
    metrics = {
        "factor_dir": str(factor),
        "sensi_d": str(args.sensi_d.resolve()),
        "list_files": [str(path) for path in paths],
        "event_interval": {
            "start_row": start_row,
            "selected_rows": selected_rows,
            "total_rows": total_rows,
            "represented_photons": represented_photons,
        },
        "physics": {**physics.to_dict(), "resolved_energy_at_440_mev": resolution},
        "diagnostics": diagnostics,
        "closure_ratio": {
            "volume_weighted_mean": volume_mean,
            "arithmetic_mean": float(np.mean(ratio)),
            "std": float(np.std(ratio)),
            "cv": float(np.std(ratio) / np.mean(ratio)),
            "min": float(np.min(ratio)),
            "median": float(np.median(ratio)),
            "max": float(np.max(ratio)),
        },
        "uniform_input_density_photons_per_mm3": represented_photons / source_volume,
        "updated_density_photons_per_mm3": {
            "min": float(np.min(updated_density)),
            "mean": float(np.mean(updated_density)),
            "max": float(np.max(updated_density)),
        },
    }
    test_sensitivity.astype(np.float32).tofile(output / "Sensi_d_independent_half")
    ratio.astype(np.float32).tofile(output / "UniformOneStepMLEM_Ratio")
    (output / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_figure(dataset.voxel_coordinates, ratio, output / "UniformOneStepMLEM_Closure.png")
    print(json.dumps(metrics["closure_ratio"], indent=2))


if __name__ == "__main__":
    with torch.no_grad():
        main()
