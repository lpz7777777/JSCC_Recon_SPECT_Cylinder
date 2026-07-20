#!/usr/bin/env python3
"""Audit the measure represented by the nonuniform polar source grid.

The system-matrix columns are point-source responses. Consequently, the
forward-model unknown is integrated activity per polar cell, not activity
density. A physically uniform source therefore has x_j proportional to the
physical volume represented by sample j.

This script is intentionally analysis-only. It does not modify Factors,
CntStat, or reconstruction outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FACTOR_DIR_NAME = "218keV_RotateNum20"
DEFAULT_RECON_RELATIVE = Path(
    "Results/R/V4L/"
    "ME_R20_E218-440_Ddd7b568a_C1e10_DS1.0_O1_SI2000_XTALK_BG1_N6.61e7"
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--factor-dir",
        type=Path,
        default=repo_root / "Factors" / DEFAULT_FACTOR_DIR_NAME,
    )
    parser.add_argument(
        "--reconstruction-dir",
        type=Path,
        default=repo_root / DEFAULT_RECON_RELATIVE,
        help="Run root containing Polar/, or the Polar directory itself.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "Results/Analysis/PolarSourceMeasure_20260720",
    )
    parser.add_argument("--phantom-radius-mm", type=float, default=120.0)
    parser.add_argument("--phantom-height-mm", type=float, default=30.0)
    parser.add_argument(
        "--profile-max-radius-mm",
        type=float,
        default=108.0,
        help="Avoid partially filled cells at the phantom edge.",
    )
    parser.add_argument("--total-source-count", type=float, default=1.0e10)
    parser.add_argument(
        "--skip-projection-comparison",
        action="store_true",
        help="Only audit geometry and existing images; do not load system matrices.",
    )
    return parser.parse_args()


def load_coordinates(factor_dir: Path) -> np.ndarray:
    path = factor_dir / "coor_polar_full.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    coordinates = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"Unexpected coordinate shape {coordinates.shape}: {path}")
    return coordinates


def midpoint_bounds(values: np.ndarray, lower_limit: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.diff(values) > 0):
        raise ValueError("Grid values must be a strictly increasing 1-D array.")
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


def build_cell_geometry(coordinates: np.ndarray) -> dict[str, np.ndarray]:
    radius = np.hypot(coordinates[:, 0], coordinates[:, 1])
    radius = np.round(radius, decimals=8)
    z_value = np.round(coordinates[:, 2], decimals=8)
    radii = np.unique(radius)
    z_values = np.unique(z_value)
    radial_inner, radial_outer = midpoint_bounds(radii, lower_limit=0.0)
    z_lower, z_upper = midpoint_bounds(z_values)

    ring_count = np.array([np.count_nonzero(radius == r) // z_values.size for r in radii])
    if ring_count[0] != 1 or np.any(ring_count < 1):
        raise ValueError(f"Unexpected points per ring: {ring_count.tolist()}")

    radial_index = np.searchsorted(radii, radius)
    z_index = np.searchsorted(z_values, z_value)
    area_per_sample_by_ring = (
        math.pi * (radial_outer**2 - radial_inner**2) / ring_count
    )
    dz_by_layer = z_upper - z_lower
    volume = area_per_sample_by_ring[radial_index] * dz_by_layer[z_index]

    return {
        "radius": radius,
        "z": z_value,
        "radii": radii,
        "z_values": z_values,
        "radial_inner": radial_inner,
        "radial_outer": radial_outer,
        "z_lower": z_lower,
        "z_upper": z_upper,
        "ring_count": ring_count,
        "radial_index": radial_index,
        "z_index": z_index,
        "area_per_sample_by_ring": area_per_sample_by_ring,
        "volume": volume,
    }


def centered_cylinder_overlap(
    geometry: dict[str, np.ndarray], radius_mm: float, height_mm: float
) -> np.ndarray:
    radial_inner = geometry["radial_inner"]
    radial_outer = np.minimum(geometry["radial_outer"], radius_mm)
    radial_inner_clipped = np.minimum(radial_inner, radius_mm)
    ring_count = geometry["ring_count"]
    area = (
        math.pi
        * np.maximum(radial_outer**2 - radial_inner_clipped**2, 0.0)
        / ring_count
    )

    half_height = 0.5 * height_mm
    lower = np.maximum(geometry["z_lower"], -half_height)
    upper = np.minimum(geometry["z_upper"], half_height)
    dz = np.maximum(upper - lower, 0.0)
    return area[geometry["radial_index"]] * dz[geometry["z_index"]]


def offcenter_disk_overlap_areas(
    coordinates: np.ndarray,
    geometry: dict[str, np.ndarray],
    center_x: float,
    center_y: float,
    disk_radius: float,
    subdivisions: int = 32,
) -> np.ndarray:
    """Integrate overlap between every polar cell and an off-center disk."""
    first_layer = geometry["z_index"] == 0
    xy = coordinates[first_layer, :2]
    points_per_layer = xy.shape[0]
    tiled_xy = np.tile(xy, (geometry["z_values"].size, 1))
    if tiled_xy.shape != coordinates[:, :2].shape or not np.allclose(
        tiled_xy, coordinates[:, :2], atol=1.0e-7
    ):
        raise ValueError("Coordinate rows are not ordered as repeated z-layer XY grids.")
    radial_index = geometry["radial_index"][first_layer]
    theta_center = np.arctan2(xy[:, 1], xy[:, 0])
    area = geometry["area_per_sample_by_ring"][radial_index]
    overlap_fraction = np.zeros(xy.shape[0], dtype=np.float64)
    midpoint = (np.arange(subdivisions, dtype=np.float64) + 0.5) / subdivisions

    for sample_index in range(xy.shape[0]):
        ring_index = radial_index[sample_index]
        r_inner = geometry["radial_inner"][ring_index]
        r_outer = geometry["radial_outer"][ring_index]
        angular_count = geometry["ring_count"][ring_index]
        if angular_count == 1:
            theta_width = 2.0 * math.pi
            theta0 = 0.0
        else:
            theta_width = 2.0 * math.pi / angular_count
            theta0 = theta_center[sample_index] - 0.5 * theta_width

        radial_samples = np.sqrt(
            r_inner**2 + midpoint * (r_outer**2 - r_inner**2)
        )
        theta_samples = theta0 + midpoint * theta_width
        sample_x = radial_samples[:, None] * np.cos(theta_samples[None, :])
        sample_y = radial_samples[:, None] * np.sin(theta_samples[None, :])
        inside = (sample_x - center_x) ** 2 + (sample_y - center_y) ** 2 <= disk_radius**2
        overlap_fraction[sample_index] = np.mean(inside)

    overlap_area_xy = area * overlap_fraction
    return overlap_area_xy[np.arange(coordinates.shape[0]) % points_per_layer]


def build_source_models(
    coordinates: np.ndarray,
    geometry: dict[str, np.ndarray],
    background_overlap: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    energies = (218, 440)
    yields = {218: 0.114, 440: 0.261}
    rod_diameters = np.arange(10.0, 31.0, 4.0)
    rod_energies = (218, 440, 218, 440, 218, 440)
    activity_ratio = 6.0
    half_height = 15.0
    dz_overlap = np.maximum(
        np.minimum(geometry["z_upper"], half_height)
        - np.maximum(geometry["z_lower"], -half_height),
        0.0,
    )

    radius = geometry["radius"]
    equal_density = {
        energy: ((radius <= 120.0) & (np.abs(geometry["z"]) <= half_height)).astype(np.float64)
        for energy in energies
    }
    volume_activity = {energy: background_overlap.copy() for energy in energies}
    rod_overlap_closure = []

    for index, (diameter, energy) in enumerate(zip(rod_diameters, rod_energies)):
        theta = index * math.pi / 3.0
        center_x = 60.0 * math.cos(theta)
        center_y = 60.0 * math.sin(theta)
        rod_radius = 0.5 * diameter
        point_inside = (
            (coordinates[:, 0] - center_x) ** 2
            + (coordinates[:, 1] - center_y) ** 2
            <= rod_radius**2
        ) & (np.abs(geometry["z"]) <= half_height)
        equal_density[energy][point_inside] += activity_ratio - 1.0

        overlap_area = offcenter_disk_overlap_areas(
            coordinates, geometry, center_x, center_y, rod_radius
        )
        overlap_volume = overlap_area * dz_overlap[geometry["z_index"]]
        volume_activity[energy] += (activity_ratio - 1.0) * overlap_volume
        numerical_volume = float(np.sum(overlap_volume))
        analytic_volume = math.pi * rod_radius**2 * 30.0
        rod_overlap_closure.append(
            {
                "rod_index": index + 1,
                "energy_keV": energy,
                "diameter_mm": float(diameter),
                "relative_error": numerical_volume / analytic_volume - 1.0,
            }
        )

    source_models: dict[str, np.ndarray] = {}
    yield_sum = sum(yields.values())
    for model_name, activity_maps in (
        ("equal_point", equal_density),
        ("physical_volume", volume_activity),
    ):
        for energy in energies:
            normalized = activity_maps[energy] / np.sum(activity_maps[energy])
            source_models[f"{model_name}_{energy}"] = normalized * yields[energy] / yield_sum

    diagnostics = {
        "rod_overlap_quadrature_subdivisions": 32,
        "rod_overlap_volume_closure": rod_overlap_closure,
        "source_fraction": {
            key: float(np.sum(value)) for key, value in source_models.items()
        },
        "equal_vs_volume_spatial_difference": {},
    }
    for energy in energies:
        equal_source = source_models[f"equal_point_{energy}"]
        volume_source = source_models[f"physical_volume_{energy}"]
        equal = equal_source / np.sum(equal_source)
        volume = volume_source / np.sum(volume_source)
        diagnostics["equal_vs_volume_spatial_difference"][str(energy)] = {
            "total_variation": float(0.5 * np.sum(np.abs(equal - volume))),
            "fraction_inside_r18_equal": float(np.sum(equal[geometry["radius"] <= 18.0])),
            "fraction_inside_r18_volume": float(np.sum(volume[geometry["radius"] <= 18.0])),
            "fraction_inside_r36_equal": float(np.sum(equal[geometry["radius"] <= 36.0])),
            "fraction_inside_r36_volume": float(np.sum(volume[geometry["radius"] <= 36.0])),
        }
    return source_models, diagnostics


def load_rotmat(path: Path, expected_pixels: int) -> np.ndarray:
    rotmat = np.loadtxt(path, delimiter=",", dtype=np.int64)
    if rotmat.ndim != 2 or rotmat.shape[0] != expected_pixels:
        raise ValueError(f"Unexpected rotation matrix shape {rotmat.shape}: {path}")
    if rotmat.min() < 1 or rotmat.max() > expected_pixels:
        raise ValueError(f"Rotation indices out of range: {path}")
    return rotmat - 1


def project_one_matrix(
    sysmat_path: Path,
    rotmat: np.ndarray,
    sources: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    import torch

    pixel_count = rotmat.shape[0]
    element_count = sysmat_path.stat().st_size // np.dtype(np.float32).itemsize
    if element_count % pixel_count:
        raise ValueError(f"System-matrix size is incompatible with {pixel_count} pixels: {sysmat_path}")
    detector_count = element_count // pixel_count
    matrix_mmap = np.memmap(
        sysmat_path, dtype=np.float32, mode="r", shape=(pixel_count, detector_count)
    )
    matrix = torch.from_numpy(np.asarray(matrix_mmap).T.copy()).cuda()
    source_gpu = {
        name: torch.from_numpy(value.astype(np.float32, copy=False)).cuda()
        for name, value in sources.items()
    }
    outputs = {
        name: np.zeros((rotmat.shape[1], detector_count), dtype=np.float32)
        for name in sources
    }
    with torch.no_grad():
        for view in range(rotmat.shape[1]):
            indices = torch.from_numpy(rotmat[:, view].copy()).cuda()
            for name, source in source_gpu.items():
                outputs[name][view, :] = (
                    matrix @ (source[indices] / rotmat.shape[1])
                ).cpu().numpy()
    del matrix, matrix_mmap, source_gpu
    torch.cuda.empty_cache()
    return outputs


def projection_shape_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed64 = observed.astype(np.float64)
    predicted64 = predicted.astype(np.float64)
    observed_sum = float(np.sum(observed64))
    predicted_sum = float(np.sum(predicted64))
    observed_shape = observed64 / observed_sum
    predicted_shape = predicted64 / predicted_sum
    relative_l2 = float(
        np.linalg.norm(predicted_shape - observed_shape) / np.linalg.norm(observed_shape)
    )
    correlation = float(np.corrcoef(observed_shape.ravel(), predicted_shape.ravel())[0, 1])
    return {
        "observed_events": observed_sum,
        "predicted_events": predicted_sum,
        "observed_to_predicted_total": observed_sum / predicted_sum,
        "normalized_shape_relative_l2": relative_l2,
        "normalized_shape_correlation": correlation,
    }


def compare_projections(
    repo_root: Path,
    factor_suffix: str,
    source_models: dict[str, np.ndarray],
    total_source_count: float,
    output_dir: Path,
) -> dict[str, object]:
    suffix_part = f"_{factor_suffix}" if factor_suffix else ""
    factor_names = {
        "A218": f"218keV_RotateNum20{suffix_part}",
        "A440": f"440keV_RotateNum20{suffix_part}",
        "C440to218": f"440keV_to218win_RotateNum20{suffix_part}",
    }
    factor_dirs = {name: repo_root / "Factors" / value for name, value in factor_names.items()}
    with (factor_dirs["A218"] / "factor_manifest.json").open("r", encoding="utf-8") as handle:
        factor_manifest = json.load(handle)
    matrix_sources = source_models
    if factor_manifest.get("maps_activity_density", False):
        volume = np.fromfile(
            factor_dirs["A218"] / "polar_cell_volume_mm3.float64", dtype="<f8"
        )
        if volume.size != next(iter(source_models.values())).size or np.any(volume <= 0.0):
            raise ValueError("Invalid density-basis polar-cell volume vector.")
        matrix_sources = {
            name: integrated_activity / volume
            for name, integrated_activity in source_models.items()
        }
    rotmat = load_rotmat(
        factor_dirs["A218"] / "RotMat_full.csv", next(iter(matrix_sources.values())).size
    )

    direct_218 = project_one_matrix(
        factor_dirs["A218"] / "SysMat_polar",
        rotmat,
        {name: value for name, value in matrix_sources.items() if name.endswith("_218")},
    )
    direct_440 = project_one_matrix(
        factor_dirs["A440"] / "SysMat_polar",
        rotmat,
        {name: value for name, value in matrix_sources.items() if name.endswith("_440")},
    )
    cross_440 = project_one_matrix(
        factor_dirs["C440to218"] / "SysMat_polar",
        rotmat,
        {name: value for name, value in matrix_sources.items() if name.endswith("_440")},
    )

    dataset = "ContrastPhantom_DualEnergy_10_30_240_30_225Ac_1e10.csv"
    observed_218 = np.loadtxt(
        repo_root / "CntStat/218keV_RotateNum20_Geant4JSCC" / f"CntStat_{dataset}",
        delimiter=",",
        dtype=np.float32,
    )
    observed_440 = np.loadtxt(
        repo_root / "CntStat/440keV_RotateNum20_Geant4JSCC" / f"CntStat_{dataset}",
        delimiter=",",
        dtype=np.float32,
    )

    predictions: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for model_name in ("equal_point", "physical_volume"):
        predicted_218 = (
            direct_218[f"{model_name}_218"] + cross_440[f"{model_name}_440"]
        ) * total_source_count
        predicted_440 = direct_440[f"{model_name}_440"] * total_source_count
        predictions[model_name] = {"218": predicted_218, "440": predicted_440}
        metrics[model_name] = {
            "218_window": projection_shape_metrics(observed_218, predicted_218),
            "440_window": projection_shape_metrics(observed_440, predicted_440),
        }

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)
    for row, (energy, observed) in enumerate((("218", observed_218), ("440", observed_440))):
        observed_detector = np.sum(observed, axis=0).astype(np.float64)
        observed_detector /= np.sum(observed_detector)
        axis = axes[row, 0]
        axis.plot(observed_detector, color="black", linewidth=1.0, label="Geant4")
        for model_name, color in (("equal_point", "#b44b32"), ("physical_volume", "#245b7a")):
            profile = np.sum(predictions[model_name][energy], axis=0).astype(np.float64)
            profile /= np.sum(profile)
            axis.plot(profile, color=color, linewidth=0.8, alpha=0.85, label=model_name)
        axis.set(title=f"{energy} window: normalized detector profile", xlabel="Detector index", ylabel="Fraction")
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)

        axis = axes[row, 1]
        labels = ["equal point", "physical volume"]
        values = [
            metrics["equal_point"][f"{energy}_window"]["normalized_shape_relative_l2"],
            metrics["physical_volume"][f"{energy}_window"]["normalized_shape_relative_l2"],
        ]
        axis.bar(labels, values, color=["#b44b32", "#245b7a"])
        axis.set(title=f"{energy} window: full CntStat shape error", ylabel="Normalized relative L2")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Geant4 CntStat versus polar source discretization")
    fig.savefig(output_dir / "projection_model_comparison.png", dpi=220)
    plt.close(fig)
    return {
        "factor_suffix": factor_suffix,
        "total_source_count": total_source_count,
        "metrics": metrics,
    }


def find_polar_dir(run_dir: Path) -> Path | None:
    if (run_dir / "run_manifest.json").is_file():
        return run_dir
    if (run_dir / "Polar" / "run_manifest.json").is_file():
        return run_dir / "Polar"
    return None


def load_reconstruction_images(run_dir: Path, expected_size: int) -> dict[str, np.ndarray]:
    polar_dir = find_polar_dir(run_dir)
    if polar_dir is None:
        return {}
    image_files = {
        "440 keV": "Image_S_440keV",
        "218 keV corrected": "Image_S_218keV_CrossTalkCorrected",
        "440+218 corrected": "Image_S_(440_218)keV_CrossTalkCorrected",
    }
    images: dict[str, np.ndarray] = {}
    for label, file_name in image_files.items():
        path = polar_dir / file_name
        if not path.is_file():
            continue
        image = np.fromfile(path, dtype="<f4").astype(np.float64)
        if image.size != expected_size:
            raise ValueError(f"{path} contains {image.size} values; expected {expected_size}.")
        images[label] = image
    return images


def rod_exclusion_mask(coordinates: np.ndarray) -> np.ndarray:
    # Exclude all six hot cylinders plus half a radial sample to keep the
    # background profile insensitive to partial-volume spill-in.
    rod_diameters = np.arange(10.0, 31.0, 4.0)
    mask = np.ones(coordinates.shape[0], dtype=bool)
    for index, diameter in enumerate(rod_diameters):
        theta = index * math.pi / 3.0
        center_x = 60.0 * math.cos(theta)
        center_y = 60.0 * math.sin(theta)
        exclusion_radius = 0.5 * diameter + 3.0
        distance = np.hypot(coordinates[:, 0] - center_x, coordinates[:, 1] - center_y)
        mask &= distance > exclusion_radius
    return mask


def ring_profile(
    values: np.ndarray,
    geometry: dict[str, np.ndarray],
    base_mask: np.ndarray,
) -> np.ndarray:
    profile = np.full(geometry["radii"].shape, np.nan, dtype=np.float64)
    for index in range(geometry["radii"].size):
        selected = base_mask & (geometry["radial_index"] == index)
        if np.any(selected):
            profile[index] = np.median(values[selected])
    return profile


def normalized_profile(profile: np.ndarray, radii: np.ndarray) -> np.ndarray:
    reference = np.isfinite(profile) & (radii >= 30.0) & (radii <= 108.0)
    scale = np.median(profile[reference])
    return profile / scale


def coefficient_of_variation(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.std(values) / np.mean(values))


def write_radial_csv(path: Path, geometry: dict[str, np.ndarray], overlap: np.ndarray) -> None:
    overlap_by_ring = np.zeros_like(geometry["radii"])
    first_layer = geometry["z_index"] == 0
    for index in range(geometry["radii"].size):
        selected = first_layer & (geometry["radial_index"] == index)
        overlap_by_ring[index] = np.mean(overlap[selected]) if np.any(selected) else 0.0

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "radius_mm",
                "radial_inner_mm",
                "radial_outer_mm",
                "points_per_ring",
                "area_per_sample_mm2",
                "full_cell_volume_mm3",
                "phantom_overlap_volume_mm3",
                "equal_point_implied_density_per_mm3",
            ]
        )
        dz = geometry["z_upper"][0] - geometry["z_lower"][0]
        for index, radius in enumerate(geometry["radii"]):
            full_volume = geometry["area_per_sample_by_ring"][index] * dz
            writer.writerow(
                [
                    f"{radius:.8g}",
                    f"{geometry['radial_inner'][index]:.8g}",
                    f"{geometry['radial_outer'][index]:.8g}",
                    int(geometry["ring_count"][index]),
                    f"{geometry['area_per_sample_by_ring'][index]:.12g}",
                    f"{full_volume:.12g}",
                    f"{overlap_by_ring[index]:.12g}",
                    f"{1.0 / full_volume:.12g}",
                ]
            )


def make_figure(
    path: Path,
    geometry: dict[str, np.ndarray],
    overlap: np.ndarray,
    images: dict[str, np.ndarray],
    profile_mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    radii = geometry["radii"]
    area = geometry["area_per_sample_by_ring"]
    dz = geometry["z_upper"][0] - geometry["z_lower"][0]
    volume_by_ring = area * dz
    expected_profile = volume_by_ring / np.median(volume_by_ring[(radii >= 30) & (radii <= 108)])
    equal_density = (1.0 / volume_by_ring)
    equal_density /= np.median(equal_density[(radii >= 30) & (radii <= 108)])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(radii, geometry["ring_count"], "o-", color="#245b7a", linewidth=1.5)
    axis.set(title="Angular samples per radius", xlabel="Radius (mm)", ylabel="Samples / ring")
    axis.grid(alpha=0.25)

    axis = axes[0, 1]
    axis.plot(radii, area, "o-", color="#b44b32", label="Area represented by one sample")
    axis.set(title="Nonuniform polar-cell area", xlabel="Radius (mm)", ylabel="Area / sample (mm$^2$)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)

    axis = axes[1, 0]
    axis.plot(radii, expected_profile, "o-", color="#2f7d4a", label="Uniform volume: expected raw x")
    axis.plot(radii, equal_density, "s--", color="#7a4e9d", label="Equal x: implied density")
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set(
        title="Source-measure mismatch (normalized)",
        xlabel="Radius (mm)",
        ylabel="Relative value",
        xlim=(0, 125),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)

    metrics: dict[str, dict[str, float]] = {}
    axis = axes[1, 1]
    colors = ["#245b7a", "#b44b32", "#2f7d4a"]
    for (label, image), color in zip(images.items(), colors):
        raw_profile = ring_profile(image, geometry, profile_mask)
        density_profile = ring_profile(image / geometry["volume"], geometry, profile_mask)
        raw_normalized = normalized_profile(raw_profile, radii)
        density_normalized = normalized_profile(density_profile, radii)
        valid = np.isfinite(raw_normalized) & (radii <= 108.0)
        expected_valid = expected_profile[valid]
        correlation = float(np.corrcoef(raw_normalized[valid], expected_valid)[0, 1])
        metrics[label] = {
            "raw_ring_profile_cv": coefficient_of_variation(raw_normalized[valid]),
            "density_corrected_ring_profile_cv": coefficient_of_variation(density_normalized[valid]),
            "raw_profile_correlation_with_cell_volume": correlation,
        }
        axis.plot(radii, raw_normalized, "-", color=color, linewidth=1.7, label=f"{label}: raw x")
        axis.plot(
            radii,
            density_normalized,
            "--",
            color=color,
            linewidth=1.4,
            label=f"{label}: x / cell volume",
        )
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set(
        title="Existing Geant4 reconstruction: background radial median",
        xlabel="Radius (mm)",
        ylabel="Relative to 30-108 mm median",
        xlim=(0, 110),
    )
    axis.grid(alpha=0.25)
    if images:
        axis.legend(frameon=False, fontsize=8, ncol=2)
    else:
        axis.text(0.5, 0.5, "Reconstruction output not found", ha="center", va="center", transform=axis.transAxes)

    fig.suptitle("Polar source-measure audit", fontsize=15)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return metrics


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    factor_dir = args.factor_dir.resolve()
    reconstruction_dir = args.reconstruction_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    coordinates = load_coordinates(factor_dir)
    geometry = build_cell_geometry(coordinates)
    overlap = centered_cylinder_overlap(
        geometry, args.phantom_radius_mm, args.phantom_height_mm
    )
    expected_volume = math.pi * args.phantom_radius_mm**2 * args.phantom_height_mm
    represented_volume = float(np.sum(overlap))

    profile_mask = (
        (geometry["radius"] <= args.profile_max_radius_mm)
        & (np.abs(geometry["z"]) <= args.phantom_height_mm / 2.0 - 1.5)
        & rod_exclusion_mask(coordinates)
    )
    images = load_reconstruction_images(reconstruction_dir, coordinates.shape[0])
    source_models, source_model_diagnostics = build_source_models(
        coordinates, geometry, overlap
    )

    write_radial_csv(output_dir / "radial_cell_metrics.csv", geometry, overlap)
    reconstruction_metrics = make_figure(
        output_dir / "polar_source_measure_audit.png",
        geometry,
        overlap,
        images,
        profile_mask,
    )
    projection_comparison = None
    if not args.skip_projection_comparison:
        factor_base = "218keV_RotateNum20"
        if factor_dir.name == factor_base:
            factor_suffix = ""
        elif factor_dir.name.startswith(factor_base + "_"):
            factor_suffix = factor_dir.name[len(factor_base) + 1 :]
        else:
            raise ValueError(f"Cannot derive factor suffix from {factor_dir.name}")
        projection_comparison = compare_projections(
            repo_root,
            factor_suffix,
            source_models,
            args.total_source_count,
            output_dir,
        )

    radii = geometry["radii"]
    volume_ring = geometry["area_per_sample_by_ring"] * (
        geometry["z_upper"][0] - geometry["z_lower"][0]
    )
    interior = radii <= args.profile_max_radius_mm
    summary = {
        "factor_dir": str(factor_dir),
        "reconstruction_dir": str(reconstruction_dir),
        "coordinate_count": int(coordinates.shape[0]),
        "z_layer_count": int(geometry["z_values"].size),
        "radii_mm": geometry["radii"].tolist(),
        "points_per_ring": geometry["ring_count"].astype(int).tolist(),
        "phantom": {
            "radius_mm": args.phantom_radius_mm,
            "height_mm": args.phantom_height_mm,
            "analytic_volume_mm3": expected_volume,
            "represented_volume_mm3": represented_volume,
            "relative_closure_error": represented_volume / expected_volume - 1.0,
        },
        "cell_volume": {
            "minimum_interior_mm3": float(np.min(volume_ring[interior])),
            "maximum_interior_mm3": float(np.max(volume_ring[interior])),
            "max_to_min_ratio_interior": float(
                np.max(volume_ring[interior]) / np.min(volume_ring[interior])
            ),
            "center_mm3": float(volume_ring[0]),
            "r6_mm3": float(volume_ring[np.where(radii == 6.0)[0][0]]),
            "r36_mm3": float(volume_ring[np.where(radii == 36.0)[0][0]]),
            "r42_mm3": float(volume_ring[np.where(radii == 42.0)[0][0]]),
            "r108_mm3": float(volume_ring[np.where(radii == 108.0)[0][0]]),
        },
        "interpretation": {
            "matrix_column": "detection probability per emitted photon at one polar sample",
            "reconstruction_unknown": "integrated emitted activity per polar cell",
            "uniform_physical_density": "x_j proportional to polar-cell overlap volume",
            "density_display": "divide reconstructed x_j by full polar-cell volume before interpolation",
            "uniform_fov_geant4_study": "equal-weight polar point array, not a uniform volume cylinder",
        },
        "source_model_diagnostics": source_model_diagnostics,
        "existing_reconstruction_metrics": reconstruction_metrics,
        "projection_comparison": projection_comparison,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
