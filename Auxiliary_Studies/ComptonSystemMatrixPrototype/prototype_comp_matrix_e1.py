from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "Result" / "prototype_e1"
DEFAULT_DATA_PATH = PROJECT_ROOT / "Auxiliary_Studies" / "FreePath" / "data.txt"
ELECTRON_REST_MEV = 0.511
NAI_DENSITY_G_CM3 = 3.67
FRONT_LAYER_CRYSTAL_MM = (3.0, 3.0, 3.0)
REAR_LAYER_CRYSTAL_MM = (2.0, 6.0, 2.0)
EPS = 1.0e-12


@dataclass
class PrototypeSummary:
    detector_csv: str
    data_path: str
    incident_energy_mev: float
    hemisphere: str
    device: str
    dtype: str
    front_detector_count: int
    rear_detector_count: int
    voxel_count: int
    e1_bin_count: int
    e1_min_mev: float
    e1_max_mev: float
    energy_resolution_662kev: float
    effective_energy_resolution: float
    energy_blur_enabled: bool
    detector_sample_shape: list[int]
    fov_shape: list[int]
    voxel_size_mm: list[float]
    output_dir: str
    response_forward_sum: float
    response_reverse_sum: float
    response_sum: float
    response_forward_max: float
    response_reverse_max: float
    response_max: float
    center_voxel_index: int
    slice_front_detector_id_1based: int
    slice_rear_detector_id_1based: int
    slice_e1_bin_index: int
    slice_e1_center_mev: float
    slice_y_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal analytic Compton system-matrix prototype implemented with torch. "
            "The script builds a small A(voxel, det1, det2, e1_bin) tensor for a tiny FOV "
            "and a small detector subset."
        )
    )
    parser.add_argument(
        "--detector-csv",
        type=Path,
        default=PROJECT_ROOT / "Factors" / "511keV_RotateNum20" / "Detector.csv",
        help="Detector.csv path.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="NaI cross-section table used for attenuation interpolation.",
    )
    parser.add_argument("--incident-energy-mev", type=float, default=0.511, help="Incident photon energy in MeV.")
    parser.add_argument(
        "--ene-resolution-662kev",
        type=float,
        default=0.1,
        help="Reference energy resolution FWHM at 662 keV.",
    )
    parser.add_argument(
        "--disable-energy-blur",
        action="store_true",
        help="Disable energy-resolution blur and assign weight to e1 bins purely by geometry-implied e1.",
    )
    parser.add_argument(
        "--hemisphere",
        choices=["positive", "negative", "all"],
        default="positive",
        help="Select detector subset from positive-y, negative-y, or all detectors.",
    )
    parser.add_argument("--front-det-count", type=int, default=8, help="Number of front-layer detectors kept.")
    parser.add_argument("--rear-det-count", type=int, default=8, help="Number of rear-layer detectors kept.")
    parser.add_argument("--target-x-mm", type=float, default=0.0, help="Detector subset target x.")
    parser.add_argument("--target-z-mm", type=float, default=0.0, help="Detector subset target z.")
    parser.add_argument("--fov-nx", type=int, default=5, help="Prototype FOV size along x.")
    parser.add_argument("--fov-ny", type=int, default=5, help="Prototype FOV size along y.")
    parser.add_argument("--fov-nz", type=int, default=3, help="Prototype FOV size along z.")
    parser.add_argument("--sx-mm", type=float, default=12.0, help="Prototype voxel size along x.")
    parser.add_argument("--sy-mm", type=float, default=12.0, help="Prototype voxel size along y.")
    parser.add_argument("--sz-mm", type=float, default=6.0, help="Prototype voxel size along z.")
    parser.add_argument("--e1-bin-count", type=int, default=12, help="Number of discretized e1 bins.")
    parser.add_argument("--e1-min-mev", type=float, default=0.0, help="Minimum discretized e1.")
    parser.add_argument("--e1-max-mev", type=float, default=0.3396666667, help="Maximum discretized e1.")
    parser.add_argument("--det-sample-nx", type=int, default=2, help="Number of detector samples along crystal x.")
    parser.add_argument("--det-sample-ny", type=int, default=2, help="Number of detector samples along crystal y.")
    parser.add_argument("--det-sample-nz", type=int, default=2, help="Number of detector samples along crystal z.")
    parser.add_argument(
        "--voxel-chunk-size",
        type=int,
        default=0,
        help="Voxel chunk size for memory-reduced computation. <=0 means use all voxels at once.",
    )
    parser.add_argument(
        "--rear-chunk-size",
        type=int,
        default=0,
        help="Rear-detector chunk size for memory-reduced computation. <=0 means use all rear detectors at once.",
    )
    parser.add_argument("--top-k", type=int, default=100, help="Top-K center-voxel bins written to CSV.")
    parser.add_argument(
        "--slice-front-index",
        type=int,
        default=None,
        help="Optional front-detector index inside the selected subset for slice plotting.",
    )
    parser.add_argument(
        "--slice-rear-index",
        type=int,
        default=None,
        help="Optional rear-detector index inside the selected subset for slice plotting.",
    )
    parser.add_argument(
        "--slice-e1-bin",
        type=int,
        default=None,
        help="Optional e1-bin index for slice plotting.",
    )
    parser.add_argument(
        "--slice-y-index",
        type=int,
        default=None,
        help="Optional y-index of the FOV slice. Defaults to the middle y plane.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Torch device used by the prototype.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Torch dtype used by the prototype.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for npz/json/csv/figures.",
    )
    return parser.parse_args()


def choose_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_dtype_from_name(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def load_cross_section_table(data_path: Path) -> dict[str, np.ndarray]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Cross-section table not found: {data_path}")

    energies = []
    incoherent = []
    total_wo_coherent = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                energies.append(float(parts[0]))
                incoherent.append(float(parts[1]))
                total_wo_coherent.append(float(parts[3]))
            except ValueError:
                continue

    if not energies:
        raise RuntimeError(f"No numeric rows parsed from cross-section table: {data_path}")

    energy_arr = np.asarray(energies, dtype=np.float64)
    order = np.argsort(energy_arr, kind="stable")
    return {
        "energy_mev": energy_arr[order],
        "incoherent_cm2_g": np.asarray(incoherent, dtype=np.float64)[order],
        "total_wo_coherent_cm2_g": np.asarray(total_wo_coherent, dtype=np.float64)[order],
    }


def interp_linear_mu_cm_inv(table: dict[str, np.ndarray], energy_mev: torch.Tensor, key: str) -> torch.Tensor:
    energy_grid = torch.from_numpy(table["energy_mev"]).to(device=energy_mev.device, dtype=energy_mev.dtype)
    coeff_grid = torch.from_numpy(table[key]).to(device=energy_mev.device, dtype=energy_mev.dtype)
    e = torch.clamp(energy_mev, min=float(energy_grid[0].item()), max=float(energy_grid[-1].item()))
    idx = torch.searchsorted(energy_grid, e, right=True)
    idx = torch.clamp(idx, 1, energy_grid.numel() - 1)
    x0 = energy_grid[idx - 1]
    x1 = energy_grid[idx]
    y0 = coeff_grid[idx - 1]
    y1 = coeff_grid[idx]
    t = (e - x0) / torch.clamp(x1 - x0, min=EPS)
    mass_coeff = y0 + t * (y1 - y0)
    return mass_coeff * NAI_DENSITY_G_CM3


def load_detector(detector_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    detector = np.genfromtxt(detector_csv, delimiter=",", dtype=np.float64)
    if detector.ndim != 2 or detector.shape[1] < 4:
        raise ValueError(f"Detector CSV must contain at least 4 columns: {detector_csv}")

    detector_pos = detector[:, 1:4]
    detector_y_abs = np.abs(detector_pos[:, 1])
    layer_values = np.sort(np.unique(np.round(detector_y_abs, decimals=6)))
    if layer_values.size < 2:
        raise ValueError("Failed to resolve detector layers from Detector.csv")

    layer_by_det = np.zeros(detector.shape[0], dtype=np.int32)
    for idx, value in enumerate(layer_values, start=1):
        layer_by_det[np.isclose(detector_y_abs, value, atol=1e-6)] = idx
    if np.any(layer_by_det == 0):
        raise ValueError("Some detector elements were not assigned to a layer.")
    return detector_pos, layer_by_det


def build_small_fov(nx: int, ny: int, nz: int, sx_mm: float, sy_mm: float, sz_mm: float) -> np.ndarray:
    x_coords = (np.arange(nx, dtype=np.float64) - (nx / 2.0 - 0.5)) * sx_mm
    y_coords = (np.arange(ny, dtype=np.float64) - (ny / 2.0 - 0.5)) * sy_mm
    z_coords = (np.arange(nz, dtype=np.float64) - (nz / 2.0 - 0.5)) * sz_mm
    yy, xx, zz = np.meshgrid(y_coords, x_coords, z_coords, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=1)


def select_detector_subset(
    detector_pos: np.ndarray,
    layer_by_det: np.ndarray,
    front_det_count: int,
    rear_det_count: int,
    target_x_mm: float,
    target_z_mm: float,
    hemisphere: str,
) -> tuple[np.ndarray, np.ndarray]:
    front_mask = layer_by_det <= (layer_by_det.max() - 1)
    rear_mask = layer_by_det == layer_by_det.max()

    if hemisphere == "positive":
        hemi_mask = detector_pos[:, 1] > 0.0
    elif hemisphere == "negative":
        hemi_mask = detector_pos[:, 1] < 0.0
    else:
        hemi_mask = np.ones(detector_pos.shape[0], dtype=bool)

    def select(mask: np.ndarray, count: int) -> np.ndarray:
        idx = np.flatnonzero(mask & hemi_mask)
        if idx.size < count:
            raise ValueError(f"Not enough detectors after filtering. Requested {count}, available {idx.size}.")
        dx = detector_pos[idx, 0] - target_x_mm
        dz = detector_pos[idx, 2] - target_z_mm
        dist_sq = dx * dx + dz * dz
        order = np.argsort(dist_sq, kind="stable")
        return idx[order[:count]]

    return select(front_mask, front_det_count), select(rear_mask, rear_det_count)


def detector_surface_normal_from_y(detector_pos: torch.Tensor) -> torch.Tensor:
    normals = torch.zeros_like(detector_pos)
    normals[:, 1] = torch.where(detector_pos[:, 1] >= 0.0, -1.0, 1.0).to(detector_pos.dtype)
    return normals


def crystal_size_mm(layer_ids: np.ndarray, max_layer_id: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sizes = crystal_size_mm_np(layer_ids, max_layer_id)
    return torch.from_numpy(sizes).to(device=device, dtype=dtype)


def crystal_size_mm_np(layer_ids: np.ndarray, max_layer_id: int) -> np.ndarray:
    layer_ids = np.asarray(layer_ids)
    is_rear = (layer_ids == max_layer_id).astype(np.float64)[:, None]
    front_sizes = np.asarray(FRONT_LAYER_CRYSTAL_MM, dtype=np.float64)[None, :]
    rear_sizes = np.asarray(REAR_LAYER_CRYSTAL_MM, dtype=np.float64)[None, :]
    return front_sizes * (1.0 - is_rear) + rear_sizes * is_rear


def crystal_face_area_and_thickness_mm(detector_sizes_mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return detector_sizes_mm[:, 0] * detector_sizes_mm[:, 2], detector_sizes_mm[:, 1]


def compton_first_deposit_from_cos(cos_theta: torch.Tensor, incident_energy_mev: float) -> tuple[torch.Tensor, torch.Tensor]:
    e0 = torch.as_tensor(float(incident_energy_mev), device=cos_theta.device, dtype=cos_theta.dtype)
    scattered_energy = e0 / (1.0 + (e0 / ELECTRON_REST_MEV) * (1.0 - cos_theta))
    return e0 - scattered_energy, scattered_energy


def klein_nishina_relative_from_cos(cos_theta: torch.Tensor, incident_energy_mev: float) -> torch.Tensor:
    e0 = torch.as_tensor(float(incident_energy_mev), device=cos_theta.device, dtype=cos_theta.dtype)
    scattered_energy = e0 / (1.0 + (e0 / ELECTRON_REST_MEV) * (1.0 - cos_theta))
    ratio = scattered_energy / e0
    sin_sq = 1.0 - cos_theta.square()
    return torch.clamp(ratio.square() * (ratio + ratio.reciprocal() - sin_sq), min=0.0)


def compute_energy_resolution(energy_mev: float, ene_resolution_662kev: float) -> float:
    return float(ene_resolution_662kev * math.sqrt(0.662 / energy_mev))


def gaussian_bin_mass_torch(x: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor, bin_width: float) -> torch.Tensor:
    bin_width_t = torch.as_tensor(float(bin_width), device=x.device, dtype=x.dtype)
    safe_sigma = torch.clamp(sigma, min=1e-8)
    coeff = 1.0 / (math.sqrt(2.0 * math.pi) * safe_sigma)
    mass = coeff * torch.exp(-0.5 * ((x - mean) / safe_sigma).square()) * bin_width_t
    hard = ((x - mean).abs() <= (0.5 * bin_width_t)).to(x.dtype)
    return torch.where(sigma > 0.0, mass, hard)


def uniform_offsets(count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if count <= 1:
        return torch.zeros((1,), device=device, dtype=dtype)
    return ((torch.arange(count, device=device, dtype=dtype) + 0.5) / float(count)) - 0.5


def build_detector_interaction_samples(
    detector_centers: torch.Tensor,
    detector_sizes_mm: torch.Tensor,
    sample_nx: int,
    sample_ny: int,
    sample_nz: int,
) -> torch.Tensor:
    ox = uniform_offsets(sample_nx, detector_centers.device, detector_centers.dtype)
    oy = uniform_offsets(sample_ny, detector_centers.device, detector_centers.dtype)
    oz = uniform_offsets(sample_nz, detector_centers.device, detector_centers.dtype)
    gy, gx, gz = torch.meshgrid(oy, ox, oz, indexing="ij")
    frac = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)
    return detector_centers.unsqueeze(1) + frac.unsqueeze(0) * detector_sizes_mm.unsqueeze(1)


def build_detector_integration_cells(
    detector_centers: torch.Tensor,
    detector_sizes_mm: torch.Tensor,
    sample_nx: int,
    sample_ny: int,
    sample_nz: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ox = uniform_offsets(sample_nx, detector_centers.device, detector_centers.dtype)
    oz = uniform_offsets(sample_nz, detector_centers.device, detector_centers.dtype)
    depth_idx = torch.arange(sample_ny, device=detector_centers.device, dtype=detector_centers.dtype)

    gx, gdepth_idx, gz = torch.meshgrid(ox, depth_idx, oz, indexing="ij")
    x_frac = gx.reshape(-1)
    depth_idx_flat = gdepth_idx.reshape(-1)
    depth_center_frac_flat = ((depth_idx_flat + 0.5) / float(sample_ny)) - 0.5
    z_frac = gz.reshape(-1)

    y_sign = torch.where(detector_centers[:, 1] >= 0.0, 1.0, -1.0).to(detector_centers.dtype).view(-1, 1)
    cell_centers = torch.stack(
        [
            detector_centers[:, 0:1] + x_frac.view(1, -1) * detector_sizes_mm[:, 0:1],
            detector_centers[:, 1:2] + (y_sign * depth_center_frac_flat.view(1, -1)) * detector_sizes_mm[:, 1:2],
            detector_centers[:, 2:3] + z_frac.view(1, -1) * detector_sizes_mm[:, 2:3],
        ],
        dim=2,
    )

    cell_area_mm2 = (detector_sizes_mm[:, 0:1] * detector_sizes_mm[:, 2:3]) / float(sample_nx * sample_nz)
    cell_thickness_mm = detector_sizes_mm[:, 1:2] / float(sample_ny)
    depth_before_mm = depth_idx_flat.view(1, -1) * cell_thickness_mm

    num_cells = x_frac.numel()
    return (
        cell_centers,
        cell_area_mm2.expand(-1, num_cells),
        cell_thickness_mm.expand(-1, num_cells),
        depth_before_mm,
    )


def projected_area_probability_torch(
    source_pos: torch.Tensor,
    detector_pos: torch.Tensor,
    detector_normal: torch.Tensor,
    face_area_mm2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ray = detector_pos.unsqueeze(0) - source_pos.unsqueeze(1)
    dist_mm = torch.linalg.norm(ray, dim=2)
    ray_unit = ray / torch.clamp(dist_mm.unsqueeze(2), min=EPS)
    cos_inc = torch.sum(-ray_unit * detector_normal.unsqueeze(0), dim=2)
    probability = face_area_mm2.unsqueeze(0) * cos_inc / (4.0 * math.pi * torch.clamp(dist_mm.square(), min=EPS))
    valid = (dist_mm > EPS) & (cos_inc > 0.0)
    probability = torch.where(valid, probability, torch.zeros_like(probability))
    return probability, dist_mm, cos_inc, ray_unit


def effective_path_length_cm(thickness_mm: torch.Tensor, cos_inc: torch.Tensor) -> torch.Tensor:
    return (thickness_mm.unsqueeze(0) / torch.clamp(cos_inc, min=EPS)) / 10.0


def build_energy_bins(e1_min_mev: float, e1_max_mev: float, e1_bin_count: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(e1_min_mev, e1_max_mev, e1_bin_count + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def detector_box_bounds_np(detector_pos: np.ndarray, detector_sizes_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * detector_sizes_mm
    return detector_pos - half, detector_pos + half


def build_voxel_to_detector_blocker_candidates(
    voxel_pos: np.ndarray,
    target_detector_indices: np.ndarray,
    detector_pos: np.ndarray,
    detector_sizes_mm: np.ndarray,
) -> list[np.ndarray]:
    fov_center = voxel_pos.mean(axis=0)
    voxel_radius = np.linalg.norm(voxel_pos - fov_center[None, :], axis=1).max()
    detector_radius = 0.5 * np.linalg.norm(detector_sizes_mm, axis=1)
    candidates: list[np.ndarray] = []

    for target_idx in target_detector_indices:
        p0 = fov_center
        p1 = detector_pos[target_idx]
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 1e-9:
            candidates.append(np.empty((0,), dtype=np.int64))
            continue
        direction = seg / seg_len
        rel = detector_pos - p0[None, :]
        proj = np.clip(rel @ direction, 0.0, seg_len)
        nearest = p0[None, :] + proj[:, None] * direction[None, :]
        dist = np.linalg.norm(detector_pos - nearest, axis=1)
        threshold = detector_radius + voxel_radius + detector_radius[target_idx]
        mask = dist <= (threshold + 1e-6)
        mask[target_idx] = False
        candidates.append(np.flatnonzero(mask).astype(np.int64))
    return candidates


def build_detector_to_detector_blocker_candidates(
    front_detector_indices: np.ndarray,
    rear_detector_indices: np.ndarray,
    detector_pos: np.ndarray,
    detector_sizes_mm: np.ndarray,
) -> list[list[np.ndarray]]:
    detector_radius = 0.5 * np.linalg.norm(detector_sizes_mm, axis=1)
    candidates: list[list[np.ndarray]] = []

    for front_idx in front_detector_indices:
        row_candidates: list[np.ndarray] = []
        for rear_idx in rear_detector_indices:
            p0 = detector_pos[front_idx]
            p1 = detector_pos[rear_idx]
            seg = p1 - p0
            seg_len = float(np.linalg.norm(seg))
            if seg_len <= 1e-9:
                row_candidates.append(np.empty((0,), dtype=np.int64))
                continue
            direction = seg / seg_len
            rel = detector_pos - p0[None, :]
            proj = np.clip(rel @ direction, 0.0, seg_len)
            nearest = p0[None, :] + proj[:, None] * direction[None, :]
            dist = np.linalg.norm(detector_pos - nearest, axis=1)
            threshold = detector_radius + detector_radius[front_idx] + detector_radius[rear_idx]
            mask = dist <= (threshold + 1e-6)
            mask[front_idx] = False
            mask[rear_idx] = False
            row_candidates.append(np.flatnonzero(mask).astype(np.int64))
        candidates.append(row_candidates)
    return candidates


def segment_box_intersection_length_torch(
    segment_start: torch.Tensor,
    segment_end: torch.Tensor,
    box_min: torch.Tensor,
    box_max: torch.Tensor,
) -> torch.Tensor:
    if box_min.shape[0] == 0:
        return torch.zeros(segment_start.shape[:-1] + (0,), device=segment_start.device, dtype=segment_start.dtype)

    direction = segment_end - segment_start
    seg_len = torch.linalg.norm(direction, dim=-1)
    num_leading_dims = segment_start.dim() - 1
    box_view_shape = (1,) * num_leading_dims + box_min.shape
    box_min_view = box_min.view(box_view_shape)
    box_max_view = box_max.view(box_view_shape)
    start_view = segment_start.unsqueeze(-2)
    direction_view = direction.unsqueeze(-2)

    parallel = direction_view.abs() < EPS
    outside_parallel = parallel & ((start_view < box_min_view) | (start_view > box_max_view))
    safe_direction = torch.where(parallel, torch.ones_like(direction_view), direction_view)

    t1 = (box_min_view - start_view) / safe_direction
    t2 = (box_max_view - start_view) / safe_direction
    neg_inf = torch.full_like(t1, -torch.inf)
    pos_inf = torch.full_like(t1, torch.inf)
    tmin_axis = torch.where(parallel, neg_inf, torch.minimum(t1, t2))
    tmax_axis = torch.where(parallel, pos_inf, torch.maximum(t1, t2))

    t_enter = torch.max(tmin_axis, dim=-1).values
    t_exit = torch.min(tmax_axis, dim=-1).values
    frac = torch.clamp(torch.minimum(t_exit, torch.ones_like(t_exit)) - torch.maximum(t_enter, torch.zeros_like(t_enter)), min=0.0)
    seg_len_view = seg_len.unsqueeze(-1)
    valid = (~outside_parallel.any(dim=-1)) & (t_exit > t_enter) & (seg_len_view > EPS)
    return torch.where(valid, frac * seg_len_view, torch.zeros_like(frac))


def ray_box_exit_distance_from_inside_torch(
    ray_start: torch.Tensor,
    ray_direction_unit: torch.Tensor,
    box_min: torch.Tensor,
    box_max: torch.Tensor,
) -> torch.Tensor:
    view_shape = [1] * ray_start.dim()
    view_shape[1] = box_min.shape[0]
    view_shape[-1] = 3
    box_min_view = box_min.view(*view_shape)
    box_max_view = box_max.view(*view_shape)
    dir_safe = torch.where(ray_direction_unit.abs() < EPS, torch.ones_like(ray_direction_unit), ray_direction_unit)
    t_to_min = (box_min_view - ray_start) / dir_safe
    t_to_max = (box_max_view - ray_start) / dir_safe
    t_candidates = torch.where(ray_direction_unit > 0.0, t_to_max, t_to_min)
    inf = torch.full_like(t_candidates, torch.inf)
    t_candidates = torch.where(ray_direction_unit.abs() < EPS, inf, t_candidates)
    t_exit = torch.min(t_candidates, dim=-1).values
    return torch.clamp(t_exit, min=0.0)


def compute_voxel_to_detector_other_attenuation(
    voxel_pos: torch.Tensor,
    front_cells: torch.Tensor,
    blocker_box_min: torch.Tensor,
    blocker_box_max: torch.Tensor,
    candidate_lists: list[torch.Tensor],
    mu_total_incident: torch.Tensor,
) -> torch.Tensor:
    num_voxels = voxel_pos.shape[0]
    num_front = front_cells.shape[0]
    num_front_cells = front_cells.shape[1]
    attenuation = torch.ones((num_voxels, num_front, num_front_cells), device=voxel_pos.device, dtype=voxel_pos.dtype)

    for front_idx, cand in enumerate(candidate_lists):
        if cand.numel() == 0:
            continue
        start = voxel_pos[:, None, :].expand(-1, num_front_cells, -1)
        end = front_cells[front_idx].unsqueeze(0).expand(num_voxels, -1, -1)
        lengths_mm = segment_box_intersection_length_torch(start, end, blocker_box_min[cand], blocker_box_max[cand])
        attenuation[:, front_idx, :] = torch.exp(-mu_total_incident * lengths_mm.sum(dim=-1) / 10.0)
    return attenuation


def compute_detector_to_detector_other_attenuation(
    front_cells: torch.Tensor,
    rear_cells: torch.Tensor,
    blocker_box_min: torch.Tensor,
    blocker_box_max: torch.Tensor,
    pair_candidate_lists: list[list[torch.Tensor]],
    mu_total_scattered: torch.Tensor,
) -> torch.Tensor:
    num_voxels = mu_total_scattered.shape[0]
    num_front = front_cells.shape[0]
    num_front_cells = front_cells.shape[1]
    num_rear = rear_cells.shape[0]
    num_rear_cells = rear_cells.shape[1]
    attenuation = torch.ones(
        (num_voxels, num_front, num_front_cells, num_rear, num_rear_cells),
        device=front_cells.device,
        dtype=front_cells.dtype,
    )

    for front_idx in range(num_front):
        for rear_idx in range(num_rear):
            cand = pair_candidate_lists[front_idx][rear_idx]
            if cand.numel() == 0:
                continue
            start = front_cells[front_idx][:, None, :].expand(-1, num_rear_cells, -1)
            end = rear_cells[rear_idx][None, :, :].expand(num_front_cells, -1, -1)
            lengths_mm = segment_box_intersection_length_torch(start, end, blocker_box_min[cand], blocker_box_max[cand])
            attenuation_cm = lengths_mm.sum(dim=-1) / 10.0
            attenuation[:, front_idx, :, rear_idx, :] = torch.exp(
                -mu_total_scattered[:, front_idx, :, rear_idx, :] * attenuation_cm.unsqueeze(0)
            )
    return attenuation


def klein_nishina_pdf_solid_angle_from_cos(
    cos_theta: torch.Tensor,
    incident_energy_mev: float,
    normalization_solid_angle: torch.Tensor,
) -> torch.Tensor:
    return klein_nishina_relative_from_cos(cos_theta, incident_energy_mev) / torch.clamp(
        normalization_solid_angle, min=EPS
    )


def klein_nishina_solid_angle_normalization(
    incident_energy_mev: float,
    device: torch.device,
    dtype: torch.dtype,
    quad_count: int = 4097,
) -> torch.Tensor:
    cos_grid = torch.linspace(-1.0, 1.0, quad_count, device=device, dtype=dtype)
    rel = klein_nishina_relative_from_cos(cos_grid, incident_energy_mev)
    return 2.0 * math.pi * torch.trapz(rel, cos_grid)


def compute_response_tensor(
    voxel_pos: torch.Tensor,
    front_pos: torch.Tensor,
    rear_pos: torch.Tensor,
    front_samples: torch.Tensor,
    rear_samples: torch.Tensor,
    front_area_mm2: torch.Tensor,
    rear_area_mm2: torch.Tensor,
    front_thickness_mm: torch.Tensor,
    rear_thickness_mm: torch.Tensor,
    front_depth_before_mm: torch.Tensor,
    rear_depth_before_mm: torch.Tensor,
    front_box_min: torch.Tensor,
    front_box_max: torch.Tensor,
    blocker_box_min: torch.Tensor,
    blocker_box_max: torch.Tensor,
    voxel_to_front_blocker_candidates: list[torch.Tensor],
    detector_to_detector_blocker_candidates: list[list[torch.Tensor]],
    e1_centers: torch.Tensor,
    e1_bin_width: float,
    incident_energy_mev: float,
    effective_energy_resolution: float,
    energy_blur_enabled: bool,
    cross_section_table: dict[str, np.ndarray],
    voxel_chunk_size: int,
    rear_chunk_size: int,
    show_progress: bool,
    progress_label: str,
) -> torch.Tensor:
    num_voxels = voxel_pos.shape[0]
    num_front = front_pos.shape[0]
    num_rear = rear_pos.shape[0]
    num_e1 = e1_centers.numel()
    num_front_cells = front_samples.shape[1]
    num_rear_cells = rear_samples.shape[1]

    front_normal = detector_surface_normal_from_y(front_pos)
    rear_normal = detector_surface_normal_from_y(rear_pos)

    incident_energy_tensor = torch.as_tensor(float(incident_energy_mev), device=voxel_pos.device, dtype=voxel_pos.dtype)
    mu_total_incident = interp_linear_mu_cm_inv(
        cross_section_table,
        incident_energy_tensor.reshape(1),
        "total_wo_coherent_cm2_g",
    )[0]
    mu_compton_incident = interp_linear_mu_cm_inv(
        cross_section_table,
        incident_energy_tensor.reshape(1),
        "incoherent_cm2_g",
    )[0]
    kn_norm = klein_nishina_solid_angle_normalization(incident_energy_mev, voxel_pos.device, voxel_pos.dtype)
    voxel_chunk_size = num_voxels if voxel_chunk_size <= 0 else min(voxel_chunk_size, num_voxels)
    rear_chunk_size = num_rear if rear_chunk_size <= 0 else min(rear_chunk_size, num_rear)

    response = torch.zeros((num_voxels, num_front, num_rear, num_e1), device=voxel_pos.device, dtype=voxel_pos.dtype)
    e1_center_grid = e1_centers.view(1, 1, 1, 1, 1, num_e1)
    num_voxel_chunks = (num_voxels + voxel_chunk_size - 1) // voxel_chunk_size
    num_rear_chunks = (num_rear + rear_chunk_size - 1) // rear_chunk_size
    total_chunk_steps = num_voxel_chunks * num_rear_chunks
    completed_chunk_steps = 0
    start_time = time.perf_counter()

    for voxel_chunk_idx, voxel_start in enumerate(range(0, num_voxels, voxel_chunk_size), start=1):
        voxel_end = min(voxel_start + voxel_chunk_size, num_voxels)
        voxel_chunk = voxel_pos[voxel_start:voxel_end]

        attenuation_voxel_to_front_other = compute_voxel_to_detector_other_attenuation(
            voxel_pos=voxel_chunk,
            front_cells=front_samples,
            blocker_box_min=blocker_box_min,
            blocker_box_max=blocker_box_max,
            candidate_lists=voxel_to_front_blocker_candidates,
            mu_total_incident=mu_total_incident,
        )

        ray_in = front_samples.unsqueeze(0) - voxel_chunk.view(voxel_chunk.shape[0], 1, 1, 3)
        dist_in_mm = torch.linalg.norm(ray_in, dim=3)
        ray_in_unit = ray_in / torch.clamp(dist_in_mm.unsqueeze(3), min=EPS)
        cos_1 = torch.sum(-ray_in_unit * front_normal.view(1, num_front, 1, 3), dim=3)
        p_reach_front_cell = front_area_mm2.view(1, num_front, num_front_cells) * cos_1 / (
            4.0 * math.pi * torch.clamp(dist_in_mm.square(), min=EPS)
        )
        p_reach_front_cell = torch.where(
            (dist_in_mm > EPS) & (cos_1 > 0.0),
            p_reach_front_cell,
            torch.zeros_like(p_reach_front_cell),
        )

        front_depth_before_cm = (
            front_depth_before_mm.view(1, num_front, num_front_cells) / torch.clamp(cos_1, min=EPS) / 10.0
        )
        front_delta_path_cm = (
            front_thickness_mm.view(1, num_front, num_front_cells) / torch.clamp(cos_1, min=EPS) / 10.0
        )
        p_first_compton = p_reach_front_cell * torch.exp(-mu_total_incident * front_depth_before_cm) * (
            1.0 - torch.exp(-mu_total_incident * front_delta_path_cm)
        )
        p_first_compton = p_first_compton * (mu_compton_incident / torch.clamp(mu_total_incident, min=EPS))
        p_first_compton = p_first_compton * attenuation_voxel_to_front_other

        for rear_chunk_idx, rear_start in enumerate(range(0, num_rear, rear_chunk_size), start=1):
            rear_end = min(rear_start + rear_chunk_size, num_rear)
            rear_chunk_len = rear_end - rear_start
            rear_samples_chunk = rear_samples[rear_start:rear_end]
            rear_normal_chunk = rear_normal[rear_start:rear_end]
            rear_area_chunk = rear_area_mm2[rear_start:rear_end]
            rear_thickness_chunk = rear_thickness_mm[rear_start:rear_end]
            rear_depth_before_chunk = rear_depth_before_mm[rear_start:rear_end]
            detector_to_detector_blocker_candidates_chunk = [
                row[rear_start:rear_end] for row in detector_to_detector_blocker_candidates
            ]

            ray_out = rear_samples_chunk.view(1, 1, 1, rear_chunk_len, num_rear_cells, 3) - front_samples.view(
                1, num_front, num_front_cells, 1, 1, 3
            )
            dist_out_mm = torch.linalg.norm(ray_out, dim=5)
            ray_out_unit = ray_out / torch.clamp(dist_out_mm.unsqueeze(5), min=EPS)

            cos_beta = torch.sum(ray_in_unit.unsqueeze(3).unsqueeze(4) * ray_out_unit, dim=5)
            cos_beta = torch.clamp(cos_beta, -1.0, 1.0)
            e1_geom, scattered_energy = compton_first_deposit_from_cos(cos_beta, incident_energy_mev)
            kn_pdf = klein_nishina_pdf_solid_angle_from_cos(cos_beta, incident_energy_mev, kn_norm)

            cos_2_signed = torch.sum(-ray_out_unit * rear_normal_chunk.view(1, 1, 1, rear_chunk_len, 1, 3), dim=5)
            cos_2 = cos_2_signed.abs()
            solid_angle_2 = rear_area_chunk.view(1, 1, 1, rear_chunk_len, num_rear_cells) * cos_2 / torch.clamp(
                dist_out_mm.square(), min=EPS
            )
            mu_total_scattered = interp_linear_mu_cm_inv(
                cross_section_table,
                torch.clamp(scattered_energy, min=1e-6),
                "total_wo_coherent_cm2_g",
            )
            front_self_exit_mm = ray_box_exit_distance_from_inside_torch(
                ray_start=front_samples.view(1, num_front, num_front_cells, 1, 1, 3),
                ray_direction_unit=ray_out_unit,
                box_min=front_box_min,
                box_max=front_box_max,
            )
            attenuation_front_self_after = torch.exp(-mu_total_scattered * (front_self_exit_mm / 10.0))
            attenuation_front_to_rear_other = compute_detector_to_detector_other_attenuation(
                front_cells=front_samples,
                rear_cells=rear_samples_chunk,
                blocker_box_min=blocker_box_min,
                blocker_box_max=blocker_box_max,
                pair_candidate_lists=detector_to_detector_blocker_candidates_chunk,
                mu_total_scattered=mu_total_scattered,
            )
            total_rear_thickness_chunk = (
                rear_depth_before_chunk.max(dim=1, keepdim=True).values + rear_thickness_chunk[:, :1]
            ).view(1, 1, 1, rear_chunk_len, 1)
            rear_depth_before_view = rear_depth_before_chunk.view(1, 1, 1, rear_chunk_len, num_rear_cells)
            rear_cell_thickness_view = rear_thickness_chunk.view(1, 1, 1, rear_chunk_len, num_rear_cells)
            rear_path_before_mm = torch.where(
                cos_2_signed >= 0.0,
                rear_depth_before_view,
                total_rear_thickness_chunk - rear_depth_before_view - rear_cell_thickness_view,
            )
            rear_depth_before_cm = (
                rear_path_before_mm / torch.clamp(cos_2, min=EPS) / 10.0
            )
            rear_delta_path_cm = (
                rear_cell_thickness_view / torch.clamp(cos_2, min=EPS) / 10.0
            )
            p_scatter_to_rear_cell = kn_pdf * solid_angle_2 * attenuation_front_self_after * attenuation_front_to_rear_other
            p_second_recorded = p_scatter_to_rear_cell * torch.exp(-mu_total_scattered * rear_depth_before_cm) * (
                1.0 - torch.exp(-mu_total_scattered * rear_delta_path_cm)
            )

            base_weight = p_first_compton.unsqueeze(3).unsqueeze(4) * p_second_recorded
            valid = (
                (p_first_compton.unsqueeze(3).unsqueeze(4) > 0.0)
                & (dist_out_mm > EPS)
                & (scattered_energy > 0.0)
                & (cos_2 > 0.0)
                & (solid_angle_2 > 0.0)
            )
            base_weight = torch.where(valid, base_weight, torch.zeros_like(base_weight))

            e1_geom_grid = e1_geom.unsqueeze(5)
            if energy_blur_enabled:
                sigma_e = torch.clamp(
                    e1_geom_grid
                    * float(effective_energy_resolution)
                    / 2.355
                    * torch.sqrt(incident_energy_tensor / torch.clamp(e1_geom_grid, min=1e-6)),
                    min=1e-6,
                )
            else:
                sigma_e = torch.zeros_like(e1_geom_grid)
            e1_mass = gaussian_bin_mass_torch(e1_center_grid, e1_geom_grid, sigma_e, e1_bin_width)
            response[voxel_start:voxel_end, :, rear_start:rear_end, :] = (
                base_weight.unsqueeze(5) * e1_mass
            ).sum(dim=(2, 4))
            completed_chunk_steps += 1
            if show_progress:
                elapsed = time.perf_counter() - start_time
                progress = completed_chunk_steps / float(total_chunk_steps)
                eta = (elapsed / progress - elapsed) if progress > 0.0 else 0.0
                print(
                    f"  [{progress_label}] chunk "
                    f"{completed_chunk_steps}/{total_chunk_steps} "
                    f"(voxel {voxel_chunk_idx}/{num_voxel_chunks}, rear {rear_chunk_idx}/{num_rear_chunks})  "
                    f"voxels {voxel_end}/{num_voxels}  rear {rear_end}/{num_rear}  "
                    f"elapsed={elapsed:.1f}s  eta={eta:.1f}s"
                )

    return torch.where(torch.isfinite(response), response, torch.zeros_like(response))


def write_detector_selection_csv(
    output_path: Path,
    detector_pos: np.ndarray,
    layer_by_det: np.ndarray,
    front_idx: np.ndarray,
    rear_idx: np.ndarray,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["role", "detector_id_1based", "layer_id", "x_mm", "y_mm", "z_mm"])
        for idx in front_idx:
            writer.writerow(["front", int(idx + 1), int(layer_by_det[idx]), *detector_pos[idx].tolist()])
        for idx in rear_idx:
            writer.writerow(["rear", int(idx + 1), int(layer_by_det[idx]), *detector_pos[idx].tolist()])


def write_voxel_table_csv(output_path: Path, voxel_pos: np.ndarray, response: np.ndarray) -> None:
    voxel_total = response.reshape(response.shape[0], -1).sum(axis=1)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["voxel_index", "x_mm", "y_mm", "z_mm", "response_sum"])
        for idx, (pos, total) in enumerate(zip(voxel_pos, voxel_total)):
            writer.writerow([idx, *pos.tolist(), float(total)])


def write_center_voxel_topbins_csv(
    output_path: Path,
    center_voxel_index: int,
    response: np.ndarray,
    front_detector_ids_1based: np.ndarray,
    rear_detector_ids_1based: np.ndarray,
    e1_centers: np.ndarray,
    top_k: int,
) -> list[tuple[int, int, int, float]]:
    center_tensor = response[center_voxel_index]
    flat = center_tensor.reshape(-1)
    top_k = min(top_k, flat.size)
    order = np.argsort(flat)[::-1][:top_k]
    top_bins: list[tuple[int, int, int, float]] = []

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "flat_index",
                "front_detector_id_1based",
                "rear_detector_id_1based",
                "e1_bin_index",
                "e1_center_mev",
                "weight",
            ]
        )
        num_front, num_rear, num_e1 = center_tensor.shape
        for rank, flat_idx in enumerate(order, start=1):
            front_idx, rear_idx, e1_idx = np.unravel_index(flat_idx, (num_front, num_rear, num_e1))
            weight = float(center_tensor[front_idx, rear_idx, e1_idx])
            writer.writerow(
                [
                    rank,
                    int(flat_idx),
                    int(front_detector_ids_1based[front_idx]),
                    int(rear_detector_ids_1based[rear_idx]),
                    int(e1_idx),
                    float(e1_centers[e1_idx]),
                    weight,
                ]
            )
            top_bins.append((front_idx, rear_idx, e1_idx, weight))
    return top_bins


def save_slice_plot(
    output_path: Path,
    voxel_pos: np.ndarray,
    response: np.ndarray,
    fov_nx: int,
    fov_ny: int,
    fov_nz: int,
    slice_y_index: int,
    front_detector_id_1based: int,
    rear_detector_id_1based: int,
    e1_center_mev: float,
) -> None:
    slice_3d = response.reshape(fov_ny, fov_nx, fov_nz)
    plane = slice_3d[slice_y_index, :, :].T

    voxel_grid = voxel_pos.reshape(fov_ny, fov_nx, fov_nz, 3)
    x_coords = voxel_grid[slice_y_index, :, 0, 0]
    z_coords = voxel_grid[slice_y_index, 0, :, 2]

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=160)
    im = ax.imshow(
        plane,
        origin="lower",
        aspect="auto",
        extent=[float(x_coords.min()), float(x_coords.max()), float(z_coords.min()), float(z_coords.max())],
        cmap="inferno",
    )
    fig.colorbar(im, ax=ax, label="Prototype response")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(
        f"x-z slice parallel to detector surface\n"
        f"front={front_detector_id_1based}, rear={rear_detector_id_1based}, e1={e1_center_mev:.4f} MeV, y-index={slice_y_index}"
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    dtype = torch_dtype_from_name(args.dtype)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector_pos_np, layer_by_det = load_detector(args.detector_csv.resolve())
    cross_section_table = load_cross_section_table(args.data_path.resolve())
    front_idx, rear_idx = select_detector_subset(
        detector_pos=detector_pos_np,
        layer_by_det=layer_by_det,
        front_det_count=args.front_det_count,
        rear_det_count=args.rear_det_count,
        target_x_mm=args.target_x_mm,
        target_z_mm=args.target_z_mm,
        hemisphere=args.hemisphere,
    )

    voxel_pos_np = build_small_fov(args.fov_nx, args.fov_ny, args.fov_nz, args.sx_mm, args.sy_mm, args.sz_mm)
    e1_edges_np, e1_centers_np = build_energy_bins(args.e1_min_mev, args.e1_max_mev, args.e1_bin_count)
    effective_energy_resolution = compute_energy_resolution(args.incident_energy_mev, args.ene_resolution_662kev)

    voxel_pos = torch.from_numpy(voxel_pos_np).to(device=device, dtype=dtype)
    front_pos = torch.from_numpy(detector_pos_np[front_idx]).to(device=device, dtype=dtype)
    rear_pos = torch.from_numpy(detector_pos_np[rear_idx]).to(device=device, dtype=dtype)
    e1_centers = torch.from_numpy(e1_centers_np).to(device=device, dtype=dtype)
    e1_bin_width = float(e1_edges_np[1] - e1_edges_np[0])
    all_sizes_np = crystal_size_mm_np(layer_by_det, int(layer_by_det.max()))
    blocker_box_min_np, blocker_box_max_np = detector_box_bounds_np(detector_pos_np, all_sizes_np)
    voxel_to_front_blocker_candidates_np = build_voxel_to_detector_blocker_candidates(
        voxel_pos=voxel_pos_np,
        target_detector_indices=front_idx,
        detector_pos=detector_pos_np,
        detector_sizes_mm=all_sizes_np,
    )
    voxel_to_rear_blocker_candidates_np = build_voxel_to_detector_blocker_candidates(
        voxel_pos=voxel_pos_np,
        target_detector_indices=rear_idx,
        detector_pos=detector_pos_np,
        detector_sizes_mm=all_sizes_np,
    )
    detector_to_detector_blocker_candidates_np = build_detector_to_detector_blocker_candidates(
        front_detector_indices=front_idx,
        rear_detector_indices=rear_idx,
        detector_pos=detector_pos_np,
        detector_sizes_mm=all_sizes_np,
    )
    detector_to_detector_blocker_candidates_rev_np = build_detector_to_detector_blocker_candidates(
        front_detector_indices=rear_idx,
        rear_detector_indices=front_idx,
        detector_pos=detector_pos_np,
        detector_sizes_mm=all_sizes_np,
    )

    front_sizes_mm = crystal_size_mm(layer_by_det[front_idx], int(layer_by_det.max()), device, dtype)
    rear_sizes_mm = crystal_size_mm(layer_by_det[rear_idx], int(layer_by_det.max()), device, dtype)
    front_samples, front_area_mm2, front_thickness_mm, front_depth_before_mm = build_detector_integration_cells(
        detector_centers=front_pos,
        detector_sizes_mm=front_sizes_mm,
        sample_nx=args.det_sample_nx,
        sample_ny=args.det_sample_ny,
        sample_nz=args.det_sample_nz,
    )
    rear_samples, rear_area_mm2, rear_thickness_mm, rear_depth_before_mm = build_detector_integration_cells(
        detector_centers=rear_pos,
        detector_sizes_mm=rear_sizes_mm,
        sample_nx=args.det_sample_nx,
        sample_ny=args.det_sample_ny,
        sample_nz=args.det_sample_nz,
    )
    front_box_min = front_pos - 0.5 * front_sizes_mm
    front_box_max = front_pos + 0.5 * front_sizes_mm
    rear_box_min = rear_pos - 0.5 * rear_sizes_mm
    rear_box_max = rear_pos + 0.5 * rear_sizes_mm
    blocker_box_min = torch.from_numpy(blocker_box_min_np).to(device=device, dtype=dtype)
    blocker_box_max = torch.from_numpy(blocker_box_max_np).to(device=device, dtype=dtype)
    voxel_to_front_blocker_candidates = [
        torch.from_numpy(candidate).to(device=device, dtype=torch.long)
        for candidate in voxel_to_front_blocker_candidates_np
    ]
    voxel_to_rear_blocker_candidates = [
        torch.from_numpy(candidate).to(device=device, dtype=torch.long)
        for candidate in voxel_to_rear_blocker_candidates_np
    ]
    detector_to_detector_blocker_candidates = [
        [torch.from_numpy(candidate).to(device=device, dtype=torch.long) for candidate in row]
        for row in detector_to_detector_blocker_candidates_np
    ]
    detector_to_detector_blocker_candidates_rev = [
        [torch.from_numpy(candidate).to(device=device, dtype=torch.long) for candidate in row]
        for row in detector_to_detector_blocker_candidates_rev_np
    ]

    print("Selected front detectors :", front_idx.size)
    print("Selected rear detectors  :", rear_idx.size)
    print("Prototype voxel count    :", voxel_pos.shape[0])
    print("e1 bin count             :", e1_centers.numel())
    print(f"Device                   : {device}")
    print(f"Dtype                    : {dtype}")
    print(f"Energy blur enabled      : {not args.disable_energy_blur}")
    print(f"Effective energy res     : {effective_energy_resolution:.6f}")
    print(
        "Detector sample shape    : "
        f"{args.det_sample_nx} x {args.det_sample_ny} x {args.det_sample_nz}"
    )
    print(f"Voxel chunk size         : {args.voxel_chunk_size if args.voxel_chunk_size > 0 else voxel_pos.shape[0]}")
    print(f"Rear chunk size          : {args.rear_chunk_size if args.rear_chunk_size > 0 else rear_pos.shape[0]}")
    print("Computing prototype tensor ...")

    with torch.no_grad():
        response_forward_torch = compute_response_tensor(
            voxel_pos=voxel_pos,
            front_pos=front_pos,
            rear_pos=rear_pos,
            front_samples=front_samples,
            rear_samples=rear_samples,
            front_area_mm2=front_area_mm2,
            rear_area_mm2=rear_area_mm2,
            front_thickness_mm=front_thickness_mm,
            rear_thickness_mm=rear_thickness_mm,
            front_depth_before_mm=front_depth_before_mm,
            rear_depth_before_mm=rear_depth_before_mm,
            front_box_min=front_box_min,
            front_box_max=front_box_max,
            blocker_box_min=blocker_box_min,
            blocker_box_max=blocker_box_max,
            voxel_to_front_blocker_candidates=voxel_to_front_blocker_candidates,
            detector_to_detector_blocker_candidates=detector_to_detector_blocker_candidates,
            e1_centers=e1_centers,
            e1_bin_width=e1_bin_width,
            incident_energy_mev=args.incident_energy_mev,
            effective_energy_resolution=effective_energy_resolution,
            energy_blur_enabled=not args.disable_energy_blur,
            cross_section_table=cross_section_table,
            voxel_chunk_size=args.voxel_chunk_size,
            rear_chunk_size=args.rear_chunk_size,
            show_progress=True,
            progress_label="front->rear",
        )
        response_reverse_ordered_torch = compute_response_tensor(
            voxel_pos=voxel_pos,
            front_pos=rear_pos,
            rear_pos=front_pos,
            front_samples=rear_samples,
            rear_samples=front_samples,
            front_area_mm2=rear_area_mm2,
            rear_area_mm2=front_area_mm2,
            front_thickness_mm=rear_thickness_mm,
            rear_thickness_mm=front_thickness_mm,
            front_depth_before_mm=rear_depth_before_mm,
            rear_depth_before_mm=front_depth_before_mm,
            front_box_min=rear_box_min,
            front_box_max=rear_box_max,
            blocker_box_min=blocker_box_min,
            blocker_box_max=blocker_box_max,
            voxel_to_front_blocker_candidates=voxel_to_rear_blocker_candidates,
            detector_to_detector_blocker_candidates=detector_to_detector_blocker_candidates_rev,
            e1_centers=e1_centers,
            e1_bin_width=e1_bin_width,
            incident_energy_mev=args.incident_energy_mev,
            effective_energy_resolution=effective_energy_resolution,
            energy_blur_enabled=not args.disable_energy_blur,
            cross_section_table=cross_section_table,
            voxel_chunk_size=args.voxel_chunk_size,
            rear_chunk_size=args.rear_chunk_size,
            show_progress=True,
            progress_label="rear->front",
        )
        response_reverse_torch = response_reverse_ordered_torch.permute(0, 2, 1, 3).contiguous()
        response_torch = response_forward_torch + response_reverse_torch
        if device.type == "cuda":
            torch.cuda.synchronize()

    response_np = response_torch.detach().cpu().numpy().astype(np.float32, copy=False)
    response_forward_np = response_forward_torch.detach().cpu().numpy().astype(np.float32, copy=False)
    response_reverse_np = response_reverse_torch.detach().cpu().numpy().astype(np.float32, copy=False)
    center_voxel_index = voxel_pos_np.shape[0] // 2

    top_bins = write_center_voxel_topbins_csv(
        output_dir / "center_voxel_topbins.csv",
        center_voxel_index=center_voxel_index,
        response=response_np,
        front_detector_ids_1based=front_idx + 1,
        rear_detector_ids_1based=rear_idx + 1,
        e1_centers=e1_centers_np,
        top_k=args.top_k,
    )

    if args.slice_front_index is None or args.slice_rear_index is None or args.slice_e1_bin is None:
        if not top_bins:
            raise RuntimeError("No nonzero response bins were found; cannot choose a slice automatically.")
        slice_front_index, slice_rear_index, slice_e1_bin, _ = top_bins[0]
    else:
        slice_front_index = int(args.slice_front_index)
        slice_rear_index = int(args.slice_rear_index)
        slice_e1_bin = int(args.slice_e1_bin)

    slice_y_index = args.slice_y_index if args.slice_y_index is not None else (args.fov_ny // 2)
    if not (0 <= slice_front_index < front_idx.size):
        raise ValueError("--slice-front-index is out of range for the selected front subset.")
    if not (0 <= slice_rear_index < rear_idx.size):
        raise ValueError("--slice-rear-index is out of range for the selected rear subset.")
    if not (0 <= slice_e1_bin < e1_centers_np.size):
        raise ValueError("--slice-e1-bin is out of range.")
    if not (0 <= slice_y_index < args.fov_ny):
        raise ValueError("--slice-y-index is out of range.")

    slice_response = response_np[:, slice_front_index, slice_rear_index, slice_e1_bin]

    summary = PrototypeSummary(
        detector_csv=str(args.detector_csv.resolve()),
        data_path=str(args.data_path.resolve()),
        incident_energy_mev=args.incident_energy_mev,
        hemisphere=args.hemisphere,
        device=str(device),
        dtype=args.dtype,
        front_detector_count=int(front_idx.size),
        rear_detector_count=int(rear_idx.size),
        voxel_count=int(voxel_pos_np.shape[0]),
        e1_bin_count=int(e1_centers_np.size),
        e1_min_mev=args.e1_min_mev,
        e1_max_mev=args.e1_max_mev,
        energy_resolution_662kev=args.ene_resolution_662kev,
        effective_energy_resolution=effective_energy_resolution,
        energy_blur_enabled=not args.disable_energy_blur,
        detector_sample_shape=[args.det_sample_nx, args.det_sample_ny, args.det_sample_nz],
        fov_shape=[args.fov_nx, args.fov_ny, args.fov_nz],
        voxel_size_mm=[args.sx_mm, args.sy_mm, args.sz_mm],
        output_dir=str(output_dir),
        response_forward_sum=float(response_forward_np.sum()),
        response_reverse_sum=float(response_reverse_np.sum()),
        response_sum=float(response_np.sum()),
        response_forward_max=float(response_forward_np.max()),
        response_reverse_max=float(response_reverse_np.max()),
        response_max=float(response_np.max()),
        center_voxel_index=int(center_voxel_index),
        slice_front_detector_id_1based=int(front_idx[slice_front_index] + 1),
        slice_rear_detector_id_1based=int(rear_idx[slice_rear_index] + 1),
        slice_e1_bin_index=int(slice_e1_bin),
        slice_e1_center_mev=float(e1_centers_np[slice_e1_bin]),
        slice_y_index=int(slice_y_index),
    )

    np.savez_compressed(
        output_dir / "prototype_response_tensor.npz",
        voxel_positions_mm=voxel_pos_np.astype(np.float32),
        front_detector_ids_1based=(front_idx + 1).astype(np.int32),
        rear_detector_ids_1based=(rear_idx + 1).astype(np.int32),
        front_detector_positions_mm=detector_pos_np[front_idx].astype(np.float32),
        rear_detector_positions_mm=detector_pos_np[rear_idx].astype(np.float32),
        front_interaction_samples_mm=front_samples.detach().cpu().numpy().astype(np.float32),
        rear_interaction_samples_mm=rear_samples.detach().cpu().numpy().astype(np.float32),
        e1_bin_edges_mev=e1_edges_np.astype(np.float32),
        e1_bin_centers_mev=e1_centers_np.astype(np.float32),
        response_forward=response_forward_np,
        response_reverse=response_reverse_np,
        response=response_np,
        slice_response=slice_response.astype(np.float32),
    )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    write_detector_selection_csv(
        output_dir / "detector_selection.csv",
        detector_pos=detector_pos_np,
        layer_by_det=layer_by_det,
        front_idx=front_idx,
        rear_idx=rear_idx,
    )
    write_voxel_table_csv(output_dir / "voxel_table.csv", voxel_pos=voxel_pos_np, response=response_np)

    save_slice_plot(
        output_path=output_dir / "xz_slice_response.png",
        voxel_pos=voxel_pos_np,
        response=slice_response,
        fov_nx=args.fov_nx,
        fov_ny=args.fov_ny,
        fov_nz=args.fov_nz,
        slice_y_index=slice_y_index,
        front_detector_id_1based=int(front_idx[slice_front_index] + 1),
        rear_detector_id_1based=int(rear_idx[slice_rear_index] + 1),
        e1_center_mev=float(e1_centers_np[slice_e1_bin]),
    )

    print(f"Output dir              : {output_dir}")
    print(f"Response forward sum    : {response_forward_np.sum():.6e}")
    print(f"Response reverse sum    : {response_reverse_np.sum():.6e}")
    print(f"Response sum            : {response_np.sum():.6e}")
    print(f"Response forward max    : {response_forward_np.max():.6e}")
    print(f"Response reverse max    : {response_reverse_np.max():.6e}")
    print(f"Response max            : {response_np.max():.6e}")
    print(f"Center voxel index      : {center_voxel_index}")
    print(
        "Slice selection         : "
        f"front={int(front_idx[slice_front_index] + 1)}, "
        f"rear={int(rear_idx[slice_rear_index] + 1)}, "
        f"e1_bin={slice_e1_bin}, "
        f"e1_center={float(e1_centers_np[slice_e1_bin]):.6f} MeV, "
        f"y_index={slice_y_index}"
    )
    if float(response_np.sum()) <= 0.0:
        print("Warning: all prototype weights are zero. Try lowering --e1-min-mev or selecting a wider detector subset.")


if __name__ == "__main__":
    main()
