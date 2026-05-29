from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "Result"
EPS = 1.0e-8


@dataclass
class FactorResult:
    factor_label: str
    factor_dir: str
    num_voxels: int
    num_detectors: int
    avg_sensitivity: float
    roi_voxel_count: int
    crc_mean: list[float]
    var_mean: list[float]
    elapsed_seconds: float


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def should_log_progress(index_1based: int, total: int, target_logs: int = 20) -> bool:
    if total <= 0:
        return False
    if index_1based <= 1 or index_1based >= total:
        return True
    step = max(1, math.ceil(total / max(1, target_logs)))
    return (index_1based % step) == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute CRC-VAR curves from single-photon SysMat_tmp factors using a full dense FIM "
            "and global direct linear algebra, without PCG."
        )
    )
    parser.add_argument("--factor-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--factor-labels", type=str, nargs="*", default=None)
    parser.add_argument("--nx", type=int, default=51)
    parser.add_argument("--ny", type=int, default=51)
    parser.add_argument("--nz", type=int, default=20)
    parser.add_argument("--sx-mm", type=float, default=6.0)
    parser.add_argument("--sy-mm", type=float, default=6.0)
    parser.add_argument("--sz-mm", type=float, default=3.0)
    parser.add_argument(
        "--interp-nx",
        type=int,
        default=None,
        help="Optional x voxel count after trilinear interpolation. Defaults to --nx.",
    )
    parser.add_argument(
        "--interp-ny",
        type=int,
        default=None,
        help="Optional y voxel count after trilinear interpolation. Defaults to --ny.",
    )
    parser.add_argument(
        "--interp-nz",
        type=int,
        default=None,
        help="Optional z voxel count after trilinear interpolation. Defaults to --nz.",
    )
    parser.add_argument(
        "--interp-detector-batch-size",
        type=int,
        default=256,
        help="Detector batch size used when interpolating SysMat_tmp to a smaller grid.",
    )
    parser.add_argument(
        "--roi-diameter-mm",
        type=float,
        default=150.0,
        help="Diameter of the center cylindrical ROI used to average CRC and VAR.",
    )
    parser.add_argument(
        "--roi-height-mm",
        type=float,
        default=None,
        help="Optional height of the center cylindrical ROI. Defaults to the full z extent.",
    )
    parser.add_argument(
        "--sample-grid-nx",
        type=int,
        default=20,
        help="Deprecated. Ignored by the direct script; kept only for CLI compatibility.",
    )
    parser.add_argument(
        "--sample-grid-ny",
        type=int,
        default=20,
        help="Deprecated. Ignored by the direct script; kept only for CLI compatibility.",
    )
    parser.add_argument(
        "--sample-grid-nz",
        type=int,
        default=5,
        help="Deprecated. Ignored by the direct script; kept only for CLI compatibility.",
    )
    parser.add_argument(
        "--sample-spacing-mm",
        type=float,
        default=10.0,
        help="Deprecated. Ignored by the direct script; kept only for CLI compatibility.",
    )
    parser.add_argument("--beta-level-start", type=int, default=-9)
    parser.add_argument("--beta-level-end", type=int, default=-1)
    parser.add_argument("--beta-multipliers", type=float, nargs="+", default=[1.0, 3.0])
    parser.add_argument(
        "--solve-batch-size",
        type=int,
        default=128,
        help="Number of sampled points solved together per direct solve batch.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Dense operator dtype.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
        help=(
            "Device used for dense direct algebra. Default is cpu, since the full dense FIM "
            "is very large."
        ),
    )
    parser.add_argument("--disable-spacing-weight", action="store_true")
    parser.add_argument("--run-name", type=str, default="SinglePhoton_CRCVAR_Direct")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--fim-row-chunk",
        type=int,
        default=512,
        help="Detector-row chunk used when accumulating the dense FIM.",
    )
    return parser.parse_args()


def resolve_factor_dirs(factor_dirs: list[Path]) -> list[Path]:
    resolved = []
    for factor_dir in factor_dirs:
        path = factor_dir.resolve() if factor_dir.is_absolute() else (PROJECT_ROOT / factor_dir).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Factor directory not found: {path}")
        if not (path / "SysMat_tmp").is_file():
            raise FileNotFoundError(f"SysMat_tmp not found under factor directory: {path}")
        resolved.append(path)
    return resolved


def beta_schedule(level_start: int, level_end: int, multipliers: list[float]) -> list[float]:
    values: list[float] = []
    for level in range(level_start, level_end):
        for multiplier in multipliers:
            values.append(float(multiplier) * (10.0 ** level))
    return values


def torch_dtype_from_name(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def choose_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sysmat_tmp(
    sysmat_path: Path,
    input_nx: int,
    input_ny: int,
    input_nz: int,
    target_nx: int,
    target_ny: int,
    target_nz: int,
    device: torch.device,
    dtype: torch.dtype,
    interp_detector_batch_size: int,
) -> tuple[torch.Tensor, int]:
    input_num_voxels = input_nx * input_ny * input_nz
    raw = np.fromfile(sysmat_path, dtype=np.float32)
    if raw.size % input_num_voxels != 0:
        raise ValueError(
            f"SysMat_tmp size is incompatible with num_voxels={input_num_voxels}: "
            f"{raw.size} float32 values at {sysmat_path}"
        )
    num_detectors = raw.size // input_num_voxels
    matrix_np = raw.reshape((input_num_voxels, num_detectors), order="F").T.copy()

    if (input_nx, input_ny, input_nz) == (target_nx, target_ny, target_nz):
        matrix = torch.from_numpy(matrix_np).to(device=device, dtype=dtype, non_blocking=True)
        return matrix, num_detectors

    target_num_voxels = target_nx * target_ny * target_nz
    matrix_interp = torch.empty((num_detectors, target_num_voxels), device=device, dtype=dtype)
    total_batches = math.ceil(num_detectors / interp_detector_batch_size)
    interp_start = time.time()

    for batch_idx, start in enumerate(range(0, num_detectors, interp_detector_batch_size), start=1):
        end = min(start + interp_detector_batch_size, num_detectors)
        batch_np = matrix_np[start:end].reshape(end - start, input_nz, input_ny, input_nx)
        batch_tensor = torch.from_numpy(batch_np).to(device=device, dtype=dtype, non_blocking=True).unsqueeze(1)
        batch_interp = torch_f.interpolate(
            batch_tensor,
            size=(target_nz, target_ny, target_nx),
            mode="trilinear",
            align_corners=True,
        )
        matrix_interp[start:end] = batch_interp[:, 0].reshape(end - start, target_num_voxels)

        if should_log_progress(batch_idx, total_batches):
            elapsed = time.time() - interp_start
            eta = elapsed / batch_idx * max(0, total_batches - batch_idx)
            print(
                f"  interpolation batch {batch_idx}/{total_batches}  detectors {end:5d}/{num_detectors} "
                f"({100.0 * end / num_detectors:5.1f}%)  "
                f"elapsed={format_seconds(elapsed)}  eta={format_seconds(eta)}",
                flush=True,
            )

    return matrix_interp, num_detectors


def effective_spacing_mm(input_n: int, input_spacing_mm: float, target_n: int) -> float:
    if target_n <= 0:
        raise ValueError("Target grid size must be positive.")
    if input_n <= 0:
        raise ValueError("Input grid size must be positive.")
    if target_n == input_n:
        return float(input_spacing_mm)
    if input_n == 1 or target_n == 1:
        return float(input_spacing_mm)
    return float(input_spacing_mm) * float(input_n - 1) / float(target_n - 1)


def build_center_cylindrical_roi_index_weights(
    nx: int,
    ny: int,
    nz: int,
    sx_mm: float,
    sy_mm: float,
    sz_mm: float,
    roi_diameter_mm: float,
    roi_height_mm: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_coords = (torch.arange(nx, dtype=torch.float64) - (nx / 2.0 - 0.5)) * sx_mm
    y_coords = (torch.arange(ny, dtype=torch.float64) - (ny / 2.0 - 0.5)) * sy_mm
    z_coords = (torch.arange(nz, dtype=torch.float64) - (nz / 2.0 - 0.5)) * sz_mm

    zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
    radial_mask = (xx.square() + yy.square()) <= (0.5 * roi_diameter_mm) ** 2

    if roi_height_mm is None:
        axial_mask = torch.ones_like(radial_mask, dtype=torch.bool)
    else:
        axial_mask = torch.abs(zz) <= (0.5 * roi_height_mm)

    roi_mask = radial_mask & axial_mask
    roi_indices = torch.nonzero(roi_mask.reshape(-1), as_tuple=False).reshape(-1).to(torch.int64)
    if roi_indices.numel() == 0:
        raise ValueError(
            "Center cylindrical ROI does not contain any voxel. "
            "Check --roi-diameter-mm / --roi-height-mm against the working grid."
        )

    weights = torch.full((roi_indices.numel(),), 1.0 / float(roi_indices.numel()), dtype=torch.float64)
    return roi_indices, weights


def build_regularizer_dense(
    nx: int,
    ny: int,
    nz: int,
    sx_mm: float,
    sy_mm: float,
    sz_mm: float,
    spacing_weighted: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    build_start = time.time()
    num_voxels = nx * ny * nz
    wx = 1.0 / (sx_mm * sx_mm) if spacing_weighted else 1.0
    wy = 1.0 / (sy_mm * sy_mm) if spacing_weighted else 1.0
    wz = 1.0 / (sz_mm * sz_mm) if spacing_weighted else 1.0

    r = torch.zeros((num_voxels, num_voxels), device=device, dtype=dtype)
    diag = torch.zeros(num_voxels, device=device, dtype=dtype)
    voxel_index = torch.arange(num_voxels, device=device, dtype=torch.int64).reshape(nz, ny, nx)

    x_left = voxel_index[:, :, :-1].reshape(-1)
    x_right = voxel_index[:, :, 1:].reshape(-1)
    if x_left.numel() > 0:
        r[x_left, x_right] = -wx
        r[x_right, x_left] = -wx
        diag.index_add_(0, x_left, torch.full((x_left.numel(),), wx, device=device, dtype=dtype))
        diag.index_add_(0, x_right, torch.full((x_right.numel(),), wx, device=device, dtype=dtype))

    y_front = voxel_index[:, :-1, :].reshape(-1)
    y_back = voxel_index[:, 1:, :].reshape(-1)
    if y_front.numel() > 0:
        r[y_front, y_back] = -wy
        r[y_back, y_front] = -wy
        diag.index_add_(0, y_front, torch.full((y_front.numel(),), wy, device=device, dtype=dtype))
        diag.index_add_(0, y_back, torch.full((y_back.numel(),), wy, device=device, dtype=dtype))

    z_lower = voxel_index[:-1, :, :].reshape(-1)
    z_upper = voxel_index[1:, :, :].reshape(-1)
    if z_lower.numel() > 0:
        r[z_lower, z_upper] = -wz
        r[z_upper, z_lower] = -wz
        diag.index_add_(0, z_lower, torch.full((z_lower.numel(),), wz, device=device, dtype=dtype))
        diag.index_add_(0, z_upper, torch.full((z_upper.numel(),), wz, device=device, dtype=dtype))

    diag_index = torch.arange(num_voxels, device=device, dtype=torch.int64)
    r[diag_index, diag_index] = diag
    print(
        f"  dense R built for {nx}x{ny}x{nz} in {format_seconds(time.time() - build_start)}",
        flush=True,
    )
    return r


def build_dense_fim(sysmat: torch.Tensor, fim_row_chunk: int) -> tuple[torch.Tensor, float]:
    num_detectors, num_voxels = sysmat.shape
    ones = torch.ones((num_voxels, 1), dtype=sysmat.dtype, device=sysmat.device)
    proj = torch.clamp(sysmat @ ones, min=EPS).squeeze(1)
    proj_inv = proj.reciprocal()
    avg_sensitivity = float(torch.sum(sysmat @ ones).item() / num_voxels)

    fim = torch.zeros((num_voxels, num_voxels), dtype=sysmat.dtype, device=sysmat.device)
    total_chunks = math.ceil(num_detectors / fim_row_chunk)
    fim_start = time.time()
    for chunk_idx, start in enumerate(range(0, num_detectors, fim_row_chunk), start=1):
        end = min(start + fim_row_chunk, num_detectors)
        s_chunk = sysmat[start:end, :]
        weighted = s_chunk * proj_inv[start:end].unsqueeze(1)
        fim.addmm_(s_chunk.T, weighted)
        if should_log_progress(chunk_idx, total_chunks):
            chunk_elapsed = time.time() - fim_start
            eta = chunk_elapsed / chunk_idx * max(0, total_chunks - chunk_idx)
            print(
                f"  FIM chunk {chunk_idx}/{total_chunks}  rows {end:5d}/{num_detectors} "
                f"({100.0 * end / num_detectors:5.1f}%)  "
                f"elapsed={format_seconds(chunk_elapsed)}  eta={format_seconds(eta)}",
                flush=True,
            )
    return fim, avg_sensitivity


def build_basis_batch(
    sample_indices: torch.Tensor,
    num_voxels: int,
    device: torch.device,
    dtype: torch.dtype,
    start: int,
    end: int,
) -> torch.Tensor:
    batch_size = end - start
    basis = torch.zeros((num_voxels, batch_size), device=device, dtype=dtype)
    cols = sample_indices[start:end]
    basis[cols, torch.arange(batch_size, device=device)] = 1.0
    return basis


def save_float32_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values.astype(np.float32, copy=False).tofile(path)


def analyze_factor(
    factor_dir: Path,
    factor_label: str,
    factor_index: int,
    total_factors: int,
    beta_values: list[float],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    r_dense: torch.Tensor,
    roi_indices_cpu: torch.Tensor,
    roi_weights_cpu: torch.Tensor,
) -> FactorResult:
    time_start = time.time()
    work_nx = args.interp_nx or args.nx
    work_ny = args.interp_ny or args.ny
    work_nz = args.interp_nz or args.nz
    work_sx_mm = effective_spacing_mm(args.nx, args.sx_mm, work_nx)
    work_sy_mm = effective_spacing_mm(args.ny, args.sy_mm, work_ny)
    work_sz_mm = effective_spacing_mm(args.nz, args.sz_mm, work_nz)
    num_voxels = work_nx * work_ny * work_nz
    print(
        f"[factor {factor_index}/{total_factors}] [{factor_label}] loading SysMat_tmp from {factor_dir / 'SysMat_tmp'}",
        flush=True,
    )
    print(
        f"[factor {factor_index}/{total_factors}] [{factor_label}] "
        f"grid {args.nx}x{args.ny}x{args.nz} -> {work_nx}x{work_ny}x{work_nz}",
        flush=True,
    )
    sysmat, num_detectors = load_sysmat_tmp(
        factor_dir / "SysMat_tmp",
        input_nx=args.nx,
        input_ny=args.ny,
        input_nz=args.nz,
        target_nx=work_nx,
        target_ny=work_ny,
        target_nz=work_nz,
        device=device,
        dtype=dtype,
        interp_detector_batch_size=args.interp_detector_batch_size,
    )
    print(
        f"[factor {factor_index}/{total_factors}] [{factor_label}] loaded "
        f"{num_detectors} detectors x {num_voxels} voxels on {device}",
        flush=True,
    )

    print(
        f"[factor {factor_index}/{total_factors}] [{factor_label}] building dense FIM",
        flush=True,
    )
    fim, avg_sensitivity = build_dense_fim(sysmat, args.fim_row_chunk)
    del sysmat
    if device.type == "cuda":
        torch.cuda.empty_cache()

    roi_indices = roi_indices_cpu.to(device=device, dtype=torch.int64)
    roi_weights = roi_weights_cpu.to(device=device, dtype=dtype)
    total_batches = math.ceil(roi_indices.numel() / args.solve_batch_size)
    print(
        f"[factor {factor_index}/{total_factors}] [{factor_label}] "
        f"starting {len(beta_values)} beta values, {roi_indices.numel()} ROI voxels, "
        f"{total_batches} direct-solve batches/beta",
        flush=True,
    )

    crc_mean_list: list[float] = []
    var_mean_list: list[float] = []

    for beta_idx, beta in enumerate(beta_values, start=1):
        beta_start = time.time()
        print(
            f"[factor {factor_index}/{total_factors}] [{factor_label}] "
            f"beta {beta_idx}/{len(beta_values)} = {beta:.3e} factorizing dense system",
            flush=True,
        )
        system = fim + float(beta) * r_dense
        chol = torch.linalg.cholesky(system)

        crc_diag_values: list[torch.Tensor] = []
        var_diag_values: list[torch.Tensor] = []

        for batch_idx, start in enumerate(range(0, roi_indices.numel(), args.solve_batch_size), start=1):
            end = min(start + args.solve_batch_size, roi_indices.numel())
            rhs_var = build_basis_batch(
                sample_indices=roi_indices,
                num_voxels=num_voxels,
                device=device,
                dtype=dtype,
                start=start,
                end=end,
            )
            rhs_crc = fim[:, roi_indices[start:end]]

            x_var = torch.cholesky_solve(rhs_var, chol)
            x_crc = torch.cholesky_solve(rhs_crc, chol)

            batch_indices = roi_indices[start:end]
            crc_batch = x_crc[batch_indices, torch.arange(end - start, device=device)]
            fim_x_var = fim @ x_var
            var_batch = torch.sum(x_var * fim_x_var, dim=0)

            crc_diag_values.append(crc_batch)
            var_diag_values.append(var_batch)

            if should_log_progress(batch_idx, total_batches):
                batch_elapsed = time.time() - beta_start
                eta = batch_elapsed / batch_idx * max(0, total_batches - batch_idx)
                print(
                    f"[factor {factor_index}/{total_factors}] [{factor_label}] "
                    f"beta {beta_idx}/{len(beta_values)} batch {batch_idx}/{total_batches} "
                    f"({100.0 * batch_idx / total_batches:5.1f}%)  "
                    f"elapsed={format_seconds(batch_elapsed)}  eta={format_seconds(eta)}",
                    flush=True,
                )

        crc_all = torch.cat(crc_diag_values)
        var_all = torch.cat(var_diag_values)
        crc_mean = float(torch.sum(crc_all * roi_weights).item())
        var_mean = float(torch.sum(var_all * roi_weights).item())
        crc_mean_list.append(crc_mean)
        var_mean_list.append(var_mean)

        beta_elapsed = time.time() - beta_start
        factor_elapsed = time.time() - time_start
        beta_eta = factor_elapsed / beta_idx * max(0, len(beta_values) - beta_idx)
        print(
            f"[factor {factor_index}/{total_factors}] [{factor_label}] "
            f"beta {beta_idx}/{len(beta_values)} = {beta:.3e} finished  "
            f"CRC={crc_mean:.6e}  VAR={var_mean:.6e}  "
            f"beta_elapsed={format_seconds(beta_elapsed)}  "
            f"factor_elapsed={format_seconds(factor_elapsed)}  "
            f"factor_eta={format_seconds(beta_eta)}",
            flush=True,
        )

        del chol
        del system
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - time_start
    print(
        f"[factor {factor_index}/{total_factors}] [{factor_label}] finished in {format_seconds(elapsed)}",
        flush=True,
    )
    return FactorResult(
        factor_label=factor_label,
        factor_dir=str(factor_dir),
        num_voxels=num_voxels,
        num_detectors=num_detectors,
        avg_sensitivity=avg_sensitivity,
        roi_voxel_count=int(roi_indices.numel()),
        crc_mean=crc_mean_list,
        var_mean=var_mean_list,
        elapsed_seconds=elapsed,
    )


def main() -> None:
    run_start = time.time()
    args = parse_args()
    factor_dirs = resolve_factor_dirs(args.factor_dirs)
    factor_labels = args.factor_labels or [path.name for path in factor_dirs]
    if len(factor_labels) != len(factor_dirs):
        raise ValueError("--factor-labels must match --factor-dirs in length.")

    beta_values = beta_schedule(args.beta_level_start, args.beta_level_end, args.beta_multipliers)
    device = choose_device(args.device)
    dtype = torch_dtype_from_name(args.dtype)
    work_nx = args.interp_nx or args.nx
    work_ny = args.interp_ny or args.ny
    work_nz = args.interp_nz or args.nz
    work_sx_mm = effective_spacing_mm(args.nx, args.sx_mm, work_nx)
    work_sy_mm = effective_spacing_mm(args.ny, args.sy_mm, work_ny)
    work_sz_mm = effective_spacing_mm(args.nz, args.sz_mm, work_nz)

    roi_indices, roi_weights = build_center_cylindrical_roi_index_weights(
        nx=work_nx,
        ny=work_ny,
        nz=work_nz,
        sx_mm=work_sx_mm,
        sy_mm=work_sy_mm,
        sz_mm=work_sz_mm,
        roi_diameter_mm=args.roi_diameter_mm,
        roi_height_mm=args.roi_height_mm,
    )

    output_dir = (args.output_root / args.run_name / "SinglePhoton").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building shared dense R once for all factors...", flush=True)
    r_build_start = time.time()
    r_dense = build_regularizer_dense(
        nx=work_nx,
        ny=work_ny,
        nz=work_nz,
        sx_mm=work_sx_mm,
        sy_mm=work_sy_mm,
        sz_mm=work_sz_mm,
        spacing_weighted=not args.disable_spacing_weight,
        device=device,
        dtype=dtype,
    )
    print(
        f"Shared dense R ready in {format_seconds(time.time() - r_build_start)}",
        flush=True,
    )

    print("=" * 72)
    print("Single-Photon CRC-VAR Direct Analysis")
    print("=" * 72)
    print(f"Device                : {device}")
    print(f"Dtype                 : {dtype}")
    print(f"Output dir            : {output_dir}")
    print(f"Factor labels         : {factor_labels}")
    print(f"Input grid            : {args.nx} x {args.ny} x {args.nz}")
    print(f"Working grid          : {work_nx} x {work_ny} x {work_nz}")
    print(f"Working spacing (mm)  : {work_sx_mm:.4f}, {work_sy_mm:.4f}, {work_sz_mm:.4f}")
    print(f"Beta count            : {len(beta_values)}")
    print(f"ROI diameter (mm)     : {args.roi_diameter_mm:.4f}")
    if args.roi_height_mm is None:
        print("ROI height (mm)       : full z extent")
    else:
        print(f"ROI height (mm)       : {args.roi_height_mm:.4f}")
    print(f"ROI voxel count       : {roi_indices.numel()}")
    print(f"Solve batch size      : {args.solve_batch_size}")
    print()

    results: list[FactorResult] = []
    total_factors = len(factor_dirs)
    for factor_idx, (factor_dir, factor_label) in enumerate(zip(factor_dirs, factor_labels), start=1):
        results.append(
            analyze_factor(
                factor_dir=factor_dir,
                factor_label=factor_label,
                factor_index=factor_idx,
                total_factors=total_factors,
                beta_values=beta_values,
                args=args,
                device=device,
                dtype=dtype,
                r_dense=r_dense,
                roi_indices_cpu=roi_indices,
                roi_weights_cpu=roi_weights,
            )
        )
        elapsed = time.time() - run_start
        eta = elapsed / factor_idx * max(0, total_factors - factor_idx)
        print(
            f"[run] completed factor {factor_idx}/{total_factors}  "
            f"elapsed={format_seconds(elapsed)}  eta={format_seconds(eta)}",
            flush=True,
        )

    save_float32_array(output_dir / "beta_values", np.asarray(beta_values, dtype=np.float32))
    for result in results:
        save_float32_array(output_dir / f"CRC_mean_{result.factor_label}", np.asarray(result.crc_mean))
        save_float32_array(output_dir / f"Var_mean_{result.factor_label}", np.asarray(result.var_mean))

    summary = {
        "config": {
            "factor_dirs": [str(path) for path in factor_dirs],
            "factor_labels": factor_labels,
            "nx": args.nx,
            "ny": args.ny,
            "nz": args.nz,
            "sx_mm": args.sx_mm,
            "sy_mm": args.sy_mm,
            "sz_mm": args.sz_mm,
            "work_nx": work_nx,
            "work_ny": work_ny,
            "work_nz": work_nz,
            "work_sx_mm": work_sx_mm,
            "work_sy_mm": work_sy_mm,
            "work_sz_mm": work_sz_mm,
            "roi_diameter_mm": args.roi_diameter_mm,
            "roi_height_mm": args.roi_height_mm,
            "roi_voxel_count": int(roi_indices.numel()),
            "beta_values": beta_values,
            "solve_batch_size": args.solve_batch_size,
            "dtype": args.dtype,
            "device": str(device),
            "fim_row_chunk": args.fim_row_chunk,
            "interp_detector_batch_size": args.interp_detector_batch_size,
            "spacing_weighted_regularizer": not args.disable_spacing_weight,
            "project_root": str(PROJECT_ROOT),
        },
        "results": [asdict(result) for result in results],
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Summary written to: {output_dir / 'summary.json'}")
    print(f"Total elapsed        : {format_seconds(time.time() - run_start)}")


if __name__ == "__main__":
    main()
