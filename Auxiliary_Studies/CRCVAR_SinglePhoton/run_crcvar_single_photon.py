from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


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
    sample_point_count: int
    unique_sample_voxel_count: int
    crc_mean: list[float]
    var_mean: list[float]
    cg_iterations_crc: list[float]
    cg_iterations_var: list[float]
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
            "Compute CRC-VAR curves from single-photon SysMat_tmp factors using a GPU-accelerated, "
            "matrix-free Fisher operator and stochastic ROI trace estimation."
        )
    )
    parser.add_argument(
        "--factor-dirs",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Factor directories containing SysMat_tmp, for example "
            "Factors/140keV_RotateNum20 Factors/140keV_RotateNum20_SPECTEHENaI"
        ),
    )
    parser.add_argument(
        "--factor-labels",
        type=str,
        nargs="*",
        default=None,
        help="Optional labels matching --factor-dirs. Defaults to each directory name.",
    )
    parser.add_argument("--nx", type=int, default=51, help="Cartesian voxel count along x.")
    parser.add_argument("--ny", type=int, default=51, help="Cartesian voxel count along y.")
    parser.add_argument("--nz", type=int, default=20, help="Cartesian voxel count along z.")
    parser.add_argument("--sx-mm", type=float, default=6.0, help="Voxel size along x in mm.")
    parser.add_argument("--sy-mm", type=float, default=6.0, help="Voxel size along y in mm.")
    parser.add_argument("--sz-mm", type=float, default=3.0, help="Voxel size along z in mm.")
    parser.add_argument(
        "--sample-grid-nx",
        type=int,
        default=20,
        help="Number of sample points along x used to average CRC and VAR.",
    )
    parser.add_argument(
        "--sample-grid-ny",
        type=int,
        default=20,
        help="Number of sample points along y used to average CRC and VAR.",
    )
    parser.add_argument(
        "--sample-grid-nz",
        type=int,
        default=5,
        help="Number of sample points along z used to average CRC and VAR.",
    )
    parser.add_argument(
        "--sample-spacing-mm",
        type=float,
        default=10.0,
        help="Physical spacing in mm between neighboring sample points.",
    )
    parser.add_argument(
        "--beta-level-start",
        type=int,
        default=-9,
        help="Start exponent for beta sweep. Beta values are 1eL and 3eL.",
    )
    parser.add_argument(
        "--beta-level-end",
        type=int,
        default=-1,
        help="End exponent for beta sweep, exclusive.",
    )
    parser.add_argument(
        "--beta-multipliers",
        type=float,
        nargs="+",
        default=[1.0, 3.0],
        help="Multipliers used inside each beta decade.",
    )
    parser.add_argument(
        "--point-batch-size",
        type=int,
        default=8,
        help="How many sampled points to solve simultaneously in each PCG batch.",
    )
    parser.add_argument(
        "--cg-tol",
        type=float,
        default=1.0e-4,
        help="Relative tolerance used by batched preconditioned conjugate gradient.",
    )
    parser.add_argument(
        "--cg-maxiter",
        type=int,
        default=200,
        help="Maximum PCG iterations for each right-hand-side batch.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Torch dtype used by the operator and solver.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device. 'auto' prefers CUDA.",
    )
    parser.add_argument(
        "--disable-spacing-weight",
        action="store_true",
        help="Use unit x/y/z neighbor weights instead of spacing-aware 1/h^2 weights.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260420,
        help="Base RNG seed for stochastic trace probes.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="SinglePhoton_CRCVAR",
        help="Result folder name created under Auxiliary_Studies/CRCVAR_SinglePhoton/Result.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory. Defaults to Auxiliary_Studies/CRCVAR_SinglePhoton/Result.",
    )
    return parser.parse_args()


def resolve_factor_dirs(factor_dirs: list[Path]) -> list[Path]:
    resolved = []
    for factor_dir in factor_dirs:
        if factor_dir.is_absolute():
            path = factor_dir.resolve()
        else:
            path = (PROJECT_ROOT / factor_dir).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Factor directory not found: {path}")
        sysmat_path = path / "SysMat_tmp"
        if not sysmat_path.is_file():
            raise FileNotFoundError(f"SysMat_tmp not found under factor directory: {path}")
        resolved.append(path)
    return resolved


def beta_schedule(level_start: int, level_end: int, multipliers: list[float]) -> list[float]:
    beta_values: list[float] = []
    for level in range(level_start, level_end):
        for multiplier in multipliers:
            beta_values.append(float(multiplier) * (10.0 ** level))
    return beta_values


def torch_dtype_from_name(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def choose_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_sysmat_tmp(
    sysmat_path: Path,
    num_voxels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    raw = np.fromfile(sysmat_path, dtype=np.float32)
    if raw.size % num_voxels != 0:
        raise ValueError(
            f"SysMat_tmp size is incompatible with num_voxels={num_voxels}: "
            f"{raw.size} float32 values at {sysmat_path}"
        )
    num_detectors = raw.size // num_voxels
    matrix_np = raw.reshape((num_voxels, num_detectors), order="F").T.copy()
    matrix = torch.from_numpy(matrix_np).to(device=device, dtype=dtype, non_blocking=True)
    return matrix, num_detectors


def build_sample_point_index_weights(
    nx: int,
    ny: int,
    nz: int,
    sx_mm: float,
    sy_mm: float,
    sz_mm: float,
    sample_grid_nx: int,
    sample_grid_ny: int,
    sample_grid_nz: int,
    sample_spacing_mm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_points = (torch.arange(sample_grid_nx, dtype=torch.float64) - (sample_grid_nx - 1) / 2.0) * sample_spacing_mm
    y_points = (torch.arange(sample_grid_ny, dtype=torch.float64) - (sample_grid_ny - 1) / 2.0) * sample_spacing_mm
    z_points = (torch.arange(sample_grid_nz, dtype=torch.float64) - (sample_grid_nz - 1) / 2.0) * sample_spacing_mm

    zz, yy, xx = torch.meshgrid(z_points, y_points, x_points, indexing="ij")
    sample_points_mm = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)

    ix = torch.round(sample_points_mm[:, 0] / sx_mm + nx / 2.0 - 0.5).to(torch.int64)
    iy = torch.round(sample_points_mm[:, 1] / sy_mm + ny / 2.0 - 0.5).to(torch.int64)
    iz = torch.round(sample_points_mm[:, 2] / sz_mm + nz / 2.0 - 0.5).to(torch.int64)

    ix = ix.clamp(0, nx - 1)
    iy = iy.clamp(0, ny - 1)
    iz = iz.clamp(0, nz - 1)

    flat_indices = iz * (nx * ny) + iy * nx + ix
    unique_indices, inverse, counts = torch.unique(
        flat_indices,
        return_inverse=True,
        return_counts=True,
        sorted=True,
    )
    weights = counts.to(torch.float64) / float(flat_indices.numel())
    return unique_indices, weights, sample_points_mm


def build_regularizer_diag(
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
    wx = 1.0 / (sx_mm * sx_mm) if spacing_weighted else 1.0
    wy = 1.0 / (sy_mm * sy_mm) if spacing_weighted else 1.0
    wz = 1.0 / (sz_mm * sz_mm) if spacing_weighted else 1.0

    diag = torch.zeros((nz, ny, nx), dtype=dtype, device=device)
    diag[:, :, 1:] += wx
    diag[:, :, :-1] += wx
    diag[:, 1:, :] += wy
    diag[:, :-1, :] += wy
    diag[1:, :, :] += wz
    diag[:-1, :, :] += wz
    return diag.reshape(-1)


def regularizer_matvec(
    x: torch.Tensor,
    nx: int,
    ny: int,
    nz: int,
    sx_mm: float,
    sy_mm: float,
    sz_mm: float,
    spacing_weighted: bool,
) -> torch.Tensor:
    wx = 1.0 / (sx_mm * sx_mm) if spacing_weighted else 1.0
    wy = 1.0 / (sy_mm * sy_mm) if spacing_weighted else 1.0
    wz = 1.0 / (sz_mm * sz_mm) if spacing_weighted else 1.0

    x4 = x.T.reshape(-1, nz, ny, nx)
    out = torch.zeros_like(x4)

    diff = x4[:, :, :, 1:] - x4[:, :, :, :-1]
    out[:, :, :, 1:] += wx * diff
    out[:, :, :, :-1] -= wx * diff

    diff = x4[:, :, 1:, :] - x4[:, :, :-1, :]
    out[:, :, 1:, :] += wy * diff
    out[:, :, :-1, :] -= wy * diff

    diff = x4[:, 1:, :, :] - x4[:, :-1, :, :]
    out[:, 1:, :, :] += wz * diff
    out[:, :-1, :, :] -= wz * diff

    return out.reshape(x.shape[1], x.shape[0]).T


class SinglePhotonOperator:
    def __init__(
        self,
        sysmat: torch.Tensor,
        nx: int,
        ny: int,
        nz: int,
        sx_mm: float,
        sy_mm: float,
        sz_mm: float,
        spacing_weighted: bool,
    ) -> None:
        self.sysmat = sysmat
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.sx_mm = sx_mm
        self.sy_mm = sy_mm
        self.sz_mm = sz_mm
        self.spacing_weighted = spacing_weighted

        ones = torch.ones((sysmat.shape[1], 1), dtype=sysmat.dtype, device=sysmat.device)
        self.proj = torch.clamp(sysmat @ ones, min=EPS).squeeze(1)
        self.proj_inv = self.proj.reciprocal()
        self.diag_fim = torch.sum(sysmat.square() * self.proj_inv.unsqueeze(1), dim=0)
        self.diag_r = build_regularizer_diag(
            nx=nx,
            ny=ny,
            nz=nz,
            sx_mm=sx_mm,
            sy_mm=sy_mm,
            sz_mm=sz_mm,
            spacing_weighted=spacing_weighted,
            device=sysmat.device,
            dtype=sysmat.dtype,
        )
        self.avg_sensitivity = float(torch.sum(sysmat @ ones).item() / sysmat.shape[1])

    def fim_matvec(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.sysmat @ x
        weighted = proj * self.proj_inv.unsqueeze(1)
        return self.sysmat.T @ weighted

    def regularizer_matvec(self, x: torch.Tensor) -> torch.Tensor:
        return regularizer_matvec(
            x=x,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            sx_mm=self.sx_mm,
            sy_mm=self.sy_mm,
            sz_mm=self.sz_mm,
            spacing_weighted=self.spacing_weighted,
        )

    def system_matvec(self, x: torch.Tensor, beta: float) -> torch.Tensor:
        return self.fim_matvec(x) + float(beta) * self.regularizer_matvec(x)

    def preconditioner(self, beta: float) -> torch.Tensor:
        diag = self.diag_fim + float(beta) * self.diag_r
        return diag.clamp_min(EPS).reciprocal()


def batched_pcg(
    matvec,
    rhs: torch.Tensor,
    m_inv: torch.Tensor,
    tol: float,
    maxiter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    z = m_inv.unsqueeze(1) * r
    p = z.clone()

    rz_old = torch.sum(r * z, dim=0)
    rhs_norm = torch.linalg.norm(rhs, dim=0).clamp_min(EPS)
    iters = torch.zeros(rhs.shape[1], dtype=torch.int32, device=rhs.device)
    converged = torch.zeros(rhs.shape[1], dtype=torch.bool, device=rhs.device)

    for iteration in range(1, maxiter + 1):
        ap = matvec(p)
        denom = torch.sum(p * ap, dim=0).clamp_min(EPS)
        alpha = rz_old / denom
        x = x + p * alpha.unsqueeze(0)
        r = r - ap * alpha.unsqueeze(0)

        rel_res = torch.linalg.norm(r, dim=0) / rhs_norm
        newly_converged = (~converged) & (rel_res <= tol)
        iters[newly_converged] = iteration
        converged |= newly_converged
        if torch.all(converged):
            break

        z = m_inv.unsqueeze(1) * r
        rz_new = torch.sum(r * z, dim=0)
        beta = rz_new / rz_old.clamp_min(EPS)
        p = z + p * beta.unsqueeze(0)
        rz_old = rz_new

    iters[~converged] = maxiter
    return x, iters


def build_basis_batch(
    sample_indices: torch.Tensor,
    num_voxels: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    start: int,
) -> torch.Tensor:
    basis = torch.zeros((num_voxels, batch_size), device=device, dtype=dtype)
    cols = sample_indices[start : start + batch_size]
    basis[cols, torch.arange(cols.numel(), device=device)] = 1.0
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
    sample_indices_cpu: torch.Tensor,
    sample_weights_cpu: torch.Tensor,
) -> FactorResult:
    time_start = time.time()
    num_voxels = args.nx * args.ny * args.nz
    sysmat_path = factor_dir / "SysMat_tmp"
    print(
        f"[factor {factor_index}/{total_factors}] [{factor_label}] loading SysMat_tmp from {sysmat_path}",
        flush=True,
    )
    sysmat, num_detectors = load_sysmat_tmp(sysmat_path, num_voxels, device, dtype)
    print(
        f"[factor {factor_index}/{total_factors}] [{factor_label}] loaded "
        f"{num_detectors} detectors x {num_voxels} voxels on {device}",
        flush=True,
    )
    operator = SinglePhotonOperator(
        sysmat=sysmat,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        sx_mm=args.sx_mm,
        sy_mm=args.sy_mm,
        sz_mm=args.sz_mm,
        spacing_weighted=not args.disable_spacing_weight,
    )

    sample_indices = sample_indices_cpu.to(device=device, dtype=torch.int64)
    sample_weights = sample_weights_cpu.to(device=device, dtype=dtype)
    if sample_indices.numel() == 0:
        raise ValueError("No sample points were resolved to valid voxels.")
    total_batches = math.ceil(sample_indices.numel() / args.point_batch_size)
    print(
        f"[factor {factor_index}/{total_factors}] [{factor_label}] "
        f"starting {len(beta_values)} beta values, {sample_indices.numel()} unique samples, "
        f"{total_batches} PCG batches/beta",
        flush=True,
    )

    crc_mean_list: list[float] = []
    var_mean_list: list[float] = []
    cg_iterations_crc: list[float] = []
    cg_iterations_var: list[float] = []

    for beta_idx, beta in enumerate(beta_values, start=1):
        beta_start = time.time()
        print(
            f"[factor {factor_index}/{total_factors}] [{factor_label}] "
            f"beta {beta_idx}/{len(beta_values)} = {beta:.3e} started",
            flush=True,
        )
        m_inv = operator.preconditioner(beta)

        crc_samples: list[torch.Tensor] = []
        var_samples: list[torch.Tensor] = []
        cg_iters_crc_batch: list[torch.Tensor] = []
        cg_iters_var_batch: list[torch.Tensor] = []

        for batch_idx, start in enumerate(range(0, sample_indices.numel(), args.point_batch_size), start=1):
            batch_size = min(args.point_batch_size, sample_indices.numel() - start)
            basis = build_basis_batch(
                sample_indices=sample_indices,
                num_voxels=num_voxels,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                start=start,
            )

            rhs_var = basis
            sol_var, it_var = batched_pcg(
                matvec=lambda x, beta_value=beta: operator.system_matvec(x, beta_value),
                rhs=rhs_var,
                m_inv=m_inv,
                tol=args.cg_tol,
                maxiter=args.cg_maxiter,
            )
            fim_sol_var = operator.fim_matvec(sol_var)
            var_batch = torch.sum(sol_var * fim_sol_var, dim=0)

            rhs_crc = operator.fim_matvec(basis)
            sol_crc, it_crc = batched_pcg(
                matvec=lambda x, beta_value=beta: operator.system_matvec(x, beta_value),
                rhs=rhs_crc,
                m_inv=m_inv,
                tol=args.cg_tol,
                maxiter=args.cg_maxiter,
            )
            batch_indices = sample_indices[start : start + batch_size]
            crc_batch = sol_crc[batch_indices, torch.arange(batch_size, device=device)]

            crc_samples.append(crc_batch.detach())
            var_samples.append(var_batch.detach())
            cg_iters_crc_batch.append(it_crc.to(dtype=torch.float32))
            cg_iters_var_batch.append(it_var.to(dtype=torch.float32))

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

        crc_all = torch.cat(crc_samples)
        var_all = torch.cat(var_samples)
        it_crc_all = torch.cat(cg_iters_crc_batch)
        it_var_all = torch.cat(cg_iters_var_batch)

        crc_mean_list.append(float(torch.sum(crc_all * sample_weights).item()))
        var_mean_list.append(float(torch.sum(var_all * sample_weights).item()))
        cg_iterations_crc.append(float(it_crc_all.mean().item()))
        cg_iterations_var.append(float(it_var_all.mean().item()))

        beta_elapsed = time.time() - beta_start
        factor_elapsed = time.time() - time_start
        beta_eta = factor_elapsed / beta_idx * max(0, len(beta_values) - beta_idx)
        print(
            f"[factor {factor_index}/{total_factors}] [{factor_label}] "
            f"beta {beta_idx}/{len(beta_values)} = {beta:.3e} finished  "
            f"CRC={crc_mean_list[-1]:.6e}  "
            f"VAR={var_mean_list[-1]:.6e}  "
            f"PCG iters CRC/VAR={cg_iterations_crc[-1]:.1f}/{cg_iterations_var[-1]:.1f}  "
            f"beta_elapsed={format_seconds(beta_elapsed)}  "
            f"factor_elapsed={format_seconds(factor_elapsed)}  "
            f"factor_eta={format_seconds(beta_eta)}",
            flush=True,
        )

    elapsed = time.time() - time_start
    del sysmat
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"[factor {factor_index}/{total_factors}] [{factor_label}] finished in {format_seconds(elapsed)}",
        flush=True,
    )

    return FactorResult(
        factor_label=factor_label,
        factor_dir=str(factor_dir),
        num_voxels=num_voxels,
        num_detectors=num_detectors,
        avg_sensitivity=operator.avg_sensitivity,
        sample_point_count=args.sample_grid_nx * args.sample_grid_ny * args.sample_grid_nz,
        unique_sample_voxel_count=int(sample_indices.numel()),
        crc_mean=crc_mean_list,
        var_mean=var_mean_list,
        cg_iterations_crc=cg_iterations_crc,
        cg_iterations_var=cg_iterations_var,
        elapsed_seconds=elapsed,
    )


def main() -> None:
    run_start = time.time()
    args = parse_args()
    factor_dirs = resolve_factor_dirs(args.factor_dirs)
    if args.factor_labels is not None and len(args.factor_labels) != len(factor_dirs):
        raise ValueError("--factor-labels must have the same length as --factor-dirs.")

    factor_labels = args.factor_labels or [path.name for path in factor_dirs]
    beta_values = beta_schedule(args.beta_level_start, args.beta_level_end, args.beta_multipliers)
    device = choose_device(args.device)
    dtype = torch_dtype_from_name(args.dtype)

    sample_indices, sample_weights, sample_points_mm = build_sample_point_index_weights(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        sx_mm=args.sx_mm,
        sy_mm=args.sy_mm,
        sz_mm=args.sz_mm,
        sample_grid_nx=args.sample_grid_nx,
        sample_grid_ny=args.sample_grid_ny,
        sample_grid_nz=args.sample_grid_nz,
        sample_spacing_mm=args.sample_spacing_mm,
    )

    output_dir = (args.output_root / args.run_name / "SinglePhoton").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Single-Photon CRC-VAR Analysis")
    print("=" * 72)
    print(f"Device                : {device}")
    print(f"Dtype                 : {dtype}")
    print(f"Project root          : {PROJECT_ROOT}")
    print(f"Output dir            : {output_dir}")
    print(f"Factor labels         : {factor_labels}")
    print(f"Beta count            : {len(beta_values)}")
    print(f"Sample point count    : {sample_points_mm.shape[0]}")
    print(f"Unique voxel samples  : {sample_indices.numel()}")
    print(f"Point batch size      : {args.point_batch_size}")
    print(f"Spacing-weighted R    : {not args.disable_spacing_weight}")
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
                sample_indices_cpu=sample_indices,
                sample_weights_cpu=sample_weights,
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
            "sample_grid_nx": args.sample_grid_nx,
            "sample_grid_ny": args.sample_grid_ny,
            "sample_grid_nz": args.sample_grid_nz,
            "sample_spacing_mm": args.sample_spacing_mm,
            "sample_point_count": int(sample_points_mm.shape[0]),
            "unique_sample_voxel_count": int(sample_indices.numel()),
            "beta_values": beta_values,
            "point_batch_size": args.point_batch_size,
            "cg_tol": args.cg_tol,
            "cg_maxiter": args.cg_maxiter,
            "dtype": args.dtype,
            "device": str(device),
            "spacing_weighted_regularizer": not args.disable_spacing_weight,
            "project_root": str(PROJECT_ROOT),
        },
        "results": [asdict(result) for result in results],
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Summary written to: {summary_path}")
    print(f"Total elapsed        : {format_seconds(time.time() - run_start)}")


if __name__ == "__main__":
    main()
