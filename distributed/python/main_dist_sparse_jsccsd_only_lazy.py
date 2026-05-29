"""
Lazy variant of main_dist_sparse_jsccsd_only.py.

Key difference: instead of storing the full t_compton tensor (shape
[num_events, coarse_pixel_num]), we only store compact per-event parameters
(cpnum1, cpnum2, e1_smeared) — 3 floats per event instead of coarse_pixel_num.
During reconstruction, t_compton is recomputed on-the-fly from these compact
parameters, trading compute for a large reduction in GPU memory.
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

try:
    from distributed.python._path_setup import setup_repo_root
except ImportError:
    from _path_setup import setup_repo_root

setup_repo_root()

from compton_sparse_ops import (
    build_compton_sparse_projector,
    materialize_sparse_event_rows_to_fine,
)
from sparse_main_utils import (
    Tee,
    build_save_path,
    downsample_projection_and_list,
    load_list_csv,
    resolve_factor_dir,
    resolve_pixel_num,
    resolve_proj_and_list_paths,
    resolve_repo_root,
)
from process_list_plane_strict import (
    ELECTRON_REST_MEV,
    MIN_EVENT_EFFECTIVE_SUPPORT,
    _build_detector_pos_sigma_sq,
    _compton_theta_from_e1,
    _compute_angle_sigma_ene_strict,
    _compute_angle_sigma_pos_strict,
    _filter_kinematically_valid_events,
)
from compton_sparse_ops import pack_sparse_event_rows, reduce_fine_rows_to_coarse
from distributed.python.recon_osem_dist_sparse_jsccsd_only_lazy import run_recon_osem_dist_sparse_jsccsd_only_lazy


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed sparse JSCCSD-only lazy reconstruction (compact params).")
    parser.add_argument("--e0-list", type=float, nargs="+", default=[0.511], help="Energy list in MeV.")
    parser.add_argument("--ene-threshold-sum-list", type=float, nargs="+", default=[0.46], help="Lower bounds for e1 + e2 in MeV.")
    parser.add_argument("--intensity-list", type=float, nargs="+", default=[1.0], help="Intensity weights.")
    parser.add_argument("--s-map-d-ratio", type=float, default=0.5, help="Scale factor applied to sparse Sensi_d.")
    parser.add_argument("--recompute-sparse-sensi-d", action="store_true", help="Force recomputing Sensi_d from sparse Compton operators.")
    parser.add_argument("--data-file-name", type=str, default="Hoffman_Big", help="Dataset name.")
    parser.add_argument("--count-level", type=str, default="1e11", help="Count level suffix.")
    parser.add_argument("--ds", type=float, default=1.0, help="Downsampling ratio in (0, 1].")
    parser.add_argument("--ene-resolution-662keV", type=float, default=0.1, help="Reference energy resolution.")
    parser.add_argument("--pixel-num-layer", type=int, default=2700, help="Legacy fallback layer pixel count.")
    parser.add_argument("--pixel-num-z", type=int, default=20, help="Axial slice count.")
    parser.add_argument("--rotate-num", type=int, default=60, help="Number of views.")
    parser.add_argument("--delta-r1", type=float, default=0.0, help="Additional isotropic position sigma for the first interaction in mm.")
    parser.add_argument("--delta-r2", type=float, default=0.0, help="Additional isotropic position sigma for the second interaction in mm.")
    parser.add_argument("--alpha", type=float, default=1.0, help="JSCC weighting parameter.")
    parser.add_argument("--jsccsd-iter", type=int, default=2000, help="JSCC-SD iteration count.")
    parser.add_argument("--save-iter-step", type=int, default=20, help="Save interval.")
    parser.add_argument("--osem-subset-num", type=int, default=16, help="OSEM subset count.")
    parser.add_argument("--t-divide-num", type=int, default=1, help="Number of t sub-blocks per subset.")
    parser.add_argument("--num-workers", type=int, default=20, help="Sub-chunks per rank during sparse list processing.")
    parser.add_argument("--compton-theta-stride", type=int, default=1, help="Angular stride used by sparse Compton grid.")
    parser.add_argument("--compton-z-stride", type=int, default=1, help="Axial stride used by sparse Compton grid.")
    parser.add_argument("--seed", type=int, default=20260332, help="Random seed.")
    parser.add_argument("--factors-dir", type=str, default="./Factors", help="Factors root directory.")
    parser.add_argument("--cntstat-dir", type=str, default="./CntStat", help="CntStat root directory.")
    parser.add_argument("--list-dir", type=str, default="./List", help="List root directory.")
    parser.add_argument("--output-root", type=str, default="./Figure_Dist_JSCCSD_Lazy", help="Output root directory.")
    return parser.parse_args()


def validate_args_jsccsd_only(args):
    if len(args.e0_list) != len(args.ene_threshold_sum_list):
        raise ValueError("--e0-list and --ene-threshold-sum-list must have the same length.")
    if len(args.e0_list) != len(args.intensity_list):
        raise ValueError("--e0-list and --intensity-list must have the same length.")
    if not (0.0 < args.ds <= 1.0):
        raise ValueError("--ds must be in (0, 1].")
    if min(args.jsccsd_iter, args.save_iter_step) <= 0:
        raise ValueError("--jsccsd-iter and --save-iter-step must be positive.")
    if args.jsccsd_iter % args.save_iter_step != 0:
        raise ValueError("--jsccsd-iter must be divisible by --save-iter-step.")
    if args.osem_subset_num <= 0 or args.t_divide_num <= 0 or args.num_workers <= 0:
        raise ValueError("--osem-subset-num, --t-divide-num and --num-workers must be positive.")
    if args.rotate_num <= 0 or args.pixel_num_z <= 0 or args.pixel_num_layer <= 0:
        raise ValueError("--rotate-num, --pixel-num-layer and --pixel-num-z must be positive.")


def setup_distributed():
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return global_rank, local_rank, world_size


def build_sparse_sensi_d_local(compact_local_all, sysmat_full_all, sparse_projector_all,
                                detector_all, delta_r1, delta_r2, e0_list, ene_resolution_list,
                                pixel_num, block_size=1024):
    """Build Sensi_d by recomputing t_compton from compact params block by block."""
    sensi_d_local = torch.zeros((pixel_num, 1), dtype=torch.float32)
    for energy_idx, (compact_l, sysmat_full, sparse_projector, detector, e0, ene_resolution) in enumerate(
        zip(compact_local_all, sysmat_full_all, sparse_projector_all, detector_all, e0_list, ene_resolution_list)
    ):
        projector_gpu = sparse_projector.to(sysmat_full.device)
        detector_gpu = detector.to(sysmat_full.device)
        for compact_rotate in compact_l:
            if compact_rotate.numel() == 0:
                continue
            for row_start in range(0, compact_rotate.size(0), block_size):
                compact_block = compact_rotate[row_start:row_start + block_size].to(sysmat_full.device, non_blocking=True)
                cpnum1 = compact_block[:, 0].long()
                cpnum2 = compact_block[:, 1].long()
                e1_smeared = compact_block[:, 2]

                # Recompute t_compton
                from distributed.python.recon_osem_dist_sparse_jsccsd_only_lazy import _recompute_t_compton_from_compact
                t_compton = _recompute_t_compton_from_compact(
                    cpnum1, cpnum2, e1_smeared,
                    detector_gpu, projector_gpu,
                    delta_r1, delta_r2, e0, ene_resolution,
                )
                event_rows = pack_sparse_event_rows(cpnum1, t_compton)
                t_fine, _ = materialize_sparse_event_rows_to_fine(event_rows, sysmat_full, projector_gpu)
                if t_fine.numel() > 0:
                    sensi_d_local = sensi_d_local + torch.sum(t_fine, dim=0, keepdim=True).transpose(0, 1).cpu()
    return sensi_d_local


def log_tensor_stats(name, tensor):
    tensor_cpu = tensor.detach().float().cpu()
    finite = torch.isfinite(tensor_cpu)
    finite_count = int(finite.sum().item())
    total_count = tensor_cpu.numel()
    zero_count = int((tensor_cpu == 0).sum().item())
    if finite_count > 0:
        finite_vals = tensor_cpu[finite]
        print(
            f"[{name}] finite={finite_count}/{total_count} zero={zero_count} "
            f"min={finite_vals.min().item():.6e} max={finite_vals.max().item():.6e} "
            f"mean={finite_vals.mean().item():.6e}"
        )
    else:
        print(f"[{name}] finite=0/{total_count} zero={zero_count}")


def get_compact_params_single_sparse(
    sysmat, detector, sparse_projector, list_origin,
    delta_r1, delta_r2, e0, ene_resolution,
    ene_threshold_max, ene_threshold_min, ene_threshold_sum, device,
):
    """Run Compton back-projection filtering, but return compact params instead of event_rows.

    Returns
    -------
    compact : Tensor [num_valid_events, 3] on CPU
        Columns: (cpnum1_as_float, cpnum2_as_float, e1_smeared)
    """
    cpnum1 = list_origin[:, 0].int()
    cpnum2 = list_origin[:, 2].int()
    e1 = list_origin[:, 1]
    e2 = list_origin[:, 3]

    sigma_1 = e1 * ene_resolution / 2.355 * (e0 / e1) ** 0.5
    sigma_2 = e2 * ene_resolution / 2.355 * (e0 / e2) ** 0.5
    e1 = e1 + sigma_1 * torch.randn(e1.shape[0], device=device)
    e2 = e2 + sigma_2 * torch.randn(e2.shape[0], device=device)

    flag = (e1 < ene_threshold_max) & (e1 > ene_threshold_min) & (e2 > ene_threshold_min) & ((e1 + e2) > ene_threshold_sum)
    cpnum1 = cpnum1[flag]
    cpnum2 = cpnum2[flag]
    e1 = e1[flag]
    e2 = e2[flag]
    cpnum1, cpnum2, e1, e2 = _filter_kinematically_valid_events(cpnum1, cpnum2, e1, e2, e0, ELECTRON_REST_MEV)

    detector_pos = detector[:, :3]
    detector_sigma_r1_sq = _build_detector_pos_sigma_sq(detector, delta_r1)
    detector_sigma_r2_sq = _build_detector_pos_sigma_sq(detector, delta_r2)

    pos1 = detector_pos[cpnum1 - 1, :]
    pos2 = detector_pos[cpnum2 - 1, :]
    sigma_pos1_sq = detector_sigma_r1_sq[cpnum1 - 1, :]
    sigma_pos2_sq = detector_sigma_r2_sq[cpnum2 - 1, :]
    flag = torch.abs(pos1[:, 1] - pos2[:, 1]) > 0.1

    cpnum1 = cpnum1[flag]
    cpnum2 = cpnum2[flag]
    e1 = e1[flag]
    pos1 = pos1[flag]
    pos2 = pos2[flag]
    sigma_pos1_sq = sigma_pos1_sq[flag]
    sigma_pos2_sq = sigma_pos2_sq[flag]

    if cpnum1.numel() == 0:
        return torch.empty((0, 3), dtype=torch.float32)

    # --- Compute full t_compton for stability filtering (same as original) ---
    coor_coarse = sparse_projector.coor_coarse
    vector01 = pos1.unsqueeze(1) - coor_coarse.unsqueeze(0)
    vector12 = (pos2 - pos1).unsqueeze(1)
    distance01 = torch.norm(vector01, dim=2)
    distance12 = torch.norm(vector12, dim=2)

    theta = _compton_theta_from_e1(e1, e0, ELECTRON_REST_MEV)
    klein_nishina = e0 / (e0 - e1) + (e0 - e1) / e0
    beta_cos = (vector01 * vector12).sum(2) / torch.clamp(distance01 * distance12, min=1e-7)
    beta = torch.acos(torch.clamp(beta_cos, -1.0 + 1e-7, 1.0 - 1e-7))

    angle_sigma_ene = _compute_angle_sigma_ene_strict(e1, e0, ene_resolution, ELECTRON_REST_MEV, beta, theta)
    angle_sigma_pos = _compute_angle_sigma_pos_strict(
        vector01, vector12, beta, sigma_pos1_sq, sigma_pos2_sq,
    )
    angle_sigma = torch.sqrt(torch.clamp(angle_sigma_pos ** 2 + angle_sigma_ene ** 2, min=1e-12))

    t_compton = torch.exp(-((beta - theta.unsqueeze(-1)) ** 2) / (2 * angle_sigma ** 2))
    t_compton = t_compton * (klein_nishina.unsqueeze(-1) - torch.sin(beta) ** 2)

    t_single = reduce_fine_rows_to_coarse(sysmat[cpnum1 - 1, :], sparse_projector)
    t = t_compton * t_single

    flag_nan = torch.isnan(t).sum(dim=1)
    flag_zero = t.sum(dim=1) == 0
    valid = (flag_nan + flag_zero) == 0
    cpnum1 = cpnum1[valid]
    e1 = e1[valid]
    t = t[valid, :]
    t_compton_f = t_compton[valid, :]
    t_single_f = t_single[valid, :]

    if t.size(0) == 0:
        return torch.empty((0, 3), dtype=torch.float32)

    # Stability filter
    t_norm = t / t.sum(dim=1, keepdim=True)
    t_compton_norm = t_compton_f / t_compton_f.sum(dim=1, keepdim=True)
    t_single_norm = t_single_f / t_single_f.sum(dim=1, keepdim=True)
    effective_support = 1.0 / torch.sum(t_norm ** 2, dim=1)
    stable = effective_support >= MIN_EVENT_EFFECTIVE_SUPPORT

    cpnum1_stable = cpnum1[stable]
    e1_stable = e1[stable]

    if cpnum1_stable.numel() == 0:
        return torch.empty((0, 3), dtype=torch.float32)

    # Return compact params: [cpnum1, cpnum2, e1_smeared]
    # Need cpnum2 as well — recover from the original indices
    cpnum2_stable = cpnum2[valid][stable]

    compact = torch.stack([
        cpnum1_stable.float(),
        cpnum2_stable.float(),
        e1_stable,
    ], dim=1)

    return compact.cpu()


def main():
    args = parse_args()
    validate_args_jsccsd_only(args)

    global_rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    repo_root = resolve_repo_root()
    factors_root = (repo_root / args.factors_dir).resolve()
    cntstat_root = (repo_root / args.cntstat_dir).resolve()
    list_root = (repo_root / args.list_dir).resolve()
    output_root = (repo_root / args.output_root).resolve()

    random.seed(args.seed + global_rank)
    np.random.seed(args.seed + global_rank)
    torch.manual_seed(args.seed + global_rank)

    logfile = None
    log_filename = None
    save_path = None
    original_stdout = sys.stdout

    try:
        if global_rank == 0:
            rand_suffix = f"{random.randint(0, 9999):04d}"
            log_filename = repo_root / f"print_log_dist_sparse_jsccsd_only_lazy_{rand_suffix}.txt"
            logfile = open(log_filename, "w", encoding="utf-8")
            sys.stdout = Tee(sys.__stdout__, logfile)
            print(f"Distributed sparse JSCCSD-only LAZY chain initialized: World Size {world_size}")
            print(f"Args: {args}")

        proj_local_all = []
        sysmat_local_all = []
        sysmat_full_all = []
        rotmat_all = []
        rotmat_inv_all = []
        sensi_s_all = []
        sensi_d_all = []
        sparse_projector_all = []
        detector_all = []
        e0_list = []
        ene_resolution_list = []

        pixel_num = None
        single_event_count_total_local = 0
        compton_event_count_total_local = 0
        compact_local_all = []  # [energy][rotate] -> CPU Tensor [N, 3]

        for e0, ene_threshold_sum, intensity in zip(args.e0_list, args.ene_threshold_sum_list, args.intensity_list):
            ene_resolution = args.ene_resolution_662keV * (0.662 / e0) ** 0.5
            ene_threshold_max = 2 * e0 ** 2 / (0.511 + 2 * e0) - 0.001
            ene_threshold_min = 0.05
            e0_list.append(e0)
            ene_resolution_list.append(ene_resolution)

            factor_dir = resolve_factor_dir(factors_root, e0, args.rotate_num)
            proj_file_path, list_dir_path = resolve_proj_and_list_paths(
                cntstat_root, list_root, e0, args.rotate_num,
                args.data_file_name, args.count_level,
            )

            sysmat_file_path = factor_dir / "SysMat_polar"
            detector_file_path = factor_dir / "Detector.csv"
            sensi_s_file_path = factor_dir / "Sensi_s"
            sensi_d_file_path = factor_dir / "Sensi_d"
            coor_polar_file_path = factor_dir / "coor_polar_full.csv"
            rotmat_file_path = factor_dir / "RotMat_full.csv"
            rotmat_inv_file_path = factor_dir / "RotMatInv_full.csv"

            detector = torch.from_numpy(np.genfromtxt(detector_file_path, delimiter=",", dtype=np.float32)[:, 1:4]).to(device)
            detector_all.append(detector)
            coor_polar = torch.from_numpy(np.genfromtxt(coor_polar_file_path, delimiter=",", dtype=np.float32))
            rotmat = torch.from_numpy(np.genfromtxt(rotmat_file_path, delimiter=",", dtype=np.int64))
            rotmat_inv = torch.from_numpy(np.genfromtxt(rotmat_inv_file_path, delimiter=",", dtype=np.int64))

            pixel_num_current = resolve_pixel_num(args.pixel_num_layer * args.pixel_num_z, args.pixel_num_z, rotmat, rotmat_inv, coor_polar, factor_dir)
            if pixel_num is None:
                pixel_num = pixel_num_current
            elif pixel_num != pixel_num_current:
                raise ValueError(f"Inconsistent pixel_num across energies: previous={pixel_num}, current={pixel_num_current}")

            full_sysmat = torch.from_numpy(np.fromfile(sysmat_file_path, dtype=np.float32).reshape(pixel_num, -1).T.copy()) * intensity
            total_bins = full_sysmat.size(0)
            bins_per_rank = total_bins // world_size
            idx_start = global_rank * bins_per_rank
            idx_end = (global_rank + 1) * bins_per_rank if global_rank != world_size - 1 else total_bins
            sysmat_local = full_sysmat[idx_start:idx_end, :].clone()
            sysmat_local_all.append(sysmat_local)
            sysmat_full_gpu = full_sysmat.to(device)
            sysmat_full_all.append(sysmat_full_gpu)

            if sensi_s_file_path.exists():
                sensi_s = torch.from_numpy(np.fromfile(sensi_s_file_path, dtype=np.float32).reshape(pixel_num, 1).copy()) * intensity
                sensi_s_all.append(sensi_s)
            else:
                sensi_s_tmp = torch.zeros([1, pixel_num], dtype=torch.float32)
                for rotate_idx in range(args.rotate_num):
                    rotmat_inv_tmp = rotmat_inv[:, rotate_idx]
                    sensi_s_tmp += torch.sum(full_sysmat[:, rotmat_inv_tmp - 1], dim=0, keepdim=True).cpu()
                sensi_s_all.append(sensi_s_tmp.transpose(0, 1) / args.rotate_num)

            if sensi_d_file_path.exists():
                sensi_d = torch.from_numpy(np.fromfile(sensi_d_file_path, dtype=np.float32).reshape(pixel_num, 1).copy()) * intensity
                sensi_d_all.append(sensi_d)
                if global_rank == 0:
                    print(f"Loaded factor Sensi_d: {sensi_d_file_path}")
            elif global_rank == 0:
                print(f"Factor Sensi_d not found, will recompute if needed: {sensi_d_file_path}")

            sparse_projector = build_compton_sparse_projector(
                coor_polar,
                theta_stride=args.compton_theta_stride,
                z_stride=args.compton_z_stride,
                rotate_num=args.rotate_num,
                dtype=torch.float32,
            )
            sparse_projector_all.append(sparse_projector)

            full_proj = torch.from_numpy(np.genfromtxt(proj_file_path, delimiter=",", dtype=np.float32).reshape(args.rotate_num, -1).T.copy())
            proj_local = full_proj[idx_start:idx_end, :].clone()
            del full_proj
            list_rotate_local = []
            for rotate_idx in range(args.rotate_num):
                full_list = load_list_csv(list_dir_path / f"{rotate_idx + 1}.csv")
                ev_per_rank = full_list.size(0) // world_size
                ev_start = global_rank * ev_per_rank
                ev_end = (global_rank + 1) * ev_per_rank if global_rank != world_size - 1 else full_list.size(0)
                list_rotate_local.append(full_list[ev_start:ev_end, :])

            proj_local, list_rotate_local = downsample_projection_and_list(proj_local, list_rotate_local, args.ds * intensity)
            proj_local_all.append(proj_local)
            single_event_count_total_local += round(proj_local.sum().item())

            # --- Compute compact params instead of full event_rows ---
            compact_rotate_local = []
            for rotate_idx in range(args.rotate_num):
                list_local_chunks = torch.chunk(list_rotate_local[rotate_idx], args.num_workers, dim=0)
                compact_parts = []
                for chunk in list_local_chunks:
                    if chunk.numel() == 0:
                        continue
                    compact_chunk = get_compact_params_single_sparse(
                        sysmat_full_gpu,
                        detector,
                        sparse_projector.to(device),
                        chunk.to(device),
                        args.delta_r1,
                        args.delta_r2,
                        e0,
                        ene_resolution,
                        ene_threshold_max,
                        ene_threshold_min,
                        ene_threshold_sum,
                        device,
                    )
                    if compact_chunk.numel() > 0:
                        compact_parts.append(compact_chunk)
                        compton_event_count_total_local += compact_chunk.size(0)
                compact_rotate = (
                    torch.cat(compact_parts, dim=0)
                    if compact_parts
                    else torch.empty((0, 3), dtype=torch.float32)
                )
                compact_rotate_local.append(compact_rotate)
            compact_local_all.append(compact_rotate_local)

            rotmat_all.append(rotmat.to(device))
            rotmat_inv_all.append(rotmat_inv.to(device))

            if global_rank == 0:
                print(
                    f"Loaded sparse projector for {e0:.3f} MeV: fine_pixels={pixel_num}, "
                    f"sparse_pixels={sparse_projector.coarse_pixel_num}, ring_strides={sparse_projector.ring_strides}"
                )

            del full_sysmat
            torch.cuda.empty_cache()

        single_event_count_tensor = torch.tensor([single_event_count_total_local], dtype=torch.float64, device=device)
        compton_event_count_tensor = torch.tensor([compton_event_count_total_local], dtype=torch.float64, device=device)
        dist.all_reduce(single_event_count_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(compton_event_count_tensor, op=dist.ReduceOp.SUM)
        single_event_count_total = int(single_event_count_tensor.item())
        compton_event_count_total = int(compton_event_count_tensor.item())

        s_map_arg = argparse.Namespace()
        s_map_arg.s = (sum(sensi_s_all)).to(device)
        if sensi_d_all and not args.recompute_sparse_sensi_d:
            if global_rank == 0:
                print("Distributed sparse JSCCSD-only LAZY chain uses file-defined Sensi_d.")
            s_map_arg.d = (sum(sensi_d_all) * args.s_map_d_ratio).to(device)
        else:
            if global_rank == 0:
                if sensi_d_all:
                    print("Distributed sparse JSCCSD-only LAZY chain recomputes Sensi_d from compact params.")
                else:
                    print("Distributed sparse JSCCSD-only LAZY chain has no file-defined Sensi_d and recomputes it.")
            sensi_d_local = build_sparse_sensi_d_local(
                compact_local_all, sysmat_full_all, sparse_projector_all,
                detector_all, args.delta_r1, args.delta_r2,
                e0_list, ene_resolution_list, pixel_num,
            )
            sensi_d_tensor = sensi_d_local.to(device)
            dist.all_reduce(sensi_d_tensor, op=dist.ReduceOp.SUM)
            if torch.sum(sensi_d_tensor) > 0:
                sensi_d_tensor = sensi_d_tensor * torch.sum(s_map_arg.s) / torch.sum(sensi_d_tensor)
                sensi_d_tensor = sensi_d_tensor * compton_event_count_total / max(single_event_count_total, 1)
            s_map_arg.d = sensi_d_tensor * args.s_map_d_ratio
        s_map_arg.j = args.alpha * s_map_arg.s + (2 - args.alpha) * s_map_arg.d

        if global_rank == 0:
            log_tensor_stats("s_map_arg.s", s_map_arg.s)
            log_tensor_stats("s_map_arg.d", s_map_arg.d)
            log_tensor_stats("s_map_arg.j", s_map_arg.j)

        iter_arg = argparse.Namespace()
        iter_arg.jsccsd = args.jsccsd_iter
        iter_arg.save_iter_step = args.save_iter_step
        iter_arg.osem_subset_num = args.osem_subset_num
        iter_arg.t_divide_num = args.t_divide_num
        iter_arg.ene_num = len(args.e0_list)
        iter_arg.num_workers = args.num_workers
        iter_arg.seed = args.seed

        base_path = build_save_path(
            output_root,
            args.e0_list,
            args.rotate_num,
            args.data_file_name,
            args.count_level,
            args.ds,
            args.s_map_d_ratio,
            args.delta_r1,
            args.alpha,
            args.ene_resolution_662keV,
            iter_arg.osem_subset_num,
            iter_arg.jsccsd,
            single_event_count_total,
            compton_event_count_total,
        )
        sparse_prefix = f"New_{args.compton_theta_stride}_{args.compton_z_stride}_"
        save_path = base_path.parent.parent / f"{sparse_prefix}{base_path.parent.name}" / "Polar"
        if global_rank == 0:
            save_path.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        run_recon_osem_dist_sparse_jsccsd_only_lazy(
            sysmat_local_all,
            sysmat_full_all,
            rotmat_all,
            rotmat_inv_all,
            proj_local_all,
            compact_local_all,
            sparse_projector_all,
            detector_all,
            args.delta_r1,
            args.delta_r2,
            e0_list,
            ene_resolution_list,
            iter_arg,
            s_map_arg,
            args.alpha,
            str(save_path) + os.sep,
        )

    finally:
        if global_rank == 0:
            sys.stdout = original_stdout
            if logfile is not None:
                logfile.close()
            if log_filename is not None:
                if save_path is not None and Path(save_path).is_dir():
                    shutil.move(str(log_filename), Path(save_path) / "print_log.txt")
                elif Path(log_filename).exists():
                    print(f"Log kept at {log_filename}")

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    with torch.no_grad():
        main()