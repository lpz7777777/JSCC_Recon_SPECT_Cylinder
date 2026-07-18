"""Distributed multi-energy multi-output reconstruction entry point.

Loads all energies' single-photon data once, plus Compton data only for energies
listed in ``--compton-energies``. Then runs one task per requested output type
(per-energy SC, per-energy Compton, all-energies SC, all-energies Compton,
all-energies joint), each with its own iteration count.

See ``multi_energy_tasks.py`` for the task model and ``recon_osem_dist_multi_energy.py``
for the reconstruction core.
"""

import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

try:
    from distributed.python._path_setup import setup_repo_root
except ImportError:
    from _path_setup import setup_repo_root

from compton_sparse_ops import (
    build_compton_sparse_projector,
    materialize_sparse_event_rows_to_fine,
)
from process_list_plane_sparse import get_compton_backproj_list_single_sparse
from sparse_main_utils import (
    Tee,
    downsample_projection_and_list,
    format_scientific_count,
    load_list_csv,
    resolve_factor_dir,
    resolve_pixel_num,
    resolve_proj_and_list_paths,
    resolve_repo_root,
)

from multi_energy_tasks import IterConfig, build_tasks, format_task_table
from recon_osem_dist_multi_energy import run_tasks_multi_energy


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed multi-energy multi-output reconstruction."
    )

    # --- energy / intensity ---
    parser.add_argument("--e0-list", type=float, nargs="+", default=[0.511],
                        help="All energies in MeV.")
    parser.add_argument("--ene-threshold-sum-list", type=float, nargs="+", default=[0.46],
                        help="Lower bounds for e1+e2 in MeV, one per energy.")
    parser.add_argument("--intensity-list", type=float, nargs="+", default=[1.0],
                        help="Intensity weights, one per energy.")
    parser.add_argument("--compton-energies", type=float, nargs="+", default=None,
                        help="Energies (MeV) whose Compton (List) data is loaded. "
                             "Default: all of --e0-list. Each must appear in --e0-list.")

    # --- per-type iteration (<=0 disables that type) ---
    parser.add_argument("--single-sc-iter", type=int, default=0,
                        help="Type1 per-energy single-photon iterations (0=skip).")
    parser.add_argument("--single-sc-save-step", type=int, default=50)
    parser.add_argument("--single-compton-iter", type=int, default=0,
                        help="Type2 per-energy Compton iterations (0=skip).")
    parser.add_argument("--single-compton-save-step", type=int, default=50)
    parser.add_argument("--joint-sc-iter", type=int, default=0,
                        help="Type3 all-energies single-photon iterations (0=skip).")
    parser.add_argument("--joint-sc-save-step", type=int, default=50)
    parser.add_argument("--joint-compton-iter", type=int, default=0,
                        help="Type4 all-energies Compton iterations (0=skip).")
    parser.add_argument("--joint-compton-save-step", type=int, default=50)
    parser.add_argument("--joint-iter", type=int, default=0,
                        help="Type5 all-energies joint iterations (0=skip).")
    parser.add_argument("--joint-save-step", type=int, default=50)

    # --- geometry / data ---
    parser.add_argument("--data-file-name", type=str, default="ContrastPhantom_240_30")
    parser.add_argument("--count-level", type=str, default="1e9")
    parser.add_argument("--ds", type=float, default=1.0, help="Downsampling ratio in (0,1].")
    parser.add_argument("--ene-resolution-662keV", type=float, default=0.1)
    parser.add_argument("--pixel-num-layer", type=int, default=1280)
    parser.add_argument("--pixel-num-z", type=int, default=20)
    parser.add_argument("--rotate-num", type=int, default=20)
    parser.add_argument("--delta-r1", type=float, default=0.0)
    parser.add_argument("--delta-r2", type=float, default=0.0)

    # --- OSEM / sparse ---
    parser.add_argument("--alpha", type=float, default=1.0, help="Joint weighting (S: alpha, D: 2-alpha).")
    parser.add_argument("--osem-subset-num", type=int, default=1)
    parser.add_argument("--t-divide-num", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--compton-theta-stride", type=int, default=1)
    parser.add_argument("--compton-z-stride", type=int, default=2)
    parser.add_argument("--s-map-d-ratio", type=float, default=1.0)
    parser.add_argument("--recompute-sparse-sensi-d", action="store_true")
    parser.add_argument("--seed", type=int, default=20260331)

    # --- paths ---
    parser.add_argument("--factors-dir", type=str, default="./Factors")
    parser.add_argument("--cntstat-dir", type=str, default="./CntStat")
    parser.add_argument("--list-dir", type=str, default="./List")
    parser.add_argument("--output-root", type=str, default="./Results/Reconstruction/Figure_Dist_MultiEnergy")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def validate_args(args):
    if not (len(args.e0_list) == len(args.ene_threshold_sum_list) == len(args.intensity_list)):
        raise ValueError("--e0-list, --ene-threshold-sum-list, --intensity-list must have equal length.")
    if not (0.0 < args.ds <= 1.0):
        raise ValueError("--ds must be in (0, 1].")
    if args.osem_subset_num <= 0 or args.t_divide_num <= 0 or args.num_workers <= 0:
        raise ValueError("--osem-subset-num, --t-divide-num, --num-workers must be positive.")
    if args.rotate_num <= 0 or args.pixel_num_z <= 0 or args.pixel_num_layer <= 0:
        raise ValueError("--rotate-num, --pixel-num-layer, --pixel-num-z must be positive.")

    # compton whitelist must be a subset of e0_list
    compton = args.compton_energies if args.compton_energies is not None else list(args.e0_list)
    for e in compton:
        if e not in args.e0_list:
            raise ValueError(f"--compton-energies entry {e} not found in --e0-list {args.e0_list}.")

    # per-type iter/step divisibility (only for enabled types)
    for it, sv, name in [
        (args.single_sc_iter, args.single_sc_save_step, "single-sc"),
        (args.single_compton_iter, args.single_compton_save_step, "single-compton"),
        (args.joint_sc_iter, args.joint_sc_save_step, "joint-sc"),
        (args.joint_compton_iter, args.joint_compton_save_step, "joint-compton"),
        (args.joint_iter, args.joint_save_step, "joint"),
    ]:
        if it > 0:
            if sv <= 0:
                raise ValueError(f"--{name}-save-step must be positive when iterations are enabled.")
            if it % sv != 0:
                raise ValueError(f"--{name}-iter ({it}) must be divisible by --{name}-save-step ({sv}).")

    # at least one type enabled
    any_enabled = any([
        args.single_sc_iter > 0,
        args.single_compton_iter > 0,
        args.joint_sc_iter > 0,
        args.joint_compton_iter > 0,
        args.joint_iter > 0,
    ])
    if not any_enabled:
        raise ValueError("At least one output type must be enabled (iter > 0).")


# --------------------------------------------------------------------------- #
# Distributed setup                                                            #
# --------------------------------------------------------------------------- #

def setup_distributed():
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return global_rank, local_rank, world_size


# --------------------------------------------------------------------------- #
# Sensi_d recomputation (only for whitelisted energies, if needed)             #
# --------------------------------------------------------------------------- #

def build_sparse_sensi_d_local(t_local_all, sysmat_full_all, sparse_projector_all,
                               compton_eidx_list, pixel_num, device, block_size=1024):
    """Per-energy Compton sensitivity by materializing T rows.

    Returns a dict {e_idx: [pixel_num, 1]} (CPU float32, local partial sum per rank).
    """
    sensi_d_local = {e_idx: torch.zeros((pixel_num, 1), dtype=torch.float32) for e_idx in compton_eidx_list}
    for c_idx, e_idx in enumerate(compton_eidx_list):
        sysmat_full = sysmat_full_all[e_idx]
        projector = sparse_projector_all[e_idx].to(sysmat_full.device)
        for t_rotate in t_local_all[e_idx]:
            if t_rotate.numel() == 0:
                continue
            for row_start in range(0, t_rotate.size(0), block_size):
                event_block = t_rotate[row_start:row_start + block_size].to(sysmat_full.device, non_blocking=True)
                t_fine, _ = materialize_sparse_event_rows_to_fine(event_block, sysmat_full, projector)
                if t_fine.numel() > 0:
                    sensi_d_local[e_idx] = sensi_d_local[e_idx] + torch.sum(t_fine, dim=0, keepdim=True).transpose(0, 1).cpu()
    return sensi_d_local


def log_tensor_stats(name, tensor, global_rank):
    if global_rank != 0:
        return
    t = tensor.detach().float().cpu()
    finite = torch.isfinite(t)
    if int(finite.sum().item()) > 0:
        print(f"[{name}] min={t[finite].min().item():.6e} max={t[finite].max().item():.6e} "
              f"mean={t[finite].mean().item():.6e} sum={t.sum().item():.6e}")
    else:
        print(f"[{name}] all-zero/non-finite")


# --------------------------------------------------------------------------- #
# Save-path construction                                                       #
# --------------------------------------------------------------------------- #

def build_save_root(output_root, e0_list, args, single_count, compton_count, sparse_theta, sparse_z):
    e0_str = "_".join(str(round(e * 1000)) for e in e0_list)
    if len(e0_list) == 1:
        prefix = f"SE_RotNum{args.rotate_num}_{args.data_file_name}_{e0_str}keV"
    else:
        prefix = f"ME_RotNum{args.rotate_num}_{args.data_file_name}_({e0_str})keV"

    single_str = format_scientific_count(single_count)
    compton_str = format_scientific_count(compton_count)
    sparse_prefix = f"{sparse_theta}_{sparse_z}_"

    name = (
        f"{prefix}_{args.count_level}_{args.ds}_SMap{args.s_map_d_ratio}_"
        f"Delta{args.delta_r1}_Alpha{args.alpha}_ER{args.ene_resolution_662keV}_"
        f"OSEM{args.osem_subset_num}_SDU{single_str}_DDU{compton_str}"
    )
    return output_root / f"{sparse_prefix}{name}" / "Polar"


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    validate_args(args)

    repo_root = resolve_repo_root()
    global_rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    start_time = time.time()

    random.seed(args.seed + global_rank)
    np.random.seed(args.seed + global_rank)
    torch.manual_seed(args.seed + global_rank)

    factors_root = (repo_root / args.factors_dir).resolve()
    cntstat_root = (repo_root / args.cntstat_dir).resolve()
    list_root = (repo_root / args.list_dir).resolve()
    output_root = (repo_root / args.output_root).resolve()

    # resolve compton whitelist -> indices into e0_list
    compton_energies = list(args.compton_energies) if args.compton_energies is not None else list(args.e0_list)
    compton_eidx_list = [args.e0_list.index(e) for e in compton_energies]

    log_filename = None
    logfile = None
    save_root = None

    try:
        if global_rank == 0:
            rand_suffix = f"{random.randint(0, 9999):04d}"
            log_filename = repo_root / f"print_log_dist_multi_energy_{rand_suffix}.txt"
            logfile = open(log_filename, "w", encoding="utf-8")
            sys.stdout = Tee(sys.__stdout__, logfile)
            print(f"Distributed initialized: world_size={world_size}, device={device}")
            print(f"Args: {args}")
            print(f"All energies (keV): {[round(e * 1000) for e in args.e0_list]}")
            print(f"Compton-whitelisted energies (keV): {[round(args.e0_list[i] * 1000) for i in compton_eidx_list]}")

        # ---- single-photon containers (one entry per energy) ----
        sysmat_local_all = []      # list[e_idx]
        proj_local_all = []        # list[e_idx]
        rotmat_all = []            # list[e_idx]
        rotmat_inv_all = []        # list[e_idx]
        sensi_s_all = []           # list[e_idx]

        # ---- compton containers (only for whitelisted energies) ----
        sysmat_full_all = {}       # dict {e_idx: GPU tensor}
        sparse_projector_all = {}  # dict {e_idx: projector}
        sensi_d_all = {}           # dict {e_idx: [pixel_num,1]}
        t_local_all = {}           # dict {e_idx: list[rotate] CPU tensor}

        pixel_num = None
        single_event_count_total_local = 0
        compton_event_count_total_local = 0

        for e_idx, (e0, ene_threshold_sum, intensity) in enumerate(
            zip(args.e0_list, args.ene_threshold_sum_list, args.intensity_list)
        ):
            is_compton = e_idx in compton_eidx_list
            ene_resolution = args.ene_resolution_662keV * (0.662 / e0) ** 0.5
            ene_threshold_max = 2 * e0 ** 2 / (0.511 + 2 * e0) - 0.001
            ene_threshold_min = 0.05

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

            rotmat = torch.from_numpy(np.genfromtxt(rotmat_file_path, delimiter=",", dtype=np.int64))
            rotmat_inv = torch.from_numpy(np.genfromtxt(rotmat_inv_file_path, delimiter=",", dtype=np.int64))
            coor_polar = torch.from_numpy(np.genfromtxt(coor_polar_file_path, delimiter=",", dtype=np.float32))

            pixel_num_current = resolve_pixel_num(
                args.pixel_num_layer * args.pixel_num_z, args.pixel_num_z,
                rotmat, rotmat_inv, coor_polar, factor_dir)
            if pixel_num is None:
                pixel_num = pixel_num_current
            elif pixel_num != pixel_num_current:
                raise ValueError(f"Inconsistent pixel_num across energies: {pixel_num} vs {pixel_num_current}")

            # ---- system matrix: load full, then bin-shard for single-photon ----
            full_sysmat = torch.from_numpy(
                np.fromfile(sysmat_file_path, dtype=np.float32).reshape(pixel_num, -1).T.copy()
            ) * intensity
            total_bins = full_sysmat.size(0)
            bins_per_rank = total_bins // world_size
            idx_start = global_rank * bins_per_rank
            idx_end = (global_rank + 1) * bins_per_rank if global_rank != world_size - 1 else total_bins
            sysmat_local = full_sysmat[idx_start:idx_end, :].clone()
            sysmat_local_all.append(sysmat_local)

            # full matrix on GPU only needed for Compton materialization
            if is_compton:
                sysmat_full_all[e_idx] = full_sysmat.to(device)
            del full_sysmat

            # ---- single-photon sensitivity ----
            if sensi_s_file_path.exists():
                sensi_s = torch.from_numpy(
                    np.fromfile(sensi_s_file_path, dtype=np.float32).reshape(pixel_num, 1).copy()
                ) * intensity
            else:
                sensi_s_tmp = torch.zeros([1, pixel_num], dtype=torch.float32)
                # use the rank-local shard (correct after the eventual all_reduce sum)
                for rotate_idx in range(args.rotate_num):
                    rotmat_inv_tmp = rotmat_inv[:, rotate_idx]
                    sensi_s_tmp += torch.sum(sysmat_local[:, rotmat_inv_tmp - 1], dim=0, keepdim=True).cpu()
                sensi_s = sensi_s_tmp.transpose(0, 1)
                # reduce across ranks to get the global single-photon sensitivity
                sensi_s_gpu = sensi_s.to(device)
                dist.all_reduce(sensi_s_gpu, op=dist.ReduceOp.SUM)
                sensi_s = sensi_s_gpu.cpu() / world_size
            sensi_s_all.append(sensi_s)

            rotmat_all.append(rotmat.to(device))
            rotmat_inv_all.append(rotmat_inv.to(device))

            # ---- projection (single-photon) ----
            full_proj = torch.from_numpy(
                np.genfromtxt(proj_file_path, delimiter=",", dtype=np.float32).reshape(args.rotate_num, -1).T.copy()
            )
            proj_local = full_proj[idx_start:idx_end, :].clone()
            del full_proj

            if is_compton:
                # ---- Compton: list -> T matrix ----
                detector = torch.from_numpy(
                    np.genfromtxt(detector_file_path, delimiter=",", dtype=np.float32)[:, 1:4]
                ).to(device)
                sparse_projector = build_compton_sparse_projector(
                    coor_polar, theta_stride=args.compton_theta_stride, z_stride=args.compton_z_stride,
                    rotate_num=args.rotate_num, dtype=torch.float32)
                sparse_projector_all[e_idx] = sparse_projector

                list_rotate_local = []
                for rotate_idx in range(args.rotate_num):
                    full_list = load_list_csv(list_dir_path / f"{rotate_idx + 1}.csv")
                    ev_per_rank = full_list.size(0) // world_size
                    ev_start = global_rank * ev_per_rank
                    ev_end = (global_rank + 1) * ev_per_rank if global_rank != world_size - 1 else full_list.size(0)
                    list_rotate_local.append(full_list[ev_start:ev_end, :])

                proj_local, list_rotate_local = downsample_projection_and_list(
                    proj_local, list_rotate_local, args.ds * intensity)

                sysmat_full_gpu = sysmat_full_all[e_idx]
                projector_gpu = sparse_projector.to(device)

                t_rotate_local = []
                for rotate_idx in range(args.rotate_num):
                    list_local_chunks = torch.chunk(list_rotate_local[rotate_idx], args.num_workers, dim=0)
                    t_rotate_parts = []
                    for chunk in list_local_chunks:
                        if chunk.numel() == 0:
                            continue
                        t_chunk, _, _ = get_compton_backproj_list_single_sparse(
                            sysmat_full_gpu, detector, projector_gpu, chunk.to(device),
                            args.delta_r1, args.delta_r2, e0, ene_resolution,
                            ene_threshold_max, ene_threshold_min, ene_threshold_sum, device)
                        if t_chunk.numel() > 0:
                            t_rotate_parts.append(t_chunk)
                            compton_event_count_total_local += t_chunk.size(0)
                    t_rotate = (
                        torch.cat(t_rotate_parts, dim=0) if t_rotate_parts
                        else torch.empty((0, sparse_projector.coarse_pixel_num + 1), dtype=torch.float32)
                    )
                    t_rotate_local.append(t_rotate)
                t_local_all[e_idx] = t_rotate_local

                # ---- Compton sensitivity ----
                if sensi_d_file_path.exists() and not args.recompute_sparse_sensi_d:
                    sensi_d = torch.from_numpy(
                        np.fromfile(sensi_d_file_path, dtype=np.float32).reshape(pixel_num, 1).copy()
                    ) * intensity
                    sensi_d_all[e_idx] = sensi_d
                    if global_rank == 0:
                        print(f"Loaded factor Sensi_d for {round(e0*1000)}keV: {sensi_d_file_path}")
                else:
                    if global_rank == 0:
                        print(f"Will recompute Sensi_d from sparse Compton ops for {round(e0*1000)}keV.")
                    # placeholder, computed in batch below
                    sensi_d_all[e_idx] = None

                if global_rank == 0:
                    print(f"[{round(e0*1000)}keV] sparse projector coarse_pixels="
                          f"{sparse_projector.coarse_pixel_num}, ring_strides={sparse_projector.ring_strides}")
            else:
                # non-compton energy: still downsample the projection (no list)
                proj_local, _ = downsample_projection_and_list(proj_local, [torch.empty(0)], args.ds * intensity)
                if global_rank == 0:
                    print(f"[{round(e0*1000)}keV] single-photon only (no Compton loaded).")

            proj_local_all.append(proj_local)
            single_event_count_total_local += round(proj_local.sum().item())

            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ---- global event counts ----
        single_t = torch.tensor([single_event_count_total_local], dtype=torch.float64, device=device)
        compton_t = torch.tensor([compton_event_count_total_local], dtype=torch.float64, device=device)
        dist.all_reduce(single_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(compton_t, op=dist.ReduceOp.SUM)
        single_event_count_total = int(single_t.item())
        compton_event_count_total = int(compton_t.item())

        # ---- recompute Sensi_d if needed (batch over compton energies) ----
        sensi_s_sum = sum(sensi_s_all)
        needs_recompute = any(v is None for v in sensi_d_all.values())
        if needs_recompute:
            sensi_d_local_dict = build_sparse_sensi_d_local(
                t_local_all, sysmat_full_all, sparse_projector_all,
                compton_eidx_list, pixel_num, device)
            # all_reduce each energy's sensi_d
            sensi_d_sum_for_norm = torch.zeros((pixel_num, 1), dtype=torch.float32)
            for e_idx in compton_eidx_list:
                sd_gpu = sensi_d_local_dict[e_idx].to(device)
                dist.all_reduce(sd_gpu, op=dist.ReduceOp.SUM)
                sd_cpu = sd_gpu.cpu()
                sensi_d_all[e_idx] = sd_cpu
                sensi_d_sum_for_norm += sd_cpu
            # normalize so total Compton sensitivity matches total single-photon sensitivity
            # (mirrors the existing chain's rescaling)
            if torch.sum(sensi_d_sum_for_norm) > 0:
                scale = torch.sum(sensi_s_sum) / torch.sum(sensi_d_sum_for_norm)
                scale = scale * compton_event_count_total / max(single_event_count_total, 1)
                for e_idx in compton_eidx_list:
                    sensi_d_all[e_idx] = sensi_d_all[e_idx] * scale
            # apply s_map_d_ratio
            for e_idx in compton_eidx_list:
                sensi_d_all[e_idx] = sensi_d_all[e_idx] * args.s_map_d_ratio
        else:
            for e_idx in compton_eidx_list:
                sensi_d_all[e_idx] = sensi_d_all[e_idx] * args.s_map_d_ratio

        if global_rank == 0:
            print("\n--- Sensitivity stats ---")
            log_tensor_stats("Sensi_s(sum)", sensi_s_sum, global_rank)
            for e_idx in compton_eidx_list:
                log_tensor_stats(f"Sensi_d({round(args.e0_list[e_idx]*1000)}keV)", sensi_d_all[e_idx], global_rank)
            print(f"single_event_count_total = {single_event_count_total}")
            print(f"compton_event_count_total = {compton_event_count_total}")

        # ---- build tasks ----
        iter_cfg = IterConfig(
            type1_single_sc=(args.single_sc_iter, args.single_sc_save_step),
            type2_single_compton=(args.single_compton_iter, args.single_compton_save_step),
            type3_joint_sc=(args.joint_sc_iter, args.joint_sc_save_step),
            type4_joint_compton=(args.joint_compton_iter, args.joint_compton_save_step),
            type5_joint=(args.joint_iter, args.joint_save_step),
        )
        tasks = build_tasks(args.e0_list, compton_eidx_list, iter_cfg)

        if not tasks:
            if global_rank == 0:
                print("No tasks to run (check iteration flags).")
            return

        if global_rank == 0:
            print("\n--- Task list ---")
            print(format_task_table(tasks, args.e0_list))

        # ---- save path ----
        save_root = build_save_root(
            output_root, args.e0_list, args, single_event_count_total,
            compton_event_count_total, args.compton_theta_stride, args.compton_z_stride)
        if global_rank == 0:
            save_root.mkdir(parents=True, exist_ok=True)
            print(f"\nSave root: {save_root}")
        dist.barrier()

        # ---- run ----
        loaded = {
            "sysmat_local_all": sysmat_local_all,
            "proj_local_all": proj_local_all,
            "rotmat_all": rotmat_all,
            "rotmat_inv_all": rotmat_inv_all,
            "sensi_s_all": sensi_s_all,
            "sysmat_full_all": sysmat_full_all,
            "sparse_projector_all": sparse_projector_all,
            "sensi_d_all": sensi_d_all,
            "t_local_all": t_local_all,
            "pixel_num": pixel_num,
            "rotate_num": args.rotate_num,
            "osem_subset_num": args.osem_subset_num,
            "t_divide_num": args.t_divide_num,
            "alpha": args.alpha,
            "seed": args.seed,
        }
        run_tasks_multi_energy(loaded, tasks, device, str(save_root) + os.sep)

        if global_rank == 0:
            print(f"\nTotal wall time: {time.time() - start_time:.2f}s")

    finally:
        if logfile is not None:
            sys.stdout = sys.__stdout__
            logfile.close()
        if log_filename is not None and save_root is not None and Path(save_root).is_dir():
            final_log = Path(save_root) / "print_log.txt"
            try:
                shutil.move(str(log_filename), final_log)
                if global_rank == 0:
                    print(f"Log moved to {final_log}")
            except Exception:
                pass
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    with torch.no_grad():
        main()
