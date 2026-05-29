import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from compton_sparse_ops import (
    build_compton_sparse_projector,
    materialize_sparse_event_rows_to_fine,
)
from process_list_plane_sparse import get_compton_backproj_list_single_sparse
from recon_osem_local_sparse_jsccsd_only import run_recon_osem_local_sparse_jsccsd_only
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


def parse_args():
    parser = argparse.ArgumentParser(description="Local sparse JSCCSD-only reconstruction for plane geometry.")
    parser.add_argument("--e0-list", type=float, nargs="+", default=[0.511], help="Energy list in MeV.")
    parser.add_argument("--ene-threshold-sum-list", type=float, nargs="+", default=[0.46], help="Lower bounds for e1 + e2 in MeV.")
    parser.add_argument("--intensity-list", type=float, nargs="+", default=[1.0], help="Intensity weights.")
    parser.add_argument("--s-map-d-ratio", type=float, default=1.0, help="Scale factor applied to sparse Sensi_d.")
    parser.add_argument("--recompute-sparse-sensi-d", action="store_true", help="Force recomputing Sensi_d from sparse Compton operators instead of reusing factor-file Sensi_d.")
    parser.add_argument("--data-file-name", type=str, default="ContrastPhantom_240_30", help="Dataset name.")
    parser.add_argument("--count-level", type=str, default="1e9", help="Count level suffix.")
    parser.add_argument("--ds", type=float, default=1.0, help="Downsampling ratio in (0, 1].")
    parser.add_argument("--ene-resolution-662keV", type=float, default=0.1, help="Reference energy resolution.")
    parser.add_argument("--pixel-num-layer", type=int, default=1280, help="Legacy fallback layer pixel count.")
    parser.add_argument("--pixel-num-z", type=int, default=20, help="Axial slice count.")
    parser.add_argument("--rotate-num", type=int, default=20, help="Number of views.")
    parser.add_argument("--delta-r1", type=float, default=0.0, help="Additional isotropic position sigma for the first interaction in mm.")
    parser.add_argument("--delta-r2", type=float, default=0.0, help="Additional isotropic position sigma for the second interaction in mm.")
    parser.add_argument("--alpha", type=float, default=1.0, help="JSCC weighting parameter.")
    parser.add_argument("--jsccsd-iter", type=int, default=5000, help="JSCC-SD iteration count.")
    parser.add_argument("--save-iter-step", type=int, default=50, help="Save interval.")
    parser.add_argument("--osem-subset-num", type=int, default=1, help="OSEM subset count.")
    parser.add_argument("--t-divide-num", type=int, default=10, help="Number of t sub-blocks per subset.")
    parser.add_argument("--num-workers", type=int, default=20, help="Sub-chunks during sparse list processing.")
    parser.add_argument("--compton-theta-stride", type=int, default=1, help="Angular stride used by sparse Compton grid.")
    parser.add_argument("--compton-z-stride", type=int, default=2, help="Axial stride used by sparse Compton grid.")
    parser.add_argument("--seed", type=int, default=20260331, help="Random seed.")
    parser.add_argument("--factors-dir", type=str, default="./Factors", help="Factors root directory.")
    parser.add_argument("--cntstat-dir", type=str, default="./CntStat", help="CntStat root directory.")
    parser.add_argument("--list-dir", type=str, default="./List", help="List root directory.")
    parser.add_argument("--output-root", type=str, default="./Fig_JSCC", help="Output root directory.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: 'auto', 'cuda', 'cuda:0', or 'cpu'.",
    )
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


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")
    return device


def build_sparse_sensi_d_local(t_local_all, sysmat_full_all, sparse_projector_all, pixel_num, device, block_size=1024):
    sensi_d_local = torch.zeros((pixel_num, 1), dtype=torch.float32)
    for t_energy, sysmat_full, sparse_projector in zip(t_local_all, sysmat_full_all, sparse_projector_all):
        projector_device = sparse_projector.to(device)
        sysmat_device = sysmat_full.to(device, non_blocking=(device.type == "cuda"))
        for t_rotate in t_energy:
            if t_rotate.numel() == 0:
                continue
            for row_start in range(0, t_rotate.size(0), block_size):
                event_block = t_rotate[row_start:row_start + block_size].to(device, non_blocking=(device.type == "cuda"))
                t_fine, _ = materialize_sparse_event_rows_to_fine(event_block, sysmat_device, projector_device)
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


def main():
    args = parse_args()
    validate_args_jsccsd_only(args)

    repo_root = resolve_repo_root()
    device = resolve_device(args.device)
    start_time = time.time()
    log_filename = None
    logfile = None
    save_path = None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    factors_root = (repo_root / args.factors_dir).resolve()
    cntstat_root = (repo_root / args.cntstat_dir).resolve()
    list_root = (repo_root / args.list_dir).resolve()
    output_root = (repo_root / args.output_root).resolve()

    try:
        rand_suffix = f"{random.randint(0, 9999):04d}"
        log_filename = repo_root / f"print_log_local_sparse_jsccsd_only_{rand_suffix}.txt"
        logfile = open(log_filename, "w", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, logfile)
        print(f"Using device: {device}")
        print(f"Args: {args}")

        proj_all = []
        sysmat_all = []
        sysmat_full_all = []
        rotmat_all = []
        rotmat_inv_all = []
        sensi_s_all = []
        sensi_d_all = []
        sparse_projector_all = []

        pixel_num = None
        single_event_count_total = 0
        compton_event_count_total = 0
        t_local_all = []

        for e0, ene_threshold_sum, intensity in zip(args.e0_list, args.ene_threshold_sum_list, args.intensity_list):
            ene_resolution = args.ene_resolution_662keV * (0.662 / e0) ** 0.5
            ene_threshold_max = 2 * e0 ** 2 / (0.511 + 2 * e0) - 0.001
            ene_threshold_min = 0.05

            factor_dir = resolve_factor_dir(factors_root, e0, args.rotate_num)
            proj_file_path, list_dir_path = resolve_proj_and_list_paths(
                cntstat_root,
                list_root,
                e0,
                args.rotate_num,
                args.data_file_name,
                args.count_level,
            )

            sysmat_file_path = factor_dir / "SysMat_polar"
            detector_file_path = factor_dir / "Detector.csv"
            sensi_s_file_path = factor_dir / "Sensi_s"
            sensi_d_file_path = factor_dir / "Sensi_d"
            coor_polar_file_path = factor_dir / "coor_polar_full.csv"
            rotmat_file_path = factor_dir / "RotMat_full.csv"
            rotmat_inv_file_path = factor_dir / "RotMatInv_full.csv"

            detector = torch.from_numpy(np.genfromtxt(detector_file_path, delimiter=",", dtype=np.float32)[:, 1:4]).to(device)
            coor_polar = torch.from_numpy(np.genfromtxt(coor_polar_file_path, delimiter=",", dtype=np.float32))
            rotmat = torch.from_numpy(np.genfromtxt(rotmat_file_path, delimiter=",", dtype=np.int64))
            rotmat_inv = torch.from_numpy(np.genfromtxt(rotmat_inv_file_path, delimiter=",", dtype=np.int64))

            pixel_num_current = resolve_pixel_num(
                args.pixel_num_layer * args.pixel_num_z,
                args.pixel_num_z,
                rotmat,
                rotmat_inv,
                coor_polar,
                factor_dir,
            )
            if pixel_num is None:
                pixel_num = pixel_num_current
            elif pixel_num != pixel_num_current:
                raise ValueError(f"Inconsistent pixel_num across energies: previous={pixel_num}, current={pixel_num_current}")

            full_sysmat = torch.from_numpy(np.fromfile(sysmat_file_path, dtype=np.float32).reshape(pixel_num, -1).T.copy()) * intensity
            sysmat_all.append(full_sysmat)
            sysmat_full_all.append(full_sysmat)

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
                print(f"Loaded factor Sensi_d: {sensi_d_file_path}")
            else:
                print(f"Factor Sensi_d not found, will recompute if needed: {sensi_d_file_path}")

            sparse_projector = build_compton_sparse_projector(
                coor_polar,
                theta_stride=args.compton_theta_stride,
                z_stride=args.compton_z_stride,
                rotate_num=args.rotate_num,
                dtype=torch.float32,
            )
            sparse_projector_all.append(sparse_projector)

            full_proj = torch.from_numpy(
                np.genfromtxt(proj_file_path, delimiter=",", dtype=np.float32).reshape(args.rotate_num, -1).T.copy()
            )
            list_rotate_all = []
            for rotate_idx in range(args.rotate_num):
                list_rotate_all.append(load_list_csv(list_dir_path / f"{rotate_idx + 1}.csv"))

            proj, list_rotate_all = downsample_projection_and_list(full_proj, list_rotate_all, args.ds * intensity)
            proj_all.append(proj)
            single_event_count_total += round(proj.sum().item())

            sysmat_device = full_sysmat.to(device, non_blocking=(device.type == "cuda"))
            sparse_projector_device = sparse_projector.to(device)

            t_rotate_all = []
            for rotate_idx in range(args.rotate_num):
                list_local_chunks = torch.chunk(list_rotate_all[rotate_idx], args.num_workers, dim=0)
                t_rotate_parts = []
                for chunk in list_local_chunks:
                    if chunk.numel() == 0:
                        continue
                    t_chunk, _, _ = get_compton_backproj_list_single_sparse(
                        sysmat_device,
                        detector,
                        sparse_projector_device,
                        chunk.to(device, non_blocking=(device.type == "cuda")),
                        args.delta_r1,
                        args.delta_r2,
                        e0,
                        ene_resolution,
                        ene_threshold_max,
                        ene_threshold_min,
                        ene_threshold_sum,
                        device,
                    )
                    if t_chunk.numel() > 0:
                        t_rotate_parts.append(t_chunk)
                        compton_event_count_total += t_chunk.size(0)
                t_rotate = (
                    torch.cat(t_rotate_parts, dim=0)
                    if t_rotate_parts
                    else torch.empty((0, sparse_projector.coarse_pixel_num + 1), dtype=torch.float32)
                )
                t_rotate_all.append(t_rotate)
            t_local_all.append(t_rotate_all)

            rotmat_all.append(rotmat)
            rotmat_inv_all.append(rotmat_inv)

            print(
                f"Loaded sparse projector for {e0:.3f} MeV: fine_pixels={pixel_num}, "
                f"sparse_pixels={sparse_projector.coarse_pixel_num}, ring_strides={sparse_projector.ring_strides}"
            )

            torch.cuda.empty_cache()

        s_map_arg = argparse.Namespace()
        s_map_arg.s = sum(sensi_s_all)
        if sensi_d_all and not args.recompute_sparse_sensi_d:
            print("Local sparse JSCCSD-only chain uses file-defined Sensi_d.")
            s_map_arg.d = sum(sensi_d_all) * args.s_map_d_ratio
        else:
            if sensi_d_all:
                print("Local sparse JSCCSD-only chain recomputes Sensi_d from sparse Compton operators.")
            else:
                print("Local sparse JSCCSD-only chain has no file-defined Sensi_d and recomputes it from sparse Compton operators.")
            sensi_d = build_sparse_sensi_d_local(
                t_local_all,
                sysmat_full_all,
                sparse_projector_all,
                pixel_num,
                device,
            )
            if torch.sum(sensi_d) > 0:
                sensi_d = sensi_d * torch.sum(s_map_arg.s) / torch.sum(sensi_d)
                sensi_d = sensi_d * compton_event_count_total / max(single_event_count_total, 1)
            s_map_arg.d = sensi_d * args.s_map_d_ratio
        s_map_arg.j = args.alpha * s_map_arg.s + (2 - args.alpha) * s_map_arg.d

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
        sparse_prefix = f"{args.compton_theta_stride}_{args.compton_z_stride}_"
        save_path = base_path.parent.parent / f"{sparse_prefix}{base_path.parent.name}" / "Polar"
        save_path.mkdir(parents=True, exist_ok=True)

        run_recon_osem_local_sparse_jsccsd_only(
            sysmat_all,
            sysmat_full_all,
            rotmat_all,
            rotmat_inv_all,
            proj_all,
            t_local_all,
            sparse_projector_all,
            iter_arg,
            s_map_arg,
            args.alpha,
            str(save_path) + os.sep,
            device,
        )

        print(f"Finished in {time.time() - start_time:.2f}s")

    finally:
        if logfile is not None:
            sys.stdout = sys.__stdout__
            logfile.close()

        if log_filename is not None and save_path is not None and Path(save_path).is_dir():
            final_log_path = Path(save_path) / "print_log.txt"
            shutil.move(str(log_filename), final_log_path)
            print(f"Log saved to {final_log_path}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
