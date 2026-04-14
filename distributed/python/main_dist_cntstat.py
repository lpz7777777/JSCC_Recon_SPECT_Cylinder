import argparse
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from _path_setup import setup_repo_root
setup_repo_root()
from recon_osem_dist_cntstat import run_recon_osem_dist_cntstat
from sparse_main_utils import build_output_name_prefix, format_scientific_count


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_distributed():
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return global_rank, local_rank, world_size


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed single-photon reconstruction using only CntStat data.")
    parser.add_argument("--e0-list", type=float, nargs="+", default=[0.511], help="Energy list in MeV.")
    parser.add_argument("--intensity-list", type=float, nargs="+", default=[1.0], help="Intensity weight for each energy.")
    parser.add_argument("--data-file-name", type=str, default="Hoffman_Big", help="Phantom or dataset name.")
    parser.add_argument("--count-level", type=str, default="5e9", help="Count level suffix in CntStat filename.")
    parser.add_argument("--ds", type=float, default=1.0, help="Projection downsampling ratio.")
    parser.add_argument("--pixel-num-layer", type=int, default=1280, help="Number of polar pixels per slice.")
    parser.add_argument("--pixel-num-z", type=int, default=20, help="Number of axial slices.")
    parser.add_argument("--rotate-num", type=int, default=20, help="Number of rotations.")
    parser.add_argument("--ene-resolution-662keV", type=float, default=0.1, help="Kept for output naming compatibility.")
    parser.add_argument("--sc-iter", type=int, default=10000, help="Number of SC OSEM iterations.")
    parser.add_argument("--save-iter-step", type=int, default=100, help="Save interval.")
    parser.add_argument("--osem-subset-num", type=int, default=4, help="Number of OSEM subsets.")
    parser.add_argument("--seed", type=int, default=20260331, help="Random seed.")
    parser.add_argument("--factors-dir", type=str, default="./Factors", help="Root directory of system factors.")
    parser.add_argument("--cntstat-dir", type=str, default="./CntStat", help="Root directory of CntStat data.")
    parser.add_argument("--output-root", type=str, default="./Figure_Dist_SC", help="Root directory of outputs.")
    return parser.parse_args()


def validate_args(args):
    if len(args.e0_list) != len(args.intensity_list):
        raise ValueError("--e0-list and --intensity-list must have the same length.")

    if not (0 < args.ds <= 1):
        raise ValueError("--ds must be within (0, 1].")

    if args.sc_iter <= 0 or args.save_iter_step <= 0:
        raise ValueError("--sc-iter and --save-iter-step must be positive.")

    if args.sc_iter % args.save_iter_step != 0:
        raise ValueError("--sc-iter must be divisible by --save-iter-step.")


def downsample_projection(full_proj, ds, seed):
    if ds >= 0.999999:
        return full_proj

    full_proj_np = np.rint(full_proj.numpy()).astype(np.int64)
    rng = np.random.default_rng(seed)
    proj_ds_np = rng.binomial(full_proj_np, ds).astype(np.float32)
    return torch.from_numpy(proj_ds_np)


def get_sysmat_bin_range(total_bins, global_rank, world_size):
    bins_per_rank = total_bins // world_size
    idx_start = global_rank * bins_per_rank
    idx_end = (global_rank + 1) * bins_per_rank if global_rank != world_size - 1 else total_bins
    return idx_start, idx_end


def load_local_sysmat(sysmat_file_path, pixel_num, intensity, global_rank, world_size):
    float_size = np.dtype(np.float32).itemsize
    element_count = os.path.getsize(sysmat_file_path) // float_size

    if element_count % pixel_num != 0:
        raise ValueError(f"SysMat size in {sysmat_file_path} is incompatible with pixel_num={pixel_num}.")

    total_bins = element_count // pixel_num
    idx_start, idx_end = get_sysmat_bin_range(total_bins, global_rank, world_size)
    sysmat_mmap = np.memmap(sysmat_file_path, dtype=np.float32, mode="r", shape=(pixel_num, total_bins))
    sysmat_local = torch.from_numpy(np.array(sysmat_mmap[:, idx_start:idx_end], dtype=np.float32).T.copy()) * intensity
    del sysmat_mmap
    return sysmat_local, total_bins, idx_start, idx_end


def resolve_pixel_num(args, rotmat, rotmat_inv, global_rank):
    pixel_num_from_args = args.pixel_num_layer * args.pixel_num_z
    pixel_num_from_rotmat = rotmat.size(0)
    pixel_num_from_rotmat_inv = rotmat_inv.size(0)

    if pixel_num_from_rotmat != pixel_num_from_rotmat_inv:
        raise ValueError(
            "Rotation matrix row mismatch: "
            f"rotmat rows={pixel_num_from_rotmat}, rotmat_inv rows={pixel_num_from_rotmat_inv}."
        )

    if pixel_num_from_rotmat != pixel_num_from_args and global_rank == 0:
        inferred_layer_msg = ""
        if args.pixel_num_z > 0 and pixel_num_from_rotmat % args.pixel_num_z == 0:
            inferred_layer = pixel_num_from_rotmat // args.pixel_num_z
            inferred_layer_msg = f", inferred pixel_num_layer={inferred_layer}"
        print(
            "Pixel count mismatch detected. "
            f"Using factor-defined pixel_num={pixel_num_from_rotmat} instead of "
            f"args pixel_num_layer * pixel_num_z = {pixel_num_from_args}{inferred_layer_msg}."
        )

    return pixel_num_from_rotmat


def compute_global_sensitivity(sysmat_local, rotmat_inv, rotate_num, device):
    sensi_local = torch.zeros([1, sysmat_local.size(1)], dtype=torch.float32)

    for rotate_idx in range(rotate_num):
        sensi_local += torch.sum(sysmat_local[:, rotmat_inv[:, rotate_idx] - 1], dim=0, keepdim=True).cpu()

    sensi_local = sensi_local.transpose(0, 1).to(device)
    dist.all_reduce(sensi_local, op=dist.ReduceOp.SUM)
    return sensi_local / rotate_num


def build_save_path(args, single_event_count_total):
    name_prefix = build_output_name_prefix(args.e0_list, args.rotate_num, args.data_file_name)
    single_event_count_str = format_scientific_count(single_event_count_total)

    return (
        f"{args.output_root}/{name_prefix}_{args.count_level}_{args.ds}_"
        f"OSEM{args.osem_subset_num}_ITER{args.sc_iter}_SDU{single_event_count_str}/Polar/"
    )


def main():
    args = parse_args()
    validate_args(args)

    global_rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    start_time = time.time()
    log_filename = None
    logfile = None
    save_path = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    try:
        if global_rank == 0:
            rand_suffix = f"{random.randint(0, 9999):04d}"
            log_filename = f"print_log_dist_sc_{rand_suffix}.txt"
            logfile = open(log_filename, "w", encoding="utf-8")
            sys.stdout = Tee(sys.__stdout__, logfile)
            print(f"Distributed initialized: world_size={world_size}, local_rank={local_rank}")
            print(f"Args: {args}")

        pixel_num = None

        proj_local_all = []
        sysmat_local_all = []
        rotmat_all = []
        rotmat_inv_all = []
        sensi_s_all = []

        single_event_count_total = torch.zeros(1, dtype=torch.float64, device=device)

        for energy_idx, (e0, intensity) in enumerate(zip(args.e0_list, args.intensity_list)):
            factor_path = os.path.join(args.factors_dir, f"{round(1000 * e0)}keV_RotateNum{args.rotate_num}")
            sysmat_file_path = os.path.join(factor_path, "SysMat_polar")
            rotmat_file_path = os.path.join(factor_path, "RotMat_full.csv")
            rotmat_inv_file_path = os.path.join(factor_path, "RotMatInv_full.csv")
            proj_file_path = os.path.join(
                args.cntstat_dir,
                f"{round(1000 * e0)}keV_RotateNum{args.rotate_num}",
                f"CntStat_{args.data_file_name}_{args.count_level}.csv",
            )

            for required_path in (sysmat_file_path, rotmat_file_path, rotmat_inv_file_path, proj_file_path):
                if not os.path.exists(required_path):
                    raise FileNotFoundError(required_path)

            rotmat = torch.from_numpy(np.genfromtxt(rotmat_file_path, delimiter=",", dtype=np.int64))
            rotmat_inv = torch.from_numpy(np.genfromtxt(rotmat_inv_file_path, delimiter=",", dtype=np.int64))
            pixel_num_current = resolve_pixel_num(args, rotmat, rotmat_inv, global_rank)

            if pixel_num is None:
                pixel_num = pixel_num_current
            elif pixel_num != pixel_num_current:
                raise ValueError(
                    f"Inconsistent pixel_num across energies: previous={pixel_num}, current={pixel_num_current}."
                )

            sysmat_local, total_bins, idx_start, idx_end = load_local_sysmat(
                sysmat_file_path, pixel_num, intensity, global_rank, world_size
            )

            full_proj = torch.from_numpy(
                np.genfromtxt(proj_file_path, delimiter=",", dtype=np.float32).reshape(args.rotate_num, -1)
            ).transpose(0, 1)

            if full_proj.size(0) != total_bins:
                raise ValueError(
                    f"Projection bin count mismatch for {proj_file_path}: "
                    f"proj bins={full_proj.size(0)}, sysmat bins={total_bins}."
                )

            full_proj = downsample_projection(full_proj, args.ds, args.seed + energy_idx)
            proj_local = full_proj[idx_start:idx_end, :]

            sensi_s = compute_global_sensitivity(sysmat_local, rotmat_inv, args.rotate_num, device)

            proj_local_all.append(proj_local)
            sysmat_local_all.append(sysmat_local)
            rotmat_all.append(rotmat.to(device))
            rotmat_inv_all.append(rotmat_inv.to(device))
            sensi_s_all.append(sensi_s)

            single_event_count_total += proj_local.sum(dtype=torch.float64).to(device)

            if global_rank == 0:
                print(
                    f"Loaded energy {e0:.3f} MeV | total_bins={total_bins} | "
                    f"local_bin_range=[{idx_start}, {idx_end})"
                )

        dist.all_reduce(single_event_count_total, op=dist.ReduceOp.SUM)

        s_map_arg = argparse.Namespace()
        s_map_arg.s = sum(sensi_s_all).to(device)

        iter_arg = argparse.Namespace()
        iter_arg.sc = args.sc_iter
        iter_arg.save_iter_step = args.save_iter_step
        iter_arg.osem_subset_num = args.osem_subset_num
        iter_arg.ene_num = len(args.e0_list)
        iter_arg.seed = args.seed

        save_path = build_save_path(args, round(single_event_count_total.item()))
        if global_rank == 0:
            os.makedirs(save_path, exist_ok=True)
        dist.barrier()

        run_recon_osem_dist_cntstat(
            sysmat_local_all,
            rotmat_all,
            rotmat_inv_all,
            proj_local_all,
            iter_arg,
            s_map_arg,
            save_path,
        )

        if global_rank == 0:
            print(f"Finished in {time.time() - start_time:.2f}s")

    finally:
        if global_rank == 0:
            if logfile is not None:
                sys.stdout = sys.__stdout__
                logfile.close()

            if log_filename is not None and save_path is not None and os.path.isdir(save_path):
                final_log_path = os.path.join(save_path, "print_log.txt")
                shutil.move(log_filename, final_log_path)
                print(f"Log saved to {final_log_path}")

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    with torch.no_grad():
        main()
