import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from recon_osem_local_cntstat import run_recon_osem_local_cntstat
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


def parse_args():
    parser = argparse.ArgumentParser(description="Local single-photon reconstruction using only CntStat data.")
    parser.add_argument("--e0-list", type=float, nargs="+", default=[0.140], help="Energy list in MeV.")
    parser.add_argument("--intensity-list", type=float, nargs="+", default=[1.0], help="Intensity weight for each energy.")
    parser.add_argument("--data-file-name", type=str, default="Hoffman_Big", help="Phantom or dataset name.")
    parser.add_argument("--count-level", type=str, default="1e10", help="Count level suffix in CntStat filename.")
    parser.add_argument("--ds", type=float, default=1.0, help="Projection downsampling ratio.")
    parser.add_argument("--pixel-num-layer", type=int, default=4500, help="Number of polar pixels per slice.")
    parser.add_argument("--pixel-num-z", type=int, default=20, help="Number of axial slices.")
    parser.add_argument("--rotate-num", type=int, default=60, help="Number of rotations.")
    parser.add_argument("--ene-resolution-662keV", type=float, default=0.1, help="Kept for output naming compatibility.")
    parser.add_argument("--sc-iter", type=int, default=200, help="Number of SC OSEM iterations.")
    parser.add_argument("--save-iter-step", type=int, default=2, help="Save interval.")
    parser.add_argument("--osem-subset-num", type=int, default=1, help="Number of OSEM subsets.")
    parser.add_argument("--seed", type=int, default=20260511, help="Random seed.")
    parser.add_argument("--factors-dir", type=str, default="./Factors", help="Root directory of system factors.")
    parser.add_argument(
        "--factor-dir-suffix",
        type=str,
        default="SPECTEHENaILowerResPbRing60120",
        help=(
            "Optional suffix appended to the factor directory name. "
            "For example, 'SPECTEHENaILowerResPbRing60120' selects "
            "'<factors-dir>/<energy>keV_RotateNum<rotate-num>_ConventionalSPECT'."
        ),
    )
    parser.add_argument("--cntstat-dir", type=str, default="./CntStat", help="Root directory of CntStat data.")
    parser.add_argument("--output-root", type=str, default="./Figure_Dist_SC", help="Root directory of outputs.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: 'auto', 'cuda', 'cuda:0', or 'cpu'.",
    )
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


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")
    return device


def downsample_projection(full_proj, ds, seed):
    if ds >= 0.999999:
        return full_proj

    full_proj_np = np.rint(full_proj.numpy()).astype(np.int64)
    rng = np.random.default_rng(seed)
    proj_ds_np = rng.binomial(full_proj_np, ds).astype(np.float32)
    return torch.from_numpy(proj_ds_np)


def load_full_sysmat(sysmat_file_path, pixel_num, intensity):
    float_size = np.dtype(np.float32).itemsize
    element_count = os.path.getsize(sysmat_file_path) // float_size

    if element_count % pixel_num != 0:
        raise ValueError(f"SysMat size in {sysmat_file_path} is incompatible with pixel_num={pixel_num}.")

    total_bins = element_count // pixel_num
    sysmat_mmap = np.memmap(sysmat_file_path, dtype=np.float32, mode="r", shape=(pixel_num, total_bins))
    sysmat = torch.from_numpy(np.array(sysmat_mmap, dtype=np.float32).T.copy()) * intensity
    del sysmat_mmap
    return sysmat, total_bins


def resolve_pixel_num(args, rotmat, rotmat_inv):
    pixel_num_from_args = args.pixel_num_layer * args.pixel_num_z
    pixel_num_from_rotmat = rotmat.size(0)
    pixel_num_from_rotmat_inv = rotmat_inv.size(0)

    if pixel_num_from_rotmat != pixel_num_from_rotmat_inv:
        raise ValueError(
            "Rotation matrix row mismatch: "
            f"rotmat rows={pixel_num_from_rotmat}, rotmat_inv rows={pixel_num_from_rotmat_inv}."
        )

    if pixel_num_from_rotmat != pixel_num_from_args:
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


def compute_sensitivity_local(sysmat, rotmat_inv, rotate_num):
    sensi = torch.zeros([1, sysmat.size(1)], dtype=torch.float32)

    for rotate_idx in range(rotate_num):
        sensi += torch.sum(sysmat[:, rotmat_inv[:, rotate_idx] - 1], dim=0, keepdim=True)

    return sensi.transpose(0, 1) / rotate_num


def build_energy_subdir_name(e0, rotate_num, dir_suffix):
    dir_name = f"{round(1000 * e0)}keV_RotateNum{rotate_num}"
    suffix = dir_suffix.strip()
    if suffix:
        suffix = suffix.lstrip("_")
        dir_name = f"{dir_name}_{suffix}"
    return dir_name


def format_factor_suffix_tag(dir_suffix):
    suffix = dir_suffix.strip()
    if not suffix:
        return ""
    return f"_FSfx{suffix.lstrip('_')}"


def build_factor_path(factors_dir, e0, rotate_num, factor_dir_suffix):
    factor_dir_name = build_energy_subdir_name(e0, rotate_num, factor_dir_suffix)
    return os.path.join(factors_dir, factor_dir_name)


def build_cntstat_energy_dir(cntstat_dir, e0, rotate_num, dir_suffix):
    cntstat_dir_name = build_energy_subdir_name(e0, rotate_num, dir_suffix)
    return os.path.join(cntstat_dir, cntstat_dir_name)


def build_proj_file_path(cntstat_dir, data_file_name, count_level, e0, rotate_num, dir_suffix):
    cntstat_energy_dir = build_cntstat_energy_dir(cntstat_dir, e0, rotate_num, dir_suffix)
    return os.path.join(
        cntstat_energy_dir,
        f"CntStat_{data_file_name}_{count_level}.csv",
    )


def build_save_path(args, single_event_count_total):
    name_prefix = build_output_name_prefix(args.e0_list, args.rotate_num, args.data_file_name)
    single_event_count_str = format_scientific_count(single_event_count_total)
    factor_suffix_tag = format_factor_suffix_tag(args.factor_dir_suffix)

    return (
        f"{args.output_root}/{name_prefix}{factor_suffix_tag}_{args.count_level}_{args.ds}_"
        f"OSEM{args.osem_subset_num}_ITER{args.sc_iter}_SDU{single_event_count_str}/Polar/"
    )


def resolve_path(repo_root, path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def main():
    args = parse_args()
    validate_args(args)

    repo_root = Path(__file__).resolve().parent
    device = resolve_device(args.device)
    start_time = time.time()
    log_filename = None
    logfile = None
    save_path = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    factors_dir = str(resolve_path(repo_root, args.factors_dir))
    cntstat_dir = str(resolve_path(repo_root, args.cntstat_dir))
    output_root = str(resolve_path(repo_root, args.output_root))

    try:
        rand_suffix = f"{random.randint(0, 9999):04d}"
        log_filename = repo_root / f"print_log_local_sc_{rand_suffix}.txt"
        logfile = open(log_filename, "w", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, logfile)
        print(f"Using device: {device}")
        print(f"Args: {args}")

        pixel_num = None

        proj_all = []
        sysmat_all = []
        rotmat_all = []
        rotmat_inv_all = []
        sensi_s_all = []
        single_event_count_total = 0.0

        for energy_idx, (e0, intensity) in enumerate(zip(args.e0_list, args.intensity_list)):
            factor_path = build_factor_path(factors_dir, e0, args.rotate_num, args.factor_dir_suffix)
            sysmat_file_path = os.path.join(factor_path, "SysMat_polar")
            rotmat_file_path = os.path.join(factor_path, "RotMat_full.csv")
            rotmat_inv_file_path = os.path.join(factor_path, "RotMatInv_full.csv")
            cntstat_energy_dir = build_cntstat_energy_dir(
                cntstat_dir, e0, args.rotate_num, args.factor_dir_suffix
            )
            proj_file_path = build_proj_file_path(
                cntstat_dir,
                args.data_file_name,
                args.count_level,
                e0,
                args.rotate_num,
                args.factor_dir_suffix,
            )

            for required_path in (sysmat_file_path, rotmat_file_path, rotmat_inv_file_path, proj_file_path):
                if not os.path.exists(required_path):
                    raise FileNotFoundError(required_path)

            rotmat = torch.from_numpy(np.genfromtxt(rotmat_file_path, delimiter=",", dtype=np.int64))
            rotmat_inv = torch.from_numpy(np.genfromtxt(rotmat_inv_file_path, delimiter=",", dtype=np.int64))
            pixel_num_current = resolve_pixel_num(args, rotmat, rotmat_inv)

            if pixel_num is None:
                pixel_num = pixel_num_current
            elif pixel_num != pixel_num_current:
                raise ValueError(
                    f"Inconsistent pixel_num across energies: previous={pixel_num}, current={pixel_num_current}."
                )

            sysmat, total_bins = load_full_sysmat(sysmat_file_path, pixel_num, intensity)

            full_proj = torch.from_numpy(
                np.genfromtxt(proj_file_path, delimiter=",", dtype=np.float32).reshape(args.rotate_num, -1)
            ).transpose(0, 1)

            if full_proj.size(0) != total_bins:
                raise ValueError(
                    f"Projection bin count mismatch for {proj_file_path}: "
                    f"proj bins={full_proj.size(0)}, sysmat bins={total_bins}."
                )

            full_proj = downsample_projection(full_proj, args.ds, args.seed + energy_idx)
            sensi_s = compute_sensitivity_local(sysmat, rotmat_inv, args.rotate_num)

            proj_all.append(full_proj)
            sysmat_all.append(sysmat)
            rotmat_all.append(rotmat)
            rotmat_inv_all.append(rotmat_inv)
            sensi_s_all.append(sensi_s)

            single_event_count_total += full_proj.sum(dtype=torch.float64).item()

            print(
                f"Loaded energy {e0:.3f} MeV from {factor_path} "
                f"with CntStat dir {cntstat_energy_dir} | total_bins={total_bins}"
            )

        s_map_arg = argparse.Namespace()
        s_map_arg.s = sum(sensi_s_all)

        iter_arg = argparse.Namespace()
        iter_arg.sc = args.sc_iter
        iter_arg.save_iter_step = args.save_iter_step
        iter_arg.osem_subset_num = args.osem_subset_num
        iter_arg.ene_num = len(args.e0_list)
        iter_arg.seed = args.seed

        save_args = argparse.Namespace(**vars(args))
        save_args.output_root = output_root
        save_path = build_save_path(save_args, round(single_event_count_total))
        os.makedirs(save_path, exist_ok=True)

        run_recon_osem_local_cntstat(
            sysmat_all,
            rotmat_all,
            rotmat_inv_all,
            proj_all,
            iter_arg,
            s_map_arg,
            save_path,
            device,
        )

        print(f"Finished in {time.time() - start_time:.2f}s")

    finally:
        if logfile is not None:
            sys.stdout = sys.__stdout__
            logfile.close()

        if log_filename is not None and save_path is not None and os.path.isdir(save_path):
            final_log_path = os.path.join(save_path, "print_log.txt")
            shutil.move(str(log_filename), final_log_path)
            print(f"Log saved to {final_log_path}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
