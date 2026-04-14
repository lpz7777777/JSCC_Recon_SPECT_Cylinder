import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
# process_list_plane_strict
from process_list_plane_strict import (
    get_compton_backproj_list_mp,
    get_compton_backproj_list_single,
)
from recon_osem_plane import run_recon_osem


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
    parser = argparse.ArgumentParser(description="Local multi-GPU JSCC reconstruction for plane geometry.")
    parser.add_argument("--num-gpus", type=int, default=torch.cuda.device_count(), help="Number of local GPUs to use.")
    parser.add_argument("--e0-list", type=float, nargs="+", default=[0.511], help="Energy list in MeV.")
    parser.add_argument(
        "--ene-threshold-sum-list",
        type=float,
        nargs="+",
        default=[0.46],
        help="Lower bounds for e1 + e2 in MeV.",
    )
    parser.add_argument("--intensity-list", type=float, nargs="+", default=[1.0], help="Intensity weights.")
    parser.add_argument("--s-map-d-ratio", type=float, default=1.0, help="Scale factor applied to file-based Sensi_d.")
    parser.add_argument("--data-file-name", type=str, default="ContrastPhantom_240_30", help="Dataset name.")
    parser.add_argument("--count-level", type=str, default="1e9", help="Count level suffix.")
    parser.add_argument("--ds", type=float, default=1.0, help="Downsampling ratio in (0, 1].")
    parser.add_argument("--ene-resolution-662keV", type=float, default=0.1, help="Reference energy resolution.")
    parser.add_argument("--pixel-num-layer", type=int, default=1160, help="Legacy fallback layer pixel count.")
    parser.add_argument("--pixel-num-z", type=int, default=20, help="Axial slice count.")
    parser.add_argument("--rotate-num", type=int, default=10, help="Number of views.")
    parser.add_argument(
        "--delta-r1",
        type=float,
        default=0.0,
        help="Additional isotropic position sigma for the first interaction, in mm, on top of crystal-size modeling.",
    )
    parser.add_argument(
        "--delta-r2",
        type=float,
        default=0.0,
        help="Additional isotropic position sigma for the second interaction, in mm, on top of crystal-size modeling.",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="JSCC weighting parameter.")
    parser.add_argument("--sc-iter", type=int, default=4000, help="SC iteration count.")
    parser.add_argument("--jsccd-iter", type=int, default=2000, help="JSCC-D iteration count.")
    parser.add_argument("--jsccsd-iter", type=int, default=4000, help="JSCC-SD iteration count.")
    parser.add_argument("--save-iter-step", type=int, default=100, help="Save interval.")
    parser.add_argument("--osem-subset-num", type=int, default=8, help="OSEM subset count.")
    parser.add_argument("--t-divide-num", type=int, default=1, help="Number of t sub-blocks per subset.")
    parser.add_argument("--num-workers", type=int, default=1, help="Sub-chunks per GPU during list processing.")
    parser.add_argument("--seed", type=int, default=20260331, help="Random seed.")
    parser.add_argument("--save-t", action="store_true", help="Keep helper save-t branch enabled.")
    parser.add_argument("--save-s", action="store_true", help="Dump sensitivity_s.")
    parser.add_argument("--save-d", action="store_true", help="Dump sensitivity_d recomputed from t.")
    parser.add_argument("--factors-dir", type=str, default="./Factors", help="Factors root directory.")
    parser.add_argument("--cntstat-dir", type=str, default="./CntStat", help="CntStat root directory.")
    parser.add_argument("--list-dir", type=str, default="./List", help="List root directory.")
    parser.add_argument("--output-root", type=str, default="./Figure", help="Output root directory.")
    return parser.parse_args()


def validate_args(args):
    if len(args.e0_list) != len(args.ene_threshold_sum_list):
        raise ValueError("--e0-list and --ene-threshold-sum-list must have the same length.")
    if len(args.e0_list) != len(args.intensity_list):
        raise ValueError("--e0-list and --intensity-list must have the same length.")
    if not (0.0 < args.ds <= 1.0):
        raise ValueError("--ds must be in (0, 1].")
    if min(args.sc_iter, args.jsccd_iter, args.jsccsd_iter, args.save_iter_step) <= 0:
        raise ValueError("Iteration counts and --save-iter-step must be positive.")
    if args.sc_iter % args.save_iter_step != 0:
        raise ValueError("--sc-iter must be divisible by --save-iter-step.")
    if args.jsccd_iter % args.save_iter_step != 0:
        raise ValueError("--jsccd-iter must be divisible by --save-iter-step.")
    if args.jsccsd_iter % args.save_iter_step != 0:
        raise ValueError("--jsccsd-iter must be divisible by --save-iter-step.")
    if args.osem_subset_num <= 0 or args.t_divide_num <= 0 or args.num_workers <= 0:
        raise ValueError("--osem-subset-num, --t-divide-num and --num-workers must be positive.")
    if args.rotate_num <= 0 or args.pixel_num_z <= 0 or args.pixel_num_layer <= 0:
        raise ValueError("--rotate-num, --pixel-num-layer and --pixel-num-z must be positive.")


def resolve_repo_root():
    return Path(__file__).resolve().parent


def resolve_factor_dir(factors_root, e0, rotate_num):
    rotate_specific = factors_root / f"{round(1000 * e0)}keV_RotateNum{rotate_num}"
    legacy = factors_root / f"{round(1000 * e0)}keV"
    if rotate_specific.is_dir():
        return rotate_specific
    if legacy.is_dir():
        return legacy
    raise FileNotFoundError(f"Factor directory not found for {e0:.3f} MeV under {factors_root}")


def resolve_proj_and_list_paths(cntstat_root, list_root, e0, rotate_num, data_file_name, count_level):
    energy_tag = f"{round(1000 * e0)}keV"

    proj_candidates = [
        cntstat_root / f"{energy_tag}_RotateNum{rotate_num}" / f"CntStat_{data_file_name}_{count_level}.csv",
        cntstat_root / f"CntStat_{data_file_name}_{energy_tag}_{count_level}.csv",
        cntstat_root / f"CntStat_{data_file_name}_{count_level}.csv",
    ]
    list_candidates = [
        list_root / f"{energy_tag}_RotateNum{rotate_num}" / f"List_{data_file_name}_{count_level}",
        list_root / f"List_{data_file_name}_{energy_tag}_{count_level}",
        list_root / f"List_{data_file_name}_{count_level}",
    ]

    proj_path = next((path for path in proj_candidates if path.exists()), None)
    list_dir = next((path for path in list_candidates if path.is_dir()), None)

    if proj_path is None:
        raise FileNotFoundError(f"CntStat file not found. Tried: {proj_candidates}")
    if list_dir is None:
        raise FileNotFoundError(f"List directory not found. Tried: {list_candidates}")
    return proj_path, list_dir


def resolve_pixel_num(pixel_num_from_args, pixel_num_z, rotmat, rotmat_inv, coor_polar, factor_dir):
    pixel_num_from_rotmat = rotmat.size(0)
    pixel_num_from_rotmat_inv = rotmat_inv.size(0)
    pixel_num_from_coor = coor_polar.size(0)

    if pixel_num_from_rotmat != pixel_num_from_rotmat_inv:
        raise ValueError(
            f"Rotation matrix mismatch in {factor_dir}: "
            f"rotmat rows={pixel_num_from_rotmat}, rotmat_inv rows={pixel_num_from_rotmat_inv}."
        )
    if pixel_num_from_rotmat != pixel_num_from_coor:
        raise ValueError(
            f"Factor pixel mismatch in {factor_dir}: "
            f"rotmat rows={pixel_num_from_rotmat}, coor rows={pixel_num_from_coor}."
        )

    if pixel_num_from_rotmat != pixel_num_from_args:
        inferred_layer_msg = ""
        if pixel_num_z > 0 and pixel_num_from_rotmat % pixel_num_z == 0:
            inferred_layer = pixel_num_from_rotmat // pixel_num_z
            inferred_layer_msg = f", inferred pixel_num_layer={inferred_layer}"
        print(
            "Pixel count mismatch detected. "
            f"Using factor-defined pixel_num={pixel_num_from_rotmat} instead of "
            f"args pixel_num_layer * pixel_num_z = {pixel_num_from_args}{inferred_layer_msg}."
        )

    return pixel_num_from_rotmat


def split_tensor_rows(tensor, parts):
    if parts <= 1:
        return [tensor]
    splits = []
    total_rows = tensor.size(0)
    for idx in range(parts):
        start = total_rows * idx // parts
        end = total_rows * (idx + 1) // parts
        splits.append(tensor[start:end, :])
    return splits


def load_list_csv(list_file_path):
    list_np = np.genfromtxt(list_file_path, delimiter=",", dtype=np.float32)
    if list_np.ndim == 1:
        list_np = np.expand_dims(list_np, axis=0)
    return torch.from_numpy(list_np[:, 0:4])


def downsample_projection_and_list(proj, list_origin, ds):
    if ds >= 0.999999:
        return proj, list_origin

    proj = proj.clone()
    list_origin = [chunk.clone() for chunk in list_origin]

    for rotate_idx in range(proj.size(1)):
        proj_tmp = proj[:, rotate_idx]
        proj_ds_tmp = torch.zeros_like(proj_tmp)
        proj_indices = torch.tensor(
            [idx for idx in range(proj_tmp.size(0)) for _ in range(round(proj_tmp[idx].item()))],
            dtype=torch.long,
        )
        if proj_indices.numel() > 0:
            selected_num = int(torch.round(proj_tmp.sum() * ds).item())
            selected_num = min(selected_num, proj_indices.numel())
            selected_indices = torch.randperm(proj_indices.numel())[:selected_num]
            proj_indices_ds = proj_indices[selected_indices]
            for bin_idx in range(proj_ds_tmp.size(0)):
                proj_ds_tmp[bin_idx] = (proj_indices_ds == bin_idx).sum()
        proj[:, rotate_idx] = proj_ds_tmp

        list_tmp = list_origin[rotate_idx]
        if list_tmp.size(0) > 0:
            selected_num = int(list_tmp.size(0) * ds)
            selected_num = min(max(selected_num, 1), list_tmp.size(0))
            selected_indices = torch.randperm(list_tmp.size(0))[:selected_num]
            list_origin[rotate_idx] = list_tmp[selected_indices, :]

    return proj, list_origin


def process_list_on_single_gpu(
    sysmat,
    detector,
    coor_polar,
    list_origin,
    rotate_num,
    pixel_num,
    num_workers,
    delta_r1,
    delta_r2,
    e0,
    ene_resolution,
    ene_threshold_max,
    ene_threshold_min,
    ene_threshold_sum,
    start_time,
    model_compton_generator,
):
    device = torch.device("cuda:0")
    sysmat_gpu = sysmat.to(device)
    detector_gpu = detector.to(device)
    coor_polar_gpu = coor_polar.to(device)
    if model_compton_generator is not None:
        model_compton_generator = model_compton_generator.to(device)

    t = []
    size_t = 0
    compton_event_count_list = torch.zeros(size=[rotate_num, 1], dtype=torch.int64)

    for rotate_idx in range(rotate_num):
        t_parts = []
        for sub_chunk in split_tensor_rows(list_origin[rotate_idx], num_workers):
            if sub_chunk.numel() == 0:
                continue
            t_chunk, _, _ = get_compton_backproj_list_single(
                sysmat_gpu,
                detector_gpu,
                coor_polar_gpu,
                sub_chunk.to(device),
                delta_r1,
                delta_r2,
                e0,
                ene_resolution,
                ene_threshold_max,
                ene_threshold_min,
                ene_threshold_sum,
                device,
                model_compton_generator,
            )
            t_parts.append(t_chunk)
            torch.cuda.empty_cache()
            print(f"Single GPU: processed rotate {rotate_idx + 1} sub-chunk, time used: {time.time() - start_time:.2f}s")

        if t_parts:
            t_tmp = torch.cat(t_parts, dim=0)
        else:
            t_tmp = torch.empty((0, pixel_num), dtype=torch.float32)

        compton_event_count_list[rotate_idx] = t_tmp.size(0)
        size_t += t_tmp.nelement() * t_tmp.element_size()
        t.append(t_tmp)
        print(f"Rotate Num {rotate_idx + 1} ends, time used: {time.time() - start_time:.2f}s")

    return t, size_t, compton_event_count_list


def process_list_on_multi_gpu(
    num_gpus,
    sysmat,
    detector,
    coor_polar,
    list_origin,
    rotate_num,
    pixel_num,
    num_workers,
    delta_r1,
    delta_r2,
    e0,
    ene_resolution,
    ene_threshold_max,
    ene_threshold_min,
    ene_threshold_sum,
    start_time,
    flag_save_t,
    model_compton_generator,
):
    t = []
    size_t = 0
    compton_event_count_list = torch.zeros(size=[rotate_num, 1], dtype=torch.int64)

    for rotate_idx in range(rotate_num):
        chunks = split_tensor_rows(list_origin[rotate_idx], num_gpus)

        with Manager() as manager:
            result_dict = manager.dict()
            processes = []

            for rank, chunk in enumerate(chunks):
                if chunk.numel() == 0:
                    result_dict[rank] = torch.empty((0, pixel_num), dtype=torch.float32)
                    continue

                process = mp.Process(
                    target=get_compton_backproj_list_mp,
                    args=(
                        rank,
                        num_gpus,
                        sysmat,
                        detector,
                        coor_polar,
                        chunk,
                        delta_r1,
                        delta_r2,
                        e0,
                        ene_resolution,
                        ene_threshold_max,
                        ene_threshold_min,
                        ene_threshold_sum,
                        result_dict,
                        num_workers,
                        start_time,
                        flag_save_t,
                        model_compton_generator,
                    ),
                )
                process.start()
                processes.append(process)

            for process in processes:
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(f"List worker exited with code {process.exitcode} on rotate {rotate_idx + 1}.")

            t_results = []
            for rank in range(num_gpus):
                if rank not in result_dict:
                    raise RuntimeError(f"Missing result from rank {rank} on rotate {rotate_idx + 1}.")
                t_results.append(result_dict[rank])
                print(f"Collected result from rank {rank}")

        t_tmp = torch.cat(t_results, dim=0) if t_results else torch.empty((0, pixel_num), dtype=torch.float32)
        compton_event_count_list[rotate_idx] = t_tmp.size(0)
        size_t += t_tmp.nelement() * t_tmp.element_size()
        t.append(t_tmp)
        print(f"Rotate Num {rotate_idx + 1} ends, time used: {time.time() - start_time:.2f}s")

    return t, size_t, compton_event_count_list


def build_save_path(output_root, e0_list, rotate_num, data_file_name, count_level, ds, s_map_d_ratio, delta_r1, alpha, ene_resolution_662keV, osem_subset_num, jsccsd_iter, single_event_count_total, compton_event_count_total):
    if len(e0_list) == 1:
        prefix = f"SingleEnergy_RotateNum{rotate_num}_{data_file_name}_{round(1000 * e0_list[0])}keV"
    else:
        e0_list_str = "_".join(str(round(e0 * 1000)) for e0 in e0_list)
        prefix = f"MultiEnergy_RotateNum{rotate_num}_{data_file_name}_({e0_list_str})keV"

    return (
        output_root
        / f"{prefix}_{count_level}_{ds}_SMap{s_map_d_ratio}_Delta{delta_r1}_Alpha{alpha}_ER{ene_resolution_662keV}_"
        f"OSEM{osem_subset_num}_ITER{jsccsd_iter}_SDU{single_event_count_total}_DDU{compton_event_count_total}"
        / "Polar"
    )


def main():
    args = parse_args()
    validate_args(args)

    repo_root = resolve_repo_root()
    factors_root = (repo_root / args.factors_dir).resolve()
    cntstat_root = (repo_root / args.cntstat_dir).resolve()
    list_root = (repo_root / args.list_dir).resolve()
    output_root = (repo_root / args.output_root).resolve()

    start_time = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_filename = None
    logfile = None
    save_path = None
    original_stdout = sys.stdout

    try:
        rand_suffix = f"{random.randint(0, 9999):04d}"
        log_filename = repo_root / f"print_log_{rand_suffix}.txt"
        logfile = open(log_filename, "w", encoding="utf-8")
        sys.stdout = Tee(original_stdout, logfile)

        print("=======================================")
        print("--------Step1: Checking Devices--------")

        available_gpus = torch.cuda.device_count()
        if available_gpus <= 0 or not torch.cuda.is_available():
            raise RuntimeError("main_plane.py requires CUDA. No available GPU was detected.")

        if args.num_gpus <= 0:
            raise ValueError("--num-gpus must be positive when CUDA is available.")

        args.num_gpus = min(args.num_gpus, available_gpus)
        print(f"CUDA is available, using {args.num_gpus} / {available_gpus} GPUs")
        print(f"Repo Root: {repo_root}")
        print(f"Args: {args}")

        proj_all = []
        list_all = []
        sysmat_all = []
        detector_all = []
        coor_polar_all = []
        rotmat_all = []
        rotmat_inv_all = []
        sensi_s_all = []
        sensi_d_all = []
        e_params = []

        pixel_num = None
        print("====================================")
        print("--------Step2: Loading Files--------")

        for e0, ene_threshold_sum, intensity in zip(args.e0_list, args.ene_threshold_sum_list, args.intensity_list):
            ene_resolution = args.ene_resolution_662keV * (0.662 / e0) ** 0.5
            ene_threshold_max = 2 * e0 ** 2 / (0.511 + 2 * e0) - 0.001
            ene_threshold_min = 0.05

            factor_dir = resolve_factor_dir(factors_root, e0, args.rotate_num)
            proj_file_path, list_dir_path = resolve_proj_and_list_paths(
                cntstat_root, list_root, e0, args.rotate_num, args.data_file_name, args.count_level
            )

            sysmat_file_path = factor_dir / "SysMat_polar"
            detector_file_path = factor_dir / "Detector.csv"
            sensi_s_file_path = factor_dir / "Sensi_s"
            sensi_d_file_path = factor_dir / "Sensi_d"
            coor_polar_file_path = factor_dir / "coor_polar_full.csv"
            rotmat_file_path = factor_dir / "RotMat_full.csv"
            rotmat_inv_file_path = factor_dir / "RotMatInv_full.csv"

            for required_path in (
                sysmat_file_path,
                detector_file_path,
                coor_polar_file_path,
                rotmat_file_path,
                rotmat_inv_file_path,
                proj_file_path,
            ):
                if not required_path.exists():
                    raise FileNotFoundError(required_path)

            detector = torch.from_numpy(np.genfromtxt(detector_file_path, delimiter=",", dtype=np.float32)[:, 1:4])
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

            sysmat_np = np.fromfile(sysmat_file_path, dtype=np.float32)
            if sysmat_np.size % pixel_num != 0:
                raise ValueError(f"SysMat size in {sysmat_file_path} is incompatible with pixel_num={pixel_num}.")
            sysmat = torch.from_numpy(sysmat_np.reshape(pixel_num, -1).T.copy()) * intensity

            if sensi_s_file_path.exists():
                sensi_s = torch.from_numpy(np.fromfile(sensi_s_file_path, dtype=np.float32).reshape(pixel_num, 1).copy()) * intensity
                sensi_s_all.append(sensi_s)

            if sensi_d_file_path.exists():
                sensi_d = torch.from_numpy(np.fromfile(sensi_d_file_path, dtype=np.float32).reshape(pixel_num, 1).copy()) * intensity
                sensi_d_all.append(sensi_d)

            proj_np = np.genfromtxt(proj_file_path, delimiter=",", dtype=np.float32)
            proj = torch.from_numpy(proj_np.reshape(args.rotate_num, -1).T.copy())

            list_origin = []
            for rotate_idx in range(args.rotate_num):
                list_file_path = list_dir_path / f"{rotate_idx + 1}.csv"
                if not list_file_path.exists():
                    raise FileNotFoundError(list_file_path)
                list_origin.append(load_list_csv(list_file_path))

            proj, list_origin = downsample_projection_and_list(proj, list_origin, args.ds * intensity)

            proj_all.append(proj)
            list_all.append(list_origin)
            sysmat_all.append(sysmat)
            detector_all.append(detector)
            coor_polar_all.append(coor_polar)
            rotmat_all.append(rotmat)
            rotmat_inv_all.append(rotmat_inv)
            e_params.append((e0, ene_resolution, ene_threshold_max, ene_threshold_min, ene_threshold_sum))

            print(f"Loaded energy {e0:.3f} MeV from {factor_dir.name}")

        print("==================================================")
        print("--------Step3: Processing List (Multi-GPU)--------")
        t_all = []
        proj_d_all = []
        single_event_count_total = 0
        compton_event_count_total = 0

        s_map_s_total = torch.zeros([pixel_num, 1], dtype=torch.float32)
        s_map_d_total = torch.zeros([pixel_num, 1], dtype=torch.float32)

        iter_arg = argparse.Namespace()
        iter_arg.sc = args.sc_iter
        iter_arg.jsccd = args.jsccd_iter
        iter_arg.jsccsd = args.jsccsd_iter
        iter_arg.admm_inner_single = 1
        iter_arg.admm_inner_compton = 1
        iter_arg.mode = 0
        iter_arg.save_iter_step = args.save_iter_step
        iter_arg.osem_subset_num = args.osem_subset_num
        iter_arg.t_divide_num = args.t_divide_num
        iter_arg.ene_num = len(args.e0_list)
        iter_arg.event_level = 2
        iter_arg.num_workers = args.num_workers
        iter_arg.seed = args.seed

        s_map_arg = argparse.Namespace()

        model_denoiser = None
        model_compton_generator = None

        for proj, list_origin, sysmat, detector, coor_polar, rotmat_inv, e_param in zip(
            proj_all,
            list_all,
            sysmat_all,
            detector_all,
            coor_polar_all,
            rotmat_inv_all,
            e_params,
        ):
            e0, ene_resolution, ene_threshold_max, ene_threshold_min, ene_threshold_sum = e_param
            print(f"Processing energy {e0:.3f} MeV ...")

            if args.num_gpus == 1:
                t, size_t, compton_event_count_list = process_list_on_single_gpu(
                    sysmat,
                    detector,
                    coor_polar,
                    list_origin,
                    args.rotate_num,
                    pixel_num,
                    iter_arg.num_workers,
                    args.delta_r1,
                    args.delta_r2,
                    e0,
                    ene_resolution,
                    ene_threshold_max,
                    ene_threshold_min,
                    ene_threshold_sum,
                    start_time,
                    model_compton_generator,
                )
            else:
                t, size_t, compton_event_count_list = process_list_on_multi_gpu(
                    args.num_gpus,
                    sysmat,
                    detector,
                    coor_polar,
                    list_origin,
                    args.rotate_num,
                    pixel_num,
                    iter_arg.num_workers,
                    args.delta_r1,
                    args.delta_r2,
                    e0,
                    ene_resolution,
                    ene_threshold_max,
                    ene_threshold_min,
                    ene_threshold_sum,
                    start_time,
                    int(args.save_t),
                    model_compton_generator,
                )

            proj_d = torch.zeros_like(proj)
            for rotate_idx in range(args.rotate_num):
                proj_tmp = proj[:, rotate_idx]
                proj_indices = torch.tensor(
                    [idx for idx in range(proj_tmp.size(0)) for _ in range(round(proj_tmp[idx].item()))],
                    dtype=torch.long,
                )
                if proj_indices.numel() == 0:
                    continue
                selected_num = min(int(compton_event_count_list[rotate_idx].item()), proj_indices.numel())
                selected_indices = torch.randperm(proj_indices.numel())[:selected_num]
                proj_d_index_tmp = proj_indices[selected_indices]
                for bin_idx in range(proj_d.size(0)):
                    proj_d[bin_idx, rotate_idx] = (proj_d_index_tmp == bin_idx).sum()

            single_event_count = round(proj.sum().item())
            compton_event_count = round(proj_d.sum().item())
            print(f"[Energy {e0:.3f}] Single events = {single_event_count}, Compton events = {compton_event_count}")
            print(f"[Energy {e0:.3f}] The size of t is {size_t / (1024 ** 3):.2f} GB")

            t_all.append(t)
            proj_d_all.append(proj_d)

            s_map_s_tmp = torch.zeros([1, pixel_num], dtype=torch.float32)
            for rotate_idx in range(args.rotate_num):
                rotmat_inv_tmp = rotmat_inv[:, rotate_idx]
                s_map_s_tmp += torch.sum(sysmat[:, rotmat_inv_tmp - 1], dim=0, keepdim=True).cpu()
            s_map_s_tmp = s_map_s_tmp.transpose(0, 1) / args.rotate_num
            s_map_s_total += s_map_s_tmp

            if single_event_count > 0:
                s_map_d_total += s_map_s_tmp * (compton_event_count / single_event_count)

            single_event_count_total += single_event_count
            compton_event_count_total += compton_event_count

        if sensi_s_all:
            print("sensi_s change to file definition")
            s_map_arg.s = sum(sensi_s_all)
        else:
            s_map_arg.s = s_map_s_total

        if sensi_d_all:
            print("sensi_d change to file definition")
            s_map_arg.d = sum(sensi_d_all) * args.s_map_d_ratio
        else:
            s_map_arg.d = s_map_d_total

        print("===========================================")
        print("--------Step4: Image Reconstruction--------")

        if args.save_s:
            with open(repo_root / "sensitivity_s", "wb") as file:
                s_map_arg.s.cpu().numpy().astype("float32").tofile(file)

        if args.save_d:
            sensi_d = torch.zeros_like(s_map_arg.s)
            for t in t_all:
                sensi_d_tmp = torch.sum(torch.cat(t, dim=0), dim=0, keepdim=True).transpose(0, 1)
                sensi_d_tmp = (
                    sensi_d_tmp
                    * torch.sum(s_map_arg.s)
                    / torch.sum(sensi_d_tmp)
                    * compton_event_count_total
                    / max(single_event_count_total, 1)
                )
                sensi_d += sensi_d_tmp
            with open(repo_root / "Sensi_d", "wb") as file:
                sensi_d.cpu().numpy().astype("float32").tofile(file)

        torch.cuda.empty_cache()

        save_path = build_save_path(
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
        save_path.mkdir(parents=True, exist_ok=True)

        run_recon_osem(
            sysmat_all,
            rotmat_all,
            rotmat_inv_all,
            proj_all,
            proj_d_all,
            t_all,
            iter_arg,
            s_map_arg,
            args.alpha,
            str(save_path) + os.sep,
            args.num_gpus,
            model_denoiser,
        )

        print(f"\nTotal time used: {time.time() - start_time:.2f}s")

    finally:
        sys.stdout = original_stdout
        if logfile is not None:
            logfile.close()

        if log_filename is not None:
            if save_path is not None and save_path.is_dir():
                final_log_name = save_path / "print_log.txt"
                shutil.move(str(log_filename), final_log_name)
            elif Path(log_filename).exists():
                print(f"Log kept at {log_filename}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    with torch.no_grad():
        main()
