import random
from pathlib import Path

import numpy as np
import torch


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


def format_scientific_count(value):
    value = float(value)
    if value == 0:
        return "0.00e0"

    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def build_output_name_prefix(e0_list, rotate_num, data_file_name):
    e0_list_str = "_".join(str(round(e0 * 1000)) for e0 in e0_list)
    if len(e0_list) == 1:
        return f"SE_RotNum{rotate_num}_{data_file_name}_{e0_list_str}keV"
    return f"ME_RotNum{rotate_num}_{data_file_name}_({e0_list_str})keV"


def build_save_path(
    output_root,
    e0_list,
    rotate_num,
    data_file_name,
    count_level,
    ds,
    s_map_d_ratio,
    delta_r1,
    alpha,
    ene_resolution_662keV,
    osem_subset_num,
    jsccsd_iter,
    single_event_count_total,
    compton_event_count_total,
):
    prefix = build_output_name_prefix(e0_list, rotate_num, data_file_name)
    single_event_count_str = format_scientific_count(single_event_count_total)
    compton_event_count_str = format_scientific_count(compton_event_count_total)

    return (
        output_root
        / f"{prefix}_{count_level}_{ds}_SMap{s_map_d_ratio}_Delta{delta_r1}_Alpha{alpha}_ER{ene_resolution_662keV}_"
        f"OSEM{osem_subset_num}_ITER{jsccsd_iter}_SDU{single_event_count_str}_DDU{compton_event_count_str}"
        / "Polar"
    )
