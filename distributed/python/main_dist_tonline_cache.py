import argparse
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

try:
    from distributed.python._path_setup import setup_repo_root
except ImportError:
    from _path_setup import setup_repo_root

setup_repo_root()
from distributed.python.recon_osem_dist_tonline_cache import run_recon_osem_dist_tonline_cache
from sparse_main_utils import build_output_name_prefix, format_scientific_count
from distributed.python.t_online_cache_dist import (
    ComptonChunkCache,
    OnlineComptonProvider,
    build_list_chunk_plan,
    load_generation_factors,
)


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
    parser = argparse.ArgumentParser(description="Distributed reconstruction with online t generation and small cache.")
    parser.add_argument("--e0-list", type=float, nargs="+", default=[0.511])
    parser.add_argument("--ene-threshold-sum-list", type=float, nargs="+", default=[0.46])
    parser.add_argument("--intensity-list", type=float, nargs="+", default=[1.0])
    parser.add_argument("--s-map-d-ratio", type=float, default=1.0)
    parser.add_argument("--data-file-name", type=str, default="ContrastPhantom_240_30")
    parser.add_argument("--count-level", type=str, default="5e10")
    parser.add_argument("--ds", type=float, default=1.0)
    parser.add_argument("--pixel-num-layer", type=int, default=1280)
    parser.add_argument("--pixel-num-z", type=int, default=20)
    parser.add_argument("--rotate-num", type=int, default=20)
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
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ene-resolution-662keV", type=float, default=0.1)
    parser.add_argument("--sc-iter", type=int, default=20000)
    parser.add_argument("--jsccd-iter", type=int, default=10000)
    parser.add_argument("--jsccsd-iter", type=int, default=20000)
    parser.add_argument("--save-iter-step", type=int, default=100)
    parser.add_argument("--osem-subset-num", type=int, default=8)
    parser.add_argument("--list-chunk-events", type=int, default=512)
    parser.add_argument("--t-cache-size-gb", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260331)
    parser.add_argument("--factors-dir", type=str, default="./Factors")
    parser.add_argument("--cntstat-dir", type=str, default="./CntStat")
    parser.add_argument("--list-dir", type=str, default="./List")
    parser.add_argument("--output-root", type=str, default="./Figure_Dist_TOnlineCache")
    return parser.parse_args()


def validate_args(args):
    lengths = {len(args.e0_list), len(args.ene_threshold_sum_list), len(args.intensity_list)}
    if len(lengths) != 1:
        raise ValueError("Energy-related lists must have the same length.")

    if not (0 < args.ds <= 1):
        raise ValueError("--ds must be in (0, 1].")

    if args.t_cache_size_gb < 0:
        raise ValueError("--t-cache-size-gb must be non-negative.")

    for name in ("sc_iter", "jsccd_iter", "jsccsd_iter", "save_iter_step", "osem_subset_num", "list_chunk_events"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")

    for iter_name in ("sc_iter", "jsccd_iter", "jsccsd_iter"):
        if getattr(args, iter_name) % args.save_iter_step != 0:
            raise ValueError(f"--{iter_name.replace('_', '-')} must be divisible by --save-iter-step.")


def downsample_projection(full_proj, ds, seed):
    if ds >= 0.999999:
        return full_proj

    full_proj_np = np.rint(full_proj.numpy()).astype(np.int64)
    rng = np.random.default_rng(seed)
    proj_ds_np = rng.binomial(full_proj_np, ds).astype(np.float32)
    return torch.from_numpy(proj_ds_np)


def downsample_list(local_list, ds, seed):
    if ds >= 0.999999 or local_list.size(0) == 0:
        return local_list

    keep_num = max(1, int(local_list.size(0) * ds))
    generator = torch.Generator()
    generator.manual_seed(seed)
    keep_ids = torch.randperm(local_list.size(0), generator=generator)[:keep_num]
    return local_list[keep_ids, :]


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


def compute_global_sensitivity(sysmat_local, rotmat_inv, rotate_num, device):
    sensi_local = torch.zeros([1, sysmat_local.size(1)], dtype=torch.float32)
    for rotate_idx in range(rotate_num):
        sensi_local += torch.sum(sysmat_local[:, rotmat_inv[:, rotate_idx] - 1], dim=0, keepdim=True).cpu()

    sensi_local = sensi_local.transpose(0, 1).to(device)
    dist.all_reduce(sensi_local, op=dist.ReduceOp.SUM)
    return sensi_local / rotate_num


def load_local_list_partition(list_root, rotate_num, world_size, global_rank, ds, seed_offset):
    list_local = []
    for rotate_idx in range(rotate_num):
        file_path = os.path.join(list_root, f"{rotate_idx + 1}.csv")
        full_list = torch.from_numpy(np.genfromtxt(file_path, delimiter=",", dtype=np.float32)[:, 0:4])
        ev_per_rank = full_list.size(0) // world_size
        ev_start = global_rank * ev_per_rank
        ev_end = (global_rank + 1) * ev_per_rank if global_rank != world_size - 1 else full_list.size(0)
        local_chunk = full_list[ev_start:ev_end, :]
        local_chunk = downsample_list(local_chunk, ds, seed_offset + rotate_idx)
        list_local.append(local_chunk)
    return list_local


def build_save_path(args, single_event_count_total, compton_event_count_total):
    name_prefix = build_output_name_prefix(args.e0_list, args.rotate_num, args.data_file_name)
    single_event_count_str = format_scientific_count(single_event_count_total)
    compton_event_count_str = format_scientific_count(compton_event_count_total)

    return (
        f"{args.output_root}/{name_prefix}_{args.count_level}_{args.ds}_SMap{args.s_map_d_ratio}_"
        f"Delta{args.delta_r1}_Alpha{args.alpha}_ER{args.ene_resolution_662keV}_"
        f"OSEM{args.osem_subset_num}_ITER{args.jsccsd_iter}_SDU{single_event_count_str}_"
        f"DDU{compton_event_count_str}/Polar/"
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
            log_filename = f"print_log_dist_tonline_cache_{rand_suffix}.txt"
            logfile = open(log_filename, "w", encoding="utf-8")
            sys.stdout = Tee(sys.__stdout__, logfile)
            print(f"Distributed initialized: world_size={world_size}, local_rank={local_rank}")
            print(f"Args: {args}")

        pixel_num = args.pixel_num_layer * args.pixel_num_z
        cache_bytes = int(args.t_cache_size_gb * (1024 ** 3))

        proj_local_all = []
        proj_d_local_all = []
        sysmat_local_all = []
        rotmat_all = []
        rotmat_inv_all = []
        sensi_s_all = []
        sensi_d_all = []
        t_online_plan_all = [[None for _ in range(len(args.e0_list))] for _ in range(args.osem_subset_num)]
        providers = []

        single_event_count_total = torch.zeros(1, dtype=torch.float64, device=device)
        compton_event_count_total = torch.zeros(1, dtype=torch.float64, device=device)

        for energy_idx, (e0, ene_threshold_sum, intensity) in enumerate(
            zip(args.e0_list, args.ene_threshold_sum_list, args.intensity_list)
        ):
            ene_resolution = args.ene_resolution_662keV * (0.662 / e0) ** 0.5
            ene_threshold_max = 2 * e0 ** 2 / (0.511 + 2 * e0) - 0.001
            ene_threshold_min = 0.05

            factor_path = os.path.join(args.factors_dir, f"{round(1000 * e0)}keV_RotateNum{args.rotate_num}")
            sysmat_file_path = os.path.join(factor_path, "SysMat_polar")
            rotmat_file_path = os.path.join(factor_path, "RotMat_full.csv")
            rotmat_inv_file_path = os.path.join(factor_path, "RotMatInv_full.csv")
            sensi_d_file_path = os.path.join(factor_path, "Sensi_d")
            proj_file_path = os.path.join(
                args.cntstat_dir,
                f"{round(1000 * e0)}keV_RotateNum{args.rotate_num}",
                f"CntStat_{args.data_file_name}_{args.count_level}.csv",
            )
            list_root = os.path.join(
                args.list_dir,
                f"{round(1000 * e0)}keV_RotateNum{args.rotate_num}",
                f"List_{args.data_file_name}_{args.count_level}",
            )

            for required_path in (sysmat_file_path, rotmat_file_path, rotmat_inv_file_path, proj_file_path, list_root):
                if not os.path.exists(required_path):
                    raise FileNotFoundError(required_path)

            sysmat_local, total_bins, idx_start, idx_end = load_local_sysmat(
                sysmat_file_path, pixel_num, intensity, global_rank, world_size
            )
            proj_full = torch.from_numpy(
                np.genfromtxt(proj_file_path, delimiter=",", dtype=np.float32).reshape(args.rotate_num, -1)
            ).transpose(0, 1)
            if proj_full.size(0) != total_bins:
                raise ValueError(
                    f"Projection bin count mismatch for {proj_file_path}: proj={proj_full.size(0)}, sysmat={total_bins}."
                )

            proj_full = downsample_projection(proj_full, args.ds, args.seed + energy_idx)
            proj_local = proj_full[idx_start:idx_end, :]
            list_local = load_local_list_partition(
                list_root, args.rotate_num, world_size, global_rank, args.ds, args.seed + energy_idx * 1000
            )

            rotmat = torch.from_numpy(np.genfromtxt(rotmat_file_path, delimiter=",", dtype=np.int64))
            rotmat_inv = torch.from_numpy(np.genfromtxt(rotmat_inv_file_path, delimiter=",", dtype=np.int64))
            sensi_s = compute_global_sensitivity(sysmat_local, rotmat_inv, args.rotate_num, device)

            sysmat_local_all.append(sysmat_local)
            proj_local_all.append(proj_local)
            rotmat_all.append(rotmat.to(device))
            rotmat_inv_all.append(rotmat_inv.to(device))
            sensi_s_all.append(sensi_s)

            if os.path.exists(sensi_d_file_path):
                sensi_d = torch.from_numpy(
                    np.reshape(np.fromfile(sensi_d_file_path, dtype=np.float32), [pixel_num, 1])
                ) * intensity
                sensi_d_all.append(sensi_d)

            cache = ComptonChunkCache(cache_bytes)
            sysmat_full_gpu, detector_gpu, coor_polar_gpu = load_generation_factors(factor_path, pixel_num, intensity, device)
            provider = OnlineComptonProvider(
                device=device,
                sysmat_full_gpu=sysmat_full_gpu,
                detector_gpu=detector_gpu,
                coor_polar_gpu=coor_polar_gpu,
                delta_r1=args.delta_r1,
                delta_r2=args.delta_r2,
                e0=e0,
                ene_resolution=ene_resolution,
                ene_threshold_max=ene_threshold_max,
                ene_threshold_min=ene_threshold_min,
                ene_threshold_sum=ene_threshold_sum,
                cache=cache,
            )
            providers.append(provider)

            subset_plan = build_list_chunk_plan(
                list_local=list_local,
                osem_subset_num=args.osem_subset_num,
                list_chunk_events=args.list_chunk_events,
                seed=args.seed,
                global_rank=global_rank,
                energy_keV=round(e0 * 1000),
            )
            for subset_idx in range(args.osem_subset_num):
                t_online_plan_all[subset_idx][energy_idx] = subset_plan[subset_idx]

            rows_local_per_rotate = torch.tensor(provider.count_rows_per_rotate(subset_plan), dtype=torch.float32, device=device)
            rows_global_per_rotate = rows_local_per_rotate.clone()
            single_rows_global_per_rotate = proj_local.sum(dim=0).to(device)
            dist.all_reduce(rows_global_per_rotate, op=dist.ReduceOp.SUM)
            dist.all_reduce(single_rows_global_per_rotate, op=dist.ReduceOp.SUM)

            proj_ratio = torch.where(
                single_rows_global_per_rotate > 0,
                rows_global_per_rotate / single_rows_global_per_rotate.clamp_min(1e-12),
                torch.zeros_like(rows_global_per_rotate),
            )
            proj_d_local_all.append(proj_local * proj_ratio.cpu().unsqueeze(0))

            single_event_count_total += proj_local.sum(dtype=torch.float64).to(device)
            compton_event_count_total += rows_local_per_rotate.sum(dtype=torch.float64).to(device)

            if global_rank == 0:
                stats = cache.stats()
                print(
                    f"Loaded energy {e0:.3f} MeV | bins=[{idx_start}, {idx_end}) | "
                    f"local single={proj_local.sum().item():.0f} | local compton={rows_local_per_rotate.sum().item():.0f} | "
                    f"cache_entries={stats['entries']} | cache_gb={stats['bytes'] / (1024 ** 3):.2f}"
                )

            cache.reset_stats()

        dist.all_reduce(single_event_count_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(compton_event_count_total, op=dist.ReduceOp.SUM)

        s_map_arg = argparse.Namespace()
        s_map_arg.s = sum(sensi_s_all).to(device)
        if len(sensi_d_all) > 0:
            s_map_arg.d = sum(sensi_d_all).to(device) * args.s_map_d_ratio
        else:
            s_map_arg.d = s_map_arg.s * args.s_map_d_ratio
        s_map_arg.j = args.alpha * s_map_arg.s + (2 - args.alpha) * s_map_arg.d

        iter_arg = argparse.Namespace()
        iter_arg.sc = args.sc_iter
        iter_arg.jsccd = args.jsccd_iter
        iter_arg.jsccsd = args.jsccsd_iter
        iter_arg.save_iter_step = args.save_iter_step
        iter_arg.osem_subset_num = args.osem_subset_num
        iter_arg.ene_num = len(args.e0_list)
        iter_arg.seed = args.seed

        save_path = build_save_path(
            args,
            round(single_event_count_total.item()),
            round(compton_event_count_total.item()),
        )
        if global_rank == 0:
            os.makedirs(save_path, exist_ok=True)
        dist.barrier()

        run_recon_osem_dist_tonline_cache(
            sysmat_l_all=sysmat_local_all,
            rotmat_all=rotmat_all,
            rotmat_inv_all=rotmat_inv_all,
            proj_l_all=proj_local_all,
            proj_dl_all=proj_d_local_all,
            t_online_plan_all=t_online_plan_all,
            providers=providers,
            iter_arg=iter_arg,
            s_map_arg=s_map_arg,
            alpha=args.alpha,
            save_path=save_path,
        )

        if global_rank == 0:
            for energy_idx, provider in enumerate(providers):
                stats = provider.cache.stats()
                print(
                    f"[Cache][Energy {energy_idx}] hit_rate={stats['hit_rate']:.4f} "
                    f"hits={stats['hits']} misses={stats['misses']} entries={stats['entries']} "
                    f"cache_gb={stats['bytes'] / (1024 ** 3):.2f}"
                )
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
