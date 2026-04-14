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
    split_tensor_rows,
    validate_args,
)
from process_list_plane_sparse import (
    get_compton_backproj_list_mp_sparse,
    get_compton_backproj_list_single_sparse,
)
from recon_osem_plane_sparse import run_recon_osem_sparse


def parse_args():
    parser = argparse.ArgumentParser(description="Local multi-GPU sparse-Compton JSCC reconstruction for plane geometry.")
    parser.add_argument("--num-gpus", type=int, default=torch.cuda.device_count(), help="Number of local GPUs to use.")
    parser.add_argument("--e0-list", type=float, nargs="+", default=[0.511], help="Energy list in MeV.")
    parser.add_argument("--ene-threshold-sum-list", type=float, nargs="+", default=[0.46], help="Lower bounds for e1 + e2 in MeV.")
    parser.add_argument("--intensity-list", type=float, nargs="+", default=[1.0], help="Intensity weights.")
    parser.add_argument("--s-map-d-ratio", type=float, default=1.0, help="Scale factor applied to sparse Sensi_d.")
    parser.add_argument("--use-file-sensi-d", action="store_true", help="Deprecated. Sparse chain now uses file-based Sensi_d by default when available.")
    parser.add_argument("--recompute-sparse-sensi-d", action="store_true", help="Force recomputing Sensi_d from sparse Compton operators instead of reusing factor-file Sensi_d.")
    parser.add_argument("--data-file-name", type=str, default="ContrastPhantom_240_30", help="Dataset name.")
    parser.add_argument("--count-level", type=str, default="1e9", help="Count level suffix.")
    parser.add_argument("--ds", type=float, default=1.0, help="Downsampling ratio in (0, 1].")
    parser.add_argument("--ene-resolution-662keV", type=float, default=0.1, help="Reference energy resolution.")
    parser.add_argument("--pixel-num-layer", type=int, default=1160, help="Legacy fallback layer pixel count.")
    parser.add_argument("--pixel-num-z", type=int, default=20, help="Axial slice count.")
    parser.add_argument("--rotate-num", type=int, default=10, help="Number of views.")
    parser.add_argument("--delta-r1", type=float, default=0.0, help="Additional isotropic position sigma for the first interaction in mm.")
    parser.add_argument("--delta-r2", type=float, default=0.0, help="Additional isotropic position sigma for the second interaction in mm.")
    parser.add_argument("--alpha", type=float, default=1.0, help="JSCC weighting parameter.")
    parser.add_argument("--sc-iter", type=int, default=4000, help="SC iteration count.")
    parser.add_argument("--jsccd-iter", type=int, default=2000, help="JSCC-D iteration count.")
    parser.add_argument("--jsccsd-iter", type=int, default=4000, help="JSCC-SD iteration count.")
    parser.add_argument("--save-iter-step", type=int, default=100, help="Save interval.")
    parser.add_argument("--osem-subset-num", type=int, default=8, help="OSEM subset count.")
    parser.add_argument("--t-divide-num", type=int, default=1, help="Number of t sub-blocks per subset.")
    parser.add_argument("--num-workers", type=int, default=1, help="Sub-chunks per GPU during list processing.")
    parser.add_argument("--compton-theta-stride", type=int, default=2, help="Angular stride used by sparse Compton grid.")
    parser.add_argument("--compton-z-stride", type=int, default=2, help="Axial stride used by sparse Compton grid.")
    parser.add_argument("--seed", type=int, default=20260331, help="Random seed.")
    parser.add_argument("--save-t", action="store_true", help="Keep helper save-t branch enabled.")
    parser.add_argument("--save-s", action="store_true", help="Dump sensitivity_s.")
    parser.add_argument("--save-d", action="store_true", help="Dump sparse sensitivity_d.")
    parser.add_argument("--factors-dir", type=str, default="./Factors", help="Factors root directory.")
    parser.add_argument("--cntstat-dir", type=str, default="./CntStat", help="CntStat root directory.")
    parser.add_argument("--list-dir", type=str, default="./List", help="List root directory.")
    parser.add_argument("--output-root", type=str, default="./Figure", help="Output root directory.")
    return parser.parse_args()


def process_list_on_single_gpu_sparse(
    sysmat,
    detector,
    sparse_projector,
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
):
    del pixel_num
    device = torch.device("cuda:0")
    sysmat_gpu = sysmat.to(device)
    detector_gpu = detector.to(device)
    sparse_projector_gpu = sparse_projector.to(device)

    t = []
    size_t = 0
    compton_event_count_list = torch.zeros(size=[rotate_num, 1], dtype=torch.int64)

    for rotate_idx in range(rotate_num):
        t_parts = []
        for sub_chunk in split_tensor_rows(list_origin[rotate_idx], num_workers):
            if sub_chunk.numel() == 0:
                continue
            t_chunk, _, _ = get_compton_backproj_list_single_sparse(
                sysmat_gpu,
                detector_gpu,
                sparse_projector_gpu,
                sub_chunk.to(device),
                delta_r1,
                delta_r2,
                e0,
                ene_resolution,
                ene_threshold_max,
                ene_threshold_min,
                ene_threshold_sum,
                device,
            )
            if t_chunk.numel() > 0:
                t_parts.append(t_chunk)
            torch.cuda.empty_cache()
            print(f"Single GPU sparse: processed rotate {rotate_idx + 1} sub-chunk, time used: {time.time() - start_time:.2f}s")

        t_tmp = torch.cat(t_parts, dim=0) if t_parts else torch.empty((0, sparse_projector.coarse_pixel_num + 1), dtype=torch.float32)
        compton_event_count_list[rotate_idx] = t_tmp.size(0)
        size_t += t_tmp.nelement() * t_tmp.element_size()
        t.append(t_tmp)
        print(f"Sparse rotate {rotate_idx + 1} ends, time used: {time.time() - start_time:.2f}s")

    return t, size_t, compton_event_count_list


def process_list_on_multi_gpu_sparse(
    num_gpus,
    sysmat,
    detector,
    sparse_projector,
    list_origin,
    rotate_num,
    num_workers,
    delta_r1,
    delta_r2,
    e0,
    ene_resolution,
    ene_threshold_max,
    ene_threshold_min,
    ene_threshold_sum,
    start_time,
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
                    result_dict[rank] = torch.empty((0, sparse_projector.coarse_pixel_num + 1), dtype=torch.float32)
                    continue

                process = mp.Process(
                    target=get_compton_backproj_list_mp_sparse,
                    args=(
                        rank,
                        num_gpus,
                        sysmat,
                        detector,
                        sparse_projector,
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
                        0,
                        None,
                    ),
                )
                process.start()
                processes.append(process)

            for process in processes:
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(f"Sparse list worker exited with code {process.exitcode} on rotate {rotate_idx + 1}.")

            t_results = []
            for rank in range(num_gpus):
                if rank not in result_dict:
                    raise RuntimeError(f"Missing sparse result from rank {rank} on rotate {rotate_idx + 1}.")
                t_results.append(result_dict[rank])

        t_tmp = torch.cat(t_results, dim=0) if t_results else torch.empty((0, sparse_projector.coarse_pixel_num + 1), dtype=torch.float32)
        compton_event_count_list[rotate_idx] = t_tmp.size(0)
        size_t += t_tmp.nelement() * t_tmp.element_size()
        t.append(t_tmp)
        print(f"Sparse rotate {rotate_idx + 1} ends, time used: {time.time() - start_time:.2f}s")

    return t, size_t, compton_event_count_list


def build_sparse_sensi_d(t_all, sysmat_all, sparse_projector_all, s_map_s, single_event_count_total, compton_event_count_total, block_size=1024):
    sensi_d = torch.zeros_like(s_map_s)
    for t_energy, sysmat, sparse_projector in zip(t_all, sysmat_all, sparse_projector_all):
        sysmat_gpu = sysmat.to("cuda:0", non_blocking=True)
        projector_gpu = sparse_projector.to("cuda:0")
        for t_rotate in t_energy:
            if t_rotate.numel() == 0:
                continue
            for row_start in range(0, t_rotate.size(0), block_size):
                event_block = t_rotate[row_start:row_start + block_size].to("cuda:0", non_blocking=True)
                t_fine, _ = materialize_sparse_event_rows_to_fine(event_block, sysmat_gpu, projector_gpu)
                if t_fine.numel() > 0:
                    sensi_d = sensi_d + torch.sum(t_fine, dim=0, keepdim=True).transpose(0, 1).cpu()

    if torch.sum(sensi_d) > 0:
        sensi_d = sensi_d * torch.sum(s_map_s) / torch.sum(sensi_d)
        sensi_d = sensi_d * compton_event_count_total / max(single_event_count_total, 1)
    return sensi_d


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
        log_filename = repo_root / f"print_log_sparse_{rand_suffix}.txt"
        logfile = open(log_filename, "w", encoding="utf-8")
        sys.stdout = Tee(original_stdout, logfile)

        available_gpus = torch.cuda.device_count()
        if available_gpus <= 0 or not torch.cuda.is_available():
            raise RuntimeError("main_plane_sparse.py requires CUDA. No available GPU was detected.")
        args.num_gpus = min(args.num_gpus, available_gpus)
        print(f"CUDA is available, using {args.num_gpus} / {available_gpus} GPUs")
        print(f"Repo Root: {repo_root}")
        print(f"Args: {args}")

        proj_all = []
        list_all = []
        sysmat_all = []
        detector_all = []
        rotmat_all = []
        rotmat_inv_all = []
        sensi_s_all = []
        sensi_d_all = []
        sparse_projector_all = []
        e_params = []

        pixel_num = None
        for e0, ene_threshold_sum, intensity in zip(args.e0_list, args.ene_threshold_sum_list, args.intensity_list):
            ene_resolution = args.ene_resolution_662keV * (0.662 / e0) ** 0.5
            ene_threshold_max = 2 * e0 ** 2 / (0.511 + 2 * e0) - 0.001
            ene_threshold_min = 0.05

            factor_dir = resolve_factor_dir(factors_root, e0, args.rotate_num)
            proj_file_path, list_dir_path = resolve_proj_and_list_paths(cntstat_root, list_root, e0, args.rotate_num, args.data_file_name, args.count_level)

            sysmat_file_path = factor_dir / "SysMat_polar"
            detector_file_path = factor_dir / "Detector.csv"
            sensi_s_file_path = factor_dir / "Sensi_s"
            sensi_d_file_path = factor_dir / "Sensi_d"
            coor_polar_file_path = factor_dir / "coor_polar_full.csv"
            rotmat_file_path = factor_dir / "RotMat_full.csv"
            rotmat_inv_file_path = factor_dir / "RotMatInv_full.csv"

            detector = torch.from_numpy(np.genfromtxt(detector_file_path, delimiter=",", dtype=np.float32)[:, 1:4])
            coor_polar = torch.from_numpy(np.genfromtxt(coor_polar_file_path, delimiter=",", dtype=np.float32))
            rotmat = torch.from_numpy(np.genfromtxt(rotmat_file_path, delimiter=",", dtype=np.int64))
            rotmat_inv = torch.from_numpy(np.genfromtxt(rotmat_inv_file_path, delimiter=",", dtype=np.int64))

            pixel_num_current = resolve_pixel_num(args.pixel_num_layer * args.pixel_num_z, args.pixel_num_z, rotmat, rotmat_inv, coor_polar, factor_dir)
            if pixel_num is None:
                pixel_num = pixel_num_current
            elif pixel_num != pixel_num_current:
                raise ValueError(f"Inconsistent pixel_num across energies: previous={pixel_num}, current={pixel_num_current}")

            sysmat_np = np.fromfile(sysmat_file_path, dtype=np.float32)
            sysmat = torch.from_numpy(sysmat_np.reshape(pixel_num, -1).T.copy()) * intensity

            if sensi_s_file_path.exists():
                sensi_s = torch.from_numpy(np.fromfile(sensi_s_file_path, dtype=np.float32).reshape(pixel_num, 1).copy()) * intensity
                sensi_s_all.append(sensi_s)
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

            proj_np = np.genfromtxt(proj_file_path, delimiter=",", dtype=np.float32)
            proj = torch.from_numpy(proj_np.reshape(args.rotate_num, -1).T.copy())

            list_origin = []
            for rotate_idx in range(args.rotate_num):
                list_origin.append(load_list_csv(list_dir_path / f"{rotate_idx + 1}.csv"))

            proj, list_origin = downsample_projection_and_list(proj, list_origin, args.ds * intensity)

            proj_all.append(proj)
            list_all.append(list_origin)
            sysmat_all.append(sysmat)
            detector_all.append(detector)
            rotmat_all.append(rotmat)
            rotmat_inv_all.append(rotmat_inv)
            sparse_projector_all.append(sparse_projector)
            e_params.append((e0, ene_resolution, ene_threshold_max, ene_threshold_min, ene_threshold_sum))

            print(
                f"Loaded energy {e0:.3f} MeV | fine pixels={pixel_num} | sparse pixels={sparse_projector.coarse_pixel_num} "
                f"(theta_stride={args.compton_theta_stride}, z_stride={args.compton_z_stride}, ring_strides={sparse_projector.ring_strides})"
            )

        t_all = []
        proj_d_all = []
        single_event_count_total = 0
        compton_event_count_total = 0
        s_map_s_total = torch.zeros([pixel_num, 1], dtype=torch.float32)

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

        for proj, list_origin, sysmat, detector, rotmat_inv, sparse_projector, e_param in zip(
            proj_all,
            list_all,
            sysmat_all,
            detector_all,
            rotmat_inv_all,
            sparse_projector_all,
            e_params,
        ):
            e0, ene_resolution, ene_threshold_max, ene_threshold_min, ene_threshold_sum = e_param
            print(f"Processing sparse Compton list for energy {e0:.3f} MeV ...")

            if args.num_gpus == 1:
                t, size_t, compton_event_count_list = process_list_on_single_gpu_sparse(
                    sysmat,
                    detector,
                    sparse_projector,
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
                )
            else:
                t, size_t, compton_event_count_list = process_list_on_multi_gpu_sparse(
                    args.num_gpus,
                    sysmat,
                    detector,
                    sparse_projector,
                    list_origin,
                    args.rotate_num,
                    iter_arg.num_workers,
                    args.delta_r1,
                    args.delta_r2,
                    e0,
                    ene_resolution,
                    ene_threshold_max,
                    ene_threshold_min,
                    ene_threshold_sum,
                    start_time,
                )

            proj_d = torch.zeros_like(proj)
            for rotate_idx in range(args.rotate_num):
                proj_tmp = proj[:, rotate_idx]
                proj_indices = torch.tensor([idx for idx in range(proj_tmp.size(0)) for _ in range(round(proj_tmp[idx].item()))], dtype=torch.long)
                if proj_indices.numel() == 0:
                    continue
                selected_num = min(int(compton_event_count_list[rotate_idx].item()), proj_indices.numel())
                proj_d_index_tmp = proj_indices[torch.randperm(proj_indices.numel())[:selected_num]]
                for bin_idx in range(proj_d.size(0)):
                    proj_d[bin_idx, rotate_idx] = (proj_d_index_tmp == bin_idx).sum()

            single_event_count = round(proj.sum().item())
            compton_event_count = round(proj_d.sum().item())
            print(f"[Energy {e0:.3f}] Single events = {single_event_count}, Sparse Compton events = {compton_event_count}")
            print(f"[Energy {e0:.3f}] The size of sparse t is {size_t / (1024 ** 3):.2f} GB")

            t_all.append(t)
            proj_d_all.append(proj_d)

            s_map_s_tmp = torch.zeros([1, pixel_num], dtype=torch.float32)
            for rotate_idx in range(args.rotate_num):
                rotmat_inv_tmp = rotmat_inv[:, rotate_idx]
                s_map_s_tmp += torch.sum(sysmat[:, rotmat_inv_tmp - 1], dim=0, keepdim=True).cpu()
            s_map_s_total += (s_map_s_tmp.transpose(0, 1) / args.rotate_num)

            single_event_count_total += single_event_count
            compton_event_count_total += compton_event_count

        s_map_arg = argparse.Namespace()
        s_map_arg.s = sum(sensi_s_all) if sensi_s_all else s_map_s_total

        if sensi_d_all and not args.recompute_sparse_sensi_d:
            print("Sparse chain uses file-defined Sensi_d.")
            s_map_arg.d = sum(sensi_d_all) * args.s_map_d_ratio
        else:
            if sensi_d_all:
                print("Sparse chain recomputes Sensi_d from sparse Compton operators.")
            else:
                print("Sparse chain has no file-defined Sensi_d and recomputes it from sparse Compton operators.")
            s_map_arg.d = build_sparse_sensi_d(
                t_all,
                sysmat_all,
                sparse_projector_all,
                s_map_arg.s,
                single_event_count_total,
                compton_event_count_total,
            ) * args.s_map_d_ratio

        if args.save_s:
            with open(repo_root / "sensitivity_s_sparse", "wb") as file:
                s_map_arg.s.cpu().numpy().astype("float32").tofile(file)

        if args.save_d:
            with open(repo_root / "Sensi_d_sparse", "wb") as file:
                s_map_arg.d.cpu().numpy().astype("float32").tofile(file)

        torch.cuda.empty_cache()

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
        sparse_prefix = f"Sparse_{args.compton_theta_stride}_{args.compton_z_stride}_"
        save_path = base_path.parent.parent / f"{sparse_prefix}{base_path.parent.name}" / "Polar"
        save_path.mkdir(parents=True, exist_ok=True)

        run_recon_osem_sparse(
            sysmat_all,
            rotmat_all,
            rotmat_inv_all,
            proj_all,
            proj_d_all,
            t_all,
            sparse_projector_all,
            iter_arg,
            s_map_arg,
            args.alpha,
            str(save_path) + os.sep,
            args.num_gpus,
            None,
        )

        print(f"\nTotal time used: {time.time() - start_time:.2f}s")

    finally:
        sys.stdout = original_stdout
        if logfile is not None:
            logfile.close()

        if log_filename is not None:
            if save_path is not None and Path(save_path).is_dir():
                shutil.move(str(log_filename), Path(save_path) / "print_log.txt")
            elif Path(log_filename).exists():
                print(f"Log kept at {log_filename}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    with torch.no_grad():
        main()
