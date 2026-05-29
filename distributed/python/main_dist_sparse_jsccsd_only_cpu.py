"""
分布式 CPU 版本 —— 仅 JSCC-SD 稀疏 Compton 重建主入口。

基于 main_dist_sparse_jsccsd_only.py，将 GPU (CUDA/NCCL) 替换为 CPU (GLOO)。

与 GPU 版本的关键区别：
  1. setup_distributed() 使用 gloo 后端，不调用 torch.cuda.set_device()
  2. device = torch.device("cpu")
  3. 所有数据加载后直接留在 CPU 上
  4. 通过 OMP_NUM_THREADS 控制每个进程的 OpenMP 线程数
  5. 不调用 torch.cuda.empty_cache()

计算逻辑与 GPU 版本完全一致，使用相同的：
  - bin 分片策略（每个 rank 负责一段 bin）
  - event 分片策略（每个 rank 负责一段 list events）
  - OSEM 子集随机划分
  - 稀疏 Compton 投影与反投影
  - all-reduce 聚合权重后 EM 更新

使用方式：
  通过 SLURM 脚本提交，使用 torchrun 启动多进程。
  详见 distributed/scripts/jsccrecon_dist_sparse_jsccsd_only_cpu.sh
"""

import argparse
import gc
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def _print_platform_info():
    """在启动时打印平台信息，辅助调试 AMD CPU 兼容性。"""
    import platform
    print(f"  Python:   {platform.python_version()}")
    print(f"  NumPy:    {np.__version__}")
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  CPU:      {platform.processor() or 'N/A'}")
    print(f"  BLAS:     {np.__config__.show() if hasattr(np.__config__, 'show') else 'N/A'}")
    print(f"  MKL avail: {torch.backends.mkl.is_available()}")
    print(f"  OpenMP:   {torch.backends.openmp.is_available()}")
    print(f"  Num threads: {torch.get_num_threads()}")

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
from process_list_plane_sparse import get_compton_backproj_list_single_sparse

# 导入 CPU 版本的重建核心
from distributed.python.recon_osem_dist_sparse_jsccsd_only_cpu import (
    run_recon_osem_dist_sparse_jsccsd_only_cpu,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed CPU sparse JSCCSD-only reconstruction for plane geometry."
    )
    parser.add_argument("--e0-list", type=float, nargs="+", default=[0.511], help="Energy list in MeV.")
    parser.add_argument("--ene-threshold-sum-list", type=float, nargs="+", default=[0.46], help="Lower bounds for e1 + e2 in MeV.")
    parser.add_argument("--intensity-list", type=float, nargs="+", default=[1.0], help="Intensity weights.")
    parser.add_argument("--s-map-d-ratio", type=float, default=1.0, help="Scale factor applied to sparse Sensi_d.")
    parser.add_argument("--recompute-sparse-sensi-d", action="store_true", help="Force recomputing Sensi_d from sparse Compton operators.")
    parser.add_argument("--data-file-name", type=str, default="Hoffman_Big", help="Dataset name.")
    parser.add_argument("--count-level", type=str, default="5e10", help="Count level suffix.")
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
    parser.add_argument("--osem-subset-num", type=int, default=16, help="OSEM subset count.")
    parser.add_argument("--t-divide-num", type=int, default=1, help="Number of t sub-blocks per subset.")
    parser.add_argument("--num-workers", type=int, default=20, help="Sub-chunks per rank during sparse list processing.")
    parser.add_argument("--compton-theta-stride", type=int, default=2, help="Angular stride used by sparse Compton grid.")
    parser.add_argument("--compton-z-stride", type=int, default=1, help="Axial stride used by sparse Compton grid.")
    parser.add_argument("--seed", type=int, default=20260331, help="Random seed.")
    parser.add_argument("--factors-dir", type=str, default="./Factors", help="Factors root directory.")
    parser.add_argument("--cntstat-dir", type=str, default="./CntStat", help="CntStat root directory.")
    parser.add_argument("--list-dir", type=str, default="./List", help="List root directory.")
    parser.add_argument("--output-root", type=str, default="./Figure_Dist_JSCCSD_CPU", help="Output root directory.")
    # CPU 专用参数
    parser.add_argument("--num-threads", type=int, default=None, help="Number of OpenMP threads per rank. Default: auto (OMP_NUM_THREADS env or 4).")
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
    """
    初始化分布式 CPU 环境。

    与 GPU 版本的区别：
    - 不调用 torch.cuda.set_device()
    - 使用 gloo 后端（而非 NCCL）
    """
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # CPU 版本：使用 gloo 后端，无需 GPU
    dist.init_process_group(backend="gloo", init_method="env://")

    return global_rank, local_rank, world_size


def build_sparse_sensi_d_local(t_local_all, sysmat_full_all, sparse_projector_all, pixel_num, block_size=1024):
    """
    从稀疏 Compton 算子本地重建 Sensi_d。
    在 CPU 上运行，block_size 控制峰值内存。
    """
    sensi_d_local = torch.zeros((pixel_num, 1), dtype=torch.float32)
    for t_energy, sysmat_full, sparse_projector in zip(t_local_all, sysmat_full_all, sparse_projector_all):
        # sparse_projector 需要在 sysmat_full 的设备上（这里是 CPU，无需转换）
        for t_rotate in t_energy:
            if t_rotate.numel() == 0:
                continue
            for row_start in range(0, t_rotate.size(0), block_size):
                event_block = t_rotate[row_start:row_start + block_size]
                t_fine, _ = materialize_sparse_event_rows_to_fine(event_block, sysmat_full, sparse_projector)
                if t_fine.numel() > 0:
                    sensi_d_local = sensi_d_local + torch.sum(t_fine, dim=0, keepdim=True).transpose(0, 1)
    return sensi_d_local


def log_tensor_stats(name, tensor):
    """打印张量统计信息。"""
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

    # 设置 OpenMP 线程数
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    elif os.environ.get("OMP_NUM_THREADS") is None:
        # 默认每个 rank 使用 4 个线程
        torch.set_num_threads(4)

    global_rank, local_rank, world_size = setup_distributed()
    # CPU 版本：所有计算在 CPU 上
    device = torch.device("cpu")

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
            log_filename = repo_root / f"print_log_dist_sparse_jsccsd_only_cpu_{rand_suffix}.txt"
            logfile = open(log_filename, "w", encoding="utf-8")
            sys.stdout = Tee(sys.__stdout__, logfile)
            print(f"Distributed CPU sparse JSCCSD-only chain initialized: World Size {world_size}")
            print(f"OpenMP threads per rank: {torch.get_num_threads()}")
            _print_platform_info()
            print(f"Args: {args}")

        proj_local_all = []
        sysmat_local_all = []
        sysmat_full_all = []
        rotmat_all = []
        rotmat_inv_all = []
        sensi_s_all = []
        sensi_d_all = []
        sparse_projector_all = []

        pixel_num = None
        single_event_count_total_local = 0
        compton_event_count_total_local = 0
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

            # 加载数据到 CPU（不再 .to(cuda_device)）
            detector = torch.from_numpy(np.genfromtxt(detector_file_path, delimiter=",", dtype=np.float32)[:, 1:4])
            coor_polar = torch.from_numpy(np.genfromtxt(coor_polar_file_path, delimiter=",", dtype=np.float32))
            rotmat = torch.from_numpy(np.genfromtxt(rotmat_file_path, delimiter=",", dtype=np.int64))
            rotmat_inv = torch.from_numpy(np.genfromtxt(rotmat_inv_file_path, delimiter=",", dtype=np.int64))

            pixel_num_current = resolve_pixel_num(
                args.pixel_num_layer * args.pixel_num_z, args.pixel_num_z,
                rotmat, rotmat_inv, coor_polar, factor_dir,
            )
            if pixel_num is None:
                pixel_num = pixel_num_current
            elif pixel_num != pixel_num_current:
                raise ValueError(
                    f"Inconsistent pixel_num across energies: previous={pixel_num}, current={pixel_num_current}"
                )

            # 加载完整系统矩阵到 CPU
            full_sysmat = (
                torch.from_numpy(np.fromfile(sysmat_file_path, dtype=np.float32).reshape(pixel_num, -1).T.copy())
                * intensity
            )
            total_bins = full_sysmat.size(0)

            # bin 分片：每个 rank 负责一段 bin
            bins_per_rank = total_bins // world_size
            idx_start = global_rank * bins_per_rank
            idx_end = (global_rank + 1) * bins_per_rank if global_rank != world_size - 1 else total_bins
            sysmat_local = full_sysmat[idx_start:idx_end, :].clone()
            sysmat_local_all.append(sysmat_local)

            # sysmat_full 在 CPU 上保留（用于 Compton 稀疏展开）
            sysmat_full_cpu = full_sysmat  # 已经在 CPU 上
            sysmat_full_all.append(sysmat_full_cpu)

            # Sensi_s
            if sensi_s_file_path.exists():
                sensi_s = (
                    torch.from_numpy(np.fromfile(sensi_s_file_path, dtype=np.float32).reshape(pixel_num, 1).copy())
                    * intensity
                )
                sensi_s_all.append(sensi_s)
            else:
                sensi_s_tmp = torch.zeros([1, pixel_num], dtype=torch.float32)
                for rotate_idx in range(args.rotate_num):
                    rotmat_inv_tmp = rotmat_inv[:, rotate_idx]
                    sensi_s_tmp += torch.sum(full_sysmat[:, rotmat_inv_tmp - 1], dim=0, keepdim=True)
                sensi_s_all.append(sensi_s_tmp.transpose(0, 1) / args.rotate_num)

            # Sensi_d
            if sensi_d_file_path.exists():
                sensi_d = (
                    torch.from_numpy(np.fromfile(sensi_d_file_path, dtype=np.float32).reshape(pixel_num, 1).copy())
                    * intensity
                )
                sensi_d_all.append(sensi_d)
                if global_rank == 0:
                    print(f"Loaded factor Sensi_d: {sensi_d_file_path}")
            elif global_rank == 0:
                print(f"Factor Sensi_d not found, will recompute if needed: {sensi_d_file_path}")

            # 构建稀疏 Compton 投影器
            sparse_projector = build_compton_sparse_projector(
                coor_polar,
                theta_stride=args.compton_theta_stride,
                z_stride=args.compton_z_stride,
                rotate_num=args.rotate_num,
                dtype=torch.float32,
            )
            sparse_projector_all.append(sparse_projector)

            # 加载投影数据和 event list
            full_proj = torch.from_numpy(
                np.genfromtxt(proj_file_path, delimiter=",", dtype=np.float32).reshape(args.rotate_num, -1).T.copy()
            )
            proj_local = full_proj[idx_start:idx_end, :].clone()
            del full_proj

            # 加载并分片 event list
            list_rotate_local = []
            for rotate_idx in range(args.rotate_num):
                full_list = load_list_csv(list_dir_path / f"{rotate_idx + 1}.csv")
                ev_per_rank = full_list.size(0) // world_size
                ev_start = global_rank * ev_per_rank
                ev_end = (global_rank + 1) * ev_per_rank if global_rank != world_size - 1 else full_list.size(0)
                list_rotate_local.append(full_list[ev_start:ev_end, :])

            # 下采样
            proj_local, list_rotate_local = downsample_projection_and_list(
                proj_local, list_rotate_local, args.ds * intensity
            )
            proj_local_all.append(proj_local)
            single_event_count_total_local += round(proj_local.sum().item())

            # ---- 康普顿反投影：在 CPU 上计算 t 矩阵 ----
            t_rotate_local = []
            for rotate_idx in range(args.rotate_num):
                list_local_chunks = torch.chunk(list_rotate_local[rotate_idx], args.num_workers, dim=0)
                t_rotate_parts = []
                for chunk in list_local_chunks:
                    if chunk.numel() == 0:
                        continue
                    # 在 CPU 上计算 Compton 反投影
                    t_chunk, _, _ = get_compton_backproj_list_single_sparse(
                        sysmat_full_cpu,      # sysmat_full 在 CPU 上
                        detector,             # detector 在 CPU 上
                        sparse_projector,     # sparse_projector 在 CPU 上
                        chunk,                # chunk 在 CPU 上
                        args.delta_r1,
                        args.delta_r2,
                        e0,
                        ene_resolution,
                        ene_threshold_max,
                        ene_threshold_min,
                        ene_threshold_sum,
                        device,               # torch.device("cpu")
                    )
                    if t_chunk.numel() > 0:
                        t_rotate_parts.append(t_chunk)
                        compton_event_count_total_local += t_chunk.size(0)
                t_rotate = (
                    torch.cat(t_rotate_parts, dim=0)
                    if t_rotate_parts
                    else torch.empty((0, sparse_projector.coarse_pixel_num + 1), dtype=torch.float32)
                )
                t_rotate_local.append(t_rotate)
            t_local_all.append(t_rotate_local)

            # rotmat / rotmat_inv 保留在 CPU
            rotmat_all.append(rotmat)
            rotmat_inv_all.append(rotmat_inv)

            if global_rank == 0:
                print(
                    f"Loaded sparse projector for {e0:.3f} MeV: fine_pixels={pixel_num}, "
                    f"sparse_pixels={sparse_projector.coarse_pixel_num}, ring_strides={sparse_projector.ring_strides}"
                )

            del full_sysmat
            gc.collect()  # CPU 版本：主动回收不再使用的临时对象

        # 全局统计事件数
        single_event_count_tensor = torch.tensor([single_event_count_total_local], dtype=torch.float64)
        compton_event_count_tensor = torch.tensor([compton_event_count_total_local], dtype=torch.float64)
        dist.all_reduce(single_event_count_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(compton_event_count_tensor, op=dist.ReduceOp.SUM)
        single_event_count_total = int(single_event_count_tensor.item())
        compton_event_count_total = int(compton_event_count_tensor.item())

        # 构建 s_map
        s_map_arg = argparse.Namespace()
        s_map_arg.s = sum(sensi_s_all)
        if sensi_d_all and not args.recompute_sparse_sensi_d:
            if global_rank == 0:
                print("Distributed CPU sparse JSCCSD-only chain uses file-defined Sensi_d.")
            s_map_arg.d = sum(sensi_d_all) * args.s_map_d_ratio
        else:
            if global_rank == 0:
                if sensi_d_all:
                    print("Distributed CPU sparse JSCCSD-only chain recomputes Sensi_d from sparse Compton operators.")
                else:
                    print("Distributed CPU sparse JSCCSD-only chain has no file-defined Sensi_d, recomputing.")
            sensi_d_local = build_sparse_sensi_d_local(
                t_local_all, sysmat_full_all, sparse_projector_all, pixel_num
            )
            dist.all_reduce(sensi_d_local, op=dist.ReduceOp.SUM)
            if torch.sum(sensi_d_local) > 0:
                sensi_d_local = sensi_d_local * torch.sum(s_map_arg.s) / torch.sum(sensi_d_local)
                sensi_d_local = sensi_d_local * compton_event_count_total / max(single_event_count_total, 1)
            s_map_arg.d = sensi_d_local * args.s_map_d_ratio
        s_map_arg.j = args.alpha * s_map_arg.s + (2 - args.alpha) * s_map_arg.d

        if global_rank == 0:
            log_tensor_stats("s_map_arg.s", s_map_arg.s)
            log_tensor_stats("s_map_arg.d", s_map_arg.d)
            log_tensor_stats("s_map_arg.j", s_map_arg.j)

        # 构建迭代参数
        iter_arg = argparse.Namespace()
        iter_arg.jsccsd = args.jsccsd_iter
        iter_arg.save_iter_step = args.save_iter_step
        iter_arg.osem_subset_num = args.osem_subset_num
        iter_arg.t_divide_num = args.t_divide_num
        iter_arg.ene_num = len(args.e0_list)
        iter_arg.num_workers = args.num_workers
        iter_arg.seed = args.seed

        # 构建保存路径
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

        # 调用 CPU 版本的重建核心
        run_recon_osem_dist_sparse_jsccsd_only_cpu(
            sysmat_local_all,
            sysmat_full_all,
            rotmat_all,
            rotmat_inv_all,
            proj_local_all,
            t_local_all,
            sparse_projector_all,
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