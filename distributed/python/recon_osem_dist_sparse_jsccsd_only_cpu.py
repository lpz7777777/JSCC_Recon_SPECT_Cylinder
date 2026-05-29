"""
分布式 CPU 版本 —— 仅 JSCC-SD 稀疏 Compton 重建核心。

基于 recon_osem_dist_sparse_jsccsd_only.py，将 GPU (CUDA/NCCL) 替换为 CPU (GLOO)：
  - device = torch.device("cpu")
  - dist backend = "gloo"
  - 删除所有 torch.cuda 相关调用
  - 通过 OMP_NUM_THREADS / torch.set_num_threads() 控制 CPU 并行度

计算逻辑与 GPU 版本完全一致，确保数值结果相同。
"""

import time

import torch
import torch.distributed as dist

try:
    from distributed.python.cpu_mem_report import log_cpu_memory_usage
except ImportError:
    from cpu_mem_report import log_cpu_memory_usage

from compton_sparse_ops import materialize_sparse_event_rows_to_fine


def get_weight_single(sysmat, proj, img_rotate):
    """单光子 bin-mode EM 权重计算。与 GPU 版本完全一致。"""
    forward = torch.matmul(sysmat, img_rotate).clamp_min(1e-12)
    return torch.matmul(sysmat.transpose(0, 1), proj / forward)


def get_weight_compton_sparse(event_block, sysmat_full, img_rotate, sparse_projector):
    """
    稀疏 Compton list-mode EM 权重计算。
    每次调用只处理一个 event_block（t_divide 的一个分片）。
    在 CPU 上，sysmat_full 已经在内存中，无需额外显存考虑。
    """
    t_fine, _ = materialize_sparse_event_rows_to_fine(event_block, sysmat_full, sparse_projector)
    if t_fine.size(0) == 0:
        return torch.zeros_like(img_rotate)
    denom = torch.clamp(torch.matmul(t_fine, img_rotate), min=1e-12)
    weight = torch.matmul(t_fine.transpose(0, 1), 1.0 / denom)
    return torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)


def safe_em_update(img, weight, s_map, eps=1e-12):
    """安全的 EM 乘性更新，防止除零和 NaN。"""
    weight_safe = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
    s_map_safe = torch.nan_to_num(s_map, nan=0.0, posinf=0.0, neginf=0.0)
    valid = s_map_safe > eps
    updated = torch.zeros_like(img)
    updated[valid] = img[valid] * torch.clamp(weight_safe[valid], min=0.0) / s_map_safe[valid]
    return updated


def summarize_image_tensor(img):
    """打印图像张量的统计摘要。"""
    img_cpu = img.detach().float().cpu()
    zero_count = int((img_cpu == 0).sum().item())
    return (
        f"min={img_cpu.min().item():.6e} max={img_cpu.max().item():.6e} "
        f"mean={img_cpu.mean().item():.6e} sum={img_cpu.sum().item():.6e} zero={zero_count}/{img_cpu.numel()}"
    )


def build_random_bin_subsets(sysmat_l_all, proj_l_all, subset_num, generator, device):
    """
    为 OSEM 随机划分 bin 子集。
    CPU 版本与 GPU 版本逻辑一致，只是 device 是 cpu。
    """
    local_bin_num = proj_l_all[0].size(0)
    cpnum_list = torch.randperm(local_bin_num, generator=generator)
    cpnum_list_chunks = list(torch.chunk(cpnum_list, subset_num, dim=0))

    sysmat_list_all = [[None for _ in range(len(sysmat_l_all))] for _ in range(subset_num)]
    proj_list_all = [[None for _ in range(len(proj_l_all))] for _ in range(subset_num)]

    for energy_idx in range(len(sysmat_l_all)):
        for subset_idx in range(subset_num):
            subset_ids = cpnum_list_chunks[subset_idx]
            sysmat_list_all[subset_idx][energy_idx] = sysmat_l_all[energy_idx][subset_ids, :].to(device, non_blocking=False)
            proj_list_all[subset_idx][energy_idx] = proj_l_all[energy_idx][subset_ids, :].to(device, non_blocking=False)

    return sysmat_list_all, proj_list_all


def resolve_iter_seed(iter_arg, global_rank, default_seed=20260413):
    """为每个 rank 生成不同的迭代种子，确保 OSEM 子集划分不同。"""
    base_seed = int(getattr(iter_arg, "seed", default_seed))
    return base_seed + 1009 * int(global_rank)


def osem_joint_mode_dist_sparse_cpu(
    sysmat_l_all, proj_l_all, t_l_all,
    sysmat_full_all, sparse_projector_all,
    rotmat_all, rotmat_inv_all,
    img, s_map, alpha, rotate_num, device,
):
    """
    CPU 版本的联合 (Single + Compton) OSEM 迭代。

    与 GPU 版本 osem_joint_mode_dist_sparse 计算逻辑完全一致：
    1. 单光子部分：sysmat @ img → forward → weight_s
    2. 康普顿部分：materialize_sparse_event_rows_to_fine → t_fine → weight_c
    3. 联合权重：alpha * weight_s + (2-alpha) * weight_c
    4. dist.all_reduce 汇总所有 rank 的 weight
    5. EM 更新图像
    """
    for sysmat_l, proj_l, t_l in zip(sysmat_l_all, proj_l_all, t_l_all):
        weight_local = torch.zeros_like(img)

        # ---- 单光子部分 ----
        for rotate_idx in range(rotate_num):
            for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_l, proj_l, rotmat_all, rotmat_inv_all):
                img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                w_s = alpha * get_weight_single(sysmat, proj[:, rotate_idx].unsqueeze(1), img_rotate)
                weight_local = weight_local + torch.index_select(w_s, 0, rotmat_inv[:, rotate_idx] - 1)

        # ---- 康普顿部分 ----
        for rotate_idx in range(rotate_num):
            for t_rotate in t_l[rotate_idx]:
                for t_block, sysmat_full, rotmat, rotmat_inv, sparse_projector in zip(
                    t_rotate, sysmat_full_all, rotmat_all, rotmat_inv_all, sparse_projector_all
                ):
                    if t_block.numel() == 0:
                        continue
                    # CPU 上不需要 device 转移，数据已在 CPU
                    img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                    w_c = (2 - alpha) * get_weight_compton_sparse(t_block, sysmat_full, img_rotate, sparse_projector)
                    weight_local = weight_local + torch.index_select(w_c, 0, rotmat_inv[:, rotate_idx] - 1)

        # 全局 all-reduce 聚合权重（GLOO 后端）
        dist.all_reduce(weight_local, op=dist.ReduceOp.SUM)
        img = safe_em_update(img, weight_local, s_map)

    return img


def save_img_dist_jsccsd_only(img_jsccsd, img_jsccsd_iter, iter_arg, save_path):
    """仅在 rank 0 保存重建结果。"""
    if dist.get_rank() != 0:
        return

    img_jsccsd.numpy().astype("float32").tofile(save_path + "Image_JSCCSD")
    img_jsccsd_iter.numpy().astype("float32").tofile(
        save_path + "Image_JSCCSD_Iter_%d_%d" % (iter_arg.jsccsd, iter_arg.jsccsd // iter_arg.save_iter_step)
    )
    print("Images saved to disk (rank 0).")


def run_recon_osem_dist_sparse_jsccsd_only_cpu(
    sysmat_l_all,
    sysmat_full_all,
    rotmat_all,
    rotmat_inv_all,
    proj_l_all,
    t_local_all,
    sparse_projector_all,
    iter_arg,
    s_map_arg,
    alpha,
    save_path,
):
    """
    CPU 分布式 JSCC-SD-only 重建主循环。

    与 GPU 版本相比：
    - device = cpu（所有张量在 CPU 上）
    - 使用 GLOO 后端进行 all-reduce
    - 不调用 torch.cuda.empty_cache()
    - 每个 rank 通过 OMP_NUM_THREADS 控制 OpenMP 并行度
    """
    global_rank = dist.get_rank()
    # CPU 版本：直接使用 CPU 设备
    device = torch.device("cpu")
    pixel_num = s_map_arg.j.size(0)
    rotate_num = rotmat_all[0].size(1)

    # 初始化重建图像（在 CPU 上）
    img_jsccsd = torch.ones([pixel_num, 1], dtype=torch.float32, device=device)
    img_jsccsd_iter = torch.ones(
        [round(iter_arg.jsccsd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32
    )

    # sysmat_full 和 sparse_projector 已经在 CPU 上，无需 .to(device)
    # 但确保它们确实在 CPU 上（防御性编程）
    sysmat_full_all = [sysmat_full.to(device) for sysmat_full in sysmat_full_all]
    sparse_projector_all = [projector.to(device) for projector in sparse_projector_all]

    # 初始化 t_list_all 结构：[subset_num][rotate_num][t_divide_num][ene_num]
    empty_t = torch.zeros((0, sparse_projector_all[0].coarse_pixel_num + 1), dtype=torch.float32, device=device)
    t_list_all = [
        [
            [[empty_t for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.t_divide_num)]
            for _ in range(rotate_num)
        ]
        for _ in range(iter_arg.osem_subset_num)
    ]

    generator = torch.Generator()
    generator.manual_seed(resolve_iter_seed(iter_arg, global_rank))

    # 将 t_local_all 按照 OSEM 子集和 t_divide 划分
    for energy_idx in range(iter_arg.ene_num):
        for rotate_idx in range(rotate_num):
            t_rotate_local = t_local_all[energy_idx][rotate_idx]
            t_subset_chunks = list(torch.chunk(t_rotate_local, iter_arg.osem_subset_num, dim=0))
            for subset_idx in range(len(t_subset_chunks)):
                t_divide_chunks = list(torch.chunk(t_subset_chunks[subset_idx], iter_arg.t_divide_num, dim=0))
                for divide_idx in range(len(t_divide_chunks)):
                    t_list_all[subset_idx][rotate_idx][divide_idx][energy_idx] = t_divide_chunks[divide_idx].to(device)

    del t_local_all
    dist.barrier()
    log_cpu_memory_usage("pre-iter-sparse-jsccsd-only-cpu")

    if global_rank == 0:
        print("\n" + "=" * 50)
        print("Starting Distributed CPU Sparse JSCCSD-Only Reconstruction")
        print(f"  device=cpu, world_size={dist.get_world_size()}, threads/rank={torch.get_num_threads()}")
        print("=" * 50)

    time_start = time.time()
    save_idx = 0

    for iter_idx in range(iter_arg.jsccsd):
        sysmat_list_all, proj_list_all = build_random_bin_subsets(
            sysmat_l_all,
            proj_l_all,
            iter_arg.osem_subset_num,
            generator,
            device,
        )
        img_jsccsd = osem_joint_mode_dist_sparse_cpu(
            sysmat_list_all,
            proj_list_all,
            t_list_all,
            sysmat_full_all,
            sparse_projector_all,
            rotmat_all,
            rotmat_inv_all,
            img_jsccsd,
            s_map_arg.j,
            alpha,
            rotate_num,
            device,
        )
        del sysmat_list_all
        del proj_list_all

        if (iter_idx + 1) % iter_arg.save_iter_step == 0:
            img_jsccsd_iter[save_idx, :] = img_jsccsd.squeeze()
            save_idx += 1
            log_cpu_memory_usage(f"iter-{iter_idx + 1}-sparse-jsccsd-only-cpu")
            if global_rank == 0:
                elapsed = time.time() - time_start
                print(
                    f"[JSCCSD-Sparse-Only-CPU] Iter: {iter_idx + 1}/{iter_arg.jsccsd} | "
                    f"Time: {elapsed:.2f}s | {summarize_image_tensor(img_jsccsd)}"
                )

    if global_rank == 0:
        print(f"Sparse JSCCSD-only CPU distributed reconstruction time: {time.time() - time_start:.2f}s")

    save_img_dist_jsccsd_only(img_jsccsd, img_jsccsd_iter, iter_arg, save_path)