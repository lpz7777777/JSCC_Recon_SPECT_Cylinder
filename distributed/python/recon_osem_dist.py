import torch
import os
import time
import torch.distributed as dist

try:
    from distributed.python.gpu_mem_report import log_gpu_memory_usage
except ImportError:
    from gpu_mem_report import log_gpu_memory_usage


def get_weight_single(sysmat, proj, img_rotate):
    # proj / (sysmat * img)
    forward = torch.matmul(sysmat, img_rotate).clamp_min(1e-12)
    return torch.matmul(sysmat.transpose(0, 1), proj / forward)


def get_weight_compton(t_block, img_rotate):
    # 1 / (t * img)
    forward = torch.matmul(t_block, img_rotate).clamp_min(1e-12)
    return torch.matmul(t_block.transpose(0, 1), 1.0 / forward)


def build_random_bin_subsets(sysmat_l_all, proj_l_all, subset_num, generator, device):
    local_bin_num = proj_l_all[0].size(0)
    cpnum_list = torch.randperm(local_bin_num, generator=generator)
    cpnum_list_chunks = list(torch.chunk(cpnum_list, subset_num, dim=0))

    sysmat_list_all = [[None for _ in range(len(sysmat_l_all))] for _ in range(subset_num)]
    proj_list_all = [[None for _ in range(len(proj_l_all))] for _ in range(subset_num)]

    for energy_idx in range(len(sysmat_l_all)):
        for subset_idx in range(subset_num):
            subset_ids = cpnum_list_chunks[subset_idx]
            sysmat_list_all[subset_idx][energy_idx] = sysmat_l_all[energy_idx][subset_ids, :].to(device, non_blocking=True)
            proj_list_all[subset_idx][energy_idx] = proj_l_all[energy_idx][subset_ids, :].to(device, non_blocking=True)

    return sysmat_list_all, proj_list_all


def resolve_iter_seed(iter_arg, global_rank, default_seed=20260413):
    base_seed = int(getattr(iter_arg, "seed", default_seed))
    return base_seed + 1009 * int(global_rank)


def apply_em_update(img, weight_local, s_map):
    safe_s_map = s_map.clamp_min(1e-12)
    update = weight_local / safe_s_map
    update = torch.nan_to_num(update, nan=1.0, posinf=1e6, neginf=0.0)
    img_next = img * update
    return torch.nan_to_num(img_next, nan=0.0, posinf=1e12, neginf=0.0)


def recompute_list_sensitivity_from_t(t_local_all, rotmat_inv_all, pixel_num, device):
    sensi_d_local = torch.zeros((pixel_num, 1), dtype=torch.float32, device=device)
    local_event_num = 0

    for t_energy, rotmat_inv in zip(t_local_all, rotmat_inv_all):
        for rotate_idx, t_rotate in enumerate(t_energy):
            if t_rotate.numel() == 0:
                continue
            sensi_rotate = t_rotate.sum(dim=0, dtype=torch.float32).to(device, non_blocking=True).unsqueeze(1)
            sensi_d_local += torch.index_select(sensi_rotate, 0, rotmat_inv[:, rotate_idx] - 1)
            local_event_num += int(t_rotate.size(0))

    event_num_tensor = torch.tensor([local_event_num], dtype=torch.float64, device=device)

    dist.all_reduce(sensi_d_local, op=dist.ReduceOp.SUM)
    dist.all_reduce(event_num_tensor, op=dist.ReduceOp.SUM)

    return sensi_d_local.clamp_min(1e-12), int(event_num_tensor.item())


def osem_bin_mode_dist(sysmat_l_all, proj_l_all, rotmat_all, rotmat_inv_all, img, s_map, rotate_num, device):
    for sysmat_l, proj_l in zip(sysmat_l_all, proj_l_all):
        weight_local = torch.zeros_like(img)
        for i in range(rotate_num):
            for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_l, proj_l, rotmat_all, rotmat_inv_all):
                if sysmat.size(0) == 0:
                    continue
                img_rotate = torch.index_select(img, 0, rotmat[:, i] - 1)
                w_tmp = get_weight_single(sysmat, proj[:, i].unsqueeze(1), img_rotate)
                weight_local = weight_local +  torch.index_select(w_tmp, 0, rotmat_inv[:, i] - 1)

        # 全局同步权重
        dist.all_reduce(weight_local, op=dist.ReduceOp.SUM)
        img = apply_em_update(img, weight_local, s_map)
    return img


def osem_list_mode_dist(t_l_all, rotmat_all, rotmat_inv_all, img, s_map, rotate_num, device):
    for t_l in t_l_all:
        weight_local = torch.zeros_like(img)
        for i in range(rotate_num):
            for t_rotate in t_l[i]:
                for t_block, rotmat, rotmat_inv in zip(t_rotate, rotmat_all, rotmat_inv_all):
                    if t_block.numel() == 0:
                        continue
                    if t_block.device != device:
                        t_block = t_block.to(device, non_blocking=True)
                    img_rotate = torch.index_select(img, 0, rotmat[:, i] - 1)
                    w_tmp = get_weight_compton(t_block, img_rotate)
                    weight_local += torch.index_select(w_tmp, 0, rotmat_inv[:, i] - 1)

        dist.all_reduce(weight_local, op=dist.ReduceOp.SUM)
        img = apply_em_update(img, weight_local, s_map)
    return img


def osem_joint_mode_dist(sysmat_l_all, proj_l_all, t_l_all, rotmat_all, rotmat_inv_all, img, s_map, alpha, rotate_num, device):
    for sysmat_l, proj_l, t_l in zip(sysmat_l_all, proj_l_all, t_l_all):
        weight_local = torch.zeros_like(img)
        # 单光子部分
        for i in range(rotate_num):
            for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_l, proj_l, rotmat_all, rotmat_inv_all):
                if sysmat.size(0) == 0:
                    continue
                img_rotate = torch.index_select(img, 0, rotmat[:, i] - 1)
                w_s = alpha * get_weight_single(sysmat, proj[:, i].unsqueeze(1), img_rotate)
                weight_local += torch.index_select(w_s, 0, rotmat_inv[:, i] - 1)
        # 康普顿部分
        for i in range(rotate_num):
            for t_rotate in t_l[i]:
                for t_block, rotmat, rotmat_inv in zip(t_rotate, rotmat_all, rotmat_inv_all):
                    if t_block.numel() == 0:
                        continue
                    if t_block.device != device:
                        t_block = t_block.to(device, non_blocking=True)
                    img_rotate = torch.index_select(img, 0, rotmat[:, i] - 1)
                    w_c = (2 - alpha) * get_weight_compton(t_block, img_rotate)
                    weight_local += torch.index_select(w_c, 0, rotmat_inv[:, i] - 1)

        dist.all_reduce(weight_local, op=dist.ReduceOp.SUM)
        img = apply_em_update(img, weight_local, s_map)
    return img


def save_img_dist(img_sc, img_scd, img_jsccd, img_jsccsd, img_sc_iter, img_scd_iter, img_jsccd_iter, img_jsccsd_iter, iter_arg, save_path):
    if dist.get_rank() == 0:
        # 与原版文件名完全一致
        img_sc.cpu().numpy().astype('float32').tofile(save_path + "Image_SC")
        img_scd.cpu().numpy().astype('float32').tofile(save_path + "Image_SCD")
        img_jsccd.cpu().numpy().astype('float32').tofile(save_path + "Image_JSCCD")
        img_jsccsd.cpu().numpy().astype('float32').tofile(save_path + "Image_JSCCSD")

        img_sc_iter.cpu().numpy().astype('float32').tofile(
            save_path + "Image_SC_Iter_%d_%d" % (iter_arg.sc, iter_arg.sc / iter_arg.save_iter_step))
        img_scd_iter.cpu().numpy().astype('float32').tofile(
            save_path + "Image_SCD_Iter_%d_%d" % (iter_arg.jsccd, iter_arg.jsccd / iter_arg.save_iter_step))
        img_jsccd_iter.cpu().numpy().astype('float32').tofile(
            save_path + "Image_JSCCD_Iter_%d_%d" % (iter_arg.jsccd, iter_arg.jsccd / iter_arg.save_iter_step))
        img_jsccsd_iter.cpu().numpy().astype('float32').tofile(
            save_path + "Image_JSCCSD_Iter_%d_%d" % (iter_arg.jsccsd, iter_arg.jsccsd / iter_arg.save_iter_step))
        print("Images saved to Rank 0 disk.")


def run_recon_osem_dist(sysmat_l_all, rotmat_all, rotmat_inv_all, proj_l_all, proj_dl_all, t_local_all, iter_arg, s_map_arg, alpha, save_path):
    global_rank = dist.get_rank()
    device = torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")
    pixel_num = s_map_arg.s.size(0)
    rotate_num = rotmat_all[0].size(1)
    print(f"[run_recon_osem_dist] rotate_num:{rotate_num}")
    s_map_d_from_t, global_compton_event_num = recompute_list_sensitivity_from_t(
        t_local_all,
        rotmat_inv_all,
        pixel_num,
        device,
    )
    has_valid_input_s_map_d = (
        hasattr(s_map_arg, "d")
        and torch.isfinite(s_map_arg.d).all()
        and float(torch.sum(s_map_arg.d).item()) > 0.0
    )
    if global_rank == 0:
        file_sum = float(torch.sum(s_map_arg.d).item()) if hasattr(s_map_arg, "d") else float("nan")
        recomputed_sum = float(torch.sum(s_map_d_from_t).item())
        print(
            "[run_recon_osem_dist] list sensitivity diagnostics "
            f"(events={global_compton_event_num}, input_sum={file_sum:.6e}, recomputed_sum={recomputed_sum:.6e}, "
            f"use_input={'yes' if has_valid_input_s_map_d else 'no'})"
        )
    if has_valid_input_s_map_d:
        s_map_arg.d = s_map_arg.d.to(device).clamp_min(1e-12)
    else:
        s_map_arg.d = s_map_d_from_t
    s_map_arg.j = alpha * s_map_arg.s + (2 - alpha) * s_map_arg.d

    # 1. 初始化图像
    img_sc = torch.ones([pixel_num, 1], dtype=torch.float32).to(device)
    img_scd = torch.ones([pixel_num, 1], dtype=torch.float32).to(device)
    img_jsccd = torch.ones([pixel_num, 1], dtype=torch.float32).to(device)
    img_jsccsd = torch.ones([pixel_num, 1], dtype=torch.float32).to(device)

    img_sc_iter = torch.ones([round(iter_arg.sc / iter_arg.save_iter_step), pixel_num])
    img_scd_iter = torch.ones([round(iter_arg.jsccd / iter_arg.save_iter_step), pixel_num])
    img_jsccd_iter = torch.ones([round(iter_arg.jsccd / iter_arg.save_iter_step), pixel_num])
    img_jsccsd_iter = torch.ones([round(iter_arg.jsccsd / iter_arg.save_iter_step), pixel_num])

    # 初始化 List 模式结构：[subset_num][rotate_num][divide_num][energy_num]
    # 使用空 Tensor 初始化，确保在某些能量/子集无数据时不会报错
    empty_t = torch.zeros((0, pixel_num), dtype=torch.float32, device=device)
    t_list_all = [[[[empty_t for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.t_divide_num)] for _ in range(rotate_num)] for _ in range(iter_arg.osem_subset_num)]
    generator = torch.Generator()
    generator.manual_seed(resolve_iter_seed(iter_arg, global_rank))

    for e in range(iter_arg.ene_num):
        # --- 处理 List 模式子集 (用于 JSCCD, JSCCSD) ---
        for i in range(rotate_num):
            t_rotate_local = t_local_all[e][i]
            # 首先按照事件总数划分为 OSEM 子集
            t_subset_chunks = list(torch.chunk(t_rotate_local, iter_arg.osem_subset_num, dim=0))

            for s in range(len(t_subset_chunks)):
                # 将每个子集进一步划分为 t_divide_num 块，以便在 get_weight_compton 中分批处理
                t_divide_chunks = list(torch.chunk(t_subset_chunks[s], iter_arg.t_divide_num, dim=0))
                for k in range(len(t_divide_chunks)):
                    t_list_all[s][i][k][e] = t_divide_chunks[k].to(device)

    del t_local_all
    dist.barrier()
    log_gpu_memory_usage("pre-iter", device)

    # 3. 开始重建循环
    if global_rank == 0:
        print("\n" + "="*50)
        print("Starting Distributed Reconstruction Process")
        print("="*50)

    time_start = time.time()
    
    # SC
    id_save = 0
    for i_sc in range(iter_arg.sc):
        sysmat_list_all, proj_list_all = build_random_bin_subsets(
            sysmat_l_all,
            proj_l_all,
            iter_arg.osem_subset_num,
            generator,
            device,
        )
        img_sc = osem_bin_mode_dist(sysmat_list_all, proj_list_all, rotmat_all, rotmat_inv_all, img_sc, s_map_arg.s, rotate_num, device)
        if (i_sc + 1) % iter_arg.save_iter_step == 0:
            img_sc_iter[id_save, :] = img_sc.squeeze().cpu()
            id_save += 1
            if global_rank == 0:
                elapsed = time.time() - time_start
                print(f"[SC] Iter: {i_sc+1}/{iter_arg.sc} | Time: {elapsed:.2f}s")

    # SCD
    for i_scd in range(iter_arg.jsccd):
        sysmat_list_all, proj_d_list_all = build_random_bin_subsets(
            sysmat_l_all,
            proj_dl_all,
            iter_arg.osem_subset_num,
            generator,
            device,
        )
        img_scd = osem_bin_mode_dist(sysmat_list_all, proj_d_list_all, rotmat_all, rotmat_inv_all, img_scd, s_map_arg.s, rotate_num, device)
        if (i_scd + 1) % iter_arg.save_iter_step == 0:
            img_scd_iter[i_scd // iter_arg.save_iter_step] = img_scd.squeeze().cpu()
            if global_rank == 0:
                elapsed = time.time() - time_start
                print(f"[SCD] Iter: {i_scd+1}/{iter_arg.jsccd} | Time: {elapsed:.2f}s")

    # JSCCD (List mode)
    for i_jsccd in range(iter_arg.jsccd):
        img_jsccd = osem_list_mode_dist(t_list_all, rotmat_all, rotmat_inv_all, img_jsccd, s_map_arg.d, rotate_num, device)
        if (i_jsccd + 1) % iter_arg.save_iter_step == 0:
            img_jsccd_iter[i_jsccd // iter_arg.save_iter_step] = img_jsccd.squeeze().cpu()
            if global_rank == 0:
                elapsed = time.time() - time_start
                print(f"[JSCCD] Iter: {i_jsccd+1}/{iter_arg.jsccd} | Time: {elapsed:.2f}s")

    # JSCCSD (Joint mode)
    for i_jsccsd in range(iter_arg.jsccsd):
        sysmat_list_all, proj_list_all = build_random_bin_subsets(
            sysmat_l_all,
            proj_l_all,
            iter_arg.osem_subset_num,
            generator,
            device,
        )
        img_jsccsd = osem_joint_mode_dist(sysmat_list_all, proj_list_all, t_list_all, rotmat_all, rotmat_inv_all, img_jsccsd, s_map_arg.j, alpha, rotate_num, device)
        if (i_jsccsd + 1) % iter_arg.save_iter_step == 0:
            img_jsccsd_iter[i_jsccsd // iter_arg.save_iter_step] = img_jsccsd.squeeze().cpu()
            if global_rank == 0:
                elapsed = time.time() - time_start
                print(f"[JSCCSD] Iter: {i_jsccsd+1}/{iter_arg.jsccsd} | Time: {elapsed:.2f}s")

    # 4. 保存
    save_img_dist(img_sc, img_scd, img_jsccd, img_jsccsd, img_sc_iter, img_scd_iter, img_jsccd_iter, img_jsccsd_iter, iter_arg, save_path)
