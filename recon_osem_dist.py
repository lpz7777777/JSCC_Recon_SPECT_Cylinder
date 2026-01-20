import torch
import os
import time
import torch.distributed as dist


def get_weight_single(sysmat, proj, img_rotate):
    # proj / (sysmat * img)
    return torch.matmul(sysmat.transpose(0, 1), proj / (torch.matmul(sysmat, img_rotate)))


def get_weight_compton(t_block, img_rotate):
    # 1 / (t * img)
    return torch.matmul(t_block.transpose(0, 1), 1.0 / (torch.matmul(t_block, img_rotate)))


def osem_bin_mode_dist(sysmat_l_all, proj_l_all, rotmat_all, rotmat_inv_all, img, s_map, rotate_num, device):
    for sysmat_l, proj_l in zip(sysmat_l_all, proj_l_all):
        weight_local = torch.zeros_like(img)
        for i in range(rotate_num):
            for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_l, proj_l, rotmat_all, rotmat_inv_all):
                img_rotate = torch.index_select(img, 0, rotmat[:, i] - 1)
                w_tmp = get_weight_single(sysmat, proj[:, i].unsqueeze(1), img_rotate)
                weight_local = weight_local +  torch.index_select(w_tmp, 0, rotmat_inv[:, i] - 1)

        # 全局同步权重
        dist.all_reduce(weight_local, op=dist.ReduceOp.SUM)
        img = img * weight_local / s_map
    return img


def osem_list_mode_dist(t_l_all, rotmat_all, rotmat_inv_all, img, s_map, rotate_num, device):
    for t_l in t_l_all:
        weight_local = torch.zeros_like(img)
        for i in range(rotate_num):
            for t_rotate in t_l[i]:
                for t_block, rotmat, rotmat_inv in zip(t_rotate, rotmat_all, rotmat_inv_all):
                    img_rotate = torch.index_select(img, 0, rotmat[:, i] - 1)
                    w_tmp = get_weight_compton(t_block, img_rotate)
                    weight_local += torch.index_select(w_tmp, 0, rotmat_inv[:, i] - 1)

        dist.all_reduce(weight_local, op=dist.ReduceOp.SUM)
        img = img * weight_local / s_map
    return img


def osem_joint_mode_dist(sysmat_l_all, proj_l_all, t_l_all, rotmat_all, rotmat_inv_all, img, s_map, alpha, rotate_num, device):
    for sysmat_l, proj_l, t_l in zip(sysmat_l_all, proj_l_all, t_l_all):
        weight_local = torch.zeros_like(img)
        # 单光子部分
        for i in range(rotate_num):
            for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_l, proj_l, rotmat_all, rotmat_inv_all):
                img_rotate = torch.index_select(img, 0, rotmat[:, i] - 1)
                w_s = alpha * get_weight_single(sysmat, proj[:, i].unsqueeze(1), img_rotate)
                weight_local += torch.index_select(w_s, 0, rotmat_inv[:, i] - 1)
        # 康普顿部分
        for i in range(rotate_num):
            for t_rotate in t_l[i]:
                for t_block, rotmat, rotmat_inv in zip(t_rotate, rotmat_all, rotmat_inv_all):
                    img_rotate = torch.index_select(img, 0, rotmat[:, i] - 1)
                    w_c = (2 - alpha) * get_weight_compton(t_block, img_rotate)
                    weight_local += torch.index_select(w_c, 0, rotmat_inv[:, i] - 1)

        dist.all_reduce(weight_local, op=dist.ReduceOp.SUM)
        img = img * weight_local / s_map
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

    # 1. 初始化图像
    img_sc = torch.ones([pixel_num, 1], dtype=torch.float32).to(device)
    img_scd = torch.ones([pixel_num, 1], dtype=torch.float32).to(device)
    img_jsccd = torch.ones([pixel_num, 1], dtype=torch.float32).to(device)
    img_jsccsd = torch.ones([pixel_num, 1], dtype=torch.float32).to(device)

    img_sc_iter = torch.ones([round(iter_arg.sc / iter_arg.save_iter_step), pixel_num])
    img_scd_iter = torch.ones([round(iter_arg.jsccd / iter_arg.save_iter_step), pixel_num])
    img_jsccd_iter = torch.ones([round(iter_arg.jsccd / iter_arg.save_iter_step), pixel_num])
    img_jsccsd_iter = torch.ones([round(iter_arg.jsccsd / iter_arg.save_iter_step), pixel_num])

    # 2. 数据子集划分 (与原版逻辑一致)
    # 初始化存储子集的列表结构：[subset_num][energy_num]
    sysmat_list_all = [[None for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.osem_subset_num)]
    proj_list_all = [[None for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.osem_subset_num)]
    proj_d_list_all = [[None for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.osem_subset_num)]

    # 初始化 List 模式结构：[subset_num][rotate_num][divide_num][energy_num]
    # 使用空 Tensor 初始化，确保在某些能量/子集无数据时不会报错
    empty_t = torch.zeros((0, pixel_num))
    t_list_all = [[[[empty_t for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.t_divide_num)] for _ in range(rotate_num)] for _ in range(iter_arg.osem_subset_num)]

    # --- 处理 Bin 模式子集 (用于 SC, SCD) ---
    local_bin_num = proj_l_all[0].size(0)
    cpnum_list = torch.arange(0, local_bin_num)
    random_id = torch.randperm(local_bin_num)
    cpnum_list = cpnum_list[random_id]

    # 将本地 Bin 索引划分为多个子集块
    cpnum_list_chunks = list(torch.chunk(cpnum_list, iter_arg.osem_subset_num, dim=0))

    for e in range(iter_arg.ene_num):
        for s in range(iter_arg.osem_subset_num):
            # 提取本地 SysMat 和投影中属于该子集的行
            sysmat_list_all[s][e] = sysmat_l_all[e][cpnum_list_chunks[s], :].to(device)
            proj_list_all[s][e] = proj_l_all[e][cpnum_list_chunks[s], :].to(device)
            proj_d_list_all[s][e] = proj_dl_all[e][cpnum_list_chunks[s], :].to(device)

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

    del sysmat_l_all, t_local_all

    # 3. 开始重建循环
    if global_rank == 0:
        print("\n" + "="*50)
        print("Starting Distributed Reconstruction Process")
        print("="*50)

    time_start = time.time()
    
    # SC
    id_save = 0
    for i_sc in range(iter_arg.sc):
        img_sc = osem_bin_mode_dist(sysmat_list_all, proj_list_all, rotmat_all, rotmat_inv_all, img_sc, s_map_arg.s, rotate_num, device)
        if (i_sc + 1) % iter_arg.save_iter_step == 0:
            img_sc_iter[id_save, :] = img_sc.squeeze().cpu()
            id_save += 1
            if global_rank == 0:
                elapsed = time.time() - time_start
                print(f"[SC] Iter: {i_sc+1}/{iter_arg.sc} | Time: {elapsed:.2f}s")

    # SCD
    for i_scd in range(iter_arg.jsccd):
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
        img_jsccsd = osem_joint_mode_dist(sysmat_list_all, proj_list_all, t_list_all, rotmat_all, rotmat_inv_all, img_jsccsd, s_map_arg.j, alpha, rotate_num, device)
        if (i_jsccsd + 1) % iter_arg.save_iter_step == 0:
            img_jsccsd_iter[i_jsccsd // iter_arg.save_iter_step] = img_jsccsd.squeeze().cpu()
            if global_rank == 0:
                elapsed = time.time() - time_start
                print(f"[JSCCSD] Iter: {i_jsccsd+1}/{iter_arg.jsccsd} | Time: {elapsed:.2f}s")

    # 4. 保存
    save_img_dist(img_sc, img_scd, img_jsccd, img_jsccsd, img_sc_iter, img_scd_iter, img_jsccd_iter, img_jsccsd_iter, iter_arg, save_path)