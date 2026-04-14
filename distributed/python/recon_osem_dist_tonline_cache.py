import time

import torch
import torch.distributed as dist

try:
    from distributed.python.gpu_mem_report import log_gpu_memory_usage
except ImportError:
    from gpu_mem_report import log_gpu_memory_usage


def get_weight_single(sysmat, proj, img_rotate):
    forward = torch.matmul(sysmat, img_rotate).clamp_min(1e-12)
    return torch.matmul(sysmat.transpose(0, 1), proj / forward)


def get_weight_compton(t_block, img_rotate):
    forward = torch.matmul(t_block, img_rotate).clamp_min(1e-12)
    return torch.matmul(t_block.transpose(0, 1), 1.0 / forward)


def osem_bin_mode_dist(sysmat_list_all, proj_list_all, rotmat_all, rotmat_inv_all, img, s_map):
    rotate_num = rotmat_all[0].size(1)

    for sysmat_list, proj_list in zip(sysmat_list_all, proj_list_all):
        weight_local = torch.zeros_like(img)
        for rotate_idx in range(rotate_num):
            for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_list, proj_list, rotmat_all, rotmat_inv_all):
                if sysmat.size(0) == 0:
                    continue

                img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                weight_tmp = get_weight_single(sysmat, proj[:, rotate_idx].unsqueeze(1), img_rotate)
                weight_local += torch.index_select(weight_tmp, 0, rotmat_inv[:, rotate_idx] - 1)

        dist.all_reduce(weight_local, op=dist.ReduceOp.SUM)
        img = img * weight_local / s_map

    return img


def osem_list_mode_dist_online(t_online_plan_all, providers, rotmat_all, rotmat_inv_all, img, s_map):
    rotate_num = rotmat_all[0].size(1)

    for subset_idx, subset_plan in enumerate(t_online_plan_all):
        weight_local = torch.zeros_like(img)

        for rotate_idx in range(rotate_num):
            for energy_idx, (rotmat, rotmat_inv) in enumerate(zip(rotmat_all, rotmat_inv_all)):
                img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                provider = providers[energy_idx]

                for record in subset_plan[energy_idx][rotate_idx]:
                    t_block = provider.get_t_block(record)
                    weight_tmp = get_weight_compton(t_block, img_rotate)
                    weight_local += torch.index_select(weight_tmp, 0, rotmat_inv[:, rotate_idx] - 1)

        dist.all_reduce(weight_local, op=dist.ReduceOp.SUM)
        img = img * weight_local / s_map

    return img


def osem_joint_mode_dist_online(
    sysmat_list_all,
    proj_list_all,
    t_online_plan_all,
    providers,
    rotmat_all,
    rotmat_inv_all,
    img,
    s_map,
    alpha,
):
    rotate_num = rotmat_all[0].size(1)

    for subset_idx, subset_plan in enumerate(t_online_plan_all):
        weight_local = torch.zeros_like(img)

        for energy_idx, (rotmat, rotmat_inv) in enumerate(zip(rotmat_all, rotmat_inv_all)):
            sysmat = sysmat_list_all[subset_idx][energy_idx]
            proj = proj_list_all[subset_idx][energy_idx]
            provider = providers[energy_idx]

            for rotate_idx in range(rotate_num):
                img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)

                if sysmat.size(0) > 0:
                    weight_single = alpha * get_weight_single(sysmat, proj[:, rotate_idx].unsqueeze(1), img_rotate)
                    weight_local += torch.index_select(weight_single, 0, rotmat_inv[:, rotate_idx] - 1)

                for record in subset_plan[energy_idx][rotate_idx]:
                    t_block = provider.get_t_block(record)
                    weight_compton = (2 - alpha) * get_weight_compton(t_block, img_rotate)
                    weight_local += torch.index_select(weight_compton, 0, rotmat_inv[:, rotate_idx] - 1)

        dist.all_reduce(weight_local, op=dist.ReduceOp.SUM)
        img = img * weight_local / s_map

    return img


def save_img_dist(
    img_sc,
    img_scd,
    img_jsccd,
    img_jsccsd,
    img_sc_iter,
    img_scd_iter,
    img_jsccd_iter,
    img_jsccsd_iter,
    iter_arg,
    save_path,
):
    if dist.get_rank() != 0:
        return

    img_sc.cpu().numpy().astype("float32").tofile(save_path + "Image_SC")
    img_scd.cpu().numpy().astype("float32").tofile(save_path + "Image_SCD")
    img_jsccd.cpu().numpy().astype("float32").tofile(save_path + "Image_JSCCD")
    img_jsccsd.cpu().numpy().astype("float32").tofile(save_path + "Image_JSCCSD")

    img_sc_iter.cpu().numpy().astype("float32").tofile(
        save_path + "Image_SC_Iter_%d_%d" % (iter_arg.sc, iter_arg.sc / iter_arg.save_iter_step)
    )
    img_scd_iter.cpu().numpy().astype("float32").tofile(
        save_path + "Image_SCD_Iter_%d_%d" % (iter_arg.jsccd, iter_arg.jsccd / iter_arg.save_iter_step)
    )
    img_jsccd_iter.cpu().numpy().astype("float32").tofile(
        save_path + "Image_JSCCD_Iter_%d_%d" % (iter_arg.jsccd, iter_arg.jsccd / iter_arg.save_iter_step)
    )
    img_jsccsd_iter.cpu().numpy().astype("float32").tofile(
        save_path + "Image_JSCCSD_Iter_%d_%d" % (iter_arg.jsccsd, iter_arg.jsccsd / iter_arg.save_iter_step)
    )
    print("Images saved to rank 0 disk.")


def run_recon_osem_dist_tonline_cache(
    sysmat_l_all,
    rotmat_all,
    rotmat_inv_all,
    proj_l_all,
    proj_dl_all,
    t_online_plan_all,
    providers,
    iter_arg,
    s_map_arg,
    alpha,
    save_path,
):
    global_rank = dist.get_rank()
    device = torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")
    pixel_num = s_map_arg.s.size(0)

    img_sc = torch.ones([pixel_num, 1], dtype=torch.float32, device=device)
    img_scd = torch.ones([pixel_num, 1], dtype=torch.float32, device=device)
    img_jsccd = torch.ones([pixel_num, 1], dtype=torch.float32, device=device)
    img_jsccsd = torch.ones([pixel_num, 1], dtype=torch.float32, device=device)

    img_sc_iter = torch.ones([round(iter_arg.sc / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_scd_iter = torch.ones([round(iter_arg.jsccd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_jsccd_iter = torch.ones([round(iter_arg.jsccd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_jsccsd_iter = torch.ones([round(iter_arg.jsccsd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)

    sysmat_list_all = [[None for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.osem_subset_num)]
    proj_list_all = [[None for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.osem_subset_num)]
    proj_d_list_all = [[None for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.osem_subset_num)]

    generator = torch.Generator()
    generator.manual_seed(iter_arg.seed)
    local_bin_num = proj_l_all[0].size(0)
    cpnum_list = torch.randperm(local_bin_num, generator=generator)
    cpnum_list_chunks = list(torch.chunk(cpnum_list, iter_arg.osem_subset_num, dim=0))

    for energy_idx in range(iter_arg.ene_num):
        for subset_idx in range(iter_arg.osem_subset_num):
            subset_ids = cpnum_list_chunks[subset_idx]
            sysmat_list_all[subset_idx][energy_idx] = sysmat_l_all[energy_idx][subset_ids, :].to(device, non_blocking=True)
            proj_list_all[subset_idx][energy_idx] = proj_l_all[energy_idx][subset_ids, :].to(device, non_blocking=True)
            proj_d_list_all[subset_idx][energy_idx] = proj_dl_all[energy_idx][subset_ids, :].to(device, non_blocking=True)

    dist.barrier()
    log_gpu_memory_usage("pre-iter", device)

    if global_rank == 0:
        print("\n" + "=" * 50)
        print("Starting distributed reconstruction with online t generation and small cache")
        print("=" * 50)

    time_start = time.time()

    save_idx = 0
    for iter_idx in range(iter_arg.sc):
        img_sc = osem_bin_mode_dist(sysmat_list_all, proj_list_all, rotmat_all, rotmat_inv_all, img_sc, s_map_arg.s)
        if (iter_idx + 1) % iter_arg.save_iter_step == 0:
            img_sc_iter[save_idx, :] = img_sc.squeeze().cpu()
            save_idx += 1
            if global_rank == 0:
                print(f"[SC] Iter: {iter_idx + 1}/{iter_arg.sc} | Time: {time.time() - time_start:.2f}s")

    for iter_idx in range(iter_arg.jsccd):
        img_scd = osem_bin_mode_dist(sysmat_list_all, proj_d_list_all, rotmat_all, rotmat_inv_all, img_scd, s_map_arg.s)
        if (iter_idx + 1) % iter_arg.save_iter_step == 0:
            img_scd_iter[iter_idx // iter_arg.save_iter_step, :] = img_scd.squeeze().cpu()
            if global_rank == 0:
                print(f"[SCD] Iter: {iter_idx + 1}/{iter_arg.jsccd} | Time: {time.time() - time_start:.2f}s")

    for iter_idx in range(iter_arg.jsccd):
        img_jsccd = osem_list_mode_dist_online(t_online_plan_all, providers, rotmat_all, rotmat_inv_all, img_jsccd, s_map_arg.d)
        if (iter_idx + 1) % iter_arg.save_iter_step == 0:
            img_jsccd_iter[iter_idx // iter_arg.save_iter_step, :] = img_jsccd.squeeze().cpu()
            if global_rank == 0:
                print(f"[JSCCD] Iter: {iter_idx + 1}/{iter_arg.jsccd} | Time: {time.time() - time_start:.2f}s")

    for iter_idx in range(iter_arg.jsccsd):
        img_jsccsd = osem_joint_mode_dist_online(
            sysmat_list_all,
            proj_list_all,
            t_online_plan_all,
            providers,
            rotmat_all,
            rotmat_inv_all,
            img_jsccsd,
            s_map_arg.j,
            alpha,
        )
        if (iter_idx + 1) % iter_arg.save_iter_step == 0:
            img_jsccsd_iter[iter_idx // iter_arg.save_iter_step, :] = img_jsccsd.squeeze().cpu()
            if global_rank == 0:
                print(f"[JSCCSD] Iter: {iter_idx + 1}/{iter_arg.jsccsd} | Time: {time.time() - time_start:.2f}s")

    save_img_dist(
        img_sc,
        img_scd,
        img_jsccd,
        img_jsccsd,
        img_sc_iter,
        img_scd_iter,
        img_jsccd_iter,
        img_jsccsd_iter,
        iter_arg,
        save_path,
    )
