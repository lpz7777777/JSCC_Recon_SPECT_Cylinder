import os
import time

import torch
import torch.multiprocessing as mp

from compton_sparse_ops import (
    project_fine_rows_to_coarse_adjoint,
    unpack_sparse_event_rows,
    upsample_coarse_rows_to_fine,
)
from recon_osem_plane import (
    get_gpu_memory_usage,
    get_t_budget_bytes_per_gpu,
    get_weight_single,
    osem_bin_mode,
    save_img,
    split_tensor_even,
)


def get_weight_compton_sparse(event_block, sysmat_full, img_fine, sparse_projector):
    device = img_fine.device
    t_gpu = event_block.to(device, non_blocking=True)
    projector_gpu = sparse_projector if sparse_projector.coor_coarse.device == device else sparse_projector.to(device)
    sysmat_gpu = sysmat_full if sysmat_full.device == device else sysmat_full.to(device, non_blocking=True)

    if t_gpu.size(0) == 0:
        return torch.zeros_like(img_fine)

    cpnum1, t_compton = unpack_sparse_event_rows(t_gpu)
    cpnum1 = torch.clamp(cpnum1 - 1, min=0, max=sysmat_gpu.size(0) - 1)

    unique_cpnum1, inverse = torch.unique(cpnum1, sorted=True, return_inverse=True)
    img_row = img_fine.transpose(0, 1)
    weight_fine = torch.zeros_like(img_fine)

    for group_idx, cp_idx in enumerate(unique_cpnum1):
        group_mask = inverse == group_idx
        if not torch.any(group_mask):
            continue

        sysmat_row = sysmat_gpu[cp_idx, :].unsqueeze(0)
        img_modulated = img_row * sysmat_row
        coarse_backproj = project_fine_rows_to_coarse_adjoint(img_modulated, projector_gpu).transpose(0, 1)

        denom = torch.clamp(torch.matmul(t_compton[group_mask, :], coarse_backproj), min=1e-12)
        weight_coarse = torch.matmul(t_compton[group_mask, :].transpose(0, 1), 1.0 / denom)
        weight_upsampled = upsample_coarse_rows_to_fine(weight_coarse.transpose(0, 1), projector_gpu).transpose(0, 1)
        weight_fine = weight_fine + sysmat_row.transpose(0, 1) * weight_upsampled

    weight_fine = torch.nan_to_num(weight_fine, nan=0.0, posinf=0.0, neginf=0.0)

    if event_block.device.type == "cpu":
        del t_gpu

    return weight_fine


def safe_em_update(img, weight, s_map, eps=1e-12):
    weight_safe = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
    s_map_safe = torch.nan_to_num(s_map, nan=0.0, posinf=0.0, neginf=0.0)
    valid = s_map_safe > eps
    updated = torch.zeros_like(img)
    updated[valid] = img[valid] * torch.clamp(weight_safe[valid], min=0.0) / s_map_safe[valid]
    return updated


def osem_list_mode_sparse(t_list_all, sysmat_all, sparse_projector_all, rotmat_all, rotmat_inv_all, img, s_map, rotate_num, model_denoiser=None):
    n = int(img.shape[0] ** 0.5)
    for t_list in t_list_all:
        weight_compton = torch.zeros_like(img)
        for rotate_idx in range(rotate_num):
            for t in t_list[rotate_idx]:
                for t_tmp, sysmat_full, rotmat, rotmat_inv, sparse_projector in zip(t, sysmat_all, rotmat_all, rotmat_inv_all, sparse_projector_all):
                    if t_tmp.numel() == 0:
                        continue
                    img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                    weight_tmp = get_weight_compton_sparse(t_tmp, sysmat_full, img_rotate, sparse_projector)
                    weight_compton = weight_compton + torch.index_select(weight_tmp, 0, rotmat_inv[:, rotate_idx] - 1)
        img = safe_em_update(img, weight_compton, s_map)

        if model_denoiser is not None:
            img_2d = img.view(1, 1, n, n)
            img = model_denoiser(img_2d).view(n * n, 1)
            img = torch.clamp(img, min=0)

    return img


def osem_joint_mode_sparse(sysmat_list_all, proj_list_all, t_list_all, sysmat_all, sparse_projector_all, rotmat_all, rotmat_inv_all, img, s_map, alpha, rotate_num, model_denoiser=None):
    n = int(img.shape[0] ** 0.5)
    for sysmat_list, proj_list, t_list in zip(sysmat_list_all, proj_list_all, t_list_all):
        weight_compton = torch.zeros_like(img)
        weight_single = torch.zeros_like(img)
        for rotate_idx in range(rotate_num):
            for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_list, proj_list, rotmat_all, rotmat_inv_all):
                img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                weight_single_tmp = get_weight_single(sysmat, proj[:, rotate_idx].unsqueeze(1), img_rotate)
                weight_single = weight_single + torch.index_select(weight_single_tmp, 0, rotmat_inv[:, rotate_idx] - 1)

            for t in t_list[rotate_idx]:
                for t_tmp, sysmat_full, rotmat, rotmat_inv, sparse_projector in zip(t, sysmat_all, rotmat_all, rotmat_inv_all, sparse_projector_all):
                    if t_tmp.numel() == 0:
                        continue
                    img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                    weight_compton_tmp = get_weight_compton_sparse(t_tmp, sysmat_full, img_rotate, sparse_projector)
                    weight_compton = weight_compton + torch.index_select(weight_compton_tmp, 0, rotmat_inv[:, rotate_idx] - 1)

        weight = (2 - alpha) * weight_compton + alpha * weight_single
        img = safe_em_update(img, weight, s_map)

        if model_denoiser is not None:
            img_2d = img.view(1, 1, n, n)
            img = model_denoiser(img_2d).view(n * n, 1)
            img = torch.clamp(img, min=0)

    return img


def osem_list_mode_mp_sparse(t_list_all, sysmat_all, sparse_projector_all, rotmat_all, rotmat_inv_all, iter_num, rank, img_queue, weight_queue, rotate_num, model_denoiser=None):
    with torch.no_grad():
        print(f"Sparse List Mode OSEM Rank{rank} Starts")
        sysmat_all = [sysmat.to(f"cuda:{rank}", non_blocking=True) for sysmat in sysmat_all]
        sparse_projector_all = [projector.to(f"cuda:{rank}") for projector in sparse_projector_all]
        for _ in range(iter_num):
            for t_list in t_list_all:
                img = img_queue.get()
                weight_compton = torch.zeros_like(img)
                img = img.to(f"cuda:{rank}")

                for rotate_idx in range(rotate_num):
                    for t in t_list[rotate_idx]:
                        for t_tmp, sysmat_full, rotmat, rotmat_inv, sparse_projector in zip(t, sysmat_all, rotmat_all, rotmat_inv_all, sparse_projector_all):
                            if t_tmp.numel() == 0:
                                continue
                            img_rotate = torch.index_select(img, 0, (rotmat[:, rotate_idx] - 1).to(f"cuda:{rank}"))
                            weight_tmp = get_weight_compton_sparse(t_tmp, sysmat_full, img_rotate, sparse_projector).to("cuda:0")
                            weight_compton = weight_compton + torch.index_select(weight_tmp, 0, rotmat_inv[:, rotate_idx] - 1)

                weight_queue.put(weight_compton)


def run_recon_osem_sparse(sysmat_all, rotmat_all, rotmat_inv_all, proj_all, proj_d_all, t_all, sparse_projector_all, iter_arg, s_map_arg, alpha, save_path, num_gpus, model_denoiser=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pixel_num = sysmat_all[0].size(1)
    rotate_num = rotmat_all[0].size(1)

    img_sc = torch.ones([pixel_num, 1], dtype=torch.float32).to("cuda:0", non_blocking=True)
    img_scd = torch.ones([pixel_num, 1], dtype=torch.float32).to("cuda:0", non_blocking=True)
    img_jsccd = torch.ones([pixel_num, 1], dtype=torch.float32).to("cuda:0", non_blocking=True)
    img_jsccsd = torch.ones([pixel_num, 1], dtype=torch.float32).to("cuda:0", non_blocking=True)

    img_sc_iter = torch.ones([round(iter_arg.sc / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_scd_iter = torch.ones([round(iter_arg.jsccd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_jsccd_iter = torch.ones([round(iter_arg.jsccd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_jsccsd_iter = torch.ones([round(iter_arg.jsccsd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)

    sysmat_list_all = [[[] for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.osem_subset_num)]
    proj_list_all = [[[] for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.osem_subset_num)]
    proj_d_list_all = [[[] for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.osem_subset_num)]
    if num_gpus == 1:
        t_list_all = [[[[[] for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.t_divide_num)] for _ in range(rotate_num)] for _ in range(iter_arg.osem_subset_num)]
    else:
        t_list_all = [[[[[[] for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.t_divide_num)] for _ in range(rotate_num)] for _ in range(iter_arg.osem_subset_num)] for _ in range(num_gpus)]

    cpnum_list = torch.arange(0, proj_all[0].size(dim=0))
    cpnum_list = cpnum_list[torch.randperm(proj_all[0].size(dim=0))]
    cpnum_list = split_tensor_even(cpnum_list, iter_arg.osem_subset_num, dim=0)

    gpu_t_budget_bytes = get_t_budget_bytes_per_gpu(num_gpus)
    gpu_vram_used = [0] * num_gpus

    for e, (sysmat, proj, proj_d, t) in enumerate(zip(sysmat_all, proj_all, proj_d_all, t_all)):
        if num_gpus == 1:
            for rotate_idx in range(rotate_num):
                t_chunks = split_tensor_even(t[rotate_idx], iter_arg.osem_subset_num, dim=0)
                for subset_idx in range(iter_arg.osem_subset_num):
                    t_sub_chunks = split_tensor_even(t_chunks[subset_idx], iter_arg.t_divide_num, dim=0)
                    for divide_idx in range(iter_arg.t_divide_num):
                        chunk = t_sub_chunks[divide_idx]
                        size_bytes = chunk.nelement() * chunk.element_size()
                        if gpu_vram_used[0] + size_bytes < gpu_t_budget_bytes[0]:
                            t_list_all[subset_idx][rotate_idx][divide_idx][e] = chunk.to("cuda:0", non_blocking=True)
                            gpu_vram_used[0] += size_bytes
                        else:
                            t_list_all[subset_idx][rotate_idx][divide_idx][e] = chunk.pin_memory()

            rotmat_all[e] = rotmat_all[e].to("cuda:0", non_blocking=True)
            rotmat_inv_all[e] = rotmat_inv_all[e].to("cuda:0", non_blocking=True)
        else:
            for rotate_idx in range(rotate_num):
                t_chunks = split_tensor_even(t[rotate_idx], num_gpus, dim=0)
                for gpu_id in range(num_gpus):
                    t_gpu_chunks = split_tensor_even(t_chunks[gpu_id], iter_arg.osem_subset_num, dim=0)
                    for subset_idx in range(iter_arg.osem_subset_num):
                        t_sub_chunks = split_tensor_even(t_gpu_chunks[subset_idx], iter_arg.t_divide_num, dim=0)
                        for divide_idx in range(iter_arg.t_divide_num):
                            chunk = t_sub_chunks[divide_idx]
                            size_bytes = chunk.nelement() * chunk.element_size()
                            if gpu_vram_used[gpu_id] + size_bytes < gpu_t_budget_bytes[gpu_id]:
                                t_list_all[gpu_id][subset_idx][rotate_idx][divide_idx][e] = chunk.to(f"cuda:{gpu_id}", non_blocking=True)
                                gpu_vram_used[gpu_id] += size_bytes
                            else:
                                t_list_all[gpu_id][subset_idx][rotate_idx][divide_idx][e] = chunk.pin_memory()

            rotmat_all[e] = rotmat_all[e].to("cuda:0", non_blocking=True)
            rotmat_inv_all[e] = rotmat_inv_all[e].to("cuda:0", non_blocking=True)

        print(
            "Sparse t loading finished. GPU usage (%) = "
            f"{[round(100 * gpu_vram_used[gpu_id] / gpu_t_budget_bytes[gpu_id], 2) for gpu_id in range(num_gpus)]}"
        )

        for subset_idx in range(iter_arg.osem_subset_num):
            sysmat_list_all[subset_idx][e] = sysmat[cpnum_list[subset_idx], :].to("cuda:0", non_blocking=True)
            proj_list_all[subset_idx][e] = proj[cpnum_list[subset_idx], :].to("cuda:0", non_blocking=True)
            proj_d_list_all[subset_idx][e] = proj_d[cpnum_list[subset_idx], :].to("cuda:0", non_blocking=True)

    if num_gpus == 1:
        sysmat_all = [sysmat.to("cuda:0", non_blocking=True) for sysmat in sysmat_all]

    del t_all

    s_map_arg.s = s_map_arg.s.to("cuda:0", non_blocking=True)
    s_map_arg.d = s_map_arg.d.to("cuda:0", non_blocking=True)
    s_map_arg.j = alpha * s_map_arg.s + (2 - alpha) * s_map_arg.d

    if model_denoiser is not None:
        model_denoiser.to("cuda:0")

    get_gpu_memory_usage(num_gpus)
    time_start = time.time()

    print("Self-Collimation OSEM starts")
    id_save = 0
    for id_iter_sc in range(iter_arg.sc):
        img_sc = osem_bin_mode(sysmat_list_all, proj_list_all, rotmat_all, rotmat_inv_all, img_sc, s_map_arg.s, rotate_num, model_denoiser)
        if (id_iter_sc + 1) % iter_arg.save_iter_step == 0:
            img_sc_iter[id_save, :] = torch.squeeze(img_sc).cpu()
            id_save += 1
            print("SC Iteration", str(id_iter_sc + 1), "ends, time used:", time.time() - time_start, "s")
    torch.cuda.empty_cache()

    print("SC-D OSEM starts")
    id_save = 0
    for id_iter_scd in range(iter_arg.jsccd):
        img_scd = osem_bin_mode(sysmat_list_all, proj_d_list_all, rotmat_all, rotmat_inv_all, img_scd, s_map_arg.s, rotate_num, model_denoiser)
        if (id_iter_scd + 1) % iter_arg.save_iter_step == 0:
            img_scd_iter[id_save, :] = torch.squeeze(img_scd).cpu()
            id_save += 1
            print("SC-D Iteration", str(id_iter_scd + 1), "ends, time used:", time.time() - time_start, "s")
    torch.cuda.empty_cache()

    if num_gpus == 1:
        print("JSCC-D Sparse OSEM starts")
        id_save = 0
        sparse_projector_gpu = [projector.to("cuda:0") for projector in sparse_projector_all]
        for id_iter_jsccd in range(iter_arg.jsccd):
            img_jsccd = osem_list_mode_sparse(t_list_all, sysmat_all, sparse_projector_gpu, rotmat_all, rotmat_inv_all, img_jsccd, s_map_arg.d, rotate_num, model_denoiser)
            if (id_iter_jsccd + 1) % iter_arg.save_iter_step == 0:
                img_jsccd_iter[id_save, :] = torch.squeeze(img_jsccd).cpu()
                id_save += 1
                print("JSCC-D Sparse Iteration", str(id_iter_jsccd + 1), "ends, time used:", time.time() - time_start, "s")
        torch.cuda.empty_cache()

        print("JSCC-SD Sparse OSEM starts")
        id_save = 0
        for id_iter_jsccsd in range(iter_arg.jsccsd):
            img_jsccsd = osem_joint_mode_sparse(sysmat_list_all, proj_list_all, t_list_all, sysmat_all, sparse_projector_gpu, rotmat_all, rotmat_inv_all, img_jsccsd, s_map_arg.j, alpha, rotate_num, model_denoiser)
            if (id_iter_jsccsd + 1) % iter_arg.save_iter_step == 0:
                img_jsccsd_iter[id_save, :] = torch.squeeze(img_jsccsd).cpu()
                id_save += 1
                print("JSCC-SD Sparse Iteration", str(id_iter_jsccsd + 1), "ends, time used:", time.time() - time_start, "s")
        torch.cuda.empty_cache()
    else:
        img_queue = mp.Queue()
        weight_queue = mp.Queue()

        print("JSCC-D Sparse OSEM starts (With Multiprocessing)")
        id_save = 0
        processes = []
        for gpu_id in range(num_gpus):
            process = mp.Process(
                target=osem_list_mode_mp_sparse,
                args=(t_list_all[gpu_id], sysmat_all, sparse_projector_all, rotmat_all, rotmat_inv_all, iter_arg.jsccd, gpu_id, img_queue, weight_queue, rotate_num, model_denoiser),
            )
            process.start()
            processes.append(process)

        for id_iter_jsccd in range(iter_arg.jsccd):
            for _ in range(iter_arg.osem_subset_num):
                weight_compton = torch.zeros_like(img_jsccd)
                for _ in range(num_gpus):
                    img_queue.put(img_jsccd)
                for _ in range(num_gpus):
                    weight_compton = weight_compton + weight_queue.get()
                img_jsccd = safe_em_update(img_jsccd, weight_compton, s_map_arg.d)
                torch.cuda.empty_cache()

            if (id_iter_jsccd + 1) % iter_arg.save_iter_step == 0:
                img_jsccd_iter[id_save, :] = torch.squeeze(img_jsccd).cpu()
                id_save += 1
                print("JSCC-D Sparse Iteration", str(id_iter_jsccd + 1), "ends, time used:", time.time() - time_start, "s")

        for process in processes:
            process.join()
        processes.clear()

        print("JSCC-SD Sparse OSEM starts (With Multiprocessing)")
        id_save = 0
        for gpu_id in range(num_gpus):
            process = mp.Process(
                target=osem_list_mode_mp_sparse,
                args=(t_list_all[gpu_id], sysmat_all, sparse_projector_all, rotmat_all, rotmat_inv_all, iter_arg.jsccsd, gpu_id, img_queue, weight_queue, rotate_num, model_denoiser),
            )
            process.start()
            processes.append(process)

        for id_iter_jsccsd in range(iter_arg.jsccsd):
            for sysmat_list, proj_list in zip(sysmat_list_all, proj_list_all):
                weight_compton = torch.zeros_like(img_jsccsd)
                weight_single = torch.zeros_like(img_jsccsd)
                for _ in range(num_gpus):
                    img_queue.put(img_jsccsd)
                for _ in range(num_gpus):
                    weight_compton = weight_compton + weight_queue.get()

                for rotate_idx in range(rotate_num):
                    for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_list, proj_list, rotmat_all, rotmat_inv_all):
                        img_rotate = torch.index_select(img_jsccsd, 0, rotmat[:, rotate_idx] - 1)
                        weight_single_tmp = get_weight_single(sysmat, proj[:, rotate_idx].unsqueeze(1), img_rotate)
                        weight_single = weight_single + torch.index_select(weight_single_tmp, 0, rotmat_inv[:, rotate_idx] - 1)

                weight = alpha * weight_single + (2 - alpha) * weight_compton
                img_jsccsd = safe_em_update(img_jsccsd, weight, s_map_arg.j)
                torch.cuda.empty_cache()

            if (id_iter_jsccsd + 1) % iter_arg.save_iter_step == 0:
                img_jsccsd_iter[id_save, :] = torch.squeeze(img_jsccsd).cpu()
                id_save += 1
                print("JSCC-SD Sparse Iteration", str(id_iter_jsccsd + 1), "ends, time used:", time.time() - time_start, "s")

        for process in processes:
            process.join()

    save_img(img_sc, img_scd, img_jsccd, img_jsccsd, img_sc_iter, img_scd_iter, img_jsccd_iter, img_jsccsd_iter, iter_arg, save_path)
