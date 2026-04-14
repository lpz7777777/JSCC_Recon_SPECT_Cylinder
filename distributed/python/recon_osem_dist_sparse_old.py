import time

import torch
import torch.distributed as dist

try:
    from distributed.python.gpu_mem_report import log_gpu_memory_usage
except ImportError:
    from gpu_mem_report import log_gpu_memory_usage

from compton_sparse_ops import materialize_sparse_event_rows_to_fine


def get_weight_single(sysmat, proj, img_rotate):
    return torch.matmul(sysmat.transpose(0, 1), proj / torch.matmul(sysmat, img_rotate))


def get_weight_compton_sparse(event_block, sysmat_full, img_rotate, sparse_projector):
    t_fine, valid = materialize_sparse_event_rows_to_fine(event_block, sysmat_full, sparse_projector)
    if t_fine.size(0) == 0:
        return torch.zeros_like(img_rotate)
    denom = torch.clamp(torch.matmul(t_fine, img_rotate), min=1e-12)
    weight = torch.matmul(t_fine.transpose(0, 1), 1.0 / denom)
    return torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)


def safe_em_update(img, weight, s_map, eps=1e-12):
    weight_safe = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
    s_map_safe = torch.nan_to_num(s_map, nan=0.0, posinf=0.0, neginf=0.0)
    valid = s_map_safe > eps
    updated = torch.zeros_like(img)
    updated[valid] = img[valid] * torch.clamp(weight_safe[valid], min=0.0) / s_map_safe[valid]
    return updated


def summarize_image_tensor(img):
    img_cpu = img.detach().float().cpu()
    zero_count = int((img_cpu == 0).sum().item())
    return (
        f"min={img_cpu.min().item():.6e} max={img_cpu.max().item():.6e} "
        f"mean={img_cpu.mean().item():.6e} sum={img_cpu.sum().item():.6e} zero={zero_count}/{img_cpu.numel()}"
    )


def osem_bin_mode_dist(sysmat_l_all, proj_l_all, rotmat_all, rotmat_inv_all, img, s_map, rotate_num):
    for sysmat_l, proj_l in zip(sysmat_l_all, proj_l_all):
        weight_local = torch.zeros_like(img)
        for rotate_idx in range(rotate_num):
            for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_l, proj_l, rotmat_all, rotmat_inv_all):
                img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                w_tmp = get_weight_single(sysmat, proj[:, rotate_idx].unsqueeze(1), img_rotate)
                weight_local = weight_local + torch.index_select(w_tmp, 0, rotmat_inv[:, rotate_idx] - 1)

        dist.all_reduce(weight_local, op=dist.ReduceOp.SUM)
        img = safe_em_update(img, weight_local, s_map)
    return img


def osem_list_mode_dist_sparse(t_l_all, sysmat_full_all, sparse_projector_all, rotmat_all, rotmat_inv_all, img, s_map, rotate_num, device):
    for t_l in t_l_all:
        weight_local = torch.zeros_like(img)
        for rotate_idx in range(rotate_num):
            for t_rotate in t_l[rotate_idx]:
                for t_block, sysmat_full, rotmat, rotmat_inv, sparse_projector in zip(t_rotate, sysmat_full_all, rotmat_all, rotmat_inv_all, sparse_projector_all):
                    if t_block.numel() == 0:
                        continue
                    if t_block.device != device:
                        t_block = t_block.to(device, non_blocking=True)
                    img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                    w_tmp = get_weight_compton_sparse(t_block, sysmat_full, img_rotate, sparse_projector)
                    weight_local = weight_local + torch.index_select(w_tmp, 0, rotmat_inv[:, rotate_idx] - 1)

        dist.all_reduce(weight_local, op=dist.ReduceOp.SUM)
        img = safe_em_update(img, weight_local, s_map)
    return img


def osem_joint_mode_dist_sparse(sysmat_l_all, proj_l_all, t_l_all, sysmat_full_all, sparse_projector_all, rotmat_all, rotmat_inv_all, img, s_map, alpha, rotate_num, device):
    for sysmat_l, proj_l, t_l in zip(sysmat_l_all, proj_l_all, t_l_all):
        weight_local = torch.zeros_like(img)
        for rotate_idx in range(rotate_num):
            for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_l, proj_l, rotmat_all, rotmat_inv_all):
                img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                w_s = alpha * get_weight_single(sysmat, proj[:, rotate_idx].unsqueeze(1), img_rotate)
                weight_local = weight_local + torch.index_select(w_s, 0, rotmat_inv[:, rotate_idx] - 1)

        for rotate_idx in range(rotate_num):
            for t_rotate in t_l[rotate_idx]:
                for t_block, sysmat_full, rotmat, rotmat_inv, sparse_projector in zip(t_rotate, sysmat_full_all, rotmat_all, rotmat_inv_all, sparse_projector_all):
                    if t_block.numel() == 0:
                        continue
                    if t_block.device != device:
                        t_block = t_block.to(device, non_blocking=True)
                    img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                    w_c = (2 - alpha) * get_weight_compton_sparse(t_block, sysmat_full, img_rotate, sparse_projector)
                    weight_local = weight_local + torch.index_select(w_c, 0, rotmat_inv[:, rotate_idx] - 1)

        dist.all_reduce(weight_local, op=dist.ReduceOp.SUM)
        img = safe_em_update(img, weight_local, s_map)
    return img


def save_img_dist(img_sc, img_scd, img_jsccd, img_jsccsd, img_sc_iter, img_scd_iter, img_jsccd_iter, img_jsccsd_iter, iter_arg, save_path):
    if dist.get_rank() != 0:
        return

    img_sc.cpu().numpy().astype("float32").tofile(save_path + "Image_SC")
    img_scd.cpu().numpy().astype("float32").tofile(save_path + "Image_SCD")
    img_jsccd.cpu().numpy().astype("float32").tofile(save_path + "Image_JSCCD")
    img_jsccsd.cpu().numpy().astype("float32").tofile(save_path + "Image_JSCCSD")

    img_sc_iter.cpu().numpy().astype("float32").tofile(save_path + "Image_SC_Iter_%d_%d" % (iter_arg.sc, iter_arg.sc / iter_arg.save_iter_step))
    img_scd_iter.cpu().numpy().astype("float32").tofile(save_path + "Image_SCD_Iter_%d_%d" % (iter_arg.jsccd, iter_arg.jsccd / iter_arg.save_iter_step))
    img_jsccd_iter.cpu().numpy().astype("float32").tofile(save_path + "Image_JSCCD_Iter_%d_%d" % (iter_arg.jsccd, iter_arg.jsccd / iter_arg.save_iter_step))
    img_jsccsd_iter.cpu().numpy().astype("float32").tofile(save_path + "Image_JSCCSD_Iter_%d_%d" % (iter_arg.jsccsd, iter_arg.jsccsd / iter_arg.save_iter_step))


def run_recon_osem_dist_sparse(sysmat_l_all, sysmat_full_all, rotmat_all, rotmat_inv_all, proj_l_all, proj_dl_all, t_local_all, sparse_projector_all, iter_arg, s_map_arg, alpha, save_path):
    global_rank = dist.get_rank()
    device = torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")
    pixel_num = s_map_arg.s.size(0)
    rotate_num = rotmat_all[0].size(1)

    img_sc = torch.ones([pixel_num, 1], dtype=torch.float32, device=device)
    img_scd = torch.ones([pixel_num, 1], dtype=torch.float32, device=device)
    img_jsccd = torch.ones([pixel_num, 1], dtype=torch.float32, device=device)
    img_jsccsd = torch.ones([pixel_num, 1], dtype=torch.float32, device=device)

    img_sc_iter = torch.ones([round(iter_arg.sc / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_scd_iter = torch.ones([round(iter_arg.jsccd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_jsccd_iter = torch.ones([round(iter_arg.jsccd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_jsccsd_iter = torch.ones([round(iter_arg.jsccsd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)

    sysmat_full_all = [sysmat_full.to(device) for sysmat_full in sysmat_full_all]
    sparse_projector_all = [projector.to(device) for projector in sparse_projector_all]

    sysmat_list_all = [[None for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.osem_subset_num)]
    proj_list_all = [[None for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.osem_subset_num)]
    proj_d_list_all = [[None for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.osem_subset_num)]
    empty_t = torch.zeros((0, sparse_projector_all[0].coarse_pixel_num + 1), dtype=torch.float32, device=device)
    t_list_all = [[[[empty_t for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.t_divide_num)] for _ in range(rotate_num)] for _ in range(iter_arg.osem_subset_num)]

    local_bin_num = proj_l_all[0].size(0)
    cpnum_list = torch.arange(0, local_bin_num)
    cpnum_list = cpnum_list[torch.randperm(local_bin_num)]
    cpnum_list_chunks = list(torch.chunk(cpnum_list, iter_arg.osem_subset_num, dim=0))

    for energy_idx in range(iter_arg.ene_num):
        for subset_idx in range(iter_arg.osem_subset_num):
            sysmat_list_all[subset_idx][energy_idx] = sysmat_l_all[energy_idx][cpnum_list_chunks[subset_idx], :].to(device)
            proj_list_all[subset_idx][energy_idx] = proj_l_all[energy_idx][cpnum_list_chunks[subset_idx], :].to(device)
            proj_d_list_all[subset_idx][energy_idx] = proj_dl_all[energy_idx][cpnum_list_chunks[subset_idx], :].to(device)

        for rotate_idx in range(rotate_num):
            t_rotate_local = t_local_all[energy_idx][rotate_idx]
            t_subset_chunks = list(torch.chunk(t_rotate_local, iter_arg.osem_subset_num, dim=0))
            for subset_idx in range(len(t_subset_chunks)):
                t_divide_chunks = list(torch.chunk(t_subset_chunks[subset_idx], iter_arg.t_divide_num, dim=0))
                for divide_idx in range(len(t_divide_chunks)):
                    t_list_all[subset_idx][rotate_idx][divide_idx][energy_idx] = t_divide_chunks[divide_idx].to(device)

    del sysmat_l_all, t_local_all
    dist.barrier()
    log_gpu_memory_usage("pre-iter-sparse", device)

    if global_rank == 0:
        print("\n" + "=" * 50)
        print("Starting Distributed Sparse Compton Reconstruction")
        print("=" * 50)

    time_start = time.time()

    save_idx = 0
    for iter_idx in range(iter_arg.sc):
        img_sc = osem_bin_mode_dist(sysmat_list_all, proj_list_all, rotmat_all, rotmat_inv_all, img_sc, s_map_arg.s, rotate_num)
        if (iter_idx + 1) % iter_arg.save_iter_step == 0:
            img_sc_iter[save_idx, :] = img_sc.squeeze().cpu()
            save_idx += 1
            if global_rank == 0:
                elapsed = time.time() - time_start
                print(f"[SC] Iter: {iter_idx + 1}/{iter_arg.sc} | Time: {elapsed:.2f}s")

    save_idx = 0
    for iter_idx in range(iter_arg.jsccd):
        img_scd = osem_bin_mode_dist(sysmat_list_all, proj_d_list_all, rotmat_all, rotmat_inv_all, img_scd, s_map_arg.s, rotate_num)
        if (iter_idx + 1) % iter_arg.save_iter_step == 0:
            img_scd_iter[save_idx, :] = img_scd.squeeze().cpu()
            save_idx += 1
            if global_rank == 0:
                elapsed = time.time() - time_start
                print(f"[SCD] Iter: {iter_idx + 1}/{iter_arg.jsccd} | Time: {elapsed:.2f}s")

    save_idx = 0
    for iter_idx in range(iter_arg.jsccd):
        img_jsccd = osem_list_mode_dist_sparse(t_list_all, sysmat_full_all, sparse_projector_all, rotmat_all, rotmat_inv_all, img_jsccd, s_map_arg.d, rotate_num, device)
        if (iter_idx + 1) % iter_arg.save_iter_step == 0:
            img_jsccd_iter[save_idx, :] = img_jsccd.squeeze().cpu()
            save_idx += 1
            if global_rank == 0:
                elapsed = time.time() - time_start
                print(f"[JSCCD-Sparse] Iter: {iter_idx + 1}/{iter_arg.jsccd} | Time: {elapsed:.2f}s | {summarize_image_tensor(img_jsccd)}")

    save_idx = 0
    for iter_idx in range(iter_arg.jsccsd):
        img_jsccsd = osem_joint_mode_dist_sparse(sysmat_list_all, proj_list_all, t_list_all, sysmat_full_all, sparse_projector_all, rotmat_all, rotmat_inv_all, img_jsccsd, s_map_arg.j, alpha, rotate_num, device)
        if (iter_idx + 1) % iter_arg.save_iter_step == 0:
            img_jsccsd_iter[save_idx, :] = img_jsccsd.squeeze().cpu()
            save_idx += 1
            if global_rank == 0:
                elapsed = time.time() - time_start
                print(f"[JSCCSD-Sparse] Iter: {iter_idx + 1}/{iter_arg.jsccsd} | Time: {elapsed:.2f}s | {summarize_image_tensor(img_jsccsd)}")

    if global_rank == 0:
        print(f"Sparse distributed reconstruction time: {time.time() - time_start:.2f}s")

    save_img_dist(img_sc, img_scd, img_jsccd, img_jsccsd, img_sc_iter, img_scd_iter, img_jsccd_iter, img_jsccsd_iter, iter_arg, save_path)
