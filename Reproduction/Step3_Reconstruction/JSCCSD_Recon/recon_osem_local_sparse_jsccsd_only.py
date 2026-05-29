import time

import torch

from compton_sparse_ops import materialize_sparse_event_rows_to_fine


def get_weight_single(sysmat, proj, img_rotate):
    forward = torch.matmul(sysmat, img_rotate).clamp_min(1e-12)
    return torch.matmul(sysmat.transpose(0, 1), proj / forward)


def get_weight_compton_sparse(event_block, sysmat_full, img_rotate, sparse_projector):
    t_fine, _ = materialize_sparse_event_rows_to_fine(event_block, sysmat_full, sparse_projector)
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


def build_random_bin_subsets(sysmat_all_device, proj_all_device, subset_num, generator):
    local_bin_num = proj_all_device[0].size(0)
    randperm_device = proj_all_device[0].device if proj_all_device[0].is_cuda else "cpu"
    cpnum_list = torch.randperm(local_bin_num, generator=generator, device=randperm_device)
    cpnum_list_chunks = list(torch.chunk(cpnum_list, subset_num, dim=0))

    sysmat_list_all = [[None for _ in range(len(sysmat_all_device))] for _ in range(subset_num)]
    proj_list_all = [[None for _ in range(len(proj_all_device))] for _ in range(subset_num)]

    for energy_idx in range(len(sysmat_all_device)):
        for subset_idx in range(subset_num):
            subset_ids = cpnum_list_chunks[subset_idx]
            sysmat_list_all[subset_idx][energy_idx] = sysmat_all_device[energy_idx][subset_ids, :]
            proj_list_all[subset_idx][energy_idx] = proj_all_device[energy_idx][subset_ids, :]

    return sysmat_list_all, proj_list_all


def osem_joint_mode_local_sparse(
    sysmat_l_all,
    proj_l_all,
    t_l_all,
    sysmat_full_all,
    sparse_projector_all,
    rotmat_all,
    rotmat_inv_all,
    img,
    s_map,
    alpha,
    rotate_num,
    device,
):
    for sysmat_l, proj_l, t_l in zip(sysmat_l_all, proj_l_all, t_l_all):
        weight_local = torch.zeros_like(img)

        for rotate_idx in range(rotate_num):
            for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_l, proj_l, rotmat_all, rotmat_inv_all):
                img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                w_s = alpha * get_weight_single(sysmat, proj[:, rotate_idx].unsqueeze(1), img_rotate)
                weight_local = weight_local + torch.index_select(w_s, 0, rotmat_inv[:, rotate_idx] - 1)

        for rotate_idx in range(rotate_num):
            for t_rotate in t_l[rotate_idx]:
                for t_block, sysmat_full, rotmat, rotmat_inv, sparse_projector in zip(
                    t_rotate,
                    sysmat_full_all,
                    rotmat_all,
                    rotmat_inv_all,
                    sparse_projector_all,
                ):
                    if t_block.numel() == 0:
                        continue
                    if t_block.device != device:
                        t_block = t_block.to(device, non_blocking=(device.type == "cuda"))
                    img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                    w_c = (2 - alpha) * get_weight_compton_sparse(t_block, sysmat_full, img_rotate, sparse_projector)
                    weight_local = weight_local + torch.index_select(w_c, 0, rotmat_inv[:, rotate_idx] - 1)

        img = safe_em_update(img, weight_local, s_map)

    return img


def save_img_local_sparse_jsccsd_only(img_jsccsd, img_jsccsd_iter, iter_arg, save_path):
    img_jsccsd.detach().cpu().numpy().astype("float32").tofile(save_path + "Image_JSCCSD")
    img_jsccsd_iter.detach().cpu().numpy().astype("float32").tofile(
        save_path + "Image_JSCCSD_Iter_%d_%d" % (iter_arg.jsccsd, iter_arg.jsccsd / iter_arg.save_iter_step)
    )
    print("Images saved to local disk.")


def run_recon_osem_local_sparse_jsccsd_only(
    sysmat_all,
    sysmat_full_all,
    rotmat_all,
    rotmat_inv_all,
    proj_all,
    t_local_all,
    sparse_projector_all,
    iter_arg,
    s_map_arg,
    alpha,
    save_path,
    device,
):
    pixel_num = s_map_arg.j.size(0)
    rotate_num = rotmat_all[0].size(1)

    img_jsccsd = torch.ones([pixel_num, 1], dtype=torch.float32, device=device)
    img_jsccsd_iter = torch.ones([round(iter_arg.jsccsd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)

    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(int(getattr(iter_arg, "seed", 20260331)))

    sysmat_all_device = [sysmat.to(device, non_blocking=(device.type == "cuda")) for sysmat in sysmat_all]
    sysmat_full_all_device = [sysmat_full.to(device, non_blocking=(device.type == "cuda")) for sysmat_full in sysmat_full_all]
    proj_all_device = [proj.to(device, non_blocking=(device.type == "cuda")) for proj in proj_all]
    rotmat_all_device = [rotmat.to(device, non_blocking=(device.type == "cuda")) for rotmat in rotmat_all]
    rotmat_inv_all_device = [rotmat_inv.to(device, non_blocking=(device.type == "cuda")) for rotmat_inv in rotmat_inv_all]
    sparse_projector_all_device = [projector.to(device) for projector in sparse_projector_all]
    s_map = s_map_arg.j.to(device, non_blocking=(device.type == "cuda"))

    empty_t = torch.zeros((0, sparse_projector_all_device[0].coarse_pixel_num + 1), dtype=torch.float32, device=device)
    t_list_all = [
        [
            [[empty_t for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.t_divide_num)]
            for _ in range(rotate_num)
        ]
        for _ in range(iter_arg.osem_subset_num)
    ]

    for energy_idx in range(iter_arg.ene_num):
        for rotate_idx in range(rotate_num):
            t_rotate_local = t_local_all[energy_idx][rotate_idx]
            t_subset_chunks = list(torch.chunk(t_rotate_local, iter_arg.osem_subset_num, dim=0))
            for subset_idx in range(len(t_subset_chunks)):
                t_divide_chunks = list(torch.chunk(t_subset_chunks[subset_idx], iter_arg.t_divide_num, dim=0))
                for divide_idx in range(len(t_divide_chunks)):
                    t_list_all[subset_idx][rotate_idx][divide_idx][energy_idx] = t_divide_chunks[divide_idx].to(
                        device,
                        non_blocking=(device.type == "cuda"),
                    )

    print("\n" + "=" * 50)
    print("Starting local sparse JSCCSD-only reconstruction")
    print("=" * 50)

    time_start = time.time()
    save_idx = 0

    for iter_idx in range(iter_arg.jsccsd):
        sysmat_list_all, proj_list_all = build_random_bin_subsets(
            sysmat_all_device,
            proj_all_device,
            iter_arg.osem_subset_num,
            generator,
        )
        img_jsccsd = osem_joint_mode_local_sparse(
            sysmat_list_all,
            proj_list_all,
            t_list_all,
            sysmat_full_all_device,
            sparse_projector_all_device,
            rotmat_all_device,
            rotmat_inv_all_device,
            img_jsccsd,
            s_map,
            alpha,
            rotate_num,
            device,
        )

        if (iter_idx + 1) % iter_arg.save_iter_step == 0:
            img_jsccsd_iter[save_idx, :] = img_jsccsd.squeeze().detach().cpu()
            save_idx += 1
            elapsed = time.time() - time_start
            print(
                f"[JSCCSD-Sparse-Only] Iter: {iter_idx + 1}/{iter_arg.jsccsd} | "
                f"Time: {elapsed:.2f}s | {summarize_image_tensor(img_jsccsd)}"
            )

    print(f"Local sparse JSCCSD-only reconstruction time: {time.time() - time_start:.2f}s")
    save_img_local_sparse_jsccsd_only(img_jsccsd, img_jsccsd_iter, iter_arg, save_path)
