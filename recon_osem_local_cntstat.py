import time

import torch


def get_weight_single(sysmat, proj, img_rotate):
    forward = torch.matmul(sysmat, img_rotate).clamp_min(1e-12)
    return torch.matmul(sysmat.transpose(0, 1), proj / forward)


def build_random_bin_subsets(sysmat_all_device, proj_all_device, subset_num, generator):
    local_bin_num = proj_all_device[0].size(0)
    if subset_num == 1:
        # MLEM uses every detector bin, so avoid shuffling and copying the full
        # system matrix on every iteration.
        return [sysmat_all_device], [proj_all_device]

    cpnum_list = torch.randperm(local_bin_num, generator=generator, device=proj_all_device[0].device)
    cpnum_list_chunks = list(torch.chunk(cpnum_list, subset_num, dim=0))

    sysmat_list_all = [[None for _ in range(len(sysmat_all_device))] for _ in range(subset_num)]
    proj_list_all = [[None for _ in range(len(proj_all_device))] for _ in range(subset_num)]

    for energy_idx in range(len(sysmat_all_device)):
        for subset_idx in range(subset_num):
            subset_ids = cpnum_list_chunks[subset_idx]
            sysmat_list_all[subset_idx][energy_idx] = sysmat_all_device[energy_idx][subset_ids, :]
            proj_list_all[subset_idx][energy_idx] = proj_all_device[energy_idx][subset_ids, :]

    return sysmat_list_all, proj_list_all


def compute_subset_sensitivity(sysmat_list, rotmat_inv_all):
    """Compute the exact sensitivity for one detector-bin subset.

    The reconstruction convention averages sensitivity over rotation because
    GenProj distributes the total source count across all views.
    """
    rotate_num = rotmat_inv_all[0].size(1)
    pixel_num = rotmat_inv_all[0].size(0)
    sensitivity = torch.zeros(
        [pixel_num, 1], dtype=sysmat_list[0].dtype, device=sysmat_list[0].device
    )

    for sysmat, rotmat_inv in zip(sysmat_list, rotmat_inv_all):
        base_sensitivity = torch.sum(sysmat, dim=0).unsqueeze(1)
        for rotate_idx in range(rotate_num):
            sensitivity += torch.index_select(
                base_sensitivity, 0, rotmat_inv[:, rotate_idx] - 1
            )

    return sensitivity / rotate_num


def osem_bin_mode_local(sysmat_list_all, proj_list_all, rotmat_all, rotmat_inv_all, img, s_map):
    rotate_num = rotmat_all[0].size(1)
    is_mlem = len(sysmat_list_all) == 1

    for sysmat_list, proj_list in zip(sysmat_list_all, proj_list_all):
        weight_local = torch.zeros_like(img)

        for rotate_idx in range(rotate_num):
            for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_list, proj_list, rotmat_all, rotmat_inv_all):
                if sysmat.size(0) == 0:
                    continue

                img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                weight_tmp = get_weight_single(sysmat, proj[:, rotate_idx].unsqueeze(1), img_rotate)
                weight_local += torch.index_select(weight_tmp, 0, rotmat_inv[:, rotate_idx] - 1)

        subset_s_map = s_map if is_mlem else compute_subset_sensitivity(
            sysmat_list, rotmat_inv_all
        )
        if torch.any(subset_s_map <= 0):
            raise ValueError("A detector-bin subset has non-positive sensitivity values.")
        img = img * weight_local / subset_s_map

    return img


def save_img_local_cntstat(img_sc, img_sc_iter, iter_arg, save_path, output_name="SC"):
    img_sc.detach().cpu().numpy().astype("float32").tofile(save_path + "Image_" + output_name)
    img_sc_iter.detach().cpu().numpy().astype("float32").tofile(
        save_path
        + "Image_%s_Iter_%d_%d"
        % (output_name, iter_arg.sc, iter_arg.sc // iter_arg.save_iter_step)
    )
    print("Images saved to local disk.")


def run_recon_osem_local_cntstat(
    sysmat_all,
    rotmat_all,
    rotmat_inv_all,
    proj_all,
    iter_arg,
    s_map_arg,
    save_path,
    device,
    output_name="SC",
):
    pixel_num = s_map_arg.s.size(0)

    img_sc = torch.ones([pixel_num, 1], dtype=torch.float32, device=device)
    img_sc_iter = torch.zeros([iter_arg.sc // iter_arg.save_iter_step, pixel_num], dtype=torch.float32)

    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(int(getattr(iter_arg, "seed", 20260331)))

    sysmat_all_device = [sysmat.to(device, non_blocking=(device.type == "cuda")) for sysmat in sysmat_all]
    proj_all_device = [proj.to(device, non_blocking=(device.type == "cuda")) for proj in proj_all]
    rotmat_all_device = [rotmat.to(device, non_blocking=(device.type == "cuda")) for rotmat in rotmat_all]
    rotmat_inv_all_device = [rotmat_inv.to(device, non_blocking=(device.type == "cuda")) for rotmat_inv in rotmat_inv_all]
    s_map = s_map_arg.s.to(device, non_blocking=(device.type == "cuda"))

    print("\n" + "=" * 50)
    algorithm_name = "MLEM" if iter_arg.osem_subset_num == 1 else "OSEM"
    print(f"Starting local single-photon {algorithm_name} reconstruction: {output_name}")
    print("=" * 50)

    time_start = time.time()
    id_save = 0

    for iter_idx in range(iter_arg.sc):
        sysmat_list_all, proj_list_all = build_random_bin_subsets(
            sysmat_all_device,
            proj_all_device,
            iter_arg.osem_subset_num,
            generator,
        )
        img_sc = osem_bin_mode_local(
            sysmat_list_all,
            proj_list_all,
            rotmat_all_device,
            rotmat_inv_all_device,
            img_sc,
            s_map,
        )

        if (iter_idx + 1) % iter_arg.save_iter_step == 0:
            img_sc_iter[id_save, :] = img_sc.squeeze().detach().cpu()
            id_save += 1
            elapsed = time.time() - time_start
            if not torch.isfinite(img_sc).all():
                raise FloatingPointError(
                    f"Non-finite image values detected in {output_name} at iteration {iter_idx + 1}."
                )
            print(
                f"[{output_name}] Iter: {iter_idx + 1}/{iter_arg.sc} | "
                f"Time: {elapsed:.2f}s | min={img_sc.min().item():.6e} "
                f"max={img_sc.max().item():.6e} sum={img_sc.sum().item():.6e}"
            )

    save_img_local_cntstat(img_sc, img_sc_iter, iter_arg, save_path, output_name)
    return img_sc
