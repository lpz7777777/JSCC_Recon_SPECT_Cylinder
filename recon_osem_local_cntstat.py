import time

import torch


def get_weight_single(sysmat, proj, img_rotate, additive_background=None):
    forward = torch.matmul(sysmat, img_rotate)
    if additive_background is not None:
        forward = forward + additive_background
    forward = forward.clamp_min(1e-12)
    return torch.matmul(sysmat.transpose(0, 1), proj / forward)


def build_random_bin_subsets(
    sysmat_all_device,
    proj_all_device,
    subset_num,
    generator,
    additive_background_all_device=None,
):
    if additive_background_all_device is None:
        additive_background_all_device = [None] * len(sysmat_all_device)
    if len(additive_background_all_device) != len(sysmat_all_device):
        raise ValueError("Additive-background and system-matrix lists must have equal length.")
    local_bin_num = proj_all_device[0].size(0)
    if subset_num == 1:
        # MLEM uses every detector bin, so avoid shuffling and copying the full
        # system matrix on every iteration.
        return [sysmat_all_device], [proj_all_device], [additive_background_all_device]

    cpnum_list = torch.randperm(local_bin_num, generator=generator, device=proj_all_device[0].device)
    cpnum_list_chunks = list(torch.chunk(cpnum_list, subset_num, dim=0))

    sysmat_list_all = [[None for _ in range(len(sysmat_all_device))] for _ in range(subset_num)]
    proj_list_all = [[None for _ in range(len(proj_all_device))] for _ in range(subset_num)]
    background_list_all = [[None for _ in range(len(proj_all_device))] for _ in range(subset_num)]

    for energy_idx in range(len(sysmat_all_device)):
        for subset_idx in range(subset_num):
            subset_ids = cpnum_list_chunks[subset_idx]
            sysmat_list_all[subset_idx][energy_idx] = sysmat_all_device[energy_idx][subset_ids, :]
            proj_list_all[subset_idx][energy_idx] = proj_all_device[energy_idx][subset_ids, :]
            background = additive_background_all_device[energy_idx]
            if background is not None:
                background_list_all[subset_idx][energy_idx] = background[subset_ids, :]

    return sysmat_list_all, proj_list_all, background_list_all


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


def forward_project_local_cntstat(sysmat, rotmat, image, device):
    """Forward-project a total-activity image into measured CntStat units.

    The reconstructed image represents activity summed over all views, while
    GenProj assigns an equal share of that activity to each rotation.  The
    returned projection therefore includes the required ``1 / rotate_num``.
    """
    sysmat_device = sysmat.to(device, non_blocking=(device.type == "cuda"))
    rotmat_device = rotmat.to(device, non_blocking=(device.type == "cuda"))
    image_device = image.to(device, non_blocking=(device.type == "cuda")).reshape(-1, 1)
    projection = torch.empty(
        [sysmat.size(0), rotmat.size(1)], dtype=sysmat.dtype, device="cpu"
    )
    for rotate_idx in range(rotmat.size(1)):
        image_rotate = torch.index_select(
            image_device, 0, rotmat_device[:, rotate_idx] - 1
        )
        projection[:, rotate_idx] = (
            torch.matmul(sysmat_device, image_rotate).squeeze(1).detach().cpu()
            / rotmat.size(1)
        )
    if not torch.isfinite(projection).all() or torch.any(projection < 0):
        raise FloatingPointError("Predicted projection background is negative or non-finite.")
    return projection


def osem_bin_mode_local(
    sysmat_list_all,
    proj_list_all,
    rotmat_all,
    rotmat_inv_all,
    img,
    s_map,
    background_list_all=None,
):
    rotate_num = rotmat_all[0].size(1)
    is_mlem = len(sysmat_list_all) == 1
    if background_list_all is None:
        background_list_all = [
            [None] * len(sysmat_list) for sysmat_list in sysmat_list_all
        ]

    for sysmat_list, proj_list, background_list in zip(
        sysmat_list_all, proj_list_all, background_list_all
    ):
        weight_local = torch.zeros_like(img)

        for rotate_idx in range(rotate_num):
            for sysmat, proj, rotmat, rotmat_inv, background in zip(
                sysmat_list, proj_list, rotmat_all, rotmat_inv_all, background_list
            ):
                if sysmat.size(0) == 0:
                    continue

                img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                background_rotate = None
                if background is not None:
                    # The low-level update uses A*x in total-activity units.
                    # Convert the measured per-view background to that same
                    # scale; the averaged sensitivity then gives the standard
                    # Poisson MLEM update for (A*x)/R + background.
                    background_rotate = (
                        background[:, rotate_idx].unsqueeze(1) * rotate_num
                    )
                weight_tmp = get_weight_single(
                    sysmat,
                    proj[:, rotate_idx].unsqueeze(1),
                    img_rotate,
                    background_rotate,
                )
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
    additive_background_all=None,
):
    """Run local MLEM/OSEM with optional measured additive backgrounds.

    Each additive background must have the same shape and physical CntStat
    scale as its observed projection.  The view-count conversion required by
    the total-activity image convention is handled inside the update.
    """
    pixel_num = s_map_arg.s.size(0)

    img_sc = torch.ones([pixel_num, 1], dtype=torch.float32, device=device)
    img_sc_iter = torch.zeros([iter_arg.sc // iter_arg.save_iter_step, pixel_num], dtype=torch.float32)

    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(int(getattr(iter_arg, "seed", 20260331)))

    sysmat_all_device = [sysmat.to(device, non_blocking=(device.type == "cuda")) for sysmat in sysmat_all]
    proj_all_device = [proj.to(device, non_blocking=(device.type == "cuda")) for proj in proj_all]
    if additive_background_all is None:
        additive_background_all = [None] * len(sysmat_all)
    if len(additive_background_all) != len(sysmat_all):
        raise ValueError("Additive-background and system-matrix lists must have equal length.")
    for energy_idx, (background, projection) in enumerate(
        zip(additive_background_all, proj_all)
    ):
        if background is None:
            continue
        if background.shape != projection.shape:
            raise ValueError(
                f"Additive background {energy_idx} has shape {tuple(background.shape)}; "
                f"expected {tuple(projection.shape)}."
            )
        if not torch.isfinite(background).all() or torch.any(background < 0):
            raise ValueError(
                f"Additive background {energy_idx} contains negative or non-finite values."
            )
    additive_background_all_device = [
        None if background is None else background.to(
            device, non_blocking=(device.type == "cuda")
        )
        for background in additive_background_all
    ]
    rotmat_all_device = [rotmat.to(device, non_blocking=(device.type == "cuda")) for rotmat in rotmat_all]
    rotmat_inv_all_device = [rotmat_inv.to(device, non_blocking=(device.type == "cuda")) for rotmat_inv in rotmat_inv_all]
    s_map = s_map_arg.s.to(device, non_blocking=(device.type == "cuda"))

    print("\n" + "=" * 50)
    algorithm_name = "MLEM" if iter_arg.osem_subset_num == 1 else "OSEM"
    print(f"Starting local single-photon {algorithm_name} reconstruction: {output_name}")
    if any(background is not None for background in additive_background_all):
        print("Using fixed additive projection background in the Poisson denominator.")
    print("=" * 50)

    time_start = time.time()
    id_save = 0

    for iter_idx in range(iter_arg.sc):
        sysmat_list_all, proj_list_all, background_list_all = build_random_bin_subsets(
            sysmat_all_device,
            proj_all_device,
            iter_arg.osem_subset_num,
            generator,
            additive_background_all_device,
        )
        img_sc = osem_bin_mode_local(
            sysmat_list_all,
            proj_list_all,
            rotmat_all_device,
            rotmat_inv_all_device,
            img_sc,
            s_map,
            background_list_all,
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
