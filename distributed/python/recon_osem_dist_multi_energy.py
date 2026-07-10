"""Distributed multi-energy multi-output reconstruction core.

Runs a list of :class:`ReconTask` objects over data loaded once by the entry
point. Each task produces one image (single-photon only / Compton only / joint),
sharing the same loaded tensors. The three OSEM mode functions mirror the
math of ``recon_osem_dist_sparse_jsccsd_only.py`` exactly; only the branching
(S-only, D-only, joint) and the per-task scheduling are new.
"""

import time

import torch
import torch.distributed as dist

from compton_sparse_ops import materialize_sparse_event_rows_to_fine


# --------------------------------------------------------------------------- #
# Weight primitives (identical to the existing sparse JSCCSD-only core)        #
# --------------------------------------------------------------------------- #

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
        f"mean={img_cpu.mean().item():.6e} sum={img_cpu.sum().item():.6e} "
        f"zero={zero_count}/{img_cpu.numel()}"
    )


def build_random_bin_subsets(sysmat_l_all, proj_l_all, subset_num, generator, device):
    """Randomly partition the rank-local detector bins into ``subset_num`` views.

    The same partition is reused across all energies within an iteration.
    Returns structures indexed ``[subset_idx][energy_idx]``.
    """
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


def resolve_iter_seed(base_seed, global_rank):
    return int(base_seed) + 1009 * int(global_rank)


# --------------------------------------------------------------------------- #
# Three OSEM mode functions                                                    #
# --------------------------------------------------------------------------- #

def osem_single_dist(sysmat_l_all, proj_l_all, rotmat_all, rotmat_inv_all,
                     img, s_map, rotate_num):
    """Single-photon only OSEM. Energy contributions are summed into one weight."""
    for sysmat_l, proj_l in zip(sysmat_l_all, proj_l_all):
        weight_local = torch.zeros_like(img)

        for rotate_idx in range(rotate_num):
            for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_l, proj_l, rotmat_all, rotmat_inv_all):
                img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                w_s = get_weight_single(sysmat, proj[:, rotate_idx].unsqueeze(1), img_rotate)
                weight_local = weight_local + torch.index_select(w_s, 0, rotmat_inv[:, rotate_idx] - 1)

        dist.all_reduce(weight_local, op=dist.ReduceOp.SUM)
        img = safe_em_update(img, weight_local, s_map)

    return img


def osem_compton_dist(t_l_all, sysmat_full_all, sparse_projector_all,
                      rotmat_all, rotmat_inv_all, img, s_map, rotate_num, device):
    """Compton only OSEM. Energy contributions are summed into one weight."""
    for t_l in t_l_all:
        weight_local = torch.zeros_like(img)

        for rotate_idx in range(rotate_num):
            for t_rotate in t_l[rotate_idx]:
                for t_block, sysmat_full, rotmat, rotmat_inv, sparse_projector in zip(
                    t_rotate, sysmat_full_all, rotmat_all, rotmat_inv_all, sparse_projector_all
                ):
                    if t_block.numel() == 0:
                        continue
                    if t_block.device != device:
                        t_block = t_block.to(device, non_blocking=True)
                    img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                    w_c = get_weight_compton_sparse(t_block, sysmat_full, img_rotate, sparse_projector)
                    weight_local = weight_local + torch.index_select(w_c, 0, rotmat_inv[:, rotate_idx] - 1)

        dist.all_reduce(weight_local, op=dist.ReduceOp.SUM)
        img = safe_em_update(img, weight_local, s_map)

    return img


def osem_joint_dist(sysmat_l_all, proj_l_all, t_l_all, sysmat_full_all,
                    sparse_projector_all, rotmat_all, rotmat_inv_all,
                    img, s_map, alpha, rotate_num, device):
    """Joint single-photon + Compton OSEM (same math as the JSCCSD-only core)."""
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
                    t_rotate, sysmat_full_all, rotmat_all, rotmat_inv_all, sparse_projector_all
                ):
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


# --------------------------------------------------------------------------- #
# Per-task T-matrix layout construction                                        #
# --------------------------------------------------------------------------- #

def build_t_list_all(t_local_subset, sparse_projector_subset, osem_subset_num,
                     t_divide_num, rotate_num, device):
    """Re-chunk a task's per-(compton-energy, rotation) event tensors into the
    ``[subset][rotate][divide][compton_energy]`` layout used by the OSEM loops.

    ``t_local_subset`` is a list indexed by compton-energy (within this task) of
    lists indexed by rotation, each a CPU tensor of packed event rows.
    """
    compton_num = len(t_local_subset)
    coarse_pixel_num_plus1 = sparse_projector_subset[0].coarse_pixel_num + 1
    empty_t = torch.zeros((0, coarse_pixel_num_plus1), dtype=torch.float32, device=device)

    t_list_all = [
        [[[empty_t for _ in range(compton_num)] for _ in range(t_divide_num)]
         for _ in range(rotate_num)]
        for _ in range(osem_subset_num)
    ]

    for c_idx in range(compton_num):
        for rotate_idx in range(rotate_num):
            t_rotate_local = t_local_subset[c_idx][rotate_idx]
            if t_rotate_local.numel() == 0:
                continue
            t_subset_chunks = list(torch.chunk(t_rotate_local, osem_subset_num, dim=0))
            for subset_idx in range(len(t_subset_chunks)):
                t_divide_chunks = list(torch.chunk(t_subset_chunks[subset_idx], t_divide_num, dim=0))
                for divide_idx in range(len(t_divide_chunks)):
                    block = t_divide_chunks[divide_idx]
                    if block.numel() > 0:
                        t_list_all[subset_idx][rotate_idx][divide_idx][c_idx] = block.to(device)

    return t_list_all


# --------------------------------------------------------------------------- #
# Save                                                                         #
# --------------------------------------------------------------------------- #

def save_task_image(img, img_iter_history, task, save_path):
    """Write the final image and the iteration history (rank 0 only)."""
    if dist.get_rank() != 0:
        return

    img.detach().cpu().numpy().astype("float32").tofile(save_path + "Image_" + task.output_name)
    img_iter_history.detach().cpu().numpy().astype("float32").tofile(
        save_path + "Image_%s_Iter_%d_%d" % (task.output_name, task.iter_num,
                                             task.iter_num // task.save_iter_step)
    )


# --------------------------------------------------------------------------- #
# Task driver                                                                  #
# --------------------------------------------------------------------------- #

def run_tasks_multi_energy(loaded, tasks, device, save_path):
    """Execute every ``ReconTask`` sequentially, sharing the loaded data.

    Parameters
    ----------
    loaded : dict
        Output of the entry point's load phase. Expected keys:
        ``sysmat_local_all`` : list per all-energies (rank-bin-sharded)
        ``proj_local_all``   : list per all-energies (rank-bin-sharded)
        ``rotmat_all``       : list per all-energies (GPU)
        ``rotmat_inv_all``   : list per all-energies (GPU)
        ``sensi_s_all``      : list per all-energies [pixel_num,1]
        ``sysmat_full_all``  : dict {e_idx: GPU tensor} for whitelisted energies
        ``sparse_projector_all`` : dict {e_idx: projector}
        ``sensi_d_all``      : dict {e_idx: [pixel_num,1]}
        ``t_local_all``      : dict {e_idx: list[rotate] CPU tensor}
        ``pixel_num``        : int
        ``rotate_num``       : int
        ``osem_subset_num``  : int
        ``t_divide_num``     : int
        ``alpha``            : float
        ``seed``             : int
    tasks : list[ReconTask]
    """
    global_rank = dist.get_rank()

    sysmat_local_all = loaded["sysmat_local_all"]
    proj_local_all = loaded["proj_local_all"]
    rotmat_all = loaded["rotmat_all"]
    rotmat_inv_all = loaded["rotmat_inv_all"]
    sensi_s_all = loaded["sensi_s_all"]
    sysmat_full_all = loaded["sysmat_full_all"]
    sparse_projector_all = loaded["sparse_projector_all"]
    sensi_d_all = loaded["sensi_d_all"]
    t_local_all = loaded["t_local_all"]

    pixel_num = loaded["pixel_num"]
    rotate_num = loaded["rotate_num"]
    osem_subset_num = loaded["osem_subset_num"]
    t_divide_num = loaded["t_divide_num"]
    alpha = loaded["alpha"]

    for task_idx, task in enumerate(tasks):
        if global_rank == 0:
            print("\n" + "=" * 70)
            print(f"[Task {task_idx + 1}/{len(tasks)}] {task.type_tag} mode={task.mode} "
                  f"-> {task.output_name}")
            print("=" * 70)

        # --- gather this task's single-photon slice ---
        s_eidx = list(task.energy_indices)
        sysmat_l_task = [sysmat_local_all[i] for i in s_eidx]
        proj_l_task = [proj_local_all[i] for i in s_eidx]
        rotmat_s_task = [rotmat_all[i] for i in s_eidx]
        rotmat_inv_s_task = [rotmat_inv_all[i] for i in s_eidx]
        sensi_s_task = [sensi_s_all[i] for i in s_eidx]

        # --- gather this task's Compton slice ---
        c_eidx = list(task.compton_energy_indices)
        sysmat_full_task = [sysmat_full_all[i] for i in c_eidx]
        projector_task = [sparse_projector_all[i] for i in c_eidx]
        rotmat_c_task = [rotmat_all[i] for i in c_eidx]
        rotmat_inv_c_task = [rotmat_inv_all[i] for i in c_eidx]
        sensi_d_task = [sensi_d_all[i] for i in c_eidx]
        t_local_task = [t_local_all[i] for i in c_eidx]

        # --- build the s_map for this task ---
        # NOTE: sensi_s_all / sensi_d_all are already global (all-reduced or
        # loaded identically on every rank) from the entry point, so summing a
        # subset of them needs no further reduction.
        if task.mode == "S":
            s_map = _sum_to(sensi_s_task, pixel_num, device)
        elif task.mode == "D":
            s_map = _sum_to(sensi_d_task, pixel_num, device)
        else:  # "J"
            s = _sum_to(sensi_s_task, pixel_num, device)
            d = _sum_to(sensi_d_task, pixel_num, device)
            s_map = alpha * s + (2 - alpha) * d

        if global_rank == 0:
            s_map_cpu = s_map.detach().float().cpu()
            print(f"s_map: min={s_map_cpu.min().item():.6e} max={s_map_cpu.max().item():.6e} "
                  f"sum={s_map_cpu.sum().item():.6e}")

        # --- init image ---
        img = torch.ones([pixel_num, 1], dtype=torch.float32, device=device)
        num_snapshots = task.iter_num // task.save_iter_step
        img_iter_history = torch.zeros([num_snapshots, pixel_num], dtype=torch.float32)

        generator = torch.Generator(device=device)
        generator.manual_seed(resolve_iter_seed(loaded["seed"], global_rank) + task_idx)

        time_start = time.time()
        save_idx = 0

        for iter_idx in range(task.iter_num):
            if task.mode == "S":
                sysmat_list_all, proj_list_all = build_random_bin_subsets(
                    sysmat_l_task, proj_l_task, osem_subset_num, generator, device)
                img = osem_single_dist(
                    sysmat_list_all, proj_list_all,
                    rotmat_s_task, rotmat_inv_s_task,
                    img, s_map, rotate_num)
                del sysmat_list_all, proj_list_all
            elif task.mode == "D":
                t_list_all = build_t_list_all(
                    t_local_task, projector_task, osem_subset_num,
                    t_divide_num, rotate_num, device)
                img = osem_compton_dist(
                    t_list_all, sysmat_full_task, projector_task,
                    rotmat_c_task, rotmat_inv_c_task,
                    img, s_map, rotate_num, device)
                del t_list_all
            else:  # "J"
                sysmat_list_all, proj_list_all = build_random_bin_subsets(
                    sysmat_l_task, proj_l_task, osem_subset_num, generator, device)
                t_list_all = build_t_list_all(
                    t_local_task, projector_task, osem_subset_num,
                    t_divide_num, rotate_num, device)
                img = osem_joint_dist(
                    sysmat_list_all, proj_list_all, t_list_all,
                    sysmat_full_task, projector_task,
                    rotmat_s_task, rotmat_inv_s_task,
                    img, s_map, alpha, rotate_num, device)
                del sysmat_list_all, proj_list_all, t_list_all

            if (iter_idx + 1) % task.save_iter_step == 0:
                img_iter_history[save_idx, :] = img.squeeze().detach().cpu()
                save_idx += 1
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if global_rank == 0:
                    elapsed = time.time() - time_start
                    print(
                        f"[{task.output_name}] Iter {iter_idx + 1}/{task.iter_num} | "
                        f"{elapsed:.2f}s | {summarize_image_tensor(img)}"
                    )

        if global_rank == 0:
            print(f"[{task.output_name}] total time {time.time() - time_start:.2f}s")

        save_task_image(img, img_iter_history, task, save_path)

        # free task-local intermediates before the next task
        del img, img_iter_history, s_map
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if global_rank == 0:
        print("\nAll tasks finished.")


def _sum_to(tensor_list, pixel_num, device):
    """Sum a list of [pixel_num, 1] tensors onto ``device`` (empty list -> zeros)."""
    if not tensor_list:
        return torch.zeros([pixel_num, 1], dtype=torch.float32, device=device)
    acc = torch.zeros([pixel_num, 1], dtype=torch.float32, device=device)
    for t in tensor_list:
        acc = acc + t.to(device, non_blocking=(device.type == "cuda"))
    return acc
