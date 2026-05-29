"""
Lazy variant of recon_osem_dist_sparse_jsccsd_only.py.

Instead of materializing the full t_compton tensor up front, we store compact
per-event parameters (cpnum1, cpnum2, e1_smeared) and recompute t_compton
on-the-fly during each OSEM iteration.  This trades compute for a large
reduction in GPU memory footprint.

Only the currently-needed t_block is kept on the GPU; it is discarded after
use, so peak GPU memory is O(max_t_block_size × fine_pixel_num) rather than
O(total_events × fine_pixel_num).
"""

import time

import torch
import torch.distributed as dist

try:
    from distributed.python.gpu_mem_report import log_gpu_memory_usage
except ImportError:
    from gpu_mem_report import log_gpu_memory_usage

try:
    from distributed.python.cpu_mem_report import log_cpu_memory_usage
except ImportError:
    from cpu_mem_report import log_cpu_memory_usage

from compton_sparse_ops import (
    pack_sparse_event_rows,
    materialize_sparse_event_rows_to_fine,
)
from process_list_plane_strict import (
    ELECTRON_REST_MEV,
    _build_detector_pos_sigma_sq,
    _compton_theta_from_e1,
    _compute_angle_sigma_ene_strict,
    _compute_angle_sigma_pos_strict,
)


# ---------------------------------------------------------------------------
# Compact helpers
# ---------------------------------------------------------------------------

def _recompute_t_compton_from_compact(cpnum1, cpnum2, e1_smeared,
                                      detector, sparse_projector,
                                      delta_r1, delta_r2, e0, ene_resolution):
    """Recompute the full t_compton tensor from compact per-event parameters.

    Returns
    -------
    t_compton : Tensor [num_events, coarse_pixel_num]  (on device)
        Raw Compton backprojection kernel values, ready for pack_sparse_event_rows.
    """
    device = detector.device
    detector_pos = detector[:, :3]
    detector_sigma_r1_sq = _build_detector_pos_sigma_sq(detector, delta_r1)
    detector_sigma_r2_sq = _build_detector_pos_sigma_sq(detector, delta_r2)

    pos1 = detector_pos[cpnum1 - 1, :]
    pos2 = detector_pos[cpnum2 - 1, :]
    sigma_pos1_sq = detector_sigma_r1_sq[cpnum1 - 1, :]
    sigma_pos2_sq = detector_sigma_r2_sq[cpnum2 - 1, :]

    coor_coarse = sparse_projector.coor_coarse

    vector01 = pos1.unsqueeze(1) - coor_coarse.unsqueeze(0)
    vector12 = (pos2 - pos1).unsqueeze(1)
    distance01 = torch.norm(vector01, dim=2)
    distance12 = torch.norm(vector12, dim=2)

    theta = _compton_theta_from_e1(e1_smeared, e0, ELECTRON_REST_MEV)
    klein_nishina = e0 / (e0 - e1_smeared) + (e0 - e1_smeared) / e0
    beta_cos = (vector01 * vector12).sum(2) / torch.clamp(distance01 * distance12, min=1e-7)
    beta = torch.acos(torch.clamp(beta_cos, -1.0 + 1e-7, 1.0 - 1e-7))

    angle_sigma_ene = _compute_angle_sigma_ene_strict(
        e1_smeared, e0, ene_resolution, ELECTRON_REST_MEV, beta, theta)
    angle_sigma_pos = _compute_angle_sigma_pos_strict(
        vector01, vector12, beta, sigma_pos1_sq, sigma_pos2_sq)
    angle_sigma = torch.sqrt(torch.clamp(angle_sigma_pos ** 2 + angle_sigma_ene ** 2, min=1e-12))

    t_compton = torch.exp(-((beta - theta.unsqueeze(-1)) ** 2) / (2 * angle_sigma ** 2))
    t_compton = t_compton * (klein_nishina.unsqueeze(-1) - torch.sin(beta) ** 2)

    return t_compton


# ---------------------------------------------------------------------------
# Weight functions (same interface as non-lazy version)
# ---------------------------------------------------------------------------

def get_weight_single(sysmat, proj, img_rotate):
    forward = torch.matmul(sysmat, img_rotate).clamp_min(1e-12)
    return torch.matmul(sysmat.transpose(0, 1), proj / forward)


def get_weight_compton_sparse_from_compact(
    compact_block, sysmat_full, img_rotate, sparse_projector,
    detector, delta_r1, delta_r2, e0, ene_resolution,
):
    """Compute Compton weight from compact parameters, fully on-the-fly.

    Events in compact_block are already stability-filtered from preprocessing.
    We recompute t_compton, pack it, and let materialize handle the rest.
    """
    cpnum1 = compact_block[:, 0].long()
    cpnum2 = compact_block[:, 1].long()
    e1_smeared = compact_block[:, 2]

    t_compton = _recompute_t_compton_from_compact(
        cpnum1, cpnum2, e1_smeared,
        detector, sparse_projector,
        delta_r1, delta_r2, e0, ene_resolution,
    )

    # Pack into event_rows format and materialize (same as non-lazy path)
    event_rows = pack_sparse_event_rows(cpnum1, t_compton)
    t_fine, _ = materialize_sparse_event_rows_to_fine(event_rows, sysmat_full, sparse_projector)
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


def run_recon_osem_dist_sparse_jsccsd_only_lazy(
    sysmat_l_all,
    sysmat_full_all,
    rotmat_all,
    rotmat_inv_all,
    proj_l_all,
    compact_local_all,       # [energy][rotate] -> CPU Tensor [N, 3]
    sparse_projector_all,
    detector_all,            # [energy] -> detector on device
    delta_r1,
    delta_r2,
    e0_list,
    ene_resolution_list,
    iter_arg,
    s_map_arg,
    alpha,
    save_path,
):
    """Lazy reconstruction: t_compton is recomputed from compact params each iteration."""
    global_rank = dist.get_rank()
    device = torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")
    pixel_num = s_map_arg.j.size(0)
    rotate_num = rotmat_all[0].size(1)

    img_jsccsd = torch.ones([pixel_num, 1], dtype=torch.float32, device=device)
    img = img_jsccsd
    img_jsccsd_iter = torch.ones([round(iter_arg.jsccsd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)

    sysmat_full_all = [sysmat_full.to(device) for sysmat_full in sysmat_full_all]
    sparse_projector_all = [projector.to(device) for projector in sparse_projector_all]

    # Pre-split compact params by subset × divide (same structure as non-lazy t_list_all)
    # compact_chunks_all[energy][subset][divide] -> CPU Tensor [N, 3]
    compact_chunks_all = []
    for energy_idx in range(iter_arg.ene_num):
        subset_list = []
        for rotate_idx in range(rotate_num):
            compact_rotate = compact_local_all[energy_idx][rotate_idx]
            subset_chunks = list(torch.chunk(compact_rotate, iter_arg.osem_subset_num, dim=0))
            rotate_list = []
            for subset_idx in range(len(subset_chunks)):
                divide_chunks = list(torch.chunk(subset_chunks[subset_idx], iter_arg.t_divide_num, dim=0))
                rotate_list.append(divide_chunks)
            subset_list.append(rotate_list)  # [rotate][subset][divide]
        compact_chunks_all.append(subset_list)

    del compact_local_all
    dist.barrier()
    log_gpu_memory_usage("pre-iter-sparse-jsccsd-only-lazy", device)

    if global_rank == 0:
        print("\n" + "=" * 50)
        print("Starting Distributed Sparse JSCCSD-Only Reconstruction (LAZY)")
        print("=" * 50)

    time_start = time.time()
    save_idx = 0

    for iter_idx in range(iter_arg.jsccsd):
        sysmat_list_all, proj_list_all = build_random_bin_subsets(
            sysmat_l_all,
            proj_l_all,
            iter_arg.osem_subset_num,
            torch.Generator().manual_seed(resolve_iter_seed(iter_arg, global_rank)),
            device,
        )

        # One full update iteration
        for energy_idx in range(iter_arg.ene_num):
            weight_local = torch.zeros_like(img)
            sysmat_full = sysmat_full_all[energy_idx]
            sparse_projector = sparse_projector_all[energy_idx]
            rotmat = rotmat_all[energy_idx]
            rotmat_inv = rotmat_inv_all[energy_idx]
            detector = detector_all[energy_idx]
            e0 = e0_list[energy_idx]
            ene_resolution = ene_resolution_list[energy_idx]

            # --- Single-photon back-projection ---
            for subset_idx in range(iter_arg.osem_subset_num):
                sysmat = sysmat_list_all[subset_idx][energy_idx]
                proj = proj_list_all[subset_idx][energy_idx]
                for rotate_idx in range(rotate_num):
                    img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                    w_s = alpha * get_weight_single(sysmat, proj[:, rotate_idx].unsqueeze(1), img_rotate)
                    weight_local = weight_local + torch.index_select(w_s, 0, rotmat_inv[:, rotate_idx] - 1)

            # --- Compton back-projection (lazy) ---
            for rotate_idx in range(rotate_num):
                for subset_idx in range(iter_arg.osem_subset_num):
                    for divide_idx in range(iter_arg.t_divide_num):
                        compact_block = compact_chunks_all[energy_idx][rotate_idx][subset_idx][divide_idx]
                        if compact_block.numel() == 0:
                            continue
                        compact_gpu = compact_block.to(device, non_blocking=True)

                        img_rotate = torch.index_select(img, 0, rotmat[:, rotate_idx] - 1)
                        w_c = (2 - alpha) * get_weight_compton_sparse_from_compact(
                            compact_gpu, sysmat_full, img_rotate, sparse_projector,
                            detector, delta_r1, delta_r2, e0, ene_resolution,
                        )
                        weight_local = weight_local + torch.index_select(w_c, 0, rotmat_inv[:, rotate_idx] - 1)
                        del compact_gpu, img_rotate, w_c

            dist.all_reduce(weight_local, op=dist.ReduceOp.SUM)
            img = safe_em_update(img, weight_local, s_map_arg.j)

        del sysmat_list_all, proj_list_all

        if (iter_idx + 1) % iter_arg.save_iter_step == 0:
            img_jsccsd_iter[save_idx, :] = img.squeeze().cpu()
            save_idx += 1
            torch.cuda.empty_cache()
            if global_rank == 0:
                elapsed = time.time() - time_start
                print(
                    f"[JSCCSD-Sparse-Only-LAZY] Iter: {iter_idx + 1}/{iter_arg.jsccsd} | "
                    f"Time: {elapsed:.2f}s | {summarize_image_tensor(img)}"
                )

    if global_rank == 0:
        print(f"Sparse JSCCSD-only lazy distributed reconstruction time: {time.time() - time_start:.2f}s")

    _save_img_dist(img, img_jsccsd_iter, iter_arg, save_path)


def _save_img_dist(img_final, img_jsccsd_iter, iter_arg, save_path):
    if dist.get_rank() != 0:
        return
    img_final.cpu().numpy().astype("float32").tofile(save_path + "Image_JSCCSD")
    img_jsccsd_iter.cpu().numpy().astype("float32").tofile(
        save_path + "Image_JSCCSD_Iter_%d_%d" % (iter_arg.jsccsd, iter_arg.jsccsd // iter_arg.save_iter_step)
    )