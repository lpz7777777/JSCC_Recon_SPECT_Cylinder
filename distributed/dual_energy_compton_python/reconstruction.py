"""Distributed MLEM kernels for the current 218/440 JSCC validation chain."""

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from compton_sparse_ops import materialize_sparse_event_rows_to_fine


@dataclass
class ReconResult:
    image: torch.Tensor
    history: torch.Tensor


def _all_reduce_sum(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _safe_update(image, weight, sensitivity, eps=1.0e-12):
    weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
    sensitivity = torch.nan_to_num(sensitivity, nan=0.0, posinf=0.0, neginf=0.0)
    valid = sensitivity > eps
    updated = torch.zeros_like(image)
    updated[valid] = image[valid] * torch.clamp(weight[valid], min=0.0) / sensitivity[valid]
    return updated


def _summary(image):
    values = image.detach().float()
    return (
        f"min={values.min().item():.6e} max={values.max().item():.6e} "
        f"mean={values.mean().item():.6e} sum={values.sum().item():.6e}"
    )


def forward_project_shard(sysmat, rotmat, image):
    """Predict this rank's detector-bin CntStat, including 1/rotate_num."""
    rotate_num = rotmat.size(1)
    projection = torch.empty(
        (sysmat.size(0), rotate_num), dtype=sysmat.dtype, device=sysmat.device
    )
    for view in range(rotate_num):
        rotated = torch.index_select(image, 0, rotmat[:, view] - 1)
        projection[:, view] = torch.matmul(sysmat, rotated).squeeze(1) / rotate_num
    return projection


def run_single_mlem_dist(
    name,
    sysmat,
    projection,
    rotmat,
    rotmat_inv,
    sensitivity,
    iterations,
    save_step,
    rank,
    additive_background=None,
):
    pixel_num = sensitivity.numel()
    rotate_num = rotmat.size(1)
    image = torch.ones((pixel_num, 1), dtype=torch.float32, device=sysmat.device)
    history = torch.empty(
        (iterations // save_step, pixel_num), dtype=torch.float32, device="cpu"
    ) if rank == 0 else torch.empty((0, pixel_num), dtype=torch.float32)
    save_index = 0

    for iteration in range(iterations):
        weight = torch.zeros_like(image)
        for view in range(rotate_num):
            rotated = torch.index_select(image, 0, rotmat[:, view] - 1)
            forward = torch.matmul(sysmat, rotated)
            if additive_background is not None:
                forward = forward + additive_background[:, view:view + 1] * rotate_num
            ratio = projection[:, view:view + 1] / forward.clamp_min(1.0e-12)
            local = torch.matmul(sysmat.transpose(0, 1), ratio)
            weight += torch.index_select(local, 0, rotmat_inv[:, view] - 1)
        _all_reduce_sum(weight)
        image = _safe_update(image, weight, sensitivity)

        if (iteration + 1) % save_step == 0:
            if rank == 0:
                history[save_index] = image.squeeze(1).detach().cpu()
                print(f"[{name}] {iteration + 1}/{iterations} | {_summary(image)}", flush=True)
            save_index += 1
    return ReconResult(image=image, history=history)


def materialize_local_event_blocks(
    packed_rows_by_view,
    sysmat_full,
    projector,
    block_events,
    storage_device,
    rank,
):
    """Materialize normalized K*B rows once and reuse them in every iteration."""
    blocks_by_view = []
    local_events = 0
    for view, packed in enumerate(packed_rows_by_view):
        view_blocks = []
        for start in range(0, packed.size(0), block_events):
            event_block = packed[start:start + block_events].to(sysmat_full.device)
            fine, _ = materialize_sparse_event_rows_to_fine(
                event_block, sysmat_full, projector
            )
            if fine.numel() == 0:
                continue
            local_events += fine.size(0)
            if storage_device == "cpu":
                fine = fine.cpu()
                if torch.cuda.is_available():
                    fine = fine.pin_memory()
            view_blocks.append(fine)
        blocks_by_view.append(view_blocks)
        print(
            f"[rank {rank}] materialized view {view + 1}: "
            f"events={sum(block.size(0) for block in view_blocks)}",
            flush=True,
        )
    return blocks_by_view, local_events


def _compton_weight(blocks, image_rotated, device):
    weight = torch.zeros_like(image_rotated)
    for stored in blocks:
        response = stored if stored.device == device else stored.to(device, non_blocking=True)
        denominator = torch.matmul(response, image_rotated).clamp_min(1.0e-12)
        weight += torch.matmul(response.transpose(0, 1), 1.0 / denominator)
        if stored.device != device:
            del response
    return weight


def run_compton_and_joint_mlem_dist(
    sysmat_440,
    projection_440,
    rotmat,
    rotmat_inv,
    local_event_blocks,
    sensi_s_440,
    sensi_d_440,
    iterations,
    save_step,
    rank,
):
    device = sysmat_440.device
    pixel_num = sensi_d_440.numel()
    rotate_num = rotmat.size(1)
    image_d = torch.ones((pixel_num, 1), dtype=torch.float32, device=device)
    image_j = torch.ones((pixel_num, 1), dtype=torch.float32, device=device)
    snapshots = iterations // save_step
    history_d = torch.empty((snapshots, pixel_num), dtype=torch.float32) if rank == 0 else torch.empty((0, pixel_num))
    history_j = torch.empty((snapshots, pixel_num), dtype=torch.float32) if rank == 0 else torch.empty((0, pixel_num))
    sensitivity_joint = sensi_s_440 + sensi_d_440
    save_index = 0

    for iteration in range(iterations):
        weight_d = torch.zeros_like(image_d)
        weight_j = torch.zeros_like(image_j)
        for view in range(rotate_num):
            ids = rotmat[:, view] - 1
            inverse_ids = rotmat_inv[:, view] - 1

            joint_rotated = torch.index_select(image_j, 0, ids)
            forward = torch.matmul(sysmat_440, joint_rotated).clamp_min(1.0e-12)
            single_local = torch.matmul(
                sysmat_440.transpose(0, 1),
                projection_440[:, view:view + 1] / forward,
            )
            weight_j += torch.index_select(single_local, 0, inverse_ids)

            d_rotated = torch.index_select(image_d, 0, ids)
            d_local = _compton_weight(local_event_blocks[view], d_rotated, device)
            j_local = _compton_weight(local_event_blocks[view], joint_rotated, device)
            weight_d += torch.index_select(d_local, 0, inverse_ids)
            weight_j += torch.index_select(j_local, 0, inverse_ids)

        _all_reduce_sum(weight_d)
        _all_reduce_sum(weight_j)
        image_d = _safe_update(image_d, weight_d, sensi_d_440)
        image_j = _safe_update(image_j, weight_j, sensitivity_joint)

        if (iteration + 1) % save_step == 0:
            if rank == 0:
                history_d[save_index] = image_d.squeeze(1).detach().cpu()
                history_j[save_index] = image_j.squeeze(1).detach().cpu()
                print(
                    f"[D+J] {iteration + 1}/{iterations} | "
                    f"D {_summary(image_d)} | J {_summary(image_j)}",
                    flush=True,
                )
            save_index += 1

    return (
        ReconResult(image=image_d, history=history_d),
        ReconResult(image=image_j, history=history_j),
    )


def save_result(output_dir, name, result, iterations, save_step, rank):
    if rank != 0:
        return
    output_dir = Path(output_dir)
    _write_float32_atomic(
        output_dir / f"Image_{name}", result.image.detach().cpu().numpy()
    )
    _write_float32_atomic(
        output_dir / f"Image_{name}_Iter_{iterations}_{iterations // save_step}",
        result.history.numpy(),
    )


def _write_float32_atomic(path, values):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        np.asarray(values, dtype=np.float32).tofile(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
