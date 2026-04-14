import math
from dataclasses import dataclass

import torch


RADIUS_TOL_MM = 1e-4
Z_TOL_MM = 1e-4


@dataclass
class ComptonSparseProjector:
    theta_stride: int
    z_stride: int
    ring_strides: tuple
    fine_layer_num: int
    fine_z_num: int
    coarse_layer_num: int
    coarse_z_num: int
    fine_pixel_num: int
    coarse_pixel_num: int
    layer_reduce: torch.Tensor
    z_reduce: torch.Tensor
    layer_expand: torch.Tensor
    z_expand: torch.Tensor
    coor_coarse: torch.Tensor

    def to(self, device):
        return ComptonSparseProjector(
            theta_stride=self.theta_stride,
            z_stride=self.z_stride,
            ring_strides=self.ring_strides,
            fine_layer_num=self.fine_layer_num,
            fine_z_num=self.fine_z_num,
            coarse_layer_num=self.coarse_layer_num,
            coarse_z_num=self.coarse_z_num,
            fine_pixel_num=self.fine_pixel_num,
            coarse_pixel_num=self.coarse_pixel_num,
            layer_reduce=self.layer_reduce.to(device),
            z_reduce=self.z_reduce.to(device),
            layer_expand=self.layer_expand.to(device),
            z_expand=self.z_expand.to(device),
            coor_coarse=self.coor_coarse.to(device),
        )


def _collect_unique_consecutive(values, tol):
    unique_values = [values[0]]
    counts = [1]
    for idx in range(1, values.numel()):
        if torch.abs(values[idx] - unique_values[-1]) <= tol:
            counts[-1] += 1
        else:
            unique_values.append(values[idx])
            counts.append(1)
    return torch.stack(unique_values), counts


def _infer_polar_layout(coor_full):
    if coor_full.ndim != 2 or coor_full.size(1) != 3:
        raise ValueError("coor_full must have shape [pixel_num, 3].")
    if coor_full.size(0) == 0:
        raise ValueError("coor_full is empty.")

    z_values, z_counts = _collect_unique_consecutive(coor_full[:, 2], Z_TOL_MM)
    if len(set(z_counts)) != 1:
        raise ValueError("coor_full does not contain a consistent z-plane layout.")

    fine_z_num = len(z_counts)
    fine_layer_num = z_counts[0]
    first_layer = coor_full[:fine_layer_num, :2]
    radii = torch.sqrt(torch.sum(first_layer ** 2, dim=1))

    ring_lengths = []
    ring_radii = []
    start_idx = 0
    while start_idx < fine_layer_num:
        current_radius = radii[start_idx]
        end_idx = start_idx + 1
        while end_idx < fine_layer_num and torch.abs(radii[end_idx] - current_radius) <= RADIUS_TOL_MM:
            end_idx += 1
        ring_lengths.append(end_idx - start_idx)
        ring_radii.append(torch.mean(radii[start_idx:end_idx]))
        start_idx = end_idx

    return {
        "z_values": z_values,
        "fine_z_num": fine_z_num,
        "fine_layer_num": fine_layer_num,
        "ring_lengths": ring_lengths,
        "ring_radii": torch.stack(ring_radii),
    }


def _build_periodic_reduce_matrix(fine_num, stride, dtype):
    if fine_num % stride != 0:
        raise ValueError(f"fine_num={fine_num} is not divisible by stride={stride}.")

    coarse_num = fine_num // stride
    reduce = torch.zeros((coarse_num, fine_num), dtype=dtype)

    for coarse_idx in range(coarse_num):
        sample_idx = coarse_idx * stride
        reduce[coarse_idx, sample_idx] = 1.0

    return reduce


def _build_linear_reduce_matrix(fine_num, stride, dtype):
    if fine_num % stride != 0:
        raise ValueError(f"fine_num={fine_num} is not divisible by stride={stride}.")

    coarse_num = fine_num // stride
    reduce = torch.zeros((coarse_num, fine_num), dtype=dtype)
    coarse_pos = (torch.arange(coarse_num, dtype=dtype) + 0.5) * stride - 0.5

    for coarse_idx in range(coarse_num):
        pos = coarse_pos[coarse_idx].item()
        left_idx = max(0, min(fine_num - 1, math.floor(pos)))
        right_idx = min(fine_num - 1, left_idx + 1)
        if right_idx == left_idx:
            reduce[coarse_idx, left_idx] = 1.0
            continue
        weight_right = pos - left_idx
        weight_left = 1.0 - weight_right
        reduce[coarse_idx, left_idx] += weight_left
        reduce[coarse_idx, right_idx] += weight_right

    return reduce


def _build_periodic_expand_matrix(fine_num, stride, dtype):
    if fine_num % stride != 0:
        raise ValueError(f"fine_num={fine_num} is not divisible by stride={stride}.")

    coarse_num = fine_num // stride
    expand = torch.zeros((fine_num, coarse_num), dtype=dtype)

    for fine_idx in range(fine_num):
        left_idx = (fine_idx // stride) % coarse_num
        right_idx = (left_idx + 1) % coarse_num
        left_pos = left_idx * stride
        right_pos = left_pos + stride
        pos_adj = fine_idx
        if right_idx == 0 and fine_idx < left_pos:
            pos_adj += fine_num
            right_pos = fine_num
        denom = max(right_pos - left_pos, 1e-12)
        weight_right = (pos_adj - left_pos) / denom
        weight_left = 1.0 - weight_right
        expand[fine_idx, left_idx] += weight_left
        expand[fine_idx, right_idx] += weight_right

    return expand


def _build_linear_expand_matrix(fine_num, stride, dtype):
    if fine_num % stride != 0:
        raise ValueError(f"fine_num={fine_num} is not divisible by stride={stride}.")

    coarse_num = fine_num // stride
    expand = torch.zeros((fine_num, coarse_num), dtype=dtype)
    coarse_pos = (torch.arange(coarse_num, dtype=dtype) + 0.5) * stride - 0.5

    for fine_idx in range(fine_num):
        pos = float(fine_idx)
        insert_idx = torch.searchsorted(coarse_pos, torch.tensor(pos, dtype=dtype)).item()
        if insert_idx <= 0:
            expand[fine_idx, 0] = 1.0
            continue
        if insert_idx >= coarse_num:
            expand[fine_idx, coarse_num - 1] = 1.0
            continue

        left_idx = insert_idx - 1
        right_idx = insert_idx
        left_pos = coarse_pos[left_idx].item()
        right_pos = coarse_pos[right_idx].item()
        denom = max(right_pos - left_pos, 1e-12)
        weight_right = (pos - left_pos) / denom
        weight_left = 1.0 - weight_right
        expand[fine_idx, left_idx] += weight_left
        expand[fine_idx, right_idx] += weight_right

    return expand


def _choose_compatible_ring_stride(ring_length, rotate_num, target_stride):
    if target_stride <= 1:
        return 1

    if ring_length % target_stride == 0 and rotate_num > 0 and ring_length % rotate_num == 0:
        rotate_step = ring_length // rotate_num
        for candidate in range(target_stride, 0, -1):
            if ring_length % candidate == 0 and rotate_step % candidate == 0:
                return candidate
        return 1

    if rotate_num > 0 and ring_length % rotate_num == 0:
        rotate_step = ring_length // rotate_num
        for candidate in range(target_stride, 0, -1):
            if ring_length % candidate == 0 and rotate_step % candidate == 0:
                return candidate
        return 1

    for candidate in range(target_stride, 0, -1):
        if ring_length % candidate == 0:
            return candidate
    return 1


def build_compton_sparse_projector(coor_full, theta_stride=2, z_stride=2, rotate_num=None, dtype=torch.float32):
    if theta_stride <= 0 or z_stride <= 0:
        raise ValueError("theta_stride and z_stride must be positive.")

    layout = _infer_polar_layout(coor_full)
    z_values = layout["z_values"].to(dtype=dtype)
    fine_z_num = layout["fine_z_num"]
    fine_layer_num = layout["fine_layer_num"]
    ring_lengths = layout["ring_lengths"]
    ring_radii = layout["ring_radii"].to(dtype=dtype)

    if fine_z_num % z_stride != 0:
        raise ValueError(f"fine_z_num={fine_z_num} is not divisible by z_stride={z_stride}.")

    layer_blocks = []
    layer_expand_blocks = []
    coarse_layer_parts = []
    ring_strides = []
    for ring_length, ring_radius in zip(ring_lengths, ring_radii):
        ring_stride = _choose_compatible_ring_stride(ring_length, rotate_num, theta_stride) if rotate_num is not None else theta_stride
        ring_reduce = _build_periodic_reduce_matrix(ring_length, ring_stride, dtype)
        ring_expand = _build_periodic_expand_matrix(ring_length, ring_stride, dtype)
        layer_blocks.append(ring_reduce)
        layer_expand_blocks.append(ring_expand)
        ring_strides.append(ring_stride)

        coarse_theta_num = ring_reduce.size(0)
        fine_theta_step = 2.0 * math.pi / ring_length
        coarse_theta_pos = torch.arange(coarse_theta_num, dtype=dtype) * ring_stride
        coarse_theta = coarse_theta_pos * fine_theta_step
        coarse_ring = torch.stack(
            (
                ring_radius * torch.cos(coarse_theta),
                ring_radius * torch.sin(coarse_theta),
            ),
            dim=1,
        )
        coarse_layer_parts.append(coarse_ring)

    layer_reduce = torch.block_diag(*layer_blocks)
    layer_expand = torch.block_diag(*layer_expand_blocks)
    coarse_layer = torch.cat(coarse_layer_parts, dim=0)

    z_reduce = _build_linear_reduce_matrix(fine_z_num, z_stride, dtype)
    z_expand = _build_linear_expand_matrix(fine_z_num, z_stride, dtype)
    coarse_z_values = torch.matmul(z_reduce, z_values.unsqueeze(1)).squeeze(1)

    coarse_planes = []
    for z_value in coarse_z_values:
        coarse_planes.append(
            torch.cat(
                (
                    coarse_layer,
                    torch.full((coarse_layer.size(0), 1), z_value.item(), dtype=dtype),
                ),
                dim=1,
            )
        )
    coor_coarse = torch.cat(coarse_planes, dim=0)

    coarse_layer_num = coarse_layer.size(0)
    coarse_z_num = coarse_z_values.numel()

    return ComptonSparseProjector(
        theta_stride=theta_stride,
        z_stride=z_stride,
        ring_strides=tuple(ring_strides),
        fine_layer_num=fine_layer_num,
        fine_z_num=fine_z_num,
        coarse_layer_num=coarse_layer_num,
        coarse_z_num=coarse_z_num,
        fine_pixel_num=fine_layer_num * fine_z_num,
        coarse_pixel_num=coarse_layer_num * coarse_z_num,
        layer_reduce=layer_reduce,
        z_reduce=z_reduce,
        layer_expand=layer_expand,
        z_expand=z_expand,
        coor_coarse=coor_coarse,
    )


def reduce_fine_image_to_coarse(img_fine, projector):
    if img_fine.ndim != 2 or img_fine.size(1) != 1:
        raise ValueError("img_fine must have shape [fine_pixel_num, 1].")
    if img_fine.size(0) != projector.fine_pixel_num:
        raise ValueError("img_fine size does not match projector.fine_pixel_num.")

    img_2d = img_fine.reshape(projector.fine_z_num, projector.fine_layer_num)
    tmp = torch.matmul(img_2d, projector.layer_reduce.transpose(0, 1))
    coarse = torch.matmul(projector.z_reduce, tmp)
    return coarse.reshape(-1, 1)


def upsample_coarse_weight_to_fine(weight_coarse, projector):
    if weight_coarse.ndim != 2 or weight_coarse.size(1) != 1:
        raise ValueError("weight_coarse must have shape [coarse_pixel_num, 1].")
    if weight_coarse.size(0) != projector.coarse_pixel_num:
        raise ValueError("weight_coarse size does not match projector.coarse_pixel_num.")

    weight_2d = weight_coarse.reshape(projector.coarse_z_num, projector.coarse_layer_num)
    tmp = torch.matmul(projector.z_reduce.transpose(0, 1), weight_2d)
    fine = torch.matmul(tmp, projector.layer_reduce)
    return fine.reshape(-1, 1)


def reduce_fine_rows_to_coarse(rows_fine, projector):
    if rows_fine.ndim != 2:
        raise ValueError("rows_fine must have shape [num_rows, fine_pixel_num].")
    if rows_fine.size(1) != projector.fine_pixel_num:
        raise ValueError("rows_fine size does not match projector.fine_pixel_num.")

    rows_3d = rows_fine.reshape(rows_fine.size(0), projector.fine_z_num, projector.fine_layer_num)
    tmp = torch.matmul(rows_3d, projector.layer_reduce.transpose(0, 1))
    tmp = tmp.permute(0, 2, 1)
    coarse = torch.matmul(tmp, projector.z_reduce.transpose(0, 1))
    coarse = coarse.permute(0, 2, 1).contiguous()
    return coarse.reshape(rows_fine.size(0), projector.coarse_pixel_num)


def upsample_coarse_rows_to_fine(rows_coarse, projector):
    if rows_coarse.ndim != 2:
        raise ValueError("rows_coarse must have shape [num_rows, coarse_pixel_num].")
    if rows_coarse.size(1) != projector.coarse_pixel_num:
        raise ValueError("rows_coarse size does not match projector.coarse_pixel_num.")

    rows_3d = rows_coarse.reshape(rows_coarse.size(0), projector.coarse_z_num, projector.coarse_layer_num)
    tmp = torch.matmul(rows_3d, projector.layer_expand.transpose(0, 1))
    tmp = tmp.permute(0, 2, 1)
    fine = torch.matmul(tmp, projector.z_expand.transpose(0, 1))
    fine = fine.permute(0, 2, 1).contiguous()
    return fine.reshape(rows_coarse.size(0), projector.fine_pixel_num)


def project_fine_rows_to_coarse_adjoint(rows_fine, projector):
    if rows_fine.ndim != 2:
        raise ValueError("rows_fine must have shape [num_rows, fine_pixel_num].")
    if rows_fine.size(1) != projector.fine_pixel_num:
        raise ValueError("rows_fine size does not match projector.fine_pixel_num.")

    rows_3d = rows_fine.reshape(rows_fine.size(0), projector.fine_z_num, projector.fine_layer_num)
    tmp = torch.matmul(rows_3d, projector.layer_expand)
    tmp = tmp.permute(0, 2, 1)
    coarse = torch.matmul(tmp, projector.z_expand)
    coarse = coarse.permute(0, 2, 1).contiguous()
    return coarse.reshape(rows_fine.size(0), projector.coarse_pixel_num)


def pack_sparse_event_rows(cpnum1, t_compton):
    if cpnum1.ndim != 1:
        raise ValueError("cpnum1 must have shape [num_events].")
    if t_compton.ndim != 2:
        raise ValueError("t_compton must have shape [num_events, coarse_pixel_num].")
    if cpnum1.size(0) != t_compton.size(0):
        raise ValueError("cpnum1 and t_compton must have the same number of rows.")

    cpnum1_col = cpnum1.to(dtype=t_compton.dtype).unsqueeze(1)
    return torch.cat((cpnum1_col, t_compton), dim=1)


def unpack_sparse_event_rows(event_rows):
    if event_rows.ndim != 2 or event_rows.size(1) < 2:
        raise ValueError("event_rows must have shape [num_events, 1 + coarse_pixel_num].")

    cpnum1 = torch.round(event_rows[:, 0]).to(torch.long)
    t_compton = event_rows[:, 1:]
    return cpnum1, t_compton


def materialize_sparse_event_rows_to_fine(event_rows, sysmat, projector):
    if event_rows.ndim != 2 or event_rows.size(1) != projector.coarse_pixel_num + 1:
        raise ValueError("event_rows size does not match sparse projector.")
    if sysmat.ndim != 2 or sysmat.size(1) != projector.fine_pixel_num:
        raise ValueError("sysmat size does not match sparse projector.")
    if event_rows.device != sysmat.device:
        raise ValueError("event_rows and sysmat must be on the same device.")
    if projector.coor_coarse.device != event_rows.device:
        raise ValueError("projector must be on the same device as event_rows.")

    cpnum1, t_compton = unpack_sparse_event_rows(event_rows)
    cpnum1 = torch.clamp(cpnum1 - 1, min=0, max=sysmat.size(0) - 1)

    t_compton_fine = upsample_coarse_rows_to_fine(t_compton, projector)
    sysmat_rows = torch.index_select(sysmat, 0, cpnum1)
    t_fine = torch.nan_to_num(t_compton_fine * sysmat_rows, nan=0.0, posinf=0.0, neginf=0.0)

    row_sum = torch.sum(t_fine, dim=1, keepdim=True)
    valid = torch.isfinite(row_sum.squeeze(1)) & (row_sum.squeeze(1) > 0)
    t_norm = t_fine[valid] / row_sum[valid]
    return t_norm, valid
