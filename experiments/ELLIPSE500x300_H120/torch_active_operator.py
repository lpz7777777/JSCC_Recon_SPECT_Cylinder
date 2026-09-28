"""Compact object-frame ellipse columns for existing JSCC GPU response rows.

Input B is already volume weighted on the complete circular grid. This module
forms B_v = B[:, P_v^{-1}(active)] * ellipse_fraction[active]. Its transpose is
the exact backprojector, for both detector rows and materialized K*B rows.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist


class ActiveGeometry:
    def __init__(self, active, fraction, inverse_rotation, device="cpu"):
        active = np.asarray(active, dtype=np.int64)
        fraction = np.asarray(fraction, dtype=np.float32)
        inverse_rotation = np.asarray(inverse_rotation, dtype=np.int64)
        if inverse_rotation.shape[0] != len(fraction) or active.ndim != 1:
            raise ValueError("Full grid dimensions disagree")
        if not np.array_equal(active, np.flatnonzero(fraction > 1e-13)):
            raise ValueError("Active cells and ellipse fractions disagree")
        if np.any(fraction < 0) or np.any(fraction > 1):
            raise ValueError("Invalid ellipse fractions")
        if inverse_rotation.min() < 0 or inverse_rotation.max() >= len(fraction):
            raise ValueError("Inverse rotation out of range")
        self.fraction = torch.as_tensor(fraction[active], device=device)
        self.object_active = torch.as_tensor(active.copy(), dtype=torch.long, device=device)
        self.indices = [torch.as_tensor(inverse_rotation[active, v].copy(),
                                       dtype=torch.long, device=device)
                        for v in range(inverse_rotation.shape[1])]
        self.views = len(self.indices)
        self.active_count = len(active)
        self.full_count = len(fraction)

    @classmethod
    def from_npz(cls, path, device="cpu"):
        with np.load(path) as geometry:
            return cls(geometry["active_indices"], geometry["ellipse_fraction"],
                       geometry["inverse_rotation"], device)

    def compact(self, rows, view):
        if rows.ndim != 2 or rows.size(1) != self.full_count:
            raise ValueError("Response rows require the complete circular grid")
        if rows.device != self.fraction.device:
            raise ValueError("Response and geometry must be on the same device")
        return torch.index_select(rows, 1, self.indices[view]) * self.fraction

    def forward(self, rows, image, view):
        if image.shape != (self.active_count, 1):
            raise ValueError("Image must contain active density columns")
        return self.compact(rows, view) @ image

    def adjoint(self, rows, measurements, view):
        return self.compact(rows, view).T @ measurements

    def single_sensitivity(self, rows):
        result = torch.zeros((self.active_count, 1),
                             dtype=rows.dtype, device=rows.device)
        for view in range(self.views):
            result += self.compact(rows, view).sum(dim=0)[:, None]
        return result / self.views

    def compton_sensitivity(self, full_sensitivity):
        """Apply object-frame overlap to a complete-grid K*B sensitivity."""
        if full_sensitivity.numel() != self.full_count:
            raise ValueError("Compton sensitivity has the wrong full grid")
        full = torch.as_tensor(full_sensitivity, device=self.fraction.device).reshape(-1)
        return (full[self.object_active] * self.fraction)[:, None]


class ViewResponse:
    def __init__(self, full_rows, geometry, cache="none"):
        if cache not in ("none", "cpu", "device"):
            raise ValueError("Unknown response cache mode")
        self.full_rows = full_rows
        self.geometry = geometry
        self.cache = cache
        self.cached = None
        if cache != "none":
            self.cached = []
            for view in range(geometry.views):
                item = geometry.compact(full_rows, view)
                if cache == "cpu":
                    item = item.cpu()
                self.cached.append(item)

    def matrix(self, view):
        if self.cached is None:
            return self.geometry.compact(self.full_rows, view)
        item = self.cached[view]
        return item if item.device == self.full_rows.device else item.to(self.full_rows.device)

    def sensitivity(self):
        result = torch.zeros((self.geometry.active_count, 1),
                             dtype=self.full_rows.dtype, device=self.full_rows.device)
        for view in range(self.geometry.views):
            result += self.matrix(view).sum(dim=0)[:, None]
        return result / self.geometry.views


def _reduce_sum(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _update(image, weight, sensitivity):
    weight = torch.nan_to_num(weight, nan=0, posinf=0, neginf=0)
    valid = torch.isfinite(sensitivity) & (sensitivity > 1e-12)
    return torch.where(valid, image * weight.clamp_min(0) /
                       sensitivity.clamp_min(1e-12), torch.zeros_like(image))


def forward_project(response, image):
    """Predict local detector counts, including 1/views acquisition factor."""
    return torch.cat([(response.matrix(v) @ image) / response.geometry.views
                      for v in range(response.geometry.views)], dim=1)


def single_mlem(response, projection, sensitivity, iterations, save_step,
                additive_background=None, save_history=True):
    if iterations <= 0 or iterations % save_step:
        raise ValueError("Iteration count must divide by save step")
    n = response.geometry.active_count
    image = torch.ones((n,1),dtype=response.full_rows.dtype,device=response.full_rows.device)
    history = []
    for iteration in range(iterations):
        weight = torch.zeros_like(image)
        for view in range(response.geometry.views):
            matrix = response.matrix(view)
            forward = matrix @ image
            if additive_background is not None:
                forward += additive_background[:,view:view+1] * response.geometry.views
            ratio = projection[:,view:view+1] / forward.clamp_min(1e-12)
            weight += matrix.T @ ratio
        image = _update(image,_reduce_sum(weight),sensitivity)
        if save_history and (iteration+1) % save_step == 0:
            history.append(image.detach().cpu().clone())
    return image, torch.stack(history) if history else None


def compact_event_blocks(blocks_by_view, geometry, storage_device="cpu", pin_memory=False):
    """Compress materialized full-grid K*B rows before long MLEM loops."""
    result=[]
    for view,blocks in enumerate(blocks_by_view):
        items=[]
        for block in blocks:
            compact=geometry.compact(block.to(geometry.fraction.device),view)
            if storage_device == "cpu":
                compact=compact.cpu()
                if pin_memory and torch.cuda.is_available():
                    compact=compact.pin_memory()
            items.append(compact)
        result.append(items)
    return result


def _event_weight(blocks, image, device):
    weight=torch.zeros_like(image)
    for stored in blocks:
        matrix=stored if stored.device==device else stored.to(device,non_blocking=True)
        weight += matrix.T @ (1.0 / (matrix @ image).clamp_min(1e-12))
    return weight


def compton_and_joint_mlem(response, projection, event_blocks,
                           single_sensitivity, compton_sensitivity,
                           iterations, save_step, save_history=True):
    if iterations <= 0 or iterations % save_step:
        raise ValueError("Iteration count must divide by save step")
    n=response.geometry.active_count
    device=response.full_rows.device
    image_d=torch.ones((n,1),dtype=torch.float32,device=device)
    image_j=image_d.clone()
    history_d=[]
    history_j=[]
    for iteration in range(iterations):
        weight_d=torch.zeros_like(image_d)
        weight_j=torch.zeros_like(image_j)
        for view in range(response.geometry.views):
            matrix=response.matrix(view)
            weight_j += matrix.T @ (projection[:,view:view+1] /
                                    (matrix @ image_j).clamp_min(1e-12))
            weight_d += _event_weight(event_blocks[view],image_d,device)
            weight_j += _event_weight(event_blocks[view],image_j,device)
        image_d=_update(image_d,_reduce_sum(weight_d),compton_sensitivity)
        image_j=_update(image_j,_reduce_sum(weight_j),
                        single_sensitivity+compton_sensitivity)
        if save_history and (iteration+1)%save_step==0:
            history_d.append(image_d.detach().cpu().clone())
            history_j.append(image_j.detach().cpu().clone())
    return ((image_d,torch.stack(history_d) if history_d else None),
            (image_j,torch.stack(history_j) if history_j else None))
