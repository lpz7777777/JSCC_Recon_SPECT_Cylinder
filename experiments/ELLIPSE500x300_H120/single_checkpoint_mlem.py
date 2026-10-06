"""Unregularized original single-MLEM equations with durable save callbacks.

The frozen torch_active_operator module is not edited. Full-event validation
compares this loop with its original single_mlem for both additive modes.
"""
import time
import torch
from torch_active_operator import _reduce_sum, _update


def single_mlem_checkpointed(response, projection, sensitivity, iterations, save_step,
                             additive_background=None, save_history=True,
                             checkpoint_callback=None, progress_label=None,
                             phase_limit_seconds=None):
    if iterations <= 0 or save_step <= 0 or iterations % save_step:
        raise ValueError('Iteration count must divide by save step')
    started = time.monotonic()
    n = response.geometry.active_count
    image = torch.ones((n, 1), dtype=response.full_rows.dtype, device=response.full_rows.device)
    history = []
    for iteration in range(iterations):
        if phase_limit_seconds and time.monotonic() - started > phase_limit_seconds:
            raise TimeoutError('Bounded single MLEM phase exceeded')
        weight = torch.zeros_like(image)
        for view in range(response.geometry.views):
            matrix = response.matrix(view)
            forward = matrix @ image
            if additive_background is not None:
                forward += additive_background[:, view:view+1] * response.geometry.views
            ratio = projection[:, view:view+1] / forward.clamp_min(1e-12)
            weight += matrix.T @ ratio
        reduced = _reduce_sum(weight)
        image = _update(image, reduced, sensitivity)
        if save_history and (iteration+1) % save_step == 0:
            history.append(image.detach().cpu().clone())
            if checkpoint_callback is not None:
                checkpoint_callback(iteration+1, history)
            if progress_label:
                print(f'ELLIPSE_ITERATION {progress_label} {iteration+1}/{iterations}', flush=True)
    return image, torch.stack(history) if history else None
