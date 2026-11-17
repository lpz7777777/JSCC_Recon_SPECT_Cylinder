"""Only the existing v5 joint branch, with durable saves and a time bound.

SC and Compton weights use the same current density estimate and their sum
is divided by S_SC + S_Compton. No post-reconstruction image addition.
The frozen torch_active_operator and its numerical primitives are unchanged.
"""
import time
import torch
from torch_active_operator import _event_weight, _reduce_sum, _update


def joint_mlem(response, projection, event_blocks, single_sensitivity,
               compton_sensitivity, iterations, save_step, save_history=True,
               checkpoint_callback=None, phase_limit_seconds=None):
    if iterations <= 0 or save_step <= 0 or iterations % save_step:
        raise ValueError('Iteration count must divide by save step')
    n = response.geometry.active_count
    device = response.full_rows.device
    image = torch.ones((n, 1), dtype=torch.float32, device=device)
    history = []
    started = time.monotonic()
    for iteration in range(iterations):
        if phase_limit_seconds and time.monotonic()-started > phase_limit_seconds:
            raise TimeoutError('Bounded joint phase exceeded')
        weight = torch.zeros_like(image)
        for view in range(response.geometry.views):
            matrix = response.matrix(view)
            forward = matrix @ image
            weight += matrix.T @ (projection[:, view:view+1] / forward.clamp_min(1e-12))
            weight += _event_weight(event_blocks[view], image, device)
        total_sensitivity = single_sensitivity + compton_sensitivity
        image = _update(image, _reduce_sum(weight), total_sensitivity)
        if save_history and (iteration+1) % save_step == 0:
            history.append(image.detach().cpu().clone())
            if checkpoint_callback is not None:
                checkpoint_callback(iteration+1, history)
            print('JSCC_JOINT_ITERATION', iteration+1, iterations, flush=True)
    return image, torch.stack(history) if history else None
