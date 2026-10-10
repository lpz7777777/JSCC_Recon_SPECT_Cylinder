"""Only the original Compton branch: no joint image or joint event passes."""
import time
import torch
from torch_active_operator import _event_weight,_reduce_sum,_update

def compton_mlem(event_blocks,sensitivity,iterations,save_step,save_history=True,checkpoint_callback=None,phase_limit_seconds=None):
    image=torch.ones_like(sensitivity);history=[];started=time.monotonic()
    for i in range(iterations):
        if phase_limit_seconds and time.monotonic()-started>phase_limit_seconds:raise TimeoutError('Bounded Compton phase exceeded')
        weight=torch.zeros_like(image)
        for blocks in event_blocks:weight+=_event_weight(blocks,image,image.device)
        image=_update(image,_reduce_sum(weight),sensitivity)
        if save_history and (i+1)%save_step==0:
            history.append(image.detach().cpu().clone())
            if checkpoint_callback:checkpoint_callback(i+1,history)
            print('JSCC_COMPTON_ITERATION',i+1,iterations,flush=True)
    return image,torch.stack(history) if history else None
