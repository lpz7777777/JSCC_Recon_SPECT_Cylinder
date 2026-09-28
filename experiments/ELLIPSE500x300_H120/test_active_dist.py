"""Tiny deterministic Gloo check: two-rank active MLEM equals one-rank MLEM."""
import os
import numpy as np
import torch
import torch.distributed as dist

from torch_active_operator import ActiveGeometry,ViewResponse,forward_project,single_mlem


def main():
    rank=int(os.environ["RANK"])
    world=int(os.environ["WORLD_SIZE"])
    if world!=2: raise ValueError("Run with exactly two torchrun ranks")
    dist.init_process_group("gloo",init_method="env://")
    rotation=np.array([[0,1,2],[1,2,0],[2,0,1]],dtype=np.int64).T
    inverse=np.argsort(rotation,axis=0)
    geometry=ActiveGeometry(np.array([0,1]),np.array([1.,.65,0.]),inverse)
    full=torch.tensor([[1.,2.,.5],[2.,.7,1.],[.8,1.5,.3],[1.2,.4,1.8]],dtype=torch.float32)
    truth=torch.tensor([[2.],[.6]])
    response_all=ViewResponse(full,geometry)
    counts=forward_project(response_all,truth)
    rows=full[rank*2:(rank+1)*2]
    local=ViewResponse(rows,geometry)
    sensitivity=local.sensitivity()
    dist.all_reduce(sensitivity)
    fit,_=single_mlem(local,counts[rank*2:(rank+1)*2],sensitivity,100,10,
                      save_history=False)
    gathered=[torch.empty_like(fit) for _ in range(world)]
    dist.all_gather(gathered,fit)
    if not torch.allclose(gathered[0],gathered[1],atol=1e-6):
        raise AssertionError("Ranks disagree")
    dist.destroy_process_group()
    if rank==0:
        reference,_=single_mlem(response_all,counts,response_all.sensitivity(),
                                100,10,save_history=False)
        if not torch.allclose(fit,reference,atol=1e-5):
            raise AssertionError((fit,reference))
        print("ELLIPSE_TWO_RANK_MLEM_MATCHES_SINGLE",fit.view(-1).tolist())


if __name__=="__main__":
    with torch.no_grad(): main()
