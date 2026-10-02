"""Two-process ablation: single, additive background, Compton and joint agree."""
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.distributed as dist

from regularized_update import AblationUpdate
from torch_active_operator import ActiveGeometry, ViewResponse, forward_project, single_mlem, compton_and_joint_mlem


def solve(folder, method, response, counts, events, ss, sd, background):
    # Tight inner accuracy isolates distributed arithmetic from truncated-solver
    # stopping differences. Production separately records its looser gap/caps.
    update=AblationUpdate({"inner_max":2000,"inner_gap_tolerance":1e-12},Path(folder)/"model.npz",
                         {"method":method,"strength":.03,"huber_delta":1.},"cpu")
    # Counts, sensitivity, and prior normalization are global on every rank.
    update.configure("single",ss,10.)
    update.configure("440_compton",sd,4.)
    update.configure("440_jscc",ss+sd,14.)
    x,_=single_mlem(response,counts,ss,30,10,background,False,"single",update)
    (d,_),(j,_)=compton_and_joint_mlem(response,counts,events,ss,sd,30,10,False,update_rule=update)
    return torch.cat((x,d,j),dim=1)


def main():
    torch.set_num_threads(1)
    rank=int(os.environ["RANK"])
    dist.init_process_group("gloo")
    if dist.get_world_size()!=2:
        raise ValueError("Requires exactly two ranks")
    geometry=ActiveGeometry(np.arange(3),np.ones(3),np.arange(3)[:,None])
    full=torch.tensor([[1.,.3,.1],[.2,.7,.2],[.1,.1,.9],[.6,.4,.3]])
    events=torch.tensor([[.7,.2,.1],[.1,.3,.6],[.15,.7,.15],[.4,.2,.4]])
    all_response=ViewResponse(full,geometry)
    background=torch.ones((4,1))*.2
    counts=forward_project(all_response,torch.tensor([[2.],[1.],[3.]]))+background
    local=ViewResponse(full[rank*2:(rank+1)*2],geometry)
    ss=all_response.sensitivity(); sd=torch.tensor([[.4],[.5],[.6]])
    with tempfile.TemporaryDirectory() as folder:
        np.savez(Path(folder)/"model.npz",edge_i=[0,1],edge_j=[1,2],graph_weight=[.5,.5],
                 gradient_scale=[1.,1.],binding_group=[0,0,1])
        fits={method:solve(folder,method,local,counts[rank*2:(rank+1)*2],
                          [[events[rank*2:(rank+1)*2]]],ss,sd,background[rank*2:(rank+1)*2])
              for method in ("binding","huber","tv")}
        for fit in fits.values():
            gathered=[torch.empty_like(fit) for _ in range(2)]
            dist.all_gather(gathered,fit)
            torch.testing.assert_close(gathered[0],gathered[1],rtol=0,atol=0)
        dist.destroy_process_group()
        if rank==0:
            for method,fit in fits.items():
                expected=solve(folder,method,all_response,counts,[[events]],ss,sd,background)
                print(method,"distributed_max_abs_difference",float((fit-expected).abs().max()),flush=True)
                torch.testing.assert_close(fit,expected,rtol=4e-6,atol=4e-6)
            print("SPIKE_ABLATION_TWO_RANK_MATCHES_SINGLE: binding, Huber, TV; three likelihoods")


if __name__=="__main__":
    with torch.no_grad():
        main()
