"""Bounded multi-node NCCL transport smoke test, independent of Factors."""
from datetime import timedelta
import os
import torch
import torch.distributed as dist

rank=int(os.environ["RANK"])
local=int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
print(f"rank={rank} init begin",flush=True)
dist.init_process_group("nccl",timeout=timedelta(minutes=2))
print(f"rank={rank} init ready",flush=True)
value=torch.tensor([rank+1.],device=f"cuda:{local}")
dist.all_reduce(value)
print(f"rank={rank} all_reduce={value.item()}",flush=True)
data=["success" if rank==0 else None]
dist.broadcast_object_list(data,src=0,device=torch.device(f"cuda:{local}"))
print(f"rank={rank} broadcast={data[0]}",flush=True)
dist.destroy_process_group()
