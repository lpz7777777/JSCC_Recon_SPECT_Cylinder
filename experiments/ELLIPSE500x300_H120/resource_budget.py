"""Conservative planning estimates; actual 10-iteration measurements decide."""
import argparse
import json
from math import ceil

PIXELS_FULL=132040
PIXELS_ACTIVE=82040
DETECTORS=10496
FLOAT=4


def estimate(accepted_events,world_size,ranks_per_node,gpu_gib,host_gib):
    if min(accepted_events,world_size,ranks_per_node,gpu_gib,host_gib)<=0:
        raise ValueError("Positive resource inputs required")
    event_rows=ceil(accepted_events/world_size)*PIXELS_ACTIVE*FLOAT
    # B_440 full for K*B preprocessing; three detector shards for 440,218,cross.
    gpu_base=(PIXELS_FULL*DETECTORS*FLOAT+
              3*ceil(DETECTORS/world_size)*PIXELS_FULL*FLOAT+
              512*PIXELS_FULL*FLOAT+2*512*PIXELS_ACTIVE*FLOAT+
              2*1024**3)
    # CPU event storage, mapped Factors share the OS page cache across ranks.
    host_base=ranks_per_node*(event_rows+ceil(DETECTORS/world_size)*PIXELS_FULL*FLOAT+
                              512*1024**2)+3*PIXELS_FULL*DETECTORS*FLOAT
    gib=1024**3
    return {"world_size":world_size,"ranks_per_node":ranks_per_node,
            "accepted_events_assumed":accepted_events,
            "event_rows_gib_per_rank":event_rows/gib,
            "estimated_gpu_peak_gib_per_rank":gpu_base/gib,
            "gpu_capacity_gib":gpu_gib,
            "estimated_host_gib_per_node":host_base/gib,
            "host_capacity_gib_per_node":host_gib,
            "gpu_20_percent_margin_pass":gpu_base<=.8*gpu_gib*gib,
            "host_20_percent_margin_pass":host_base<=.8*host_gib*gib,
            "note":"Estimates omit allocator fragmentation, packed event rows and CUDA framework overhead; full-event 10-iteration pilot is mandatory."}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--accepted-events",type=int,required=True)
    p.add_argument("--gpu-gib",type=float,required=True)
    p.add_argument("--host-gib",type=float,required=True)
    p.add_argument("--nodes",type=int)
    p.add_argument("--gpus-per-node",type=int)
    a=p.parse_args()
    if (a.nodes is None) != (a.gpus_per_node is None):
        p.error("--nodes and --gpus-per-node must be supplied together")
    topologies = ([(a.nodes,a.gpus_per_node)] if a.nodes is not None
                  else [(8,g) for g in (3,4,6,8)])
    print(json.dumps([estimate(a.accepted_events,n*g,g,a.gpu_gib,a.host_gib)
                      for n,g in topologies],indent=2))
