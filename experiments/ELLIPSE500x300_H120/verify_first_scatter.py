"""Strict regression/pilot/formal verification for the four paired histories."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()

def allocated_host_bytes(allocation,world):
    text=Path(allocation).read_text()
    nodes=re.search(r"\bNumNodes=(\d+)",text)
    tres=re.search(r"\bAllocTRES=([^\s]+)",text)
    if not nodes or int(nodes[1])!=world or not tres:
        raise ValueError("Actual Slurm node allocation missing or differs")
    memory=re.search(r"(?:^|,)mem=([0-9.]+)([KMGT])(?:,|$)",tres[1])
    if not memory:raise ValueError("Actual Slurm allocated memory missing")
    return int(float(memory[1])*1024**("KMGT".index(memory[2])+1)/world)

def verify(result,config,geometry,baseline,mode,allocation=None):
    cfg=json.loads(config.read_text());run=json.loads((result/"run_manifest.json").read_text())
    gate_path=config.parent/"validation_gate.json"
    if digest(gate_path)!=cfg["validation_gate_sha256"] or json.loads(gate_path.read_text())["status"]!="PASSED":
        raise ValueError("Independent sensitivity gate is not passed/frozen")
    if digest(geometry)!=cfg["geometry_sha256"]:raise ValueError("Geometry differs")
    iterations,save={"regression":(50,50),"pilot":(10,10),"formal":(2000,50)}[mode]
    if (run["iterations"],run["save_step"])!=(iterations,save):raise ValueError("Iteration contract differs")
    world=run["world_size"]
    if world not in (4,8) or run["pixels_active"]!=82040 or run["pixels_full"]!=132040:
        raise ValueError("Topology/grid differs")
    resources=run["resources"]
    if sorted(r["rank"] for r in resources)!=list(range(world)) or len({r["node"] for r in resources})!=world:
        raise ValueError("Distinct node per GPU/rank required")
    info=run["response_mismatch"];filtered=mode!="regression"
    if info["filter_enabled"]!=filtered or info["config_sha256"]!=digest(config):raise ValueError("Filter/config differs")
    if info["event_policy"]!=cfg["group"] or info["validation_gate_sha256"]!=digest(gate_path):
        raise ValueError("Wrong event policy or validation gate")
    if (run["input_sha256"]!=cfg["baseline_input_sha256"] or
        run["factor_manifest_sha256"]!=cfg["baseline_factor_manifest_sha256"] or run["geometry_sha256"]!=digest(geometry)):
        raise ValueError("Frozen inputs/Factor/geometry differ")
    count=cfg["kept_compton_events"] if filtered else 97299
    if run["accepted_compton_events"]!=count or sum(r["accepted_events"] for r in resources)!=count:
        raise ValueError("Global/rank event closure differs")
    if run["accepted_compton_events_per_view"]!=(cfg["kept_per_view"] if filtered else cfg["baseline_per_view"]):
        raise ValueError("Per-view closure differs")
    if run["sensi_d_sha256"]!=(cfg["sensi_d_sha256"] if filtered else cfg["baseline_sensi_d_sha256"]):
        raise ValueError("Matched sensitivity differs")
    g=np.load(geometry);active=g["active_indices"];inactive=np.ones(132040,dtype=bool);inactive[active]=False
    outputs=[]
    for channel in cfg["channels"]:
        paths={k:result/f"Image_{channel}_{k}.float32" for k in ("active","full","history")}
        sizes={"active":82040,"full":132040,"history":82040*(iterations//save)}
        for key,path in paths.items():
            if path.stat().st_size!=sizes[key]*4:raise ValueError("Image/history size differs")
            a=np.memmap(path,"<f4",mode="r")
            if not np.isfinite(a).all() or np.any(a<0):raise ValueError("Invalid image/history values")
        a=np.fromfile(paths["active"],"<f4");f=np.fromfile(paths["full"],"<f4")
        history=np.memmap(paths["history"],"<f4",mode="r",shape=(iterations//save,82040))
        if not np.array_equal(a,history[-1]) or not np.array_equal(f[active],a) or np.any(f[inactive]!=0):
            raise ValueError("Last frame/active/full ellipse closure differs")
        output=dict(channel=channel,frames=iterations//save,sha256={k:digest(p) for k,p in paths.items()})
        if mode=="regression":
            old=np.memmap(baseline/f"Image_{channel}_history.float32","<f4",mode="r",shape=(200,82040))[0]
            error=float(np.linalg.norm(a.astype(np.float64)-old)/np.linalg.norm(old.astype(np.float64)))
            output["baseline_frame50_relative_l2"]=error
            if error>1e-5:raise ValueError("Legacy unfiltered baseline regression differs")
        if mode=="formal":
            for frame in range(40):
                iteration=(frame+1)*50;folder=result/f"checkpoint_{iteration:06d}"
                snapshot=json.loads((folder/"checkpoint_manifest.json").read_text())
                if snapshot["iteration"]!=iteration or snapshot["config_sha256"]!=digest(config):
                    raise ValueError("Persistent checkpoint identity differs")
                for kind in ("active","full"):
                    p=folder/f"Image_{channel}_{kind}.float32"
                    if digest(p)!=snapshot["outputs"][channel][kind]:raise ValueError("Checkpoint hash differs")
                if not np.array_equal(np.fromfile(folder/f"Image_{channel}_active.float32","<f4"),history[frame]):
                    raise ValueError("Persistent checkpoint/history differ")
        outputs.append(output)
    for r in resources:
        if allocation is not None and r["host_allocated_bytes"]!=allocated_host_bytes(allocation,world):
            raise ValueError("Host memory denominator differs from actual Slurm allocation")
        if r["host_allocated_bytes"]<=0:raise ValueError("Actual host allocation not recorded")
        if (r["peak_reserved_bytes"]/r["total_device_bytes"]>.8 or
            r["host_peak_rss_bytes"]/r["host_allocated_bytes"]>.8):raise ValueError("20% resource margin fails")
    report=dict(passed=True,study="compton_first_scatter_v2",group=cfg["group"],mode=mode,
        outputs=outputs,resources=resources,accepted_events=count,iterations=iterations,save_step=save,
        config_sha256=digest(config),run_manifest_sha256=digest(result/"run_manifest.json"),
        allocation_sha256=digest(allocation) if allocation is not None else None)
    (result/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print("FIRST_SCATTER_VERIFIED",cfg["group"],mode,count)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--allocation-only",type=Path);p.add_argument("--nodes",type=int)
    import sys
    if "--allocation-only" in sys.argv:
        a=p.parse_args()
        if a.nodes not in (4,8):raise ValueError("Actual 4/8-node allocation required")
        print(allocated_host_bytes(a.allocation_only,a.nodes));return
    for name in ("result","config","geometry","baseline"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--allocation",type=Path,required=True)
    p.add_argument("--mode",choices=("regression","pilot","formal"),required=True)
    a=p.parse_args();verify(a.result,a.config,a.geometry,a.baseline,a.mode,a.allocation)
if __name__=="__main__":main()
