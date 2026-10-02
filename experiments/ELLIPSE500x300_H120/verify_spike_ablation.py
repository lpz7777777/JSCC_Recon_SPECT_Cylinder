"""Fail-closed numeric, provenance and actual resource gate for each ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np

CHANNELS=("440_SinglePhoton","440_ComptonOnly","440_SinglePlusCompton",
          "218_SinglePhoton_CrossTalkCorrected","440SinglePlus218Single",
          "440SingleComptonPlus218Single")


def digest(path):
    sha=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):
            sha.update(block)
    return sha.hexdigest()


def array(path,count):
    if path.stat().st_size!=count*4:
        raise ValueError(f"Wrong byte count: {path.name}")
    x=np.memmap(path,mode="r",dtype="<f4",shape=(count,))
    if not np.isfinite(x).all() or np.any(x<0):
        raise ValueError(f"Invalid numeric data: {path.name}")
    return x


def slurm_memory(job):
    # Accounting can lag the just-finished srun step. A missing value cannot pass.
    for attempt in range(6):
        text=subprocess.check_output(["sacct","-n","-P","-j",job,"--format=JobID,MaxRSS,ReqMem"],text=True,timeout=20)
        sizes=[]
        for line in text.splitlines():
            parts=line.split("|")
            if len(parts)<3:
                continue
            match=re.fullmatch(r"([0-9.]+)([KMGT]?)",parts[1].strip())
            if match:
                sizes.append(float(match[1])*1024**({"":0,"K":1,"M":2,"G":3,"T":4}[match[2]]))
        if sizes:
            return {"job":job,"sacct":text,"max_rss_bytes":max(sizes)}
        if attempt<5:
            time.sleep(5)
    raise RuntimeError("No Slurm MaxRSS yet; gate remains closed, verify again after accounting updates")


def verify(base,study_path,result,phase,variant,job):
    study=json.loads(study_path.read_text())
    expected=next(v for v in study["variants"] if v["id"]==variant)
    iterations,step={"pilot":(10,10),"short":(200,50),"formal":(10000,50)}[phase]
    run=json.loads((result/"run_manifest.json").read_text())
    frozen=run.get("spike_ablation") or {}
    if (result.resolve().parent!=(base/"generated/Results").resolve() or
        run["experiment"]!="ELLIPSE500x300_H120" or run["dataset"]!=study["dataset"] or
        run["count_level"]!=study["level"] or run["iterations"]!=iterations or run["save_step"]!=step or
        run["pixels_full"]!=132040 or run["pixels_active"]!=82040 or
        frozen.get("variant")!=expected or frozen.get("study_sha256")!=digest(study_path) or
        run["geometry_sha256"]!=study["geometry_sha256"] or
        run["sensi_d_sha256"]!=study["baseline_sensi_d_sha256"] or
        run["input_sha256"]!=study["baseline_input_sha256"] or
        run["factor_manifest_sha256"]!=study["baseline_factor_manifest_sha256"] or
        run["accepted_compton_events"]!=study["baseline_accepted_compton_events"]):
        raise ValueError("Frozen variant, data, geometry, sensitivity or event mismatch")
    if bool(run["pilot_only"])!=(phase=="pilot") or bool(run["study_short"])!=(phase=="short"):
        raise ValueError("Run phase mismatch")
    release=json.loads((Path(__file__).parent/"release.json").read_text())
    for name,sha in frozen["code_sha256"].items():
        if sha!=release["files"][name]["sha256"]:
            raise ValueError(f"Code provenance differs from deployed release: {name}")
    if digest(base/"generated/Geometry/geometry.npz")!=run["geometry_sha256"]:
        raise ValueError("Geometry changed since reconstruction")
    if digest(base/"generated/FactorsCalibrated/440keV_RotateNum20/Sensi_d")!=run["sensi_d_sha256"]:
        raise ValueError("Compton sensitivity changed since reconstruction")
    for name,sha in run["input_sha256"].items():
        path=(base/"generated"/name).resolve()
        if not path.is_relative_to((base/"generated").resolve()) or digest(path)!=sha:
            raise ValueError(f"Changed input: {name}")
    for name,sha in run["factor_manifest_sha256"].items():
        if digest(base/"generated/FactorsCalibrated"/name/"factor_manifest.json")!=sha:
            raise ValueError("Changed Factor manifest")
    collection=json.loads((base/"generated/collections/NEMA_Body_H60_5e9.json").read_text())
    if (sum(collection["primary_counts"])!=5*10**9 or collection["views"]!=list(range(1,21)) or
        sorted(collection["worker_indices"])!=list(range(200)) or
        len(collection["seeds"])!=200 or len(set(collection["seeds"]))!=200):
        raise ValueError("Source provenance not closed")
    with np.load(base/"generated/Geometry/geometry.npz") as g:
        active=g["active_indices"]
    spatial=study_path.parent/"spatial_model.npz"
    if digest(spatial)!=study["spatial_model_sha256"] or frozen["spatial_model_sha256"]!=digest(spatial):
        raise ValueError("Spatial model changed")
    with np.load(spatial) as model:
        anchor=model["binding_anchor"]
    history={}; final={}; outputs={}
    for name in CHANNELS:
        paths={suffix:result/f"Image_{name}_{suffix}.float32" for suffix in ("active","full","history")}
        x=array(paths["active"],82040); full=array(paths["full"],132040)
        h=array(paths["history"],(iterations//step)*82040).reshape(iterations//step,82040)
        if not np.array_equal(x,full[active]) or np.count_nonzero(full)!=np.count_nonzero(x) or not np.array_equal(h[-1],x):
            raise ValueError(f"Ellipse support or final/history mismatch: {name}")
        if expected["method"]=="binding" and not np.array_equal(h,h[:,anchor]):
            raise ValueError(f"Tied density constraint violated: {name}")
        history[name]=h; final[name]=x
        outputs[name]={suffix:digest(path) for suffix,path in paths.items()}
    for name,left,right in ((CHANNELS[4],CHANNELS[0],CHANNELS[3]),(CHANNELS[5],CHANNELS[2],CHANNELS[3])):
        if not np.array_equal(final[name],final[left]+final[right]) or not np.array_equal(history[name],history[left]+history[right]):
            raise ValueError("Composite gamma channel sum mismatch")
    prediction=result/"PredictedCntStat_218_From440.float32"
    array(prediction,10496*20)
    optimization=json.loads((result/"optimization.json").read_text())
    if optimization["variant"]!=expected or not optimization["prior_applied_once_after_global_backprojection"]:
        raise ValueError("Prior application mismatch")
    stage_names=("440_single","218_corrected","440_compton","440_jscc")
    if set(optimization["states"])!=set(stage_names):
        raise ValueError("Optimization stages missing")
    for name,state in optimization["states"].items():
        if (state["updates"]!=iterations or state["max_objective_increase"]>2e-5 or
            state["max_surrogate_increase"]>1e-10 or not np.isfinite(state["max_inner_gap"])):
            raise ValueError(f"Optimization descent/closure failed: {name}")
        rows=[r for r in optimization["history"] if r["channel"]==name and "objective" in r]
        required=([0,iterations] if phase=="pilot" else list(range(0,iterations+1,50)))
        if [r["iteration"] for r in rows]!=required or not all(np.isfinite(r["objective"]) for r in rows):
            raise ValueError(f"Objective history missing: {name}")
    resources=run["resources"]
    if sorted(r["rank"] for r in resources)!=list(range(run["world_size"])) or sum(r["accepted_events"] for r in resources)!=484936:
        raise ValueError("Rank/event closure failed")
    gpu=max(r["peak_reserved_bytes"]/r["total_device_bytes"] for r in resources)
    nodes={r["node"] for r in resources}
    host=max(sum(r["host_peak_rss_bytes"] for r in resources if r["node"]==node)/
             min(r["host_allocated_bytes"] for r in resources if r["node"]==node) for node in nodes)
    if any(r["host_peak_rss_bytes"]<=0 or r["host_allocated_bytes"]<=0 for r in resources):
        raise ValueError("Missing actual host resource measurement")
    accounting=slurm_memory(job)
    slurm_host=accounting["max_rss_bytes"]/min(r["host_allocated_bytes"] for r in resources)
    if gpu>.8 or host>.8 or slurm_host>.8:
        raise ValueError(f"20% resource margin fails: GPU={gpu}, host={host}, Slurm={slurm_host}")
    record={"passed":True,"study":study["study"],"variant":variant,"phase":phase,
            "result":str(result),"iterations":iterations,"frames":iterations//step,
            "primaries":5*10**9,"accepted_events":484936,"nodes":len(nodes),
            "gpu_reserved_fraction":gpu,"host_peak_fraction":host,"slurm_host_fraction":slurm_host,
            "elapsed_max_seconds":max(r["elapsed_seconds"] for r in resources),"accounting":accounting,
            "study_sha256":digest(study_path),"release_sha256":digest(Path(__file__).parent/"release.json"),
            "run_manifest_sha256":digest(result/"run_manifest.json"),
            "optimization_sha256":digest(result/"optimization.json"),"outputs_sha256":outputs,
            "prediction_sha256":digest(prediction),"optimization_states":optimization["states"]}
    (result/"ablation_integrity.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-root",type=Path,required=True)
    p.add_argument("--study-json",type=Path,required=True)
    p.add_argument("--result",type=Path,required=True)
    p.add_argument("--phase",choices=("pilot","short","formal"),required=True)
    p.add_argument("--variant",required=True)
    p.add_argument("--job",required=True)
    a=p.parse_args()
    report=verify(a.experiment_root,a.study_json,a.result,a.phase,a.variant,a.job)
    gate=a.study_json.parent/"gates"/f"{a.variant}.{a.phase}.json"
    gate.parent.mkdir(exist_ok=True)
    with gate.open("x") as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("SPIKE_ABLATION_GATE_PASSED",json.dumps({k:report[k] for k in ("variant","phase","result","gpu_reserved_fraction","host_peak_fraction","slurm_host_fraction")}))


if __name__=="__main__":
    main()
