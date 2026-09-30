"""In-situ integrity check for a 10000-iteration ellipse reconstruction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/"generated"
CHANNELS=("440_SinglePhoton","440_ComptonOnly","440_SinglePlusCompton",
          "218_SinglePhoton_CrossTalkCorrected","440SinglePlus218Single",
          "440SingleComptonPlus218Single")

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()

def check_array(path,count):
    if path.stat().st_size!=count*4:
        raise ValueError(f"File length mismatch: {path}")
    values=np.memmap(path,mode="r",dtype="<f4",shape=(count,))
    if not np.isfinite(values).all() or np.any(values<0):
        raise ValueError(f"Nonfinite or negative values: {path}")
    return values

def main():
    p=argparse.ArgumentParser()
    p.add_argument("result",type=Path)
    args=p.parse_args()
    result=args.result.resolve()
    if result.parent!= (ROOT/"Results").resolve():
        raise ValueError("Result must be directly under this experiment's Results directory")
    run=json.loads((result/"run_manifest.json").read_text())
    if (run["experiment"]!="ELLIPSE500x300_H120" or
        run["iterations"]!=10000 or run["save_step"]!=50 or
        run["pilot_only"] or run["pixels_full"]!=132040 or
        run["pixels_active"]!=82040 or run["world_size"]<1):
        raise ValueError("Formal run metadata mismatch")
    collection_path=ROOT/"collections"/f'{run["dataset"]}_{run["count_level"]}.json'
    collection=json.loads(collection_path.read_text())
    if (collection["dataset"]!=run["dataset"] or
        collection["level"]!=run["count_level"] or
        collection["views"]!=list(range(1,21)) or
        sum(collection["primary_counts"])!=10**9 or
        len(set(collection["worker_indices"]))!=200):
        raise ValueError("Geant4 collection provenance mismatch")
    with np.load(ROOT/"Geometry/geometry.npz") as geometry:
        active=geometry["active_indices"]
    if len(active)!=82040 or np.any(np.diff(active)<=0):
        raise ValueError("Active index mismatch")
    geometry_hash=sha256(ROOT/"Geometry/geometry.npz")
    sensi_hash=sha256(ROOT/"FactorsCalibrated/440keV_RotateNum20/Sensi_d")
    if geometry_hash!=run["geometry_sha256"] or sensi_hash!=run["sensi_d_sha256"]:
        raise ValueError("Geometry or Compton sensitivity hash mismatch")
    if sum(x["accepted_events"] for x in run["resources"])!=run["accepted_compton_events"]:
        raise ValueError("Accepted event total mismatch")
    if len(run["resources"])!=run["world_size"] or run["accepted_compton_events"]<=0:
        raise ValueError("Incomplete distributed resources")
    gpu_fraction=max(x["peak_reserved_bytes"]/x["total_device_bytes"]
                     for x in run["resources"])
    if gpu_fraction>.8:
        raise ValueError(f"GPU reserve exceeds 80% of device capacity: {gpu_fraction}")
    outputs=[]
    for channel in CHANNELS:
        paths={suffix:result/f"Image_{channel}_{suffix}.float32"
               for suffix in ("active","full","history")}
        active_values=check_array(paths["active"],82040)
        full=check_array(paths["full"],132040)
        history=check_array(paths["history"],200*82040).reshape(200,82040)
        if not np.array_equal(full[active],active_values):
            raise ValueError(f"Full/active image mismatch: {channel}")
        if np.count_nonzero(full)!=np.count_nonzero(active_values):
            raise ValueError(f"Non-active image voxels are nonzero: {channel}")
        if not np.array_equal(history[-1],active_values):
            raise ValueError(f"Last history frame differs from final image: {channel}")
        outputs.append({"channel":channel,"density_sum":float(active_values.sum(dtype=np.float64)),
                        "nonzero_voxels":int(np.count_nonzero(active_values)),
                        "sha256":{key:sha256(path) for key,path in paths.items()}})
    predicted_path=result/"PredictedCntStat_218_From440.float32"
    predicted=check_array(predicted_path,10496*20)
    report={"experiment":run["experiment"],"dataset":run["dataset"],
            "job_result":result.name,"primaries":sum(collection["primary_counts"]),
            "views":len(collection["views"]),"worker_count":len(collection["worker_indices"]),
            "accepted_compton_events":run["accepted_compton_events"],
            "history_frames_per_channel":200,"gpu_peak_reserved_fraction":gpu_fraction,
            "geometry_sha256":geometry_hash,"sensi_d_sha256":sensi_hash,
            "collection_sha256":sha256(collection_path),"outputs":outputs,
            "predicted_sha256":sha256(predicted_path),
            "predicted_sum":float(predicted.sum(dtype=np.float64))}
    out=result/"integrity_report.json"
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="outputs"},indent=2))
    print("ELLIPSE_FORMAL_INTEGRITY_OK",out)

if __name__=="__main__": main()
