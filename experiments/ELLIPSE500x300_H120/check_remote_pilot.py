"""Read small pilot outputs over SFTP and verify six complete finite images."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"FOV120"))
from reconstruction_ssh import connect

ROOT=("/data/run01/scxi717/lpz/"
      "20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/"
      "experiments/ELLIPSE500x300_H120/generated/Results")
CHANNELS=("440_SinglePhoton","440_ComptonOnly","440_SinglePlusCompton",
          "218_SinglePhoton_CrossTalkCorrected","440SinglePlus218Single",
          "440SingleComptonPlus218Single")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("result_name")
    args=p.parse_args()
    if "/" in args.result_name or "\\" in args.result_name:
        p.error("result_name must be a single directory name")
    folder=f"{ROOT}/{args.result_name}"
    with connect() as ssh, ssh.open_sftp() as sftp:
        with sftp.open(f"{folder}/run_manifest.json","r") as stream:
            manifest=json.load(stream)
        if manifest["iterations"]!=10 or not manifest["pilot_only"]:
            raise ValueError("Expected 10-iteration pilot")
        results=[]
        for channel in CHANNELS:
            shapes=(("active",82040),("full",132040),("history",82040))
            for suffix,count in shapes:
                name=f"Image_{channel}_{suffix}.float32"
                with sftp.open(f"{folder}/{name}","rb") as stream:
                    data=np.frombuffer(stream.read(),dtype="<f4")
                if data.size!=count or not np.isfinite(data).all() or (data<0).any():
                    raise ValueError(f"Invalid output {name}: {data.size} values")
                if suffix=="full":
                    results.append({"channel":channel,"sum":float(data.sum()),
                                    "positive_voxels":int(np.count_nonzero(data))})
        with sftp.open(f"{folder}/PredictedCntStat_218_From440.float32","rb") as stream:
            predicted=np.frombuffer(stream.read(),dtype="<f4")
        if predicted.size!=10496*20 or not np.isfinite(predicted).all() or (predicted<0).any():
            raise ValueError("Invalid cross-talk prediction")
        report={"job":args.result_name,"accepted_events":manifest["accepted_compton_events"],
                "world_size":manifest["world_size"],"channels":results,
                "peak_reserved_gib":max(r["peak_reserved_bytes"] for r in manifest["resources"])/2**30,
                "min_gpu_capacity_gib":min(r["total_device_bytes"] for r in manifest["resources"])/2**30}
        print(json.dumps(report,indent=2))

if __name__=="__main__": main()
