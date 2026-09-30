"""Volume-weighted axial/radial uniformity for a CircleNewDist result."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
CHANNELS=("440_SinglePhoton","440_ComptonOnly","440_SinglePlusCompton",
          "218_SinglePhoton_CrossTalkCorrected")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("result",type=Path)
    args=p.parse_args()
    result=args.result.resolve()
    if result.parent!=(HERE/"generated/Results").resolve():
        raise ValueError("Result must belong to this experiment")
    run=json.loads((result/"run_manifest.json").read_text())
    if run["dataset"]!="CircleNewDist" or run["iterations"]!=10000:
        raise ValueError("Only complete CircleNewDist reconstruction is supported")
    with np.load(HERE/"generated/Geometry/geometry.npz") as g:
        coords=g["coordinates_mm"]
        volume=g["cell_volume_mm3"]*g["ellipse_fraction"]
        active=g["active_indices"]
    r=np.linalg.norm(coords[:,:2],axis=1)
    absz=np.abs(coords[:,2])
    bands=(("center",absz<=30),("mid",(absz>30)&(absz<=45)),
           ("edge",(absz>45)&(absz<=60)))
    report={"result":result.name,"source":"uniform cylinder radius 150 mm, |z|<=60 mm",
            "roi":"r<=135 mm, full axial bands; volume-weighted density",
            "channels":{}}
    for channel in CHANNELS:
        image=np.memmap(result/f"Image_{channel}_full.float32",mode="r",
                        dtype="<f4",shape=(132040,))
        rows={}
        history=np.memmap(result/f"Image_{channel}_history.float32",mode="r",
                          dtype="<f4",shape=(200,82040))
        for name,zmask in bands:
            mask=(r<=135)&zmask&(volume>0)
            values=np.asarray(image[mask],dtype=np.float64)
            weight=volume[mask]
            mean=np.average(values,weights=weight)
            var=np.average((values-mean)**2,weights=weight)
            rows[name]={"voxels":int(mask.sum()),"mean_density":float(mean),
                        "cv":float(np.sqrt(var)/mean) if mean>0 else None,
                        "median_density":float(np.median(values)),
                        "p99_density":float(np.percentile(values,99))}
            active_mask=mask[active]
            active_weight=weight
            curve=[]
            for iteration in (50,100,500,1000,2000,5000,10000):
                frame=np.asarray(history[iteration//50-1,active_mask],dtype=np.float64)
                frame_mean=np.average(frame,weights=active_weight)
                frame_var=np.average((frame-frame_mean)**2,weights=active_weight)
                curve.append({"iteration":iteration,"cv":float(np.sqrt(frame_var)/frame_mean)
                              if frame_mean>0 else None})
            rows[name]["cv_history"]=curve
        center=rows["center"]["mean_density"]
        for name in ("mid","edge"):
            rows[name]["mean_over_center"]=rows[name]["mean_density"]/center if center>0 else None
        report["channels"][channel]=rows
    path=result/"circle_uniformity_report.json"
    path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
