"""Object-frame uniformity of the independent ellipse volume source."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
CHANNELS=("440_SinglePhoton","440_ComptonOnly","440_SinglePlusCompton",
          "218_SinglePhoton_CrossTalkCorrected")
ITERATIONS=(50,100,500,1000,2000,5000,10000)

def stats(values,weights):
    mean=np.average(values,weights=weights)
    variance=np.average((values-mean)**2,weights=weights)
    return {"voxels":int(len(values)),"mean_density":float(mean),
            "cv":float(np.sqrt(variance)/mean) if mean>0 else None,
            "median_density":float(np.median(values)),
            "p99_density":float(np.percentile(values,99))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("result",type=Path)
    args=p.parse_args()
    result=args.result.resolve()
    if result.parent!=(HERE/"generated/Results").resolve():
        raise ValueError("Result must belong to this experiment")
    run=json.loads((result/"run_manifest.json").read_text())
    if run["dataset"]!="EllipseUniform" or run["iterations"]!=10000:
        raise ValueError("Only complete EllipseUniform reconstruction is supported")
    with np.load(HERE/"generated/Geometry/geometry.npz") as g:
        coords=g["coordinates_mm"]
        volume=g["cell_volume_mm3"]*g["ellipse_fraction"]
        active=g["active_indices"]
    x=coords[:,0]/250
    y=coords[:,1]/150
    rho=np.sqrt(x*x+y*y)
    absz=np.abs(coords[:,2])
    supported=volume>0
    masks={
        "core":(rho<=.5)&(absz<=30)&supported,
        "middle":(rho>.5)&(rho<=.9)&(absz<=30)&supported,
        "outer":(rho>.9)&(rho<=.98)&(absz<=30)&supported,
        "z_center":(rho<=.9)&(absz<=30)&supported,
        "z_mid":(rho<=.9)&(absz>30)&(absz<=45)&supported,
        "z_edge":(rho<=.9)&(absz>45)&(absz<=60)&supported,
        "long_axis_outer":(rho>.75)&(rho<=.98)&(np.abs(x)>=np.abs(y))&
                          (absz<=45)&supported,
        "short_axis_outer":(rho>.75)&(rho<=.98)&(np.abs(x)<np.abs(y))&
                           (absz<=45)&supported,
    }
    report={"result":result.name,"source":"uniform elliptical cylinder a=250 mm, b=150 mm, |z|<=60 mm",
            "basis":"volume-weighted density; physical overlap fraction once",
            "channels":{}}
    for channel in CHANNELS:
        image=np.memmap(result/f"Image_{channel}_full.float32",mode="r",
                        dtype="<f4",shape=(132040,))
        history=np.memmap(result/f"Image_{channel}_history.float32",mode="r",
                          dtype="<f4",shape=(200,82040))
        rows={}
        for name,mask in masks.items():
            values=np.asarray(image[mask],dtype=np.float64)
            weight=volume[mask]
            row=stats(values,weight)
            if name in ("core","z_center","z_edge"):
                active_mask=mask[active]
                row["cv_history"]=[]
                for iteration in ITERATIONS:
                    frame=np.asarray(history[iteration//50-1,active_mask],dtype=np.float64)
                    row["cv_history"].append({"iteration":iteration,
                                               "cv":stats(frame,weight)["cv"]})
            rows[name]=row
        center=rows["z_center"]["mean_density"]
        for name in ("z_mid","z_edge"):
            rows[name]["mean_over_z_center"]=rows[name]["mean_density"]/center
        rows["outer"]["mean_over_core"]=rows["outer"]["mean_density"]/rows["core"]["mean_density"]
        rows["long_axis_outer"]["mean_over_short_axis_outer"]=(
            rows["long_axis_outer"]["mean_density"]/
            rows["short_axis_outer"]["mean_density"])
        report["channels"][channel]=rows
    out=result/"ellipse_uniformity_report.json"
    out.write_text(json.dumps(report,indent=2)+"\n")
    print("ELLIPSE_UNIFORMITY_OK",out)

if __name__=="__main__": main()
