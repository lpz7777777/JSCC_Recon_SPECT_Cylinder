"""Compare XCAT final gamma-channel density with the 3-mm source truth."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

HERE=Path(__file__).resolve().parent
ROOT=HERE/"generated"
CHANNELS=(("440_SinglePhoton","bi"),("440_ComptonOnly","bi"),
          ("440_SinglePlusCompton","bi"),
          ("218_SinglePhoton_CrossTalkCorrected","fr"))
ORGANS=("body","kidney","liver","lesion")

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("result_name")
    args=p.parse_args()
    result=ROOT/"RemoteResults"/args.result_name
    integrity=json.loads((HERE/"reports"/f"{args.result_name}_integrity.json").read_text())
    if integrity["dataset"]!="XCAT" or integrity["job_result"]!=args.result_name:
        raise ValueError("A verified XCAT formal result is required")
    truth_file=ROOT/"XCAT_1e9_fullx/truth_3mm.npz"
    with np.load(truth_file) as f:
        truth={key:np.asarray(f[f"{key}_zyx"],dtype=np.float64) for key in ("fr","bi")}
        masks={key:np.asarray(f[f"{key}_fraction_zyx"],dtype=np.float64)
               for key in ORGANS}
        x,y,z=(f[f"{axis}_mm"] for axis in ("x","y","z"))
    with np.load(ROOT/"Geometry/geometry.npz") as f:
        coords=f["coordinates_mm"]
    xy=coords[:3301,:2]
    if not np.allclose(coords[::3301,2],z,atol=1e-8):
        raise ValueError("Polar and XCAT axial coordinates differ")
    xx,yy=np.meshgrid(x,y,indexing="xy")
    in_ellipse=(xx/250)**2+(yy/150)**2<=1
    tri=Delaunay(xy)
    query=np.column_stack((xx.ravel(),yy.ravel()))
    report={"result":args.result_name,"truth_sha256":digest(truth_file),
            "method":"Linear XY interpolation per 3-mm z slice; global integral normalization within physical ellipse; no smoothing",
            "kidney_truth_retained_fraction":.8191108946709337,
            "channels":{}}
    images={}
    for channel,truth_name in CHANNELS:
        path=result/f"Image_{channel}_full.float32"
        expected=next(row["sha256"]["full"] for row in integrity["outputs"]
                      if row["channel"]==channel)
        if digest(path)!=expected:
            raise ValueError(f"Image changed after integrity validation: {channel}")
        polar=np.memmap(path,mode="r",dtype="<f4",shape=(40,3301))
        cart=np.empty((40,len(y),len(x)),dtype=np.float64)
        for k in range(40):
            interpolator=LinearNDInterpolator(tri,polar[k],fill_value=0.)
            cart[k]=interpolator(query).reshape(len(y),len(x))
        cart=np.maximum(cart,0)
        cart[:,~in_ellipse]=0
        target=truth[truth_name]
        scale=target.sum(dtype=np.float64)/cart.sum(dtype=np.float64)
        scaled=cart*scale
        organs={}
        for organ in ORGANS:
            mask=masks[organ]
            expected_integral=np.sum(target*mask,dtype=np.float64)*27
            got_integral=np.sum(scaled*mask,dtype=np.float64)*27
            organs[organ]={"truth_integral":float(expected_integral),
                           "recon_integral":float(got_integral),
                           "relative_recovery":float(got_integral/expected_integral)
                           if expected_integral>0 else None}
        body=masks["body"]>.5
        difference=scaled[body]-target[body]
        nrmse=np.sqrt(np.mean(difference*difference))/np.mean(target[body])
        corr=np.corrcoef(scaled[body],target[body])[0,1]
        report["channels"][channel]={"truth_channel":truth_name,
                                    "global_scale":float(scale),
                                    "body_mean_normalized_rmse":float(nrmse),
                                    "body_pearson":float(corr),
                                    "organs":organs}
        if channel in ("440_SinglePlusCompton","218_SinglePhoton_CrossTalkCorrected"):
            images[channel]=scaled
    report_path=HERE/"reports"/f"{args.result_name}_xcat_spatial.json"
    report_path.write_text(json.dumps(report,indent=2)+"\n")
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained")
    for row,(channel,truth_name,label) in enumerate((("218_SinglePhoton_CrossTalkCorrected","fr","218 corrected"),
                                                      ("440_SinglePlusCompton","bi","440 joint"))):
        reference=truth[truth_name][20]
        estimate=images[channel][20]
        vmax=2*np.percentile(reference[masks["body"][20]>.5],99)
        for col,(data,title) in enumerate(((reference,f"{truth_name.upper()} truth"),
                                            (estimate,label))):
            axes[row,col].imshow(data,origin="lower",extent=(x[0]-1.5,x[-1]+1.5,
                                 y[0]-1.5,y[-1]+1.5),vmin=0,vmax=vmax,cmap="inferno")
            axes[row,col].set_title(title)
            axes[row,col].set_xlabel("x (mm)")
            axes[row,col].set_ylabel("y (mm)")
    fig.suptitle("XCAT central axial slice, 10⁹ primaries, 10000 iterations\n"
                 "Display clipped at 2× truth p99; metrics use unmodified values")
    figure=report_path.with_suffix(".png")
    fig.savefig(figure,dpi=170)
    print(report_path)
    print(figure)
    for key,value in report["channels"].items():
        print(key,"NRMSE",value["body_mean_normalized_rmse"],
              "kidney",value["organs"]["kidney"]["relative_recovery"],
              "lesion",value["organs"]["lesion"]["relative_recovery"])

if __name__=="__main__": main()
