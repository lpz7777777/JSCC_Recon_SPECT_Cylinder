"""Single-energy MLEM and axial PSF metrics for a 20-view point dataset."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"distributed/dual_energy_compton_python")]
from fov_config import factor_geometry
from reconstruction import run_single_mlem_dist,save_result


def axial_fwhm(z,profile):
    peak=int(np.argmax(profile));half=profile[peak]/2
    left=np.flatnonzero(profile[:peak]<half)
    right=np.flatnonzero(profile[peak+1:]<half)+peak+1
    if not len(left) or not len(right):
        return {"fwhm_mm":None,"censored_at_boundary":True}
    a,b=int(left[-1]),int(right[0])
    lo=z[a]+(half-profile[a])*(z[a+1]-z[a])/(profile[a+1]-profile[a])
    hi=z[b-1]+(half-profile[b-1])*(z[b]-z[b-1])/(profile[b]-profile[b-1])
    return {"fwhm_mm":float(hi-lo),"censored_at_boundary":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor",type=Path,required=True)
    parser.add_argument("--projection",type=Path,required=True)
    parser.add_argument("--position-mm",type=float,nargs=3,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--iterations",type=int,default=1000)
    args=parser.parse_args()
    if args.iterations<=0 or args.iterations%50:
        raise ValueError("Iterations must be a positive multiple of 50")
    if args.output.exists():raise FileExistsError(args.output)
    coords,nxy,nz=factor_geometry(args.factor)
    matrix=np.fromfile(args.factor/"SysMat_polar",dtype="<f4").reshape(len(coords),-1).T.copy()
    a=torch.from_numpy(matrix).to(args.device)
    rot=torch.tensor(np.loadtxt(args.factor/"RotMat_full.csv",delimiter=",",dtype=np.int64),device=args.device)
    inv=torch.tensor(np.loadtxt(args.factor/"RotMatInv_full.csv",delimiter=",",dtype=np.int64),device=args.device)
    counts=np.loadtxt(args.projection,delimiter=",")
    if counts.shape!=(20,a.shape[0]) or not np.isfinite(counts).all() or np.any(counts<0):
        raise ValueError("Expected 20-view point counts")
    sensitivity=torch.stack([a.sum(0)[inv[:,v]-1] for v in range(20)]).mean(0)[:,None]
    result=run_single_mlem_dist("Point",a,torch.tensor(counts.T,dtype=torch.float32,device=args.device),
        rot,inv,sensitivity,args.iterations,50,0)
    args.output.mkdir(parents=True,exist_ok=False)
    save_result(args.output,"Point",result,args.iterations,50,0)
    image=result.image.cpu().numpy().reshape(nz,nxy)
    xy=coords[:nxy,:2];index=int(np.argmin(np.linalg.norm(xy-np.asarray(args.position_mm)[:2],axis=1)))
    z=np.unique(coords[:,2]);profile=image[:,index]
    peak=coords[int(np.argmax(image))]
    metrics={"position_mm":args.position_mm,"peak_position_mm":peak.tolist(),
             "peak_error_mm":float(np.linalg.norm(peak-args.position_mm)),
             "profile_xy_mm":xy[index].tolist(),"profile_z_mm":z.tolist(),
             "profile":profile.tolist(),**axial_fwhm(z,profile)}
    (args.output/"point_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")


if __name__=="__main__":main()
