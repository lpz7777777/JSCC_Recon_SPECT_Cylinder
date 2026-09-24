"""Compare uncalibrated 60/120-mm Cartesian response columns at common coordinates."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(old,new,detectors=11520):
    for path,nz in ((old,20),(new,40)):
        if path.stat().st_size!=detectors*nz*51*51*4:
            raise ValueError(f"Wrong raw matrix byte size: {path}")
    a=np.memmap(old,dtype="<f4",mode="r",shape=(detectors,20,51,51))
    b=np.memmap(new,dtype="<f4",mode="r",shape=(detectors,40,51,51))
    error2=reference2=maximum=0.
    for start in range(0,detectors,16):
        x=np.asarray(a[start:start+16],dtype=np.float64)
        y=np.asarray(b[start:start+16,10:30],dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x<0) or np.any(y<0):
            raise ValueError("Invalid response in overlap")
        difference=y-x
        error2+=float(np.sum(difference**2));reference2+=float(np.sum(x**2))
        maximum=max(maximum,float(np.max(np.abs(difference))))
    return {"relative_l2":float(np.sqrt(error2/max(reference2,1e-300))),"max_abs":maximum,
            "old":str(old),"new":str(new),"new_z_slice_half_open":[10,30]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old",type=Path,required=True);parser.add_argument("--new",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--relative-tolerance",type=float,default=1e-5)
    args=parser.parse_args();report=compare(args.old,args.new)
    report["passed"]=report["relative_l2"]<=args.relative_tolerance
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    if not report["passed"]:raise SystemExit("Common-column regression failed; inspect before production")
