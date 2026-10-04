"""Reproduce one frozen event on CPU/CUDA in both arithmetic precisions.

Diagnostic only: does not write a List, sensitivity or reconstruction.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from diagnose_compton_geometry_stability import stable_geometry, sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--release',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    sys.path.insert(0,str(args.release))
    from compton_event_response import (ComptonEventSettings, prepare_compton_events,
        build_detector_position_variance,compton_theta_from_e1,
        _position_angle_sigma,_energy_angle_sigma)
    from detector_csv import load_detector_coordinates
    torch.set_grad_enabled(False);torch.set_num_threads(4)
    source=args.base/'generated/compton_first_scatter_v2/analysis_inputs/NEMA/ideal_v04.csv'
    gp=args.release/'geometry.npz'
    dp=args.base/'generated/FactorsCalibrated/440keV_RotateNum20/Detector.csv'
    raw=np.loadtxt(source,delimiter=',',dtype=np.float32,ndmin=2)[11822:11823,:4]
    coords=np.load(gp)['coordinates_mm'].astype(np.float32)
    detector=load_detector_coordinates(dp,10496).astype(np.float32)
    s=ComptonEventSettings(.440,.13*np.sqrt(511/440),2*.440**2/(.511+2*.440)-.001,.05,.35)
    results=[]
    for device in ('cpu','cuda:0'):
        if device.startswith('cuda') and not torch.cuda.is_available():continue
        for dtype in (torch.float32,torch.float64):
            d=torch.tensor(detector,device=device,dtype=dtype)
            c=torch.tensor(coords,device=device,dtype=dtype)
            v=build_detector_position_variance(d,0)
            p,_=prepare_compton_events(torch.tensor(raw,device=device,dtype=dtype),s,d,v,v,
                input_energies_already_smeared=True)
            assert p.count==1
            a=p.pos1[:,None]-c[None];b=(p.pos2-p.pos1)[:,None]
            beta=torch.acos(((a*b).sum(2)/(a.norm(dim=2)*b.norm(dim=2))).clamp(-1+1e-7,1-1e-7))
            theta=compton_theta_from_e1(p.e1,.440)
            se=_energy_angle_sigma(p.e1,s,beta,theta)
            sp=_position_angle_sigma(a,b,p.sigma_pos1_sq,p.sigma_pos2_sq,True)
            q=(beta-theta[:,None]).abs()/(se**2+sp**2).sqrt();j=int(q.argmin())
            u=a/a.norm(dim=2,keepdim=True);w=b/b.norm(dim=2,keepdim=True)
            cosine=(u*w).sum(2)
            sine=torch.linalg.cross(u,w.expand_as(u),dim=2).norm(dim=2)
            sb,ss,upper=stable_geometry(p.pos1.double()[:,None]-c.double()[None],
                (p.pos2.double()-p.pos1.double())[:,None],p.sigma_pos1_sq,p.sigma_pos2_sq)
            st=compton_theta_from_e1(p.e1.double(),.440)
            es=_energy_angle_sigma(p.e1.double(),s,sb,st)
            sq=(sb-st[:,None]).abs()/(es**2+ss**2).sqrt()
            results.append(dict(device=device,dtype=str(dtype),old_q_min=float(q.min()),
                old_best_index=j,old_best_position_sigma_deg=float(sp[0,j]*180/np.pi),
                old_position_sigma_max_deg=float(sp.max()*180/np.pi),
                old_best_cosine=float(cosine[0,j]),old_best_cross_product_sine=float(sine[0,j]),
                stable_q_min=float(sq.min()),stable_position_sigma_max_deg=float(ss.max()*180/np.pi)))
    out=dict(diagnostic_only=True,view=4,zero_based_merged_row=11822,raw=raw.tolist(),
        source_sha256=sha(source),geometry_sha256=sha(gp),detector_sha256=sha(dp),
        kernel_sha256=sha(args.release/'compton_event_response.py'),code_sha256=sha(__file__),
        prototype_sha256=sha(Path(__file__).with_name('diagnose_compton_geometry_stability.py')),
        torch_version=torch.__version__,cuda_version=torch.version.cuda,results=results)
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
