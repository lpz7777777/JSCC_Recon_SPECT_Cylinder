"""Bounded read-only audit of collinear Compton angular-uncertainty arithmetic.

The stable implementation below is a diagnostic prototype, not a production
response change. No reconstruction, new sensitivity or event list is written.
"""
import argparse
import csv
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

def stable_geometry(a,b,var1,var2):
    """Float64 atan2/cross-product tangent calculation with a finite pole limit.

    At an exactly collinear pole the scalar angle gradient has no unique
    direction. Use a conservative finite covariance upper bound there only;
    this is documented explicitly instead of dividing by an artificial sine.
    """
    a,b,var1,var2=(x.double() for x in (a,b,var1,var2))
    da=a.norm(dim=2,keepdim=True);db=b.norm(dim=2,keepdim=True)
    if bool((da<=0).any() or (db<=0).any()):raise ValueError('Coincident positions')
    u,v=a/da,b/db
    cross=torch.linalg.cross(u,v.expand_as(u),dim=2)
    sine=cross.norm(dim=2,keepdim=True);cosine=(u*v).sum(2,keepdim=True)
    beta=torch.atan2(sine,cosine).squeeze(2)
    normal=cross/sine.clamp_min(1e-12)
    t1=torch.linalg.cross(normal,u,dim=2)/da
    t2=torch.linalg.cross(v.expand_as(u),normal,dim=2)/db
    grad1=t2-t1;grad2=-t2
    variance=(grad1.square()*var1[:,None]).sum(2)+(grad2.square()*var2[:,None]).sum(2)
    upper=(var1.amax(1)[:,None]*(1/da.squeeze(2)+1/db.squeeze(2))**2
           +var2.amax(1)[:,None]/db.squeeze(2)**2)
    variance=torch.where(sine.squeeze(2)<=1e-12,upper,variance)
    if bool((variance>upper*(1+1e-10)).any()):raise ValueError('Physical derivative bound exceeded')
    return beta,variance.clamp_min(1e-12).sqrt(),upper.sqrt()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('base','release','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--sample-per-view',type=int,default=64)
    ap.add_argument('--all-events',action='store_true')
    ap.add_argument('--images',type=Path)
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.release))
    from compton_event_response import (ComptonEventSettings,PreparedComptonEvents,
        prepare_compton_events,build_detector_position_variance,compton_theta_from_e1,
        _position_angle_sigma,_energy_angle_sigma,min_standardized_compton_arm)
    from detector_csv import load_detector_coordinates
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    if args.device.startswith('cuda'):torch.cuda.set_device(args.device)
    base=args.base;study=base/'generated/compton_first_scatter_v2';factor=base/'generated/FactorsCalibrated/440keV_RotateNum20'
    gp=base/'generated/Geometry/geometry.npz'
    if not gp.exists():gp=args.release/'geometry.npz'
    gate=json.loads((study/'analysis/validation_gate.json').read_text())
    assert sha(args.release/'compton_event_response.py')==gate['kernel_sha256']
    g=np.load(gp);coords=torch.tensor(g['coordinates_mm'],dtype=torch.float32,device=args.device)
    detector=torch.tensor(load_detector_coordinates(factor/'Detector.csv',10496),device=args.device)
    var=build_detector_position_variance(detector,0)
    s=ComptonEventSettings(.440,.13*np.sqrt(511/440),2*.440**2/(.511+2*.440)-.001,.05,.35)
    manifest=json.loads((study/'analysis_inputs/input_manifest.json').read_text())['files']
    raw_b=np.memmap(factor/'SysMat_polar','<f4',mode='r',shape=(132040,10496))
    matrix=torch.tensor(np.array(raw_b.T,copy=True),device=args.device);del raw_b
    rng=np.random.default_rng(20261004);rows=[];sources={};start=time.monotonic()
    images={}
    if args.images:
        for name,array in np.load(args.images).items():
            assert array.shape==(82040,)
            images[name]=(torch.tensor(array,device=args.device,dtype=torch.float64),int(array.argmax()))
    for view in range(1,21):
        path=study/f'analysis_inputs/NEMA/ideal_v{view:02d}.csv';file_sha=sha(path)
        assert file_sha==manifest[path.relative_to(study/'analysis_inputs').as_posix()]
        sources[path.name]=file_sha
        kept=np.load(study/f'analysis/NEMA_ideal_v{view:02d}_kept_rows.npy')
        selected=kept if args.all_events else np.sort(rng.choice(kept,min(args.sample_per_view,len(kept)),replace=False))
        # Include the discovered witness in addition to the predeclared random sample.
        if view==4 and 11822 not in selected:selected=np.sort(np.r_[selected,11822])
        raw=np.loadtxt(path,delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
        for off in range(0,len(selected),8):
            ix=selected[off:off+8];p,_=prepare_compton_events(torch.tensor(raw[ix],device=args.device),s,detector,var,var,input_energies_already_smeared=True)
            assert p.count==len(ix)
            a=p.pos1[:,None]-coords[None];b=(p.pos2-p.pos1)[:,None]
            beta=torch.acos(((a*b).sum(2)/(a.norm(dim=2)*b.norm(dim=2))).clamp(-1+1e-7,1-1e-7))
            theta=compton_theta_from_e1(p.e1,.440)
            se=_energy_angle_sigma(p.e1,s,beta,theta)
            old_sp=_position_angle_sigma(a,b,p.sigma_pos1_sq,p.sigma_pos2_sq,True)
            old_q=(beta-theta[:,None]).abs()/(se**2+old_sp**2).sqrt()
            # Cast inputs before vector subtraction; avoid float32 cancellation first.
            aa=p.pos1.double()[:,None]-coords.double()[None];bb=(p.pos2.double()-p.pos1.double())[:,None]
            sb,sp,upper=stable_geometry(aa,bb,p.sigma_pos1_sq,p.sigma_pos2_sq)
            st=compton_theta_from_e1(p.e1.double(),.440);es=_energy_angle_sigma(p.e1.double(),s,sb,st)
            new_q=(sb-st[:,None]).abs()/(es**2+sp**2).sqrt()
            # Original formula in float64 is a useful reference away from a pole.
            ref_sp=_position_angle_sigma(aa,bb,p.sigma_pos1_sq.double(),p.sigma_pos2_sq.double(),True)
            ref_beta=torch.acos(((aa*bb).sum(2)/(aa.norm(dim=2)*bb.norm(dim=2))).clamp(-1+1e-7,1-1e-7))
            ref_se=_energy_angle_sigma(p.e1.double(),s,ref_beta,st)
            ref_q=(ref_beta-st[:,None]).abs()/(ref_se**2+ref_sp**2).sqrt()
            old_kn=(.440/(.440-p.e1)+(.440-p.e1)/.440)[:,None]-beta.sin()**2
            new_kn=(.440/(.440-p.e1.double())+(.440-p.e1.double())/.440)[:,None]-sb.sin()**2
            old=torch.exp(-old_q**2/2)*old_kn*matrix[p.cpnum1-1]
            new=torch.exp(-new_q**2/2)*new_kn*matrix[p.cpnum1-1].double()
            old/=old.sum(1,keepdim=True);new/=new.sum(1,keepdim=True)
            indices=torch.tensor(g['inverse_rotation'][g['active_indices'],view-1],device=args.device,dtype=torch.long)
            fractions=torch.tensor(g['ellipse_fraction'][g['active_indices']],device=args.device)
            oa=old[:,indices]*fractions;na=new[:,indices]*fractions
            oa/=oa.sum(1,keepdim=True);na/=na.sum(1,keepdim=True)
            for k,row in enumerate(ix):
                j=int(old_q[k].argmin());bad=old_sp[k].double()>upper[k]*(1+1e-4)
                rec=dict(view=view,input_row=int(row),witness=bool(view==4 and row==11822),
                    q_current_gpu=float(old_q[k].min()),q_reference_float64=float(ref_q[k].min()),q_stable=float(new_q[k].min()),
                    max_old_sigma_deg=float(old_sp[k].max()*180/np.pi),max_stable_sigma_deg=float(sp[k].max()*180/np.pi),
                    columns_exceeding_derivative_bound=int(bad.sum()),
                    old_q_min_grid_index=j,old_q_min_sigma_deg=float(old_sp[k,j]*180/np.pi),
                    full_row_total_variation=float((old[k].double()-new[k]).abs().sum()/2),
                    ellipse_conditional_total_variation=float((oa[k]-na[k]).abs().sum()/2))
                for name,(image,peak) in images.items():
                    rec[name+'_peak_responsibility_old']=float(oa[k,peak]*image[peak]/torch.dot(oa[k],image))
                    rec[name+'_peak_responsibility_stable']=float(na[k,peak]*image[peak]/torch.dot(na[k],image))
                rows.append(rec)
        print('STABILITY_VIEW',view,len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    with (args.output/'events.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    sample=rows if args.all_events else [r for r in rows if not r['witness']]
    stats=dict(events=len(sample),events_exceeding_derivative_bound=sum(r['columns_exceeding_derivative_bound']>0 for r in sample),
        old_kept_but_stable_q_over_3=sum(r['q_current_gpu']<=3 and r['q_stable']>3 for r in sample),
        max_old_sigma_deg=max(r['max_old_sigma_deg'] for r in sample),
        full_row_TV_quantiles=np.quantile([r['full_row_total_variation'] for r in sample],[.5,.9,.99,1]).tolist(),
        ellipse_row_TV_quantiles=np.quantile([r['ellipse_conditional_total_variation'] for r in sample],[.5,.9,.99,1]).tolist(),
        ellipse_row_TV_above_01=sum(r['ellipse_conditional_total_variation']>.1 for r in sample),
        current_q_over_3=sum(r['q_current_gpu']>3.00001 for r in sample),
        max_abs_stable_reference_q_difference=max(abs(r['q_stable']-r['q_reference_float64']) for r in sample))
    attribution={}
    for name in images:
        total=sum(r[name+'_peak_responsibility_old'] for r in sample)
        attribution[name]=dict(total_peak_responsibility_old=total,
            total_peak_responsibility_stable_same_image=sum(r[name+'_peak_responsibility_stable'] for r in sample),
            old_responsibility_from_new_q_rejects=sum(r[name+'_peak_responsibility_old'] for r in sample if r['q_stable']>3),
            old_responsibility_from_TV_gt_01=sum(r[name+'_peak_responsibility_old'] for r in sample if r['ellipse_conditional_total_variation']>.1),
            note='Single likelihood evaluation at frozen old image, not a reconstruction or matched-sensitivity prediction')
    result=dict(diagnostic_only=True,production_changed=False,sample_per_view=args.sample_per_view,
        seed=20261004,all_events=args.all_events,sample_stats=stats,peak_attribution=attribution,witness=[r for r in rows if r['witness']],
        production_kernel_sha256=gate['kernel_sha256'],prototype_sha256=sha(__file__),geometry_sha256=sha(gp),input_sha256=sources,
        runtime_seconds=time.monotonic()-start,gpu_max_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
        interpretation='Random frozen ideal accepted events plus one separately flagged witness; row TV includes original B and ellipse overlap. No image updates or sensitivity changes.')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(stats),flush=True)

if __name__=='__main__':main()
