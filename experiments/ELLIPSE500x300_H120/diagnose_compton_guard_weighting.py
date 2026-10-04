"""Existing independent point-source events weight the A interpolation residual.

Neither NEMA truth nor a reconstructed image enters this bounded diagnostic.
It is not a cell integral or a matched-S gate and never authorizes imaging.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from generate_compton_a_guard import digest,axes
from build_compton_a_guard_field import combined
from compton_boundary_quadrature import GuardedPolarResponseField
from compton_event_response import (ComptonEventSettings,prepare_compton_events,
    build_detector_position_variance,build_compton_cone_weights,min_standardized_compton_arm)
from detector_csv import load_detector_coordinates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('field','guard','radial','inputs','factors','geometry','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    manifest=json.loads((a.field/'field_manifest.json').read_text())
    if digest(a.field/'A_field.float64')!=manifest['A_field_sha256']:raise ValueError('A field hash differs')
    if digest(a.geometry)!=manifest['baseline_geometry_sha256']:raise ValueError('Frozen geometry differs')
    a.output.mkdir(parents=True,exist_ok=False)
    fg=np.load(a.field/'field_geometry.npz');field=np.memmap(a.field/'A_field.float64',mode='r',dtype='<f8',shape=tuple(manifest['shape']))
    interp=GuardedPolarResponseField(fg['xy_mm'],int(fg['original_points']),fg['z_mm'])
    selected=fg['selected_raw_detectors'];scale=np.asarray(manifest['calibration_scales'])
    controls=[]
    for directory,parts in ((a.guard,('mid_minus','mid_plus')),(a.radial,('radial_mid',))):
        for name in parts:
            values=combined(directory,name);spec=json.loads((directory/name/'complete.json').read_text())['spec']
            xs,ys,zs=axes(spec)
            for zi,z in enumerate(zs):
                for yi,y in enumerate(ys):
                    for xi,x in enumerate(xs):
                        if np.hypot(x,y)>255:continue
                        point=np.array([[x,y,z]])
                        controls.append(dict(part=name,xyz=point[0],cache=interp.cache(point),
                            actual=values[selected,zi,yi,xi].astype(float)*scale))
    torch.set_grad_enabled(False);torch.set_num_threads(8);device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    factor=a.factors/'440keV_RotateNum20';detector=torch.tensor(load_detector_coordinates(factor/'Detector.csv',10496),device=device)
    var=build_detector_position_variance(detector,0)
    settings=ComptonEventSettings(.440,.13*np.sqrt(511/440),2*.440**2/(.511+2*.440)-.001,.05,.35,geometry_mode='stable_float64')
    coords=torch.tensor(np.load(a.geometry)['coordinates_mm'],dtype=torch.float32,device=device)
    locations=torch.tensor(np.array([c['xyz'] for c in controls]),dtype=torch.float64,device=device)
    B=np.memmap(factor/'SysMat_polar',dtype='<f4',mode='r',shape=(132040,10496))
    sums=np.zeros((len(controls),4));event_counts=[];cache={};start=time.monotonic()
    for folder in sorted(a.inputs.glob('point_*')):
        for view in (1,6,11,16):
            path=folder/f'ideal_v{view:02d}.csv'
            raw=np.loadtxt(path,delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)[:32]
            prepared,_=prepare_compton_events(torch.tensor(raw,device=device),settings,detector,var,var,input_energies_already_smeared=True)
            if prepared is None:continue
            q=min_standardized_compton_arm(prepared,coords,settings);mask=q<=3
            c1=(prepared.cpnum1[mask].cpu().numpy()-1).astype(int)
            if not len(c1):continue
            k=build_compton_cone_weights(prepared,locations,settings)[mask].double().cpu().numpy()
            fullk=build_compton_cone_weights(prepared,coords,settings)[mask].double()
            norm=(fullk*torch.tensor(np.asarray(B[:,c1]).T,device=device)).sum(1).cpu().numpy()
            if not np.isfinite(norm).all() or np.any(norm<=0):raise ValueError('Zero reference in diagnostic sample')
            actual=np.column_stack([c['actual'][c1] for c in controls])
            pred=np.empty_like(actual)
            for row,det in enumerate(c1):
                for col,control in enumerate(controls):
                    key=(int(det),col)
                    if key not in cache:cache[key]=float(interp.evaluate(field[det],control['cache'])[0])
                    pred[row,col]=cache[key]
            # Same current kernel and the same fixed reference for both A fields.
            truth_response=k*actual/norm[:,None];interpolated=k*pred/norm[:,None]
            delta=interpolated-truth_response
            sums[:,0]+=truth_response.sum(0);sums[:,1]+=interpolated.sum(0)
            sums[:,2]+=np.square(truth_response).sum(0);sums[:,3]+=np.square(delta).sum(0)
            event_counts.append(dict(dataset=folder.name,view=view,raw_prefix_rows=len(raw),
                prepared=prepared.count,q3_sample=len(c1),input_sha256=digest(path)))
            print(json.dumps(dict(dataset=folder.name,view=view,events=len(c1),elapsed=time.monotonic()-start)),flush=True)
    rows=[]
    for c,s in zip(controls,sums):
        rows.append(dict(part=c['part'],xyz_mm=c['xyz'].tolist(),total_normalized_response=s[0],
            weighted_relative_sum=float(s[1]/s[0]-1) if s[0]>0 else None,
            weighted_relative_l2=float(np.sqrt(s[3]/s[2])) if s[2]>0 else None,
            statistically_valid_transport_efficiency=False))
    result=dict(status='DIAGNOSTICS_COMPLETE_NOT_A_GATE',cases=rows,event_samples=event_counts,
        sample_events=sum(v['q3_sample'] for v in event_counts),selection='First32 rows of seven independent point-source datasets at four fixed views, existing energy/kinematic rules and stable full-circle q<=3',
        field_manifest_sha256=digest(a.field/'field_manifest.json'),geometry_sha256=digest(a.geometry),
        kernel_sha256=digest(Path(__import__('compton_event_response').__file__)),
        limitation='Deterministic bounded sample; point responses, not 3D integrals or independent S validation. No inference about NEMA spikes.',
        reconstruction_permitted=False,new_transport_photons=0,elapsed_seconds=time.monotonic()-start)
    (a.output/'weighted_interpolation.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
