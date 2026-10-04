"""Bounded anisotropic A interpolation study using existing physical matrices.

Subsample the same .375 mm PE-v4 plus full-scatter Cartesian boxes. No new
transport, matrix calculation, event selection or imaging response is enabled.
Results describe only the frozen 84 independent point-source cases.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from geometry import grid
from generate_compton_a_guard import digest,axes,PE_HASH,SCATTER_HASH
from build_compton_a_guard_field import combined
from compton_cartesian_patch import CartesianPatch
from compton_boundary_quadrature import cell_quadrature,rotate_to_detector
from compton_event_response import (ComptonEventSettings,prepare_compton_events,
    build_detector_position_variance,build_compton_cone_weights,min_standardized_compton_arm)
from detector_csv import load_detector_coordinates

# Fixed nested axes keep the physical support and its end faces unchanged.
CANDIDATES={
    'xy0375_z075':(1,1,2),
    'xy0375_z150':(1,1,4),
    'xy0375_z300':(1,1,8),
    'x075_y0375_z0375':(2,1,1),
    'x0375_y075_z0375':(1,2,1),
    'xy075_z0375':(2,2,1),
    'xy075_z150':(2,2,4),
    'xy150_z0375':(4,4,1),
}


def subset_axes(xyz,stride):
    indices=[]
    for axis,step in zip(xyz,stride):
        if (len(axis)-1)%step:raise ValueError('Nested sampling does not preserve both faces')
        indices.append(np.arange(0,len(axis),step))
    return [v[i] for v,i in zip(xyz,indices)],indices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('field','patches','cases','geometry','config','inputs','factors','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(8);torch.set_grad_enabled(False);device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    start=time.monotonic();cfg=json.loads(a.config.read_text());geo=np.load(a.geometry)
    coords,cells,_=grid(cfg);np.testing.assert_allclose(coords,geo['coordinates_mm'],rtol=0,atol=1e-10)
    fm=json.loads((a.field/'field_manifest.json').read_text());fg=np.load(a.field/'field_geometry.npz')
    if digest(a.geometry)!=fm['baseline_geometry_sha256'] or digest(a.field/'field_geometry.npz')!=fm['field_geometry_sha256']:
        raise ValueError('Frozen field geometry differs')
    ready=json.loads((a.patches/'patch_ready.json').read_text())
    if ready['pe_binary_sha256']!=PE_HASH or ready['scatter_binary_sha256']!=SCATTER_HASH:
        raise ValueError('Physical A model differs')
    boxes={}
    for layer,name in ((0,'near_minus'),(20,'near_middle'),(39,'near_plus')):
        receipt=json.loads((a.patches/name/'complete.json').read_text())
        xyz=axes(receipt['spec'])
        boxes[layer]=dict(xyz=xyz,values=combined(a.patches,name),interpolator=CartesianPatch(xyz),subsets={})
        for tag,stride in CANDIDATES.items():
            sub,indices=subset_axes(xyz,stride)
            boxes[layer]['subsets'][tag]=(CartesianPatch(sub),indices)
    cases=list(csv.DictReader(a.cases.open()))
    if len(cases)!=84:raise ValueError('Frozen point case budget differs')
    factor=a.factors/'440keV_RotateNum20'
    detector=torch.tensor(load_detector_coordinates(factor/'Detector.csv',10496),device=device)
    variance=build_detector_position_variance(detector,0)
    settings=ComptonEventSettings(.440,.13*np.sqrt(511/440),2*.440**2/(.511+2*.440)-.001,.05,.35,
                                  geometry_mode='stable_float64')
    manifest=json.loads((a.inputs/'input_manifest.json').read_text())
    events={};rows=[];cache={};a.output.mkdir(parents=True,exist_ok=False)
    for original in cases:
        view,layer,cell=map(int,(original['view'],original['layer'],original['cell']))
        identity=(original['dataset'],view,int(original['input_row']))
        if identity not in events:
            path=a.inputs/original['dataset']/f'ideal_v{view:02d}.csv'
            if digest(path)!=manifest['files'][path.relative_to(a.inputs).as_posix()]:raise ValueError('Point input changed')
            raw=np.loadtxt(path,delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)[identity[2]:identity[2]+1]
            prepared,_=prepare_compton_events(torch.tensor(raw,device=device),settings,detector,variance,variance,
                                            input_energies_already_smeared=True)
            if prepared is None or prepared.count!=1:raise ValueError('Frozen event preparation differs')
            q=min_standardized_compton_arm(prepared,torch.tensor(coords,dtype=torch.float32,device=device),settings)
            if float(q)>3:raise ValueError('Frozen event q3 acceptance differs')
            events[identity]=prepared
        prepared=events[identity];c1=int(prepared.cpnum1[0])-1
        if c1+1!=int(original['c1']):raise ValueError('First crystal differs')
        nodes,w=cell_quadrature(cells[cell],coords[layer*cfg['points_per_layer']+cell,2],32,12,12,
                                ellipse=original['domain']=='object_intersection')
        nodes=rotate_to_detector(nodes,view-1);box=boxes[layer]
        k=build_compton_cone_weights(prepared,torch.tensor(nodes,dtype=torch.float64,device=device),settings)[0].double().cpu().numpy()
        full=box['values'][fg['selected_raw_detectors'][c1]]
        scale=fm['calibration_scales'][c1]
        baseline=float(np.dot(w*k,box['interpolator'].evaluate(full,box['interpolator'].cache(nodes))*scale))
        norm=float(original['reference_norm']);expected=float(original['patch_0p375mm'])
        if abs(baseline-expected)>1e-7*abs(expected)+1e-12*norm:
            raise ValueError('Full-resolution interpolation does not reproduce frozen evidence')
        for tag,(interpolator,indices) in box['subsets'].items():
            key=(layer,c1,tag)
            if key not in cache:
                ix,iy,iz=indices;cache[key]=np.array(full[np.ix_(iz,iy,ix)],copy=True)
            value=float(np.dot(w*k,interpolator.evaluate(cache[key],interpolator.cache(nodes))*scale))
            delta=abs(value-baseline);relative=delta/max(abs(baseline),1e-10*norm)
            rows.append(dict(dataset=identity[0],view=view,input_row=identity[2],c1=c1+1,layer=layer,cell=cell,
                domain=original['domain'],sampling=tag,baseline=baseline,subsampled=value,
                relative_change=relative,absolute_change_over_common_Z=delta/norm,
                passed=delta<=.01*abs(baseline)+1e-10*norm))
        if len(rows)%96==0:print(json.dumps(dict(point_cases=len(rows)//len(CANDIDATES),elapsed=time.monotonic()-start)),flush=True)
    with (a.output/'sampling_cases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={}
    for tag,stride in CANDIDATES.items():
        subset=[r for r in rows if r['sampling']==tag]
        summary[tag]=dict(spacing_mm=(.375*np.array(stride)).tolist(),cases=len(subset),
            passed=sum(r['passed'] for r in subset),maximum_relative_change=max(r['relative_change'] for r in subset),
            maximum_absolute_change_over_common_Z=max(r['absolute_change_over_common_Z'] for r in subset),
            cases_over_one_percent=sum(not r['passed'] for r in subset))
    result=dict(status='SAMPLING_DIAGNOSTIC_COMPLETE_GLOBAL_ACCURACY_HOLD',candidates=summary,
        baseline_cases=len(cases),output_sha256=digest(a.output/'sampling_cases.csv'),cases_sha256=digest(a.cases),
        field_manifest_sha256=digest(a.field/'field_manifest.json'),patch_ready_sha256=digest(a.patches/'patch_ready.json'),
        geometry_sha256=digest(a.geometry),code_sha256=digest(Path(__file__)),
        kernel_sha256=digest(Path(__import__('compton_event_response').__file__)),elapsed_seconds=time.monotonic()-start,
        new_transport_photons=0,new_matrix_points=0,reconstruction_permitted=False,S2_generated=False,
        interpretation='Anisotropic A subsampling on the same local 84 point cases; preserved PE plus full physical scatter, '
                       'all box faces and frozen calibration. Not a global accuracy certificate.')
    (a.output/'sampling_gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
