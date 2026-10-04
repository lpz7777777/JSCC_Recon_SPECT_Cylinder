"""Separate PE/scatter contributions to the already observed local A error.

Reuses the 84 predetermined held-out point-source cases and all original
matrix components. No response model, selection, calibration or MLEM changes.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from geometry import grid
from generate_compton_a_guard import digest, ENGINE_REL, SOURCE_RUN, NDET, axes, PE_HASH, SCATTER_HASH
from build_compton_a_guard_field import interpolate_cartesian
from compton_boundary_quadrature import GuardedPolarResponseField, cell_quadrature, rotate_to_detector
from compton_cartesian_patch import CartesianPatch
from compton_event_response import (ComptonEventSettings,prepare_compton_events,
    build_detector_position_variance,build_compton_cone_weights,min_standardized_compton_arm)
from detector_csv import load_detector_coordinates


def component(folder,name,kind):
    part=folder/name;receipt=json.loads((part/'complete.json').read_text())
    matches=[p for p in receipt['matrices'] if p.startswith(kind)]
    if len(matches)!=1:raise ValueError('Component file identity is not unique')
    path=part/matches[0]
    sha=digest(path)
    if sha!=receipt['matrices'][matches[0]]['sha256']:raise ValueError('Component file changed')
    return np.memmap(path,dtype='<f4',mode='r',shape=(NDET,*reversed(receipt['spec']['shape']))),sha


def coarse_component_row(crystal,raw,halos,xy,original_count,scale):
    cart=np.empty((40,87,87),dtype=float)
    cart[:,1:-1,1:-1]=raw[crystal]
    cart[:,:,0]=halos['x_minus'][crystal,:,:,0]
    cart[:,:,-1]=halos['x_plus'][crystal,:,:,0]
    cart[:,0,1:-1]=halos['y_minus'][crystal,:,0,:]
    cart[:,-1,1:-1]=halos['y_plus'][crystal,:,0,:]
    out=np.empty((42,len(xy)),dtype=float)
    out[1:-1]=interpolate_cartesian(cart,xy)*scale
    out[0]=interpolate_cartesian(halos['z_minus'][crystal,0].astype(float),xy)*scale
    out[-1]=interpolate_cartesian(halos['z_plus'][crystal,0].astype(float),xy)*scale
    if not np.isfinite(out).all() or np.any(out<0):raise ValueError('Invalid component field')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','guard','field','patches','cases','geometry','config','inputs','factors','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(8);torch.set_grad_enabled(False);device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    start=time.monotonic();cfg=json.loads(a.config.read_text());geo=np.load(a.geometry)
    coords,cells,_=grid(cfg);np.testing.assert_allclose(coords,geo['coordinates_mm'],rtol=0,atol=1e-10)
    fm=json.loads((a.field/'field_manifest.json').read_text());fg=np.load(a.field/'field_geometry.npz')
    if digest(a.geometry)!=fm['baseline_geometry_sha256']:raise ValueError('Geometry differs')
    if digest(a.field/'field_geometry.npz')!=fm['field_geometry_sha256']:raise ValueError('Field coordinates differ')
    ready=json.loads((a.patches/'patch_ready.json').read_text())
    if ready['pe_binary_sha256']!=PE_HASH or ready['scatter_binary_sha256']!=SCATTER_HASH:
        raise ValueError('Fine matrix model differs')
    source=a.root/ENGINE_REL/SOURCE_RUN;stem='shift_0.000000_0.000000_0.000000'
    input_hashes={};raw={};halos={};fine={}
    for tag,file,kind in [('PE',f'PE_Windowed_SysMat_{stem}_v4.sysmat','pe_windowed.sysmat'),
                          ('Scatter',f'Scatter_SysMat_{stem}.sysmat','Scatter_SysMat')]:
        path=source/file;input_hashes[file]=digest(path)
        raw[tag]=np.memmap(path,dtype='<f4',mode='r',shape=(NDET,40,85,85));halos[tag]={};fine[tag]={}
        for name in ('x_minus','x_plus','y_minus','y_plus','z_minus','z_plus'):
            matrix,h=component(a.guard,name,kind);halos[tag][name]=matrix;input_hashes['halo/'+name+'/'+tag]=h
        for layer,name in ((0,'near_minus'),(20,'near_middle'),(39,'near_plus')):
            matrix,h=component(a.patches,name,kind);fine[tag][layer]=matrix;input_hashes['fine/'+name+'/'+tag]=h
    factor=a.factors/'440keV_RotateNum20'
    detector=torch.tensor(load_detector_coordinates(factor/'Detector.csv',10496),device=device)
    variance=build_detector_position_variance(detector,0)
    settings=ComptonEventSettings(.440,.13*np.sqrt(511/440),2*.440**2/(.511+2*.440)-.001,.05,.35,
        geometry_mode='stable_float64')
    interp=GuardedPolarResponseField(fg['xy_mm'],int(fg['original_points']),fg['z_mm'])
    patch={layer:CartesianPatch(axes(json.loads((a.patches/name/'complete.json').read_text())['spec']))
           for layer,name in ((0,'near_minus'),(20,'near_middle'),(39,'near_plus'))}
    cases=list(csv.DictReader(a.cases.open()))
    if len(cases)!=84:raise ValueError('Predetermined 84-case set differs')
    events={};rows=[];cache={};selected=fg['selected_raw_detectors'];scales=fm['calibration_scales']
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.inputs/'input_manifest.json').read_text())
    for original in cases:
        view,layer,cell=map(int,(original['view'],original['layer'],original['cell']))
        event_key=(original['dataset'],view,int(original['input_row']))
        if event_key not in events:
            path=a.inputs/original['dataset']/f'ideal_v{view:02d}.csv'
            if digest(path)!=manifest['files'][path.relative_to(a.inputs).as_posix()]:raise ValueError('Point input changed')
            event=np.loadtxt(path,delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)[event_key[2]:event_key[2]+1]
            prepared,_=prepare_compton_events(torch.tensor(event,device=device),settings,detector,variance,variance,
                                            input_energies_already_smeared=True)
            if prepared is None or prepared.count!=1:raise ValueError('Predetermined event preparation differs')
            score=min_standardized_compton_arm(prepared,torch.tensor(geo['coordinates_mm'],dtype=torch.float32,device=device),settings)
            if float(score)>3:raise ValueError('Predetermined point event no longer passes R1 q3')
            events[event_key]=prepared
        prepared=events[event_key];c1=int(prepared.cpnum1[0])-1
        if c1+1!=int(original['c1']):raise ValueError('Crystal definition changed')
        nodes,w=cell_quadrature(cells[cell],coords[layer*cfg['points_per_layer']+cell,2],32,12,12,
                                ellipse=original['domain']=='object_intersection')
        nodes=rotate_to_detector(nodes,view-1)
        k=build_compton_cone_weights(prepared,torch.tensor(nodes,dtype=torch.float64,device=device),settings)[0].double().cpu().numpy()
        old_cache=interp.cache(nodes);new_cache=patch[layer].cache(nodes);values={}
        for tag in ('PE','Scatter'):
            key=(tag,c1)
            if key not in cache:
                cache[key]=coarse_component_row(int(selected[c1]),raw[tag],halos[tag],fg['xy_mm'],
                    int(fg['original_points']),scales[c1])
            values['coarse_'+tag]=float(np.dot(k*interp.evaluate(cache[key],old_cache),w))
            values['fine_'+tag]=float(np.dot(k*patch[layer].evaluate(
                fine[tag][layer][selected[c1]],new_cache)*scales[c1],w))
        norm=float(original['reference_norm']);guarded=float(original['fine']);exact=float(original['patch_0p375mm'])
        oldsum=values['coarse_PE']+values['coarse_Scatter'];newsum=values['fine_PE']+values['fine_Scatter']
        olderr=abs(oldsum-guarded)/max(abs(guarded),1e-10*norm)
        newerr=abs(newsum-exact)/max(abs(exact),1e-10*norm)
        passed=abs(oldsum-guarded)<=1e-5*abs(guarded)+1e-10*norm and abs(newsum-exact)<=1e-5*abs(exact)+1e-10*norm
        rows.append(dict(dataset=original['dataset'],view=view,input_row=event_key[2],c1=c1+1,
            layer=layer,cell=cell,domain=original['domain'],**values,original_guarded=guarded,original_fine=exact,
            component_sum_passed=passed,coarse_sum_relative_error=olderr,fine_sum_relative_error=newerr,
            PE_delta_over_common_Z=(values['fine_PE']-values['coarse_PE'])/norm,
            Scatter_delta_over_common_Z=(values['fine_Scatter']-values['coarse_Scatter'])/norm,
            original_delta_over_common_Z=(exact-guarded)/norm))
        if len(rows)%12==0:print(json.dumps(dict(cases=len(rows),elapsed=time.monotonic()-start)),flush=True)
    with (a.output/'component_cases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    pe=sum(abs(r['PE_delta_over_common_Z']) for r in rows)
    scatter=sum(abs(r['Scatter_delta_over_common_Z']) for r in rows)
    passed=all(r['component_sum_passed'] for r in rows)
    result=dict(status='PASSED_DIAGNOSTIC_ONLY' if passed else 'HOLD_COMPONENT_ADDITIVITY',cases=len(rows),
        component_sums_passed=sum(r['component_sum_passed'] for r in rows),
        maximum_coarse_sum_relative_error=max(r['coarse_sum_relative_error'] for r in rows),
        maximum_fine_sum_relative_error=max(r['fine_sum_relative_error'] for r in rows),
        aggregate_absolute_PE_delta_over_common_Z=pe,aggregate_absolute_Scatter_delta_over_common_Z=scatter,
        PE_share_of_component_absolute_changes=pe/max(pe+scatter,1e-30),
        inputs_sha256=input_hashes,cases_sha256=digest(a.cases),output_sha256=digest(a.output/'component_cases.csv'),
        geometry_sha256=digest(a.geometry),field_manifest_sha256=digest(a.field/'field_manifest.json'),
        patch_ready_sha256=digest(a.patches/'patch_ready.json'),
        kernel_sha256=digest(Path(__import__('compton_event_response').__file__)),
        code_sha256=digest(Path(__file__)),elapsed_seconds=time.monotonic()-start,
        new_transport_photons=0,new_matrix_points=0,reconstruction_permitted=False,S2_generated=False,
        interpretation='PE/scatter error decomposition on the same 84 local point cases only; '
                       'not a global FOV accuracy certificate or permission to omit physical scatter sources.')
    (a.output/'component_gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='inputs_sha256'},indent=2),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
