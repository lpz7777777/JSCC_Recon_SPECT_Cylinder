"""Bounded R2 feasibility gate: geometry, held-out kernels and A interpolation.

Does not build or publish a production response. Failures produce HOLD with
the required matrix/support remedy, rather than silently extrapolating A.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from geometry import grid
from compton_boundary_quadrature import cell_quadrature,rotate_to_detector,PolarResponseField
from compton_event_response import ComptonEventSettings,build_detector_position_variance,prepare_compton_events,build_compton_cone_weights,min_standardized_compton_arm
from detector_csv import load_detector_coordinates

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('geometry','grid-config','study-config','inputs','factors','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    study=json.loads(a.study_config.read_text());cfg=json.loads(a.grid_config.read_text())
    if sha(a.geometry)!=study['geometry_sha256']:raise ValueError('Frozen geometry differs')
    a.output.mkdir(parents=True,exist_ok=False)
    geo=np.load(a.geometry);coords,cells,_=grid(cfg);n=cfg['points_per_layer']
    f=geo['ellipse_fraction'][:n];volumes=geo['cell_volume_mm3'];partial=np.flatnonzero((f>1e-12)&(f<1-1e-12))
    if len(partial)*40!=study['partial_cells']:raise ValueError('Partial cell definition differs')
    interpolator=PolarResponseField(coords[:n,:2]);volume_rows=[];transverse=[]
    for cell in partial:
        q,w=cell_quadrature(cells[cell],1.5,16,4,4)
        q2,w2=cell_quadrature(cells[cell],1.5,32,4,4)
        reference=volumes[cell]*f[cell]
        volume_rows.append(dict(cell=int(cell),fraction=float(f[cell]),frozen_volume=float(reference),
            integral_volume=float(w2.sum()),relative_error=float(w2.sum()/reference-1),
            refinement_error=float(w2.sum()/w.sum()-1)))
        full,fw=cell_quadrature(cells[cell],1.5,8,4,4,ellipse=False)
        missing=interpolator.triangulation.find_simplex(full[:,:2])<0
        if missing.any():transverse.append(dict(cell=int(cell),fraction=float(f[cell]),
            full_cell_weight_outside_response_hull=float(fw[missing].sum()/fw.sum()),
            max_radius_mm=float(np.hypot(full[missing,0],full[missing,1]).max())))
    # Predefined cells: fraction quantiles plus eight object-frame directions.
    ordered=partial[np.argsort(f[partial])]
    controls=set(ordered[np.linspace(0,len(ordered)-1,8,dtype=int)].tolist())
    angle=np.arctan2(coords[partial,1],coords[partial,0])
    for target in np.arange(8)*np.pi/4:
        controls.add(int(partial[np.argmin(np.abs(np.angle(np.exp(1j*(angle-target)))))]))
    controls=sorted(controls);torch.set_num_threads(8);torch.set_grad_enabled(False)
    device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    factor=a.factors/'440keV_RotateNum20';detector=torch.tensor(load_detector_coordinates(factor/'Detector.csv',10496),device=device)
    var=build_detector_position_variance(detector,0)
    settings=ComptonEventSettings(.440,.13*np.sqrt(511/440),2*.440**2/(.511+2*.440)-.001,.05,.35,geometry_mode='stable_float64')
    memory=np.memmap(factor/'SysMat_polar',dtype='<f4',mode='r',shape=(132040,10496))
    results=[];start=time.monotonic();allcoords=torch.tensor(coords,dtype=torch.float32,device=device)
    for folder in sorted(a.inputs.glob('point_*')):
        for view in (1,6,11,16):
            file=folder/f'ideal_v{view:02d}.csv'
            if not file.stat().st_size:continue
            raw=np.loadtxt(file,delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
            # The first stable-q eligible event is deterministic and independent of NEMA images.
            prepared=None;row=None
            for index in range(min(len(raw),32)):
                candidate,_=prepare_compton_events(torch.tensor(raw[index:index+1],device=device),settings,detector,var,var,input_energies_already_smeared=True)
                if candidate is not None and float(min_standardized_compton_arm(candidate,allcoords,settings))<=3:
                    prepared=candidate;row=index;break
            if prepared is None:continue
            c1=int(prepared.cpnum1[0])-1
            b=np.asarray(memory[:,c1],dtype=np.float64);field=(b/volumes).reshape(40,n)
            norm=float((build_compton_cone_weights(prepared,allcoords,settings)[0].double()*torch.tensor(b,device=device)).sum())
            if not norm>0:raise ValueError('Control event has zero reference response')
            for layer in (0,20,39):
                for cell in controls:
                    values=[];endpoint=0.;centroid=0.;point=0.;overlap=0.
                    for theta,radial,axial in ((8,4,4),(16,8,8),(32,12,12)):
                        nodes,w=cell_quadrature(cells[cell],coords[layer*n+cell,2],theta,radial,axial)
                        detector_nodes=rotate_to_detector(nodes,view-1)
                        cache=interpolator.cache(detector_nodes)
                        spatial=interpolator.evaluate(field,cache)
                        k=build_compton_cone_weights(prepared,torch.tensor(detector_nodes,dtype=torch.float32,device=device),settings)[0].double().cpu().numpy()
                        value=float(np.dot(k*spatial,w));values.append(value);overlap=float(w.sum())
                        if theta==32:
                            clamped=interpolator.evaluate(field,interpolator.cache(detector_nodes,'clamped_endpoint'))
                            endpoint=float(np.dot(k*clamped,w))
                            for name,location in (('centroid',np.sum(nodes*w[:,None],axis=0)/overlap),('point',coords[layer*n+cell])):
                                location=rotate_to_detector(location[None],view-1)
                                response=float(build_compton_cone_weights(prepared,torch.tensor(location,dtype=torch.float32,device=device),settings)[0,0])
                                val=response*float(interpolator.evaluate(field,interpolator.cache(location))[0])*overlap
                                if name=='centroid':centroid=val
                                else:point=val
                    floor=1e-10*norm
                    converged=abs(values[-1]-values[-2])<=.01*abs(values[-1])+floor
                    endpoint_ok=abs(endpoint-values[-1])<=.01*abs(values[-1])+floor
                    results.append(dict(dataset=folder.name,view=view,input_row=row,input_sha256=sha(file),
                        c1=c1+1,c2=int(prepared.cpnum2[0]),layer=layer,cell=cell,fraction=float(f[cell]),
                        overlap_volume=overlap,representative=point,centroid=centroid,
                        coarse=values[0],medium=values[1],fine=values[2],clamped_endpoint=endpoint,
                        reference_norm=norm,converged=converged,endpoint_passed=endpoint_ok))
            print(json.dumps(dict(dataset=folder.name,view=view,cases=len(results),elapsed_seconds=time.monotonic()-start)),flush=True)
            with (a.output/'response_cases.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    volume_ok=all(abs(r['relative_error'])<=.001 for r in volume_rows)
    gate=dict(study=study['study'],status='PASSED' if results and volume_ok and not transverse and all(r['converged'] and r['endpoint_passed'] for r in results) else 'HOLD',
        geometry_sha256=sha(a.geometry),study_config_sha256=sha(a.study_config),kernel_sha256=sha(Path(__import__('compton_event_response').__file__)),
        partial_cells_per_layer=len(partial),volume_passed=volume_ok,volume_rows=volume_rows,
        transverse_reference_supported=not transverse,transverse_missing=transverse,
        predefined_cells=controls,response_cases=len(results),converged=sum(r['converged'] for r in results),
        endpoint_passed=sum(r['endpoint_passed'] for r in results),
        explanation='HOLD forbids R2 imaging. Any transverse gap requires original Cartesian response interpolation or matrix support extension and a new validation, not silent radial extrapolation.',
        elapsed_seconds=time.monotonic()-start,new_transport_photons=0)
    (a.output/'boundary_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    print('BOUNDARY_GATE',gate['status'],flush=True)

if __name__=='__main__':main()
