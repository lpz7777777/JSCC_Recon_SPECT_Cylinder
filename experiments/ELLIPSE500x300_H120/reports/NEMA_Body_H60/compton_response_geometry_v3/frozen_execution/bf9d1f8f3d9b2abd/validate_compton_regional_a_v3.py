"""Held-out K-weighted object/full-reference regional A precision diagnostic.

Common-point production, quadrature convergence and physical-grid precision
are separate checks. A local pass does not certify global A or permit imaging.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from geometry import grid
from generate_compton_a_guard import digest,write,axes,NDET,PE_HASH,SCATTER_HASH
from compton_cartesian_patch import CartesianPatch
from compton_boundary_quadrature import cell_quadrature,rotate_to_detector
from detector_csv import load_detector_coordinates
from compton_event_response import (ComptonEventSettings,prepare_compton_events,
    build_detector_position_variance,build_compton_cone_weights,min_standardized_compton_arm)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('plan','samples','field','geometry','config','measure','inputs','factors','scan','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();start=time.monotonic()
    torch.set_num_threads(8);torch.set_grad_enabled(False);device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    plan=json.loads(a.plan.read_text());geometry=np.load(a.geometry);cfg=json.loads(a.config.read_text())
    coords,cells,_=grid(cfg);np.testing.assert_allclose(coords,geometry['coordinates_mm'],atol=1e-10,rtol=0)
    fm=json.loads((a.field/'field_manifest.json').read_text());fg=a.field/'field_geometry.npz'
    gate=json.loads((a.scan/'validation_gate.json').read_text());kernel=Path(__import__('compton_event_response').__file__)
    mm=json.loads((a.measure/'measure_manifest.json').read_text());measure=np.load(a.measure/'measure.npz')
    if (gate['status']!='PASSED' or gate['geometry_sha256']!=digest(a.geometry)
            or gate['kernel_sha256']!=digest(kernel) or plan['geometry_sha256']!=digest(a.geometry)
            or fm['baseline_geometry_sha256']!=digest(a.geometry) or digest(fg)!=fm['field_geometry_sha256']
            or mm['status']!='PASSED' or digest(a.measure/'measure.npz')!=mm['measure_sha256']
            or plan['measure_npz_sha256']!=mm['measure_sha256']):
        raise ValueError('Frozen regional response/measure/kernel identity differs')
    selected=np.load(fg)['selected_raw_detectors'];scales=np.asarray(fm['calibration_scales'])
    factor=a.factors/'440keV_RotateNum20'
    if digest(factor/'SysMat_polar')!=fm['baseline_B_sha256']:raise ValueError('Frozen B differs')
    B=np.memmap(factor/'SysMat_polar',mode='r',dtype='<f4',shape=(132040,10496))
    detector=torch.tensor(load_detector_coordinates(factor/'Detector.csv',10496),device=device)
    variance=build_detector_position_variance(detector,0)
    settings=ComptonEventSettings(.440,.13*np.sqrt(511/440),2*.440**2/(.511+2*.440)-.001,
        .05,.35,geometry_mode='stable_float64')
    allcoords=torch.tensor(geometry['coordinates_mm'],device=device,dtype=torch.float32)
    manifest=json.loads((a.inputs/'input_manifest.json').read_text());prepared_cache={};event_records={};missing=[]
    def events(view):
        if view in prepared_cache:return prepared_cache[view]
        raw_rows=[];identities=[]
        for folder in sorted(a.inputs.glob('point_*')):
            path=folder/f'ideal_v{view:02d}.csv';sha=digest(path)
            if manifest['files'][path.relative_to(a.inputs).as_posix()]!=sha:raise ValueError('Point List differs')
            raw=np.loadtxt(path,delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
            for row in range(min(len(raw),32)):
                candidate,_=prepare_compton_events(torch.tensor(raw[row:row+1],device=device),settings,
                    detector,variance,variance,input_energies_already_smeared=True)
                if candidate is not None and float(min_standardized_compton_arm(candidate,allcoords,settings))<=3:
                    raw_rows.append(raw[row]);identities.append(dict(dataset=folder.name,view=view,
                        input_row=row,input_sha256=sha));break
            else:missing.append(dict(dataset=folder.name,view=view,reason='No stable-q eligible event in first32'))
        if not raw_rows:raise ValueError('No predetermined point events; do not change selection')
        prepared,_=prepare_compton_events(torch.tensor(np.array(raw_rows),device=device),settings,
            detector,variance,variance,input_energies_already_smeared=True)
        if prepared.count!=len(identities):raise ValueError('Diagnostic event identity lost')
        crystals=prepared.cpnum1.cpu().numpy()-1
        kb=build_compton_cone_weights(prepared,allcoords,settings)*torch.tensor(np.array(B[:,crystals].T,copy=True),device=device)
        Z=kb.double().sum(1).cpu().numpy()
        if np.any(Z<=0) or not np.isfinite(Z).all():raise ValueError('Common complete-circle Z invalid')
        prepared_cache[view]=(prepared,crystals,Z);event_records[str(view)]=identities
        return prepared_cache[view]
    production={};source_gates={}
    for group in range(3):
        folder=a.samples/f'regional_a_g{group}';path=folder/'regional_ready.json';r=json.loads(path.read_text())
        if (r['status']!='REGIONAL_PHYSICAL_COMMON_POINTS_PASSED_ACCURACY_PENDING'
                or r['plan_sha256']!=digest(a.plan) or r['pe_binary_sha256']!=PE_HASH
                or r['scatter_binary_sha256']!=SCATTER_HASH):raise ValueError('Regional physical production gate differs')
        source_gates[str(group)]=digest(path)
        for entry in r['results']:production[entry['case']['name']]=(folder,entry)
    a.output.mkdir(parents=True,exist_ok=False);rows=[];volume_error=[]
    for case in plan['cases']:
        folder,entry=production[case['name']]
        if entry['case']!=case:raise ValueError('Frozen regional control changed')
        prepared,crystals,Z=events(case['view']);fields=[]
        for receipt in entry['receipts']:
            name=next(n for n in receipt['matrices'] if n.startswith('SysMat_withScatter'))
            path=folder/receipt['spec']['name']/name
            if digest(path)!=receipt['matrices'][name]['sha256']:raise ValueError('Immutable regional A changed')
            fields.append((np.memmap(path,mode='r',dtype='<f4',shape=(NDET,*reversed(receipt['spec']['shape']))),
                CartesianPatch(axes(receipt['spec']))))
        for domain in ('object_intersection','full_reference_cell'):
            results=[]
            for order in ((32,12,12),(48,16,16)):
                points,weights=cell_quadrature(cells[case['object_xy_index']],case['z_mm'],*order,
                    ellipse=domain=='object_intersection')
                physical=rotate_to_detector(points,case['view']-1)
                i=case['layer']*3301+case['object_xy_index']
                V=measure['effective_volume_mm3'][i] if domain=='object_intersection' else geometry['cell_volume_mm3'][i]
                volume_error.append(abs(float(weights.sum()/V-1)))
                K=build_compton_cone_weights(prepared,torch.tensor(physical,device=device,dtype=torch.float64),settings).double().cpu().numpy()
                values=[]
                for raw,interp in fields:
                    cache=interp.cache(physical)
                    values.append(np.array([np.dot(K[e]*interp.evaluate(raw[selected[c]],cache)*scales[c],weights)
                        for e,c in enumerate(crystals)]))
                results.append(np.stack(values,axis=1))
            coarse,fine=results[-1].T;delta=abs(fine-coarse);floor=1e-10*Z
            for e,identity in enumerate(event_records[str(case['view'])]):
                qdiff=np.abs(results[1][e]-results[0][e])
                rows.append(dict(control=case['name'],domain=domain,**identity,
                    c1=int(prepared.cpnum1[e]),c2=int(prepared.cpnum2[e]),common_R1_Z=float(Z[e]),
                    coarse_A_integral=float(coarse[e]),fine_A_integral=float(fine[e]),
                    A_relative_change=float(delta[e]/max(abs(fine[e]),floor[e])),
                    A_absolute_change_over_Z=float(delta[e]/Z[e]),
                    A_passed=bool(delta[e]<=.01*abs(fine[e])+floor[e]),
                    informative=bool(abs(fine[e])>floor[e]),
                    quadrature_relative_change=float(np.max(qdiff/np.maximum(abs(results[1][e]),floor[e]))),
                    quadrature_passed=bool(np.all(qdiff<=.01*abs(results[1][e])+floor[e]))))
        with (a.output/'regional_cases.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print(json.dumps(dict(completed_control=case['name'],entries=len(rows),elapsed_seconds=time.monotonic()-start)),flush=True)
    passed=all(r['A_passed'] and r['quadrature_passed'] for r in rows) and max(volume_error)<=.001 and not missing
    result=dict(status='REGIONAL_REFINEMENT_PASSED_LOCAL_ONLY' if passed else 'HOLD_REGIONAL_REFINEMENT',
        entries=len(rows),A_entries_passed=sum(r['A_passed'] for r in rows),
        quadrature_entries_passed=sum(r['quadrature_passed'] for r in rows),informative_entries=sum(r['informative'] for r in rows),
        maximum_A_relative_change=max(r['A_relative_change'] for r in rows),
        maximum_A_absolute_change_over_Z=max(r['A_absolute_change_over_Z'] for r in rows),
        maximum_quadrature_relative_change=max(r['quadrature_relative_change'] for r in rows),
        maximum_volume_relative_error=max(volume_error),missing_events=missing,events=event_records,
        by_control={c['name']:dict(entries=sum(r['control']==c['name'] for r in rows),
            informative=sum(r['control']==c['name'] and r['informative'] for r in rows),
            passed=sum(r['control']==c['name'] and r['A_passed'] and r['quadrature_passed'] for r in rows)) for c in plan['cases']},
        plan_sha256=digest(a.plan),production_gates_sha256=source_gates,kernel_sha256=digest(kernel),
        geometry_sha256=digest(a.geometry),field_manifest_sha256=digest(a.field/'field_manifest.json'),
        csv_sha256=digest(a.output/'regional_cases.csv'),source_sha256=digest(Path(__file__)),elapsed_seconds=time.monotonic()-start,
        new_transport_photons=0,reconstruction_permitted=False,S2_generated=False,
        interpretation='Same predefined independent point events, stable K and common Z; full-reference and true object dV. '
            'Near-zero entries are explicitly marked. Local convergence is not global physical A validation or spike reduction.')
    write(a.output/'regional_validation.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('events','by_control')},indent=2))


if __name__=='__main__':main()
