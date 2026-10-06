"""Matched legacy-policy sensitivities for the isolated existing 5e9 comparison.

Uses the frozen v5 response law unchanged. Tests its transfer to legacy events,
with independent circle, ellipse, spatial and point-category validation.
"""
import argparse
import json
import math
from pathlib import Path
try:
    import resource
except ImportError:
    resource=None
import time
import numpy as np
import torch
from geometry import grid
from detector_csv import load_detector_coordinates
from compton_event_response import (prepare_compton_events,build_detector_position_variance,
    build_compton_cone_weights)
from process_list_global_audit_v4 import digest,write,table,vector,settings,fine_labels,source_matrix,map_source_cell
from prepare_energy_5e9_v5 import STUDY,metadata,selected_rows
from compton_energy_probability_v5 import ContinuousTransferLaw,normalized_proxy_response

def prepare(raw,detector,variance,device):
    p,_=prepare_compton_events(torch.tensor(raw,device=device),settings(),detector,
        variance,variance,input_energies_already_smeared=True)
    if p is None or p.count!=len(raw):raise ValueError('Frozen event fails original preparation')
    return p


def category(raw,detector):
    l1=np.rint((abs(detector[raw[:,0].astype(int)-1,1])-300)/30).astype(int)
    l2=np.rint((abs(detector[raw[:,2].astype(int)-1,1])-300)/30).astype(int)
    if np.any(l1==l2):raise ValueError('Same-layer frozen event')
    pair=l1*3+l2-(l2>l1)
    energy=np.searchsorted([.125,.20],raw[:,1],side='right')
    second=(raw[:,3]>=.25).astype(int)
    return (pair*3+energy)*2+second


def source_contexts(a,detector,coords,slices):
    contexts=[];lookup={}
    for point in range(7):
        for view in range(1,21):
            folder=a.inputs/f'point_{point}'
            first=next(metadata(folder,view))
            source=vector(first,'source_')+np.array([0,345,0])
            cell=int(map_source_cell(source[None],slices)[0])
            contexts.append(dict(point=point,view=view,source=source,cell=cell))
            lookup[(point,view)]=contexts[-1]
    return contexts,lookup


def spatial(a,d,var,law,coords,slices,geo):
    device=a.device;vol=geo['cell_volume_mm3'];V=float(vol.sum());labels=fine_labels(coords)
    masks=np.eye(192)[labels]
    averaged=sum(masks[geo['inverse_rotation'][:,v]] for v in range(20))/20
    mask=torch.tensor(averaged,device=device,dtype=torch.float32)
    co=torch.tensor(coords,device=device,dtype=torch.float32)
    matrix=np.memmap(a.factors/'440keV_RotateNum20/SysMat_polar',dtype='<f4',mode='r',shape=(132040,10496))
    B=torch.tensor(np.array(matrix.T,copy=True),device=device);del matrix
    folder=a.inputs/'circle_train';indices=selected_rows(a,'circle_train',1)
    meta={int(r['global_legacy_row']):r for r in metadata(folder,1,set(map(int,indices)))}
    raw=np.loadtxt(folder/'legacy_v01.csv',delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
    contexts,_=source_contexts(a,d.cpu().numpy(),coords,slices)
    cells=np.array([r['cell'] for r in contexts]);point_volume=vol[cells]
    workers={name:np.zeros((200,192)) for name in ('angular','continuous_energy')}
    spectral={name:np.zeros((200,140,72)) for name in workers}
    sums={name:torch.zeros(132040,device=device,dtype=torch.float64) for name in workers}
    limit=min(len(indices),a.probe_events) if a.probe_events else len(indices)
    ellipse_workers={name:np.zeros(200) for name in workers}
    ellipse_volume=float((vol*geo['ellipse_fraction']).sum())
    ellipse_mask=torch.tensor(geo['ellipse_fraction'][geo['inverse_rotation']].mean(1),device=device,dtype=torch.float32)
    warmup=0.
    if a.compiled_bins or a.energy_backend!='analytic':
        if not device.startswith('cuda'):raise ValueError('Fusion probe requires CUDA')
        begin=time.monotonic();p=prepare(raw[indices[:a.batch]],d,var,device)
        eager=normalized_proxy_response(p,co,B,law,node_chunk=a.node_chunk)
        fused=normalized_proxy_response(p,co,B,law,node_chunk=a.node_chunk,compiled_bins=a.compiled_bins,backend=a.energy_backend,gaussian_float32=a.gaussian_float32)
        relative=float(torch.linalg.vector_norm(fused-eager)/torch.linalg.vector_norm(eager))
        variation=float((fused-eager).abs().sum(1).max()/2)
        threshold=1e-5 if a.compiled_bins else .001
        if relative>threshold or variation>threshold:raise ValueError('Numerical response disagrees with analytic reference')
        torch.cuda.synchronize();warmup=time.monotonic()-begin
        write(a.output/'fusion_gate.json',dict(response_relative_L2=relative,max_total_variation=variation,threshold=threshold,backend=a.energy_backend,passed=True,
            events=p.count,full_grid_points=132040,warmup_seconds=warmup,geometry_dtype='float64',
            gaussian_evaluation_dtype='float32' if a.gaussian_float32 else 'float64',new_model_parameters=False))
        del eager,fused,p
    start=time.monotonic()
    for offset in range(0,limit,a.batch):
        if device.startswith('cuda') and torch.cuda.memory_reserved()>.60*torch.cuda.get_device_properties(device).total_memory:
            torch.cuda.empty_cache()
        rows=indices[offset:min(offset+a.batch,limit)]
        p=prepare(raw[rows],d,var,device)
        old=build_compton_cone_weights(p,co,settings())*B[p.cpnum1-1]
        old=old/old.sum(1,keepdim=True)
        new=normalized_proxy_response(p,co,B,law,node_chunk=a.node_chunk,compiled_bins=a.compiled_bins,backend=a.energy_backend,gaussian_float32=a.gaussian_float32)
        ids=np.array([int(meta[int(i)]['worker']) for i in rows]);cat=category(raw[rows],d.cpu().numpy())
        for name,response in (('angular',old),('continuous_energy',new)):
            if not bool(torch.isfinite(response).all()) or bool((response<0).any()) or not bool(torch.allclose(response.sum(1),torch.ones(len(rows),device=device),atol=1e-5,rtol=0)):
                raise ValueError('Invalid normalized calibration response')
            sums[name]+=response.double().sum(0)
            np.add.at(ellipse_workers[name],ids,(response@ellipse_mask).double().cpu().numpy())
            np.add.at(workers[name],ids,(response@mask).double().cpu().numpy())
            point=response[:,cells].double().cpu().numpy()/point_volume[None]
            for e,worker in enumerate(ids):spectral[name][worker,:,cat[e]]+=point[e]
        if offset%512==0:print('CANDIDATE_TRAIN_EVENTS',offset,limit,'seconds',round(time.monotonic()-start,2),flush=True)
        del p,old,new,response
        if device.startswith('cuda') and torch.cuda.max_memory_reserved()>.75*torch.cuda.get_device_properties(device).total_memory:
            write(a.output/'resource_hold.json',dict(reason='Runtime reserved peak exceeds proactive 75% limit',
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_total_bytes=torch.cuda.get_device_properties(device).total_memory,processed_events=min(offset+a.batch,limit)))
            raise RuntimeError('Runtime GPU memory gate; partial S rejected')
    if a.probe_events:
        write(a.output/'resource_probe.json',dict(events=limit,total_events=len(indices),
            elapsed_seconds=time.monotonic()-start,estimated_full_seconds=(time.monotonic()-start)*len(indices)/limit,
            compile_warmup_seconds=warmup,
            scientific_validation=False))
        return
    N=int(json.loads((folder/'collection.json').read_text())['primary_counts'][1])
    sensi={}
    for name in sums:
        unrot=sums[name].cpu().numpy()*V/N
        averaged_s=unrot[geo['inverse_rotation']].mean(1)
        sensi[name]=averaged_s
        averaged_s.astype('<f4').tofile(a.output/(name+'_Sensi_full'))
        unrot.astype('<f4').tofile(a.output/(name+'_Sensi_detector'))
    validation=a.inputs/'circle_validation';selected=selected_rows(a,'circle_validation',1)
    vm={int(r['global_legacy_row']):r for r in metadata(validation,1,set(map(int,selected)))}
    observed=source_matrix([vm[int(i)] for i in selected],labels,slices)
    ow=np.zeros((200,192));np.add.at(ow,np.array([int(vm[int(i)]['worker']) for i in selected]),observed)
    Nv=int(json.loads((validation/'collection.json').read_text())['primary_counts'][1])
    volumes=np.bincount(labels,weights=vol,minlength=192);reports=[]
    for name in workers:
        wt=workers[name]*V/(N/200*volumes);wv=ow*V/(Nv/200*volumes)
        pred=wt.mean(0);actual=wv.mean(0);se=np.hypot(wt.std(0,ddof=1),wv.std(0,ddof=1))/np.sqrt(200)
        for j in range(192):
            adequate=float(ow[:,j].sum())>=400;bias=float(pred[j]/actual[j]-1) if actual[j]>0 else None
            uncertainty=float(se[j]/actual[j]) if actual[j]>0 else None
            hold=bool(adequate and abs(bias)>.2 and abs(bias)>3*uncertainty)
            reports.append(dict(model=name,bin=j,volume_mm3=float(volumes[j]),
                accepted_mass=float(ow[:,j].sum()),adequate=adequate,predicted_efficiency=float(pred[j]),
                observed_efficiency=float(actual[j]),relative_bias=bias,relative_se=uncertainty,hold=hold))
    table(a.output/'spatial_efficiency.csv',reports)
    validate_joint(a,d.cpu().numpy(),vol,V,N,contexts,spectral)
    ellipse_accepted=sum(len(selected_rows(a,'ellipse_validation',v)) for v in range(1,21))
    Ne=int(json.loads((a.inputs/'ellipse_validation/collection.json').read_text())['primary_counts'][1])
    ellipse={name:dict(observed_efficiency=ellipse_accepted/Ne,
        predicted_efficiency=float((s*geo['ellipse_fraction']).sum()/(vol*geo['ellipse_fraction']).sum())) for name,s in sensi.items()}
    circle={}
    for name,item in ellipse.items():
        item['relative_bias']=item['predicted_efficiency']/item['observed_efficiency']-1
        train_ellipse=ellipse_workers[name]*V/(N/200*ellipse_volume)
        se=np.hypot(train_ellipse.std(ddof=1)/np.sqrt(200),np.sqrt(ellipse_accepted)/Ne)
        item['relative_se']=float(se/item['observed_efficiency'])
        item['hold']=abs(item['relative_bias'])>max(.05,3*item['relative_se'])
        wt=workers[name].sum(1)/(N/200);wv=ow.sum(1)/(Nv/200)
        bias=float(wt.mean()/wv.mean()-1);se=float(np.hypot(wt.std(ddof=1),wv.std(ddof=1))/np.sqrt(200)/wv.mean())
        circle[name]=dict(relative_bias=bias,relative_se=se,hold=abs(bias)>max(.02,3*se),
            predicted_efficiency=float(wt.mean()),observed_efficiency=float(wv.mean()))

    np.savez_compressed(a.output/'calibration_worker_evidence.npz',**workers,observed=ow,volume_mm3=volumes)
    write(a.output/'spatial_gate.json',dict(candidate_hold_bins=sum(r['hold'] for r in reports if r['model']=='continuous_energy'),
        legacy_hold_bins=sum(r['hold'] for r in reports if r['model']=='angular'),
        adequate_bins=sum(r['adequate'] for r in reports if r['model']=='continuous_energy'),
        event_policy='legacy',circle=circle,train_events=len(indices),validation_events=len(selected),
        actual_training_photons=N,actual_validation_photons=Nv,ellipse=ellipse,
        normalization='actual photons, full cell volume, same frozen events; no accepted-fraction rescaling',
        candidate_max_cell_detection_probability=float(np.max(sensi['continuous_energy']/vol)),
        candidate_min_cell_detection_probability=float(np.min(sensi['continuous_energy']/vol)),
        active_indices_from='separate whole-polar-cell geometry; full S is not yet deployed'))


def validate_joint(a,detector,vol,V,N,contexts,spectral):
    reports=[]
    for point in range(7):
        observed=np.zeros(72);worker_variance=np.zeros(72)
        for view in range(1,21):
            folder=a.inputs/f'point_{point}'
            raw=np.loadtxt(folder/f'legacy_v{view:02d}.csv',delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
            selected=selected_rows(a,f'point_{point}',view)
            observed+=np.bincount(category(raw[selected],detector),minlength=72)
        source_indices=[i for i,r in enumerate(contexts) if r['point']==point]
        for name in spectral:
            # All 20 rotations share the same 200 calibration workers.
            wp=spectral[name][:,source_indices].mean(1)*V/(N/200)
            rate=wp.mean(0);se=wp.std(0,ddof=1)/np.sqrt(200)
            expected=rate*10_000_000
            combined=np.sqrt((se*10_000_000)**2+observed)
            for cat in range(72):
                adequate=observed[cat]>=50
                bias=float(expected[cat]/observed[cat]-1) if observed[cat]>0 else None
                relse=float(combined[cat]/observed[cat]) if observed[cat]>0 else None
                hold=bool(adequate and abs(bias)>.3 and abs(bias)>3*relse)
                reports.append(dict(model=name,point=point,category=cat,observed=float(observed[cat]),
                    predicted=float(expected[cat]),relative_bias=bias,relative_se=relse,adequate=bool(adequate),hold=hold))
    table(a.output/'independent_joint_categories.csv',reports)
    write(a.output/'joint_gate.json',dict(categories=72,source_positions=7,
        fixed_E1_edges_MeV=[.05,.125,.20,settings().energy_threshold_max_mev],
        fixed_E2_edges_MeV=[.05,.25,'infinity'],layer_pairs=12,
        adequacy=50,hold_relative_bias=.3,hold_standard_errors=3,
        candidate_hold_categories=sum(r['hold'] for r in reports if r['model']=='continuous_energy'),
        legacy_hold_categories=sum(r['hold'] for r in reports if r['model']=='angular'),
        source_location_evaluation='nearest complete-circle polar representative; discretization remains a limitation',
        calibration_workers=200,point_workers=20,rotations_not_independent=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('inputs','analysis','factors','geometry','grid-config','training-law','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--probe-events',type=int,default=0)
    a=parser.parse_args();a.device='cuda:0';a.batch=32;a.node_chunk=16
    a.compiled_bins=False;a.energy_backend='tail16_mid2';a.gaussian_float32=True
    start=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.set_grad_enabled(False);torch.cuda.set_device(0)
    gate=json.loads((a.analysis/'selection_gate.json').read_text())
    if gate['status']!='PASSED' or gate['event_policy']!='legacy' or gate['geometry_mode']!='stable_float64':
        raise ValueError('This experiment requires frozen legacy stable selections')
    import compton_event_response
    if digest(compton_event_response.__file__)!=gate['kernel_sha256']:raise ValueError('Kernel changed')
    if digest(a.geometry)!=gate['geometry_sha256']:raise ValueError('Grid changed')
    if digest(a.inputs/'input_manifest.json')!=gate['input_manifest_sha256']:raise ValueError('Manifest changed')
    if digest(a.training_law)!='9b752ae4a8568646c2225b53736662410f5acceb4dbcb4d542a2167fe0ac784d':
        raise ValueError('Frozen material law changed; retraining is not authorized')
    manifest=json.loads((a.inputs/'input_manifest.json').read_text())['files'];consumed={}
    names=['circle_train','circle_validation','ellipse_validation']+[f'point_{i}' for i in range(7)]
    for name,expected in manifest.items():
        if name.split('/')[0] in names and (name.endswith('collection.json') or '/legacy_v' in name or '/events_v' in name):
            if digest(a.inputs/name)!=expected:raise ValueError('Input changed: '+name)
            consumed[name]=expected
    for r in gate['records']:
        if digest(a.analysis/f"{r['dataset']}_legacy_v{r['view']:02d}_kept_rows.npy")!=r['selection_sha256']:
            raise ValueError('Selection changed')
    coords,_,slices=grid(json.loads(a.grid_config.read_text()));geo=np.load(a.geometry)
    if not np.allclose(coords,geo['coordinates_mm'],atol=1e-5):raise ValueError('Geometry coordinates differ')
    detector=load_detector_coordinates(a.factors/'440keV_RotateNum20/Detector.csv',10496)
    d=torch.tensor(detector,device=a.device,dtype=torch.float32);var=build_detector_position_variance(d,0)
    law=ContinuousTransferLaw.load(a.training_law)
    write(a.output/'consumed_inputs.json',dict(input_files=consumed,selection_gate_sha256=digest(a.analysis/'selection_gate.json'),
        transfer_law_sha256=digest(a.training_law),event_policy='legacy'))
    spatial(a,d,var,law,coords,slices,geo)
    memory=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemTotal:')))*1024
    cgroup=Path('/sys/fs/cgroup/memory.max')
    value=cgroup.read_text().strip() if cgroup.exists() else 'max'
    limit=min(mem,int(value)) if value.isdigit() else mem
    execution=dict(elapsed_seconds=time.monotonic()-start,host_peak_rss_bytes=memory,host_allocated_bytes=limit,
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        candidate_sha256=digest(Path(__file__).with_name('compton_energy_probability_v5.py')),
        original_kernel_sha256=gate['kernel_sha256'],source_sha256=digest(__file__),geometry_sha256=digest(a.geometry),
        training_law_sha256=digest(a.training_law),event_policy='legacy',geometry_dtype='float64',
        gaussian_evaluation_dtype='float32',backend='tail16_mid2',node_chunk=16,new_photons=0,new_training=False)
    write(a.output/'execution.json',execution)
    if memory>.8*limit or execution['gpu_peak_reserved_bytes']>.8*execution['gpu_total_bytes']:
        raise ValueError('Actual resources exceed 80 percent')
    if not a.probe_events:
        spatial_gate=json.loads((a.output/'spatial_gate.json').read_text())
        joint=json.loads((a.output/'joint_gate.json').read_text())
        failures=[]
        for model,prefix in [('angular','legacy'),('continuous_energy','candidate')]:
            if spatial_gate[prefix+'_hold_bins']:failures.append(model+' spatial bias')
            if spatial_gate['circle'][model]['hold']:failures.append(model+' circle closure')
            if spatial_gate['ellipse'][model]['hold']:failures.append(model+' ellipse closure')
            if joint[prefix+'_hold_categories']:failures.append(model+' independent joint categories')
        files={p.name:digest(p) for p in a.output.iterdir() if p.is_file()}
        write(a.output/'calibration_gate.json',dict(study=STUDY,status='HOLD' if failures else 'PASSED',
            failures=failures,event_policy='legacy',files=files,selection_gate_sha256=digest(a.analysis/'selection_gate.json'),
            models=['angular','continuous_energy'],normalization='actual emitted photons, never accepted counts',
            limitation='Frozen material law trained on ideal events; this is a legacy-policy transfer test. Sparse joint cells stay unresolved.'))
        if failures:raise ValueError('Legacy calibration HOLD: '+', '.join(failures))
    print('LEGACY_CALIBRATION_FINISHED',a.probe_events,time.monotonic()-start,flush=True)

if __name__=='__main__':main()

