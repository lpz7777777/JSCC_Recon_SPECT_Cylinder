"""Bounded, read-only diagnostics on frozen first-scatter transport.

Never emits a replacement List, changes q selection, or launches reconstruction.
Source labels are used only for diagnostics. Matrix B remains the existing
full-circle density basis; fine A-field production is not a dependency.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
try:
    import resource
except ImportError:  # Windows can run unit tests; production diagnostics run on Linux.
    resource = None
import time

import numpy as np
import torch
from geometry import grid
from detector_csv import load_detector_coordinates
from compton_event_response import (
    ComptonEventSettings, build_detector_position_variance,
    prepare_compton_events, build_compton_cone_weights,
    min_standardized_compton_arm, _stable_event_angles,
)

E0 = .440
MASS = .511


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def table(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0)
    return dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x)>1 else 0.,
                rms=float(np.sqrt(np.mean(x*x))),
                quantiles=dict(zip(('p01','p05','p50','p95','p99'),
                                  map(float, np.quantile(x,[.01,.05,.5,.95,.99])))))


def settings():
    return ComptonEventSettings(E0, .13*np.sqrt(.511/E0),
        2*E0*E0/(MASS+2*E0)-.001, .05, .35, geometry_mode='stable_float64')


def metadata(folder, view, wanted=None):
    with (folder/f'events_v{view:02d}.csv').open() as stream:
        for r in csv.DictReader(stream):
            index = int(r['global_ideal_row'])
            if wanted is None or index in wanted:
                yield r


def vector(r, prefix):
    return np.array([float(r[prefix+k]) for k in 'xyz'])


def transfer(first, second, source):
    u, v = first-source, second-first
    cosine = np.clip(np.sum(u*v,axis=-1)/(np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)), -1, 1)
    beta = np.arctan2(np.linalg.norm(np.cross(u,v),axis=-1), np.sum(u*v,axis=-1))
    return E0-E0/(1+E0/MASS*(1-cosine)), beta


def selected_rows(a, dataset, view):
    return np.load(a.analysis/f'{dataset}_ideal_v{view:02d}_kept_rows.npy')


def point_diagnostics(a, detector):
    rows=[]; process_counts={}
    var=build_detector_position_variance(torch.tensor(detector,dtype=torch.float32),0)
    for folder in sorted(a.inputs.glob('point_*')):
        for view in range(1,21):
            kept=set(map(int,selected_rows(a,folder.name,view)))
            meta=list(metadata(folder,view))
            process_counts[folder.name+f'/v{view:02d}']=dict(
                recorded_candidates=len(meta), all_primary_interactions_available=False,
                reason='Recorder writes summary only for >=2 measured crystal hits or an accepted legacy/ideal event')
            serialized=np.loadtxt(folder/f'ideal_v{view:02d}.csv',delimiter=',',usecols=(0,1,2,3),dtype=np.float64,ndmin=2)
            raw=serialized.astype(np.float32)
            if not raw.size:continue
            prepared,_=prepare_compton_events(torch.tensor(raw),settings(),torch.tensor(detector,dtype=torch.float32),var,var,
                                             input_energies_already_smeared=True)
            qs={}; sigmas={}
            if prepared is not None:
                valid_meta=[r for r in meta if int(r['global_ideal_row'])>=0]
                if not valid_meta:raise ValueError('Missing point metadata')
                source=vector(valid_meta[0],'source_')+np.array([0,345,0])
                beta,theta,sigma=_stable_event_angles(prepared,torch.tensor(source[None],dtype=torch.float32),settings())
                q=((beta-theta[:,None]).abs()/sigma)[:,0].numpy()
                for i,index in enumerate(prepared.source_row_indices.tolist()):
                    qs[index]=float(q[i]);sigmas[index]=float(sigma[i,0])
            for r in meta:
                index=int(r['global_ideal_row'])
                if index<0:continue
                c1,c2=int(r['c1'])-1,int(r['c2'])-1
                if c1<0 or c2<0:raise ValueError('Ideal event has invalid crystals')
                src=vector(r,'source_')+np.array([0,345,0])
                p1=vector(r,'p1_')+np.array([0,345,0]);p2=vector(r,'p2_')+np.array([0,345,0])
                free,beta=transfer(p1,p2,src);center,bc=transfer(detector[c1],detector[c2],src)
                true=float(r['transfer_mev']);deposit=float(r['true_e1']);measured=float(r['measured_e1'])
                # Residual decomposition is signed and exactly additive.
                pieces=np.array([true-free,free-center,deposit-true,measured-deposit])*1000
                if abs(pieces.sum()-(measured-center)*1000)>1e-8:raise ValueError('Residual bookkeeping failure')
                sd=.13/2.355*np.sqrt(.511*max(center,1e-12))*1000
                # Actual six-significant-digit List values, not rounded metadata replacement.
                error=max(abs(serialized[index,1]-measured),abs(serialized[index,3]-float(r['measured_e2'])))
                for column,key in ((1,'measured_e1'),(3,'measured_e2')):
                    tolerance=.5001*10**(math.floor(math.log10(abs(serialized[index,column])))-5)
                    if abs(serialized[index,column]-float(r[key]))>tolerance:
                        raise ValueError('List and same-smear metadata disagree beyond six-digit serialization')
                rows.append(dict(dataset=folder.name,view=view,worker=int(r['worker']),seed=int(r['seed']),
                    event_id=int(r['event_id']),input_row=index,reco_energy_pass=index in qs,stable_q3_pass=index in kept,
                    c1=int(r['c1']),c2=int(r['c2']),layer1=int(round((abs(detector[c1,1])-300)/30)),
                    layer2=int(round((abs(detector[c2,1])-300)/30)),beta_true_deg=float(np.degrees(beta)),
                    beta_center_deg=float(np.degrees(bc)),free_transfer_keV=float(free*1000),
                    center_transfer_keV=float(center*1000),primary_transfer_keV=true*1000,
                    atomic_residual_keV=float(pieces[0]),position_residual_keV=float(pieces[1]),
                    accumulation_residual_keV=float(pieces[2]),measurement_residual_keV=float(pieces[3]),
                    total_residual_keV=float(pieces.sum()),free_energy_pull=float(pieces.sum()/sd),
                    stable_arm_q=qs.get(index),sigma_angle_deg=float(np.degrees(sigmas[index])) if index in sigmas else None,
                    measured_e1_keV=measured*1000,measured_e2_keV=float(r['measured_e2'])*1000,
                    list_metadata_max_energy_difference_MeV=error))
    table(a.output/'point_residuals.csv',rows)
    reports=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        for cohort in ('ideal_before_reco','energy_selected','stable_q3_selected'):
            chosen=[r for r in rows if r['dataset']==dataset and
                    (cohort=='ideal_before_reco' or (r['reco_energy_pass'] if cohort=='energy_selected' else r['stable_q3_pass']))]
            fields=['atomic_residual_keV','position_residual_keV','accumulation_residual_keV','measurement_residual_keV']
            x=np.array([[r[k] for k in fields] for r in chosen])
            reports.append(dict(dataset=dataset,cohort=cohort,events=len(chosen),
                residuals={k:summary([r[k] for r in chosen]) for k in fields+['total_residual_keV']},
                covariance_keV2=np.cov(x,rowvar=False).tolist() if len(x)>1 else None,
                arm_outside_3_fraction=float(np.mean([r['stable_arm_q']>3 for r in chosen if r['stable_arm_q'] is not None]))
                    if any(r['stable_arm_q'] is not None for r in chosen) else None,
                energy_pull_coverage={str(k):float(np.mean([abs(r['free_energy_pull'])<=k for r in chosen])) for k in (1,2,3)}))
    write(a.output/'point_residual_summary.json',dict(cohorts=reports,candidate_record_coverage=process_counts,
        normalized_likelihood_comparison=False,note='Free energy pull is diagnostic; selected/truncated samples are not standard-normal reference data'))
    print('POINT_RESIDUALS',len(rows),flush=True)


def fine_labels(coords):
    radius=np.hypot(coords[:,0],coords[:,1]);angle=np.mod(np.arctan2(coords[:,1],coords[:,0])+np.pi,2*np.pi)
    radial=np.searchsorted([81,159,219],radius,side='right')
    axial=np.searchsorted([-45,-24,0,24,45],coords[:,2],side='right')
    azimuth=np.minimum(7,np.floor(angle/(np.pi/4)).astype(int))
    return (radial*8+azimuth)*6+axial


def map_source_cell(position, slices, rotate=0):
    radius=np.hypot(position[:,0],position[:,1]);ring=np.floor((radius+3)/6).astype(int)
    if np.any(radius>255+1e-6) or np.any(abs(position[:,2])>60+1e-6):raise ValueError('Source outside circle support')
    starts=np.array([s[0] for s in slices]);counts=np.array([s[1] for s in slices])
    xy=np.zeros(len(ring),dtype=int);other=ring>0
    n=counts[ring[other]-1]
    angle=np.arctan2(position[other,1],position[other,0])+rotate*np.pi/10
    xy[other]=starts[ring[other]-1]+np.floor(angle*n/(2*np.pi)+.5).astype(int)%n
    layer=np.minimum(39,np.floor((position[:,2]+60)/3).astype(int))
    return layer*3301+xy


def source_matrix(rows, labels, slices):
    pos=np.array([vector(r,'source_')+np.array([0,345,0]) for r in rows])
    result=np.zeros((len(rows),192))
    for view in range(20):
        result[np.arange(len(rows)),labels[map_source_cell(pos,slices,view)]]+=1/20
    return result


def spatial_diagnostics(a, detector):
    cfg=json.loads(a.grid_config.read_text());coords,cells,slices=grid(cfg)
    geo=np.load(a.geometry);vol=geo['cell_volume_mm3'];V=float(vol.sum())
    labels=fine_labels(coords);binvol=np.bincount(labels,weights=vol,minlength=192)
    mask=np.eye(192)[labels]
    averaged=sum(mask[geo['inverse_rotation'][:,view]] for view in range(20))/20
    device=torch.device(a.device)
    d=torch.tensor(detector,dtype=torch.float32,device=device);var=build_detector_position_variance(d,0)
    coordinate=torch.tensor(coords,dtype=torch.float32,device=device)
    masks=torch.tensor(averaged,dtype=torch.float32,device=device)
    matrix=np.memmap(a.factors/'440keV_RotateNum20/SysMat_polar',dtype='<f4',mode='r',shape=(132040,10496))
    # One shared full matrix, never one copy per batch or process.
    B=torch.tensor(np.array(matrix.T,copy=True),device=device);del matrix
    workers={};counts={};totals={};start=time.monotonic()
    for dataset in ('circle_train','circle_validation'):
        folder=a.inputs/dataset;collection=json.loads((folder/'collection.json').read_text())
        indices=selected_rows(a,dataset,1);wanted=set(map(int,indices))
        meta={int(r['global_ideal_row']):r for r in metadata(folder,1,wanted)}
        if len(meta)!=len(indices):raise ValueError('Missing circle event metadata')
        totals[dataset]=int(collection['primary_counts'][1]);w=np.zeros((200,192))
        if dataset=='circle_validation':
            ordered=[meta[int(i)] for i in indices];contribution=source_matrix(ordered,labels,slices)
            np.add.at(w,np.array([int(r['worker']) for r in ordered]),contribution)
            workers[dataset]=w;counts[dataset]=contribution.sum(0);continue
        raw=np.loadtxt(folder/'ideal_v01.csv',delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
        limit=min(len(indices),a.probe_events) if a.probe_events else len(indices)
        for offset in range(0,limit,32):
            rows=indices[offset:min(offset+32,limit)]
            p,_=prepare_compton_events(torch.tensor(raw[rows],device=device),settings(),d,var,var,input_energies_already_smeared=True)
            if p is None or p.count!=len(rows):raise ValueError('Frozen identities fail preparation')
            response=build_compton_cone_weights(p,coordinate,settings())*B[p.cpnum1-1]
            normal=response/response.sum(1,keepdim=True)
            contribution=(normal@masks).double().cpu().numpy()
            np.add.at(w,np.array([int(meta[int(i)]['worker']) for i in rows]),contribution)
            if offset%3200==0:print('FINE_SPATIAL_EVENTS',offset,limit,'seconds',round(time.monotonic()-start,2),flush=True)
        if a.probe_events:
            write(a.output/'spatial_probe.json',dict(events=limit,full_training_events=len(indices),
                elapsed_seconds=time.monotonic()-start,scientific_validation=False,
                extrapolated_compute_seconds=(time.monotonic()-start)*len(indices)/limit))
            return
        workers[dataset]=w;counts[dataset]=w.sum(0)
    sensitivity=np.fromfile(a.analysis/'ideal/Sensi_d',dtype='<f4')
    predicted=np.bincount(labels,weights=sensitivity,minlength=192)/binvol
    recomputed=counts['circle_train']*V/totals['circle_train']/binvol
    if not np.allclose(predicted,recomputed,rtol=2e-5,atol=1e-12):raise ValueError('Fine worker sums do not reproduce frozen sensitivity')
    wt=workers['circle_train']*V/(totals['circle_train']/200*binvol)
    wv=workers['circle_validation']*V/(totals['circle_validation']/200*binvol)
    train_se=wt.std(0,ddof=1)/math.sqrt(200);validation_se=wv.std(0,ddof=1)/math.sqrt(200)
    observed=wv.mean(0);combined=np.hypot(train_se,validation_se)
    reports=[]
    for i in range(192):
        if binvol[i]<=0:continue
        error=float(predicted[i]/observed[i]-1) if observed[i]>0 else None
        se=float(combined[i]/observed[i]) if observed[i]>0 else None
        adequate=counts['circle_validation'][i]>=400
        fail=bool(adequate and abs(error)>.2 and abs(error)>3*se)
        reports.append(dict(bin=i,radial_bin=i//48,azimuth_bin=i//6%8,axial_bin=i%6,
            volume_mm3=float(binvol[i]),accepted_rotation_averaged_mass=float(counts['circle_validation'][i]),
            predicted_efficiency=float(predicted[i]),observed_efficiency=float(observed[i]),relative_error=error,
            relative_standard_error=se,statistically_adequate=bool(adequate),hold=fail))
    table(a.output/'fine_spatial_efficiency.csv',reports)
    write(a.output/'fine_spatial_gate.json',dict(status='HOLD' if any(r['hold'] for r in reports) else 'PASSED_AT_THIS_RESOLUTION',
        bins=192,gates=reports,adequate_bins=sum(r['statistically_adequate'] for r in reports),
        failing_bins=sum(r['hold'] for r in reports),normalization='actual primary total times exact polar-bin volume fraction',
        uncertainties='200 independent workers; 20 rotations combined within worker, not independent samples',
        all_primary_positions_recorded=False,expected_emissions_used=True,
        exact_per_bin_primary_counts_not_available=True,sensitivity_sha256=digest(a.analysis/'ideal/Sensi_d'),
        interpretation='Conditional selected-event efficiency; not an unconditional first-Compton model certificate'))
    np.savez_compressed(a.output/'worker_spatial_evidence.npz',train=wt,validation=wv,bin_volume_mm3=binvol)


def full_sector(cell,z,order):
    lo,hi,start,end=cell
    node,weight=np.polynomial.legendre.leggauss(order)
    theta=(start+end)/2+(end-start)*node/2
    r=np.sqrt(lo*lo+(hi*hi-lo*lo)*(node+1)/2)
    t,r,zz=np.meshgrid(theta,r,z+1.5*node,indexing='ij')
    w=weight[:,None,None]*weight[None,:,None]*weight[None,None,:]
    points=np.column_stack((r.ravel()*np.cos(t.ravel()),r.ravel()*np.sin(t.ravel()),zz.ravel()))
    return points,w.ravel()/w.sum()


def sampling_diagnostics(a,detector):
    cfg=json.loads(a.grid_config.read_text());coords,cells,_=grid(cfg)
    # Fixed physical controls, independent of image peaks and NEMA sphere positions.
    targets=[(r*np.cos(t),r*np.sin(t),z) for r in (0,60,120,180) for t in (0,np.pi/2,np.pi,3*np.pi/2)
             for z in (-21,1.5,22.5)]
    targets += [(225,0,-55.5),(225,0,55.5),(-225,0,-55.5),(-225,0,55.5)]
    selected=np.unique([int(np.argmin(np.sum((coords-np.array(p))**2,axis=1))) for p in targets])
    assert len(selected)<=64
    device=torch.device(a.device);d=torch.tensor(detector,dtype=torch.float32,device=device)
    var=build_detector_position_variance(d,0);results=[]
    for view in range(1,21):
        kept=selected_rows(a,'NEMA',view)
        # Deterministic even coverage of input order, not image-derived selection.
        chosen=kept[np.linspace(0,len(kept)-1,min(16,len(kept)),dtype=int)]
        raw=np.loadtxt(a.inputs/f'NEMA/ideal_v{view:02d}.csv',delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)[chosen]
        p,_=prepare_compton_events(torch.tensor(raw,device=device),settings(),d,var,var,input_energies_already_smeared=True)
        if p is None or p.count!=len(chosen):raise ValueError('Sampling event identities changed')
        angle=(view-1)*np.pi/10
        def rotate(points):
            q=points.copy();q[:,0]=points[:,0]*np.cos(angle)+points[:,1]*np.sin(angle)
            q[:,1]=points[:,1]*np.cos(angle)-points[:,0]*np.sin(angle);return q
        center=build_compton_cone_weights(p,torch.tensor(rotate(coords[selected]),dtype=torch.float32,device=device),settings()).double().cpu().numpy()
        values={}
        for order in (2,4,8):
            parts=[]
            for j in selected:
                point,w=full_sector(cells[j%3301],coords[j,2],order)
                kernel=build_compton_cone_weights(p,torch.tensor(rotate(point),dtype=torch.float32,device=device),settings())
                parts.append((kernel.double()@torch.tensor(w,dtype=torch.float64,device=device)).cpu().numpy())
            values[order]=np.column_stack(parts)
        for e,row in enumerate(chosen):
            for k,j in enumerate(selected):
                ref=values[8][e,k];adequate=ref>1e-6
                results.append(dict(view=view,input_row=int(row),cell=int(j),x_mm=float(coords[j,0]),y_mm=float(coords[j,1]),z_mm=float(coords[j,2]),
                    center_K=float(center[e,k]),average_K_2=float(values[2][e,k]),average_K_4=float(values[4][e,k]),average_K_8=float(ref),
                    response_adequate=bool(adequate),center_relative_error=float(center[e,k]/ref-1) if adequate else None,
                    refinement_relative_error=float(abs(values[4][e,k]/ref-1)) if adequate else None))
        print('INTERIOR_K_SAMPLING_VIEW',view,flush=True)
    table(a.output/'whole_cell_K_sampling.csv',results)
    adequate=[r for r in results if r['response_adequate']]
    write(a.output/'whole_cell_K_sampling_summary.json',dict(events=20*16,control_cells=len(selected),cases=len(results),
        adequate_cases=len(adequate),converged_cases=sum(r['refinement_relative_error']<=.01 for r in adequate),
        center_relative_error=summary([r['center_relative_error'] for r in adequate]),
        refinement_relative_error=summary([r['refinement_relative_error'] for r in adequate]),
        A_assumption='constant within each full polar cell; only K integrated',
        interpretation='Isolates K sampling error; not a complete K*A integral or a reconstruction result'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('points','spatial','sampling'))
    for name in ('inputs','analysis','factors','geometry','grid-config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--probe-events',type=int,default=0)
    a=p.parse_args()
    for source in (a.inputs,a.analysis,a.factors):
        if a.output.resolve().is_relative_to(source.resolve()):
            raise ValueError('Output must not be inside a frozen input directory')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.set_grad_enabled(False)
    if torch.device(a.device).type=='cuda':torch.cuda.set_device(torch.device(a.device))
    gate=json.loads((a.analysis/'validation_gate.json').read_text())
    if gate['geometry_mode']!='stable_float64' or gate['status']!='PASSED':raise ValueError('Frozen stable selection required')
    if gate['geometry_sha256']!=digest(a.geometry):raise ValueError('Wrong geometry')
    import compton_event_response
    if digest(compton_event_response.__file__)!=gate['kernel_sha256']:raise ValueError('Frozen kernel differs')
    if digest(a.inputs/'input_manifest.json')!=gate['input_manifest_sha256']:raise ValueError('Input manifest differs')
    names={'points':['point_'+str(i) for i in range(7)],
           'spatial':['circle_train','circle_validation'],'sampling':['NEMA']}[a.action]
    manifest=json.loads((a.inputs/'input_manifest.json').read_text())['files']
    consumed={}
    for name,sha in manifest.items():
        if name.split('/')[0] in names and (name.endswith('collection.json') or
                name.split('/')[-1].startswith(('ideal_v','events_v'))):
            actual=digest(a.inputs/name)
            if actual!=sha:raise ValueError('Frozen input hash differs: '+name)
            consumed[name]=actual
    selections={p.name:digest(p) for dataset in names for p in a.analysis.glob(dataset+'_ideal_v*_kept_rows.npy')}
    write(a.output/'consumed_inputs.json',dict(input_files=consumed,selection_files=selections))
    detector=load_detector_coordinates(a.factors/'440keV_RotateNum20/Detector.csv',10496)
    start=time.monotonic()
    {'points':point_diagnostics,'spatial':spatial_diagnostics,'sampling':sampling_diagnostics}[a.action](a,detector)
    import compton_event_response
    record=dict(action=a.action,elapsed_seconds=time.monotonic()-start,new_transport_photons=0,
        full_event_response_saved=False,input_manifest_sha256=digest(a.inputs/'input_manifest.json'),
        stable_selection_sha256=digest(a.analysis/'validation_gate.json'),geometry_sha256=digest(a.geometry),
        kernel_sha256=digest(compton_event_response.__file__),source_sha256=digest(__file__),
        detector_sha256=digest(a.factors/'440keV_RotateNum20/Detector.csv'),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None)
    if torch.device(a.device).type=='cuda':
        record.update(gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      gpu_total_bytes=torch.cuda.get_device_properties(torch.device(a.device)).total_memory)
    write(a.output/'execution.json',record)
    print('AUDIT_COMPLETED',a.action,round(record['elapsed_seconds'],2),flush=True)


if __name__=='__main__':main()
