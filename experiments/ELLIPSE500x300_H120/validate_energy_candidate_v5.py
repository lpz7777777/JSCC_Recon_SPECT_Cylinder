"""Read-only validation and new S for one frozen energy-response candidate."""
import argparse
import csv
import json
import math
from pathlib import Path
import resource
import time
import numpy as np
import torch
from scipy.special import logsumexp
from compton_event_response import (prepare_compton_events,build_detector_position_variance,
    build_compton_cone_weights,min_standardized_compton_arm)
from detector_csv import load_detector_coordinates
from geometry import grid
from process_list_global_audit_v4 import (digest,write,table,metadata,vector,settings,
    selected_rows,fine_labels,source_matrix,map_source_cell)
from compton_energy_probability_v5 import (ContinuousTransferLaw,geometry_arrays,
    selected_logpdf_pit,log_interval_mass,fixed_q_min,fixed_q_intervals,
    normalized_proxy_response)


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


def point_scores(a,d,variance,law,coords,slices):
    points=list(csv.DictReader(a.points.open()))
    _,contexts=source_contexts(a,d.cpu().numpy(),coords,slices)
    raw_cache={};records=[];quadrature=[];hybrid_error=[]
    chosen=[r for r in points if r['reco_energy_pass']=='True']
    refinement_sample=set()
    for point in range(7):
        group=[r for r in chosen if r['dataset']==f'point_{point}']
        for i in np.linspace(0,len(group)-1,min(32,len(group)),dtype=int):
            r=group[i];refinement_sample.add((r['dataset'],r['view'],r['input_row']))
    for r in chosen:
        key=(r['dataset'],int(r['view']));point=int(r['dataset'].split('_')[1]);view=key[1]
        if key not in raw_cache:
            raw_cache[key]=np.loadtxt(a.inputs/key[0]/f'ideal_v{view:02d}.csv',delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
        raw=raw_cache[key][int(r['input_row']):int(r['input_row'])+1]
        p=prepare(raw,d,variance,a.device)
        source=torch.tensor(contexts[(point,view)]['source'][None],device=a.device,dtype=torch.float32)
        beta,sp=geometry_arrays(p,source)
        e1,e2=float(raw[0,1]),float(raw[0,3]);lo=max(.05,.35-e2);hi=settings().energy_threshold_max_mev
        for position,b,s in [('true',math.radians(float(r['beta_true_deg'])),0.),
                             ('centre',float(beta[0,0]),float(sp[0,0]))]:
            for name,material in [('free',False),('material',True)]:
                ll,pit,z,extra=law.selected_probability(e1,[(lo,hi)],b,int(r['layer1']),s,material)
                if material:
                    hm,hs,hw,_=law.components(b,int(r['layer1']),s,True,order='tail16_mid2')
                    hybrid_pdf=logsumexp(-.5*((e1-hm)/hs)**2-np.log(hs)-.5*np.log(2*np.pi)+np.log(hw))
                    hybrid_error.append(float(abs(np.expm1(hybrid_pdf-(ll+z)))))
                if material and (r['dataset'],r['view'],r['input_row']) in refinement_sample:
                    m4,s4,w4,_=law.components(b,int(r['layer1']),s,True,order=16)
                    ll4,pit4,z4=selected_logpdf_pit(e1,[(lo,hi)],m4,s4,w4)
                    quadrature.append(dict(point=point,position=position,input_row=int(r['input_row']),
                        density_relative_change=float(abs(np.expm1(ll4-ll))),
                        mass_relative_change=float(abs(np.expm1(z4-z))),pit_absolute_change=abs(pit4-pit)))
                records.append(dict(dataset=r['dataset'],view=view,seed=int(r['seed']),input_row=int(r['input_row']),
                    model=name+'_'+position,log_density=ll,pit=pit,log_energy_acceptance_mass=z,
                    endpoint_extrapolation=extra,stable_q3_pass=r['stable_q3_pass']=='True'))
    table(a.output/'point_probability.csv',records)
    reports=[]
    for point in range(7):
        for position in ('true','centre'):
            group=[r for r in records if r['dataset']==f'point_{point}']
            old=[r for r in group if r['model']=='free_'+position]
            new=[r for r in group if r['model']=='material_'+position]
            gain=np.array([b['log_density']-aa['log_density'] for aa,b in zip(old,new)])
            seeds=np.array([r['seed'] for r in old]);wm=np.array([gain[seeds==s].mean() for s in sorted(set(seeds))])
            if len(wm)!=20:raise ValueError('Missing independent point views')
            reports.append(dict(dataset=f'point_{point}',position=position,events=len(old),
                mean_gain_nats=float(gain.mean()),median_gain_nats=float(np.median(gain)),
                seed_mean_gain_nats=float(wm.mean()),seed_standard_error=float(wm.std(ddof=1)/np.sqrt(20)),
                free_tail_fraction=float(np.mean([r['pit']<.01 or r['pit']>.99 for r in old])),
                material_tail_fraction=float(np.mean([r['pit']<.01 or r['pit']>.99 for r in new])),
                endpoint_extrapolated_events=sum(r['endpoint_extrapolation'] for r in new)))
    numerical=dict(density_method='analytic vs GL16',selection_mass_orders=[8,16],contexts=len(quadrature),fixed_relative_target=.01,
        max_density_relative_change=max(r['density_relative_change'] for r in quadrature),
        max_mass_relative_change=max(r['mass_relative_change'] for r in quadrature),
        max_pit_absolute_change=max(r['pit_absolute_change'] for r in quadrature))
    numerical['passed']=numerical['max_density_relative_change']<=.01 and numerical['max_mass_relative_change']<=.01
    numerical['hybrid']=dict(backend='tail16_mid2',contexts=len(hybrid_error),
        max_density_relative_change=max(hybrid_error),relative_target=.01,passed=max(hybrid_error)<=.01)
    write(a.output/'point_gate.json',dict(scored_events=len(chosen),dropped_events=0,quadrature=numerical,
        reports=reports,position_variance='original uniform-crystal propagation, zero bias, unchanged sizes',
        q_in_this_score=False,interpretation='Energy-window conditional diagnostic; complete q handled separately'))
    print('ALL_POINT_SCORES_FINISHED',len(chosen),flush=True)


def q_scores(a,d,var,law,coords):
    points=list(csv.DictReader(a.points.open()));results=[]
    co=torch.tensor(coords,device=a.device,dtype=torch.float32)
    for point in range(7):
        group=[r for r in points if r['dataset']==f'point_{point}' and r['stable_q3_pass']=='True']
        chosen=[group[i] for i in np.linspace(0,len(group)-1,min(a.q_per_source,len(group)),dtype=int)]
        raw_cache={}
        for index,r in enumerate(chosen):
            view=int(r['view']);folder=a.inputs/f'point_{point}'
            if view not in raw_cache:
                raw_cache[view]=np.loadtxt(folder/f'ideal_v{view:02d}.csv',delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
            raw=raw_cache[view][int(r['input_row']):int(r['input_row'])+1]
            p=prepare(raw,d,var,a.device);beta,sp=geometry_arrays(p,co)
            beta,sp=beta[0],sp[0]
            e1,e2=float(raw[0,1]),float(raw[0,3]);lo=max(.05,.35-e2);hi=settings().energy_threshold_max_mev
            q=float(fixed_q_min([e1],beta,sp)[0])
            direct=float(min_standardized_compton_arm(p,co,settings())[0])
            if abs(q-direct)>1e-10 or q>3:raise ValueError('Frozen q membership or formula differs')
            coarse=fixed_q_intervals(lo,hi,beta,sp,256)
            fine=fixed_q_intervals(lo,hi,beta,sp,512)
            # Dense refinement checks numerical selection integration, not an analytic certificate.
            context=next(metadata(folder,view));source=vector(context,'source_')+np.array([0,345,0])
            bs,sps=geometry_arrays(p,torch.tensor(source[None],device=a.device,dtype=torch.float32))
            for material in (False,True):
                bs0,sps0=float(bs[0,0]),float(sps[0,0])
                means,sd,weights,extra=law.components(bs0,int(r['layer1']),sps0,material)
                ll,pit,z,_=law.selected_probability(e1,fine,bs0,int(r['layer1']),sps0,material)
                z0=logsumexp([log_interval_mass(l,h,means,sd,weights) for l,h in coarse])
                refinement=float(abs(np.expm1(z0-z)))
                results.append(dict(dataset=f'point_{point}',view=view,seed=int(r['seed']),input_row=int(r['input_row']),
                    model='material' if material else 'free',q=q,intervals=fine,
                    log_selected_density=ll,pit=pit,refinement_relative_mass=refinement,
                    endpoint_extrapolation=extra))
            print('FIXED_Q_CONTEXT',point,index+1,len(chosen),flush=True)
    write(a.output/'q_selected_probability.json',dict(records=results,
        membership_matches=True,scan_nodes=[256,512],integration_target=.001,
        all_refinements_pass=all(r['refinement_relative_mass']<=.001 for r in results),
        normalized_over='energy window intersect frozen complete-circle stable q<=3',
        used_in_forward_response=False,new_event_selection=False,
        limitation='Finite interval scans; components narrower than both grids remain unproved'))


def spatial(a,d,var,law,coords,slices,geo):
    device=a.device;vol=geo['cell_volume_mm3'];V=float(vol.sum());labels=fine_labels(coords)
    masks=np.eye(192)[labels]
    averaged=sum(masks[geo['inverse_rotation'][:,v]] for v in range(20))/20
    mask=torch.tensor(averaged,device=device,dtype=torch.float32)
    co=torch.tensor(coords,device=device,dtype=torch.float32)
    matrix=np.memmap(a.factors/'440keV_RotateNum20/SysMat_polar',dtype='<f4',mode='r',shape=(132040,10496))
    B=torch.tensor(np.array(matrix.T,copy=True),device=device);del matrix
    folder=a.inputs/'circle_train';indices=selected_rows(a,'circle_train',1)
    meta={int(r['global_ideal_row']):r for r in metadata(folder,1,set(map(int,indices)))}
    raw=np.loadtxt(folder/'ideal_v01.csv',delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
    contexts,_=source_contexts(a,d.cpu().numpy(),coords,slices)
    cells=np.array([r['cell'] for r in contexts]);point_volume=vol[cells]
    workers={name:np.zeros((200,192)) for name in ('legacy','candidate')}
    spectral={name:np.zeros((200,140,72)) for name in workers}
    sums={name:torch.zeros(132040,device=device,dtype=torch.float64) for name in workers}
    limit=min(len(indices),a.probe_events) if a.probe_events else len(indices)
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
        for name,response in (('legacy',old),('candidate',new)):
            sums[name]+=response.double().sum(0)
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
    old_s=np.fromfile(a.analysis/'ideal/Sensi_d',dtype='<f4')
    error=float(np.linalg.norm(sensi['legacy']-old_s)/np.linalg.norm(old_s))
    if error>1e-5:raise ValueError('Original S regression failed')
    validation=a.inputs/'circle_validation';selected=selected_rows(a,'circle_validation',1)
    vm={int(r['global_ideal_row']):r for r in metadata(validation,1,set(map(int,selected)))}
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
    for item in ellipse.values():item['relative_bias']=item['predicted_efficiency']/item['observed_efficiency']-1
    np.savez_compressed(a.output/'calibration_worker_evidence.npz',**workers,observed=ow,volume_mm3=volumes)
    write(a.output/'spatial_gate.json',dict(candidate_hold_bins=sum(r['hold'] for r in reports if r['model']=='candidate'),
        legacy_hold_bins=sum(r['hold'] for r in reports if r['model']=='legacy'),
        adequate_bins=sum(r['adequate'] for r in reports if r['model']=='candidate'),
        original_S_relative_L2=error,train_events=len(indices),validation_events=len(selected),
        actual_training_photons=N,actual_validation_photons=Nv,ellipse=ellipse,
        normalization='actual photons, full cell volume, same frozen events; no accepted-fraction rescaling',
        candidate_max_cell_detection_probability=float(np.max(sensi['candidate']/vol)),
        candidate_min_cell_detection_probability=float(np.min(sensi['candidate']/vol)),
        active_indices_from='separate whole-polar-cell geometry; full S is not yet deployed'))


def validate_joint(a,detector,vol,V,N,contexts,spectral):
    reports=[]
    for point in range(7):
        observed=np.zeros(72);worker_variance=np.zeros(72)
        for view in range(1,21):
            folder=a.inputs/f'point_{point}'
            raw=np.loadtxt(folder/f'ideal_v{view:02d}.csv',delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
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
        candidate_hold_categories=sum(r['hold'] for r in reports if r['model']=='candidate'),
        legacy_hold_categories=sum(r['hold'] for r in reports if r['model']=='legacy'),
        source_location_evaluation='nearest complete-circle polar representative; discretization remains a limitation',
        calibration_workers=200,point_workers=20,rotations_not_independent=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['points','q','spatial'])
    for key in ('inputs','analysis','factors','geometry','grid-config','training-law','points','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--device',default='cpu');parser.add_argument('--probe-events',type=int,default=0)
    parser.add_argument('--batch',type=int,default=32);parser.add_argument('--node-chunk',type=int,default=16)
    parser.add_argument('--compiled-bins',action='store_true')
    parser.add_argument('--energy-backend',choices=['analytic','tail16_mid2'],default='analytic')
    parser.add_argument('--gaussian-float32',action='store_true')
    parser.add_argument('--q-per-source',type=int,default=32)
    a=parser.parse_args();start=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.set_grad_enabled(False)
    if a.device.startswith('cuda'):torch.cuda.set_device(a.device)
    gate=json.loads((a.analysis/'validation_gate.json').read_text())
    if gate['status']!='PASSED' or gate['geometry_mode']!='stable_float64':raise ValueError('Frozen stable selection required')
    import compton_event_response
    if digest(compton_event_response.__file__)!=gate['kernel_sha256']:raise ValueError('Original kernel changed')
    if digest(a.geometry)!=gate['geometry_sha256']:raise ValueError('Original complete-circle geometry changed')
    if digest(a.inputs/'input_manifest.json')!=gate['input_manifest_sha256']:raise ValueError('Input manifest changed')
    manifest=json.loads((a.inputs/'input_manifest.json').read_text())['files'];consumed={}
    names=[f'point_{i}' for i in range(7)]
    if a.action=='spatial':names+=['circle_train','circle_validation','ellipse_validation']
    for name,expected in manifest.items():
        if name.split('/')[0] in names and name.endswith(('.csv','collection.json')):
            actual=digest(a.inputs/name)
            if actual!=expected:raise ValueError('Input differs: '+name)
            consumed[name]=actual
    coords,_,slices=grid(json.loads(a.grid_config.read_text()));geo=np.load(a.geometry)
    detector=load_detector_coordinates(a.factors/'440keV_RotateNum20/Detector.csv',10496)
    d=torch.tensor(detector,device=a.device,dtype=torch.float32);var=build_detector_position_variance(d,0)
    law=ContinuousTransferLaw.load(a.training_law)
    selections={f.name:digest(f) for name in names for f in a.analysis.glob(name+'_ideal_v*_kept_rows.npy')}
    write(a.output/'consumed_inputs.json',dict(input_files=consumed,selection_files=selections,
        transfer_law_sha256=digest(a.training_law),points_sha256=digest(a.points)))
    if a.action=='points':point_scores(a,d,var,law,coords,slices)
    elif a.action=='q':q_scores(a,d,var,law,coords)
    else:spatial(a,d,var,law,coords,slices,geo)
    memory=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    cuda=a.device.startswith('cuda')
    write(a.output/'execution.json',dict(action=a.action,elapsed_seconds=time.monotonic()-start,
        host_peak_rss_bytes=memory,gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else 0,
        gpu_total_bytes=torch.cuda.get_device_properties(a.device).total_memory if cuda else 0,
        source_sha256=digest(__file__),candidate_sha256=digest(Path(__file__).with_name('compton_energy_probability_v5.py')),
        original_kernel_sha256=gate['kernel_sha256'],geometry_sha256=gate['geometry_sha256'],
        compiled_float64_bins=a.compiled_bins,
        numerical_energy_backend=a.energy_backend,
        gaussian_evaluation_dtype='float32' if a.gaussian_float32 else 'float64',geometry_dtype='float64',
        new_photons=0,new_fine_A_matrices=0,new_reconstruction=False,full_response_saved=False))
    print('ENERGY_CANDIDATE_PHASE_FINISHED',a.action,round(time.monotonic()-start,2),flush=True)


if __name__=='__main__':main()
