"""Monoenergy SC and the unchanged 440-keV SC+Compton MLEM branch."""
import argparse
from datetime import timedelta
import hashlib
import os
import re
from pathlib import Path
import resource
import time
import numpy as np
import torch
import torch.distributed as dist
from derenzo_campaign_common import *
from jscc_5e10_runtime import DeviceMonitor
from jscc_5e10_common import host_allocated_bytes
from torch_active_operator import ActiveGeometry,ViewResponse,single_mlem,compton_and_joint_mlem,_event_weight,_reduce_sum,_update
from single_checkpoint_mlem import single_mlem_checkpointed
from run_reconstruction import local_rows,full_rows,split_bins,read_event_partition,collect_response
from run_energy_preflight_v5 import settings,rows_for,partition_indices,numerical_checks,ContinuousTransferLaw,load_detector_coordinates,build_detector_position_variance
from compton_event_response import prepare_compton_events,build_compton_cone_weights,min_standardized_compton_arm
from prepare_energy_5e9_v5 import subset
from run_jscc_5e10 import operator_checks,relative,publish_images


def joint_mlem(response,projection,blocks,single_s,compton_s,iterations,step,save_history=True,callback=None,limit=None):
    """Exact original joint branch, preserving view/batch/update order."""
    image=torch.ones((response.geometry.active_count,1),dtype=torch.float32,device=response.full_rows.device)
    history=[];started=time.monotonic()
    for i in range(iterations):
        if limit and time.monotonic()-started>limit:raise TimeoutError('Bounded joint phase exceeded')
        weight=torch.zeros_like(image)
        for view in range(response.geometry.views):
            matrix=response.matrix(view);forward=matrix@image
            weight+=matrix.T@(projection[:,view:view+1]/forward.clamp_min(1e-12))
            weight+=_event_weight(blocks[view],image,image.device)
        image=_update(image,_reduce_sum(weight),single_s+compton_s)
        if save_history and (i+1)%step==0:
            history.append(image.detach().cpu().clone())
            if callback:callback(i+1,history)
    return image,torch.stack(history) if history else None


def setup(plan,allocation):
    rank,world,local=(int(os.environ[n]) for n in ['RANK','WORLD_SIZE','LOCAL_RANK'])
    if world!=plan['world_size'] or torch.cuda.device_count()!=plan['gpus_per_node'] or not 0<=local<plan['gpus_per_node']:
        raise ValueError('Actual GPU/rank topology differs from this plan')
    torch.cuda.set_device(local);torch.set_num_threads(8);torch.set_grad_enabled(False)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    device=torch.device('cuda',local);monitor=DeviceMonitor(device)
    memory=host_allocated_bytes(Path(allocation).read_text(),plan['nodes'])
    return rank,world,local,device,monitor,memory


def resource_record(rank,local,device,monitor,memory,started):
    monitor.finish();prop=torch.cuda.get_device_properties(device)
    return dict(rank=rank,local_rank=local,node=os.environ['SLURMD_NODENAME'],gpu_uuid=monitor.uuid,
                host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                host_allocated_bytes_node=memory,peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                total_device_bytes=prop.total_memory,measured_gpu_used_peak_bytes=monitor.peak,
                gpu_memory_samples=monitor.samples,elapsed_seconds=time.monotonic()-started)


def verify_resources(records,plan,allocation):
    allocation_text=Path(allocation).read_text();memory=host_allocated_bytes(allocation_text,plan['nodes'])
    tres=re.search(r'\bAllocTRES=([^\s]+)',allocation_text)[1]
    if not re.search(r'(?:^|,)gres/gpu='+str(plan['world_size'])+r'(?:,|$)',tres):raise ValueError('Actual allocated GPU count differs')
    if len(records)!=plan['world_size'] or sorted(x['rank'] for x in records)!=list(range(plan['world_size'])):raise ValueError('All actual rank identities required')
    nodes={x['node'] for x in records}
    if len(nodes)!=plan['nodes'] or len({x['gpu_uuid'] for x in records})!=len(records):raise ValueError('Actual unique nodes/GPU UUIDs differ')
    for node in nodes:
        group=[x for x in records if x['node']==node]
        if sorted(x['local_rank'] for x in group)!=list(range(plan['gpus_per_node'])):raise ValueError('Local GPU identities differ')
        if sum(x['host_peak_rss_bytes'] for x in group)>.8*memory:raise MemoryError('Node RSS lacks 20% margin')
        for x in group:
            if (x['host_allocated_bytes_node']!=memory or x['host_peak_rss_bytes']<=0 or x['gpu_memory_samples']<2
                    or x['peak_reserved_bytes']>.8*x['total_device_bytes'] or x['measured_gpu_used_peak_bytes']>.8*x['total_device_bytes']):
                raise MemoryError('Actual GPU/RSS evidence fails 80% budget')
    return memory


def initial(a):
    cfg=read(a.release/'gpu_campaign.json');verify_files(a.release,read(a.release/'release_manifest.json')['sha256'])
    spec=cfg['datasets'][a.dataset];collection=validate_collection(read(a.input/'collection.json'),a.dataset)
    plan=read(a.plan)
    if (plan['dataset']!=a.dataset or plan['collection_sha256']!=digest(a.input/'collection.json')
            or digest(a.plan)!=a.plan_sha256 or plan['release_sha256']!=digest(a.release/'release_manifest.json')
            or plan['world_size']!=plan['nodes']*plan['gpus_per_node']):raise ValueError('Plan/input/release identity differs')
    started=time.monotonic();rank,world,local,device,monitor,memory=setup(plan,a.allocation)
    error=[None]
    if rank==0:
        try:
            verify_files(a.input,collection['files']);verify_files(Path(spec['factor_root']),spec['factor_files'])
            if digest(Path(spec['factor_root'])/'factor_manifest.json')!=spec['factor_manifest_sha256']:raise ValueError('Matrix manifest differs')
            a.output.mkdir(parents=True,exist_ok=False)
        except Exception as e:error=[str(e)]
    dist.broadcast_object_list(error,src=0)
    if error[0]:raise ValueError(error[0])
    geometry=ActiveGeometry.from_npz(a.release/'whole_geometry.npz',device)
    if (geometry.full_count,geometry.active_count)!=(132040,78920):raise ValueError('Original full/active grid differs')
    return cfg,spec,collection,plan,started,rank,world,local,device,monitor,memory,geometry


def save_npy(path,x):
    with Path(path).open('wb') as f:np.save(f,x);f.flush();os.fsync(f.fileno())


def select(a):
    if a.dataset!='JSCC_440':raise ValueError('Compton selection is only authorized at 440 keV')
    cfg,spec,coll,plan,started,rank,world,local,device,monitor,memory,geometry=initial(a)
    if rank==0:
        for name in ['parts','selections','selected_rows']:(a.output/name).mkdir()
    dist.barrier()
    geo=np.load(a.release/'whole_geometry.npz');coords=torch.tensor(geo['coordinates_mm'],dtype=torch.float32,device=device)
    root=Path(spec['factor_root']);rawB=np.memmap(root/'SysMat_polar','<f4',mode='c',shape=(132040,10496));B=full_rows(rawB,device)
    detector=torch.tensor(load_detector_coordinates(root/'Detector.csv',10496),dtype=torch.float32,device=device)
    variance=build_detector_position_variance(detector,0)
    for v in range(1,21):
        raw,byte_partition=read_event_partition(a.input/'List'/f'{v}.csv',rank,world)
        counts=[None]*world;dist.all_gather_object(counts,len(raw));offset=sum(counts[:rank])
        prepared,diag=prepare_compton_events(raw.to(device),settings(),detector,variance,variance,input_energies_already_smeared=True)
        kept=[];original=removed=0
        if prepared is not None:
            for start in range(0,prepared.count,32):
                e=subset(prepared,slice(start,start+32));w=build_compton_cone_weights(e,coords,settings())*B[e.cpnum1-1]
                mass=w.sum(1);valid=torch.isfinite(w).all(1)&torch.isfinite(mass)&(mass>0)
                e=subset(e,valid);del w,mass
                if not e.count:continue
                original+=e.count;score=min_standardized_compton_arm(e,coords,settings())
                if not bool(torch.isfinite(score).all()):raise ValueError('Nonfinite original q score')
                mask=score<=3;kept.extend((e.source_row_indices[mask].cpu().numpy()+offset).tolist());removed+=int((~mask).sum())
        name=f'{v:02d}_rank{rank:02d}'
        save_npy(a.output/'parts'/(name+'.npy'),np.asarray(kept,dtype='<i8'))
        save_npy(a.output/'parts'/(name+'_rows.npy'),raw[np.asarray(kept,dtype=np.int64)-offset].numpy())
        write(a.output/'parts'/(name+'.json'),dict(rank=rank,view=v,raw_rows=len(raw),offset=offset,
              byte_partition=byte_partition,original_accepted=original,removed=removed,kept=len(kept),
              files={n:digest(a.output/'parts'/n) for n in [name+'.npy',name+'_rows.npy']}))
        print('DERENZO_SELECTED',rank,v,len(kept),flush=True)
    record=resource_record(rank,local,device,monitor,memory,started);records=[None]*world;dist.all_gather_object(records,record)
    verify_resources(records,plan,a.allocation);dist.barrier()
    if rank==0:
        events=[];total_raw=0;original=removed=0
        for v in range(1,21):
            indices=[];rows=[];cursor=0
            for r in range(world):
                name=f'{v:02d}_rank{r:02d}';receipt=read(a.output/'parts'/(name+'.json'));verify_files(a.output/'parts',receipt['files'])
                if receipt['offset']!=cursor or receipt['kept']+receipt['removed']!=receipt['original_accepted']:raise ValueError('Raw event partitions do not close')
                cursor+=receipt['raw_rows'];original+=receipt['original_accepted'];removed+=receipt['removed']
                indices.append(np.load(a.output/'parts'/(name+'.npy')));rows.append(np.load(a.output/'parts'/(name+'_rows.npy')))
            values=np.concatenate(indices)
            if np.any(values[1:]<=values[:-1]):raise ValueError('Selection overlaps or changes order')
            save_npy(a.output/'selections'/f'{v}.npy',values);save_npy(a.output/'selected_rows'/f'{v}.npy',np.concatenate(rows));events.append(len(values));total_raw+=cursor
        if total_raw!=coll['raw_list_rows']:raise ValueError('All raw rows must be inspected')
        write(a.output/'selection_manifest.json',dict(passed=True,study=STUDY,dataset=a.dataset,events_per_view=events,
              accepted_events=sum(events),original_accepted=original,removed=removed,resources=records,plan=plan,
              collection_sha256=digest(a.input/'collection.json'),release_sha256=digest(a.release/'release_manifest.json'),
              selection_policy='Original legacy positive support and stable_float64 full-circle q<=3',files=hashes(a.output)))
    dist.barrier();dist.destroy_process_group()


def reconstruct(a):
    cfg,spec,coll,plan,started,rank,world,local,device,monitor,memory,geometry=initial(a)
    info=dataset_info(a.dataset);iterations,step=(10,10) if a.mode=='validation' else (info['iterations'],info['save_step'])
    if a.mode=='formal':
        if digest(a.authority)!=a.authority_sha256:raise ValueError('Formal authority bytes changed after startup gate')
        authority=read(a.authority)
        if (not authority['passed'] or authority['dataset']!=a.dataset or authority['plan_sha256']!=digest(a.plan)
                or authority['collection_sha256']!=digest(a.input/'collection.json') or authority['release_sha256']!=digest(a.release/'release_manifest.json')):raise ValueError('Strict current validation authority required')
        verify_files(Path(authority['validation_result']),authority['validation_files_sha256'])
    root=Path(spec['factor_root']);bins=info['bins'];begin,end=split_bins(bins,rank,world)
    raw=np.memmap(root/'SysMat_polar','<f4',mode='c',shape=(132040,bins))
    response=ViewResponse(local_rows(raw,begin,end,device),geometry,'none');sensitivity=response.sensitivity();dist.all_reduce(sensitivity)
    y=np.loadtxt(a.input/f"projection_{info['energy_keV']}.csv",delimiter=',',dtype=np.float32)
    if y.shape!=(20,bins) or not np.isfinite(y).all() or np.any(y<0):raise ValueError('All actual observation bins required')
    projection=torch.tensor(np.ascontiguousarray(y[:,begin:end].T),device=device)
    checks=operator_checks(response) if a.mode=='validation' else None
    phase_times={};regression={};histories={};phase_limit=float(os.environ['DERENZO_PHASE_SECONDS'])
    def checkpoint(channel,i,h):
        folder=a.output/'checkpoints'/channel/f'checkpoint_{i:06d}';temporary=folder.with_name('.'+folder.name+'.partial')
        temporary.mkdir(parents=True,exist_ok=False);collect_response(temporary,channel,h[-1],None,geometry,0)
        for p in temporary.iterdir():
            with p.open('r+b') as f:os.fsync(f.fileno())
        write(temporary/'manifest.json',dict(iteration=i,channel=channel,plan_sha256=digest(a.plan),files=hashes(temporary)))
        if folder.exists():raise FileExistsError('Preserve existing checkpoint')
        temporary.rename(folder)
        fd=os.open(folder.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        write(a.output/'progress.json',dict(channel=channel,iteration=i,target=iterations,updated_epoch=time.time()))
    channel=info['channels'][0];began=time.monotonic()
    image,h=single_mlem_checkpointed(response,projection,sensitivity,iterations,step,save_history=rank==0,
        checkpoint_callback=(lambda i,h:checkpoint(channel,i,h)) if rank==0 else None,phase_limit_seconds=phase_limit,progress_label=channel)
    phase_times[channel]=time.monotonic()-began
    if a.mode=='validation':
        ref,rh=single_mlem(response,projection,sensitivity,10,10,save_history=rank==0)
        regression[channel]=relative(image,ref)
        if regression[channel]>1e-5 or (rank==0 and not torch.equal(h,rh)):raise ValueError('Original single MLEM/history regression failed')
        del ref,rh
    publish_images(a.output,channel,image,h,geometry,rank)
    if rank==0:sensitivity.cpu().numpy().astype('<f4').tofile(a.output/'S_single.float32')
    del image,h
    prep_seconds=0.;accepted_events=0;partition_proof=[];event_checks=None;event_bytes=0;minimum_mass=None
    if a.dataset=='JSCC_440':
        manifest=read(a.selection/'selection_manifest.json');verify_files(a.selection,manifest['files'])
        if manifest['collection_sha256']!=digest(a.input/'collection.json'):raise ValueError('Selection input differs')
        coords=torch.tensor(np.load(a.release/'whole_geometry.npz')['coordinates_mm'],dtype=torch.float32,device=device)
        detector=torch.tensor(load_detector_coordinates(root/'Detector.csv',10496),dtype=torch.float32,device=device)
        variance=build_detector_position_variance(detector,0);law=ContinuousTransferLaw.load(a.release/'transfer_training_summary.json')
        B=full_rows(raw,device);blocks=[];began=time.monotonic()
        for v in range(20):
            indices=np.load(a.selection/'selections'/f'{v+1}.npy');assigned,lo,hi=partition_indices(indices,rank,world)
            rows=np.load(a.selection/'selected_rows'/f'{v+1}.npy',mmap_mode='r');items=[]
            if v==0:event_checks=numerical_checks(np.array(rows[:32],copy=True),detector,variance,coords,B,law,geometry,'continuous_energy',rank,world)
            for off in range(lo,hi,32):
                if time.monotonic()-began>float(os.environ['DERENZO_PREPARE_SECONDS']):raise TimeoutError('Complete response preparation exceeded bound')
                block=geometry.compact(rows_for(np.array(rows[off:min(off+32,hi)],copy=True),detector,variance,coords,B,law,'continuous_energy'),v)
                mass=block.double().sum(1)
                if not bool(torch.isfinite(block).all()) or bool((block<0).any()) or bool((mass<=0).any()):raise ValueError('Do not remove any selected event with invalid response')
                minimum_mass=float(mass.min()) if minimum_mass is None else min(minimum_mass,float(mass.min()))
                event_bytes+=block.numel()*block.element_size()
                items.append(block.cpu())
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>.8*memory/plan['gpus_per_node']:raise MemoryError('Conservative node RSS budget exceeded')
                if torch.cuda.max_memory_reserved(device)>.8*torch.cuda.get_device_properties(device).total_memory:raise MemoryError('GPU response budget exceeded')
            blocks.append(items);accepted_events+=len(assigned)
            partition_proof.append(dict(view=v+1,lo=lo,hi=hi,indices_sha256=hashlib.sha256(assigned.astype('<i8').tobytes()).hexdigest()))
            print('DERENZO_ALL_EVENTS_PREPARED',rank,v+1,len(assigned),flush=True)
        del B,coords,detector,variance;torch.cuda.empty_cache();prep_seconds=time.monotonic()-began
        sf=np.fromfile(a.release/'continuous_energy_Sensi_full','<f4');compton_s=geometry.compton_sensitivity(torch.tensor(sf,device=device))
        if rank==0:compton_s.cpu().numpy().astype('<f4').tofile(a.output/'S_Compton.float32')
        channel=info['channels'][1];began=time.monotonic()
        image,h=joint_mlem(response,projection,blocks,sensitivity,compton_s,iterations,step,rank==0,
                           (lambda i,h:checkpoint(channel,i,h)) if rank==0 else None,phase_limit)
        phase_times[channel]=time.monotonic()-began
        if a.mode=='validation':
            _,(ref,rh)=compton_and_joint_mlem(response,projection,blocks,sensitivity,compton_s,10,10,save_history=rank==0)
            regression[channel]=relative(image,ref)
            if regression[channel]>1e-5 or (rank==0 and not torch.equal(h,rh)):raise ValueError('Original joint branch/history regression failed')
        publish_images(a.output,channel,image,h,geometry,rank)
    record=resource_record(rank,local,device,monitor,memory,started)
    record.update(phase_solve_seconds=phase_times,prepare_seconds=prep_seconds,operator_checks=checks,
                  regression_relative_L2=regression,accepted_events=accepted_events,partitions=partition_proof,
                  event_numerical_checks=event_checks,event_response_bytes=event_bytes,minimum_active_event_mass=minimum_mass)
    records=[None]*world;dist.all_gather_object(records,record);verify_resources(records,plan,a.allocation)
    if rank==0:
        write(a.output/'run_manifest.json',dict(passed=True,study=STUDY,dataset=a.dataset,mode=a.mode,
              iterations=iterations,save_step=step,channels=info['channels'],resources=records,plan=plan,
              plan_sha256=digest(a.plan),collection_sha256=digest(a.input/'collection.json'),
              release_sha256=digest(a.release/'release_manifest.json'),factor_manifest_sha256=spec['factor_manifest_sha256'],
              monoenergy_background='zero; no independent 440 phantom inserted into 218 data',
              initial_density=1,regularization='none',physical_calibration_claim=False,selection=str(a.selection) if a.selection else None,
              authority_sha256=digest(a.authority) if a.authority else None,authority_file=str(a.authority) if a.authority else None))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['release','input','output','allocation','plan']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--mode',choices=['selection','validation','formal'],required=True)
    p.add_argument('--plan-sha256',required=True)
    p.add_argument('--selection',type=Path);p.add_argument('--authority',type=Path);p.add_argument('--authority-sha256');a=p.parse_args()
    if a.mode=='selection':select(a)
    else:reconstruct(a)
