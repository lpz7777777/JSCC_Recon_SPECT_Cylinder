"""Actual 5e10, 8x3 GPUs with bounded node-local response replay, only 440 single/218 corrected/440 Compton."""
import argparse, hashlib, os, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from jscc_5e10_common import *
from jscc_5e10_streaming_contract import load_contract,write_checkpoint,verify_topology
from jscc_5e10_streaming_runtime import setup,resource_record,gather_resources
from jscc_5e10_compton_mlem import compton_mlem
from jscc_5e10_streaming_cache import CacheWriter,DiskBlocks,attach_cache,POLICY
from jscc_5e10_streaming_probe import local_storage
from run_energy_preflight_v5 import (rows_for,numerical_checks,partition_indices,validate_whole_geometry,
    split_bins,load_matrix,local_rows,full_rows,ActiveGeometry,ViewResponse,ContinuousTransferLaw,
    load_detector_coordinates,build_detector_position_variance,NAMES)
from torch_active_operator import forward_project,single_mlem,compton_and_joint_mlem
from single_checkpoint_mlem import single_mlem_checkpointed
from run_reconstruction import collect_response

def relative(x,y):return float(torch.linalg.vector_norm((x-y).double())/torch.linalg.vector_norm(y.double()).clamp_min(1e-30))

def operator_checks(response):
    """Every view and local detector row against full-grid float64 scatter/gather."""
    geo=response.geometry;rows=response.full_rows;device=rows.device
    x=(1+torch.arange(geo.active_count,device=device,dtype=torch.float64)/geo.active_count)[:,None]
    u=(1+torch.arange(len(rows),device=device,dtype=torch.float64)/max(len(rows),1))[:,None]
    base=rows.double();forward=[];adjoint=[];sensitivity=torch.zeros_like(x)
    for v in range(20):
        full=torch.zeros((geo.full_count,1),device=device,dtype=torch.float64)
        full.index_add_(0,geo.indices[v],x*geo.fraction.double()[:,None])
        ref=base@full;actual=response.matrix(v)@x.float()
        forward.append(relative(actual.double(),ref))
        ref_adj=(base.T@u)[geo.indices[v]]*geo.fraction.double()[:,None]
        actual_adj=response.matrix(v).T@u.float();adjoint.append(relative(actual_adj.double(),ref_adj))
        sensitivity+=(base.sum(0)[geo.indices[v]]*geo.fraction.double())[:,None]/20
    dist.all_reduce(sensitivity)
    actual_s=response.sensitivity().double();dist.all_reduce(actual_s)
    checks=dict(views=20,local_detector_rows=len(rows),full_points=132040,active_points=78920,
        forward_relative_L2=forward,adjoint_relative_L2=adjoint,sensitivity_relative_L2=relative(actual_s,sensitivity),threshold=1e-5)
    if max(forward+adjoint+[checks['sensitivity_relative_L2']])>1e-5:raise ValueError('All-view/all-row forward/transpose/S equivalence failed')
    return checks

def publish_images(output,channel,image,history,geometry,rank):
    if rank:return
    stage=output/('.publish_'+channel);stage.mkdir(exist_ok=False)
    collect_response(stage,channel,image,history,geometry,0)
    for f in stage.iterdir():
        with f.open('r+b') as s:os.fsync(s.fileno())
        target=output/f.name
        if target.exists():raise FileExistsError('Never overwrite published final output')
        os.replace(f,target)
    stage.rmdir()

def main():
    p=argparse.ArgumentParser()
    for n in ('contract','input-root','factors','output','allocation'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--mode',choices=('validation','formal'),required=True)
    p.add_argument('--authority',type=Path);p.add_argument('--authority-sha256');a=p.parse_args()
    iterations,save_step=execution_policy(a.mode);root=a.contract.parent;started=time.monotonic()
    cfg=load_contract(a.contract,a.mode,iterations,save_step)
    if a.mode=='formal':
        if digest(a.authority)!=a.authority_sha256:raise ValueError('Full-input authority SHA differs')
        authority=read(a.authority)
        if not authority['passed'] or authority['contract_sha256']!=digest(a.contract):raise ValueError('Actual current input validation authority required')
        for n,sha in authority['evidence_sha256'].items():
            if digest(Path(authority['result'])/n)!=sha:raise ValueError('Accepted validation evidence changed')
    rank,world,local,device=setup();error=[None]
    if rank==0:
        try:
            validate_collection(read(a.input_root/'collection.json'));verify_files(a.input_root,cfg['input_sha256'])
            verify_files(a.factors,cfg['factor_payload_sha256']);a.output.mkdir(exist_ok=False)
        except Exception as e:error=[str(e)]
    dist.broadcast_object_list(error,src=0)
    if error[0]:raise ValueError(error[0])
    validate_whole_geometry(root/'whole_geometry.npz',cfg['whole_geometry_sha256'])
    geometry=ActiveGeometry.from_npz(root/'whole_geometry.npz',device);begin,end=split_bins(10496,rank,world)
    rawB=load_matrix(a.factors,NAMES['A440'],132040,10496)
    response=ViewResponse(local_rows(rawB,begin,end,device),geometry,'none')
    sensitivity440=response.sensitivity();dist.all_reduce(sensitivity440)
    phase_times={};single_checks={};all_operator_checks={};contract_sha=digest(a.contract)
    limit=float(os.environ['JSCC_PHASE_SECONDS']);phase_start=time.monotonic()
    def progress(phase,iteration):
        if rank==0:write(a.output/'progress.json',dict(phase=phase,iteration=iteration,limit=iterations,contract_sha256=contract_sha,updated_epoch=time.time()))
    def checkpoint(phase,i,h):
        write_checkpoint(a.output,phase,i,{PHASE_CHANNELS[phase][0]:h[-1]},geometry,contract_sha,a.mode);progress(phase,i)
    def observations(e):
        values=np.loadtxt(a.input_root/f'projection_{e}.csv',delimiter=',',dtype=np.float32)
        if values.shape!=(20,10496) or not np.isfinite(values).all() or np.any(values<0):raise ValueError('Complete observations differ')
        return torch.tensor(np.ascontiguousarray(values[:,begin:end].T),device=device)
    if a.mode=='validation':all_operator_checks['A440']=operator_checks(response)
    projection440=observations(440);progress('440_single',0);phase_start=time.monotonic()
    image440,h440=single_mlem_checkpointed(response,projection440,sensitivity440,iterations,save_step,
        save_history=rank==0,checkpoint_callback=(lambda i,h:checkpoint('440_single',i,h)) if rank==0 else None,
        progress_label='JSCC5e10_440',phase_limit_seconds=limit)
    phase_times['440_single']=time.monotonic()-phase_start
    if a.mode=='validation':
        ref,rh=single_mlem(response,projection440,sensitivity440,10,10,save_history=rank==0)
        single_checks['440_SinglePhoton']=relative(image440,ref)
        if single_checks['440_SinglePhoton']>1e-5 or (rank==0 and not torch.equal(h440,rh)):raise ValueError('Original single MLEM/history differs')
        del ref,rh
    publish_images(a.output,CHANNELS[0],image440,h440,geometry,rank)
    cross_raw=load_matrix(a.factors,NAMES['C440to218'],132040,10496)
    cross=ViewResponse(local_rows(cross_raw,begin,end,device),geometry,'none')
    if a.mode=='validation':all_operator_checks['C440to218']=operator_checks(cross)
    predicted=forward_project(cross,image440)
    if not bool(torch.isfinite(predicted).all()) or bool((predicted<0).any()):raise ValueError('Invalid fixed additive background')
    del cross,cross_raw;torch.cuda.empty_cache()
    raw218=load_matrix(a.factors,NAMES['A218'],132040,10496)
    response218=ViewResponse(local_rows(raw218,begin,end,device),geometry,'none')
    sensitivity218=response218.sensitivity();dist.all_reduce(sensitivity218)
    if a.mode=='validation':all_operator_checks['A218']=operator_checks(response218)
    projection218=observations(218);phase_start=time.monotonic();progress('218_corrected',0)
    image218,h218=single_mlem_checkpointed(response218,projection218,sensitivity218,iterations,save_step,
        additive_background=predicted,save_history=rank==0,
        checkpoint_callback=(lambda i,h:checkpoint('218_corrected',i,h)) if rank==0 else None,
        progress_label='JSCC5e10_218',phase_limit_seconds=limit)
    phase_times['218_corrected']=time.monotonic()-phase_start
    if a.mode=='validation':
        ref,rh=single_mlem(response218,projection218,sensitivity218,10,10,predicted,rank==0)
        single_checks['218_SinglePhoton_CrossTalkCorrected']=relative(image218,ref)
        if single_checks['218_SinglePhoton_CrossTalkCorrected']>1e-5 or (rank==0 and not torch.equal(h218,rh)):raise ValueError('Original corrected MLEM/history differs')
        del ref,rh
    publish_images(a.output,CHANNELS[1],image218,h218,geometry,rank)
    padded=torch.zeros(((10496+world-1)//world,20),device=device);padded[:end-begin]=predicted
    gathered=[torch.empty_like(padded) for _ in range(world)];dist.all_gather(gathered,padded)
    if rank==0:
        full=np.concatenate([x[:split_bins(10496,k,world)[1]-split_bins(10496,k,world)[0]].cpu().numpy() for k,x in enumerate(gathered)])
        full.astype('<f4').tofile(a.output/'PredictedCntStat_218_From440.float32')
        sensitivity440.cpu().numpy().astype('<f4').tofile(a.output/'S_440.float32')
        sensitivity218.cpu().numpy().astype('<f4').tofile(a.output/'S_218.float32')
    del raw218,response218,projection218,predicted,padded,gathered;torch.cuda.empty_cache()
    prep_started=time.monotonic();progress('compton_response',0)
    B=full_rows(rawB,device);g=np.load(root/'whole_geometry.npz')
    coords=torch.tensor(g['coordinates_mm'],dtype=torch.float32,device=device)
    detector=torch.tensor(load_detector_coordinates(a.factors/NAMES['A440']/'Detector.csv',10496),device=device,dtype=torch.float32)
    variance=build_detector_position_variance(detector,0);law=ContinuousTransferLaw.load(root/'transfer_training_summary.json')
    blocks=[];partitions=[];accepted=[];minimum_mass=float('inf');floor_events=0;check=None
    storage=local_storage('/tmp')
    cache_root=Path(storage['root'])/'jscc_geant4_5e10_streaming'/contract_sha[:16]/f'rank{rank:02d}'
    cache_manifest_path=cache_root/'cache_manifest.json'
    cache_identity=dict(policy=POLICY,contract_sha256=contract_sha,rank=rank,node=os.environ['SLURMD_NODENAME'])
    existing_manifest=None
    if a.mode=='formal':
        # Reuse only the fully accepted validation's exact per-rank disk cache.
        source=Path(os.environ['JSCC_VALIDATION_RESULT'])/f'cache_rank{rank:02d}.json'
        existing_manifest,blocks=attach_cache(cache_manifest_path,cache_identity)
        if digest(source)!=digest(cache_manifest_path):raise ValueError('Accepted validation/cache manifest SHA differs')
        partitions=existing_manifest['partitions'];accepted=[x['events'] for x in existing_manifest['views']]
        minimum_mass=existing_manifest['minimum_active_event_mass'];floor_events=existing_manifest['initial_forward_floor_events']
        check=existing_manifest['numerical_checks']
    else:
        cache_root.mkdir(parents=True,exist_ok=False)
    cache_receipts=[]
    for view in (range(20) if existing_manifest is None else []):
        indices=np.load(root/'selections'/f'{view+1}.npy');selected,lo,hi=partition_indices(indices,rank,world)
        all_rows=np.load(root/'selected_rows'/f'{view+1}.npy',mmap_mode='r');raw=np.array(all_rows[lo:hi],copy=True)
        if len(raw)!=len(selected):raise ValueError('This view all accepted event rows required')
        # Every rank checks the same sample before taking its disjoint rows.
        # The unchanged numerical oracle all-reduces rank slices of this sample.
        if view==0:check=numerical_checks(np.array(all_rows[:32],copy=True),detector,variance,coords,B,law,geometry,'continuous_energy',rank,world)
        writer=CacheWriter(cache_root/f'view{view+1:02d}.cache',78920)
        for offset in range(0,len(raw),32):
            if time.monotonic()-prep_started>float(os.environ['JSCC_PREPARE_SECONDS']):raise TimeoutError('Bounded complete response preparation exceeded')
            rows=rows_for(raw[offset:offset+32],detector,variance,coords,B,law,'continuous_energy')
            compact=geometry.compact(rows,view);mass=compact.double().sum(1)
            if not bool(torch.isfinite(mass).all()) or bool((mass<=0).any()):raise ValueError('Accepted event has zero/invalid active support; do not silently remove')
            minimum_mass=min(minimum_mass,float(mass.min()));floor_events+=int((mass<1e-12).sum())
            writer.append(compact);del rows,compact,mass
            if torch.cuda.max_memory_reserved(device)>.8*torch.cuda.get_device_properties(device).total_memory:raise MemoryError('GPU preparation margin fails')
            if offset%2048==0:
                import resource
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>.8*int(os.environ['JSCC_HOST_BYTES_NODE'])/3:
                    raise MemoryError('Conservative three-rank node RSS operational guard fails')
        receipt=writer.finish();cache_receipts.append(receipt);blocks.append(DiskBlocks(receipt));accepted.append(len(raw))
        partitions.append(dict(view=view+1,first_selected_position=lo,last_selected_position_exclusive=hi,events=len(raw),
            original_rows_sha256=hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest()))
        print('JSCC_EVENT_PREPARED',rank,view+1,len(raw),flush=True)
    if existing_manifest is None:
        existing_manifest=dict(cache_identity,complete=True,views=cache_receipts,partitions=partitions,
            minimum_active_event_mass=minimum_mass,initial_forward_floor_events=floor_events,numerical_checks=check)
        write(cache_manifest_path,existing_manifest)
    write(a.output/f'cache_rank{rank:02d}.json',existing_manifest)
    del B,rawB,detector,variance,coords;torch.cuda.empty_cache()
    counts=torch.tensor(accepted,dtype=torch.int64,device=device);dist.all_reduce(counts)
    if counts.cpu().tolist()!=cfg['events_per_view']:raise ValueError('All accepted event ranks/views do not close')
    sf=np.fromfile(root/'continuous_energy_Sensi_full','<f4')
    if sf.shape!=(132040,) or not np.isfinite(sf).all() or np.any(sf<0):raise ValueError('Full matched Compton sensitivity required')
    sensitivity=geometry.compton_sensitivity(torch.tensor(sf,device=device))
    if rank==0:sensitivity.cpu().numpy().astype('<f4').tofile(a.output/'S_Compton.float32')
    prepare_seconds=time.monotonic()-prep_started;solve_started=time.monotonic();progress('440_compton',0)
    imageC,hC=compton_mlem(blocks,sensitivity,iterations,save_step,rank==0,
        (lambda i,h:checkpoint('440_compton',i,h)) if rank==0 else None,limit)
    phase_times['440_compton']=time.monotonic()-solve_started;compton_regression=None
    if a.mode=='validation':
        # Only ten iterations of the unchanged historical implementation, as numerical reference.
        (ref,rh),(_,jh)=compton_and_joint_mlem(response,projection440,blocks,sensitivity440,sensitivity,10,10,save_history=rank==0)
        compton_regression=relative(imageC,ref)
        if compton_regression>1e-5 or (rank==0 and not torch.equal(hC,rh)):raise ValueError('Standalone original Compton branch/history differs')
        del ref,rh,jh
    publish_images(a.output,CHANNELS[2],imageC,hC,geometry,rank)
    record=resource_record(rank,local,device,started)
    record.update(accepted_events=sum(accepted),partitions=partitions,event_bytes_cpu_bound=32*78920*4,event_bytes_disk_dense=sum(x['raw_bytes'] for x in existing_manifest['views']),
        event_bytes_disk_encoded=sum(x['encoded_bytes'] for x in existing_manifest['views']),
        cache_reused_from_validation=a.mode=='formal',cache_read_seconds=sum(x.read_seconds for x in blocks),
        cache_read_passes=[x.read_passes for x in blocks],local_storage=storage,
        minimum_active_event_mass=minimum_mass,initial_forward_floor_events=floor_events,numerical_checks=check,
        phase_solve_seconds=phase_times,prepare_seconds=prepare_seconds,single_regression_relative_L2=single_checks,
        original_compton_regression_relative_L2=compton_regression,single_operator_checks=all_operator_checks)
    resources=gather_resources(record);verify_topology(resources,a.allocation)
    if rank==0:
        write(a.output/'run_manifest.json',dict(study=STUDY,mode=a.mode,model='continuous_energy',iterations=iterations,save_step=save_step,
            pixels_active=78920,pixels_full=132040,channels=list(CHANNELS),world_size=24,nodes=8,gpus_per_node=3,response_storage_policy=POLICY,
            accepted_compton_events=cfg['accepted_events'],events_per_view=cfg['events_per_view'],resources=resources,
            contract_sha256=digest(a.contract),helper_sha256=cfg['files'],input_sha256=cfg['input_sha256'],
            factor_payload_sha256=cfg['factor_payload_sha256'],factor_manifest_sha256=cfg['factor_manifest_sha256'],
            geometry_sha256=cfg['whole_geometry_sha256'],sensitivity_sha256=cfg['files']['continuous_energy_Sensi_full'],
            initial_density=1.,regularization='none',joint_solver_enabled=False,event_policy='legacy',actual_primary_photons=TOTAL,
            cross_prediction_source='440_SinglePhoton_final',authority_sha256=a.authority_sha256,authority_file=str(a.authority) if a.authority else None))
        progress('complete',iterations)
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
