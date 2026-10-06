"""Full six-output continuous-energy legacy 5e9 reconstruction: 10000/save50."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
try:
    import resource
except ImportError:
    resource=None
import time
import numpy as np
import torch
import torch.distributed as dist
from run_energy_preflight_v5 import (write,settings,partition_indices,validate_whole_geometry,
    prepare_frozen,rows_for,numerical_checks,split_bins,load_matrix,local_rows,full_rows,
    collect_response,digest,ActiveGeometry,ViewResponse,compton_and_joint_mlem,
    ContinuousTransferLaw,load_detector_coordinates,build_detector_position_variance,validate,NAMES)
from energy_full10000_v5_contract import load_contract,write_checkpoint,CHANNELS
from energy_5e9_v5_contract import validate_transport_collection
from single_checkpoint_mlem import single_mlem_checkpointed
from torch_active_operator import single_mlem,forward_project
from verify_energy_full10000_v5 import validate_authority


def setup(backend):
    """Same rank/device layout, bounded 15-minute collectives for complete SHA reads.

    Rank zero streams all three full matrices before broadcasting acceptance.
    The original 5-minute timeout is shorter than cold shared-storage SHA reads.
    Response computation and MLEM are unchanged; total startup remains 25 minutes.
    """
    if backend!='nccl':raise ValueError('Explicit NCCL topology required')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if local!=0:raise ValueError('Exactly one GPU per node')
    torch.cuda.set_device(local);device=torch.device('cuda:0')
    dist.init_process_group(backend=backend,init_method='env://',timeout=timedelta(minutes=15))
    return rank,world,device


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('contract','factors','input-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',choices=('continuous_energy',),default='continuous_energy')
    p.add_argument('--mode',choices=('validation','formal'),required=True)
    p.add_argument('--iterations',type=int,required=True);p.add_argument('--save-step',type=int,required=True)
    p.add_argument('--authority',type=Path);p.add_argument('--authority-sha256')
    a=p.parse_args();start=time.monotonic();torch.set_grad_enabled(False)
    root=a.contract.parent;cfg=load_contract(a.contract,a.mode,a.iterations,a.save_step)
    if a.mode=='formal':validate_authority(a.authority,a.authority_sha256,a.contract)
    gate=json.loads((root/'calibration/calibration_gate.json').read_text())
    if gate['status']!='PASSED' or gate['failures']:
        raise ValueError('Independent diagnostic gate is not passed')
    validate_whole_geometry(root/'whole_geometry.npz',cfg['whole_geometry_sha256'])
    if os.environ.get('NCCL_SOCKET_IFNAME')!='bond0':raise ValueError('NCCL must use bond0')
    rank,world,device=setup('nccl')
    if world!=cfg['nodes']:raise ValueError('Only the frozen 8-node, one-GPU topology')
    error=[None]
    if rank==0:
        try:
            a.output.mkdir(parents=True,exist_ok=False)
            for relative,sha in cfg['input_sha256'].items():
                if digest(a.input_root/relative)!=sha:raise ValueError('Original NEMA input differs: '+relative)
            collection=json.loads((a.input_root/'collections/NEMA_Body_H60_5e9.json').read_text())
            validate_transport_collection(collection)
            validate(a.factors)
            for relative,sha in cfg['factor_payload_sha256'].items():
                if digest(a.factors/relative)!=sha:raise ValueError('Full matrix/geometry SHA differs: '+relative)
            for folder,sha in cfg['factor_manifest_sha256'].items():
                if digest(a.factors/folder/'factor_manifest.json')!=sha:raise ValueError('Frozen Factor manifest differs')
        except Exception as ex:error=[str(ex)]
    dist.broadcast_object_list(error,src=0)
    if error[0]:raise ValueError('Original input validation failed: '+error[0])
    geometry=ActiveGeometry.from_npz(root/'whole_geometry.npz',device)
    begin,end=split_bins(10496,rank,world)
    rawB=load_matrix(a.factors,NAMES['A440'],132040,10496)
    response=ViewResponse(local_rows(rawB,begin,end,device),geometry,'none')
    single_sensitivity=response.sensitivity();dist.all_reduce(single_sensitivity,op=dist.ReduceOp.SUM)
    counts=np.loadtxt(a.input_root/'CntStat/440keV_RotateNum20_Geant4JSCC/CntStat_NEMA_Body_H60_5e9.csv',delimiter=',',dtype=np.float32)
    if counts.shape!=(20,10496) or not np.isfinite(counts).all() or np.any(counts<0):raise ValueError('440 counts differ')
    projection=torch.from_numpy(np.ascontiguousarray(counts[:,begin:end].T)).to(device)
    phase_seconds=float(os.environ.get('ENERGY_FULL_V5_PHASE_SECONDS','2700'))
    phase_times={}; single_regression={}; prefix_regression={}
    contract_sha=digest(a.contract)
    def stage_guard():
        memory=int(os.environ.get('ABLATION_HOST_ALLOCATED_BYTES','0'))
        rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
        if memory<=0 or rss>.8*memory or torch.cuda.max_memory_reserved()>.8*torch.cuda.get_device_properties(device).total_memory:
            raise RuntimeError('Actual six-output resource margin fails')
    def publish(phase,iteration,frames):
        write_checkpoint(a.output,phase,iteration,frames,geometry,contract_sha,a.mode)
        write(a.output/'progress.json',dict(phase=phase,iteration=iteration,elapsed_seconds=time.monotonic()-start,
            contract_sha256=contract_sha,peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    def single_checkpoint(iteration,history):
        publish('440_single',iteration,{'440_SinglePhoton':history[-1]})
    if rank==0:
        print('ENERGY_FULL_PHASE 440_single',flush=True)
        write(a.output/'progress.json',dict(phase='440_single',iteration=0,contract_sha256=contract_sha))
    stage_start=time.monotonic()
    image440,h440=single_mlem_checkpointed(response,projection,single_sensitivity,a.iterations,a.save_step,
        save_history=rank==0,checkpoint_callback=single_checkpoint if rank==0 else None,
        progress_label='full_440_single',phase_limit_seconds=phase_seconds)
    phase_times['440_single']=time.monotonic()-stage_start
    if a.mode=='validation':
        original,original_history=single_mlem(response,projection,single_sensitivity,10,10,save_history=rank==0)
        value=float(torch.linalg.vector_norm((image440-original).double())/torch.linalg.vector_norm(original.double()))
        if not np.isfinite(value) or value>1e-5:raise ValueError('Single checkpoint loop differs from original MLEM')
        single_regression['440_SinglePhoton']=value
        if rank==0 and not torch.equal(h440,original_history):raise ValueError('Original single history differs')
        del original,original_history
    collect_response(a.output,'440_SinglePhoton',image440,h440,geometry,rank)
    stage_guard()
    if rank==0:
        print('ENERGY_FULL_PHASE 218_corrected',flush=True)
        write(a.output/'progress.json',dict(phase='218_corrected',iteration=0,contract_sha256=contract_sha))
    cross_raw=load_matrix(a.factors,NAMES['C440to218'],132040,10496)
    cross=ViewResponse(local_rows(cross_raw,begin,end,device),geometry,'none')
    predicted=forward_project(cross,image440)
    if not bool(torch.isfinite(predicted).all()) or bool((predicted<0).any()):raise ValueError('Invalid fixed 440-to-218 prediction')
    del cross,cross_raw;torch.cuda.empty_cache()
    raw218=load_matrix(a.factors,NAMES['A218'],132040,10496)
    response218=ViewResponse(local_rows(raw218,begin,end,device),geometry,'none')
    sensitivity218=response218.sensitivity();dist.all_reduce(sensitivity218,op=dist.ReduceOp.SUM)
    counts218=np.loadtxt(a.input_root/'CntStat/218keV_RotateNum20_Geant4JSCC/CntStat_NEMA_Body_H60_5e9.csv',delimiter=',',dtype=np.float32)
    if counts218.shape!=(20,10496) or not np.isfinite(counts218).all() or np.any(counts218<0):raise ValueError('218 counts differ')
    projection218=torch.from_numpy(np.ascontiguousarray(counts218[:,begin:end].T)).to(device)
    def corrected_checkpoint(iteration,history):
        publish('218_corrected',iteration,{'218_SinglePhoton_CrossTalkCorrected':history[-1],
            '440SinglePlus218Single':h440[iteration//a.save_step-1]+history[-1]})
    stage_start=time.monotonic()
    image218,h218=single_mlem_checkpointed(response218,projection218,sensitivity218,a.iterations,a.save_step,
        additive_background=predicted,save_history=rank==0,
        checkpoint_callback=corrected_checkpoint if rank==0 else None,
        progress_label='full_218_corrected',phase_limit_seconds=phase_seconds)
    phase_times['218_corrected']=time.monotonic()-stage_start
    if a.mode=='validation':
        original,original_history=single_mlem(response218,projection218,sensitivity218,10,10,predicted,rank==0)
        value=float(torch.linalg.vector_norm((image218-original).double())/torch.linalg.vector_norm(original.double()))
        if not np.isfinite(value) or value>1e-5:raise ValueError('Corrected checkpoint loop differs from original MLEM')
        single_regression['218_SinglePhoton_CrossTalkCorrected']=value
        if rank==0 and not torch.equal(h218,original_history):raise ValueError('Original corrected history differs')
        del original,original_history
    collect_response(a.output,'218_SinglePhoton_CrossTalkCorrected',image218,h218,geometry,rank)
    collect_response(a.output,'440SinglePlus218Single',image440+image218,h440+h218 if rank==0 else None,geometry,rank)
    padded=torch.zeros(((10496+world-1)//world,20),device=device);padded[:end-begin]=predicted
    gathered=[torch.empty_like(padded) for _ in range(world)];dist.all_gather(gathered,padded)
    if rank==0:
        predicted_all=np.concatenate([item[:split_bins(10496,k,world)[1]-split_bins(10496,k,world)[0]].cpu().numpy()
            for k,item in enumerate(gathered)]).astype('<f4')
        predicted_all.tofile(a.output/'PredictedCntStat_218_From440.float32')
        from energy_5e9_v5_contract import sync_file
        sync_file(a.output/'PredictedCntStat_218_From440.float32')
    del response218,raw218,projection218,sensitivity218,predicted,padded,gathered
    torch.cuda.empty_cache();stage_guard()
    response_started=time.monotonic()
    if rank==0:
        print('ENERGY_FULL_PHASE compton_response',flush=True)
        write(a.output/'progress.json',dict(phase='compton_response',iteration=0,contract_sha256=contract_sha))
    B=full_rows(rawB,device)
    geo=np.load(root/'whole_geometry.npz');coords=torch.tensor(geo['coordinates_mm'],dtype=torch.float32,device=device)
    detector=torch.tensor(load_detector_coordinates(a.factors/NAMES['A440']/'Detector.csv',10496),device=device,dtype=torch.float32)
    variance=build_detector_position_variance(detector,0.)
    law=ContinuousTransferLaw.load(root/'transfer_training_summary.json')
    blocks=[];partitions=[];accepted=[];check=None
    minimum_active_mass=float('inf');initial_floor_events=0
    prepare_started=time.monotonic()
    for view in range(20):
        list_path=a.input_root/f'List/218-440keV_RotateNum20_Geant4JSCC/List_NEMA_Body_H60_5e9/{view+1}.csv'
        raw=np.loadtxt(list_path,delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
        indices=np.load(root/f'selections/{view+1}.npy')
        selected,lo,hi=partition_indices(indices,rank,world)
        if len(indices)!=cfg['events_per_view'][view] or indices[-1]>=len(raw):raise ValueError('Frozen selection closure differs')
        if view==0:check=numerical_checks(raw[indices],detector,variance,coords,B,law,geometry,a.model,rank,world)
        items=[]
        for offset in range(0,len(selected),32):
            if time.monotonic()-response_started>1500:raise TimeoutError('Bounded startup/response generation exceeded 25 minutes')
            if torch.cuda.memory_reserved()>.6*torch.cuda.get_device_properties(device).total_memory:torch.cuda.empty_cache()
            rows=rows_for(raw[selected[offset:offset+32]],detector,variance,coords,B,law,a.model)
            if a.model=='angular' and bool((1/rows.square().sum(1)<1.).any()):
                raise ValueError('Frozen event fails original effective-support threshold; no silent event removal')
            compact=geometry.compact(rows,view)
            mass=compact.double().sum(1)
            invalid_mass=~torch.isfinite(mass)|(mass<=0)
            if bool(invalid_mass.any()):
                write(a.output/f'response_support_hold_rank{rank}.json',dict(model=a.model,view=view+1,
                    original_rows=selected[offset:offset+32][invalid_mass.cpu().numpy()].tolist(),
                    reason='Frozen event has zero response in actual active basis; cannot silently ignore it',
                    contract_sha256=digest(a.contract)))
                raise ValueError('Zero active-basis event response; diagnose before imaging')
            minimum_active_mass=min(minimum_active_mass,float(mass.min()))
            initial_floor_events+=int((mass<1e-12).sum())
            items.append(compact.cpu());del rows,compact,mass
            if torch.cuda.max_memory_reserved()>.75*torch.cuda.get_device_properties(device).total_memory:
                raise RuntimeError('Pilot runtime reserved peak exceeds proactive 75% limit')
        blocks.append(items);accepted.append(len(selected))
        partitions.append(dict(view=view+1,first_selected_position=lo,last_selected_position_exclusive=hi,
            events=len(selected),original_rows_sha256=hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest()))
        print('ENERGY_FULL10000_V5_PREPARE',a.model,rank,view+1,len(selected),flush=True)
    del B,rawB,detector,variance,coords;torch.cuda.empty_cache()
    total=torch.tensor(accepted,dtype=torch.int64,device=device);dist.all_reduce(total,op=dist.ReduceOp.SUM)
    if total.cpu().tolist()!=cfg['events_per_view'] or int(total.sum())!=cfg['accepted_events']:raise ValueError('Full event/rank closure failed')
    sensitivity_path=root/(a.model+'_Sensi_full')
    full_s=np.fromfile(sensitivity_path,dtype='<f4')
    if len(full_s)!=132040 or not np.isfinite(full_s).all() or np.any(full_s<=0):raise ValueError('Invalid matched full-circle S')
    sensitivity=geometry.compton_sensitivity(torch.tensor(full_s,device=device))
    prepare_seconds=time.monotonic()-prepare_started; solve_started=time.monotonic()
    def checkpoint(iteration,history_d,history_j):
        if time.monotonic()-solve_started>phase_seconds:raise TimeoutError('Bounded Compton/JSCC phase exceeded')
        publish('compton_jscc',iteration,{'440_ComptonOnly':history_d[-1],
            '440_SinglePlusCompton':history_j[-1],
            '440SingleComptonPlus218Single':history_j[-1]+h218[iteration//a.save_step-1]})
        if a.mode=='formal' and iteration==2000:
            for name,frame in [('440_ComptonOnly',history_d[-1]),('440_SinglePlusCompton',history_j[-1])]:
                ref=cfg['formal_prefix_reference'];path=Path(ref['result'])/('Image_'+name+'_active.float32')
                if digest(path)!=ref['sha256'][path.name]:raise ValueError('Delivered 2000 frame changed')
                values=np.fromfile(path,'<f4');current=frame.numpy().reshape(-1)
                value=float(np.linalg.norm(current.astype(float)-values.astype(float))/np.linalg.norm(values.astype(float)))
                if not np.isfinite(value) or value>1e-5:raise ValueError('10000 prefix differs from delivered 2000 result')
                prefix_regression[name]=value
            write(a.output/'prefix_regression_2000.json',dict(passed=True,relative_L2=prefix_regression,reference_job=1667869))
    (id_,hd),(ij,hj)=compton_and_joint_mlem(response,projection,blocks,single_sensitivity,sensitivity,
        a.iterations,a.save_step,save_history=rank==0,progress_label='energy_5e9_v5_'+a.model,
        checkpoint_callback=checkpoint if rank==0 else None)
    solve_seconds=time.monotonic()-solve_started
    phase_times['compton_jscc']=solve_seconds
    for name,image,history in [('440_ComptonOnly',id_,hd),('440_SinglePlusCompton',ij,hj),
        ('440SingleComptonPlus218Single',ij+image218,hj+h218 if rank==0 else None)]:
        collect_response(a.output,name,image,history,geometry,rank)
    reserved=torch.cuda.max_memory_reserved();gpu_total=torch.cuda.get_device_properties(device).total_memory
    if resource is None:raise RuntimeError('Actual Linux resource accounting required')
    rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    memory=int(os.environ.get('ABLATION_HOST_ALLOCATED_BYTES','0'))
    record=dict(rank=rank,node=os.environ.get('SLURMD_NODENAME'),accepted_events=sum(accepted),
        partitions=partitions,host_peak_rss_bytes=rss,host_allocated_bytes=memory,
        peak_reserved_bytes=reserved,total_device_bytes=gpu_total,
        event_bytes_cpu=sum(x.numel()*x.element_size() for items in blocks for x in items),
        minimum_active_event_mass=minimum_active_mass,initial_forward_floor_events=initial_floor_events,
        elapsed_seconds=time.monotonic()-start,prepare_seconds=prepare_seconds,solve_seconds=solve_seconds,numerical_checks=check,phase_solve_seconds=phase_times,single_regression_relative_L2=single_regression)
    if memory<=0 or reserved>.8*gpu_total or rss>.8*memory:raise RuntimeError('Actual pilot resource margin fails')
    resources=[None]*world;dist.all_gather_object(resources,record)
    if len({r['node'] for r in resources})!=world:raise ValueError('Distinct rank/node allocation failed')
    if rank==0:
        write(a.output/'run_manifest.json',dict(study=cfg['study'],model=a.model,mode=a.mode,iterations=a.iterations,save_step=a.save_step,
            world_size=world,pixels_active=78920,pixels_full=132040,accepted_compton_events=cfg['accepted_events'],
            accepted_compton_events_per_view=total.cpu().tolist(),resources=resources,
            contract_sha256=digest(a.contract),geometry_sha256=cfg['whole_geometry_sha256'],
            sensitivity_sha256=digest(sensitivity_path),input_sha256=cfg['input_sha256'],
            factor_manifest_sha256=cfg['factor_manifest_sha256'],source_sha256=digest(__file__),
            factor_payload_sha256=cfg['factor_payload_sha256'],
            initial_density=1.,algorithm='unchanged torch_active_operator JSCC MLEM',event_policy='legacy',
            new_photons=0,new_fine_A_matrices=0,calibration_release=cfg['calibration_release'],
            actual_primary_gamma=5_000_000_000,workers=200,views=20,
            channels=list(CHANNELS),cross_prediction_source=cfg['cross_prediction_source'],
            prefix_regression_relative_L2=prefix_regression,helper_sha256=cfg['files'],
            transport=validate_transport_collection(collection),
            authority_file=str(a.authority) if a.mode=='formal' else None,
            authority_sha256=a.authority_sha256 if a.mode=='formal' else None))
    dist.barrier();dist.destroy_process_group()
    print('ENERGY_FULL10000_V5_FORMAL_ENTRY_FINISHED',a.model,rank,flush=True)


if __name__=='__main__':main()
