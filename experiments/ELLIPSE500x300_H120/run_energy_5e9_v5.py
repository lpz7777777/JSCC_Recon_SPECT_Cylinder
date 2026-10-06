"""Existing legacy 5e9 comparison with the unchanged v5 response helpers.
Validation mode is explicitly 10/save10; formal mode requires pinned successful
validation authority and is explicitly 2000/save50. Original pilot/core unchanged.
"""
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
from energy_5e9_v5_contract import load_contract,write_checkpoint,validate_transport_collection
from verify_energy_5e9_v5 import validate_authority


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
    p.add_argument('--model',choices=('angular','continuous_energy'),default='angular')
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
            if time.monotonic()-start>1500:raise TimeoutError('Bounded startup/response generation exceeded 25 minutes')
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
        print('ENERGY_5E9_V5_PREPARE',a.model,rank,view+1,len(selected),flush=True)
    del B,rawB,detector,variance,coords;torch.cuda.empty_cache()
    total=torch.tensor(accepted,dtype=torch.int64,device=device);dist.all_reduce(total,op=dist.ReduceOp.SUM)
    if total.cpu().tolist()!=cfg['events_per_view'] or int(total.sum())!=cfg['accepted_events']:raise ValueError('Full event/rank closure failed')
    sensitivity_path=root/(a.model+'_Sensi_full')
    full_s=np.fromfile(sensitivity_path,dtype='<f4')
    if len(full_s)!=132040 or not np.isfinite(full_s).all() or np.any(full_s<=0):raise ValueError('Invalid matched full-circle S')
    sensitivity=geometry.compton_sensitivity(torch.tensor(full_s,device=device))
    prepare_seconds=time.monotonic()-prepare_started; solve_started=time.monotonic()
    def checkpoint(iteration,history_d,history_j):
        write_checkpoint(a.output,a.model,iteration,history_d,history_j,geometry,digest(a.contract))
        write(a.output/'progress.json',dict(model=a.model,iteration=iteration,elapsed_seconds=time.monotonic()-start,
            contract_sha256=digest(a.contract),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    (id_,hd),(ij,hj)=compton_and_joint_mlem(response,projection,blocks,single_sensitivity,sensitivity,
        a.iterations,a.save_step,save_history=rank==0,progress_label='energy_5e9_v5_'+a.model,
        checkpoint_callback=checkpoint if a.mode=='formal' and rank==0 else None)
    solve_seconds=time.monotonic()-solve_started
    for name,image,history in [('440_ComptonOnly',id_,hd),('440_SinglePlusCompton',ij,hj)]:
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
        elapsed_seconds=time.monotonic()-start,prepare_seconds=prepare_seconds,solve_seconds=solve_seconds,numerical_checks=check)
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
            transport=validate_transport_collection(collection),
            authority_file=str(a.authority) if a.mode=='formal' else None,
            authority_sha256=a.authority_sha256 if a.mode=='formal' else None))
    dist.barrier();dist.destroy_process_group()
    print('ENERGY_5E9_V5_FORMAL_ENTRY_FINISHED',a.model,rank,flush=True)


if __name__=='__main__':main()
