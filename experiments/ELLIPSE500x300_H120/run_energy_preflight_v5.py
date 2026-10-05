"""Isolated, complete-event 10-iteration pilot; no formal imaging option.

Both models consume the exact frozen R1 indices. Candidate rows are the same
full-circle probability proxy used to generate their S, then compacted once.
The original production entry and MLEM solver remain unchanged.
"""
import argparse
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

from run_reconstruction import (setup,split_bins,load_matrix,local_rows,full_rows,
    collect_response,digest)
from torch_active_operator import ActiveGeometry,ViewResponse,compton_and_joint_mlem
from compton_event_response import (ComptonEventSettings,prepare_compton_events,
    build_detector_position_variance,build_compton_cone_weights)
from compton_energy_probability_v5 import ContinuousTransferLaw,normalized_proxy_response
from compton_sparse_ops import build_compton_sparse_projector,materialize_sparse_event_rows_to_fine
from process_list_plane_sparse import get_compton_backproj_list_single_sparse
from detector_csv import load_detector_coordinates
from validate_factors import validate,NAMES


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def settings():
    return ComptonEventSettings(.440,.13*np.sqrt(.511/.440),
        2*.440**2/(.511+2*.440)-.001,.05,.350,geometry_mode='stable_float64')


def partition_indices(indices,rank,world):
    indices=np.asarray(indices)
    if (world<=0 or not 0<=rank<world or indices.ndim!=1 or indices.dtype.kind not in 'iu' or
        np.any(indices[1:]<=indices[:-1]) or np.any(indices<0)):
        raise ValueError('Frozen sorted unique integer indices required')
    start=len(indices)*rank//world;stop=len(indices)*(rank+1)//world
    return indices[start:stop],start,stop


def validate_whole_geometry(path,expected):
    if digest(path)!=expected:raise ValueError('Whole-cell geometry SHA differs')
    g=np.load(path);xyz=g['coordinates_mm'];active=g['active_indices'];f=g['ellipse_fraction']
    mask=((xyz[:,0]/250)**2+(xyz[:,1]/150)**2<=1+1e-12)
    if len(xyz)!=132040 or len(active)!=78920 or not np.array_equal(f,mask.astype(float)):
        raise ValueError('Frozen whole-polar-cell basis differs')
    if not np.array_equal(active,np.flatnonzero(mask)) or g['inverse_rotation'].shape!=(132040,20):
        raise ValueError('Whole-cell columns/rotations differ')
    return g


def prepare_frozen(raw,detector,variance,device):
    prepared,_=prepare_compton_events(torch.tensor(raw,device=device),settings(),detector,
        variance,variance,input_energies_already_smeared=True)
    if prepared is None or prepared.count!=len(raw):
        raise ValueError('Frozen accepted event failed preparation; do not silently drop')
    return prepared


def rows_for(raw,detector,variance,coords,B,law,model):
    prepared=prepare_frozen(raw,detector,variance,B.device)
    if model=='angular':
        rows=build_compton_cone_weights(prepared,coords,settings())*B[prepared.cpnum1-1]
        rows=rows/rows.sum(1,keepdim=True)
    elif model=='continuous_energy':
        rows=normalized_proxy_response(prepared,coords,B,law,
            node_chunk=16,backend='tail16_mid2',gaussian_float32=True)
    else:raise ValueError('Unknown isolated pilot response')
    if not bool(torch.isfinite(rows).all()) or bool((rows<0).any()):
        raise ValueError('Invalid response; no event removal is permitted')
    return rows


def relative_error(a,b):
    return float(torch.linalg.vector_norm(a.double()-b.double())/torch.linalg.vector_norm(b.double()).clamp_min(1e-300))


def numerical_checks(raw,detector,variance,coords,B,law,geometry,model,rank,world):
    """Actual rows, full grid: chunk/rank, transpose and event-constant tests."""
    raw=raw[:32]
    reference=rows_for(raw,detector,variance,coords,B,law,model)
    pieces=torch.cat([rows_for(raw[i:i+8],detector,variance,coords,B,law,model)
        for i in range(0,len(raw),8)])
    chunk_error=relative_error(pieces,reference)
    partition,start,stop=partition_indices(np.arange(len(raw)),rank,world)
    compact=geometry.compact(reference,0).double()
    assigned=geometry.compact(rows_for(raw[partition],detector,variance,coords,B,law,model),0).double()
    idx=torch.arange(geometry.active_count,device=B.device,dtype=torch.float64)[:,None]
    image=1+idx/geometry.active_count
    y=1+torch.arange(len(raw),device=B.device,dtype=torch.float64)[:,None]/len(raw)
    lhs=(compact@image*y).sum();rhs=(image*(compact.T@y)).sum()
    adjoint=float((lhs-rhs).abs()/lhs.abs())
    base=compact.T@(1/(compact@image));local=assigned.T@(1/(assigned@image))
    dist.all_reduce(local,op=dist.ReduceOp.SUM)
    rank_error=relative_error(local,base)
    scale=(.5+y)
    scaled=compact*scale
    scale_error=relative_error(scaled.T@(1/(scaled@image)),base)
    sparse_error=None
    if model=='angular':
        projector=build_compton_sparse_projector(coords,theta_stride=1,z_stride=1,rotate_num=20,dtype=torch.float32).to(B.device)
        packed,_,_=get_compton_backproj_list_single_sparse(B,detector,projector,
            torch.tensor(raw,device=B.device),0.,0.,.440,settings().energy_resolution,
            settings().energy_threshold_max_mev,.05,.350,B.device,
            input_energies_already_smeared=True,max_min_standardized_arm=3.,geometry_mode='stable_float64')
        prior,valid=materialize_sparse_event_rows_to_fine(packed.to(B.device),B,projector)
        if len(prior)!=len(raw) or not bool(valid.all()):raise ValueError('Frozen sample fails original sparse route')
        sparse_error=relative_error(reference,prior)
    receipt=dict(model=model,sample_events=len(raw),full_points=132040,active_points=78920,
        chunk_relative_L2=chunk_error,rank_weight_relative_L2=rank_error,
        forward_adjoint_relative_error=adjoint,event_constant_relative_error=scale_error,
        original_sparse_relative_L2=sparse_error,threshold=1e-5,
        source_truth_used=False,physical_generative_validation=False)
    if any(v>1e-5 for v in (chunk_error,rank_error,adjoint,scale_error)) or (sparse_error is not None and sparse_error>1e-5):
        raise ValueError('Actual response numerical checks failed: '+json.dumps(receipt))
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('contract','factors','input-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',choices=('angular','continuous_energy'),default='angular')
    a=p.parse_args();start=time.monotonic();torch.set_grad_enabled(False)
    root=a.contract.parent;cfg=json.loads(a.contract.read_text())
    if cfg['study']!='compton_energy_probability_v5_preflight' or cfg['formal_submission_permitted']:
        raise ValueError('Pilot-only frozen contract required')
    for name,sha in cfg['files'].items():
        if digest(root/name)!=sha:raise ValueError('Frozen release differs: '+name)
    gate=json.loads((root/'diagnostic_gate.json').read_text())
    if gate['status']!='DIAGNOSTIC_GATES_PASSED' or gate['failures']:
        raise ValueError('Independent diagnostic gate is not passed')
    validate_whole_geometry(root/'whole_geometry.npz',cfg['whole_geometry_sha256'])
    if os.environ.get('NCCL_SOCKET_IFNAME')!='bond0':raise ValueError('NCCL must use bond0')
    rank,world,device=setup('nccl')
    if world not in (4,8):raise ValueError('Only distinct 4/8-node, one-GPU topology')
    error=[None]
    if rank==0:
        try:
            a.output.mkdir(parents=True,exist_ok=False)
            for relative,sha in cfg['input_sha256'].items():
                if digest(a.input_root/relative)!=sha:raise ValueError('Original NEMA input differs: '+relative)
            collection=json.loads((a.input_root/'collections/NEMA_Body_H60_1e9.json').read_text())
            if sum(collection['primary_counts'])!=1_000_000_000 or len(set(collection['seeds']))!=200 or collection['views']!=list(range(1,21)):
                raise ValueError('Original transport closure failed')
            validate(a.factors)
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
    counts=np.loadtxt(a.input_root/'CntStat/440keV_RotateNum20_Geant4JSCC/CntStat_NEMA_Body_H60_1e9.csv',delimiter=',',dtype=np.float32)
    if counts.shape!=(20,10496) or not np.isfinite(counts).all() or np.any(counts<0):raise ValueError('440 counts differ')
    projection=torch.from_numpy(np.ascontiguousarray(counts[:,begin:end].T)).to(device)
    B=full_rows(rawB,device)
    geo=np.load(root/'whole_geometry.npz');coords=torch.tensor(geo['coordinates_mm'],dtype=torch.float32,device=device)
    detector=torch.tensor(load_detector_coordinates(a.factors/NAMES['A440']/'Detector.csv',10496),device=device,dtype=torch.float32)
    variance=build_detector_position_variance(detector,0.)
    law=ContinuousTransferLaw.load(root/'transfer_training_summary.json')
    blocks=[];partitions=[];accepted=[];check=None
    for view in range(20):
        list_path=a.input_root/f'List/218-440keV_RotateNum20_Geant4JSCC/List_NEMA_Body_H60_1e9/{view+1}.csv'
        raw=np.loadtxt(list_path,delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
        indices=np.load(root/f'selections/{view+1}.npy')
        selected,lo,hi=partition_indices(indices,rank,world)
        if len(indices)!=cfg['events_per_view'][view] or indices[-1]>=len(raw):raise ValueError('Frozen selection closure differs')
        if view==0:check=numerical_checks(raw[indices],detector,variance,coords,B,law,geometry,a.model,rank,world)
        items=[]
        for offset in range(0,len(selected),32):
            if torch.cuda.memory_reserved()>.6*torch.cuda.get_device_properties(device).total_memory:torch.cuda.empty_cache()
            rows=rows_for(raw[selected[offset:offset+32]],detector,variance,coords,B,law,a.model)
            items.append(geometry.compact(rows,view).cpu());del rows
            if torch.cuda.max_memory_reserved()>.75*torch.cuda.get_device_properties(device).total_memory:
                raise RuntimeError('Pilot runtime reserved peak exceeds proactive 75% limit')
        blocks.append(items);accepted.append(len(selected))
        partitions.append(dict(view=view+1,first_selected_position=lo,last_selected_position_exclusive=hi,
            events=len(selected),original_rows_sha256=hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest()))
        print('ENERGY_V5_PREPARE',a.model,rank,view+1,len(selected),flush=True)
    del B,rawB,detector,variance,coords;torch.cuda.empty_cache()
    total=torch.tensor(accepted,dtype=torch.int64,device=device);dist.all_reduce(total,op=dist.ReduceOp.SUM)
    if total.cpu().tolist()!=cfg['events_per_view'] or int(total.sum())!=91225:raise ValueError('Full event/rank closure failed')
    sensitivity_path=root/(a.model+'_Sensi_full')
    full_s=np.fromfile(sensitivity_path,dtype='<f4')
    if len(full_s)!=132040 or not np.isfinite(full_s).all() or np.any(full_s<=0):raise ValueError('Invalid matched full-circle S')
    sensitivity=geometry.compton_sensitivity(torch.tensor(full_s,device=device))
    (id_,hd),(ij,hj)=compton_and_joint_mlem(response,projection,blocks,single_sensitivity,sensitivity,
        10,10,save_history=rank==0,progress_label='energy_v5_'+a.model)
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
        elapsed_seconds=time.monotonic()-start,numerical_checks=check)
    if memory<=0 or reserved>.8*gpu_total or rss>.8*memory:raise RuntimeError('Actual pilot resource margin fails')
    resources=[None]*world;dist.all_gather_object(resources,record)
    if len({r['node'] for r in resources})!=world:raise ValueError('Distinct rank/node allocation failed')
    if rank==0:
        write(a.output/'run_manifest.json',dict(study=cfg['study'],model=a.model,iterations=10,save_step=10,
            world_size=world,pixels_active=78920,pixels_full=132040,accepted_compton_events=91225,
            accepted_compton_events_per_view=total.cpu().tolist(),resources=resources,
            contract_sha256=digest(a.contract),geometry_sha256=cfg['whole_geometry_sha256'],
            sensitivity_sha256=digest(sensitivity_path),input_sha256=cfg['input_sha256'],
            factor_manifest_sha256=cfg['factor_manifest_sha256'],source_sha256=digest(__file__),
            initial_density=1.,algorithm='unchanged torch_active_operator JSCC MLEM',
            new_photons=0,new_fine_A_matrices=0,formal_submission_permitted=False))
    dist.barrier();dist.destroy_process_group()
    print('ENERGY_V5_PILOT_FINISHED',a.model,rank,flush=True)


if __name__=='__main__':main()
