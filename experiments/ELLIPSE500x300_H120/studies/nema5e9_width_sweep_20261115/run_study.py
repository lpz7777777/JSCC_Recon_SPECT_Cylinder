"""One formal allocation: matched calibration, paired benchmark, three JSCC solves.

No new transport, event selection or external jobs. Shared disk contains only
small outputs; one candidate's full response resides in distributed CPU RAM.
"""
import argparse,gc,hashlib,json,os,resource,time
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch_active_operator as op
from detector_csv import load_detector_coordinates
from compton_event_response import ComptonEventSettings,prepare_compton_events,build_detector_position_variance
from compton_energy_probability_v5 import ContinuousTransferLaw
from width_kernel import scaled_response
from jscc_5e10_joint_mlem import joint_mlem
from adapters import BoundedViewResponse,OrderedTransferPool,ViewBlocks,current_cuda_budget

LIFETIME_GPU_PEAK=0

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def save_json(p,x,replace=False):
    data=(json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
    if replace:
        p=Path(p);tmp=p.with_name(p.name+'.partial')
        with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    else:atomic(p,data)
def settings():return ComptonEventSettings(.440,.13*np.sqrt(.511/.440),2*.440**2/(.511+2*.440)-.001,.05,.350,geometry_mode='stable_float64')
def max_time(value,device):
    x=torch.tensor(value,dtype=torch.float64,device=device);dist.all_reduce(x,op=dist.ReduceOp.MAX);return float(x)
def all_true(value,device):
    x=torch.tensor(int(value),device=device);dist.all_reduce(x,op=dist.ReduceOp.MIN);return bool(x)
def sync():torch.cuda.synchronize();dist.barrier();torch.cuda.synchronize()
def prepared(raw,det,var,device):
    p,_=prepare_compton_events(torch.tensor(raw,device=device),settings(),det,var,var,input_energies_already_smeared=True)
    if p is None or p.count!=len(raw):raise ValueError('Frozen rows changed; event deletion forbidden')
    return p
def guard(host_bytes):
    global LIFETIME_GPU_PEAK
    LIFETIME_GPU_PEAK=max(LIFETIME_GPU_PEAK,torch.cuda.max_memory_reserved())
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if rss>.8*host_bytes or LIFETIME_GPU_PEAK>.8*torch.cuda.get_device_properties(0).total_memory:
        raise MemoryError('Actual RSS/GPU historical peak exceeds 80%')
    return dict(rss_peak_bytes=rss,gpu_reserved_peak_bytes=LIFETIME_GPU_PEAK,phase_gpu_peak_bytes=torch.cuda.max_memory_reserved(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,host_allocated_bytes=host_bytes)

def begin_cache_phase(host_bytes):
    # Preserve lifetime peak for the guard; budget a new phase only after the
    # preceding phase has synchronized and released its temporaries/caches.
    sync();guard(host_bytes);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()

def actual_row_checks(raw,det,var,coords,B,law,scale,geo):
    reference=scaled_response(prepared(raw,det,var,B.device),coords,B,law,scale)
    pieces=torch.cat([scaled_response(prepared(raw[i:i+8],det,var,B.device),coords,B,law,scale) for i in range(0,len(raw),8)])
    err=float(torch.linalg.vector_norm((pieces-reference).double())/torch.linalg.vector_norm(reference.double()))
    errors=[]
    x=1+torch.arange(geo.active_count,device=B.device,dtype=torch.float64)[:,None]/geo.active_count
    y=1+torch.arange(len(raw),device=B.device,dtype=torch.float64)[:,None]/len(raw)
    for v in (0,9,19):
        C=geo.compact(reference,v).double();lhs=((C@x)*y).sum();rhs=(x*(C.T@y)).sum()
        errors.append(float((lhs-rhs).abs()/lhs.abs()))
    if max([err]+errors)>1e-5:raise ValueError('Actual full-grid response numerical check failed')
    return dict(full_points=geo.full_count,active_points=geo.active_count,events=len(raw),scale=scale,chunk_relative_L2=err,adjoint_errors=errors,threshold=1e-5)

def profile_update(provider,projection,events,ss,sc,device,pool):
    """One diagnostic update, excluded from throughput timing; CUDA completion times."""
    if pool:pool.reset_stats();pool.timing=True
    measured={};cpu_submit={};x=torch.ones_like(ss);w=torch.zeros_like(x)
    def segment(name,fn):
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        before=time.perf_counter();begin.record();value=fn();end.record()
        cpu_submit[name]=cpu_submit.get(name,0.)+time.perf_counter()-before
        measured.setdefault(name,[]).append((begin,end));return value
    sync();start=time.monotonic()
    for v in range(provider.geometry.views):
        A=segment('SC_matrix',lambda:provider.matrix(v))
        f=segment('SC_forward',lambda:A@x)
        w+=segment('SC_backprojection',lambda:A.T@(projection[:,v:v+1]/f.clamp_min(1e-12)))
        wd=torch.zeros_like(x)
        for stored in events[v]:
            C=segment('H2D_pageable',lambda:stored.to(device,non_blocking=True)) if stored.device!=device else stored
            f=segment('Compton_forward',lambda:C@x)
            wd+=segment('Compton_backprojection',lambda:C.T@(1/f.clamp_min(1e-12)))
        w+=wd
    segment('communication',lambda:dist.all_reduce(w))
    result=segment('update',lambda:op._update(x,w,ss+sc));sync()
    wall=max_time(time.monotonic()-start,device)
    timing={k:sum(a.elapsed_time(b)/1000 for a,b in pairs) for k,pairs in measured.items()}
    transfer=pool.finish_stats() if pool else None
    if pool:pool.timing=False
    reference,_=joint_mlem(provider,projection,events,ss,sc,1,1,False);sync()
    if not all_true(torch.equal(result,reference),device):raise ValueError('Profiled update differs')
    return dict(cuda_seconds=timing,cpu_submit_seconds=cpu_submit,wall_max_rank_seconds=wall,transfer=transfer,
                note='Instrumented one-step diagnostic; segment totals overlap for prefetch and are not added to wall time; not used for speed selection')

def benchmark(root,out,response,projection,blocks,ss,sc,geo,rank,device,host_bytes):
    """Paired complete input, same rank order, 50 steps, three repeats."""
    records=[];reference=None;valid={};times={}
    for mode in ['baseline','cache','grouped_sync','grouped_prefetch']:
        cache=None;pool=None;begin_cache_phase(host_bytes);began=time.monotonic()
        if mode!='baseline':
            cache=BoundedViewResponse(op,response.full_rows,geo,budget=current_cuda_budget(device,512<<20),phase='joint_solve_after_compton_prepare')
        provider=cache or response;events=blocks
        if mode.startswith('grouped'):
            pool=OrderedTransferPool(geo.active_count,device,max_host_staging_bytes=160<<20,
                device_budget=current_cuda_budget(device,256<<20),asynchronous=mode=='grouped_prefetch',timing=False)
            events=[ViewBlocks(pool,b) for b in blocks]
        sync();setup=max_time(time.monotonic()-began,device)
        S=provider.sensitivity();dist.all_reduce(S)
        sameS=all_true(torch.equal(S,ss),device)
        joint_mlem(provider,projection,events,ss,sc,1,1,False);sync()
        if pool:pool.reset_stats()
        durations=[];equal=True;diffs=[]
        for repeat in range(3):
            sync();start=time.monotonic()
            image,history=joint_mlem(provider,projection,events,ss,sc,50,50,rank==0)
            sync();durations.append(max_time(time.monotonic()-start,device))
            if reference is None:reference=image.clone()
            same=all_true(torch.equal(reference,image),device);equal &= same
            delta=(image-reference).double();norm=torch.linalg.vector_norm(reference.double()).clamp_min(1e-30)
            diffs.append(dict(relative_L2=float(torch.linalg.vector_norm(delta)/norm),peak_scaled_absolute=float(delta.abs().max()/reference.abs().max().clamp_min(1e-30))))
        valid[mode]=bool(equal and sameS);times[mode]=float(np.median(durations))
        stats=pool.finish_stats() if pool else dict(transfer_calls=150*sum(len(b) for b in blocks),transfer_bytes=150*sum(t.numel()*4 for b in blocks for t in b),h2d_measured=False)
        diagnostic=profile_update(provider,projection,events,ss,sc,device,pool)
        record=dict(mode=mode,bitwise=valid[mode],S_bitwise=sameS,setup_max_rank_seconds=setup,iterations=50,repeats=3,
            max_rank_seconds=durations,median_seconds=times[mode],errors=diffs,transfer_stats=stats,
            cache_views=cache.cached_views if cache else 0,cache_bytes=cache.cache_bytes if cache else 0,resources=guard(host_bytes),diagnostic_one_step=diagnostic)
        save_json(out/f'benchmark_{mode}_rank{rank}.json',record);records.append(record)
        if pool:pool.close()
        if cache:cache.close()
        del image,history,S;gc.collect();torch.cuda.empty_cache()
    chosen='cache' if valid['cache'] and times['cache']<times['baseline'] else 'baseline'
    if not valid['baseline']:raise ValueError('Baseline paired runs are not reproducible')
    for mode in ['grouped_sync','grouped_prefetch']:
        if valid[mode] and times[mode]<.95*times[chosen]:chosen=mode
    choice=dict(chosen=chosen,times=times,bitwise_gates=valid,cache_equivalent=valid['cache'],no_width_model_comparison_in_timing=True)
    if rank==0:save_json(out/'benchmark_choice.json',choice)
    return chosen

def main():
    p=argparse.ArgumentParser();p.add_argument('--input-root',type=Path,required=True);p.add_argument('--factors',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--host-bytes',type=int,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent;cfg=json.loads((root/'study.json').read_text());release=json.loads((root/'release_manifest.json').read_text())
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=8 or local!=0:raise ValueError('Frozen 8 nodes x 1 GPU topology required')
    torch.set_num_threads(8);torch.set_grad_enabled(False);torch.cuda.set_device(0);device=torch.device('cuda:0')
    dist.init_process_group('nccl',timeout=timedelta(minutes=30));started=time.monotonic()
    if torch.cuda.device_count()!=1 or '5090' not in torch.cuda.get_device_name():raise RuntimeError('Expected one ordinary 5090')
    error=[None]
    if rank==0:
        try:
            a.output.mkdir(parents=True,exist_ok=False)
            for n,s in release['files'].items():assert sha(root/n)==s,n
            for n,s in cfg['input_sha256'].items():assert sha(a.input_root/n)==s,n
            for n,s in cfg['factor_sha256'].items():assert sha(a.factors/n)==s,n
        except Exception as e:error=[str(e)]
    dist.broadcast_object_list(error,0)
    if error[0]:raise ValueError(error[0])
    geom=np.load(root/'whole_geometry.npz');geo=op.ActiveGeometry.from_npz(root/'whole_geometry.npz',device)
    rawB=np.memmap(a.factors/'440keV_RotateNum20/SysMat_polar',dtype='<f4',mode='c',shape=(132040,10496))
    lo=10496*rank//world;hi=10496*(rank+1)//world
    # Preserve the actual frozen loaders' strides as well as values.
    localB=torch.from_numpy(np.ascontiguousarray(rawB[:,lo:hi].T)).to(device)
    response=op.ViewResponse(localB,geo,'none');ss=response.sensitivity();dist.all_reduce(ss)
    counts=np.loadtxt(a.input_root/'CntStat/440keV_RotateNum20_Geant4JSCC/CntStat_NEMA_Body_H60_5e9.csv',delimiter=',',dtype=np.float32)
    assert counts.shape==(20,10496)
    projection=torch.tensor(np.ascontiguousarray(counts[:,lo:hi].T),device=device)
    law=ContinuousTransferLaw.load(root/'transfer_training_summary.json')
    det=torch.tensor(load_detector_coordinates(a.factors/'440keV_RotateNum20/Detector.csv',10496),dtype=torch.float32,device=device)
    var=build_detector_position_variance(det,0.);coords=torch.tensor(geom['coordinates_mm'],dtype=torch.float32,device=device)
    cal=np.load(root/'calibration_rows.npy');assert cal.shape==(170247,4)
    if rank==0:save_json(a.output/'started.json',dict(job=os.environ['SLURM_JOB_ID'],release=release['key'],ranks=world,scales=cfg['scales'],torch=torch.__version__,cuda=torch.version.cuda,precision_flags=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)))
    choice=None
    for scale in cfg['scales']:
        label=f'width{round(scale*100)}';out=a.output/label
        if rank==0:out.mkdir();save_json(a.output/'progress.json',dict(phase=label+'_prepare',iteration=0),replace=True)
        dist.barrier();start=time.monotonic();B=torch.from_numpy(rawB.T).to(device)
        if scale==1.:
            sfull=np.fromfile(root/'continuous_energy_Sensi_full','<f4');cal_time=0.
        else:
            ct=time.monotonic();total=torch.zeros(132040,dtype=torch.float64,device=device)
            # Partition original 32-event calibration groups; each event once.
            for batch,start_i in enumerate(range(0,len(cal),32)):
                if batch%world!=rank:continue
                if time.monotonic()-ct>cfg['calibration_limit_hours']*3600:raise TimeoutError('Calibration time bound')
                if torch.cuda.memory_reserved()>.6*torch.cuda.get_device_properties(0).total_memory:torch.cuda.empty_cache()
                rows=scaled_response(prepared(cal[start_i:start_i+32],det,var,device),coords,B,law,scale)
                if not bool(torch.allclose(rows.sum(1),torch.ones(len(rows),device=device),atol=1e-5,rtol=0)):raise ValueError('Calibration rows not normalized')
                total+=rows.double().sum(0)
                del rows
                if batch%400==rank:guard(a.host_bytes);print('CALIBRATION',label,batch,flush=True)
            dist.all_reduce(total);volume=float(geom['cell_volume_mm3'].sum());unrot=total.cpu().numpy()*volume/1e9
            averaged=unrot[geom['inverse_rotation']].mean(1);sfull=averaged.astype('<f4')
            expected=len(cal)*volume/1e9
            if not np.isclose(unrot.sum(),expected,rtol=1e-5):raise ValueError('Independent S normalization failure')
            if rank==0:
                atomic(out/'S_calibration_detector.float64',unrot.astype('<f8').tobytes())
                save_json(out/'S_provenance.json',dict(source='Independent frozen circle_train, not NEMA observed C',events=len(cal),primary_gamma_440=1000000000,scale=scale,selection_sha256=json.loads((root/'calibration_extract.json').read_text())['selection_sha256'],integral_relative_error=float(unrot.sum()/expected-1),row_batch=32,full_volume_mm3=volume))
            cal_time=max_time(time.monotonic()-ct,device)
            del total
        sc=geo.compton_sensitivity(torch.tensor(sfull,device=device))
        if not bool(torch.isfinite(sc).all()) or bool((sc<0).any()):raise ValueError('Invalid matching S')
        if rank==0:
            atomic(out/'S_Compton_full.float32',sfull.tobytes());atomic(out/'S_Compton_active.float32',sc.cpu().numpy().astype('<f4').tobytes())
            atomic(out/'S_SC_active.float32',ss.cpu().numpy().astype('<f4').tobytes())
        blocks=[];parts=[]
        for view in range(20):
            raw=np.loadtxt(a.input_root/f'List/218-440keV_RotateNum20_Geant4JSCC/List_NEMA_Body_H60_5e9/{view+1}.csv',delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
            selected=np.load(root/f'selections/{view+1}.npy');assert len(selected)==cfg['events_per_view'][view]
            first=len(selected)*rank//world;last=len(selected)*(rank+1)//world;idx=selected[first:last];items=[]
            if view==0:
                receipt=actual_row_checks(raw[idx[:32]],det,var,coords,B,law,scale,geo)
                save_json(out/f'numerical_rank{rank}.json',receipt);guard(a.host_bytes);torch.cuda.empty_cache()
            for start_i in range(0,len(idx),32):
                if torch.cuda.memory_reserved()>.6*torch.cuda.get_device_properties(0).total_memory:torch.cuda.empty_cache()
                rows=scaled_response(prepared(raw[idx[start_i:start_i+32]],det,var,device),coords,B,law,scale)
                compact=geo.compact(rows,view)
                if not bool(torch.isfinite(compact).all()) or bool((compact.sum(1)<=0).any()):raise ValueError('Zero/invalid active support; cannot drop event')
                items.append(compact.cpu())
                del rows,compact
                if start_i%512==0:guard(a.host_bytes)
            blocks.append(items);parts.append(dict(view=view+1,first=first,last=last,events=len(idx),selection_sha256=sha(root/f'selections/{view+1}.npy')))
            print('WIDTH_PREPARE',label,rank,view+1,len(idx),flush=True)
        del B,raw;gc.collect();torch.cuda.empty_cache();sync()
        prepared_time=max_time(time.monotonic()-start,device)
        eventcount=torch.tensor(sum(x['events'] for x in parts),device=device);dist.all_reduce(eventcount);assert int(eventcount)==483743
        if scale==1.:choice=benchmark(root,out,response,projection,blocks,ss,sc,geo,rank,device,a.host_bytes)
        cache=None;pool=None;begin_cache_phase(a.host_bytes);t=time.monotonic()
        if choice!='baseline':cache=BoundedViewResponse(op,localB,geo,budget=current_cuda_budget(device,512<<20),phase='joint_solve_after_compton_prepare')
        events=blocks
        if choice.startswith('grouped'):
            pool=OrderedTransferPool(geo.active_count,device,max_host_staging_bytes=160<<20,device_budget=current_cuda_budget(device,256<<20),asynchronous=choice=='grouped_prefetch',timing=False)
            events=[ViewBlocks(pool,b) for b in blocks]
        sync();cache_time=max_time(time.monotonic()-t,device);checkpoint_times=[];checksums={}
        def checkpoint(iteration,history):
            begin=time.monotonic();frame=history[-1].numpy().astype('<f4');data=frame.tobytes();name=f'checkpoints/iteration_{iteration:05d}.float32'
            if not np.isfinite(frame).all() or (frame<0).any():raise ValueError('Invalid saved density')
            if scale==1. and iteration in (50,10000):
                reference=np.fromfile(root/f'original_iteration_{iteration:05d}.float32','<f4')
                delta=frame.reshape(-1).astype(float)-reference.astype(float)
                rel=float(np.linalg.norm(delta)/np.linalg.norm(reference.astype(float)))
                save_json(out/f'original_regression_{iteration:05d}.json',dict(relative_L2=rel,bitwise=np.array_equal(frame.reshape(-1),reference),threshold=1e-5,reference_job=1669255))
                if not np.isfinite(rel) or rel>1e-5:raise ValueError('Alpha100 differs from delivered original')
            atomic(out/name,data);checksums[name]=hashlib.sha256(data).hexdigest()
            save_json(out/'checkpoint_manifest.json',dict(iteration=iteration,shape=list(frame.shape),scale=scale,release=release['key'],sha256=checksums),replace=True)
            save_json(a.output/'progress.json',dict(phase=label,iteration=iteration,elapsed_seconds=time.monotonic()-started),replace=True)
            checkpoint_times.append(time.monotonic()-begin)
        sync();solve=time.monotonic()
        im,h=joint_mlem(cache or response,projection,events,ss,sc,10000,50,rank==0,checkpoint if rank==0 else None,cfg['phase_limit_hours']*3600)
        sync();elapsed=max_time(time.monotonic()-solve,device)
        if rank==0:
            atomic(out/'Image_JSCC_active.float32',im.cpu().numpy().astype('<f4').tobytes());atomic(out/'Image_JSCC_history.float32',h.numpy().astype('<f4').tobytes())
            full=np.zeros((geo.full_count,1),np.float32);full[geom['active_indices']]=im.cpu().numpy();atomic(out/'Image_JSCC_full.float32',full.tobytes())
        stats=pool.finish_stats() if pool else None
        record=dict(rank=rank,node=os.environ['SLURMD_NODENAME'],scale=scale,partitions=parts,prepare_max_rank_seconds=prepared_time,
            calibration_max_rank_seconds=cal_time,cache_setup_max_rank_seconds=cache_time,solve_max_rank_seconds=elapsed,seconds_per_iteration=elapsed/10000,
            chosen=choice,cache_views=cache.cached_views if cache else 0,transfer_stats=stats,checkpoint_seconds=sum(checkpoint_times),resources=guard(a.host_bytes))
        save_json(out/f'rank{rank}.json',record)
        if pool:pool.close()
        if cache:cache.close()
        del blocks,items,events,im,h,sc,cache,pool;gc.collect();torch.cuda.empty_cache();dist.barrier()
        if rank==0:save_json(out/'completed.json',dict(scale=scale,iterations=10000,save_step=50,frames=200,release=release['key'],files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    if rank==0:save_json(a.output/'completed.json',dict(scales=cfg['scales'],elapsed_seconds=time.monotonic()-started,release=release['key'],scientific_model_changed_except_alpha100=True))
    dist.destroy_process_group()

if __name__=='__main__':main()
