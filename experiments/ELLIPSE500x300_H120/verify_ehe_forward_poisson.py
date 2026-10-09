"""Read-only complete matrix-Poisson data, operator, checkpoint and resource acceptance."""
import argparse,math,re
from pathlib import Path
import numpy as np
from ehe_common import *

def source_macro_matches(original,actual):
    # The frozen launcher reads ASCII text and writes it on Linux. Only newline
    # representation may differ from the registered Windows source bytes.
    return Path(original).read_bytes().replace(b'\r\n',b'\n')==Path(actual).read_bytes()

def operator_closure(result,release,responses,geometry):
    """Exercise the frozen helper on every view, in bounded CPU detector-row blocks.

    This is read-only acceptance work. Its CPU resource use is not the Slurm
    certificate for the separately completed GPU reconstruction.
    """
    import sys,time,torch
    sys.path.insert(0,str(release))
    from torch_active_operator import ActiveGeometry
    torch.set_num_threads(6)
    active=geometry['active_indices'];inverse=geometry['inverse_rotation']
    g=ActiveGeometry(active,geometry['ellipse_fraction'],inverse,'cpu')
    views=g.views;bins=2312;started=time.monotonic();proof={}
    x=torch.linspace(.2,1.2,g.active_count)[:,None]
    z=torch.linspace(.1,.9,bins)[:,None]
    final=torch.from_numpy(np.fromfile(result/f'Image_{CHANNELS[0]}_final.float32','<f4').copy())[:,None]
    background=np.zeros((bins,views),np.float32)
    for name in RESPONSES:
        raw=np.memmap(responses/name/'SysMat_polar','<f4',mode='r',shape=(g.full_count,bins))
        sensitivity=np.zeros(g.active_count,np.float64)
        full_sums=np.zeros(g.full_count,np.float64)
        lhs=np.zeros(views,np.float64);rhs=np.zeros(views,np.float64)
        for start in range(0,bins,64):
            stop=min(start+64,bins)
            rows=torch.from_numpy(np.array(raw[:,start:stop].T,copy=True))
            full_sums+=rows.sum(dim=0,dtype=torch.float64).numpy()
            for view in range(views):
                matrix=g.compact(rows,view)
                sensitivity+=matrix.sum(dim=0,dtype=torch.float64).numpy()/views
                lhs[view]+=float(((matrix@x)*z[start:stop]).sum(dtype=torch.float64))
                rhs[view]+=float((x*(matrix.T@z[start:stop])).sum(dtype=torch.float64))
                if name=='C440to218':background[start:stop,view]=((matrix@final)/views).numpy().reshape(-1)
        saved_s=np.fromfile(responses/name/'S_active.float64','<f8')
        saved_full=np.fromfile(responses/name/'S_full.float64','<f8')
        if saved_s.shape!=sensitivity.shape or saved_full.shape!=full_sums.shape:raise ValueError('Own sensitivity shape differs')
        relative=lambda a,b:float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)),1e-30))
        active_l2=relative(sensitivity,saved_s);full_l2=relative(full_sums,saved_full)
        adjoint=np.abs(lhs-rhs)/np.maximum(np.abs(lhs),1e-30)
        if active_l2>1e-5 or full_l2>1e-5 or np.any(adjoint>1e-5):raise ValueError('Frozen operator/sensitivity/transpose closure failed: '+name)
        proof[name]=dict(sensitivity_l2=active_l2,full_row_sum_l2=full_l2,adjoint_relative_error_by_view=adjoint.tolist())
        del raw
    saved=np.load(result/'fixed_cross_background.npy')
    if saved.shape!=background.shape or np.any(~np.isfinite(saved)) or np.any(saved<0):raise ValueError('Fixed cross-window background invalid')
    background_l2=relative(background,saved)
    if background_l2>1e-5:raise ValueError('218 additive background is not the forward prediction of this EHE final440 image')
    return dict(passed=True,views=views,responses=proof,background_l2=background_l2,
        background_source_sha256=digest(result/f'Image_{CHANNELS[0]}_final.float32'),
        method='Unmodified frozen ActiveGeometry, all rows/all views, CPU row blocks; no image rescaling',
        elapsed_seconds=time.monotonic()-started,imaging_resource_certificate=False)

from ehe_forward_poisson_data import verify_counts,verify_forward_means

def check_generation_resources(collection,accounting):
    from ehe_slurm_status import stage_completed
    if not stage_completed(accounting,collection['allocation']['job']):raise ValueError('Generation not actually completed')
    alloc=collection['allocation'];maxrss=0
    if allocated_bytes(alloc['scontrol'])!=alloc['host_allocated_bytes'] or not alloc['imaging_allocation_certificate']:
        raise ValueError('Generation actual AllocTRES denominator differs')
    for row in accounting.splitlines():
        fields=row.split('|')
        if len(fields)>3:
            m=re.fullmatch(r'([0-9.]+)([KMGT]?)',fields[3])
            if m:maxrss=max(maxrss,int(float(m[1])*1024**(' KMGT'.index(m[2]) if m[2] else 0)))
    if not 0<maxrss<=.8*alloc['host_allocated_bytes']:raise ValueError('Generation Slurm resource reserve failed')
    if set(collection['resources'])!=set(RESPONSES) or any(v['rss_fraction']>.8 or not 0<v['gpu_reserved_fraction']<=.8 for v in collection['resources'].values()):
        raise ValueError('Generation process/GPU resource reserve failed')
    return dict(passed=True,slurm_maxrss_bytes=maxrss,host_fraction=maxrss/alloc['host_allocated_bytes'],accounting=accounting)

def verify(result,release,responses,counts,accounting,generation_accounting):
    from ehe_slurm_status import stage_completed
    run=read(result/'run_manifest.json');policy(run['mode'],run['iterations'],run['save_step'])
    if not stage_completed(accounting,run['allocation']['job']):raise ValueError('Imaging has not actually completed')
    release_manifest=read(release/'release_manifest.json');verify_files(release,release_manifest['sha256'])
    if run['release_key']!=release_manifest['release_key']:raise ValueError('Release differs')
    if run['channels']!=list(CHANNELS) or run['geometry_sha256']!=digest(release/'whole_geometry.npz'):raise ValueError('Channel/whole geometry identity differs')
    if run['baseline_helper_sha256']!={n:digest(release/n) for n in ('torch_active_operator.py','single_checkpoint_mlem.py')}:raise ValueError('Frozen MLEM helper identity differs')
    collection=verify_counts(counts,release);config=read(release/'config.json')
    if digest(counts/'collection.json')!=run['counts_sha256'] or run['data_kind']!=collection['data_kind'] or run['physical_calibration_claim']:
        raise ValueError('Independent synthetic observation identity differs')
    mean_closure=verify_forward_means(counts,release,responses)
    generation_resources=check_generation_resources(collection,generation_accounting)
    for name in RESPONSES:
        f=read(responses/name/'factor_manifest.json');verify_files(responses/name,f['files'])
        if digest(responses/name/'factor_manifest.json')!=run['factor_sha256'][name]:raise ValueError('Factor identity differs')
        if run['factor_sha256'][name]!=config['factor_sha256'][name]:raise ValueError('Registered synthetic factor identity differs')
        if not f['passed'] or (f['bins'],f['full_points'],f['active_points'])!=(2312,132040,78920):raise ValueError('Complete factor dimensions differ')
        if digest(responses/name/'whole_geometry.npz')!=run['geometry_sha256']:raise ValueError('Response coordinates/rotation/volume differ')
        verify_files(responses/name,config['params_sha256'][name])
        s=np.fromfile(responses/name/'S_active.float64','<f8')
        if s.shape!=(78920,) or np.any(~np.isfinite(s)) or np.any(s<=0):raise ValueError('Sensitivity invalid')
    g=np.load(release/'whole_geometry.npz');active=g['active_indices'];frames=run['iterations']//10;hist={};checkpoints=[]
    for channel in CHANNELS:
        path=result/f'Image_{channel}_history.float32'
        if path.stat().st_size!=frames*78920*4:raise ValueError('History length differs')
        h=np.fromfile(path,'<f4').reshape(frames,78920);hist[channel]=h
        if np.any(~np.isfinite(h)) or np.any(h<0):raise ValueError('History invalid')
        if not np.array_equal(h[-1],np.fromfile(result/f'Image_{channel}_final.float32','<f4')):raise ValueError('Final/history mismatch')
        if channel==CHANNELS[2]:continue
        folders=sorted((result/'checkpoints'/channel).iterdir())
        if len(folders)!=frames:raise ValueError('Checkpoint count differs')
        for i,folder in enumerate(folders):
            m=read(folder/'manifest.json');verify_files(folder,m['files'])
            if m['iteration']!=(i+1)*10 or m['channel']!=channel or m['release_key']!=run['release_key']:raise ValueError('Checkpoint identity differs')
            a=np.fromfile(folder/'active.float32','<f4');f=np.fromfile(folder/'full.float32','<f4')
            expected=np.zeros(132040,'<f4');expected[active]=h[i]
            if not np.array_equal(a,h[i]) or not np.array_equal(f,expected):raise ValueError('Complete checkpoint/support domain mismatch')
            checkpoints.append(dict(channel=channel,iteration=m['iteration'],sha256=digest(folder/'manifest.json')))
    if not np.array_equal(hist[CHANNELS[2]],hist[CHANNELS[0]]+hist[CHANNELS[1]]):raise ValueError('Combined sum differs')
    if run['mode']=='validation':
        if len(run['tests'])!=2 or any(not math.isfinite(v['l2']) or not math.isfinite(v['adjoint_relative_error']) or v['l2']>1e-5 or v['adjoint_relative_error']>1e-5 or not v['history_equal'] for v in run['tests'].values()):raise ValueError('Actual numerical checks missing/failed')
    if not any(line.split('|')[:3]==[str(run['allocation']['job']),'COMPLETED','0:0'] for line in accounting.splitlines()):raise ValueError('Actual successful exit required')
    maxrss=0
    for line in accounting.splitlines():
        fields=line.split('|')
        for value in fields[3:4]:
            m=re.fullmatch(r'([0-9.]+)([KMGT]?)',value)
            if m:maxrss=max(maxrss,int(float(m[1])*1024**(' KMGT'.index(m[2]) if m[2] else 0)))
    if maxrss<=0:raise ValueError('Final Slurm MaxRSS unavailable')
    if maxrss>.8*run['allocation']['host_allocated_bytes']:raise MemoryError('Actual Slurm memory reserve failed')
    if not run['allocation']['imaging_allocation_certificate'] or allocated_bytes(run['allocation']['scontrol'])!=run['allocation']['host_allocated_bytes']:raise ValueError('Actual imaging memory allocation identity differs')
    if len(run['resources'])!=2*frames or any(not math.isfinite(v['rss_fraction']) or not math.isfinite(v['gpu_reserved_fraction']) or v['rss_fraction']>.8 or v['gpu_reserved_fraction']>.8 or v['gpu_reserved_peak_bytes']<=0 or v['gpu_total_bytes']<=0 for v in run['resources']):raise MemoryError('Runtime resource proof failed')
    closure=operator_closure(result,release,responses,g)
    return dict(passed=True,mode=run['mode'],release_key=run['release_key'],frames_per_channel=frames,checkpoints=checkpoints,
        iterations=run['iterations'],counts_sha256=run['counts_sha256'],data_kind=run['data_kind'],physical_calibration_claim=False,factor_sha256=run['factor_sha256'],
        phase_seconds=run['phase_seconds'],slurm_maxrss_bytes=maxrss,accounting=accounting,operator_closure=closure,
        verification_source_sha256={n:digest(HERE/n) for n in ('verify_ehe_forward_poisson.py','ehe_common.py','ehe_forward_poisson_data.py')},files=hashes(result),forward_mean_closure=mean_closure,generation_resource_acceptance=generation_resources)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('result','release','responses','counts','accounting','generation-accounting','output','verification-manifest','evidence'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verifier=read(a.verification_manifest);verify_files(HERE,verifier['files'])
    a.evidence.mkdir(parents=True,exist_ok=False);alloc=allocation();write(a.evidence/'allocation.json',alloc)
    import time,resource
    started=time.monotonic()
    try:
        proof=verify(a.result,a.release,a.responses,a.counts,a.accounting.read_text(),a.generation_accounting.read_text())
        proof['verification_manifest_sha256']=digest(a.verification_manifest)
        write(a.output,proof)
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if rss>.8*alloc['host_allocated_bytes']:raise MemoryError('Read-only verifier reserve failed')
        write(a.evidence/'receipt.json',dict(passed=True,authority_sha256=digest(a.output),elapsed_seconds=time.monotonic()-started,
            allocation=alloc,rss_peak_bytes=rss,host_fraction=rss/alloc['host_allocated_bytes'],gpu_calculation=False,imaging_resource_certificate=False))
        print('MATRIX_POISSON_VERIFIED',proof['mode'],flush=True)
    except BaseException as exc:
        write(a.evidence/'failure.json',dict(passed=False,error=str(exc)))
        raise
