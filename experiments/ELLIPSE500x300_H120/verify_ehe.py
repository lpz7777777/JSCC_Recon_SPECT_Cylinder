"""Read-only actual outputs/checkpoints and immutable acquisition/factor verification."""
import argparse,math,re
from pathlib import Path
import numpy as np
from ehe_common import *
from ehe_execution_policy import physical_permission

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

def verify(result,release,responses,counts,physical,accounting,physical_policy=None):
    run=read(result/'run_manifest.json');policy(run['mode'],run['iterations'],run['save_step'])
    release_manifest=read(release/'release_manifest.json');verify_files(release,release_manifest['sha256'])
    if run['release_key']!=release_manifest['release_key']:raise ValueError('Release differs')
    if run['channels']!=list(CHANNELS) or run['geometry_sha256']!=digest(release/'whole_geometry.npz'):raise ValueError('Channel/whole geometry identity differs')
    if run['baseline_helper_sha256']!={n:digest(release/n) for n in ('torch_active_operator.py','single_checkpoint_mlem.py')}:raise ValueError('Frozen baseline helper identity differs')
    if digest(counts/'collection.json')!=run['counts_sha256'] or digest(physical/'physical_gate.json')!=run['physical_gate_sha256']:raise ValueError('Observation/physics identity differs')
    collection=read(counts/'collection.json');verify_files(counts,collection['files'])
    if collection['total_primary_photons']!=5_000_000_000 or collection['workers']!=200 or collection['views']!=20:raise ValueError('Actual transport incomplete')
    config=read(release/'config.json');receipts=collection['receipts']
    registry=counts/'source_registry'
    if digest(registry/'jobs.json')!=config['simulation_manifest_sha256']:raise ValueError('Frozen source registry identity differs')
    jobs=read(registry/'jobs.json')['jobs']
    if len(jobs)!=200:raise ValueError('Actual registered worker count differs')
    if len(receipts)!=200 or collection['seeds']!=list(range(config['seed_base'],config['seed_base']+200)):raise ValueError('Independent seed identity differs')
    for index,receipt in enumerate(receipts):
        if receipt['index']!=index or receipt['view']!=index//10+1 or receipt['worker']!=index%10 or receipt['photons']!=25_000_000 or receipt['pilot'] or not receipt['passed']:raise ValueError('Actual per-worker/view identity differs')
        job=jobs[index];original=registry/job['macro']
        if receipt['seed']!=collection['seeds'][index] or receipt['seed']!=job['seed'] or receipt['registered_macro_sha256']!=job['macro_sha256'] or digest(original)!=job['macro_sha256']:raise ValueError('Actual source macro/seed differs')
        folder=counts/f'worker_{index:03d}';verify_files(folder,receipt['files'])
        if read(folder/'receipt.json')!=receipt:raise ValueError('Aggregated worker receipt differs')
        if digest(folder/'source.mac')!=receipt['actual_macro_sha256'] or not source_macro_matches(original,folder/'source.mac'):raise ValueError('Actual source content differs from frozen macro')
    gate=read(physical/'physical_gate.json');verify_files(physical,gate['files'])
    permission=physical_permission(physical,counts,responses,release_manifest,physical_policy)
    if any(run.get(k)!=v for k,v in permission.items()):raise ValueError('Recorded continuation permission differs')
    if gate['collection_sha256']!=digest(counts/'collection.json') or gate['source_sha256']!=config['truth_sha256']:raise ValueError('Physical source/observation identity differs')
    for name in RESPONSES:
        f=read(responses/name/'factor_manifest.json');verify_files(responses/name,f['files'])
        if digest(responses/name/'factor_manifest.json')!=run['factor_sha256'][name]:raise ValueError('Factor identity differs')
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
        iterations=run['iterations'],counts_sha256=run['counts_sha256'],physical_gate_sha256=run['physical_gate_sha256'],factor_sha256=run['factor_sha256'],
        phase_seconds=run['phase_seconds'],slurm_maxrss_bytes=maxrss,accounting=accounting,operator_closure=closure,
        verification_source_sha256={n:digest(HERE/n) for n in ('verify_ehe.py','ehe_common.py','ehe_execution_policy.py')},files=hashes(result),**permission)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--responses',type=Path,required=True);p.add_argument('--counts',type=Path,required=True);p.add_argument('--physical',type=Path,required=True);p.add_argument('--accounting',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--verification-manifest',type=Path,required=True)
    p.add_argument('--physical-policy',type=Path)
    a=p.parse_args();verifier=read(a.verification_manifest);verify_files(HERE,verifier['files'])
    proof=verify(a.result,a.release,a.responses,a.counts,a.physical,a.accounting.read_text(),a.physical_policy);proof['verification_manifest_sha256']=digest(a.verification_manifest)
    write(a.output,proof);print('EHE_VERIFIED',proof['mode'])
