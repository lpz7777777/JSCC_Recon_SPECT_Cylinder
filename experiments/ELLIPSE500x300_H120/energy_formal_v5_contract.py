"""Bounded formal authority and durable checkpoints; no response/MLEM changes."""
import json
import os
from pathlib import Path
import shutil
import numpy as np
import torch
from run_reconstruction import collect_response,digest

CHANNELS=('440_ComptonOnly','440_SinglePlusCompton')


def validate_transport_collection(collection):
    """PrimaryCount aggregates energy categories, not individual workers."""
    counts=collection['primary_counts']
    if (len(counts)!=3 or any(type(n) is not int or n<0 for n in counts) or
        sum(counts)!=1_000_000_000 or
        collection['seeds']!=list(range(30093001,30093201)) or
        collection['worker_indices']!=list(range(200)) or
        collection['views']!=list(range(1,21)) or
        collection['dataset']!='NEMA_Body_H60' or collection['level']!='1e9' or
        collection['event_policy']!='ideal'):
        raise ValueError('Original transport closure failed')
    return dict(actual_primary_gamma=sum(counts),primary_by_energy=counts,
        workers=len(collection['worker_indices']),views=len(collection['views']),
        seeds=collection['seeds'],worker_indices=collection['worker_indices'])


def load_contract(path,mode,iterations,save_step):
    path=Path(path);root=path.parent;cfg=json.loads(path.read_text())
    expected={'validation':(10,10),'formal':(2000,50)}
    if mode not in expected or (iterations,save_step)!=expected[mode]:
        raise ValueError('Explicit validation10 or formal2000/save50 required')
    if (cfg['study']!='compton_energy_probability_v5_formal' or
        cfg['iterations']!=2000 or cfg['save_step']!=50 or cfg['accepted_events']!=91225 or
        cfg['new_photons']!=0 or cfg['models']!=['angular','continuous_energy'] or
        cfg['channels']!=list(CHANNELS)):
        raise ValueError('Unique bounded paired imaging contract differs')
    for name,sha in cfg['files'].items():
        if digest(root/name)!=sha:raise ValueError('Frozen formal payload differs: '+name)
    prior=json.loads((root/'preflight_contract.json').read_text())
    summary=json.loads((root/'preflight_summary.json').read_text())
    if summary['status']!='PASSED' or summary['job']!=cfg['preflight_job']:
        raise ValueError('Current complete-event preflight has not actually passed')
    if {p['phase'] for p in summary['phases']}!={'regression','angular','continuous_energy'}:
        raise ValueError('All three preflight phases required')
    for name,sha in prior['files'].items():
        if cfg['files'].get(name)!=sha:raise ValueError('Validated code/data/operator changed: '+name)
    for key in ('input_sha256','factor_manifest_sha256','whole_geometry_sha256','events_per_view'):
        if cfg[key]!=prior[key]:raise ValueError('Paired data identity differs: '+key)
    if sum(cfg['events_per_view'])!=91225:raise ValueError('Complete frozen events required')
    for phase in summary['phases']:
        folder=root/'preflight_evidence'/phase['phase']
        if (digest(folder/'verification.json')!=phase['verification_sha256'] or
            digest(folder/'run_manifest.json')!=phase['run_manifest_sha256']):
            raise ValueError('Actual preflight proof changed')
        proof=json.loads((folder/'verification.json').read_text())
        if not proof['passed']:raise ValueError('Preflight proof is not successful')
    return cfg


def sync_file(path):
    with Path(path).open('rb+') as f:os.fsync(f.fileno())


def write_checkpoint(output,model,iteration,history_d,history_j,geometry,contract_sha):
    """Publish a complete two-channel snapshot only after all files are durable."""
    if not 0<iteration<=2000 or iteration%50:raise ValueError('Bounded save50 checkpoint required')
    if len(history_d)!=iteration//50 or len(history_j)!=iteration//50:
        raise ValueError('Checkpoint histories incomplete')
    output=Path(output);target=output/f'checkpoint_{iteration:06d}'
    temp=output/f'.checkpoint_{iteration:06d}.partial'
    if target.exists() or temp.exists():raise ValueError('Checkpoint already exists; never overwrite')
    temp.mkdir()
    try:
        images={}
        for name,history in zip(CHANNELS,(history_d,history_j)):
            frame=history[-1]
            if frame.numel()!=geometry.active_count or not bool(torch.isfinite(frame).all()) or bool((frame<0).any()):
                raise ValueError('Invalid persistent checkpoint')
            collect_response(temp,name,frame,None,geometry,0)
            images[name]={}
            for kind in ('active','full'):
                p=temp/f'Image_{name}_{kind}.float32';sync_file(p);images[name][kind]=digest(p)
        record=dict(study='compton_energy_probability_v5_formal',model=model,iteration=iteration,
            contract_sha256=contract_sha,outputs=images)
        p=temp/'checkpoint_manifest.json';p.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');sync_file(p)
        temp.rename(target)
        if os.name=='posix':
            fd=os.open(str(output),os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
        return record
    except Exception:
        # Failed unpublished snapshots are small, generated files beneath output.
        if temp.resolve().parent!=output.resolve():raise ValueError('Unsafe checkpoint cleanup')
        shutil.rmtree(temp)
        raise


def verify_checkpoints(result,history,active,full_count,model,contract_sha):
    result=Path(result);frames=40
    expected={f'checkpoint_{i:06d}' for i in range(50,2001,50)}
    actual={p.name for p in result.glob('checkpoint_*') if p.is_dir()}
    if actual!=expected:raise ValueError('Forty unique persistent checkpoints required')
    records=[];inactive=np.ones(full_count,bool);inactive[active]=False
    for frame,i in enumerate(range(50,2001,50)):
        folder=result/f'checkpoint_{i:06d}';p=folder/'checkpoint_manifest.json';r=json.loads(p.read_text())
        if (r['study'],r['model'],r['iteration'],r['contract_sha256'])!=(
            'compton_energy_probability_v5_formal',model,i,contract_sha):raise ValueError('Checkpoint identity changed')
        for name in CHANNELS:
            for kind,count in (('active',len(active)),('full',full_count)):
                f=folder/f'Image_{name}_{kind}.float32'
                if f.stat().st_size!=count*4 or digest(f)!=r['outputs'][name][kind]:
                    raise ValueError('Checkpoint size/SHA changed')
            a=np.fromfile(folder/f'Image_{name}_active.float32','<f4')
            f=np.fromfile(folder/f'Image_{name}_full.float32','<f4')
            if (not np.array_equal(a,history[name][frame]) or not np.array_equal(f[active],a) or
                np.any(f[inactive]!=0) or not np.isfinite(f).all() or np.any(f<0)):
                raise ValueError('Checkpoint/history/full support differs')
        records.append(dict(iteration=i,manifest_sha256=digest(p)))
    return records
