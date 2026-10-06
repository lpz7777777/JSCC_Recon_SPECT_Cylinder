"""Isolated legacy 5e9 contract and bounded durable paired checkpoints."""
import json
import os
from pathlib import Path
import shutil
import numpy as np
import torch
from run_reconstruction import collect_response,digest
from prepare_energy_5e9_v5 import STUDY,validate_transport as validate_transport_collection

CHANNELS=('440_ComptonOnly','440_SinglePlusCompton')

def load_contract(path,mode,iterations,save_step):
    path=Path(path);root=path.parent;cfg=json.loads(path.read_text())
    expected={'validation':(10,10),'formal':(2000,50)}
    if mode not in expected or (iterations,save_step)!=expected[mode]:
        raise ValueError('Explicit validation10 or formal2000/save50 required')
    if (cfg['study']!=STUDY or cfg['iterations']!=2000 or cfg['save_step']!=50
        or cfg['new_photons']!=0 or cfg['new_training'] or cfg['event_policy']!='legacy'
        or cfg['models']!=['angular','continuous_energy'] or cfg['channels']!=list(CHANNELS)
        or cfg['nodes']!=8 or cfg['accepted_events']<=0 or cfg['accepted_events']>484936
        or len(cfg['events_per_view'])!=20 or sum(cfg['events_per_view'])!=cfg['accepted_events']):
        raise ValueError('Unique legacy 5e9 paired contract differs')
    for name,sha in cfg['files'].items():
        if digest(root/name)!=sha:raise ValueError('Frozen 5e9 payload differs: '+name)
    collection=json.loads((root/'transport_collection.json').read_text())
    validate_transport_collection(collection)
    gate=json.loads((root/'calibration/calibration_gate.json').read_text())
    if gate['study']!=STUDY or gate['status']!='PASSED' or gate['failures'] or gate['event_policy']!='legacy':
        raise ValueError('Actual independent legacy calibration must pass')
    for name,sha in gate['files'].items():
        if digest(root/'calibration'/name)!=sha:raise ValueError('Calibration evidence changed')
    selection=json.loads((root/'selection_gate.json').read_text())
    if (selection['study']!=STUDY or selection['status']!='PASSED' or selection['event_policy']!='legacy'
        or selection['nema_kept']!=cfg['accepted_events'] or selection['original_nema_accepted']!=484936
        or selection['geometry_mode']!='stable_float64' or selection['grid_points']!=132040):
        raise ValueError('Complete stable legacy selection required')
    if digest(root/'selection_gate.json')!=gate['selection_gate_sha256']:
        raise ValueError('Calibration and NEMA selection release differ')
    nema=sorted((r for r in selection['records'] if r['dataset']=='NEMA'),key=lambda r:r['view'])
    if [r['view'] for r in nema]!=list(range(1,21)) or [r['kept'] for r in nema]!=cfg['events_per_view']:
        raise ValueError('Twenty-view selection closure differs')
    for r in nema:
        if digest(root/f"selections/{r['view']}.npy")!=r['selection_sha256']:
            raise ValueError('NEMA raw-row identity changed')
    for model in cfg['models']:
        if digest(root/(model+'_Sensi_full'))!=gate['files'][model+'_Sensi_full']:
            raise ValueError('Wrong policy sensitivity')
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
        record=dict(study='compton_energy_probability_v5_5e9',model=model,iteration=iteration,
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
            'compton_energy_probability_v5_5e9',model,i,contract_sha):raise ValueError('Checkpoint identity changed')
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
