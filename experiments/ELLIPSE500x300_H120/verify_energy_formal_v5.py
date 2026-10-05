"""Strict current-publication identity, resource, checkpoint and frame verification."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from verify_first_scatter import digest,allocated_host_bytes
from energy_formal_v5_contract import load_contract,verify_checkpoints,CHANNELS,validate_transport_collection


def verify(result,contract,allocation,mode,write_receipt=True):
    root=contract.parent;iterations,save_step={'validation':(10,10),'formal':(2000,50)}[mode]
    cfg=load_contract(contract,mode,iterations,save_step)
    run=json.loads((result/'run_manifest.json').read_text());model=run['model']
    if model not in ('angular','continuous_energy'):raise ValueError('Unknown pilot model')
    if mode=='formal':validate_authority(Path(run['authority_file']),run['authority_sha256'],contract)
    for name,sha in cfg['files'].items():
        if digest(root/name)!=sha:raise ValueError('Frozen release changed: '+name)
    if run['contract_sha256']!=digest(contract):raise ValueError('Pilot contract changed')
    if (run['study'],run['iterations'],run['save_step'],run['pixels_active'],run['pixels_full'])!=(
        'compton_energy_probability_v5_formal',iterations,save_step,78920,132040):
        raise ValueError('Pilot/grid dimensions differ')
    if run['initial_density']!=1. or run['new_photons'] or run['new_fine_A_matrices']:
        raise ValueError('Pilot scope differs')
    for key,expected in [('geometry_sha256',cfg['whole_geometry_sha256']),
        ('sensitivity_sha256',cfg['files'][model+'_Sensi_full']),
        ('source_sha256',cfg['files']['run_energy_formal_v5.py']),
        ('input_sha256',cfg['input_sha256']),('factor_manifest_sha256',cfg['factor_manifest_sha256'])]:
        if run[key]!=expected:raise ValueError('Frozen identity differs: '+key)
    counts=cfg['events_per_view'];resources=run['resources'];world=run['world_size']
    if world not in (4,8) or sorted(r['rank'] for r in resources)!=list(range(world)):
        raise ValueError('Distinct rank closure differs')
    if len({r['node'] for r in resources})!=world:raise ValueError('Distinct nodes required')
    if (run['accepted_compton_events']!=91225 or sum(counts)!=91225 or
        run['accepted_compton_events_per_view']!=counts or sum(r['accepted_events'] for r in resources)!=91225):
        raise ValueError('Complete event closure differs')
    actual=allocated_host_bytes(allocation,world)
    for r in resources:
        if (r['host_allocated_bytes']!=actual or actual<=0 or
            r['peak_reserved_bytes']/r['total_device_bytes']>.8 or r['host_peak_rss_bytes']/actual>.8):
            raise ValueError('Actual resource margin fails')
        check=r['numerical_checks']
        if (check['model']!=model or check['sample_events']!=32 or check['active_points']!=78920 or
            check['full_points']!=132040 or check['source_truth_used'] or check['threshold']!=1e-5):
            raise ValueError('Actual-row numerical check identity differs')
        for name in ('chunk_relative_L2','rank_weight_relative_L2','forward_adjoint_relative_error',
                     'event_constant_relative_error','original_sparse_relative_L2'):
            value=check[name]
            if value is None:
                if name!='original_sparse_relative_L2' or model!='continuous_energy':raise ValueError('Missing numerical check')
            elif not np.isfinite(value) or value>1e-5 or value<0:raise ValueError('Numerical equivalence failed: '+name)
    for v,count in enumerate(counts,1):
        indices=np.load(root/f'selections/{v}.npy');cursor=0
        if len(indices)!=count:raise ValueError('Frozen view count differs')
        for r in sorted(resources,key=lambda x:x['rank']):
            part=r['partitions'][v-1];lo=part['first_selected_position'];hi=part['last_selected_position_exclusive']
            if part['view']!=v or lo!=cursor or hi<lo or part['events']!=hi-lo:
                raise ValueError('Event partition missing/overlapping')
            sha=hashlib.sha256(indices[lo:hi].astype('<i8').tobytes()).hexdigest()
            if sha!=part['original_rows_sha256']:raise ValueError('Rank event identity differs')
            cursor=hi
        if cursor!=count:raise ValueError('View partition incomplete')
    g=np.load(root/'whole_geometry.npz');active=g['active_indices'];inactive=np.ones(132040,bool);inactive[active]=False
    outputs=[];histories={}
    for channel in CHANNELS:
        paths={k:result/f'Image_{channel}_{k}.float32' for k in ('active','full','history')}
        arrays={}
        for kind,path in paths.items():
            size=132040 if kind=='full' else 78920*(iterations//save_step if kind=='history' else 1)
            if path.stat().st_size!=size*4:raise ValueError('Pilot output size differs')
            arrays[kind]=np.fromfile(path,'<f4')
            if kind=='history':arrays[kind]=arrays[kind].reshape(iterations//save_step,78920)
            if not np.isfinite(arrays[kind]).all() or np.any(arrays[kind]<0):raise ValueError('Invalid pilot image')
        if (not np.array_equal(arrays['active'],arrays['history'][-1]) or
            not np.array_equal(arrays['full'][active],arrays['active']) or np.any(arrays['full'][inactive]!=0)):
            raise ValueError('Pilot last frame/full support differs')
        item=dict(channel=channel,frames=iterations//save_step,sha256={k:digest(p) for k,p in paths.items()})
        histories[channel]=arrays['history']
        if mode=='validation':
            reference=root/'preflight_evidence'/model/'reference'/paths['active'].name
            if digest(reference)!=cfg['files'][reference.relative_to(root).as_posix()]:
                raise ValueError('Frozen original pilot reference differs')
            old=np.fromfile(reference,'<f4').astype(np.float64);now=arrays['active'].astype(np.float64)
            error=float(np.linalg.norm(now-old)/max(np.linalg.norm(old),1e-300))
            if not np.isfinite(error) or error>1e-5:raise ValueError('Formal entry differs from complete-event pilot')
            item['pilot_frame10_relative_L2']=error
        outputs.append(item)
    checkpoints=verify_checkpoints(result,histories,active,132040,model,digest(contract)) if mode=='formal' else []
    if (run.get('mode')!=mode or run.get('preflight_job')!=cfg['preflight_job'] or
        run.get('actual_primary_gamma')!=1_000_000_000 or run.get('workers')!=200 or run.get('views')!=20):
        raise ValueError('Formal execution/transport closure differs')
    original=json.loads((root/'transport_collection.json').read_text())
    if (digest(root/'transport_collection.json')!=cfg['input_sha256']['collections/NEMA_Body_H60_1e9.json'] or
        run.get('transport')!=validate_transport_collection(original)):
        raise ValueError('Actual energy-category counts/worker/seed identity differs')
    receipt=dict(passed=True,model=model,mode=mode,iterations=iterations,save_step=save_step,accepted_events=91225,outputs=outputs,checkpoints=checkpoints,
        resources=resources,contract_sha256=digest(contract),run_manifest_sha256=digest(result/'run_manifest.json'),
        allocation_sha256=digest(allocation),authority_sha256=run.get('authority_sha256'))
    if write_receipt:(result/'verification.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print('ENERGY_V5_FORMAL_VERIFIED',mode,model,91225)
    return receipt


def validate_authority(path,expected_sha,contract):
    if path is None or expected_sha is None or digest(path)!=expected_sha:
        raise ValueError('Pinned validation authority is required before formal imaging')
    authority=json.loads(path.read_text())
    if (authority['contract_sha256']!=digest(contract) or authority['iterations']!=2000 or
        authority['save_step']!=50 or not authority['passed'] or
        set(authority['models'])!={'angular','continuous_energy'}):
        raise ValueError('Formal authority differs')
    for model,record in authority['models'].items():
        result=Path(record['result']);allocation=Path(record['allocation'])
        for name,sha in record['sha256'].items():
            path=allocation if name=='allocation.txt' else result/name
            if digest(path)!=sha:raise ValueError('Validated formal-entry evidence changed')
        receipt=verify(result,contract,allocation,'validation',write_receipt=False)
        if receipt['model']!=model:raise ValueError('Wrong validated model')
    return authority


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('result','contract','allocation'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--mode',choices=('validation','formal'),required=True)
    a=p.parse_args();verify(a.result,a.contract,a.allocation,a.mode)
