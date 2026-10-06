"""Strict current-publication identity, resource, checkpoint and frame verification."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from verify_first_scatter import digest,allocated_host_bytes
from energy_full10000_v5_contract import load_contract,verify_checkpoints,CHANNELS,STUDY,execution_policy
from energy_5e9_v5_contract import validate_transport_collection


def verify_support_record(record):
    """Accepted events must have actual support; small positive rows are retained.

    The sub-1e-12 row-sum count is diagnostic. It does not impose a new cut or
    change the original MLEM forward floor, and is not a later-iteration count.
    """
    minimum=record['minimum_active_event_mass'];count=record['initial_forward_floor_events']
    if not np.isfinite(minimum) or minimum<=0:
        raise ValueError('Frozen event has invalid active-basis response')
    if type(count) is not int or not 0<=count<=record['accepted_events']:
        raise ValueError('Invalid original forward-floor diagnostic')


def verify(result,contract,allocation,mode,write_receipt=True):
    root=contract.parent;iterations,save_step=execution_policy(mode)
    cfg=load_contract(contract,mode,iterations,save_step)
    run=json.loads((result/'run_manifest.json').read_text());model=run['model']
    if model!='continuous_energy':raise ValueError('Unknown pilot model')
    if mode=='formal':validate_authority(Path(run['authority_file']),run['authority_sha256'],contract)
    for name,sha in cfg['files'].items():
        if digest(root/name)!=sha:raise ValueError('Frozen release changed: '+name)
    if run['contract_sha256']!=digest(contract):raise ValueError('Pilot contract changed')
    if (run['study'],run['iterations'],run['save_step'],run['pixels_active'],run['pixels_full'])!=(
        STUDY,iterations,save_step,78920,132040):
        raise ValueError('Pilot/grid dimensions differ')
    if run.get('event_policy')!='legacy' or run['initial_density']!=1. or run['new_photons'] or run['new_fine_A_matrices']:
        raise ValueError('Pilot scope differs')
    for key,expected in [('geometry_sha256',cfg['whole_geometry_sha256']),
        ('sensitivity_sha256',cfg['files'][model+'_Sensi_full']),
        ('source_sha256',cfg['files']['run_energy_full10000_v5.py']),
        ('input_sha256',cfg['input_sha256']),('factor_manifest_sha256',cfg['factor_manifest_sha256']),
        ('factor_payload_sha256',cfg['factor_payload_sha256'])]:
        if run[key]!=expected:raise ValueError('Frozen identity differs: '+key)
    counts=cfg['events_per_view'];resources=run['resources'];world=run['world_size']
    if world!=cfg['nodes'] or sorted(r['rank'] for r in resources)!=list(range(world)):
        raise ValueError('Distinct rank closure differs')
    if len({r['node'] for r in resources})!=world:raise ValueError('Distinct nodes required')
    if (run['accepted_compton_events']!=cfg['accepted_events'] or sum(counts)!=cfg['accepted_events'] or
        run['accepted_compton_events_per_view']!=counts or sum(r['accepted_events'] for r in resources)!=cfg['accepted_events']):
        raise ValueError('Complete event closure differs')
    actual=allocated_host_bytes(allocation,world)
    for r in resources:
        verify_support_record(r)
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
        outputs.append(item)
    checkpoints=verify_checkpoints(result,histories,active,digest(contract),mode)
    if run['channels']!=list(CHANNELS) or run['helper_sha256']!=cfg['files']:
        raise ValueError('Six-output release/algorithm identity differs')
    if run['cross_prediction_source']!='440_SinglePhoton_final':raise ValueError('Cross-talk estimator changed')
    for combined,left,right in [('440SinglePlus218Single','440_SinglePhoton','218_SinglePhoton_CrossTalkCorrected'),
        ('440SingleComptonPlus218Single','440_SinglePlusCompton','218_SinglePhoton_CrossTalkCorrected')]:
        if not np.array_equal(histories[combined],histories[left]+histories[right]):raise ValueError('Composite history is not the specified gamma-density sum')
    prediction=result/'PredictedCntStat_218_From440.float32'
    if prediction.stat().st_size!=10496*20*4:raise ValueError('Cross-talk prediction shape differs')
    values=np.fromfile(prediction,'<f4')
    if not np.isfinite(values).all() or np.any(values<0):raise ValueError('Invalid fixed cross-talk prediction')
    regression={}
    reference=cfg['validation_reference'] if mode=='validation' else cfg['formal_prefix_reference']
    for name in ('run_manifest.json','verification.json'):
        if digest(Path(reference['result'])/name)!=reference['sha256'][name]:raise ValueError('Frozen delivered reference proof changed')
    frame_index=0 if mode=='validation' else 39
    for channel in ('440_ComptonOnly','440_SinglePlusCompton'):
        name='Image_'+channel+'_active.float32';path=Path(reference['result'])/name
        if digest(path)!=reference['sha256'][name]:raise ValueError('Frozen actual reference image changed')
        original=np.fromfile(path,'<f4');current=histories[channel][frame_index]
        value=float(np.linalg.norm(current.astype(float)-original.astype(float))/np.linalg.norm(original.astype(float)))
        if not np.isfinite(value) or value>1e-5:raise ValueError('Actual delivered kernel/2000-prefix regression failed')
        regression[channel]=value
    if mode=='validation':
        for resource in resources:
            checks=resource['single_regression_relative_L2']
            if set(checks)!={'440_SinglePhoton','218_SinglePhoton_CrossTalkCorrected'} or any(not np.isfinite(x) or not 0<=x<=1e-5 for x in checks.values()):
                raise ValueError('Actual full single-MLEM equivalence failed')
    else:
        recorded=run['prefix_regression_relative_L2']
        if set(recorded)!=set(regression) or any(not np.isfinite(x) or not 0<=x<=1e-5 for x in recorded.values()):
            raise ValueError('Missing actual 2000-prefix gate')
    if (run.get('mode')!=mode or run.get('calibration_release')!=cfg['calibration_release'] or
        run.get('actual_primary_gamma')!=5_000_000_000 or run.get('workers')!=200 or run.get('views')!=20):
        raise ValueError('Formal execution/transport closure differs')
    original=json.loads((root/'transport_collection.json').read_text())
    if (digest(root/'transport_collection.json')!=cfg['input_sha256']['collections/NEMA_Body_H60_5e9.json'] or
        run.get('transport')!=validate_transport_collection(original)):
        raise ValueError('Actual energy-category counts/worker/seed identity differs')
    receipt=dict(passed=True,model=model,mode=mode,iterations=iterations,save_step=save_step,accepted_events=cfg['accepted_events'],outputs=outputs,checkpoints=checkpoints,
        prediction_sha256=digest(prediction),reference_regression_relative_L2=regression,resources=resources,contract_sha256=digest(contract),run_manifest_sha256=digest(result/'run_manifest.json'),
        allocation_sha256=digest(allocation),authority_sha256=run.get('authority_sha256'))
    if write_receipt:
        path=result/'verification.json';data=(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode()
        if path.exists() and path.read_bytes()!=data:raise ValueError('Existing proof differs; never overwrite frozen evidence')
        if not path.exists():path.write_bytes(data)
    print('ENERGY_FULL10000_V5_VERIFIED',mode,model,cfg['accepted_events'])
    return receipt


def validate_authority(path,expected_sha,contract):
    if path is None or expected_sha is None or digest(path)!=expected_sha:
        raise ValueError('Pinned actual full six-channel validation required')
    authority=json.loads(path.read_text())
    if (authority['contract_sha256']!=digest(contract) or authority['iterations']!=10000 or
        authority['save_step']!=50 or not authority['passed'] or
        set(authority['models'])!={'continuous_energy'}):raise ValueError('Full-10000 authority differs')
    record=authority['models']['continuous_energy'];result=Path(record['result']);allocation=Path(record['allocation'])
    for name,sha in record['sha256'].items():
        file=allocation if name=='allocation.txt' else result/name
        if digest(file)!=sha:raise ValueError('Actual full-entry validation evidence changed')
    verify(result,contract,allocation,'validation',write_receipt=False)
    return authority


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('result','contract','allocation'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--mode',choices=('validation','formal'),required=True)
    p.add_argument('--read-only',action='store_true')
    a=p.parse_args();verify(a.result,a.contract,a.allocation,a.mode,write_receipt=not a.read_only)
