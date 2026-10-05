"""Verify bounded whole-cell pilots, frozen event identity and actual resources."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from verify_first_scatter import digest,allocated_host_bytes,verify as verify_first_scatter

CHANNELS=('440_ComptonOnly','440_SinglePlusCompton')


def verify_reused_regression(contract):
    root=contract.parent;cfg=json.loads(contract.read_text());reuse=cfg['regression_reuse']
    for name,sha in cfg['files'].items():
        if digest(root/name)!=sha:raise ValueError('Frozen resumed release differs: '+name)
    result=Path(reuse['result']);allocation=Path(reuse['allocation'])
    if digest(result/'run_manifest.json')!=reuse['run_manifest_sha256'] or digest(allocation)!=reuse['allocation_sha256']:
        raise ValueError('Previously passed regression evidence changed')
    verify_first_scatter(result,root/'legacy_regression/R1.json',root/'geometry.npz',
        Path(reuse['baseline']),'regression',allocation)
    receipt=json.loads((result/'verification.json').read_text())
    if digest(result/'verification.json')!=reuse['verification_sha256']:
        raise ValueError('Reverified historical regression differs from frozen receipt')
    if not receipt['passed'] or receipt['accepted_events']!=91231 or receipt['iterations']!=50:
        raise ValueError('Historical regression incomplete')
    print('ENERGY_V5_REUSED_REGRESSION_VERIFIED',reuse['job'])
    return receipt


def verify(result,contract,allocation):
    root=contract.parent;cfg=json.loads(contract.read_text())
    run=json.loads((result/'run_manifest.json').read_text());model=run['model']
    if model not in ('angular','continuous_energy'):raise ValueError('Unknown pilot model')
    if cfg['formal_submission_permitted'] or run['formal_submission_permitted']:
        raise ValueError('Pilot cannot authorize formal imaging')
    for name,sha in cfg['files'].items():
        if digest(root/name)!=sha:raise ValueError('Frozen release changed: '+name)
    if run['contract_sha256']!=digest(contract):raise ValueError('Pilot contract changed')
    if (run['study'],run['iterations'],run['save_step'],run['pixels_active'],run['pixels_full'])!=(
        'compton_energy_probability_v5_preflight',10,10,78920,132040):
        raise ValueError('Pilot/grid dimensions differ')
    if run['initial_density']!=1. or run['new_photons'] or run['new_fine_A_matrices']:
        raise ValueError('Pilot scope differs')
    for key,expected in [('geometry_sha256',cfg['whole_geometry_sha256']),
        ('sensitivity_sha256',cfg['files'][model+'_Sensi_full']),
        ('source_sha256',cfg['files']['run_energy_preflight_v5.py']),
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
    outputs=[]
    for channel in CHANNELS:
        paths={k:result/f'Image_{channel}_{k}.float32' for k in ('active','full','history')}
        arrays={}
        for kind,path in paths.items():
            size=132040 if kind=='full' else 78920
            if path.stat().st_size!=size*4:raise ValueError('Pilot output size differs')
            arrays[kind]=np.fromfile(path,'<f4')
            if not np.isfinite(arrays[kind]).all() or np.any(arrays[kind]<0):raise ValueError('Invalid pilot image')
        if (not np.array_equal(arrays['active'],arrays['history']) or
            not np.array_equal(arrays['full'][active],arrays['active']) or np.any(arrays['full'][inactive]!=0)):
            raise ValueError('Pilot last frame/full support differs')
        outputs.append(dict(channel=channel,frames=1,sha256={k:digest(p) for k,p in paths.items()}))
    receipt=dict(passed=True,model=model,iterations=10,accepted_events=91225,outputs=outputs,
        resources=resources,contract_sha256=digest(contract),run_manifest_sha256=digest(result/'run_manifest.json'),
        allocation_sha256=digest(allocation),formal_submission_permitted=False)
    (result/'verification.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print('ENERGY_V5_PREFLIGHT_VERIFIED',model,91225)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--result',type=Path);p.add_argument('--allocation',type=Path)
    p.add_argument('--reuse-regression-only',action='store_true')
    a=p.parse_args()
    if a.reuse_regression_only:verify_reused_regression(a.contract)
    else:
        if a.result is None or a.allocation is None:p.error('result and allocation are required for a pilot')
        verify(a.result,a.contract,a.allocation)
