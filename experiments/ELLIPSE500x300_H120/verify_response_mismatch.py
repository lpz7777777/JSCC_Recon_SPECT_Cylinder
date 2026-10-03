"""Gate the filter-off regression, full-data pilot and two-channel formal run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--mode',choices=('regression','pilot','formal'),required=True)
    args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    if digest(args.geometry)!=cfg['geometry_sha256']: raise ValueError('Frozen geometry differs')
    active_indices=np.load(args.geometry)['active_indices']
    inactive=np.ones(132040,dtype=bool); inactive[active_indices]=False
    run=json.loads((args.result/'run_manifest.json').read_text())
    steps={'regression':(50,50),'pilot':(10,10),'formal':(10000,50)}
    iterations,save=steps[args.mode]
    if (run['iterations'],run['save_step'])!=(iterations,save): raise ValueError('Iteration contract differs')
    if run['world_size']!=8 or run['pixels_active']!=82040 or run['pixels_full']!=132040:
        raise ValueError('Topology or full-grid geometry differs')
    if sorted(r['rank'] for r in run['resources'])!=list(range(8)) or len({r['node'] for r in run['resources']})!=8:
        raise ValueError('Eight unique ranks on eight distinct nodes required')
    if (run['input_sha256']!=cfg['baseline_input_sha256'] or
        run['factor_manifest_sha256']!=cfg['baseline_factor_manifest_sha256'] or
        run['geometry_sha256']!=cfg['geometry_sha256']):
        raise ValueError('Frozen input/geometry/Factor contract differs')
    filtered=args.mode!='regression'
    info=run['response_mismatch']
    if info['filter_enabled']!=filtered or info['config_sha256']!=digest(args.config):
        raise ValueError('Wrong filter mode or config hash')
    expected=cfg['kept_compton_events'] if filtered else 484936
    if run['accepted_compton_events']!=expected or sum(r['accepted_events'] for r in run['resources'])!=expected:
        raise ValueError('Rank/total event closure differs')
    expected_views=cfg['kept_per_view'] if filtered else cfg['baseline_per_view']
    if run['accepted_compton_events_per_view']!=expected_views:
        raise ValueError('Per-view selection differs from full scan')
    if run['sensi_d_sha256']!=(cfg['sensi_d_sha256'] if filtered else cfg['baseline_sensi_d_sha256']):
        raise ValueError('Sensitivity differs from frozen mode')
    outputs=[]
    for channel in cfg['channels']:
        paths={key:args.result/f'Image_{channel}_{key}.float32' for key in ('active','full','history')}
        sizes={'active':82040,'full':132040,'history':82040*(iterations//save)}
        for key,path in paths.items():
            if path.stat().st_size!=sizes[key]*4: raise ValueError(f'Wrong {channel}/{key} bytes')
            values=np.memmap(path,'<f4',mode='r')
            if not np.isfinite(values).all() or np.any(values<0): raise ValueError('Invalid image values')
        active=np.fromfile(paths['active'],'<f4')
        full=np.fromfile(paths['full'],'<f4')
        if not np.array_equal(full[active_indices],active) or np.any(full[inactive]!=0):
            raise ValueError('Full image must be zero outside the ellipse and match active columns')
        history=np.memmap(paths['history'],'<f4',mode='r',shape=(iterations//save,82040))
        if not np.array_equal(active,history[-1]): raise ValueError('Final does not equal last saved frame')
        entry={'channel':channel,'frames':iterations//save,'sha256':{k:digest(v) for k,v in paths.items()}}
        if args.mode=='regression':
            original=np.memmap(args.baseline/f'Image_{channel}_history.float32','<f4',mode='r',shape=(200,82040))[0]
            error=float(np.linalg.norm(active.astype(np.float64)-original)/np.linalg.norm(original.astype(np.float64)))
            entry['relative_l2_baseline_frame50']=error
            if error>1e-5: raise ValueError(f'Baseline regression differs: {error}')
        outputs.append(entry)
    resource=[]
    for r in run['resources']:
        gpu=r['peak_reserved_bytes']/r['total_device_bytes']
        allocated=r['host_allocated_bytes']
        if allocated<=0: raise ValueError('Actual allocated host memory missing')
        host=r['host_peak_rss_bytes']/allocated
        if max(gpu,host)>.8: raise ValueError('GPU/host 20% margin failed')
        resource.append({'rank':r['rank'],'node':r['node'],'gpu_reserved_fraction':gpu,'host_rss_fraction':host,
                         'allocated_host_bytes':allocated})
    report={'study':cfg['study'],'mode':args.mode,'passed':True,'result':str(args.result),
        'iterations':iterations,'save_step':save,'accepted_events':expected,'outputs':outputs,'resources':resource,
        'run_manifest_sha256':digest(args.result/'run_manifest.json'),'config_sha256':digest(args.config)}
    (args.result/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
