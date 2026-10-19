"""Independent CPU-only checks of all selected raw rows and every saved image."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import shutil
import tarfile
import time
import numpy as np
from derenzo_campaign_common import *
from derenzo_campaign_reconstruction import verify_resources
from verify_jscc_5e10 import own_s,rel


def verify(a):
    started=time.monotonic();verify_files(a.release,read(a.release/'release_manifest.json')['sha256'])
    cfg=read(a.release/'gpu_campaign.json');spec=cfg['datasets'][a.dataset];info=dataset_info(a.dataset)
    collection=validate_collection(read(a.input/'collection.json'),a.dataset);verify_files(a.input,collection['files'])
    plan=read(a.plan);before=hashes(a.result)
    if a.mode=='selection':
        m=read(a.result/'selection_manifest.json');verify_files(a.result,m['files'])
        if a.dataset!='JSCC_440' or m['collection_sha256']!=digest(a.input/'collection.json'):raise ValueError('Actual source selection identity differs')
        records=m['resources'];verify_resources(records,plan,a.allocation)
        total=0
        for v in range(1,21):
            raw=np.loadtxt(a.input/'List'/f'{v}.csv',delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
            indices=np.load(a.result/'selections'/f'{v}.npy');rows=np.load(a.result/'selected_rows'/f'{v}.npy')
            if len(indices)!=m['events_per_view'][v-1] or np.any(indices<0) or np.any(indices>=len(raw)) or np.any(indices[1:]<=indices[:-1]):raise ValueError('All selected row indices must close without overlaps')
            if not np.array_equal(rows,raw[indices]):raise ValueError('Selected cache differs from original raw rows')
            cursor=0;merged=[];merged_rows=[]
            for r in range(plan['world_size']):
                name=f'{v:02d}_rank{r:02d}';p=read(a.result/'parts'/(name+'.json'));verify_files(a.result/'parts',p['files'])
                if p['offset']!=cursor or p['kept']+p['removed']!=p['original_accepted']:raise ValueError('Rank receipt/raw offset closure failed')
                ix=np.load(a.result/'parts'/(name+'.npy'));rr=np.load(a.result/'parts'/(name+'_rows.npy'))
                if len(ix)!=p['kept'] or not np.array_equal(rr,raw[ix]):raise ValueError('Every partition array must equal original rows')
                cursor+=p['raw_rows'];merged.append(ix);merged_rows.append(rr)
            if cursor!=len(raw) or not np.array_equal(np.concatenate(merged),indices) or not np.array_equal(np.concatenate(merged_rows),rows):raise ValueError('All raw/rank/final arrays do not close')
            total+=len(raw);print('DERENZO_ALL_SELECTED_RAW_ROWS_VERIFIED',v,len(rows),flush=True)
        if total!=collection['raw_list_rows'] or sum(m['events_per_view'])!=m['accepted_events'] or m['accepted_events']<=0:raise ValueError('All selected event counts must close')
        extra=dict(accepted_events=m['accepted_events'],events_per_view=m['events_per_view'],all_selected_raw_rows_exact=True)
    else:
        run=read(a.result/'run_manifest.json');records=run['resources'];verify_resources(records,plan,a.allocation)
        iterations,step=(10,10) if a.mode=='validation' else (info['iterations'],info['save_step'])
        if (run['study'],run['dataset'],run['mode'],run['channels'],run['iterations'],run['save_step'])!=(STUDY,a.dataset,a.mode,info['channels'],iterations,step):raise ValueError('Exact authorized output routes/iterations required')
        if (run['plan_sha256']!=digest(a.plan) or run['collection_sha256']!=digest(a.input/'collection.json')
                or run['release_sha256']!=digest(a.release/'release_manifest.json') or run['initial_density']!=1 or run['regularization']!='none'):
            raise ValueError('Actual input/algorithm identity differs')
        if run['factor_manifest_sha256']!=spec['factor_manifest_sha256']:raise ValueError('Executed complete matrix differs')
        geo=np.load(a.release/'whole_geometry.npz');active=geo['active_indices'];inactive=np.ones(132040,bool);inactive[active]=False
        allowed={f'Image_{c}_{k}.float32' for c in info['channels'] for k in ['active','full','history']}
        if {p.name for p in a.result.glob('Image_*.float32')}!=allowed:raise ValueError('Missing/extra unauthorized image channel')
        for c in info['channels']:
            history=np.fromfile(a.result/f'Image_{c}_history.float32','<f4').reshape(iterations//step,78920)
            image=np.fromfile(a.result/f'Image_{c}_active.float32','<f4');full=np.fromfile(a.result/f'Image_{c}_full.float32','<f4')
            if image.shape!=(78920,) or full.shape!=(132040,) or not np.isfinite(history).all() or np.any(history<0):raise ValueError('Complete finite nonnegative histories required')
            if not np.array_equal(history[-1],image) or not np.array_equal(full[active],image) or np.any(full[inactive]!=0):raise ValueError('Final/history/full active support differs')
            parent=a.result/'checkpoints'/c
            expected={f'checkpoint_{i:06d}' for i in range(step,iterations+1,step)}
            if {p.name for p in parent.iterdir()}!=expected:raise ValueError('Missing/extra/partial checkpoints')
            for frame,i in enumerate(range(step,iterations+1,step)):
                folder=parent/f'checkpoint_{i:06d}';m=read(folder/'manifest.json');verify_files(folder,m['files'])
                if (m['iteration'],m['channel'],m['plan_sha256'])!=(i,c,digest(a.plan)):raise ValueError('Checkpoint execution identity differs')
                x=np.fromfile(folder/f'Image_{c}_active.float32','<f4');y=np.fromfile(folder/f'Image_{c}_full.float32','<f4')
                if not np.array_equal(x,history[frame]) or not np.array_equal(y[active],x) or np.any(y[inactive]!=0):raise ValueError('Every checkpoint must equal original history')
        if a.mode=='validation':
            for r in records:
                if set(r['regression_relative_L2'])!=set(info['channels']) or any(not math.isfinite(v) or v>1e-5 for v in r['regression_relative_L2'].values()):raise ValueError('Original MLEM regression failed')
                n=r['operator_checks'];errors=n['forward_relative_L2']+n['adjoint_relative_L2']+[n['sensitivity_relative_L2']]
                if n['views']!=20 or any(not math.isfinite(x) or x>1e-5 for x in errors):raise ValueError('All matrix views/rows/transpose/S validation failed')
        verify_files(Path(spec['factor_root']),spec['factor_files'])
        matrix=np.memmap(Path(spec['factor_root'])/'SysMat_polar','<f4',mode='r',shape=(132040,info['bins']))
        expected_s=own_s(matrix,geo);actual_s=np.fromfile(a.result/'S_single.float32','<f4');s_error=rel(actual_s,expected_s)
        if not math.isfinite(s_error) or s_error>1e-5:raise ValueError('Independent full matrix own sensitivity differs')
        extra=dict(iterations=iterations,save_step=step,channels=info['channels'],own_s_relative_L2=s_error)
        if a.dataset=='JSCC_440':
            selection=Path(run['selection']);m=read(selection/'selection_manifest.json');verify_files(selection,m['files'])
            if sum(r['accepted_events'] for r in records)!=m['accepted_events']:raise ValueError('Every selected event must contribute')
            if sum(r['event_response_bytes'] for r in records)!=m['accepted_events']*78920*4:raise ValueError('All complete float32 event response bytes required')
            for v,count in enumerate(m['events_per_view'],1):
                indices=np.load(selection/'selections'/f'{v}.npy');cursor=0
                for r in sorted(records,key=lambda x:x['rank']):
                    p=r['partitions'][v-1]
                    if p['view']!=v or p['lo']!=cursor or p['hi']<cursor or p['indices_sha256']!=hashlib.sha256(indices[p['lo']:p['hi']].astype('<i8').tobytes()).hexdigest():raise ValueError('All response event partitions must close')
                    cursor=p['hi']
                if cursor!=count:raise ValueError('Missing original selected response events')
            for r in records:
                n=r['event_numerical_checks']
                if (n['sample_events'],n['active_points'],n['full_points'])!=(32,78920,132040) or n['source_truth_used']:raise ValueError('Original event oracle dimensions differ')
                for k in ['chunk_relative_L2','rank_weight_relative_L2','forward_adjoint_relative_error','event_constant_relative_error']:
                    if not math.isfinite(n[k]) or not 0<=n[k]<=1e-5:raise ValueError('Original event numerical oracle failed')
                if not math.isfinite(r['minimum_active_event_mass']) or r['minimum_active_event_mass']<=0:raise ValueError('Invalid retained event support')
            expected=np.fromfile(a.release/'continuous_energy_Sensi_full','<f4')[active]*geo['ellipse_fraction'][active]
            if not np.array_equal(expected.astype('<f4'),np.fromfile(a.result/'S_Compton.float32','<f4')):raise ValueError('Own matched Compton sensitivity differs')
            extra['accepted_events']=m['accepted_events']
        if a.mode=='formal':
            authority=read(run['authority_file'])
            if digest(run['authority_file'])!=run['authority_sha256'] or not authority['passed'] or authority['dataset']!=a.dataset:raise ValueError('Formal full-input authority differs')
    if hashes(a.result)!=before:raise ValueError('Scientific results changed during read-only verification')
    a.output.mkdir(parents=True,exist_ok=False)
    proof=dict(passed=True,study=STUDY,dataset=a.dataset,mode=a.mode,resources=records,plan=plan,
               plan_sha256=digest(a.plan),collection_sha256=digest(a.input/'collection.json'),
               release_sha256=digest(a.release/'release_manifest.json'),result_files_sha256=before,
               allocation_sha256=digest(a.allocation),elapsed_seconds=time.monotonic()-started,
               independent_cpu_read_only_verification=True,**extra)
    write(a.output/'verification.json',proof);shutil.copy2(a.allocation,a.output/'allocation.txt')
    archive=a.output/'accepted_result.tar.gz'
    with tarfile.open(str(archive)+'.writing','w:gz',compresslevel=1) as t:
        for n in sorted(before):t.add(a.result/n,arcname=n,recursive=False)
        for n in ['verification.json','allocation.txt']:t.add(a.output/n,arcname=n,recursive=False)
    os.replace(str(archive)+'.writing',archive)
    files=dict(before,**{n:digest(a.output/n) for n in ['verification.json','allocation.txt']})
    write(a.output/'package.json',dict(passed=True,study=STUDY,dataset=a.dataset,mode=a.mode,files=files,
          archive_sha256=digest(archive),archive_bytes=archive.stat().st_size))
    verify_files(a.result,before)
    print('DERENZO_READ_ONLY_ACCEPTED',a.dataset,a.mode,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['release','input','result','allocation','output','plan']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--mode',choices=['selection','validation','formal'],required=True)
    verify(p.parse_args())
