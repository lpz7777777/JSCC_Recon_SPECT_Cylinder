"""Strict r5 delivery after all three branches and all Slurm steps succeed.

This is an explicit command, not a watcher. It cannot submit or cancel jobs.
"""
import argparse,hashlib,json,os,tarfile
from pathlib import Path
import numpy as np
from workflow import REPORT,DATA,REMOTE,FREEZE,read,write,sha,now,connect,command,script,q

def verify(root,reg):
    frozen=read(REPORT/'freeze_resume_r5.json');execution=read(DATA/'execution_manifest_resume_r5.json')
    assert reg['execution_key']==execution['key'] and read(root/'release/release_manifest.json')==execution
    for name,digest in execution['files'].items():assert sha(root/'release'/name)==digest
    g=np.load(DATA/frozen['payload_dir']/'whole_geometry.npz');active=g['active_indices']
    rows=[];single=None
    for percent in (100,80,60):
        folder=root/'result'/f'width{percent}';m=read(folder/'completed.json')
        assert (m['iterations'],m['save_step'],m['frames'])==(10000,50,200)
        assert m['release']==reg['execution_key']
        for n,s in m['files'].items():assert sha(folder/n)==s,n
        final=np.fromfile(folder/'Image_JSCC_active.float32','<f4');assert final.shape==(78920,)
        history=np.fromfile(folder/'Image_JSCC_history.float32','<f4').reshape(200,78920)
        full=np.fromfile(folder/'Image_JSCC_full.float32','<f4');assert full.shape==(132040,)
        assert np.isfinite(history).all() and (history>=0).all()
        assert np.array_equal(history[-1],final) and np.array_equal(full[active],final)
        outside=np.ones(132040,bool);outside[active]=False;assert np.count_nonzero(full[outside])==0
        cp=read(folder/'checkpoint_manifest.json');assert cp['iteration']==10000 and len(cp['sha256'])==200
        assert len(list((folder/'checkpoints').glob('*.float32')))==200
        for i in range(200):
            name=f'checkpoints/iteration_{(i+1)*50:05d}.float32'
            assert sha(folder/name)==cp['sha256'][name]
            assert np.array_equal(history[i],np.fromfile(folder/name,'<f4'))
        ss=np.fromfile(folder/'S_SC_active.float32','<f4');sc=np.fromfile(folder/'S_Compton_active.float32','<f4')
        sf=np.fromfile(folder/'S_Compton_full.float32','<f4')
        assert ss.shape==sc.shape==(78920,) and sf.shape==(132040,)
        assert np.isfinite(sc).all() and (sc>=0).all()
        assert np.array_equal(sc,sf[active]*g['ellipse_fraction'][active].astype(np.float32))
        if single is None:single=ss
        else:assert np.array_equal(single,ss)
        if percent==100:
            assert sha(folder/'S_Compton_full.float32')==read(FREEZE)['files']['continuous_energy_Sensi_full']
            assert read(folder/'original_regression_00050.json')['relative_L2']<=1e-5
            diagnostic=read(folder/'original_regression_10000.json')
            assert np.isfinite(diagnostic['relative_L2']) and diagnostic['threshold']==1e-5
            assert diagnostic['continuation_authorized'] and reg['historical_discrepancy_explicitly_accepted']
            assert diagnostic['within_original_threshold']==bool(diagnostic['relative_L2']<=1e-5)
            plan=read(REPORT/'resume_plan_r5.json');lineage=read(root/'result/resume_lineage.json')
            assert lineage==plan and plan['completed_iterations']==9950
            for iteration in range(50,9951,50):
                name=f'checkpoints/iteration_{iteration:05d}.float32'
                assert sha(folder/name)==plan['parent_files']['width100/'+name]
            for name,digest in plan['parent_files'].items():
                if Path(name).name.startswith('benchmark_'):assert sha(folder/Path(name).name)==digest
            assert sha(folder/'S_SC_active.float32')==plan['parent_files']['width100/S_SC_active.float32']
        else:
            s=read(folder/'S_provenance.json');assert s['events']==170247 and s['primary_gamma_440']==1000000000
            assert abs(s['integral_relative_error'])<=1e-5 and s['scale']==percent/100
        parts=[];rank_records=[]
        for rank in range(8):
            r=read(folder/f'rank{rank}.json');assert r['rank']==rank and r['scale']==percent/100
            assert (r['inherited_iterations'],r['new_iterations'])==((9950,50) if percent==100 else (0,10000))
            guard=r['resources'];assert guard['rss_peak_bytes']<=.8*guard['host_allocated_bytes']
            assert guard['gpu_reserved_peak_bytes']<=.8*guard['gpu_total_bytes']
            n=read(folder/f'numerical_rank{rank}.json');assert n['chunk_relative_L2']<=1e-5 and max(n['adjoint_errors'])<=1e-5
            parts.append(r['partitions']);rank_records.append(r)
        assert len({r['node'] for r in rank_records})==8
        expected=read(FREEZE)['study']['events_per_view']
        for view in range(20):
            chunks=[r[view] for r in parts];assert chunks[0]['first']==0 and chunks[-1]['last']==expected[view]
            assert all(a['last']==b['first'] for a,b in zip(chunks,chunks[1:]))
            assert sum(r['events'] for r in chunks)==expected[view]
        rows.append(dict(width_percent=percent,frames=200,seconds_per_iteration=rank_records[0]['seconds_per_iteration'],chosen=rank_records[0]['chosen'],final_sha256=sha(folder/'Image_JSCC_active.float32')))
    choice=read(root/'result/width100/benchmark_choice.json');assert choice['bitwise_gates'][choice['chosen']]
    result=dict(time=now(),passed=True,job=reg['job'],execution_key=reg['execution_key'],frames=600,branches=rows,
        benchmark=choice,physical_resolution_certified=False,image_comparison_completed=False,
        inherited_checkpoint_count=199,historical_comparison=read(root/'result/width100/original_regression_10000.json'),
        accepted_continuation_does_not_change_historical_pass_status=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    reg=read(REPORT/'formal_registration_resume_r5.json');job=reg['job'];target=DATA/('delivered_'+job)
    receipt=REPORT/('strict_delivery_'+job+'.json')
    if receipt.exists():
        print('ALREADY STRICTLY DELIVERED',job);return
    if a.verify_only:
        print(json.dumps(verify(target,reg),indent=2));return
    with connect() as c:
        state=command(c,'sacct -n -P -j '+job+' --format=JobID,State,ExitCode',45)
        rows=[line.split('|') for line in state.splitlines() if line.strip()]
        assert {job,job+'.batch',job+'.extern',job+'.0'}<={r[0] for r in rows},state
        if any(r[1]!='COMPLETED' or r[2]!='0:0' for r in rows):raise RuntimeError('Job not fully successful; no delivery: '+state)
        code='''import hashlib,json,tarfile,pathlib,os
base=pathlib.Path(BASE);result=pathlib.Path(RESULT);release=pathlib.Path(RELEASE);job=JOB
receipt=base/('delivery_'+job+'.json');archive=base/('delivery_'+job+'.tar')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
if receipt.exists():
 m=json.loads(receipt.read_text());assert sha(archive)==m['archive_sha256'];print(json.dumps(m));raise SystemExit
complete=json.loads((result/'completed.json').read_text());assert complete['scales']==[1.,.8,.6]
files={}
for prefix,root in [('result',result),('release',release)]:
 for f in sorted(root.rglob('*')):
  if f.is_file():
   assert not f.is_symlink();files[prefix+'/'+f.relative_to(root).as_posix()]=f
for suffix in ('out','err'):files['evidence/'+job+'.'+suffix]=base/'logs'/(job+'.'+suffix)
files['evidence/allocation.txt']=base/'allocations'/(job+'.txt')
hashes={n:sha(f) for n,f in files.items()}
partial=archive.with_suffix('.tar.partial')
with tarfile.open(partial,'w') as t:
 for n,f in files.items():t.add(f,arcname=n,recursive=False)
os.replace(partial,archive)
m=dict(job=job,files=hashes,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
with receipt.open('x') as f:json.dump(m,f,indent=2)
print(json.dumps(m))
'''.replace('BASE',repr(REMOTE)).replace('RESULT',repr(reg['result'])).replace('RELEASE',repr(reg['release'])).replace('JOB',repr(job))
        manifest=json.loads(script(c,code,300));write(REPORT/('archive_manifest_'+job+'.json'),manifest)
        archive=DATA/('delivery_'+job+'.tar');partial=archive.with_suffix('.tar.partial')
        if not archive.exists():
            with c.open_sftp() as s:
                s.get(REMOTE+'/'+archive.name,str(partial),prefetch=True,max_concurrent_prefetch_requests=16)
            assert partial.stat().st_size==manifest['archive_bytes'] and sha(partial)==manifest['archive_sha256']
            partial.replace(archive)
        assert sha(archive)==manifest['archive_sha256']
    target.mkdir(exist_ok=True);seen=set()
    with tarfile.open(archive) as t:
        for member in t:
            name=member.name
            assert member.isfile() and name in manifest['files'] and name not in seen
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            data=t.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==manifest['files'][name]
            dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha(dest)==manifest['files'][name]
            else:
                with dest.open('xb') as f:f.write(data)
            seen.add(name)
    assert seen==set(manifest['files'])
    quality=verify(target,reg);quality.update(member_count=len(seen),accounting=state,archive_sha256=manifest['archive_sha256'])
    write(receipt,quality);print(json.dumps(quality,indent=2))

if __name__=='__main__':main()
