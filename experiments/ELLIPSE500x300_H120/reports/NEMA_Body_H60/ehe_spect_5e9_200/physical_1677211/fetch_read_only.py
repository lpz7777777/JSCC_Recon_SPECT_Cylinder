"""Read-only strict retrieval of the existing scientific HOLD; never submit."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path('experiments/ELLIPSE500x300_H120').resolve()))
from ehe_5e9_workflow import *
from ehe_conversion_workflow import response_root

a=read(REPORT/'physical_job.json');job=a['job'];remote=base('gpu')
paths={'physical_gate.json':remote+'/physical/physical_gate.json',
       'physical_audit.csv':remote+'/physical/physical_audit.csv',
       'physical.log':remote+f'/logs/physical_{job}_4294967294.log'}
paths.update({n:release('gpu')+'/'+n for n in ['ehe_gpu_pipeline.py','ehe_common.py','truth_3mm.npz','config.json']})
paths['collection.json']=remote+'/counts/collection.json'
paths['worker_counts.npz']=remote+'/counts/worker_counts.npz'
for n in RESPONSES:paths[n+'/factor_manifest.json']=response_root()+'/'+n+'/factor_manifest.json'
code='import hashlib,json;from pathlib import Path;p='+repr(paths)+';print(json.dumps({n:{"sha256":hashlib.sha256(Path(v).read_bytes()).hexdigest(),"bytes":Path(v).stat().st_size} for n,v in p.items()}))'
with connection('gpu') as c:
    queue=command(c,'squeue -h -u scxi717 -o "%i|%T|%M|%R"')
    own=[line for line in queue.splitlines() if line.split('|')[0]==str(job)]
    assert not own,own
    accounting=command(c,'sacct -n -P -j '+q(job)+' --format=JobID,State,ExitCode,MaxRSS,Elapsed,AllocTRES,Start,End,NodeList')
    print(accounting)
    assert f'{job}|FAILED|1:0|' in accounting and f'{job}.batch|FAILED|1:0|' in accounting and f'{job}.extern|COMPLETED|0:0|' in accounting
    before=json.loads(command(c,q(GPU_PYTHON)+' -c '+q(code)))
    out=DATA/f'physical_{job}';out.mkdir(exist_ok=True)
    rpt=REPORT/f'physical_{job}';rpt.mkdir(exist_ok=True)
    targets={'physical_gate.json':REPORT/'physical_gate.json',
             'physical_audit.csv':out/'physical_audit.csv','physical.log':rpt/'physical.log'}
    with c.open_sftp() as s:
        for n,p in targets.items():
            if p.exists():assert digest(p)==before[n]['sha256'],'Existing physical evidence differs'
            else:s.get(paths[n],str(p))
            assert digest(p)==before[n]['sha256'] and p.stat().st_size==before[n]['bytes']
    after=json.loads(command(c,q(GPU_PYTHON)+' -c '+q(code)))
    assert before==after,'Remote evidence changed during fetch'
    (rpt/'accounting.txt').write_bytes(accounting.encode())
    (rpt/'queue.txt').write_bytes(('\n'.join(own)+'\n').encode() if own else b'')
    gate=read(REPORT/'physical_gate.json')
    assert not gate['passed'] and gate['hold_count']>0
    assert gate['files']['physical_audit.csv']==before['physical_audit.csv']['sha256']
    fg=frozen('gpu')
    for n in ['ehe_gpu_pipeline.py','ehe_common.py','truth_3mm.npz','config.json']:assert before[n]['sha256']==fg['sha256'][n]
    ident=read(REPORT/'response_conversion_identity_acceptance.json')
    for n in RESPONSES:assert before[n+'/factor_manifest.json']['sha256']==ident['factor_manifest_sha256'][n]
    collection=read(DATA/'transport/collection.json')
    assert before['worker_counts.npz']['sha256']==collection['files']['worker_counts.npz']==digest(DATA/'transport/worker_counts.npz')
    assert before['collection.json']['sha256']==gate['collection_sha256']==digest(DATA/'transport/collection.json')
    assert before['truth_3mm.npz']['sha256']==gate['source_sha256']
    assert 'Physical response HOLD: '+str(gate['hold_count'])+' diagnostics' in targets['physical.log'].read_text()
    assert not (REPORT/'validation_job.json').exists() and not (REPORT/'formal_job.json').exists()
    write(REPORT/'physical_hold_fetch_acceptance.json',dict(passed=True,scope='Strict retrieval and immutable input/exit identity of scientific HOLD; physical gate failed',
        job=job,science_gate_passed=False,hold_count=gate['hold_count'],no_queue=True,accounting_file=f'physical_{job}/accounting.txt',
        physical_audit_local=str(out/'physical_audit.csv'),remote_before=before,remote_after_equal=True,
        local_evidence_sha256={str(p.relative_to(REPORT)):digest(p) for p in targets.values() if p.is_relative_to(REPORT)},
        accounting_sha256=digest(rpt/'accounting.txt'),queue_sha256=digest(rpt/'queue.txt'),data_csv_sha256=digest(out/'physical_audit.csv'),
        fetch_code_sha256=digest(__file__),science_release_key=a['release_key'],response_root=response_root(),validation_submitted=False,formal_submitted=False))
    print('STRICT_HOLD_FETCH_PASS',json.dumps(before,indent=2))
