"""Independent 8x1 formal, explicitly retaining the existing 16-GPU job."""
import argparse
import difflib
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import prejune_compton_workflow as parent
from prejune_8x1_adapter import adapt
from ehe_5e9_workflow import connection, command, q, put_tree
from jscc_5e10_common import GPU_PROJECT,GPU_PYTHON

HERE=Path(__file__).resolve().parent
TAG='nema_5e9_prejune_compton_8x1_20261026'
DATA=HERE/'generated'/TAG
REPORT=HERE/'reports/NEMA_Body_H60'/TAG
REMOTE=GPU_PROJECT+'/generated/'+TAG
PAYLOAD=DATA/'payload_v2'
digest,read,write=parent.digest,parent.read,parent.write


def prepare():
    DATA.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    marker=REPORT/'freeze_v2.json'
    if marker.exists():
        f=read(marker)
        for name,sha in f['files'].items():assert digest(PAYLOAD/name)==sha,name
        return f
    old=parent.freeze()
    for name in ('test_prejune_compton.py','test_prejune_8x1.py'):
        result=subprocess.run([sys.executable,'-X','utf8',str(HERE/name)],capture_output=True,text=True)
        (REPORT/(name+'.log')).write_text(result.stdout+result.stderr,encoding='utf8')
        if result.returncode:raise RuntimeError(name+' failed')
    capacity=read(HERE/'reports/NEMA_Body_H60/nema_5e9_prejune_compton_4x1_20261026/capacity_analysis.json')
    per=[capacity['logical16_event_upper_bounds'][i]+capacity['logical16_event_upper_bounds'][i+8] for i in range(8)]
    bound=[n*78920*4 for n in per]
    analysis=dict(events_upper=sum(per),per_gpu_event_upper=per,per_gpu_response_bytes_upper=bound,
        max_response_GiB=max(bound)/2**30,node_expected_memory_MiB=126000,node_80pct_GiB=98.4375,
        preparation_matrix_GiB=5543567360/2**30,matrix_freed_before_gpu_response_cache=True,
        gpu_runtime_guard_fraction=.8,host_runtime_guard_fraction=.8,
        cache_plus_2GiB_under_80pct_32GiB=max(bound)+2*2**30 < .8*32*2**30,
        input_counts_source_sha256=digest(HERE/'reports/NEMA_Body_H60/nema_5e9_prejune_compton_4x1_20261026/input_capacity_counts.json'),
        complete_gpu_run_passed=False)
    assert analysis['cache_plus_2GiB_under_80pct_32GiB']
    if not (REPORT/'capacity_analysis.json').exists():write(REPORT/'capacity_analysis.json',analysis)
    else:assert read(REPORT/'capacity_analysis.json')==analysis
    source=parent.DATA/'payload';target=PAYLOAD
    shutil.copytree(source,target)
    original=(source/'run_prejune_compton.py').read_text('utf8');changed=adapt(original)
    (target/'run_prejune_compton.py').write_text(changed,encoding='utf8',newline='\n')
    (REPORT/'runner_execution_only.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile='16gpu_frozen',tofile='8x1_logical16')),encoding='utf8')
    contract=read(source/'contract.json')
    contract.update(study=TAG,gpus_per_node=1,ranks=8,logical_ranks=16,
        logical_ownership='physical rank r owns original shards [r,r+8]; original per-shard raw ranges, batch32, Neff and OSEM subset membership preserved',
        parent_contract_sha256=old['contract_sha256'],retained_parent_job=1739454,
        new_authorization='User requested 8 nodes x 1 RTX5090 instead of 4x1, retain original 16-GPU job; no 4-GPU submit',
        execution_changes='Two original logical ranks per GPU; float32 collective association differs; resident full responses; no host streaming; Python-compatible chunked SHA256',
        release_files={str(p.relative_to(target)).replace('\\','/'):digest(p) for p in sorted(target.rglob('*')) if p.is_file() and p.name!='contract.json'})
    (target/'contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    files={str(p.relative_to(target)).replace('\\','/'):digest(p) for p in sorted(target.rglob('*')) if p.is_file()}
    unchanged={k:v for k,v in old['files'].items() if k not in ('contract.json','run_prejune_compton.py')}
    assert all(files[k]==v for k,v in unchanged.items())
    key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    f=dict(release=REMOTE+'/releases/'+key,key=key,files=files,contract_sha256=digest(target/'contract.json'),
        parent_release=old['release'],parent_contract_sha256=old['contract_sha256'],unchanged_files=unchanged,
        changes=['contract.json','run_prejune_compton.py'],capacity=analysis)
    write(marker,f)
    return f


def script(f):
    original=parent.launcher(f)
    body=original.replace(parent.REMOTE,REMOTE).replace('--nproc_per_node=2','--nproc_per_node=1').replace('/tmp/nema_prejune_', '/tmp/nema_prejune8x1_')
    assert '--nproc_per_node=2' not in body and parent.REMOTE not in body
    return body


def upload_bundle(c, f, launch):
    """Transfer small frozen files together, avoiding per-file SFTP stalls."""
    bundle=io.BytesIO()
    expected={name:sha for name,sha in f['files'].items() if name!='calibration/legacy_v01.csv'}
    expected['__launcher__']=digest(launch)
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in expected:
            archive.writestr(name,(launch if name=='__launcher__' else PAYLOAD/name).read_bytes())
    payload=bundle.getvalue()
    code='''import hashlib,io,pathlib,sys,zipfile
root=pathlib.Path(%r);base=pathlib.Path(%r);expected=%r
data=sys.stdin.buffer.read()
assert hashlib.sha256(data).hexdigest()==%r
print('BUNDLE_RECEIVED',len(data),flush=True)
with zipfile.ZipFile(io.BytesIO(data)) as archive:
 assert set(archive.namelist())==set(expected)
 for name,sha in expected.items():
  data=archive.read(name)
  assert hashlib.sha256(data).hexdigest()==sha,name
  p=(base/'formal.sh.staging') if name=='__launcher__' else root/(name+'.staging')
  p.write_bytes(data)
  print('STAGED',name,flush=True)
''' % (f['release'],REMOTE,expected,hashlib.sha256(payload).hexdigest())
    stdin,out,err=c.exec_command(GPU_PYTHON+' -B -c '+q(code),timeout=60)
    stdin.write(payload);stdin.flush();stdin.channel.shutdown_write()
    channel=out.channel;errors=[];last=time.monotonic();began=last
    while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
        if channel.recv_ready():
            print(channel.recv(65536).decode(errors='replace'),end='',flush=True);last=time.monotonic()
        if channel.recv_stderr_ready():errors.append(channel.recv_stderr(65536))
        if time.monotonic()-last>120 or time.monotonic()-began>600:
            channel.close();raise TimeoutError('Frozen bundle upload stalled before submission')
        time.sleep(.05)
    if channel.recv_exit_status():raise RuntimeError(b''.join(errors).decode(errors='replace'))


def submit():
    f=prepare()
    if (REPORT/'formal_job.json').exists():return read(REPORT/'formal_job.json')
    if (REPORT/'submission_intent.json').exists():raise RuntimeError('Unresolved intent; inspect Slurm before retry')
    f=dict(f,release=f['release']+'_localcopy')
    if not (REPORT/'delivery_freeze.json').exists():write(REPORT/'delivery_freeze.json',f)
    else:assert read(REPORT/'delivery_freeze.json')==f
    body=script(f);launch=DATA/'formal_delivery.sh'
    if launch.exists():assert launch.read_text('utf8')==body
    else:launch.write_text(body,encoding='utf8',newline='\n')
    with connection('gpu') as c:
        queue=command(c,"squeue -h -u scxi717 -o '%i|%j|%T|%R'",timeout=30)
        if 'NEMA5e9_preJune8x1' in queue:raise RuntimeError('New 8x1 already in queue; inspect registration')
        original=command(c,'scontrol show job 1739454',timeout=30)
        (REPORT/'original_job_before.txt').write_text(original,encoding='utf8')
        disk=int(command(c,GPU_PYTHON+' -c '+q('import os;s=os.statvfs('+repr(GPU_PROJECT)+');print(s.f_bavail*s.f_frsize)')).strip())
        if disk<16*2**30:raise RuntimeError('Insufficient shared disk reserve')
        command(c,'mkdir -p -- '+q(f['release'])+' '+q(REMOTE+'/logs'))
        # Avoid the blocked source helper inode. Use local immutable bytes for
        # every code/geometry file; only reuse the separately verified raw CSV.
        stage_code='''import hashlib,pathlib
dest=pathlib.Path(%r);source=pathlib.Path(%r);files=%r
for name in files:(dest/name).parent.mkdir(parents=True,exist_ok=True)
name='calibration/legacy_v01.csv'
data=(source/name).read_bytes()
assert hashlib.sha256(data).hexdigest()==files[name]
p=dest/name
if not p.exists():p.write_bytes(data)
assert hashlib.sha256(p.read_bytes()).hexdigest()==files[name]
print('CALIBRATION_REUSED_SHA_OK')
''' % (f['release'],f['parent_release'],f['files'])
        print(command(c,GPU_PYTHON+' -B -c '+q(stage_code),timeout=60),flush=True)
        upload_bundle(c,f,launch)
        finalize='''import hashlib,pathlib
base=pathlib.Path(%r);root=pathlib.Path(%r);expected=%r
for name,sha in expected.items():
 p=(root/name) if name=='formal.sh' else (base/name)
 temp=p.with_name(p.name+'.staging')
 assert hashlib.sha256(temp.read_bytes()).hexdigest()==sha,name
 temp.replace(p)
print('CHANGED_SOURCE_AND_LAUNCHER_SHA_VERIFIED')
''' % (f['release'],REMOTE,{name:sha for name,sha in f['files'].items() if name!='calibration/legacy_v01.csv'}|{'formal.sh':digest(launch)})
        print(command(c,GPU_PYTHON+' -B -c '+q(finalize),timeout=60),flush=True)
        code='import hashlib,pathlib; r=pathlib.Path('+repr(f['release'])+'); h='+repr(f['files'])+'; assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==v for n,v in h.items());print("FROZEN_RELEASE_SHA_OK",len(h))'
        proof=command(c,GPU_PYTHON+' -B -c '+q(code),timeout=120)
        compatibility_code='''import pathlib,sys,json
root=pathlib.Path(%r);sys.path.insert(0,str(root))
import run_prejune_compton as runner
from prejune_compton_model import historical_namespace
assert runner.sha(root/'contract.json')==%r
old=historical_namespace(root/'legacy_original')
assert old.MIN_EVENT_EFFECTIVE_SUPPORT==50
assert callable(old.safe_em_update)
print(json.dumps(dict(python=sys.version,torch=runner.torch.__version__,import_passed=True,sha_passed=True,historical_functions_loaded=True,gpu_test_performed=False)))
''' % (f['release'],f['contract_sha256'])
        compatibility=command(c,GPU_PYTHON+' -B -c '+q(compatibility_code),timeout=60)
        (REPORT/'remote_import_check.json').write_text(compatibility,encoding='utf8')
        command(c,'bash -n '+q(REMOTE+'/formal.sh'))
        opts='-p gpu_5090 --qos=gpugpu -N 8 -n 8 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1 --time=3-00:00:00 --job-name=NEMA5e9_preJune8x1 --chdir='+q(f['release'])+' --output='+q(REMOTE+'/logs/formal.%j.out')+' --error='+q(REMOTE+'/logs/formal.%j.err')+' '+q(REMOTE+'/formal.sh')
        feasibility=command(c,'sbatch --test-only '+opts+' 2>&1',timeout=60)
        if 'error' in feasibility.lower():raise RuntimeError(feasibility)
        write(REPORT/'deployment.json',dict(release=f['release'],proof=proof,launcher_sha256=digest(launch),queue_before=queue,scheduler_test_only=feasibility,disk_free=disk,remote_import_check_sha256=digest(REPORT/'remote_import_check.json')))
        write(REPORT/'submission_intent.json',dict(command='sbatch --parsable '+opts,time=time.time(),resolved=False,original_job_unchanged=1739454))
        job=command(c,'sbatch --parsable '+opts,timeout=60).strip().split(';')[0]
        if not job.isdigit():raise RuntimeError('Uncertain submit outcome; do not retry: '+job)
        rec=dict(job=int(job),study=TAG,stage='formal',release=f['release'],contract_sha256=f['contract_sha256'],output=REMOTE+'/formal_'+job,
            nodes=8,gpus_per_node=1,physical_gpus=8,logical_ranks=16,channel='440 keV pure Compton',iterations=10000,save_every=50,subsets=8,
            retained_parent_job=1739454,new_transport=False,separate_validation_jobs=0,four_gpu_submitted=False)
        write(REPORT/'formal_job.json',rec)
        status=command(c,'scontrol show job '+job+'; squeue -h -u scxi717 -o "%i|%j|%T|%R"',timeout=30)
        (REPORT/'submitted_status.txt').write_text(status,encoding='utf8')
        print(status,flush=True)
        return rec


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit']);a=p.parse_args()
    print(json.dumps(prepare() if a.action=='prepare' else submit(),ensure_ascii=False,indent=2))
