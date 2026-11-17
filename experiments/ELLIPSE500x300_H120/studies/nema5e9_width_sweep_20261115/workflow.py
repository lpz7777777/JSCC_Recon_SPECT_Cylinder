"""Explicit, idempotent submission/status/fetch for the independent width study.

No scheduler loop, no cancellation and no restoration of old controllers.
An ambiguous sbatch is never retried automatically.
"""
import argparse,datetime,hashlib,json,re,shlex,sys,tarfile,time
from pathlib import Path
from prepare import CODE,EXP,ROOT,DATA,REPORT,sha,read,write
sys.path[:0]=[str(EXP),str(ROOT/'experiments/FOV120')]
from reconstruction_ssh import connect
from ehe_5e9_workflow import command,q

GPU='/data/run01/scxi717/lpz/'+ROOT.name+'/experiments/ELLIPSE500x300_H120'
REMOTE=GPU+'/generated/nema5e9_width_sweep_20261115'
PY='/data/home/scxi717/.conda/envs/torch/bin/python'
FREEZE=REPORT/'freeze_r3.json'
ATTEMPT='r4'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def script(c,code,timeout=120):return command(c,q(PY)+' -B -c '+q(code),timeout)
def immutable(path,obj):
    if path.exists():
        if read(path)!=obj:raise FileExistsError(path)
    else:write(path,obj)

def capacity(c):
    raw=command(c,'timeout 30s /usr/lpp/mmfs/bin/mmlsquota -u scxi717 hpc1',40)
    lines=[l.split() for l in raw.splitlines() if re.match(r'^hpc1\s+run01\s+USR\s',l)]
    if len(lines)!=1:raise ValueError('Actual run01 user quota unavailable')
    row=lines[0];used,soft,hard,doubt=[int(v)*1024 for v in row[3:7]]
    limits=[x for x in (soft,hard) if x>0]
    if not limits:raise ValueError('No finite actual quota bound')
    limit=min(limits);free=limit-used-doubt;budget=2<<30
    if free<budget/.8:raise ValueError('Insufficient actual quota for bounded new output')
    return dict(time=now(),filesystem='hpc1',fileset='run01',used_bytes=used,quota_bytes=limit,
        in_doubt_bytes=doubt,conservative_free_bytes=free,new_output_cap_bytes=budget,
        required_free_bytes=int(budget/.8),raw=raw,
        policy='At least 20% headroom on new write budget; pre-existing user-wide utilization reported honestly. GPU/RAM maximum 80%. No response C stored on disk.')

def package():
    f=read(FREEZE);payload=DATA/f['payload_dir']
    for n,s in f['files'].items():assert sha(payload/n)==s,n
    launch='''#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NCCL_SOCKET_IFNAME=bond0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RELEASE="$1"
test -f "$RELEASE/release_manifest.json"
export WIDTH_RELEASE="$RELEASE"
export WIDTH_REMOTE=__REMOTE__
export WIDTH_GPU_BASE=__GPU__
export WIDTH_PYTHON=__PY__
scontrol show job -dd "$SLURM_JOB_ID" > "$WIDTH_REMOTE/allocations/${SLURM_JOB_ID}.txt"
WIDTH_HOST_BYTES=$("$WIDTH_PYTHON" -B "$RELEASE/verify_first_scatter.py" --allocation-only "$WIDTH_REMOTE/allocations/${SLURM_JOB_ID}.txt" --nodes 8)
export WIDTH_HOST_BYTES
export MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((35000+SLURM_JOB_ID%20000))
timeout --signal=TERM --kill-after=60s 258600s srun --label --kill-on-bad-exit=1 --chdir=/tmp "$WIDTH_PYTHON" -B "$RELEASE/node_launcher.py"
'''.replace('__REMOTE__',q(REMOTE)).replace('__GPU__',q(GPU)).replace('__PY__',q(PY))
    node='''import os,subprocess
from pathlib import Path
root=Path(os.environ['WIDTH_RELEASE']);job=os.environ['SLURM_JOB_ID']
# All caches are per-job local temporaries; never scientific cross-job data.
cache=Path('/tmp')/('width_'+job+'_'+os.environ['SLURM_PROCID']);cache.mkdir(exist_ok=True)
os.environ['TMPDIR']=str(cache);os.environ['TORCHINDUCTOR_CACHE_DIR']=str(cache/'torch')
os.environ['TRITON_CACHE_DIR']=str(cache/'triton');os.environ['CUDA_CACHE_PATH']=str(cache/'cuda')
cmd=[os.environ['WIDTH_PYTHON'],'-B','-m','torch.distributed.run','--nnodes=8','--nproc_per_node=1',
'--node_rank='+os.environ['SLURM_PROCID'],'--master_addr='+os.environ['MASTER_ADDR'],
'--master_port='+os.environ['MASTER_PORT'],'--rdzv_backend=static','--rdzv_id='+job+'_width','--max_restarts=0',
str(root/'run_study.py'),'--input-root',os.environ['WIDTH_GPU_BASE']+'/generated',
'--factors',os.environ['WIDTH_GPU_BASE']+'/generated/FactorsCalibrated',
'--output',os.environ['WIDTH_REMOTE']+'/formal_'+job,'--host-bytes',os.environ['WIDTH_HOST_BYTES']]
os.execv(cmd[0],cmd)
'''
    extra=DATA/('launcher_'+ATTEMPT);extra.mkdir(exist_ok=True)
    for n,t in [('launch.sh',launch),('node_launcher.py',node)]:
        p=extra/n;raw=t.encode()
        if p.exists() and p.read_bytes()!=raw:raise FileExistsError('Frozen launcher differs')
        if not p.exists():p.write_bytes(raw)
    files=dict(f['files']);files.update({p.name:sha(p) for p in extra.iterdir() if p.is_file()})
    key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    manifest=dict(key=key,files=files,source_freeze=f['key'],source_freeze_sha256=sha(FREEZE))
    immutable(DATA/('execution_manifest_'+ATTEMPT+'.json'),manifest)
    archive=DATA/('release_'+key+'.tar.gz')
    if not archive.exists():
        with tarfile.open(archive,'w:gz') as t:
            for n in sorted(files):t.add(payload/n if n in f['files'] else extra/n,arcname=n,recursive=False)
            t.add(DATA/('execution_manifest_'+ATTEMPT+'.json'),arcname='release_manifest.json',recursive=False)
    return manifest,archive

def deploy():
    if (REPORT/('deployment_'+ATTEMPT+'.json')).exists():return read(REPORT/('deployment_'+ATTEMPT+'.json'))
    m,archive=package();release=REMOTE+'/releases/'+m['key'];remote_tar=REMOTE+'/'+archive.name
    with connect() as c:
        cap=capacity(c);write(REPORT/('capacity_before_deploy_'+ATTEMPT+'.json'),cap)
        command(c,'mkdir -p '+q(REMOTE+'/releases')+' '+q(REMOTE+'/logs')+' '+q(REMOTE+'/allocations'))
        # New unique namespace only; exclusive creation or identical bytes.
        with c.open_sftp() as s:
            try:
                st=s.stat(remote_tar)
                if st.st_size!=archive.stat().st_size:raise ValueError('Existing archive size differs')
            except FileNotFoundError:s.put(str(archive),remote_tar)
        check=script(c,"import hashlib;from pathlib import Path;p=Path("+repr(remote_tar)+");print(hashlib.sha256(p.read_bytes()).hexdigest())").strip()
        assert check==sha(archive)
        code="""import pathlib,tarfile,json,hashlib
r=pathlib.Path(RELEASE);archive=pathlib.Path(ARCHIVE)
with tarfile.open(archive) as t:
 for member in t:
  if not member.isfile() or pathlib.PurePosixPath(member.name).is_absolute() or '..' in pathlib.PurePosixPath(member.name).parts:raise ValueError('Unsafe archive member')
  p=r/member.name;data=t.extractfile(member).read();p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==data,str(p)
  else:
   with p.open('xb') as f:f.write(data)
m=json.loads((r/'release_manifest.json').read_text())
for n,s in m['files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==s,n
print('DEPLOY_SHA_PASS',m['key'],len(m['files']))
""".replace('RELEASE',repr(release)).replace('ARCHIVE',repr(remote_tar))
        proof=script(c,code)
        test=command(c,'cd '+q(release)+' && '+q(PY)+' -B test_study.py',120)
        if 'WIDTH_SWEEP_LOCAL_PASS' not in test:raise ValueError('Linux CPU tests failed')
        (REPORT/('linux_tests_'+ATTEMPT+'.txt')).write_text(test,encoding='utf8')
        result=dict(time=now(),release=release,execution_key=m['key'],archive_sha256=sha(archive),proof=proof,linux_tests_passed=True,linux_tests_sha256=sha(REPORT/('linux_tests_'+ATTEMPT+'.txt')))
        write(REPORT/('deployment_'+ATTEMPT+'.json'),result);print(json.dumps(result),flush=True)
        return result

def submit():
    registration=REPORT/('formal_registration_'+ATTEMPT+'.json');intent=REPORT/('submission_intent_'+ATTEMPT+'.json')
    if registration.exists():print(json.dumps(read(registration)));return
    if intent.exists():raise RuntimeError('Submission intent exists; query Slurm before any retry')
    deployed=deploy()
    with connect() as c:
        old=read(REPORT/'formal_registration.json')
        terminal=command(c,'sacct -n -P -j '+old['job']+' --format=JobID,State,ExitCode',45)
        rows=[l.split('|') for l in terminal.splitlines() if l.strip()]
        if not rows or any(r[1].split()[0] not in ('FAILED','COMPLETED','CANCELLED') for r in rows):raise ValueError('Old launch has not fully exited')
        absent=script(c,'from pathlib import Path;print(Path('+repr(old['result'])+').exists())').strip()
        if absent!='False':raise ValueError('Old attempt has scientific output; do not rerun blindly')
        write(REPORT/('launch_failure_closure_'+ATTEMPT+'.json'),dict(time=now(),old_job=old['job'],accounting=terminal,no_scientific_output=True,fix='Pass immutable release as explicit sbatch argument; never use dirname($0), which Slurm relocates'))
        cap=capacity(c);write(REPORT/('capacity_before_submit_'+ATTEMPT+'.json'),cap)
        queue=command(c,"squeue -u scxi717 -h -o '%i|%j|%T|%D|%b'",45)
        if 'NEMA5e9_width' in queue:raise ValueError('Matching active job exists')
        partition=command(c,'scontrol show partition gpu_5090',30)
        if 'State=UP' not in partition:raise ValueError('GPU partition unavailable')
        args=['sbatch','--parsable','-p','gpu_5090','--qos=gpugpu','-N','8','-n','8','--ntasks-per-node=1','--cpus-per-task=8','--gres=gpu:1','--time=4320',
            '--job-name=NEMA5e9_width','--chdir='+REMOTE,'--output='+REMOTE+'/logs/%j.out','--error='+REMOTE+'/logs/%j.err',deployed['release']+'/launch.sh',deployed['release']]
        dry=command(c,' '.join(q(s) for s in args[:1]+['--test-only']+args[1:]),45)
        write(REPORT/('submission_preflight_'+ATTEMPT+'.json'),dict(time=now(),existing_queue=queue,partition=partition,test_only=dry,capacity=cap,command=args))
        write(intent,dict(time=now(),execution_key=deployed['execution_key'],command=args,auto_retry=False))
        returned=command(c,' '.join(q(s) for s in args),45).strip()
        if not re.fullmatch(r'\d+(?:;\S+)?',returned):raise ValueError('Ambiguous sbatch output: '+returned)
        job=returned.split(';')[0]
        reg=dict(job=job,time=now(),release=deployed['release'],execution_key=deployed['execution_key'],result=REMOTE+'/formal_'+job,nodes=8,gpus_per_node=1,total_gpus=8,scales=[100,80,60],iterations=10000,save_step=50)
        write(registration,reg)
        actual=command(c,'scontrol show job -dd '+job,30)
        (REPORT/('submitted_slurm_'+ATTEMPT+'.txt')).write_text(actual,encoding='utf8')
        if 'NumNodes=8' not in actual or 'gres/gpu=8' not in actual:raise ValueError('Actual submitted topology differs; inspect')
        print(json.dumps(reg),flush=True);print(actual,flush=True)

def status():
    reg=read(REPORT/('formal_registration_'+ATTEMPT+'.json'));job=reg['job'];stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with connect() as c:
        state=command(c,'sacct -n -P -j '+job+' --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS',45)
        out=dict(time=now(),job=job,accounting=state)
        code="""import pathlib,json
r=pathlib.Path(RESULT);d={}
for name in ('started.json','progress.json','completed.json','width100/benchmark_choice.json'):
 p=r/name
 if p.exists():d[name]=json.loads(p.read_text())
for suffix in ('out','err'):
 p=pathlib.Path(LOGS)/(JOB+'.'+suffix)
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6000));d[suffix]=f.read().decode(errors='replace')
print(json.dumps(d))
""".replace('RESULT',repr(reg['result'])).replace('LOGS',repr(REMOTE+'/logs')).replace('JOB',repr(job))
        out.update(json.loads(script(c,code,45)));write(REPORT/f'status_{stamp}.json',out)
        print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['deploy','submit','status']);a=p.parse_args()
    globals()[a.action]()
if __name__=='__main__':main()
