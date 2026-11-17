"""Isolated formal OpenGATE campaign; immutable release and non-repeating submit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
from opengate_ehe_common_20261113 import *
from campaign_20261102_io import execute,python_json,upload_files,sftp,mkdirs
from ehe_5e9_workflow import q

SCIENCE_FILES=["opengate_ehe_common_20261113.py","opengate_ehe_transport_20261113.py",
               "opengate_ehe_projector_20261113.py","opengate_ehe_collect_20261113.py",
               "opengate_ehe_recon_20261113.py","opengate_ehe_guard_20261113.py"]
GPU_PYTHON=GPU_ROOT+"/environment/py310/bin/python"


def freeze(repair_reason=None):
    previous=None
    if (REPORT/"freeze.json").exists():
        if not repair_reason:raise FileExistsError("Release already frozen")
        if any((REPORT/(s+"_job.json")).exists() for s in ("imaging","response","collect","reconstruction")):
            raise ValueError("A registered release requires an explicit exited-job repair, not this preparation helper")
        previous=read(REPORT/"freeze.json")
        history=REPORT/"releases"/previous['release'];history.mkdir(parents=True,exist_ok=True)
        for n in ("freeze.json","deployment_maty.json","imaging_submission_intent.json"):
            if (REPORT/n).exists():shutil.copy2(REPORT/n,history/n)
    payload=DATA/("payload" if previous is None else "payload_repair_"+str(time.time_ns()))
    payload.mkdir(parents=True,exist_ok=False)
    for n in SCIENCE_FILES:shutil.copy2(HERE/n,payload/n)
    shutil.copytree(DATA/"inputs",payload/"inputs")
    env=CPU_ROOT+"/environment"
    shell=("#!/bin/bash\nset -euo pipefail\n"
           f"export SINGULARITY_CACHEDIR={q(env+'/cache')}\n"
           f"export SINGULARITYENV_LD_LIBRARY_PATH={q(env+'/native_libs/usr/lib/x86_64-linux-gnu')}\n"
           "export SINGULARITYENV_OMP_NUM_THREADS=1\n"
           "export SINGULARITYENV_OPENBLAS_NUM_THREADS=1\n"
           "export SINGULARITYENV_ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1\n"
           "export SINGULARITYENV_SLURM_JOB_ID=\"${SLURM_JOB_ID:-}\"\n"
           "export SINGULARITYENV_SLURM_CPUS_PER_TASK=\"${SLURM_CPUS_PER_TASK:-1}\"\n"
           f"exec singularity exec --cleanenv --bind /WORK:/WORK {q(env+'/python311_bookworm_ecr.sif')} "
           f"{q(env+'/venv/bin/python')} \"$@\"\n")
    (payload/"cpu_python.sh").write_text(shell,encoding="ascii",newline="\n")
    members={p.relative_to(payload).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))
             for p in sorted(payload.rglob('*')) if p.is_file()}
    identity=hashlib.sha256(json.dumps(members,sort_keys=True).encode()).hexdigest()[:16]
    record=dict(study=STUDY,release=identity,created_utc=now(),files=members,
                payload_relative=payload.relative_to(DATA).as_posix(),
                source_truth_sha256=sha(TRUTH),condition="original_vacuum",
                imaging_budget=5_000_000_000,response_characterization_budget=4_400_000_000,
                numerical_checks="8 local tests passed; geometry, directions, units, exact transpose/S, MLEM, FOV integral")
    if previous:record.update(previous_release=previous['release'],execution_repair_reason=repair_reason)
    save(payload/"release.json",record);save(REPORT/"freeze.json",record)
    return record


def release(host):
    f=read(REPORT/"freeze.json")
    return (CPU_ROOT if host=="maty" else GPU_ROOT)+"/releases/"+f["release"]


def deploy(host):
    f=read(REPORT/"freeze.json")
    payload=DATA/f.get('payload_relative','payload')
    files={p.relative_to(payload).as_posix():p for p in payload.rglob('*') if p.is_file()}
    for n,r in f["files"].items():
        if sha(payload/n)!=r["sha256"]:raise ValueError("Frozen local payload changed")
    upload_files(host,release(host),files)
    code='''import hashlib,json,pathlib,sys
a=json.loads(sys.argv[1]);root=pathlib.Path(a['root']);f=json.loads((root/'release.json').read_text())
for n,v in f['files'].items():
 p=root/n
 if p.stat().st_size!=v['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=v['sha256']:raise ValueError(n)
print('CAMPAIGN_JSON '+json.dumps({'passed':True,'release':f['release'],'files':len(f['files'])}))'''
    proof=python_json(host,"python3",code,dict(root=release(host)),timeout=90)
    save(REPORT/f"deployment_{host}.json",dict(**proof,created_utc=now(),remote_release=release(host)))


def scripts():
    r=release("maty");py=CPU_ROOT+"/environment/venv/bin/python"
    out={}
    for kind,n in (("imaging",200),("response",88)):
        lines=["#!/bin/bash","set -euo pipefail",f"cd {q(r)}",
               f"mkdir -p {q(CPU_ROOT+'/workers/'+kind)} {q(CPU_ROOT+'/resources')} {q(CPU_ROOT+'/preflight')}",
               'index="${SLURM_ARRAY_TASK_ID:?}"',
               'printf -v index4 "%04d" "$index"',
               f'allocation={q(CPU_ROOT+"/resources/"+kind+"_")}\"$index4\".allocation.txt',
               'scontrol show job "${SLURM_JOB_ID:?}" > "$allocation"']
        if kind=="imaging":
            lines.extend(['if [ "$index" = 0 ]; then',
                f" bash {q(r+'/cpu_python.sh')} {q(r+'/opengate_ehe_guard_20261113.py')} "
                f"--record {q(CPU_ROOT+'/resources/inline_readout.json')} --allocation \"$allocation\" --seconds 900 -- "
                f"{q(py)} {q(r+'/opengate_ehe_transport_20261113.py')} self_check "
                f"--inputs {q(r+'/inputs')} --output {q(CPU_ROOT+'/preflight/inline_readout')}","fi"])
        lines.extend([
            f"bash {q(r+'/cpu_python.sh')} {q(r+'/opengate_ehe_guard_20261113.py')} "
            f"--record {q(CPU_ROOT+'/resources/'+kind+'_')}\"$index4\".json --allocation \"$allocation\" --seconds 250000 -- "
            f"{q(py)} {q(r+'/opengate_ehe_transport_20261113.py')} run --inputs {q(r+'/inputs')} "
            f"--kind {kind} --index \"$index\" --output {q(CPU_ROOT+'/workers/'+kind+'/')}\"$index4\""])
        out[kind]="\n".join(lines)+"\n"
    out["collect"]=("#!/bin/bash\nset -euo pipefail\n"
        f"ln -s {q(r+'/inputs')} {q(CPU_ROOT+'/inputs')}\n"
        f"bash {q(r+'/cpu_python.sh')} {q(r+'/opengate_ehe_collect_20261113.py')} --root {q(CPU_ROOT)}\n")
    g=release("gpu")
    release_reservation="""import json,pathlib,os
r=pathlib.Path(ROOT_VALUE);p=r/'control/output_reservation.bin'
a=json.loads((r/'control/output_capacity.json').read_text())
if not a['passed'] or p.stat().st_size!=a['reserved_bytes']:raise ValueError('Actual bounded output reservation is missing')
p.unlink()
print('OUTPUT_CAPACITY_RESERVATION_RELEASED',a['reserved_bytes'],flush=True)
""".replace('ROOT_VALUE',repr(GPU_ROOT))
    out["reconstruction"]=("#!/bin/bash\nset -euo pipefail\n"
        "export OMP_NUM_THREADS=8\nexport MKL_NUM_THREADS=8\n"
        f"export XDG_CACHE_HOME={q(GPU_ROOT+'/environment/cache')}\n"
        f"cd {q(g)}\n"
        f"python3 -c {q(release_reservation)}\n"
        f"{q(GPU_PYTHON)} -u {q(g+'/opengate_ehe_recon_20261113.py')} --root {q(GPU_ROOT)} "
        f"--inputs {q(g+'/inputs')}\n")
    folder=DATA/"scripts";folder.mkdir(exist_ok=True)
    for name,body in out.items():
        (folder/(name+".sh")).write_text(body,encoding="ascii",newline="\n")
    return out


def reserve_gpu_output():
    """Enforce bounded real allocation, rather than confuse volume free with quota."""
    required=int(400*126*126*34*8/.8)+(1<<30)
    source='''import fcntl,json,os,pathlib,sys,time
a=json.loads(sys.argv[1]);root=pathlib.Path(a['root']);control=root/'control';control.mkdir(exist_ok=True)
lock=(control/'output_capacity.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
proof=control/'output_capacity.json';p=control/'output_reservation.bin'
if proof.exists():
 r=json.loads(proof.read_text())
 if not p.exists() or p.stat().st_size!=a['bytes']:raise ValueError('Prior reservation missing or already used')
else:
 if p.exists():raise ValueError('Unresolved allocation attempt; inspect before retry')
 with p.open('xb') as f:
  os.posix_fallocate(f.fileno(),0,a['bytes']);f.flush();os.fsync(f.fileno())
 s=p.stat()
 if s.st_size!=a['bytes'] or s.st_blocks*512<a['bytes']:raise ValueError('Filesystem did not confirm actual allocated blocks')
 r=dict(passed=True,reserved_bytes=a['bytes'],allocated_blocks_bytes=s.st_blocks*512,
  planned_uncompressed_checkpoints_bytes=400*126*126*34*8,reserve_fraction=.2,
  extra_allowance_bytes=1<<30,actual_directory=str(root),created_epoch=time.time(),
  method='POSIX real allocation in the actual output directory; released once by the formal batch')
 with proof.open('x') as f:json.dump(r,f,indent=2);f.flush();os.fsync(f.fileno())
print('CAMPAIGN_JSON '+json.dumps(r))'''
    result=python_json('gpu','python3',source,dict(root=GPU_ROOT,bytes=required),timeout=120,readonly=False)
    save(REPORT/'gpu_output_capacity.json',result)
    return result


def submit(stage,dependency=None):
    host="gpu" if stage=="reconstruction" else "maty"
    root=GPU_ROOT if host=="gpu" else CPU_ROOT
    receipt=REPORT/(stage+"_job.json")
    if receipt.exists():return read(receipt)
    if not (REPORT/f"deployment_{host}.json").exists():raise ValueError("Deploy and verify frozen release first")
    if stage=='reconstruction':
        for name in ('gpu_actual_input_verified.json','gpu_output_capacity.json'):
            if not (REPORT/name).exists() or not read(REPORT/name)['passed']:
                raise ValueError('Actual input/capacity gate is incomplete: '+name)
    scripts();script=DATA/"scripts"/(stage+".sh")
    script_root=root+"/scripts/"+read(REPORT/"freeze.json")['release']
    upload_files(host,script_root,{script.name:script})
    name="og_ehe5e9_"+stage
    flags=(["--partition=gpu_5090","--account=scxi717","--qos=gpugpu","-N","1","-n","1","-c","8","--gres=gpu:1","--time=24:00:00"]
           if host=="gpu" else ["--partition=cnmix","-N","1","-n","1","-c","1","--time=72:00:00"])
    if stage in ("imaging","response"):
        flags+=["--array="+("0-199%1" if stage=="imaging" else "0-87%1"),"--hold"]
    if dependency:flags+=["--dependency=afterok:"+dependency]
    flags+=["--job-name="+name,"--output="+root+"/logs/"+stage+"_%A_%a.log"]
    plan=dict(stage=stage,release=read(REPORT/"freeze.json")["release"],
              root=root,name=name,flags=flags,script=script_root+"/"+script.name,
              script_sha256=sha(script),created_utc=now())
    save(REPORT/(stage+"_submission_intent.json"),plan)
    source='''import fcntl,hashlib,json,os,pathlib,re,subprocess,sys,time
a=json.loads(sys.argv[1]);root=pathlib.Path(a['root']);(root/'control').mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
lock=(root/'control'/('submit_'+a['stage']+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipt=root/'control'/(a['stage']+'_job.json');intent=root/'control'/(a['stage']+'_intent.json')
def save(p,d):
 t=p.with_suffix('.writing');f=t.open('w');json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno());f.close();os.replace(t,p)
if receipt.exists():
 r=json.loads(receipt.read_text())
 if r['release']!=a['release'] or r['script_sha256']!=a['script_sha256']:raise ValueError('Different prior registration')
else:
 if intent.exists():raise RuntimeError('Unresolved remote submission intent: inspect Slurm, never blindly resubmit')
 active=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j'],text=True)
 if any(v.split('|')[-1]==a['name'] for v in active.splitlines()):raise RuntimeError('Same-name active job')
 if hashlib.sha256(pathlib.Path(a['script']).read_bytes()).hexdigest()!=a['script_sha256']:raise ValueError('Changed batch script')
 save(intent,a)
 p=subprocess.run(['sbatch','--parsable']+a['flags']+[a['script']],capture_output=True,text=True,timeout=60)
 if p.returncode:save(root/'control'/(a['stage']+'_submission_failed.json'),{'returncode':p.returncode,'stderr':p.stderr});raise RuntimeError(p.stderr)
 number=p.stdout.strip().split(';')[0]
 if not number.isdigit():raise RuntimeError('Uncertain scheduler response, inspect before retry')
 r=dict(a,job=int(number),submitted_epoch=time.time());save(receipt,r)
 if a['stage'] in ['imaging','response']:
  subprocess.run(['scontrol','release',number+'_0'],check=True,timeout=30)
print('CAMPAIGN_JSON '+json.dumps(r))'''
    record=python_json(host,"python3",source,plan,timeout=90,readonly=False)
    save(receipt,record)
    return record


def status():
    records={}
    for stage in ("imaging","response","collect","reconstruction"):
        file=REPORT/(stage+"_job.json")
        if not file.exists():continue
        job=read(file);host="gpu" if stage=="reconstruction" else "maty"
        cmd=f"squeue -h -r -j {job['job']} -o '%i|%T|%M|%R'; timeout 30s sacct -n -P -S 2026-11-13 -j {job['job']} --format=JobID,State,ExitCode,MaxRSS,Elapsed,AllocTRES"
        raw=execute(host,cmd,timeout=90,readonly=True)
        lines=[s.split('|') for s in raw.splitlines() if '|' in s]
        accounting=[v for v in lines if len(v)>=6]
        bad=[v for v in accounting if v[1].split()[0] in ("FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","BOOT_FAIL","PREEMPTED")]
        complete=bool(accounting) and all(v[1]=="COMPLETED" and v[2]=="0:0" for v in accounting)
        records[stage]=dict(job=job['job'],complete=complete,failures=bad,raw=raw)
    save(REPORT/"latest_status.json",dict(created_utc=now(),stages=records))
    return records


def raise_throttle(kind):
    job=read(REPORT/(kind+"_job.json"))["job"]
    marker=REPORT/(kind+"_throttle.json")
    if marker.exists():return read(marker)
    raw=execute("maty",f"timeout 30s sacct -n -P -S 2026-11-13 -j {job} --format=JobID,State,ExitCode",45,True)
    rows=[x.split('|') for x in raw.splitlines() if '|' in x and
          (x.split('|')[0]==f'{job}_0' or x.split('|')[0].startswith(f'{job}_0.'))]
    if not rows or any(v[1]!="COMPLETED" or v[2]!="0:0" for v in rows):return None
    target=160 if kind=="imaging" else 40
    end=199 if kind=="imaging" else 87
    result=execute("maty",f"scontrol update JobId={job} ArrayTaskThrottle={target} && scontrol release {q(str(job)+'_[1-'+str(end)+']')} && scontrol show job {job}",300,False)
    if f"ArrayTaskThrottle={target}" not in result and f"%{target}" not in result:
        raise ValueError("Scheduler did not confirm the new array concurrency")
    record=dict(job=job,target_concurrency=target,created_utc=now(),verified_scontrol=result)
    save(marker,record);return record


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["freeze","deploy_cpu","deploy_gpu","submit_cpu","status","throttle"])
    a=p.parse_args()
    if a.action=="freeze":print(json.dumps(freeze(),indent=2))
    elif a.action.startswith("deploy_"):deploy("maty" if a.action=="deploy_cpu" else "gpu")
    elif a.action=="submit_cpu":
        first=submit("imaging");second=submit("response")
        third=submit("collect",f"{first['job']}:{second['job']}")
        print(json.dumps({k:v['job'] for k,v in [('imaging',first),('response',second),('collect',third)]}))
    elif a.action=="throttle":
        for k in ("imaging","response"):print(k,raise_throttle(k))
    else:
        r=status()
        print(json.dumps({k:dict(job=v['job'],complete=v['complete'],failed=len(v['failures']),tail=v['raw'][-1700:]) for k,v in r.items()},indent=2))
