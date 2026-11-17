"""Bounded local CPU execution of the frozen PyTomography reconstruction.

No scientific operation changes: the reversible source patch affects only the
device, allocation metadata and memory guard. Unstarted GPU job is retained in
history and must be confirmed cancelled before starting this execution.
"""
from __future__ import annotations
import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import psutil
from opengate_ehe_common_20261113 import DATA,REPORT,HERE,read,save,sha,now


CPU_ALLOCATION='''def allocation():
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("This execution is explicitly local CPU, not Slurm")
    actual=psutil.virtual_memory()
    limit=int(os.environ["OPENGATE_LOCAL_HOST_CAP_BYTES"])
    return dict(job=None,execution_kind="LOCAL_CPU",host_bytes=limit,
        host_total_bytes=actual.total,host_available_at_start_bytes=actual.available,
        pid=os.getpid(),create_time=psutil.Process().create_time(),
        CPU_threads=8,scontrol=None,CUDA_used=False)
'''
CPU_RESOURCE='''def resource_check(alloc):
    memory=psutil.Process().memory_info()
    rss=max(memory.rss,getattr(memory,"peak_wset",0))
    available=psutil.virtual_memory().available
    if rss>.8*alloc["host_bytes"] or available<.2*alloc["host_bytes"]:
        raise MemoryError("Local CPU 20 percent host resource margin violated")
    return dict(rss_bytes=rss,allocated_host_bytes=alloc["host_bytes"],
        actual_available_host_bytes=available,execution_kind="LOCAL_CPU",
        gpu_reserved_peak_bytes=None,actual_gpu_total_bytes=None)
'''


def prepare():
    freeze=read(REPORT/'freeze.json');source=DATA/freeze['payload_relative']
    for name,r in freeze['files'].items():
        if sha(source/name)!=r['sha256']:raise ValueError('Changed frozen source: '+name)
    original=(source/'opengate_ehe_recon_20261113.py').read_text(encoding='utf8')
    functions={n.name:n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef)}
    lines=original.splitlines(keepends=True);text=original;changes=[]
    for name,new in [('allocation',CPU_ALLOCATION),('resource_check',CPU_RESOURCE)]:
        node=functions[name];old=''.join(lines[node.lineno-1:node.end_lineno])
        replacement=new.rstrip()+'\n'
        if text.count(old)!=1:raise ValueError('Ambiguous execution function')
        text=text.replace(old,replacement,1);changes.append((old,replacement))
    for old,new in [
        ('    if not torch.cuda.is_available():raise RuntimeError("Allocated CUDA GPU unavailable")\n',''),
        ('pytomography.set_device("cuda:0")','pytomography.set_device("cpu")'),
        ('gpu_name=torch.cuda.get_device_name()','gpu_name=None'),
        ('torch.cuda.empty_cache()','pass  # Local CPU: no CUDA allocator')]:
        if text.count(old)!=1:raise ValueError('Unexpected execution patch context')
        text=text.replace(old,new,1);changes.append((old,new))
    # Undo all edits and demand complete byte identity, including MLEM/callbacks.
    restored=text
    for old,new in reversed(changes):
        if new:
            if restored.count(new)!=1:raise ValueError('Ambiguous reversible source patch')
            restored=restored.replace(new,old,1)
        else:
            anchor='    torch.set_num_threads(8)\n'
            if restored.count(anchor)!=1:raise ValueError('Lost execution guard anchor')
            restored=restored.replace(anchor,old+anchor,1)
    if restored!=original:raise ValueError('A scientific source operation changed')
    ast.parse(text)
    identity=hashlib.sha256(text.encode()).hexdigest()[:16]
    dest=DATA/'local_cpu_releases'/identity;dest.mkdir(parents=True,exist_ok=True)
    for name in ('opengate_ehe_common_20261113.py','opengate_ehe_projector_20261113.py'):
        p=dest/name
        if p.exists() and sha(p)!=sha(source/name):raise ValueError('Changed local frozen dependency')
        if not p.exists():shutil.copy2(source/name,p)
    script=dest/'opengate_ehe_recon_cpu_20261113.py'
    if script.exists() and script.read_text(encoding='utf8')!=text:raise ValueError('Changed local execution release')
    if not script.exists():script.write_text(text,encoding='utf8',newline='\n')
    diff=''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),
        fromfile='frozen_gpu_reconstruction',tofile='local_CPU_execution'))
    (dest/'execution_only.diff').write_text(diff,encoding='utf8')
    proof=dict(passed=True,execution_release=identity,original_release=freeze['release'],
        reversible_scientific_source_identity_passed=True,execution_kind='LOCAL_CPU',
        modifications=['Real local allocation/RSS guard','CUDA device to CPU','No CUDA availability/allocator calls','Explicitly null GPU metadata'],
        original_reconstruction_sha256=sha(source/'opengate_ehe_recon_20261113.py'),
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in dest.iterdir() if p.is_file()})
    save(dest/'release.json',proof);save(REPORT/'local_cpu_execution_freeze.json',proof)
    return dest,proof


def run():
    if (REPORT/'local_cpu_exit.json').exists():raise FileExistsError('Local reconstruction already exited; preserve its outputs')
    cancelled=read(REPORT/'gpu_unstarted_job_cancellation_verified.json')
    if not cancelled['passed'] or cancelled['job']!=1806310:raise ValueError('Unstarted GPU cancellation not confirmed')
    bridge=read(REPORT/'maty_bridge_strict_fetch.json')
    if not bridge['passed'] or not bridge['all_files_SHA_verified']:raise ValueError('Unverified actual input')
    root=DATA/'delivery'
    for name,r in bridge['files'].items():
        p=root/name
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('Changed accepted input')
    if (root/'reconstruction').exists():raise FileExistsError('Refuse a parallel/repeated reconstruction')
    release,freeze=prepare()
    budget=int(min(8*(1<<30),.8*psutil.virtual_memory().available))
    if budget<4*(1<<30):raise MemoryError('Insufficient local host memory with reserve')
    env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',
             OPENGATE_LOCAL_HOST_CAP_BYTES=str(budget))
    command=[str(DATA/'local_env/Scripts/python.exe'),'-u',str(release/'opengate_ehe_recon_cpu_20261113.py'),
             '--root',str(root),'--inputs',str(root/'inputs')]
    execution=root/'local_execution';execution.mkdir()
    for p in release.iterdir():
        if p.is_file():shutil.copy2(p,execution/p.name)
    shutil.copy2(Path(__file__),execution/Path(__file__).name)
    start=time.monotonic();peak=0;last=0
    with (execution/'stdout.log').open('xb') as stdout,(execution/'stderr.log').open('xb') as stderr:
        child=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr)
        proc=psutil.Process(child.pid)
        registration=dict(status='RUNNING',pid=child.pid,create_time=proc.create_time(),command=proc.cmdline(),
            controller_pid=os.getpid(),controller_create_time=psutil.Process().create_time(),
            execution_kind='LOCAL_CPU',threads=8,host_bound_bytes=budget,started_utc=now(),
            execution_release=freeze['execution_release'],superseded_unstarted_gpu_job=1806310)
        save(REPORT/'local_cpu_registration.json',registration)
        print('LOCAL_CPU_FORMAL_STARTED',child.pid,freeze['execution_release'],flush=True)
        try:
            while child.poll() is None:
                try:
                    # Windows venv redirectors wait on a separate Python child.
                    # Count the actual descendant process as well as the launcher.
                    family=[proc]+proc.children(recursive=True)
                    usage=0
                    for member in family:
                        try:
                            mem=member.memory_info();usage+=max(mem.rss,getattr(mem,'peak_wset',0))
                        except psutil.NoSuchProcess:pass
                    peak=max(peak,usage)
                except psutil.NoSuchProcess:pass
                if peak>.8*budget or psutil.virtual_memory().available<.2*budget:
                    raise MemoryError('Local process exceeded the actual 20 percent resource reserve')
                if time.monotonic()-start>86400:raise TimeoutError('Bounded local formal reconstruction timeout')
                if time.monotonic()-last>30:
                    progress={}
                    for energy in ('440','218'):
                        p=root/'reconstruction'/energy/'progress.json'
                        if p.exists():progress[energy]=read(p)['iteration']
                    print('LOCAL_CPU_PROGRESS',progress,'rss_peak',peak,flush=True);last=time.monotonic()
                time.sleep(.5)
            if child.returncode:raise RuntimeError('Local reconstruction exit '+str(child.returncode))
            if not read(root/'reconstruction_complete.json')['passed']:raise ValueError('Missing complete result')
        except BaseException:
            if child.poll() is None:
                try:family=proc.children(recursive=True)+[proc]
                except psutil.NoSuchProcess:family=[]
                for member in family:
                    try:member.terminate()
                    except psutil.NoSuchProcess:pass
                _,alive=psutil.wait_procs(family,timeout=10)
                for member in alive:
                    try:member.kill()
                    except psutil.NoSuchProcess:pass
                child.wait(timeout=20)
            save(REPORT/'local_cpu_exit.json',dict(registration,status='FAILED',returncode=child.returncode,
                peak_rss_bytes=peak,elapsed_seconds=time.monotonic()-start,ended_utc=now()))
            raise
    result=dict(registration,status='COMPLETED',returncode=0,peak_rss_bytes=peak,
        elapsed_seconds=time.monotonic()-start,ended_utc=now(),passed=True)
    save(REPORT/'local_cpu_exit.json',result);save(execution/'exit.json',result)
    print('LOCAL_CPU_FORMAL_COMPLETE',result['elapsed_seconds'],flush=True)


def finalize(root=None,exit_path=None):
    cpu=read(REPORT/'maty_transport_strict_fetch.json');exit=read(exit_path or REPORT/'local_cpu_exit.json')
    if not cpu['passed'] or not cpu['all_files_SHA_verified'] or not exit['passed'] or exit['returncode']!=0:
        raise ValueError('Full successful raw transport and actual local exit required')
    root=Path(root or DATA/'delivery');allocation=read(root/'reconstruction_allocation.json')
    bridge=read(REPORT/'maty_bridge_strict_fetch.json')
    for name,r in bridge['files'].items():
        p=root/name
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:
            raise ValueError('Final accepted input identity changed: '+name)
    if allocation['execution_kind']!='LOCAL_CPU' or allocation['CUDA_used']:
        raise ValueError('Execution provenance does not identify actual CPU use')
    frames=0;actual_peak=0
    for energy in ('440','218'):
        folder=root/'reconstruction'/energy;complete=read(folder/'complete.json')
        if not complete['passed'] or complete['checkpoints']!=200 or complete['iterations']!=200:raise ValueError('Incomplete actual MLEM')
        actual_peak=max(actual_peak,complete['resources']['rss_bytes'])
        actual_peak=max(actual_peak,max(row['rss_bytes'] for row in read(folder/'history.json')))
        for name,r in complete['files'].items():
            p=folder/name
            if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('Changed reconstruction member')
        with np.load(folder/'sensitivity.npz') as z:fraction=z['fraction_xyz']
        for i in range(1,201):
            with np.load(folder/f'iteration_{i:04d}.npz') as z:a=z['density_xyz']
            if not np.isfinite(a).all() or np.any(a<0) or np.any(a[fraction==0]!=0):raise ValueError('Invalid native reconstruction frame')
            frames+=1
    files={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))
           for p in root.rglob('*') if p.is_file()}
    if any(n.endswith('.writing') for n in files):raise ValueError('Unfinished output remains')
    result=dict(passed=True,strict_fetch_passed=True,all_files_SHA_verified=True,
        files=files,checkpoints=frames,actual_imaging_primaries=5_000_000_000,
        execution_backend='LOCAL_CPU_8_THREADS',remote_reconstruction_job=None,
        delivery_root=str(root),actual_scientific_peak_rss_bytes=actual_peak,
        reconstruction_provenance='Reconstructed locally from fully SHA-verified OpenGATE input; no remote GPU reconstruction or GPU fetch is claimed',
        transport_receipt_sha256=sha(REPORT/'maty_transport_strict_fetch.json'),
        local_execution_exit_sha256=sha(exit_path or REPORT/'local_cpu_exit.json'),
        resume_plan_sha256=sha(REPORT/'local_cpu_resume_plan.json') if exit_path else None,
        all_native_frames_finite_nonnegative_outside_zero=True,
        completed_utc=now(),physical_calibration_passed=False,visual_QA_performed=False)
    save(REPORT/'strict_delivery.json',result)
    print('STRICT_LOCAL_RECONSTRUCTION_DELIVERY',frames,len(files),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','finalize'])
    a=p.parse_args()
    if a.action=='prepare':print(prepare()[1]['execution_release'])
    elif a.action=='run':run()
    else:finalize()
