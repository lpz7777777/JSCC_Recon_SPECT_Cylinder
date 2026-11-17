"""Explicit controller ownership handoff and bounded real-byte transfer benchmark."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import zlib
import psutil
import stream_transfer as st
from repair_workflow import verify,LOCAL_REG,DATA,REPORT,REG
from common import read,write,now,sha


def local_checks():
    records=[]
    with tempfile.TemporaryDirectory(prefix='cart_transfer_check_') as tmp:
        folder=Path(tmp);source=folder/'source.bin'
        data=bytes(range(256))*(8192)+bytes(3*(1<<20))+b'tail-31'
        source.write_bytes(data)
        for compression in ('raw','zlib1'):
            ranges=[[19,2*(1<<20)+7],[len(data)-35,35]]
            cfg=dict(source=str(source),source_bytes=len(data),ranges=ranges,compression=compression)
            x=subprocess.run([sys.executable,'-B','-c',st.SENDER,json.dumps(cfg)],capture_output=True,check=True)
            expected=b''.join(data[a:a+n] for a,n in ranges)
            assert (zlib.decompress(x.stdout) if compression=='zlib1' else x.stdout)==expected
            for offset in (0,12345):
                target=folder/(compression+str(offset)+'.bin')
                if offset:target.write_bytes(expected[:offset])
                payload=zlib.compress(expected[offset:],1) if compression=='zlib1' else expected[offset:]
                cfg=dict(target=str(target),expected_bytes=len(expected),expected_sha256=hashlib.sha256(expected).hexdigest(),
                    compression=compression,offset=offset,prefix_sha256=hashlib.sha256(expected[:offset]).hexdigest() if offset else None)
                result=subprocess.run([sys.executable,'-B','-c',st.RECEIVER,json.dumps(cfg)],input=payload,capture_output=True,check=True)
                r=json.loads(result.stdout);assert r['passed'] and target.read_bytes()==expected
                records.append(dict(compression=compression,resume_offset=offset,passed=True,tail_bytes=35))
        target=folder/'negative.bin';target.write_bytes(b'original prefix')
        cfg=dict(target=str(target),expected_bytes=20,expected_sha256='0'*64,compression='raw',offset=15,prefix_sha256='0'*64)
        x=subprocess.run([sys.executable,'-B','-c',st.RECEIVER,json.dumps(cfg)],input=b'bad',capture_output=True)
        assert x.returncode and target.read_bytes()==b'original prefix'
        records.append(dict(invalid_prefix_rejected_without_write=True))
    result=dict(passed=True,checks=records,synthetic_byte_transport_only=True,scientific_model_modified=False)
    write(REPORT/'transfer_local_checks_20261117.json',result)
    return result


def stop_old():
    verify()
    marker=REPORT/'transfer_ownership_stop_20261117.json'
    if marker.exists():raise FileExistsError('Already handed off; inspect, never repeat stop')
    if (REPORT/'formal_registration.json').exists() or (REPORT/'formal_submission_intent.json').exists():
        raise RuntimeError('Formal submit exists; stop and inspect before transfer-only handoff')
    old=read(LOCAL_REG);parent=psutil.Process(old['pid'])
    if parent.create_time()!=old['create_time'] or parent.cmdline()!=old['cmdline']:raise ValueError('Old PID identity differs')
    children=parent.children(recursive=True)
    senders=[]
    for child in children:
        args=child.cmdline()
        if Path(child.exe()).name.lower()=='ssh.exe' and any(st.SERVER==a for a in args) and any('B_detector_zyx.float32' in a for a in args):
            senders.append(dict(pid=child.pid,create_time=child.create_time(),cmdline=args))
    plan=dict(time=now(),reason='User requested matrix-transfer acceleration; no scientific job cancellation',
        original_registration=old,source_senders=senders,old_controller_source_sha256=sha(Path(old['cmdline'][-1])),
        original_frozen_math_preserved=True)
    write(REPORT/'transfer_ownership_stop_intent_20261117.json',plan)
    parent.terminate();parent.wait(timeout=30)
    for record in senders:
        try:
            q=psutil.Process(record['pid'])
            if q.create_time()==record['create_time'] and q.cmdline()==record['cmdline']:
                q.terminate();q.wait(timeout=30)
        except psutil.NoSuchProcess:pass
    result=dict(plan,passed=True,parent_exited=True,source_senders_exited=True,stopped_utc=now(),
                old_registration_retained=True,remote_jobs_cancelled=False,partial_preserved=True)
    write(marker,result);return result


def benchmark():
    stopped=read(REPORT/'transfer_ownership_stop_20261117.json');assert stopped['passed']
    old=stopped['original_registration']
    try:
        q=psutil.Process(old['pid'])
        if q.create_time()==old['create_time'] and q.cmdline()==old['cmdline']:raise RuntimeError('Old controller still alive')
    except psutil.NoSuchProcess:pass
    original=read(REG);matrix=read(REPORT/'matrix_repair_latest_status_20261116.json')['completed.json']
    source=original['output']+'/A440/B_detector_zyx.float32';size=matrix['channels']['A440']['bytes']
    block=16<<20;ranges=[[0,block],[size//2,block],[size-block,block]]
    # Only read a fixed 48MiB across a real frozen matrix, never create scientific histories.
    code='''import hashlib,json,time,zlib
p=FILE;rr=RANGES;h=hashlib.sha256();comp=zlib.compressobj(1);n=0;wire=0;t=time.monotonic()
with open(p,'rb') as f:
 for a,count in rr:
  f.seek(a);b=f.read(count);assert len(b)==count;h.update(b);n+=len(b);wire+=len(comp.compress(b))
wire+=len(comp.flush())
print(json.dumps(dict(sha256=h.hexdigest(),bytes=n,zlib1_bytes=wire,source_compress_seconds=time.monotonic()-t)))'''.replace('FILE',repr(source)).replace('RANGES',repr(ranges))
    sample=json.loads(st.server(code,timeout=120))
    root=st.REMOTE_GPU+'/transfer_bench_20261117';results=[]
    with st.connect() as client:
        for compression in ('raw','zlib1'):
            target=root+'/'+compression+'_'+str(time.time_ns())+'.bin'
            result=st.bridge(client,source,size,ranges,target,sample['bytes'],sample['sha256'],compression,
                             progress=False,maximum_seconds=300)
            results.append(result);print('BENCHMARK',compression,json.dumps(result),flush=True)
    best=max(results,key=lambda r:r['effective_MiB_per_second'])
    result=dict(passed=True,time=now(),source=source,source_full_sha256=matrix['channels']['A440']['sha256'],
        sample_ranges=ranges,sample=sample,results=results,selected_compression=best['compression'],
        measured_best_MiB_per_second=best['effective_MiB_per_second'],
        original_observed_MiB_per_second=2.1125571867987096,
        measured_speedup_vs_previous_observation=best['effective_MiB_per_second']/2.1125571867987096,
        benchmark_is_IO_only=True,no_Slurm_jobs_submitted=True)
    write(REPORT/'transfer_benchmark_20261117.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','stop','benchmark']);a=p.parse_args()
    r={'check':local_checks,'stop':stop_old,'benchmark':benchmark}[a.action]()
    print(json.dumps({k:v for k,v in r.items() if k!='original_registration'},indent=2))
