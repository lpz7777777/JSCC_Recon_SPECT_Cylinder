"""New exclusive finite handoff; frozen scientific code stays unchanged."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import time
import traceback
import psutil
import framed_stream_r2 as st
import chunked_transfer_r2 as ct
from repair_workflow import verify as verify_original,status as matrix_status,LOCAL_REG as OLD_REG,BINDING
from common import read,write,sha,now
from prepare import DATA,REPORT
from workflow import gpu_preflight
import handoff

REG=DATA/'completion_transfer_r2_registration_20261117.json'
MANIFEST=REPORT/'transfer_r2_execution_manifest_20261117.json'


def bind():
    original=verify_original()
    old=read(OLD_REG)
    try:
        p=psutil.Process(old['pid'])
        if p.create_time()==old['create_time'] and p.cmdline()==old['cmdline']:raise RuntimeError('Original controller still alive')
    except psutil.NoSuchProcess:pass
    assert read(REPORT/'transfer_ownership_stop_20261117.json')['passed']
    stopped=read(REPORT/'transfer_r2_stop_20261117.json');assert stopped['passed']
    prior=stopped['original_registration']
    try:
        p=psutil.Process(prior['pid'])
        if p.create_time()==prior['create_time'] and p.cmdline()==prior['cmdline']:raise RuntimeError('r1 controller still alive')
    except psutil.NoSuchProcess:pass
    assert read(REPORT/'transfer_benchmark_20261117.json')['passed']
    assert read(REPORT/'transfer_parallel_benchmark_20261117.json')['passed']
    paths=[Path(__file__),Path(st.__file__),Path(ct.__file__),Path(__file__).parent/'test_framed_r2.py',Path(__file__).parent/'prepare_stream.py',Path(__file__).parent/'benchmark_parallel.py',
           REPORT/'transfer_ownership_stop_20261117.json',REPORT/'transfer_benchmark_20261117.json',
           REPORT/'transfer_local_checks_20261117.json',REPORT/'transfer_framed_checks_r2_20261117.json',REPORT/'transfer_r2_stop_20261117.json',Path(st.original.__file__),REPORT/'transfer_merge_checks_20261117.json',REPORT/'transfer_parallel_benchmark_20261117.json',BINDING]
    records={str(p):sha(p) for p in paths}
    result=dict(time=now(),deadline_utc=original['deadline_utc'],original_binding_sha256=sha(BINDING),
        files=records,scientific_release='95022158ffedcdd7',matrix_execution_repair='009433dd7e2b3b50',
        modified_scope='Lossless I/O stream and verified prefix resume only; original handoff math/reconstruction/Slurm unchanged',
        periodic_automation=False,scientific_resubmission=False,remote_jobs_cancelled=False)
    result['key']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()[:16]
    write(MANIFEST,result);return result


def main():
    if REG.exists() or MANIFEST.exists():raise RuntimeError('Inspect existing new handoff; never duplicate')
    lock=DATA/'completion_transfer_r2_20261117.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    d={}
    try:
        manifest=bind();p=psutil.Process();deadline=datetime.datetime.fromisoformat(manifest['deadline_utc']).timestamp()
        d=dict(pid=p.pid,create_time=p.create_time(),cmdline=p.cmdline(),status='ACTIVE',started=now(),
            deadline_utc=manifest['deadline_utc'],execution_transfer=manifest['key'],manifest_sha256=sha(MANIFEST),
            periodic_chat_automation=False,source_scientific_release=manifest['scientific_release'],
            can_cancel=False,scientific_resubmission=False)
        write(REG,d)
        write(DATA/'completion_current_controller_20261117.json',dict(registration=str(REG),**d),replace=True)
        handoff.matrix_status=matrix_status
        handoff.transfer_one=ct.transfer_one
        while time.time()<deadline:
            verify_original()
            for path,h in manifest['files'].items():
                if sha(path)!=h:raise ValueError('Transfer-only binding changed: '+path)
            if not (REPORT/'formal_registration.json').exists():
                if (REPORT/'formal_submission_intent.json').exists():raise RuntimeError('Uncertain submission; inspect scheduler before any retry')
                ready=gpu_preflight()
                if not ready['passed']:
                    d.update(last_check=now(),stage='GPU_CONNECTION_OR_QUOTA_WAIT',reason=ready.get('error','Resource gate'))
                    write(REG,d,replace=True);time.sleep(60);continue
                d.update(last_check=now(),stage='LOSSLESS_MATRIX_STREAM_AND_FORMAL_SUBMIT');write(REG,d,replace=True)
                job=handoff.transfer_and_submit();d['job']=job['job']
                print('CARTESIAN_FORMAL_SUBMITTED',json.dumps(job),flush=True)
            try:s=handoff.formal_status()
            except Exception as error:
                d.update(last_check=now(),stage='FORMAL_NETWORK_WAIT',error=repr(error));write(REG,d,replace=True)
                time.sleep(60);continue
            write(REPORT/'formal_latest_status.json',s,replace=True)
            d.update(last_check=now(),stage='FORMAL_'+s['decision'],progress=s['details'].get('progress.json'));write(REG,d,replace=True)
            if s['decision']=='FAILED':raise RuntimeError('Formal scientific failure; preserve outputs, no automatic re-compute')
            if s['decision']=='SUCCESS':
                delivery=handoff.fetch()
                d.update(status='STRICTLY_DELIVERED_VISUAL_QA_PENDING',completed=now(),delivery=delivery)
                write(REG,d,replace=True);return
            time.sleep(120)
        raise TimeoutError('Original finite deadline reached; no remote job cancelled')
    except BaseException:
        if d:
            d.update(status='STOPPED_FAILURE',stopped=now(),traceback=traceback.format_exc())
            write(REG,d,replace=True)
        raise
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()
