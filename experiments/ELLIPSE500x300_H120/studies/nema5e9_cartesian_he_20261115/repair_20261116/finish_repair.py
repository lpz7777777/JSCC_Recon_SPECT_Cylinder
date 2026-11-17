"""Single finite handoff, original deadline retained; no automatic reruns."""
import datetime
import os
import time
import traceback
import psutil
from repair_workflow import verify, status, BINDING, LOCAL_REG, MANIFEST, DATA, REPORT
from common import read, write, sha, now
from workflow import gpu_preflight
import handoff


def main():
    binding=verify()
    if LOCAL_REG.exists():raise RuntimeError('Recovery controller already registered')
    lock=DATA/'completion_repair_20261116.lock'
    fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    p=psutil.Process()
    deadline=datetime.datetime.fromisoformat(binding['deadline_utc']).timestamp()
    d=dict(pid=p.pid,create_time=p.create_time(),cmdline=p.cmdline(),status='ACTIVE',started=now(),
        deadline_utc=binding['deadline_utc'],binding_sha256=sha(BINDING),execution_repair=read(MANIFEST)['key'],
        periodic_chat_automation=False,can_cancel=False,scientific_resubmission=False)
    write(LOCAL_REG,d)
    # Reuse frozen handoff implementation, bind only its matrix status provider.
    handoff.matrix_status=status
    failures=0
    try:
        while time.time()<deadline:
            verify()
            if not (REPORT/'formal_registration.json').exists():
                try:s=status()
                except Exception as e:
                    failures+=1;d.update(last_check=now(),stage='MATRIX_NETWORK_WAIT',error=repr(e));write(LOCAL_REG,d,replace=True)
                    if failures>144:raise
                    time.sleep(300);continue
                failures=0
                if not s['alive'] and 'completed.json' not in s:
                    raise RuntimeError('Recovery matrix exited without completion; no automatic retry')
                if s['alive']:
                    d.update(last_check=now(),stage='MATRIX_RUNNING',progress=s.get('progress.json'));write(LOCAL_REG,d,replace=True)
                    time.sleep(300);continue
                ready=gpu_preflight()
                if not ready['passed']:
                    d.update(last_check=now(),stage='GPU_CONNECTION_OR_QUOTA_WAIT',reason=ready.get('error','Quota/partition not ready'))
                    write(LOCAL_REG,d,replace=True);time.sleep(300);continue
                d.update(last_check=now(),stage='STRICT_MATRIX_TRANSFER_AND_FORMAL_SUBMIT');write(LOCAL_REG,d,replace=True)
                proof=REPORT/'formal_repair_binding_20261116.json'
                if not proof.exists():write(proof,dict(execution_repair=read(MANIFEST)['key'],matrix_completed=s['completed.json'],binding_sha256=sha(BINDING)))
                job=handoff.transfer_and_submit();d['job']=job['job']
            try:s=handoff.formal_status()
            except Exception as e:
                failures+=1;d.update(last_check=now(),stage='FORMAL_NETWORK_WAIT',error=repr(e));write(LOCAL_REG,d,replace=True)
                if failures>144:raise
                time.sleep(300);continue
            failures=0
            write(REPORT/'formal_latest_status.json',s,replace=True)
            d.update(last_check=now(),stage='FORMAL_'+s['decision'],progress=s['details'].get('progress.json'));write(LOCAL_REG,d,replace=True)
            if s['decision']=='FAILED':raise RuntimeError('Formal failed; no automatic resubmission')
            if s['decision']=='SUCCESS':
                verify();delivery=handoff.fetch()
                d.update(status='STRICTLY_DELIVERED_VISUAL_QA_PENDING',completed=now(),delivery=delivery);write(LOCAL_REG,d,replace=True);return
            time.sleep(300)
        raise TimeoutError('Original finite deadline reached; remote jobs untouched')
    except Exception as e:
        d.update(status='STOPPED_FAILURE',stopped=now(),error=repr(e),traceback=traceback.format_exc());write(LOCAL_REG,d,replace=True);raise
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()
