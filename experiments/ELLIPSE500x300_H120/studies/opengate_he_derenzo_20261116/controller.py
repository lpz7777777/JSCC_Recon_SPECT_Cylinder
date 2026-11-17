"""Finite, exclusive controller for this one HE Derenzo request (12-hour limit)."""
import json
import os
from pathlib import Path
import sys
import time
import traceback
import msvcrt
import psutil
import workflow as w
from campaign_20261102_io import python_json


def remaining_submission():
    route = '5e10_440'
    if (w.REPORT/route/'collect_job.json').exists():
        return
    # Count individual array elements. No submit retry while the account cap binds.
    source = '''import subprocess,json
x=subprocess.run(['squeue','-r','-h','-u','maty','-o','%i'],capture_output=True,text=True,timeout=30)
if x.returncode:raise RuntimeError(x.stderr)
print('CAMPAIGN_JSON '+json.dumps(dict(active_submitted_elements=len(x.stdout.splitlines()))))'''
    count = python_json('maty','python3',source,timeout=60,readonly=True)
    w.save(w.REPORT/'remaining_submit_capacity.json',dict(**count,checked_utc=w.now(),account_limit=2000))
    if count['active_submitted_elements'] > 990:
        return
    prior = w.read(w.REPORT/route/'submission_limit_resolution.json')
    assert prior['passed']
    imaging = w.submit(route,'imaging',w.read(w.REPORT/'5e9_440/collect_job.json')['job'])
    w.submit(route,'collect',imaging['job'])
    w.submit_all()  # All already registered; publish the complete authority map only.


def main():
    w.REPORT.mkdir(parents=True,exist_ok=True)
    lock=(w.REPORT/'controller.lock').open('a+b')
    lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    current=psutil.Process()
    bindings={str(p):w.sha(p) for p in [Path(__file__),w.STUDY/'workflow.py']}
    frozen=w.read(w.REPORT/'freeze.json')
    deadline=time.time()+12*3600
    reg=dict(pid=os.getpid(),create_time=current.create_time(),cmdline=current.cmdline(),status='ACTIVE',
             started_utc=w.now(),deadline_epoch=deadline,bindings=bindings,release=frozen['release'],
             scope='four HE Derenzo acquisitions only; strict fetch, local 8-CPU MLEM200/save1, final numerical QA',
             recurring_automation=False,automatic_scientific_resubmission=False)
    w.save(w.REPORT/'controller_registration.json',reg)
    try:
        while time.time()<deadline:
            for p,h in bindings.items():
                if w.sha(p)!=h:raise RuntimeError('Bound controller source changed')
            w.verify(w.DATA/'payload',frozen['files'])
            remaining_submission()
            state=w.status()
            print('STATUS',w.now(),json.dumps({k:{n:v for n,v in x.items() if n!='accounting'} for k,x in state.items()}),flush=True)
            if any(x['errors'] for x in state.values()):raise RuntimeError('Confirmed scientific job failure; preserve and stop')
            for route, x in state.items():
                if (w.REPORT/route/'strict_delivery.json').exists():continue
                if x['fully_exited_successfully']:
                    w.fetch(route,'bridge')
                    w.reconstruct(route)
                    w.fetch(route,'transport')
                    print('STRICT_ROUTE_COMPLETE',w.finalize(route),flush=True)
            done=[r for r in w.ROUTES if (w.REPORT/r/'strict_delivery.json').exists()]
            if len(done)==4:
                w.save(w.REPORT/'controller_exit.json',dict(reg,status='COMPLETED',passed=True,ended_utc=w.now(),
                    completed_routes=done,PPT_replacement_complete=False,
                    next_step='Generate HE panels, review rendered PPT, replace the user deck after visual QA'))
                return
            time.sleep(45)
        w.save(w.REPORT/'controller_exit.json',dict(reg,status='DEADLINE_REACHED',passed=False,ended_utc=w.now()))
    except BaseException:
        w.save(w.REPORT/'controller_exit.json',dict(reg,status='FAILED',passed=False,ended_utc=w.now(),error=traceback.format_exc()))
        raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
