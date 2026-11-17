"""Finite completion of ONE registered job: read status, strict fetch, then plot.

Cannot submit, cancel, resubmit or modify any HPC job. No scheduled automation.
Stops after success/failure or 96 hours. Bound source files must not be edited
while this process is alive. Network errors wait within the fixed deadline.
"""
import argparse,datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path
import psutil
from prepare import CODE,DATA,REPORT,sha,read,write
from workflow import connect,command,script,now

FREEZE=REPORT/'completion_pipeline_r5.json'
REG=DATA/'completion_once_r5_registration.json'
LOCK=DATA/'completion_once_r5.lock'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}

def save(path,obj):
    temp=path.with_name(path.name+'.partial')
    with temp.open('w',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def bound():
    f=read(FREEZE)
    for name,digest in f['source_and_input_sha256'].items():
        if sha(Path(name))!=digest:raise ValueError('Bound completion source/input changed: '+name)
    return f

def classify(text,job):
    rows=[line.split('|') for line in text.splitlines() if line.strip()]
    required={job,job+'.batch',job+'.extern',job+'.0'}
    if required<={r[0] for r in rows} and all(r[1]=='COMPLETED' and r[2]=='0:0' for r in rows):return 'SUCCESS'
    root=next((r for r in rows if r[0]==job),None)
    if root and root[1].split()[0] in TERMINAL:
        if root[1]!='COMPLETED' or root[2]!='0:0':return 'FAILED'
        if any(r[1].split()[0] in TERMINAL and (r[1]!='COMPLETED' or r[2]!='0:0') for r in rows):return 'FAILED'
    return 'WAIT'

def snapshot(reg):
    job=reg['job']
    with connect() as c:
        account=command(c,'sacct -n -P -j '+job+' --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS',45)
        code='''from pathlib import Path
import json
p=Path(RESULT);data={}
for n in ('started.json','progress.json','completed.json','width100/completed.json','width80/completed.json','width60/completed.json'):
 f=p/n
 if f.exists():
  d=json.loads(f.read_text());d.pop('files',None);data[n]=d
print(json.dumps(data))
'''.replace('RESULT',repr(reg['result']))
        details=json.loads(script(c,code,45))
    return dict(time=now(),job=job,accounting=account,decision=classify(account,job),details=details)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');a=parser.parse_args()
    if a.self_test:
        job='123';good='\n'.join(n+'|COMPLETED|0:0' for n in (job,job+'.batch',job+'.extern',job+'.0'))
        assert classify(good,job)=='SUCCESS' and classify(good.replace('123|COMPLETED','123|FAILED'),job)=='FAILED'
        assert classify('123|RUNNING|0:0',job)=='WAIT' and classify('123|COMPLETED|0:0',job)=='WAIT'
        assert classify(good.replace('123.0|COMPLETED|0:0','123.0|FAILED|1:0'),job)=='FAILED'
        print('FINITE_COMPLETION_STATE_TEST_PASS');return
    f=bound();registered=read(REPORT/'formal_registration_resume_r5.json');assert f['job']==registered['job']
    if REG.exists():raise RuntimeError('Completion process already registered; inspect identity before any restart')
    lock=os.open(str(LOCK),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(lock,str(os.getpid()).encode());os.close(lock)
    process=psutil.Process();start=time.time();deadline=start+96*3600
    status=dict(pid=os.getpid(),create_time=process.create_time(),cmdline=process.cmdline(),job=registered['job'],
        started=now(),deadline_utc=datetime.datetime.fromtimestamp(deadline,datetime.timezone.utc).isoformat(),
        source_manifest_sha256=sha(FREEZE),status='ACTIVE',can_submit=False,can_cancel=False,
        only_job=registered['job'],automatically_resubmit=False,periodic_chat_automation=False)
    save(REG,status)
    try:
        errors=0
        while time.time()<deadline:
            bound()
            try:
                snap=snapshot(registered);save(REPORT/'completion_latest_status_r5.json',snap);errors=0
            except Exception as e:
                errors+=1;save(REPORT/'completion_network_issue_r5.json',dict(time=now(),consecutive_errors=errors,error=repr(e)))
                if errors>=60:raise RuntimeError('60 consecutive read-only status failures') from e
                time.sleep(min(300,max(0,deadline-time.time())));continue
            status.update(last_check=snap['time'],last_decision=snap['decision'],progress=snap['details'].get('progress.json'));save(REG,status)
            print(snap['time'],registered['job'],snap['decision'],status['progress'],flush=True)
            if snap['decision']=='FAILED':raise RuntimeError('Formal job failed; outputs preserved; no auto resubmission: '+snap['accounting'])
            if snap['decision']=='SUCCESS':break
            time.sleep(min(300,max(0,deadline-time.time())))
        else:raise TimeoutError('Finite 96 hour completion deadline; remote job untouched')
        for name,stage in [('resume_delivery.py','STRICT_FETCH'),('compare_results.py','GENERATE_COMPARISON')]:
            bound();status.update(status=stage);save(REG,status)
            subprocess.run([sys.executable,'-B','-X','utf8',str(CODE/name)],cwd=CODE.parents[3],check=True,timeout=3*3600)
        status.update(status='COMPARISON_GENERATED_VISUAL_QA_PENDING',completed=now(),
            report=str(CODE.parents[3]/'Analysis_Plots/research/nema5e9_width_sweep_20261115/index.html'))
        save(REG,status)
    except Exception as e:
        status.update(status='STOPPED_FAILURE',stopped=now(),error=repr(e),traceback=traceback.format_exc());save(REG,status);raise
    finally:
        if LOCK.read_text()==str(os.getpid()):LOCK.unlink()

if __name__=='__main__':main()
