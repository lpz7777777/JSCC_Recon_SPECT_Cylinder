"""Idempotent submission; optionally wait for this account's submission slot."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'FOV120'))
from reconstruction_ssh import connect

REPORT=HERE/'reports/NEMA_Body_H60/response_mismatch_cut3_v1'
REMOTE='/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-seconds',type=int,default=0)
    args=parser.parse_args()
    target=REPORT/'job.json'
    if target.exists(): print(target.read_text()); return
    deployment=json.loads((REPORT/'deployment.json').read_text())
    release=deployment['release']
    deadline=time.monotonic()+args.wait_seconds
    attempt=0
    with connect() as client:
        while True:
            _,out,err=client.exec_command("squeue -u scxi717 -h -o '%i %j %T'",timeout=40)
            queued=out.read().decode(); code=out.channel.recv_exit_status()
            if code: raise RuntimeError(err.read().decode(errors='replace'))
            if any('NEMA_response_cut3' in line for line in queued.splitlines()):
                raise RuntimeError('Existing response-cut job: inspect before any new submission')
            attempt+=1
            if len(queued.splitlines())<50:
                cmd=('sbatch --parsable -N 8 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=6 '
                    '-p gpu_4090,gpu_5090 --qos=gpugpu --time=48:00:00 '
                    '--job-name=NEMA_response_cut3 --export='+shlex.quote('ALL,RESPONSE_RELEASE='+release)+' '+
                    shlex.quote(release+'/reconstruct_response_mismatch.sh'))
                _,out,err=client.exec_command(cmd,timeout=40)
                text=out.read().decode().strip(); error=err.read().decode(errors='replace')
                if out.channel.recv_exit_status()==0:
                    job=text.split(';')[0]
                    if not job.isdigit(): raise RuntimeError('Ambiguous submission outcome; inspect queue')
                    row={'study':'response_mismatch_cut3_v1','job_id':job,'job_name':'NEMA_response_cut3',
                        'submitted_utc':datetime.now(timezone.utc).isoformat(),'nodes':8,'gpus_per_node':1,
                        'release':release,'phases':['filter-off 50','cut3 full-data 10','cut3 formal 10000'],
                        'formal_output':REMOTE+'/generated/response_mismatch_cut3_v1/formal_'+job,
                        'nccl_interface':'bond0','attempts':attempt}
                    target.write_text(json.dumps(row,indent=2)+'\n'); print(json.dumps(row,indent=2),flush=True); return
                if 'AssocMaxSubmitJobLimit' not in error:
                    raise RuntimeError(error or text)
            row={'study':'response_mismatch_cut3_v1','status':'waiting_for_account_submission_slot',
                 'queued_jobs':len(queued.splitlines()),'attempts':attempt,
                 'checked_utc':datetime.now(timezone.utc).isoformat(),'release':release,
                 'other_project_jobs_untouched':True}
            (REPORT/'submission_wait.json').write_text(json.dumps(row,indent=2)+'\n')
            if time.monotonic()>=deadline: print(json.dumps(row,indent=2),flush=True); return
            if attempt==1 or attempt%30==0: print(json.dumps(row),flush=True)
            time.sleep(10)


if __name__=='__main__': main()
