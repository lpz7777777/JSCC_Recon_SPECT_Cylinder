"""Explicit r5 continuation: one formal allocation, no scheduler or cancellation."""
import argparse,hashlib,json,re,tarfile
from pathlib import Path
import workflow as base
from workflow import DATA,REPORT,REMOTE,PY,read,write,sha,now,connect,command,script,q,capacity,immutable

FREEZE=REPORT/'freeze_resume_r5.json'
ATTEMPT='resume_r5'

def package():
    f=read(FREEZE);payload=DATA/f['payload_dir'];old=read(DATA/'execution_manifest_r4.json')
    for n,s in f['files'].items():assert sha(payload/n)==s,n
    extra=DATA/'launcher_resume_r5';extra.mkdir(exist_ok=True)
    for name in ('launch.sh','node_launcher.py'):
        source=DATA/'launcher_r4'/name;assert sha(source)==old['files'][name]
        data=source.read_text(encoding='utf8')
        if name=='node_launcher.py':
            assert data.count("root/'run_study.py'")==1
            data=data.replace("root/'run_study.py'","root/'run_resume.py'")
        p=extra/name;raw=data.encode()
        if p.exists():assert p.read_bytes()==raw
        else:p.write_bytes(raw)
    files=dict(f['files']);files.update({n:sha(extra/n) for n in ('launch.sh','node_launcher.py')})
    key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    m=dict(key=key,files=files,source_freeze=f['key'],source_freeze_sha256=sha(FREEZE),
           parent_execution='cae5636aa17bca3c',parent_job='1813383',resume_iteration=9950)
    manifest=DATA/'execution_manifest_resume_r5.json';immutable(manifest,m)
    archive=DATA/('release_'+key+'.tar.gz')
    if not archive.exists():
        with tarfile.open(archive,'w:gz') as t:
            for n in sorted(files):t.add(payload/n if n in f['files'] else extra/n,arcname=n,recursive=False)
            t.add(manifest,arcname='release_manifest.json',recursive=False)
    return m,archive

base.ATTEMPT=ATTEMPT;base.FREEZE=FREEZE;base.package=package

def deploy():
    deployed=base.deploy()
    evidence=REPORT/'linux_resume_tests_r5.txt'
    if not evidence.exists():
        with connect() as c:
            result=command(c,'cd '+q(deployed['release'])+' && '+q(PY)+' -B test_resume.py',120)
        if 'WIDTH_RESUME_PASS' not in result:raise ValueError('Linux resume regression failed')
        evidence.write_text(result,encoding='utf8')
    return deployed

def submit():
    registration=REPORT/'formal_registration_resume_r5.json';intent=REPORT/'submission_intent_resume_r5.json'
    if registration.exists():print(json.dumps(read(registration)));return
    if intent.exists():raise RuntimeError('Unresolved submission intent; reconcile Slurm, never resubmit blindly')
    deployed=deploy();plan=read(REPORT/'resume_plan_r5.json')
    with connect() as c:
        state=command(c,'sacct -n -P -j '+plan['parent_job']+' --format=JobID,State,ExitCode',45)
        rows=[line.split('|') for line in state.splitlines() if line.strip()]
        assert {plan['parent_job'],plan['parent_job']+'.batch',plan['parent_job']+'.extern',plan['parent_job']+'.0'}<={r[0] for r in rows}
        assert all(r[1] in ('COMPLETED','FAILED','CANCELLED') for r in rows),state
        check='import sys,json;from pathlib import Path;sys.path.insert(0,'+repr(deployed['release'])+');from resume_support import verify_parent;verify_parent(json.loads((Path('+repr(deployed['release'])+')/"resume_plan.json").read_text()));print("PARENT_199_CHECKPOINTS_SHA_PASS")'
        proof=script(c,check,120);assert 'PARENT_199_CHECKPOINTS_SHA_PASS' in proof
        cap=capacity(c)
        queue=command(c,"squeue -u scxi717 -h -o '%i|%j|%T|%D|%b'",45)
        if 'NEMA5e9_width' in queue:raise ValueError('Existing width job, no parallel continuation')
        partition=command(c,'scontrol show partition gpu_5090',30);assert 'State=UP' in partition
        args=['sbatch','--parsable','-p','gpu_5090','--qos=gpugpu','-N','8','-n','8','--ntasks-per-node=1','--cpus-per-task=8','--gres=gpu:1','--time=4320',
              '--job-name=NEMA5e9_width_r5','--chdir='+REMOTE,'--output='+REMOTE+'/logs/%j.out','--error='+REMOTE+'/logs/%j.err',deployed['release']+'/launch.sh',deployed['release']]
        test=command(c,' '.join(q(x) for x in args[:1]+['--test-only']+args[1:]),45)
        write(REPORT/'submission_preflight_resume_r5.json',dict(time=now(),parent_accounting=state,parent_verification=proof,
            existing_queue=queue,partition=partition,capacity=cap,test_only=test,command=args,
            linux_resume_tests_sha256=sha(REPORT/'linux_resume_tests_r5.txt')))
        write(intent,dict(time=now(),execution_key=deployed['execution_key'],command=args,auto_retry=False))
        returned=command(c,' '.join(q(x) for x in args),45).strip()
        if not re.fullmatch(r'\d+(?:;\S+)?',returned):raise RuntimeError('Ambiguous sbatch output: '+returned)
        job=returned.split(';')[0]
        reg=dict(job=job,time=now(),release=deployed['release'],execution_key=deployed['execution_key'],result=REMOTE+'/formal_'+job,
            nodes=8,gpus_per_node=1,total_gpus=8,scales=[100,80,60],iterations=10000,save_step=50,
            parent_job=plan['parent_job'],inherited_iterations=9950,new_iterations={'100':50,'80':10000,'60':10000},
            historical_discrepancy_explicitly_accepted=True,original_comparison_threshold_unchanged=1e-5)
        write(registration,reg)
        actual=command(c,'scontrol show job -dd '+job,30)
        (REPORT/'submitted_slurm_resume_r5.txt').write_text(actual,encoding='utf8')
        assert 'NumNodes=8' in actual and 'gres/gpu=8' in actual,actual
        print(json.dumps(reg,indent=2));print(actual)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['deploy','submit','status']);a=p.parse_args()
    if a.action=='status':base.status()
    else:globals()[a.action]()

if __name__=='__main__':main()
