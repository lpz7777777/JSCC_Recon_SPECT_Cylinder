"""Independent 5e9 full six-output continuous-energy 10000 workflow."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import tarfile
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from energy_preflight_v5_workflow import (HERE,BASE,PYTHON,
    connect,command,write,digest,accounting_for,peak_rss)
from prepare_energy_5e9_v5 import validate_transport
from energy_full10000_v5_contract import STUDY,CHANNELS
DATA=HERE/'generated'/STUDY
REPORT=HERE/'reports/NEMA_Body_H60'/STUDY
REMOTE=BASE+'/generated/'+STUDY
MODELS=('continuous_energy',)

def read_only_verification_command(client,text,timeout_seconds=600):
    """Drain both SSH streams under a total deadline for complete SHA verification."""
    _,out,err=client.exec_command(text,timeout=30)
    channel=out.channel;stdout=bytearray();stderr=bytearray()
    started=time.monotonic();next_notice=started+60
    while True:
        while channel.recv_ready():stdout.extend(channel.recv(65536))
        while channel.recv_stderr_ready():stderr.extend(channel.recv_stderr(65536))
        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
            break
        now=time.monotonic()
        if now-started>=timeout_seconds:
            raise TimeoutError('Read-only complete SHA verification exceeded total deadline')
        if now>=next_notice:
            print('FETCH_READ_ONLY_VERIFY_RUNNING_SECONDS',round(now-started),flush=True)
            next_notice=now+60
        time.sleep(.1)
    data=stdout.decode(errors='replace');error=stderr.decode(errors='replace')
    if channel.recv_exit_status():raise RuntimeError(error or data)
    return data.strip()

def freeze():
    DATA.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    old=HERE/'generated/compton_energy_probability_v5_5e9/formal_payload'
    previous=HERE/'reports/NEMA_Body_H60/compton_energy_probability_v5_5e9'
    accepted=json.loads((previous/'formal_freeze.json').read_text())
    delivered=json.loads((previous/'formal_summary.json').read_text())
    pilot=json.loads((previous/'validation_summary.json').read_text())
    if not delivered['passed'] or not delivered['paired_imaging_completed'] or not pilot['passed']:
        raise ValueError('Actual legacy v5 delivery and full-event pilot required')
    payload=DATA/'formal_payload';payload.mkdir(exist_ok=False)
    for name,sha in accepted['sha256'].items():
        source=old/name
        if digest(source)!=sha:raise ValueError('Delivered source bytes changed: '+name)
        target=payload/('source_v5_contract.json' if name=='contract.json' else name)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    for name in ('energy_full10000_v5_contract.py','single_checkpoint_mlem.py','run_energy_full10000_v5.py',
                 'verify_energy_full10000_v5.py','test_energy_full10000_v5.py','reconstruct_energy_full10000_v5.sh'):
        shutil.copy2(HERE/name,payload/name)
    old_cfg=json.loads((payload/'source_v5_contract.json').read_text())
    files={p.relative_to(payload).as_posix():digest(p) for p in payload.rglob('*') if p.is_file()}
    cfg=dict(old_cfg,study=STUDY,files=files,model='continuous_energy',models=list(MODELS),
        channels=list(CHANNELS),iterations=10000,save_step=50,source_v5_contract_sha256=files['source_v5_contract.json'],
        cross_prediction_source='440_SinglePhoton_final',composites_are_gamma_density_sums=True,
        source_delivered_job=1667869,validation_reference=pilot['models']['continuous_energy'],
        formal_prefix_reference=delivered['models']['continuous_energy'])
    write(payload/'contract.json',cfg);files['contract.json']=digest(payload/'contract.json')
    from energy_full10000_v5_contract import load_contract
    load_contract(payload/'contract.json','validation',10,10)
    key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    write(REPORT/'formal_freeze.json',dict(release_key=key,sha256=files,source_delivered_job=1667869,
        calibration_release=cfg['calibration_release'],
        source_v5_contract_sha256=cfg['source_v5_contract_sha256'],accepted_events=483743,
        iterations=10000,save_step=50,channels=list(CHANNELS),model='continuous_energy',formal_submitted=False))
    print('ENERGY_FULL10000_V5_FROZEN',key)


def deploy():
    record=json.loads((REPORT/'formal_freeze.json').read_text());payload=DATA/'formal_payload'
    for name,sha in record['sha256'].items():
        if digest(payload/name)!=sha:raise ValueError('Frozen formal payload changed')
    release=REMOTE+'/formal_releases/'+record['release_key'];archive=DATA/'formal_payload.tar.gz'
    with tarfile.open(archive,'w:gz') as f:
        for name in sorted(record['sha256']):f.add(payload/name,arcname=name)
    with connect() as c:
        command(c,'mkdir -p -- '+shlex.quote(release)+' '+shlex.quote(REMOTE+'/logs'))
        with c.open_sftp() as s:s.put(str(archive),release+'/payload.tar.gz')
        if command(c,'sha256sum -- '+shlex.quote(release+'/payload.tar.gz')).split()[0]!=digest(archive):
            raise ValueError('Formal archive transfer differs')
        command(c,'tar --no-same-owner -xzf '+shlex.quote(release+'/payload.tar.gz')+' -C '+shlex.quote(release))
        checks={release+'/'+n:s for n,s in record['sha256'].items()}
        cfg=json.loads((payload/'contract.json').read_text())
        checks.update({BASE+'/generated/'+n:s for n,s in cfg['input_sha256'].items()})
        checks.update({BASE+'/generated/FactorsCalibrated/'+n+'/factor_manifest.json':s for n,s in cfg['factor_manifest_sha256'].items()})
        checks.update({BASE+'/generated/FactorsCalibrated/'+n:s for n,s in cfg['factor_payload_sha256'].items()})
        script='''import hashlib,json
checks=json.loads('''+repr(json.dumps(checks))+''')
for p,sha in checks.items():
 h=hashlib.sha256();chunks=0
 with open(p,'rb') as stream:
  for block in iter(lambda:stream.read(8<<20),b''):
   h.update(block);chunks+=1
   if chunks%64==0:print('SHA_READ_PROGRESS',p,chunks,flush=True)
 if h.hexdigest()!=sha:raise ValueError(p)
 print('SHA_CHECKED',p,flush=True)
print('FORMAL_INPUT_SHA_VERIFIED',len(checks))
'''
        print(command(c,'timeout --signal=TERM --kill-after=15s 600s '+PYTHON+' -c '+shlex.quote(script)))
        tests=command(c,'cd '+shlex.quote(release)+' && JSCC_PROJECT_ROOT='+shlex.quote(release)+' '+PYTHON+
            ' -m unittest test_energy_full10000_v5 -v 2>&1')
        if '\nOK' not in tests or 'skipped' in tests:raise ValueError('New formal entry tests failed or incomplete: '+tests)
        (REPORT/'formal_tests.txt').write_bytes(tests.encode())
        command(c,'bash -n '+shlex.quote(release+'/reconstruct_energy_full10000_v5.sh'))
        print(command(c,'cd '+shlex.quote(release)+' && JSCC_PROJECT_ROOT='+shlex.quote(release)+' '+PYTHON+' -c '+
            shlex.quote("from pathlib import Path; from energy_full10000_v5_contract import load_contract; load_contract(Path('contract.json'),'validation',10,10); print('FORMAL_CONTRACT_PASSED')")))
    write(REPORT/'formal_deployment.json',dict(release=release,sha256=record['sha256'],
        tests_sha256=digest(REPORT/'formal_tests.txt'),calibration_release=record['calibration_release'],formal_submitted=False))
    print('ENERGY_FULL10000_V5_FORMAL_DEPLOYED',release)


def submit(mode):
    name='formal_validation_job.json' if mode=='validation' else 'formal_job.json'
    if (REPORT/name).exists():raise ValueError('Already registered; inspect before retry')
    deployed=json.loads((REPORT/'formal_deployment.json').read_text());release=deployed['release']
    exports='ALL,ENERGY_FULL_V5_RELEASE='+release+',ENERGY_FULL_V5_EXECUTION='+mode
    seconds=2700;minutes=120
    authority=None
    if mode=='formal':
        authority=json.loads((REPORT/'formal_authority.json').read_text())
        if authority['contract_sha256']!=deployed['sha256']['contract.json'] or not authority['passed']:
            raise ValueError('Current formal entry validation has not passed')
        exports+=',ENERGY_FULL_V5_AUTHORITY='+authority['remote']+',ENERGY_FULL_V5_AUTHORITY_SHA='+authority['sha256']
        estimate=authority['estimated_max_phase_seconds']
        seconds=max(1800,math.ceil(1.5*estimate+300))
        if seconds>86400:raise ValueError('Estimate exceeds bounded phase budget; diagnose throughput first')
        minutes=math.ceil((1.5*authority['estimated_total_seconds']+1200)/60)
        if minutes>2880:raise ValueError('Full task exceeds bounded 48-hour allocation; diagnose throughput')
    exports+=',ENERGY_FULL_V5_PHASE_SECONDS='+str(seconds)+',ENERGY_FULL_V5_TOTAL_SECONDS='+str(minutes*60-120)
    with connect() as c:
        queue=command(c,"squeue -u scxi717 -h -o '%i %j %T'")
        if any('Energy_full5e9_v5_' in x for x in queue.splitlines()):raise ValueError('Another v5 job is active')
        if len(queue.splitlines())>=50:
            write(REPORT/'formal_submission_wait.json',dict(reason='account_50_job_limit',mode=mode,other_jobs_untouched=True));return
        cmd=(f'sbatch --parsable -N 8 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=6 '+
            f'-p gpu_4090,gpu_5090 --qos=gpugpu --time={minutes} --chdir=/tmp --job-name=Energy_full5e9_v5_{mode} '+
            '--exclude=wqd10nba06g6 --export='+shlex.quote(exports)+
            ' --output='+shlex.quote(REMOTE+'/logs/'+mode+'.%j.out')+
            ' --error='+shlex.quote(REMOTE+'/logs/'+mode+'.%j.err')+' '+shlex.quote(release+'/reconstruct_energy_full10000_v5.sh'))
        job=command(c,cmd).split(';')[0]
        if not job.isdigit():raise ValueError('Ambiguous submission; inspect queue before retry')
    write(REPORT/name,dict(job=int(job),mode=mode,nodes=8,gpus_per_node=1,release=release,
        contract_sha256=deployed['sha256']['contract.json'],authority=authority,
        iterations=10 if mode=='validation' else 10000,save_step=10 if mode=='validation' else 50,
        models=list(MODELS),nccl_interface='bond0',walltime_minutes=minutes,phase_timeout_seconds=seconds,
        submitted_utc=datetime.now(timezone.utc).isoformat(),formal_submitted=mode=='formal'))
    print('ENERGY_FULL10000_V5_'+mode.upper()+'_JOB',job)


def latest():
    p=REPORT/('formal_job.json' if (REPORT/'formal_job.json').exists() else 'formal_validation_job.json')
    return json.loads(p.read_text())


def status():
    r=latest();job=r['job']
    with connect() as c:
        q=command(c,"squeue -u scxi717 -h -o '%i %j %T %M %D %R'")
        print('\n'.join(x for x in q.splitlines() if x.split()[0]==str(job)) or 'Not in active queue')
        print(accounting_for(c,job))
        for ext in ('out','err'):
            p=shlex.quote(REMOTE+'/logs/'+r['mode']+'.'+str(job)+'.'+ext)
            print(command(c,'if test -f '+p+'; then tail -n 12 '+p+'; fi'))
        result=REMOTE+'/'+r['mode']+'_continuous_energy_'+str(job)
        script="""import json
from pathlib import Path
p=Path("""+repr(result)+""")
if (p/'progress.json').exists():
 print('ACTUAL_PROGRESS',json.dumps(json.loads((p/'progress.json').read_text()),sort_keys=True))
 for phase in ('440_single','218_corrected','compton_jscc'):
  print('DURABLE_CHECKPOINTS',phase,len(list((p/('checkpoints_'+phase)).glob('checkpoint_*'))))
"""
        print(command(c,PYTHON+' -c '+shlex.quote(script)))
        if any(x.split()[0]==str(job) and ' RUNNING ' in x for x in q.splitlines()):
            print(command(c,f'sstat -j {job} --allsteps --noheader --parsable2 --format=JobID,MaxRSS,AveRSS'))


def fetch():
    r=latest();job=str(r['job']);mode=r['mode'];target=DATA/'formal_results'/job
    target.mkdir(parents=True,exist_ok=True);proofs={}
    with connect() as c:
        accounting=accounting_for(c,job)
        if not any(x.split('|')[:3]==[job,'COMPLETED','0:0'] for x in accounting.splitlines()):
            raise ValueError('Current job has not completely exited successfully')
        allocation=REMOTE+'/formal_allocation_'+job+'.txt'
        with c.open_sftp() as s:s.get(allocation,str(target/'allocation.txt'))
        allocation_sha=command(c,'sha256sum -- '+shlex.quote(allocation)).split()[0]
        if digest(target/'allocation.txt')!=allocation_sha:raise ValueError('Allocation transfer differs')
        for model in MODELS:
            remote=REMOTE+'/'+mode+'_'+model+'_'+job;folder=target/model;folder.mkdir(exist_ok=True)
            print(read_only_verification_command(c,PYTHON+' '+shlex.quote(r['release']+'/verify_energy_full10000_v5.py')+
                ' --result '+shlex.quote(remote)+' --contract '+shlex.quote(r['release']+'/contract.json')+
                ' --allocation '+shlex.quote(allocation)+' --mode '+mode+' --read-only'))
            with c.open_sftp() as s:
                for n in ('run_manifest.json','verification.json'):
                    sha=command(c,'sha256sum -- '+shlex.quote(remote+'/'+n)).split()[0]
                    s.get(remote+'/'+n,str(folder/n))
                    if digest(folder/n)!=sha:raise ValueError('Strict proof transfer differs')
            v=json.loads((folder/'verification.json').read_text())
            if (not v['passed'] or v['mode']!=mode or v['model']!=model or
                v['contract_sha256']!=r['contract_sha256'] or v['allocation_sha256']!=allocation_sha or
                v['run_manifest_sha256']!=digest(folder/'run_manifest.json')):
                raise ValueError('Actual current release/transport/resource proof differs')
            if mode=='formal' and v['authority_sha256']!=r['authority']['sha256']:
                raise ValueError('Actual formal authority differs')
            sha_map={'run_manifest.json':digest(folder/'run_manifest.json'),
                'verification.json':digest(folder/'verification.json'),'allocation.txt':allocation_sha}
            for item in v['outputs']:
                for kind,sha in item['sha256'].items():
                    n='Image_'+item['channel']+'_'+kind+'.float32'
                    with c.open_sftp() as s:s.get(remote+'/'+n,str(folder/n))
                    if digest(folder/n)!=sha:raise ValueError('Image transfer differs')
                    sha_map[n]=sha
                print('FETCH_CHANNEL_SHA_VERIFIED',item['channel'],flush=True)
            with c.open_sftp() as snapshots:
                for checkpoint_count,snapshot in enumerate(v['checkpoints'],1):
                    relative=f"checkpoints_{snapshot['phase']}/checkpoint_{snapshot['iteration']:06d}"
                    local=folder/relative;local.mkdir(parents=True,exist_ok=True)
                    n='checkpoint_manifest.json'
                    snapshots.get(remote+'/'+relative+'/'+n,str(local/n))
                    if digest(local/n)!=snapshot['manifest_sha256']:raise ValueError('Checkpoint manifest transfer differs')
                    metadata=json.loads((local/n).read_text())
                    for channel,kinds in metadata['outputs'].items():
                        for kind,sha in kinds.items():
                            n='Image_'+channel+'_'+kind+'.float32'
                            if (local/n).exists() and digest(local/n)==sha:continue
                            snapshots.get(remote+'/'+relative+'/'+n,str(local/n))
                            if digest(local/n)!=sha:raise ValueError('Persistent checkpoint image transfer differs')
                    if checkpoint_count%50==0:
                        print('FETCH_PERSISTENT_CHECKPOINTS_SHA_VERIFIED',checkpoint_count,flush=True)
            with c.open_sftp() as transfer:
                transfer.get(remote+'/PredictedCntStat_218_From440.float32',str(folder/'PredictedCntStat_218_From440.float32'))
            if digest(folder/'PredictedCntStat_218_From440.float32')!=v['prediction_sha256']:raise ValueError('Cross prediction transfer differs')
            sha_map['PredictedCntStat_218_From440.float32']=v['prediction_sha256']
            if peak_rss(accounting)>.8*min(x['host_allocated_bytes'] for x in v['resources']):
                raise ValueError('Slurm MaxRSS resource margin fails')
            proofs[model]=dict(result=remote,allocation=allocation,sha256=sha_map,resources=v['resources'])
            evidence=REPORT/'formal_evidence'/job/model;evidence.mkdir(parents=True,exist_ok=True)
            for n in ('run_manifest.json','verification.json'):shutil.copy2(folder/n,evidence/n)
    write(REPORT/(mode+'_summary.json'),dict(passed=True,job=int(job),mode=mode,
        contract_sha256=r['contract_sha256'],models=proofs,accounting=accounting,
        full_six_imaging_completed=mode=='formal',slurm_peak_rss_bytes=peak_rss(accounting)))
    if mode=='validation':
        rr=proofs['continuous_energy']['resources']
        phases={name:max(x['phase_solve_seconds'][name] for x in rr)*1000 for name in ('440_single','218_corrected','compton_jscc')}
        phases['compton_jscc']+=max(x['prepare_seconds'] for x in rr)
        estimate=max(phases.values());total_estimate=sum(phases.values())
        authority=dict(passed=True,contract_sha256=r['contract_sha256'],models=proofs,iterations=10000,save_step=50)
        path=DATA/'formal_authority.json';write(path,authority);sha=digest(path)
        remote=REMOTE+'/formal_authorities/'+sha+'.json'
        with connect() as c:
            command(c,'mkdir -p -- '+shlex.quote(REMOTE+'/formal_authorities'))
            with c.open_sftp() as s:s.put(str(path),remote)
            if command(c,'sha256sum -- '+shlex.quote(remote)).split()[0]!=sha:raise ValueError('Authority transfer differs')
        write(REPORT/'formal_authority.json',dict(passed=True,sha256=sha,remote=remote,
            contract_sha256=r['contract_sha256'],validation_job=int(job),estimated_max_phase_seconds=estimate,estimated_total_seconds=total_estimate,estimated_phase_seconds=phases))
    print('ENERGY_FULL10000_V5_'+mode.upper()+'_FETCH_VERIFIED',job)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('freeze','deploy','submit','status','fetch'))
    p.add_argument('--mode',choices=('validation','formal'),default='validation');a=p.parse_args()
    if a.action=='submit':submit(a.mode)
    else:globals()[a.action]()
