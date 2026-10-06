"""Existing legacy 5e9: matched calibration -> full10 validation -> unique 2000 pair."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import tarfile
import numpy as np
from energy_preflight_v5_workflow import (HERE,BASE,PYTHON,
    connect,command,write,digest,accounting_for,peak_rss)
from prepare_energy_5e9_v5 import STUDY,validate_transport
DATA=HERE/'generated'/STUDY
REPORT=HERE/'reports/NEMA_Body_H60'/STUDY
REMOTE=BASE+'/generated/'+STUDY
MODELS=('angular','continuous_energy')

def freeze():
    calibration=DATA/'calibration_results/calibration'
    selections=DATA/'calibration_results/selections'
    gate=json.loads((calibration/'calibration_gate.json').read_text())
    selection=json.loads((selections/'selection_gate.json').read_text())
    if gate['status']!='PASSED' or gate['failures'] or gate['event_policy']!='legacy':
        raise ValueError('Independent legacy calibration has not passed')
    for name,sha in gate['files'].items():
        if digest(calibration/name)!=sha:raise ValueError('Fetched calibration changed')
    if digest(selections/'selection_gate.json')!=gate['selection_gate_sha256']:
        raise ValueError('Calibration and selection release differ')
    inputs=json.loads((HERE/'generated/nema_h60_imaging_5e9_files.json').read_text())
    collection=HERE/'generated/collections/NEMA_Body_H60_5e9.json'
    validate_transport(json.loads(collection.read_text()))
    if digest(collection)!=inputs['collections/NEMA_Body_H60_5e9.json']['sha256']:
        raise ValueError('Original transport changed')
    payload=DATA/'formal_payload';payload.mkdir(exist_ok=False)
    # Reuse the accepted original code, never silently modify its response/core.
    old=HERE/'generated/compton_energy_probability_v5/formal_payload'
    accepted=json.loads((HERE/'reports/NEMA_Body_H60/compton_energy_probability_v5/formal_freeze.json').read_text())['sha256']
    needed=['run_energy_preflight_v5.py','run_reconstruction.py','torch_active_operator.py',
        'compton_event_response.py','compton_energy_probability_v5.py','compton_sparse_ops.py',
        'process_list_plane_sparse.py','detector_csv.py','validate_factors.py',
        'geometry.py','config.json','verify_first_scatter.py','whole_geometry.npz','transfer_training_summary.json']
    for name in needed:
        if digest(old/name)!=accepted[name]:raise ValueError('Accepted v5 helper changed: '+name)
        shutil.copy2(old/name,payload/name)
    # Other dependency modules in the accepted payload retain their exact bytes.
    for p in old.glob('*.py'):
        if p.name not in ('run_energy_formal_v5.py','verify_energy_formal_v5.py','energy_formal_v5_contract.py'):
            if p.name in accepted and not (payload/p.name).exists():
                if digest(p)!=accepted[p.name]:raise ValueError('Original dependency changed')
                shutil.copy2(p,payload/p.name)
    for name in ('run_energy_5e9_v5.py','verify_energy_5e9_v5.py','energy_5e9_v5_contract.py',
        'prepare_energy_5e9_v5.py','process_list_global_audit_v4.py','test_energy_5e9_v5.py',
        'test_energy_5e9_formal_v5.py','reconstruct_energy_5e9_v5.sh'):
        shutil.copy2(HERE/name,payload/name)
    shutil.copytree(calibration,payload/'calibration')
    shutil.copy2(selections/'selection_gate.json',payload/'selection_gate.json')
    shutil.copy2(collection,payload/'transport_collection.json')
    (payload/'selections').mkdir()
    nema=sorted((r for r in selection['records'] if r['dataset']=='NEMA'),key=lambda r:r['view'])
    for r in nema:
        source=selections/f"NEMA_legacy_v{r['view']:02d}_kept_rows.npy"
        if digest(source)!=r['selection_sha256']:raise ValueError('NEMA selection changed')
        shutil.copy2(source,payload/f"selections/{r['view']}.npy")
    for model in MODELS:shutil.copy2(calibration/(model+'_Sensi_full'),payload/(model+'_Sensi_full'))
    source_contract=json.loads((old/'contract.json').read_text())
    factors=json.loads((REPORT/'factor_identity.json').read_text())
    if not factors['passed']:raise ValueError('Cross-host complete matrix identity is required')
    shutil.copy2(REPORT/'factor_identity.json',payload/'factor_identity.json')
    files={p.relative_to(payload).as_posix():digest(p) for p in payload.rglob('*') if p.is_file()}
    cfg=dict(study=STUDY,files=files,event_policy='legacy',nodes=8,new_photons=0,new_training=False,
        input_sha256={n:i['sha256'] for n,i in inputs.items()},factor_manifest_sha256=source_contract['factor_manifest_sha256'],
        factor_payload_sha256={n:item['sha256'] for n,item in factors['files'].items()},
        whole_geometry_sha256=files['whole_geometry.npz'],events_per_view=[r['kept'] for r in nema],
        accepted_events=selection['nema_kept'],models=list(MODELS),
        channels=['440_ComptonOnly','440_SinglePlusCompton'],iterations=2000,save_step=50,
        calibration_release=json.loads((REPORT/'calibration_deployment.json').read_text())['release'],
        prior_1e9_formal_job=1666673,prior_1e9_policy='ideal',
        response_helpers_unchanged=True,mlem_core_unchanged=True,validation_required=True)
    write(payload/'contract.json',cfg);files['contract.json']=digest(payload/'contract.json')
    key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    from energy_5e9_v5_contract import load_contract
    load_contract(payload/'contract.json','validation',10,10)
    write(REPORT/'formal_freeze.json',dict(release_key=key,sha256=files,calibration_release=cfg['calibration_release'],
        formal_submitted=False,validation_required=True,accepted_events=cfg['accepted_events']))
    print('ENERGY_5E9_V5_FROZEN',key)

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
            ' -m unittest test_energy_5e9_v5 test_energy_5e9_formal_v5 -v 2>&1')
        if '\nOK' not in tests:raise ValueError('New formal entry tests failed: '+tests)
        (REPORT/'formal_tests.txt').write_bytes(tests.encode())
        command(c,'bash -n '+shlex.quote(release+'/reconstruct_energy_5e9_v5.sh'))
        print(command(c,'cd '+shlex.quote(release)+' && JSCC_PROJECT_ROOT='+shlex.quote(release)+' '+PYTHON+' -c '+
            shlex.quote("from pathlib import Path; from energy_5e9_v5_contract import load_contract; load_contract(Path('contract.json'),'validation',10,10); print('FORMAL_CONTRACT_PASSED')")))
    write(REPORT/'formal_deployment.json',dict(release=release,sha256=record['sha256'],
        tests_sha256=digest(REPORT/'formal_tests.txt'),calibration_release=record['calibration_release'],formal_submitted=False))
    print('ENERGY_5E9_V5_FORMAL_DEPLOYED',release)


def submit(mode):
    name='formal_validation_job.json' if mode=='validation' else 'formal_job.json'
    if (REPORT/name).exists():raise ValueError('Already registered; inspect before retry')
    deployed=json.loads((REPORT/'formal_deployment.json').read_text());release=deployed['release']
    exports='ALL,ENERGY_5E9_V5_FORMAL_RELEASE='+release+',ENERGY_5E9_V5_EXECUTION='+mode
    seconds=1500;minutes=90
    authority=None
    if mode=='formal':
        authority=json.loads((REPORT/'formal_authority.json').read_text())
        if authority['contract_sha256']!=deployed['sha256']['contract.json'] or not authority['passed']:
            raise ValueError('Current formal entry validation has not passed')
        exports+=',ENERGY_5E9_V5_AUTHORITY='+authority['remote']+',ENERGY_5E9_V5_AUTHORITY_SHA='+authority['sha256']
        estimate=authority['estimated_max_phase_seconds']
        seconds=max(1800,math.ceil(1.5*estimate+300))
        if seconds>21600:raise ValueError('Estimate exceeds bounded phase budget; diagnose throughput first')
        minutes=math.ceil((seconds*2+600)/60)
    exports+=',ENERGY_5E9_V5_PHASE_SECONDS='+str(seconds)
    with connect() as c:
        queue=command(c,"squeue -u scxi717 -h -o '%i %j %T'")
        if any('Energy_5e9_v5_' in x for x in queue.splitlines()):raise ValueError('Another v5 job is active')
        if len(queue.splitlines())>=50:
            write(REPORT/'formal_submission_wait.json',dict(reason='account_50_job_limit',mode=mode,other_jobs_untouched=True));return
        cmd=(f'sbatch --parsable -N 8 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=6 '+
            f'-p gpu_4090,gpu_5090 --qos=gpugpu --time={minutes} --chdir=/tmp --job-name=Energy_5e9_v5_{mode} '+
            '--exclude=wqd10nba06g6 --export='+shlex.quote(exports)+
            ' --output='+shlex.quote(REMOTE+'/logs/'+mode+'.%j.out')+
            ' --error='+shlex.quote(REMOTE+'/logs/'+mode+'.%j.err')+' '+shlex.quote(release+'/reconstruct_energy_5e9_v5.sh'))
        job=command(c,cmd).split(';')[0]
        if not job.isdigit():raise ValueError('Ambiguous submission; inspect queue before retry')
    write(REPORT/name,dict(job=int(job),mode=mode,nodes=8,gpus_per_node=1,release=release,
        contract_sha256=deployed['sha256']['contract.json'],authority=authority,
        iterations=10 if mode=='validation' else 2000,save_step=10 if mode=='validation' else 50,
        models=list(MODELS),nccl_interface='bond0',walltime_minutes=minutes,phase_timeout_seconds=seconds,
        submitted_utc=datetime.now(timezone.utc).isoformat(),formal_submitted=mode=='formal'))
    print('ENERGY_5E9_V5_'+mode.upper()+'_JOB',job)


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
            print(command(c,PYTHON+' '+shlex.quote(r['release']+'/verify_energy_5e9_v5.py')+
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
            if mode=='formal':
                for snapshot in v['checkpoints']:
                    relative=f"checkpoint_{snapshot['iteration']:06d}"
                    local=folder/relative;local.mkdir(exist_ok=True)
                    n='checkpoint_manifest.json'
                    with c.open_sftp() as s:s.get(remote+'/'+relative+'/'+n,str(local/n))
                    if digest(local/n)!=snapshot['manifest_sha256']:raise ValueError('Checkpoint manifest transfer differs')
                    metadata=json.loads((local/n).read_text())
                    for channel,kinds in metadata['outputs'].items():
                        for kind,sha in kinds.items():
                            n='Image_'+channel+'_'+kind+'.float32'
                            if (local/n).exists() and digest(local/n)==sha:continue
                            with c.open_sftp() as s:s.get(remote+'/'+relative+'/'+n,str(local/n))
                            if digest(local/n)!=sha:raise ValueError('Persistent checkpoint image transfer differs')
            if peak_rss(accounting)>.8*min(x['host_allocated_bytes'] for x in v['resources']):
                raise ValueError('Slurm MaxRSS resource margin fails')
            proofs[model]=dict(result=remote,allocation=allocation,sha256=sha_map,resources=v['resources'])
            evidence=REPORT/'formal_evidence'/job/model;evidence.mkdir(parents=True,exist_ok=True)
            for n in ('run_manifest.json','verification.json'):shutil.copy2(folder/n,evidence/n)
    write(REPORT/(mode+'_summary.json'),dict(passed=True,job=int(job),mode=mode,
        contract_sha256=r['contract_sha256'],models=proofs,accounting=accounting,
        paired_imaging_completed=mode=='formal',slurm_peak_rss_bytes=peak_rss(accounting)))
    if mode=='validation':
        estimate=max(x['prepare_seconds']+200*x['solve_seconds'] for p in proofs.values() for x in p['resources'])
        authority=dict(passed=True,contract_sha256=r['contract_sha256'],models=proofs,iterations=2000,save_step=50)
        path=DATA/'formal_authority.json';write(path,authority);sha=digest(path)
        remote=REMOTE+'/formal_authorities/'+sha+'.json'
        with connect() as c:
            command(c,'mkdir -p -- '+shlex.quote(REMOTE+'/formal_authorities'))
            with c.open_sftp() as s:s.put(str(path),remote)
            if command(c,'sha256sum -- '+shlex.quote(remote)).split()[0]!=sha:raise ValueError('Authority transfer differs')
        write(REPORT/'formal_authority.json',dict(passed=True,sha256=sha,remote=remote,
            contract_sha256=r['contract_sha256'],validation_job=int(job),estimated_max_phase_seconds=estimate))
    print('ENERGY_5E9_V5_'+mode.upper()+'_FETCH_VERIFIED',job)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('freeze','deploy','submit','status','fetch'))
    p.add_argument('--mode',choices=('validation','formal'),default='validation');a=p.parse_args()
    if a.action=='submit':submit(a.mode)
    else:globals()[a.action]()
