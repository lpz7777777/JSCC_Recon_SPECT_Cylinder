"""Immutable independent matrix-Poisson generation, validation, formal200 and strict fetch."""
import argparse, contextlib, hashlib, json, math, os, shutil, sys, time
from pathlib import Path
from ehe_common import HERE, GPU_BASE, GPU_PYTHON, RESPONSES, digest, read, write, hashes, verify_files
from ehe_forward_poisson_data import STUDY, verify_counts, dose_budget
from ehe_5e9_workflow import connection, command, q, put_tree, env_gpu
from ehe_slurm_status import stage_completed

DATA=HERE/'generated'/STUDY
REPORT=HERE/'reports/NEMA_Body_H60'/STUDY
BASE=GPU_BASE.rsplit('/',1)[0]+'/'+STUDY
OLD_DATA=HERE/'generated/ehe_spect_5e9_200'
OLD_REPORT=HERE/'reports/NEMA_Body_H60/ehe_spect_5e9_200'


def freeze():
    path=REPORT/'freeze.json'
    if path.exists():
        f=read(path);verify_files(DATA/f['payload_dir'],f['sha256']);return f
    from ehe_conversion_workflow import response_root
    old=read(OLD_REPORT/'reconstruction_execution_freeze.json');old_payload=OLD_DATA/old['payload_dir']
    cfg=read(old_payload/'config.json')
    config={k:cfg[k] for k in ('gamma_yields','truth_sha256','geometry_sha256','params_sha256')}
    config.update(study=STUDY,expected_emitted_photons=5_000_000_000,views=20,bins=2312,iterations=200,save_step=10,
        noise_seeds={'A218':32100101,'A440':32100102,'C440to218':32100103},bit_generator='PCG64',
        data_kind='matrix_forward_plus_independent_Poisson',relative_activity_integral_mm3=read(HERE/'reports/NEMA_Body_H60/manifest.json')['relative_activity_integral_mm3'],
        factor_sha256={n:digest(OLD_DATA/'factor_evidence'/n/'factor_manifest.json') for n in RESPONSES},
        response_root=response_root(),physical_calibration_claim=False,transport_performed=False,
        source_basis='Original 3mm truth voxel mass -> Cartesian bilinear stencil, clockwise rotation',
        reconstruction_basis='Original full Polar volume-weighted density operator',
        background_source='This new experiment final440 single image, fixed additive Poisson term',
        initial_density=1,regularization=None,source_integral_normalization='Expected emitted dose only; no detected-count matching',
        human_instruction='Use system matrix forward projection plus noise for a new 5e9 EHE reconstruction')
    payload=DATA/'payload';payload.mkdir(parents=True,exist_ok=False)
    for name in ('ehe_forward_poisson_data.py','run_ehe_forward_poisson.py','verify_ehe_forward_poisson.py','ehe_slurm_status.py'):
        shutil.copy2(HERE/name,payload/name)
    for name in ('ehe_common.py','torch_active_operator.py','single_checkpoint_mlem.py','whole_geometry.npz','truth_3mm.npz'):
        if digest(old_payload/name)!=old['sha256'][name]:raise ValueError('Original execution helper/source identity differs')
        shutil.copy2(old_payload/name,payload/name)
    scientific=read(OLD_REPORT/'response_repair_freeze.json');source=OLD_DATA/scientific['payload_dir']/'ehe_gpu_pipeline.py'
    if digest(source)!=scientific['sha256']['ehe_gpu_pipeline.py']:raise ValueError('Original source stencil implementation differs')
    shutil.copy2(source,payload/'ehe_gpu_pipeline.py')
    write(payload/'config.json',config)
    import numpy as np
    budget=dose_budget(np.load(payload/'truth_3mm.npz'),config)
    if not math.isclose(budget['expected_primary_photons']['218']/5e9,.29380779868182727,rel_tol=1e-13):raise ValueError('True emitted energy allocation differs')
    files=hashes(payload);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    f=dict(study=STUDY,release_key=key,payload_dir='payload',root=BASE+'/releases/'+key,sha256=files,
           response_root=config['response_root'],budget=budget,original_response_release_key=scientific['release_key'],
           original_reconstruction_release_key=old['release_key'],preregistered_noise_seeds=config['noise_seeds'])
    write(payload/'release_manifest.json',f);write(path,f)
    return f


def accounting(c,job):
    return command(c,'sacct -j '+q(job)+' --parsable2 --noheader --format=JobID,State,ExitCode,MaxRSS,Elapsed,AllocTRES,NodeList')


def registered(stage):
    path=REPORT/(stage+'_job.json');return read(path) if path.exists() else None


def submit(c,stage,lines,minutes):
    previous=registered(stage)
    if previous:return previous
    jobs=command(c,'squeue -u scxi717 -h -o "%i"').splitlines()
    if len(jobs)>=50:return None
    intent_path=REPORT/(stage+'_submission_intent.json')
    if intent_path.exists():raise RuntimeError('Unresolved submission intent retained; inspect queue/accounting before any repeat')
    command(c,'mkdir -p '+q(BASE+'/scripts')+' '+q(BASE+'/logs'))
    script=DATA/'scripts'/(stage+'.sh');script.parent.mkdir(parents=True,exist_ok=True)
    script.write_bytes(('#!/bin/bash\nset -euo pipefail\n'+env_gpu()+lines+'\n').encode())
    remote=BASE+'/scripts/'+script.name
    with c.open_sftp() as s:s.put(str(script),remote)
    # One GPU, no explicit memory request; actual AllocTRES is checked after exit.
    cmd='sbatch --parsable --partition=gpu_5090 --account=scxi717 --qos=gpugpu -N 1 -n 1 -c 6 --gres=gpu:1 --exclude=wqd10nba06g6 --time='+str(minutes)
    cmd+=' --job-name='+q('ehe_fp_'+stage)+' --output='+q(BASE+'/logs/'+stage+'_%j.log')+' '+q(remote)
    write(intent_path,dict(stage=stage,script_sha256=digest(script),created_epoch=time.time(),status='submitting'))
    job=int(command(c,cmd).strip().split(';')[0])
    value=dict(job=job,stage=stage,script=remote,script_sha256=digest(script),output=BASE+('/validation' if stage=='generate_validation' else '/'+stage),
               submitted_epoch=time.time(),slurm_minutes=minutes,release_key=read(REPORT/'freeze.json')['release_key'])
    write(REPORT/(stage+'_job.json'),value);return value


def deploy(c,f):
    payload=DATA/f['payload_dir'];verify_files(payload,f['sha256'])
    command(c,'mkdir -p '+q(BASE+'/releases'))
    with c.open_sftp() as s:put_tree(s,payload,f['root'])
    code='from pathlib import Path;from ehe_common import *;r=Path('+repr(f['root'])+');verify_files(r,read(r/"release_manifest.json")["sha256"]);print("NEW_RELEASE_SHA_PASS")'
    print(command(c,'cd '+q(f['root'])+' && '+q(GPU_PYTHON)+' -c '+q(code)))
    write(REPORT/'deployment.json',dict(passed=True,release_key=f['release_key'],release_manifest_sha256=digest(payload/'release_manifest.json')))


def verification_release(c,f):
    path=REPORT/'verification_freeze.json'
    if path.exists():return read(path)
    source=DATA/f['payload_dir'];folder=DATA/'verification_payload';folder.mkdir(exist_ok=False)
    for name in ('ehe_common.py','ehe_gpu_pipeline.py','torch_active_operator.py','ehe_forward_poisson_data.py','verify_ehe_forward_poisson.py','ehe_slurm_status.py'):
        shutil.copy2(source/name,folder/name)
    files=hashes(folder);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    v=dict(key=key,files=files,root=BASE+'/verification_releases/'+key,payload_dir='verification_payload')
    write(folder/'verification_manifest.json',v)
    command(c,'mkdir -p '+q(BASE+'/verification_releases'))
    with c.open_sftp() as s:put_tree(s,folder,v['root'])
    write(path,v);return v


def fetch_file(s,remote,local,expected=None):
    local=Path(local);local.parent.mkdir(parents=True,exist_ok=True)
    if local.exists() and expected and digest(local)==expected:return
    partial=local.with_name(local.name+'.fetching');s.get(remote,str(partial))
    if expected and digest(partial)!=expected:raise ValueError('Strict fetched SHA differs: '+str(local))
    os.replace(partial,local)


def fetch_authority(c,mode,f):
    path=REPORT/(mode+'_summary.json')
    if path.exists():return read(path)
    v=read(REPORT/'verification_freeze.json');job=registered(mode+'_acceptance')['job'];text=accounting(c,job)
    if not stage_completed(text,job):return None
    dest=BASE+'/'+mode+'_acceptance';target=DATA/(mode+'_acceptance')
    with c.open_sftp() as s:
        fetch_file(s,dest+'/authority.json',target/'authority.json')
        proof=read(target/'authority.json')
        fetch_file(s,dest+'/evidence/receipt.json',target/'receipt.json')
        receipt=read(target/'receipt.json')
        if receipt['authority_sha256']!=digest(target/'authority.json') or not receipt['passed']:raise ValueError('Verifier receipt SHA differs')
        if proof['verification_manifest_sha256']!=digest(DATA/v['payload_dir']/'verification_manifest.json'):raise ValueError('Actual immutable verifier differs')
        if proof['release_key']!=f['release_key'] or proof['mode']!=mode or not proof['passed']:raise ValueError('Acceptance authority identity differs')
        result=DATA/'results'/mode
        for name,sha in proof['files'].items():fetch_file(s,BASE+'/'+mode+'/'+name,result/name,sha)
        for stage in ('generate_validation',mode+'_acceptance')+(() if mode=='validation' else ('formal',)):
            j=registered(stage)['job'];fetch_file(s,BASE+'/logs/'+stage+'_'+str(j)+'.log',DATA/'logs'/(stage+'_'+str(j)+'.log'))
    verify_files(DATA/'results'/mode,proof['files'])
    write(REPORT/(mode+'_acceptance_receipt.json'),receipt)
    write(REPORT/(mode+'_acceptance_sacct.json'),dict(job=job,accounting=text,passed=True))
    write(path,proof);return proof


def advance():
    f=freeze()
    with connection('gpu') as c:
        if not (REPORT/'deployment.json').exists():deploy(c,f)
        r=f['root'];resp=f['response_root'];counts=BASE+'/counts'
        generation=registered('generate_validation')
        if generation is None:
            line='cd '+q(r)+'\ntimeout --signal=TERM --kill-after=10s 2400s '+q(GPU_PYTHON)+' -u ehe_forward_poisson_data.py --release '+q(r)+' --responses '+q(resp)+' --output '+q(counts)
            line+='\ntimeout --signal=TERM --kill-after=10s 2400s '+q(GPU_PYTHON)+' -u run_ehe_forward_poisson.py --release '+q(r)+' --responses '+q(resp)+' --counts '+q(counts)+' --output '+q(BASE+'/validation')+' --mode validation --iterations 10 --limit 1800'
            print(submit(c,'generate_validation',line,90));return
        text=accounting(c,generation['job'])
        if not stage_completed(text,generation['job']):print(text);return
        if not (REPORT/'generation_summary.json').exists():
            with c.open_sftp() as s:
                fetch_file(s,counts+'/collection.json',DATA/'counts/collection.json');collection=read(DATA/'counts/collection.json')
                for name,sha in collection['files'].items():fetch_file(s,counts+'/'+name,DATA/'counts'/name,sha)
            verify_counts(DATA/'counts',DATA/f['payload_dir']);write(REPORT/'generation_summary.json',collection)
        for mode in ('validation','formal'):
            if mode=='formal' and registered('formal') is None:
                authority=read(REPORT/'validation_summary.json');limit=math.ceil(max(authority['phase_seconds'].values())*20*1.8+300)
                line='cd '+q(r)+'\ntimeout --signal=TERM --kill-after=10s '+str(2*limit+900)+'s '+q(GPU_PYTHON)+' -u run_ehe_forward_poisson.py --release '+q(r)+' --responses '+q(resp)+' --counts '+q(counts)+' --output '+q(BASE+'/formal')+' --mode formal --iterations 200 --limit '+str(limit)+' --authority '+q(BASE+'/validation_acceptance/authority.json')
                print(submit(c,'formal',line,math.ceil((2*limit+900)/60)+5));return
            imaging=generation if mode=='validation' else registered('formal');text=accounting(c,imaging['job'])
            if not stage_completed(text,imaging['job']):print(text);return
            if registered(mode+'_acceptance') is None:
                v=verification_release(c,f);folder=DATA/'accounting';folder.mkdir(exist_ok=True)
                (folder/(mode+'.txt')).write_bytes(text.encode());(folder/'generation.txt').write_bytes(accounting(c,generation['job']).encode())
                command(c,'mkdir -p '+q(BASE+'/accounting'))
                with c.open_sftp() as s:put_tree(s,folder,BASE+'/accounting')
                dest=BASE+'/'+mode+'_acceptance'
                line='cd '+q(v['root'])+'\nmkdir -p '+q(dest)+'\ntimeout --signal=TERM --kill-after=10s 2700s '+q(GPU_PYTHON)+' -u verify_ehe_forward_poisson.py --result '+q(BASE+'/'+mode)+' --release '+q(r)+' --responses '+q(resp)+' --counts '+q(counts)+' --accounting '+q(BASE+'/accounting/'+mode+'.txt')+' --generation-accounting '+q(BASE+'/accounting/generation.txt')+' --output '+q(dest+'/authority.json')+' --verification-manifest '+q(v['root']+'/verification_manifest.json')+' --evidence '+q(dest+'/evidence')
                print(submit(c,mode+'_acceptance',line,50));return
            proof=fetch_authority(c,mode,f)
            if proof is None:print(accounting(c,registered(mode+'_acceptance')['job']));return
        print('FORMAL200_STRICTLY_FETCHED')


@contextlib.contextmanager
def controller():
    DATA.mkdir(parents=True,exist_ok=True);path=DATA/'controller.json'
    if path.exists():
        previous=read(path)
        if previous['status']=='running':
            try:os.kill(previous['pid'],0)
            except OSError:pass
            else:raise RuntimeError('Registered local controller still alive; no concurrent advance/fetch/submit')
    value=dict(pid=os.getpid(),status='running',started_epoch=time.time());write(path,value)
    try:yield
    except BaseException:
        value.update(status='failed',finished_epoch=time.time());write(path,value);raise
    else:value.update(status='complete',exit_code=0,finished_epoch=time.time());write(path,value)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['advance','status']);a=p.parse_args()
    if a.action=='advance':
        with controller():advance()
    else:
        with connection('gpu') as c:
            for stage in ('generate_validation','validation_acceptance','formal','formal_acceptance'):
                value=registered(stage)
                if value:print(stage,accounting(c,value['job']))
