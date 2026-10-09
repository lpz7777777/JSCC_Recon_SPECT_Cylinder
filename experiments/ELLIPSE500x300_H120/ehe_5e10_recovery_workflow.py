"""Bounded srun WaitTime repair; original scientific execution bytes stay frozen."""
import hashlib, json, shutil, time
from pathlib import Path
from ehe_common import read, write, digest, hashes, verify_files
from ehe_5e9_workflow import command, q, put_tree, env_cpu


def prepare(c):
    from ehe_5e10_workflow import DATA, REPORT, CPU_BASE, BASE, HERE, cpu_freeze
    path=REPORT/'transport_recovery_freeze.json'
    if path.exists():
        f=read(path);verify_files(DATA/f['payload_dir'],f['sha256']);return f
    stop=read(REPORT/'transport_stop_acceptance.json')
    if not stop['passed'] or stop['original_state']!='FAILED' or len(stop['completed'])!=13:
        raise ValueError('Actual stopped source and thirteen verified completed workers required')
    if command(c,'squeue -h -j '+str(stop['original_job'])+' 2>/dev/null || true').strip():
        raise ValueError('Original allocation is still active')
    original=cpu_freeze();payload=DATA/'transport_recovery_payload';payload.mkdir(exist_ok=False)
    config=read(DATA/original['payload_dir']/'config.json')
    recovery=dict(original_failed_job=stop['original_job'],original_state='FAILED',
        original_failure_accounting=stop['original_accounting'],
        original_transport_root=CPU_BASE+'/transport',original_worker_release=original['root'],
        original_worker_source_sha256=original['sha256']['ehe_5e10_transport.py'],
        reused_workers=stop['completed'],reused_count=13,missing_count=987,
        stop_acceptance_sha256=digest(REPORT/'transport_stop_acceptance.json'),
        launcher_change='Explicit srun --wait=0; same 1000-task/18-node allocation, 240-minute cap')
    config['transport_recovery']=recovery
    for name in ('ehe_common.py','ehe_slurm_status.py'):
        shutil.copy2(DATA/original['payload_dir']/name,payload/name)
    shutil.copy2(HERE/'ehe_5e10_recovery.py',payload/'ehe_5e10_recovery.py')
    shutil.copy2(HERE/'ehe_5e10_recovered_transport.py',payload/'ehe_5e10_transport.py')
    shutil.copy2(REPORT/'transport_stop_acceptance.json',payload/'source_stop_acceptance.json')
    write(payload/'config.json',config)
    files=hashes(payload);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    f=dict(release_key=key,payload_dir=payload.name,sha256=files,
        root=CPU_BASE+'/releases/'+key,transport_root=CPU_BASE+'/transport_recovery_'+key,
        phase_limit_seconds=original['phase_limit_seconds'],source_registry_sha256=original['source_registry_sha256'],
        binary_sha256=original['binary_sha256'],original_release_key=original['release_key'],
        reused_workers=13,missing_workers=987,original_worker_unchanged=True)
    write(payload/'release_manifest.json',f);write(path,f)
    with c.open_sftp() as s:put_tree(s,payload,f['root'])
    code='from pathlib import Path;from ehe_common import *;p=Path('+repr(f['root'])+');verify_files(p,read(p/"release_manifest.json")["sha256"]);print("RECOVERY_RELEASE_SHA_PASS")'
    print(command(c,'cd '+q(f['root'])+' && python3 -c '+q(code),120),flush=True)
    write(REPORT/'transport_recovery_deployment.json',dict(passed=True,release_key=key,release_manifest_sha256=digest(payload/'release_manifest.json')))
    return f


def submit(c,f):
    from ehe_5e10_workflow import DATA, REPORT, CPU_BASE
    path=REPORT/'transport_recovery_job.json'
    if path.exists():return read(path)
    intent=REPORT/'transport_recovery_submission_intent.json'
    if intent.exists():raise RuntimeError('Unresolved recovery submission intent retained')
    name='ehe5e10_recover'
    if name in command(c,'squeue -u maty -h -o %j').splitlines():
        raise RuntimeError('Recovery already exists; reconcile before any submission')
    script=DATA/'scripts/transport_recovery.sh'
    lines='#!/bin/bash\nset -euo pipefail\n'+env_cpu()+'export OMP_NUM_THREADS=1\ncd '+q(f['root'])+'\n'
    lines+='srun --wait=0 --kill-on-bad-exit=0 --ntasks=1000 --cpus-per-task=1 --cpu-bind=cores'
    lines+=' --output='+q(CPU_BASE+'/logs/recovery_%j_task_%t.log')+' --error='+q(CPU_BASE+'/logs/recovery_%j_task_%t.err')
    lines+=' python3 -u ehe_5e10_recovery.py --release '+q(f['root'])+' --simulation '+q(CPU_BASE+'/simulation')+' --output '+q(f['transport_root'])+' --limit '+str(f['phase_limit_seconds'])+'\n'
    script.write_bytes(lines.encode());remote=CPU_BASE+'/scripts/transport_recovery.sh'
    with c.open_sftp() as s:s.put(str(script),remote)
    cmd='sbatch --parsable --partition=cnmix --nodes=18 --ntasks=1000 --cpus-per-task=1 --time=240 --job-name='+name+' --output='+q(CPU_BASE+'/logs/transport_recovery_%j.log')+' '+q(remote)
    write(intent,dict(status='submitting',created_epoch=time.time(),script_sha256=digest(script),command=cmd))
    job=int(command(c,cmd).strip().split(';')[0])
    record=dict(job=job,stage='transport',host='maty',release_key=f['release_key'],
        script=remote,script_sha256=digest(script),submitted_epoch=time.time(),minutes=240,
        allocation='single 18-node allocation; 1000 rank identities, 987 compute and 13 strict reuse',
        nodes=18,parallel_limit=1000,workers=1000,new_workers=987,reused_workers=13,
        photons_per_worker=50000000,total_primary_photons=50000000000,
        seed_first=33100101,seed_last=33101100,original_failed_job=15683333,srun_wait_seconds=0)
    write(path,record);write(intent,dict(status='submitted',job=job,script_sha256=digest(script)))
    shutil.copy2(REPORT/'transport_job.json',REPORT/'transport_job_initial_15683333.json')
    write(REPORT/'transport_job.json',record)
    print('EHE_5E10_MISSING_WORKER_RECOVERY_SUBMITTED',job,flush=True)
    return record


def prepare_gpu(job):
    from ehe_5e10_workflow import DATA, REPORT, BASE
    path=REPORT/'reconstruction_recovery_freeze.json'
    if path.exists():return read(path)
    if any((REPORT/(s+'_job.json')).exists() for s in ('validation','formal')):
        raise ValueError('Cannot replace a started reconstruction release')
    old=read(REPORT/'freeze.json');original=DATA/old['payload_dir'];verify_files(original,old['sha256'])
    payload=DATA/'reconstruction_recovery_payload';shutil.copytree(original,payload)
    (payload/'release_manifest.json').unlink()
    recovery=read(DATA/'transport_recovery_payload/config.json')['transport_recovery']
    config=read(payload/'config.json');config.update(transport_recovery=recovery,transport_job=job['job'])
    write(payload/'config.json',config)
    shutil.copy2(DATA/'transport_recovery_payload/ehe_5e10_transport.py',payload/'ehe_5e10_transport.py')
    for name in ('run_ehe_5e10_reconstruction.py','verify_ehe_5e10.py','torch_active_operator.py','single_checkpoint_mlem.py','whole_geometry.npz','truth_3mm.npz'):
        if digest(payload/name)!=old['sha256'][name]:raise ValueError('Original scientific reconstruction changed')
    files=hashes(payload);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    f={**old,'release_key':key,'payload_dir':payload.name,'root':BASE+'/releases/'+key,'sha256':files,
        'initial_unstarted_release_key':old['release_key'],'transport_recovery_job':job['job']}
    write(payload/'release_manifest.json',f);write(path,f)
    shutil.copy2(REPORT/'freeze.json',REPORT/'freeze_initial_unstarted.json')
    write(REPORT/'freeze.json',f)
    if (REPORT/'deployment_gpu.json').exists():
        shutil.copy2(REPORT/'deployment_gpu.json',REPORT/'deployment_gpu_initial_unstarted.json')
    return f


if __name__=='__main__':
    from ehe_5e10_workflow import connection,controller,deploy_gpu
    with controller():
        with connection('maty') as c:f=prepare(c);job=submit(c,f)
        gpu=prepare_gpu(job)
        with connection('gpu') as c:deploy_gpu(c,gpu)
