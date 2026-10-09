"""Registered independent 5e10 transport, original MLEM, strict acceptance/fetch."""
import argparse, contextlib, hashlib, json, math, os, shutil, sys, time
from pathlib import Path
from ehe_common import HERE, MATY_BASE, GPU_BASE, GPU_PYTHON, RESPONSES
from ehe_common import read, write, digest, hashes, verify_files
from ehe_5e9_workflow import connection, command, q, put_tree, env_cpu, env_gpu
from ehe_5e10_transport import STUDY, TOTAL, WORKERS, PER_VIEW, PER_WORKER, SEED_BASE
from ehe_5e10_transport import extract, verify_counts
from ehe_slurm_status import stage_completed

DATA = HERE/'generated'/STUDY
REPORT = HERE/'reports/NEMA_Body_H60'/STUDY
OLD_DATA = HERE/'generated/ehe_spect_5e9_200'
OLD_REPORT = HERE/'reports/NEMA_Body_H60/ehe_spect_5e9_200'
CPU_BASE = MATY_BASE.rsplit('/',1)[0]+'/'+STUDY
BASE = GPU_BASE.rsplit('/',1)[0]+'/'+STUDY
PARALLEL = 1000
NODES = 18


def cpu_freeze():
    path = REPORT/'transport_freeze.json'
    if path.exists():
        f=read(path);verify_files(DATA/f['payload_dir'],f['sha256'])
        verify_files(DATA/'simulation',f['source_registry_sha256']);return f
    original=read(OLD_REPORT/'transport_repair_freeze.json')
    accepted=read(OLD_REPORT/'transport_identity_acceptance.json')
    if not accepted['passed'] or accepted['release_key']!=original['release_key']:
        raise ValueError('Accepted executable/source identity required')
    old=OLD_DATA/'simulation';registry=read(old/'jobs.json')
    expected=read(OLD_DATA/'transport/collection.json')['files']
    for name,sha in hashes(old).items():
        if expected.get('source_registry/'+name)!=sha:
            raise ValueError('Accepted original source registry bytes differ')
    DATA.mkdir(parents=True,exist_ok=True)
    simulation=DATA/'simulation';simulation.mkdir(exist_ok=False)
    (simulation/'macros').mkdir()
    macro_rows=[];jobs=[]
    for row in registry['macros']:
        source=old/row['path']
        raw=source.read_bytes()
        if digest(source)!=row['sha256'] or raw.count(b'/run/beamOn 25000000')!=1:
            raise ValueError('Accepted source macro anchor differs')
        if b'/xcat/centerY -345' not in raw:
            raise ValueError('Original EHE -345 source center required')
        target=simulation/row['path']
        target.write_bytes(raw.replace(b'/run/beamOn 25000000',b'/run/beamOn 50000000'))
        macro_rows.append(dict(view=row['view'],path=row['path'],
                              bytes=target.stat().st_size,sha256=digest(target)))
        for worker in range(PER_VIEW):
            index=len(jobs)
            jobs.append(dict(index=index,dataset=registry['jobs'][0]['dataset'],level='5e10',
                view=row['view'],worker=worker,photons=PER_WORKER,seed=SEED_BASE+index,
                mono_keV=None,role='imaging',macro=row['path'],macro_sha256=digest(target)))
    registry.update(study=STUDY,level='5e10',jobs=jobs,macros=macro_rows,
        total_primary_photons=TOTAL,workers_per_view=PER_VIEW,seed_base=SEED_BASE)
    write(simulation/'jobs.json',registry)
    identity=dict(study=STUDY,registry_sha256=digest(simulation/'jobs.json'),
                  worker_source_sha256=digest(HERE/'ehe_5e10_transport.py'),
                  binary_sha256=accepted['binary_sha256'],original_release=original['release_key'])
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    payload=DATA/'transport_payload';payload.mkdir()
    for name in ('ehe_5e10_transport.py','ehe_common.py','ehe_slurm_status.py'):
        shutil.copy2(HERE/name,payload/name)
    original_root=MATY_BASE+'/releases/'+original['release_key']
    geometry={p.name:digest(p) for p in (OLD_REPORT/'geometry_evidence/raw').iterdir()}
    if set(geometry)!=set(['EHE_CollimatorHoles.csv','EHE_DetectorGeometry.csv','EHE_GeometrySummary.txt']):
        raise ValueError('Three authoritative original geometry members required')
    config=dict(study=STUDY,total_primary_photons=TOTAL,workers=WORKERS,
        workers_per_view=PER_VIEW,photons_per_worker=PER_WORKER,views=20,bins=2312,
        seed_base=SEED_BASE,transport_release_key=key,
        simulation_manifest_sha256=digest(simulation/'jobs.json'),
        source_registry_sha256=hashes(simulation),binary_path=original_root+'/build/ehe_spect',
        binary_sha256=accepted['binary_sha256'],original_transport_root=original_root,
        original_transport_source_sha256=original['sha256'],
        geometry_evidence_sha256=geometry,truth_sha256=read(OLD_REPORT/'freeze.json')['sha256']['truth_3mm.npz'],
        parallel_tasks=PARALLEL,nodes=NODES,
        human_instruction='EHE 5e10 Geant4 simulation plus reconstruction; increase parallel limit',
        scientific_changes='Dose/new seeds only; original source commands and actual executable retained')
    write(payload/'config.json',config)
    times=read(OLD_REPORT/'transport_measurement.json')['phase_seconds']
    limit=math.ceil((times['initialization_seconds']['maximum']+
                    times['beam_seconds']['maximum']*2)*1.8+300)
    f=dict(study=STUDY,release_key=key,payload_dir='transport_payload',
           root=CPU_BASE+'/releases/'+key,sha256=hashes(payload),
           source_registry_sha256=hashes(simulation),binary_sha256=config['binary_sha256'],
           original_release_key=original['release_key'],phase_limit_seconds=limit,
           time_basis='Original actual max initialization + 2x actual max beam, 1.8 margin +300s',
           binding_identity=identity)
    write(payload/'release_manifest.json',f);write(path,f)
    return f


def deploy_cpu(c,f):
    config=read(DATA/f['payload_dir']/'config.json')
    command(c,'mkdir -p '+q(CPU_BASE+'/releases')+' '+q(CPU_BASE+'/logs')+' '+q(CPU_BASE+'/scripts'))
    with c.open_sftp() as s:
        put_tree(s,DATA/f['payload_dir'],f['root'])
        put_tree(s,DATA/'simulation',CPU_BASE+'/simulation')
    code='from pathlib import Path;from ehe_common import *;r=Path('+repr(f['root'])+');f=read(r/"release_manifest.json");verify_files(r,f["sha256"]);c=read(r/"config.json");verify_files(Path(c["original_transport_root"]),c["original_transport_source_sha256"]);assert digest(c["binary_path"])==c["binary_sha256"];verify_files(Path('+repr(CPU_BASE+'/simulation')+'),c["source_registry_sha256"]);print("CPU_SOURCE_EXECUTABLE_REGISTRY_SHA_PASS")'
    print(command(c,'cd '+q(f['root'])+' && python3 -c '+q(code),120))
    symbols=command(c,'nm -C '+q(config['binary_path']))
    required=('G4MultiUnion::InsideWithExclusion','G4MultiUnion::InsideNoVoxels','G4MultiUnion::G4MultiUnion')
    names=[line.strip() for line in symbols.splitlines() if any(' T '+n in line for n in required)]
    if any(not any(' T '+n in line for line in names) for n in required):
        raise ValueError('Actual repaired Geant4 symbols must remain linked')
    write(REPORT/'transport_reuse_acceptance.json',dict(passed=True,
        release_key=f['release_key'],actual_binary_sha256=config['binary_sha256'],
        source_registry_sha256=f['source_registry_sha256'],linked_text_symbols=names,
        geometry_evidence_sha256=config['geometry_evidence_sha256'],read_only_original=True))
    write(REPORT/'deployment_cpu.json',dict(passed=True,release_key=f['release_key'],root=f['root']))


def registered(stage):
    path=REPORT/(stage+'_job.json')
    return read(path) if path.exists() else None


def accounting(c,job):
    return command(c,'sacct -j '+q(job)+' --parsable2 --noheader --format=JobID,State,ExitCode,MaxRSS,Elapsed,AllocTRES,NodeList',90)


def submit_transport(c,f):
    existing=registered('transport')
    if existing:return existing
    intent=REPORT/'transport_submission_intent.json'
    if intent.exists():raise RuntimeError('Unresolved submit intent: inspect scheduler before repeat')
    name='ehe5e10_transport'
    if name in command(c,'squeue -u maty -h -o %j').splitlines():
        raise ValueError('Existing same-name job; register it before any repeat')
    script=DATA/'scripts/transport.sh';script.parent.mkdir(exist_ok=True)
    text='#!/bin/bash\nset -euo pipefail\n'+env_cpu()
    text+='export OMP_NUM_THREADS=1\ncd '+q(f['root'])+'\n'
    text+='srun --kill-on-bad-exit=0 --ntasks=1000 --cpus-per-task=1 --cpu-bind=cores --output='+q(CPU_BASE+'/logs/task_%t.log')+' --error='+q(CPU_BASE+'/logs/task_%t.err')
    text+=' python3 -u ehe_5e10_transport.py worker --release '+q(f['root'])+' --simulation '+q(CPU_BASE+'/simulation')+' --output '+q(CPU_BASE+'/transport')+' --limit '+str(f['phase_limit_seconds'])+'\n'
    script.write_bytes(text.encode())
    remote=CPU_BASE+'/scripts/transport.sh'
    with c.open_sftp() as s:s.put(str(script),remote)
    minutes=math.ceil(f['phase_limit_seconds']/60)+10
    cmd='sbatch --parsable --partition=cnmix --nodes=18 --ntasks=1000 --ntasks-per-node=56 --cpus-per-task=1 --time='+str(minutes)+' --job-name='+name+' --output='+q(CPU_BASE+'/logs/transport_%j.log')+' '+q(remote)
    write(intent,dict(status='submitting',created_epoch=time.time(),script_sha256=digest(script),command=cmd))
    job=int(command(c,cmd).strip().split(';')[0])
    record=dict(job=job,stage='transport',host='maty',release_key=f['release_key'],
        script=remote,script_sha256=digest(script),submitted_epoch=time.time(),minutes=minutes,
        allocation='single multinode allocation, independent srun tasks',nodes=NODES,
        parallel_limit=PARALLEL,workers=WORKERS,photons_per_worker=PER_WORKER,
        total_primary_photons=TOTAL,seed_first=SEED_BASE,seed_last=SEED_BASE+WORKERS-1)
    write(REPORT/'transport_job.json',record)
    write(intent,dict(status='submitted',job=job,script_sha256=digest(script)))
    print('EHE_5E10_TRANSPORT_SUBMITTED',job,flush=True)
    return record


def fetch_file(s,remote,local,expected=None):
    local=Path(local);local.parent.mkdir(parents=True,exist_ok=True)
    if local.exists() and expected and digest(local)==expected:return
    temporary=local.with_name(local.name+'.fetching');s.get(remote,str(temporary))
    if expected and digest(temporary)!=expected:raise ValueError('Strict fetched SHA differs')
    os.replace(temporary,local)


def collect_transport(c,f,job,text):
    if (REPORT/'transport_acceptance.json').exists():return read(REPORT/'transport_acceptance.json')
    account=DATA/'transport_accounting.txt';account.write_bytes(text.encode())
    with c.open_sftp() as s:s.put(str(account),CPU_BASE+'/transport_accounting.txt')
    # archive(Path('transport_counts.tar.gz')) writes its receipt via with_suffix:
    # transport_counts.tar.json. A completed archive survives an SSH timeout.
    remote_receipt=CPU_BASE+'/transport_counts.tar.json'
    ready=command(c,'if [ -f '+q(remote_receipt)+' ] && [ -f '+q(CPU_BASE+'/transport_counts.tar.gz')+' ] && [ -f '+q(CPU_BASE+'/counts/collection.json')+' ]; then echo READY; fi').strip()
    if ready!='READY':
        line='cd '+q(f['root'])+' && python3 -u ehe_5e10_transport.py collect --release '+q(f['root'])+' --simulation '+q(CPU_BASE+'/simulation')+' --transport '+q(f.get('transport_root',CPU_BASE+'/transport'))+' --output '+q(CPU_BASE+'/counts')+' --job '+str(job)+' --accounting '+q(CPU_BASE+'/transport_accounting.txt')
        print(command(c,line,1800),flush=True)
    else:print('EHE_5E10_EXISTING_ARCHIVE_REUSED_NO_COLLECTION_RECOMPUTE',flush=True)
    bundle=DATA/'transport_counts.tar.gz';receipt=DATA/'transport_counts.json'
    with c.open_sftp() as s:
        fetch_file(s,remote_receipt,receipt)
        proof=read(receipt)
        if not proof['passed']:raise ValueError('Completed transport archive receipt required')
        fetch_file(s,CPU_BASE+'/transport_counts.tar.gz',bundle,proof['archive_sha256'])
    folder=DATA/'counts'
    if not folder.exists():extract(bundle,folder)
    if digest(folder/'collection.json')!=proof['collection_sha256']:
        raise ValueError('Fetched collection authority SHA differs')
    if 'transport_root' in f:
        from ehe_5e10_recovered_transport import verify_counts as check_counts
    else:check_counts=verify_counts
    collection=check_counts(folder,DATA/f['payload_dir'])
    write(REPORT/'transport_acceptance.json',dict(passed=True,job=job,
        collection_sha256=digest(folder/'collection.json'),strict_fetch_files=proof['files'],
        primary_counts=collection['primary_counts'],window_counts=collection['window_counts'],
        workers=WORKERS,actual_primary_photons=TOTAL,seed_first=SEED_BASE,seed_last=SEED_BASE+WORKERS-1,
        accounting=text,source_registry_sha256=f['source_registry_sha256'],
        actual_binary_sha256=f['binary_sha256'],cpu_operational_guard_only=True,
        imaging_resource_certificate=False,physical_calibration_claim=False,
        recovery=collection.get('recovery')))
    return read(REPORT/'transport_acceptance.json')


@contextlib.contextmanager
def controller():
    DATA.mkdir(parents=True,exist_ok=True);path=DATA/'controller.json'
    if path.exists():
        value=read(path)
        if value['status']=='running':
            if pid_alive(value['pid']):raise RuntimeError('Registered local controller alive; do not duplicate')
    value=dict(pid=os.getpid(),status='running',started_epoch=time.time())
    write(path,value)
    try:yield
    except BaseException:
        value.update(status='failed',finished_epoch=time.time());write(path,value);raise
    else:
        value.update(status='complete',exit_code=0,finished_epoch=time.time());write(path,value)


def pid_alive(pid):
    # On Windows os.kill(pid,0) can terminate a process. Only query its handle.
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        api.OpenProcess.restype=wintypes.HANDLE
        api.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        api.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=api.OpenProcess(0x1000,False,pid)
        if not handle:return ctypes.get_last_error()==5
        try:
            code=wintypes.DWORD()
            if not api.GetExitCodeProcess(handle,ctypes.byref(code)):raise OSError('Cannot inspect controller process')
            return code.value==259
        finally:api.CloseHandle(handle)
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    except PermissionError:return True
    return True


def advance():
    f=cpu_freeze()
    with connection('maty') as c:
        if not (REPORT/'deployment_cpu.json').exists():deploy_cpu(c,f)
        job=submit_transport(c,f)['job']
        text=accounting(c,job)
        if not stage_completed(text,job):
            return False
        if (REPORT/'transport_recovery_freeze.json').exists():
            f=read(REPORT/'transport_recovery_freeze.json')
            verify_files(DATA/f['payload_dir'],f['sha256'])
        collect_transport(c,f,job,text)
    return advance_reconstruction()


def gpu_freeze():
    path=REPORT/'freeze.json'
    if path.exists():
        f=read(path);verify_files(DATA/f['payload_dir'],f['sha256']);return f
    from ehe_conversion_workflow import response_root
    cpu=cpu_freeze();config=read(DATA/cpu['payload_dir']/'config.json')
    old=read(OLD_REPORT/'reconstruction_execution_freeze.json')
    old_payload=OLD_DATA/old['payload_dir'];original_config=read(old_payload/'config.json')
    config.update({k:original_config[k] for k in ('gamma_yields','truth_sha256','geometry_sha256','params_sha256')})
    config.update(data_kind='Geant4_transport',iterations=200,save_step=10,
        transport_job=registered('transport')['job'],transport_performed=True,
        factor_sha256={n:digest(OLD_DATA/'factor_evidence'/n/'factor_manifest.json') for n in RESPONSES},
        response_root=response_root(),physical_calibration_claim=False,
        emission_solid_angle_sr='4*pi',dose_equivalent_multiplier=1,
        source_angular_audit_sha256=digest(REPORT/'source_angular_audit.json'),
        relative_activity_integral_mm3=read(HERE/'reports/NEMA_Body_H60/manifest.json')['relative_activity_integral_mm3'],
        source_basis='Original 3mm 3D Xcat cuboids, full-sphere isotropic Geant4 photons',
        reconstruction_basis='Original full Polar volume-weighted density operator',
        background_source='This independent acquisition final440 single200 image, fixed additive Poisson term',
        initial_density=1,regularization=None,
        human_continuation='User explicitly continued original method despite existing response discrepancy; no physical calibration claim')
    payload=DATA/'payload';payload.mkdir(parents=True,exist_ok=False)
    # CPU worker execution bytes remain untouched. They also validate the exact
    # same receipts on the reconstruction host and during independent acceptance.
    shutil.copy2(DATA/cpu['payload_dir']/'ehe_5e10_transport.py',payload/'ehe_5e10_transport.py')
    for name in ('run_ehe_5e10_reconstruction.py','verify_ehe_5e10.py','ehe_slurm_status.py'):
        shutil.copy2(HERE/name,payload/name)
    for name in ('ehe_common.py','torch_active_operator.py','single_checkpoint_mlem.py','whole_geometry.npz','truth_3mm.npz'):
        if digest(old_payload/name)!=old['sha256'][name]:raise ValueError('Original complete execution helper/source identity differs')
        shutil.copy2(old_payload/name,payload/name)
    write(payload/'config.json',config)
    files=hashes(payload);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    f=dict(study=STUDY,release_key=key,payload_dir='payload',root=BASE+'/releases/'+key,
        sha256=files,response_root=config['response_root'],transport_release_key=cpu['release_key'],
        original_reconstruction_release_key=old['release_key'],
        data_kind='Geant4_transport',physical_calibration_claim=False)
    write(payload/'release_manifest.json',f);write(path,f);return f


def deploy_gpu(c,f):
    payload=DATA/f['payload_dir'];verify_files(payload,f['sha256'])
    command(c,'mkdir -p '+q(BASE+'/releases'))
    with c.open_sftp() as s:put_tree(s,payload,f['root'])
    code='from pathlib import Path;from ehe_common import *;r=Path('+repr(f['root'])+');verify_files(r,read(r/"release_manifest.json")["sha256"]);print("NEW_GPU_RELEASE_SHA_PASS")'
    print(command(c,'cd '+q(f['root'])+' && '+q(GPU_PYTHON)+' -c '+q(code)),flush=True)
    write(REPORT/'deployment_gpu.json',dict(passed=True,release_key=f['release_key'],
        release_manifest_sha256=digest(payload/'release_manifest.json')))


def sync_counts(c,f):
    storage=read(DATA/f['payload_dir']/'config.json').get('archive_storage')
    if storage:
        path=REPORT/'transport_gpu_archive_acceptance.json'
        if path.exists():
            v=read(path)
            if not v['passed'] or v['release_key']!=f['release_key'] or v['archive_sha256']!=storage['archive_sha256']:
                raise ValueError('Registered archive transfer identity differs')
            return v
        code='from ehe_common import digest;from pathlib import Path;p=Path('+repr(storage['archive_path'])+');assert p.stat().st_size=='+str(storage['archive_bytes'])+';assert digest(p)=='+repr(storage['archive_sha256'])+';print("ACCEPTED_GPU_ARCHIVE_SHA_PASS_NO_REUPLOAD")'
        print(command(c,'cd '+q(f['root'])+' && '+q(GPU_PYTHON)+' -c '+q(code),120),flush=True)
        value=dict(passed=True,archive_sha256=storage['archive_sha256'],archive_bytes=storage['archive_bytes'],
            collection_sha256=storage['collection_sha256'],release_key=f['release_key'],
            full_gpu_worker_acceptance_pending=True,scope='Archive transfer identity only; actual complete verification in each allocated stage',
            physical_calibration_claim=False)
        write(path,value);return value
    path=REPORT/'transport_gpu_sync_acceptance.json'
    if path.exists():return read(path)
    accepted=read(REPORT/'transport_acceptance.json')
    if not accepted['passed'] or accepted['actual_primary_photons']!=TOTAL:raise ValueError('Full actual transport acceptance required')
    bundle=DATA/'transport_counts.tar.gz';receipt=read(DATA/'transport_counts.json')
    if digest(bundle)!=receipt['archive_sha256']:raise ValueError('Accepted counts archive changed')
    command(c,'mkdir -p '+q(BASE))
    with c.open_sftp() as s:
        s.put(str(bundle),BASE+'/transport_counts.tar.gz')
    code='from pathlib import Path;from ehe_common import *;from ehe_5e10_transport import extract,verify_counts;p=Path('+repr(BASE)+');r=Path('+repr(f['root'])+');assert digest(p/"transport_counts.tar.gz")=='+repr(receipt['archive_sha256'])+';folder=p/"counts";extract(p/"transport_counts.tar.gz",folder) if not folder.exists() else None;assert digest(folder/"collection.json")=='+repr(accepted['collection_sha256'])+';v=verify_counts(folder,r);print("FULL1000_WORKER_GPU_COUNTS_SHA_PASS",v["total_primary_photons"])'
    print(command(c,'cd '+q(f['root'])+' && '+q(GPU_PYTHON)+' -c '+q(code),600),flush=True)
    value=dict(passed=True,actual_primary_photons=TOTAL,workers=WORKERS,
        collection_sha256=accepted['collection_sha256'],archive_sha256=receipt['archive_sha256'],
        strict_sha_files=receipt['files'],release_key=f['release_key'],physical_calibration_claim=False)
    write(path,value);return value


def submit_gpu(c,stage,lines,minutes):
    previous=registered(stage)
    if previous:return previous
    if len(command(c,'squeue -u scxi717 -h -o "%i"').splitlines())>=50:return None
    intent=REPORT/(stage+'_submission_intent.json')
    if intent.exists():raise RuntimeError('Unresolved submission intent retained; reconcile scheduler before repeat')
    name='ehe5e10_'+stage
    if name in command(c,'squeue -u scxi717 -h -o %j').splitlines():raise RuntimeError('Existing same-name stage: register before repeat')
    command(c,'mkdir -p '+q(BASE+'/scripts')+' '+q(BASE+'/logs'))
    script=DATA/'scripts'/(stage+'.sh');script.parent.mkdir(exist_ok=True)
    script.write_bytes(('#!/bin/bash\nset -euo pipefail\n'+env_gpu()+lines+'\n').encode())
    remote=BASE+'/scripts/'+script.name
    with c.open_sftp() as s:s.put(str(script),remote)
    cmd='sbatch --parsable --partition=gpu_5090 --account=scxi717 --qos=gpugpu -N 1 -n 1 -c 6 --gres=gpu:1 --exclude=wqd10nba06g6 --time='+str(minutes)
    cmd+=' --job-name='+q(name)+' --output='+q(BASE+'/logs/'+stage+'_%j.log')+' '+q(remote)
    write(intent,dict(status='submitting',stage=stage,created_epoch=time.time(),script_sha256=digest(script),command=cmd))
    job=int(command(c,cmd).strip().split(';')[0])
    value=dict(job=job,stage=stage,script=remote,script_sha256=digest(script),
        submitted_epoch=time.time(),slurm_minutes=minutes,release_key=read(REPORT/'freeze.json')['release_key'])
    write(REPORT/(stage+'_job.json'),value);write(intent,dict(status='submitted',job=job,script_sha256=digest(script)))
    print('EHE_5E10_GPU_SUBMITTED',stage,job,flush=True);return value


def verification_release(c,f):
    path=REPORT/'verification_freeze.json'
    if path.exists():return read(path)
    source=DATA/f['payload_dir'];folder=DATA/'verification_payload';folder.mkdir(exist_ok=False)
    names=['ehe_common.py','torch_active_operator.py','ehe_5e10_transport.py','verify_ehe_5e10.py','ehe_slurm_status.py']
    if f.get('input_storage')=='node_local_archive':names.append('ehe_5e10_archive_input.py')
    for name in names:
        shutil.copy2(source/name,folder/name)
    files=hashes(folder);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    v=dict(key=key,files=files,root=BASE+'/verification_releases/'+key,payload_dir='verification_payload')
    write(folder/'verification_manifest.json',v)
    command(c,'mkdir -p '+q(BASE+'/verification_releases'))
    with c.open_sftp() as s:put_tree(s,folder,v['root'])
    write(path,v);return v


def fetch_authority(c,mode,f):
    path=REPORT/(mode+'_summary.json')
    if path.exists():return read(path)
    v=read(REPORT/'verification_freeze.json');job=registered(mode+'_acceptance')['job'];text=accounting(c,job)
    if not stage_completed(text,job):return None
    dest=BASE+'/'+mode+'_acceptance';target=DATA/(mode+'_acceptance')
    with c.open_sftp() as s:
        fetch_file(s,dest+'/authority.json',target/'authority.json')
        proof=read(target/'authority.json')
        fetch_file(s,dest+'/evidence/receipt.json',target/'receipt.json');receipt=read(target/'receipt.json')
        if not receipt['passed'] or receipt['authority_sha256']!=digest(target/'authority.json'):raise ValueError('Immutable verifier receipt differs')
        if proof['verification_manifest_sha256']!=digest(DATA/v['payload_dir']/'verification_manifest.json'):raise ValueError('Actual verification code differs')
        if not proof['passed'] or proof['mode']!=mode or proof['release_key']!=f['release_key']:raise ValueError('Acceptance authority differs')
        if proof['counts_sha256']!=read(REPORT/'transport_acceptance.json')['collection_sha256']:raise ValueError('Actual acquisition authority differs')
        for name,sha in proof['files'].items():fetch_file(s,BASE+'/'+mode+'/'+name,DATA/'results'/mode/name,sha)
        for stage in (mode,mode+'_acceptance'):
            j=registered(stage)['job'];fetch_file(s,BASE+'/logs/'+stage+'_'+str(j)+'.log',DATA/'logs'/(stage+'_'+str(j)+'.log'))
            if f.get('input_storage')=='node_local_archive':
                local=target/(stage+'_input_receipt.json')
                fetch_file(s,BASE+'/input_receipts/'+stage+'.json',local)
                bootstrap=read(local);storage=read(DATA/f['payload_dir']/'config.json')['archive_storage']
                if not bootstrap['passed'] or bootstrap['status']!='complete' or str(bootstrap['job'])!=str(j):
                    raise ValueError('Actual allocated archive input stage did not complete')
                if bootstrap['release_key']!=f['release_key'] or bootstrap['bootstrap_sha256']!=f['sha256']['ehe_5e10_archive_input.py']:
                    raise ValueError('Archive bootstrap actual execution identity differs')
                if bootstrap['archive_sha256']!=storage['archive_sha256'] or bootstrap['collection_sha256']!=proof['counts_sha256']:
                    raise ValueError('Allocated input receipt differs from strict full-worker authority')
                write(REPORT/(stage+'_input_acceptance.json'),dict(**bootstrap,receipt_sha256=digest(local)))
    verify_files(DATA/'results'/mode,proof['files'])
    write(REPORT/(mode+'_acceptance_receipt.json'),receipt)
    write(REPORT/(mode+'_acceptance_sacct.json'),dict(job=job,accounting=text,passed=True))
    write(path,proof);print('EHE_5E10_STRICT_FETCH',mode,len(proof['files']),flush=True);return proof


def gpu_entry(f,stage,program,program_root,arguments):
    if f.get('input_storage')=='node_local_archive':
        return q(GPU_PYTHON)+' -u '+q(program_root+'/ehe_5e10_archive_input.py')+' --release '+q(f['root'])+' --program '+q(program_root+'/'+program)+' --receipt '+q(BASE+'/input_receipts/'+stage+'.json')+' -- '+arguments
    return q(GPU_PYTHON)+' -u '+q(program_root+'/'+program)+' '+arguments+' --counts '+q(BASE+'/counts')


def advance_reconstruction():
    f=gpu_freeze()
    with connection('gpu') as c:
        if not (REPORT/'deployment_gpu.json').exists() or read(REPORT/'deployment_gpu.json')['release_key']!=f['release_key']:deploy_gpu(c,f)
        sync_counts(c,f);r=f['root'];resp=f['response_root'];counts=BASE+'/counts'
        for mode in ('validation','formal'):
            if registered(mode) is None:
                if mode=='validation':limit=1800;iterations=10;wall=5400
                else:
                    authority=read(REPORT/'validation_summary.json')
                    if not authority['passed'] or not authority['operator_closure']['passed']:raise ValueError('Strict full-input validation authority required')
                    limit=math.ceil(max(authority['phase_seconds'].values())*20*1.8+300)
                    iterations=200;wall=2*limit+2400
                args='--release '+q(r)+' --responses '+q(resp)+' --output '+q(BASE+'/'+mode)+' --mode '+mode+' --iterations '+str(iterations)+' --limit '+str(limit)
                if mode=='formal':args+=' --authority '+q(BASE+'/validation_acceptance/authority.json')
                line='cd '+q(r)+'\ntimeout --signal=TERM --kill-after=10s '+str(wall)+'s '+gpu_entry(f,mode,'run_ehe_5e10_reconstruction.py',r,args)
                submit_gpu(c,mode,line,math.ceil(wall/60)+5);return False
            imaging=registered(mode);text=accounting(c,imaging['job'])
            if not stage_completed(text,imaging['job']):return False
            if registered(mode+'_acceptance') is None:
                v=verification_release(c,f);folder=DATA/'accounting';folder.mkdir(exist_ok=True)
                (folder/(mode+'.txt')).write_bytes(text.encode())
                shutil.copy2(DATA/'transport_accounting.txt',folder/'transport.txt')
                command(c,'mkdir -p '+q(BASE+'/accounting'))
                with c.open_sftp() as s:put_tree(s,folder,BASE+'/accounting')
                dest=BASE+'/'+mode+'_acceptance'
                args='--result '+q(BASE+'/'+mode)+' --release '+q(r)+' --responses '+q(resp)+' --accounting '+q(BASE+'/accounting/'+mode+'.txt')+' --transport-accounting '+q(BASE+'/accounting/transport.txt')+' --output '+q(dest+'/authority.json')+' --verification-manifest '+q(v['root']+'/verification_manifest.json')+' --evidence '+q(dest+'/evidence')
                line='cd '+q(v['root'])+'\nmkdir -p '+q(dest)+'\ntimeout --signal=TERM --kill-after=10s 2700s '+gpu_entry(f,mode+'_acceptance','verify_ehe_5e10.py',v['root'],args)
                submit_gpu(c,mode+'_acceptance',line,50);return False
            if fetch_authority(c,mode,f) is None:return False
    print('EHE_5E10_FORMAL200_STRICTLY_FETCHED',flush=True);return True


def watch(hours):
    began=time.monotonic()
    while time.monotonic()-began<hours*3600:
        if advance():return
        time.sleep(30)
    raise TimeoutError('Bounded controller time exhausted; registered jobs/results retained')


def status():
    if registered('transport'):
        with connection('maty') as c:
            job=registered('transport')['job']
            print(command(c,'squeue -h -j '+str(job)+' -o "%i|%T|%M|%R"'))
            print(accounting(c,job))
    if any(registered(stage) for stage in ('validation','validation_acceptance','formal','formal_acceptance')):
        with connection('gpu') as c:
            for stage in ('validation','validation_acceptance','formal','formal_acceptance'):
                value=registered(stage)
                if value:print(stage,accounting(c,value['job']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['advance','status','watch']);p.add_argument('--hours',type=float,default=8)
    a=p.parse_args()
    if a.action=='advance':
        with controller():advance()
    elif a.action=='watch':
        with controller():watch(a.hours)
    else:status()
