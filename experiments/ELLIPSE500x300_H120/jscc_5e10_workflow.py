"""Registered, bounded JSCC actual-5e10 acquisition and three-route reconstruction."""
import argparse, contextlib, ctypes, hashlib, json, math, os, re, shlex, shutil, sys, tarfile, time
from pathlib import Path
from jscc_5e10_common import *
from ehe_5e9_workflow import connection, command
from jscc_5e10_array import array_accounting

q=lambda x:shlex.quote(str(x))

def accounting(c,job):
    return command(c,f'sacct -j {job} -n -P --format=JobIDRaw,State,ExitCode,MaxRSS,Elapsed,AllocTRES,NodeList',timeout=60)

def completed(text,job):
    lines=[x.split('|') for x in text.splitlines()]
    return any(x[:3]==[str(job),'COMPLETED','0:0'] for x in lines) and all(x[1]=='COMPLETED' and x[2]=='0:0' for x in lines if x[0].startswith(str(job)+'.'))

def job_state(c,record):
    if record.get('array'):
        job=record['job']
        s=command(c,f'sacct -j {job} -n -P --format=JobID%40,JobIDRaw%40,State,ExitCode,NNodes,NCPUS,MaxRSS,Elapsed,AllocTRES%200,NodeList%120',timeout=120)
        proof=array_accounting(s,job)
        all_queue=command(c,"squeue -h -u maty -o '%i|%j|%T|%M|%D|%R'",timeout=60)
        queue='\n'.join(x for x in all_queue.splitlines() if x.split('|')[0].startswith(str(job)+'_'))
        write(REPORT/'transport_array_accounting.json',proof)
        write(REPORT/'transport_latest_status.json',dict(job=job,queue=queue,state_counts=proof['state_counts'],checked_epoch=time.time(),accounting_sha256=digest(REPORT/'transport_array_accounting.json')))
        if proof['failures']:raise RuntimeError('Array worker failure; preserve successful workers and diagnose: '+str(proof['failures']))
        return ('complete' if proof['passed'] and not queue else 'waiting'),s
    job=record['job'];s=accounting(c,job)
    user='maty' if record['host']=='maty' else 'scxi717'
    all_queue=command(c,f"squeue -h -u {user} -o '%i|%j|%T|%M|%D|%R'",timeout=60)
    queue='\n'.join(x for x in all_queue.splitlines() if x.split('|')[0]==str(job))
    write(REPORT/(record['stage']+'_latest_status.json'),dict(job=job,accounting=s,queue=queue,checked_epoch=time.time()))
    if queue.strip():return 'waiting',s
    if completed(s,job):return 'complete',s
    if any(x.split('|')[0]==str(job) and x.split('|')[1] in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED') for x in s.splitlines()):
        raise RuntimeError('Registered '+record['stage']+' failed; preserve outputs and diagnose before any repair: '+s)
    return 'waiting',s

def prepare_transport():
    if (REPORT/'transport_freeze.json').exists():return
    DATA.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    old=HERE/'generated/NEMA_Body_H60/Simulation_5e9';source=read(old/'jobs.json')
    sim=DATA/'source_registry';sim.mkdir(exist_ok=False);(sim/'macros').mkdir()
    macros={}
    for v in range(1,21):
        original=old/source['jobs'][(v-1)*10]['macro']
        body=original.read_bytes()
        new=re.sub(rb'/run/beamOn\s+25000000',b'/run/beamOn 50000000',body)
        if new==body or new.replace(b'/run/beamOn 50000000',b'/run/beamOn 25000000')!=body:
            raise ValueError('Only the photon count may change in the original 3D source macro')
        dest=sim/f'macros/NEMA_Body_H60_v{v:02d}.mac';dest.write_bytes(new);macros[v]=dest
    jobs=[]
    for i in range(WORKERS):
        v=i//50+1;oldjob=source['jobs'][(v-1)*10]
        jobs.append(dict(oldjob,index=i,view=v,worker=i%50,photons=PER_WORKER,seed=SEED_BASE+i,level='5e10',
            macro=f'macros/NEMA_Body_H60_v{v:02d}.mac',macro_sha256=digest(macros[v])))
    r={k:v for k,v in source.items() if k not in ('jobs','macros','macro_files')}
    r.update(jobs=jobs,total_primary_photons=TOTAL,workers_per_view=50,source_solid_angle='4pi',dose_multiplier=1,
        source_registry_reference_sha256=digest(old/'jobs.json'))
    write(sim/'jobs.json',r);registry(sim/'jobs.json')
    payload=DATA/'transport_payload';payload.mkdir(exist_ok=False)
    for n in ('jscc_5e10_common.py','jscc_5e10_transport.py','ehe_common.py'):shutil.copy2(HERE/n,payload/n)
    write(payload/'transport_config.json',dict(study=STUDY,binary_sha256=BINARY_SHA,crystal_sha256=CRYSTAL_SHA,
        source_registry_sha256=hashes(sim),total_primary_photons=TOTAL,seed_first=SEED_BASE,seed_last=SEED_BASE+999,
        original_source_registry_sha256=digest(old/'jobs.json'),source_solid_angle='4pi',dose_multiplier=1,
        source_primary_generator_sha256=digest(HERE.parents[1]/'Geant4Sim/Geant4Code/src/PrimaryGeneratorAction.cc')))
    members=hashes(payload);key=hashlib.sha256(json.dumps(members,sort_keys=True).encode()).hexdigest()[:16]
    write(payload/'release_manifest.json',dict(release_key=key,sha256=members))
    write(REPORT/'transport_freeze.json',dict(release_key=key,sha256=hashes(payload),source_registry_sha256=hashes(sim),
        binary_sha256=BINARY_SHA,crystal_sha256=CRYSTAL_SHA,workers=1000,total_primary_photons=TOTAL))

def put_archive(c,folder,remote):
    archive=DATA/(folder.name+'.tar.gz')
    with tarfile.open(archive,'w:gz') as t:
        for n in sorted(hashes(folder)):t.add(folder/n,arcname=n,recursive=False)
    command(c,'test ! -e '+q(remote)+' && mkdir -p '+q(remote),timeout=60)
    with c.open_sftp() as s:s.put(str(archive),remote+'/payload.tar.gz')
    if command(c,'sha256sum '+q(remote+'/payload.tar.gz')).split()[0]!=digest(archive):raise ValueError('Archive SHA transfer differs')
    command(c,'tar --no-same-owner -xzf '+q(remote+'/payload.tar.gz')+' -C '+q(remote),timeout=600)
    return hashes(folder)

def deploy_transport():
    if (REPORT/'transport_deployment.json').exists():return
    f=read(REPORT/'transport_freeze.json');verify_files(DATA/'transport_payload',f['sha256'])
    release=CPU_BASE+'/transport_releases/'+f['release_key'];simulation=CPU_BASE+'/source_registry'
    with connection('maty') as c:
        command(c,'mkdir -p '+q(CPU_BASE+'/logs'),timeout=60)
        put_archive(c,DATA/'transport_payload',release);put_archive(c,DATA/'source_registry',simulation)
        command(c,'cp -- '+q(CPU_PROJECT+'/Geant4Build/gamma01')+' '+q(release+'/gamma01')+' && cp -- '+
            q(CPU_PROJECT+'/Geant4Sim/Geant4Code/CrystalMatrix.txt')+' '+q(release+'/CrystalMatrix.txt'),timeout=60)
        result=command(c,'sha256sum '+q(release+'/gamma01')+' '+q(release+'/CrystalMatrix.txt'))
        if [x.split()[0] for x in result.splitlines()]!=[BINARY_SHA,CRYSTAL_SHA]:raise ValueError('Previous accepted actual binary changed')
        check='from pathlib import Path;from jscc_5e10_common import *;verify_files(Path('+repr(release)+'),read(Path('+repr(release)+')/"release_manifest.json")["sha256"]);registry(Path('+repr(simulation)+')/"jobs.json");print("TRANSPORT_IDENTITY_PASSED")'
        print(command(c,'cd '+q(release)+' && python3 -c '+q(check),timeout=60))
    write(REPORT/'transport_deployment.json',dict(release=release,simulation=simulation,binary_evidence=result,sha256=f['sha256']))

def prepare_array_transport():
    """Separate control release; original pilot release/source remain immutable."""
    if (REPORT/'transport_array_deployment.json').exists():return
    old=read(REPORT/'transport_deployment.json');cancel=read(REPORT/'transport_15705120_cancellation.json')
    if not cancel['passed'] or cancel['actual_primary_photons']!=0:raise ValueError('Old queued allocation must be fully cancelled without workers')
    payload=DATA/'transport_array_payload'
    if not payload.exists():
        payload.mkdir()
        for n in ('jscc_5e10_common.py','jscc_5e10_transport.py','jscc_5e10_array.py','test_jscc_5e10_array.py','ehe_common.py'):shutil.copy2(HERE/n,payload/n)
        shutil.copy2(DATA/'transport_payload/transport_config.json',payload/'transport_config.json')
        members=hashes(payload);key=hashlib.sha256(json.dumps(members,sort_keys=True).encode()).hexdigest()[:16]
        write(payload/'release_manifest.json',dict(release_key=key,sha256=members))
        write(REPORT/'transport_array_freeze.json',dict(release_key=key,sha256=hashes(payload),source_registry_sha256=read(REPORT/'transport_freeze.json')['source_registry_sha256'],
            original_freeze_sha256=digest(REPORT/'transport_freeze.json'),original_release=old['release'],binary_sha256=BINARY_SHA,crystal_sha256=CRYSTAL_SHA,
            change='Independent one-node one-core allocation and scheduler identity only; identical worker physics, source, dose, seed and binary',
            workers=WORKERS,total_primary_photons=TOTAL,max_concurrent_workers=1000,nodes_per_worker=1,cpus_per_worker=1))
    f=read(REPORT/'transport_array_freeze.json');verify_files(payload,f['sha256'])
    release=CPU_BASE+'/transport_array_releases/'+f['release_key']
    with connection('maty') as c:
        put_archive(c,payload,release)
        command(c,'cp -- '+q(old['release']+'/gamma01')+' '+q(release+'/gamma01')+' && cp -- '+q(old['release']+'/CrystalMatrix.txt')+' '+q(release+'/CrystalMatrix.txt'),timeout=120)
        check='from pathlib import Path;from jscc_5e10_common import *;verify_files(Path("."),read("release_manifest.json")["sha256"]);assert digest("gamma01")==BINARY_SHA;assert digest("CrystalMatrix.txt")==CRYSTAL_SHA;verify_files(Path('+repr(old['simulation'])+'),read("transport_config.json")["source_registry_sha256"]);print("ARRAY_RELEASE_SOURCE_BINARY_SHA_PASSED")'
        identity=command(c,'cd '+q(release)+' && python3 -c '+q(check),timeout=120)
        tests=command(c,'cd '+q(release)+' && python3 -m unittest test_jscc_5e10_array -v 2>&1',timeout=120)
        if '\nOK' not in tests:raise ValueError('Actual array metadata tests failed')
    (REPORT/'transport_array_linux_tests.txt').write_bytes(tests.encode())
    write(REPORT/'transport_array_deployment.json',dict(release=release,simulation=old['simulation'],sha256=f['sha256'],
        actual_identity_evidence=identity,actual_linux_tests_sha256=digest(REPORT/'transport_array_linux_tests.txt'),source_registry_reused_read_only=True))

def submit_array_transport():
    path=REPORT/'transport_job.json'
    if path.exists() and read(path).get('array'):return
    if path.exists() and read(path)['job']!=15705120:raise ValueError('Unexpected original registered transport')
    prepare_array_transport();d=read(REPORT/'transport_array_deployment.json');release=d['release'];sim=d['simulation']
    proof=read(REPORT/'transport_pilot_acceptance.json')
    if not proof['passed']:raise ValueError('Actual pilot must remain passed')
    # Keep the original measured bounded runtime; allocation change does not alter physics.
    limit=max(3600,math.ceil(proof['resource']['elapsed_seconds']*500*4+300));minutes=math.ceil((limit+300)/60)
    out=CPU_BASE+'/transport_array_'+read(REPORT/'transport_array_freeze.json')['release_key']
    script='''#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load compilers/gcc/v12.2.0
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[[ "${SLURM_CPUS_PER_TASK}" == 1 && "${SLURM_NNODES}" == 1 && "${SLURM_NTASKS}" == 1 ]]
index="${SLURM_ARRAY_TASK_ID}"
printf -v suffix "worker_%04d" "$index"
srun --chdir=/tmp --kill-on-bad-exit=1 --wait=0 python3 '''+q(release+'/jscc_5e10_transport.py')+' worker --release '+q(release)+' --simulation '+q(sim)+' --output '+q(out)+'/"$suffix" --index "$index" --limit '+str(limit)+'\n'
    local=DATA/'transport_array.sh';local.write_bytes(script.encode());remote=CPU_BASE+'/transport_array.sh'
    intent=REPORT/'transport_array_submission_intent.json'
    if intent.exists():raise ValueError('Unresolved array submission intent; inspect instead of submitting twice')
    with connection('maty') as c:
        # The explicitly superseded combined allocation must still have no workers or queue entry.
        old_acc=accounting(c,15705120)
        if not any(x.startswith('15705120|CANCELLED') for x in old_acc.splitlines()):raise ValueError('Original allocation is not fully cancelled')
        command(c,'test ! -d '+q(out),timeout=60)
        with c.open_sftp() as s:s.put(str(local),remote)
        if command(c,'sha256sum '+q(remote)).split()[0]!=digest(local):raise ValueError('Array submission SHA differs')
        command(c,'bash -n '+q(remote))
        write(intent,dict(stage='transport',array=True,script_sha256=digest(local),started_epoch=time.time(),workers=1000,max_concurrent_workers=1000))
        job=command(c,'sbatch --parsable -p cnmix -N 1 -n 1 --ntasks-per-node=1 --cpus-per-task=1 --array=0-999%1000 --time='+str(minutes)+' --job-name=JSCC5e10_worker --chdir=/tmp --output='+q(CPU_BASE+'/logs/transport.%A_%a.out')+' --error='+q(CPU_BASE+'/logs/transport.%A_%a.err')+' '+q(remote),timeout=60).split(';')[0].strip()
        if not job.isdigit():raise ValueError('Ambiguous array submission; do not retry')
    write(path,dict(job=int(job),stage='transport',host='maty',array=True,array_range='0-999%1000',nodes_per_worker=1,cpus_per_worker=1,
        tasks=1000,max_concurrent_workers=1000,release=release,simulation=sim,output=out,limit_seconds=limit,walltime_minutes=minutes,
        script_sha256=digest(local),superseded_queued_job=15705120,cancellation_sha256=digest(REPORT/'transport_15705120_cancellation.json'),submitted_epoch=time.time()))
    write(intent,dict(read(intent),registered_job=int(job),resolved=True));print('REGISTERED_INDEPENDENT_CPU_ARRAY',job,flush=True)

def submit_cpu(stage):
    if stage=='transport':return submit_array_transport()
    path=REPORT/(stage+'_job.json')
    if path.exists():return
    intent=REPORT/(stage+'_submission_intent.json')
    if intent.exists():raise ValueError('Unresolved submission intent; inspect queue rather than submitting twice')
    d=read(REPORT/'transport_deployment.json');release=d['release'];sim=d['simulation']
    if stage=='transport':
        proof=read(REPORT/'transport_pilot_acceptance.json')
        if not proof['passed']:raise ValueError('Actual binary/source pilot is required')
        # Conservative four-times measured throughput, including launch/I/O overhead.
        limit=max(3600,math.ceil(proof['resource']['elapsed_seconds']*500*4+300));minutes=math.ceil((limit+300)/60)
        tasks,nodes=1000,18;out=CPU_BASE+'/transport';pilot=''
    else:limit,minutes,tasks,nodes,out,pilot=600,15,1,1,CPU_BASE+'/transport_pilot',' --pilot'
    script='''#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load compilers/gcc/v12.2.0
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JSCC_RELEASE='''+q(release)+''' JSCC_SIMULATION='''+q(sim)+''' JSCC_OUTPUT='''+q(out)+''' JSCC_LIMIT='''+str(limit)+'''
srun --chdir=/tmp --kill-on-bad-exit=1 --wait=0 bash -c '
 index=${SLURM_PROCID:-0}
 if [[ "''' +stage+'''" == transport ]]; then printf -v suffix "worker_%04d" "$index"; output="$JSCC_OUTPUT/$suffix"; else output="$JSCC_OUTPUT"; fi
 exec python3 "$JSCC_RELEASE/jscc_5e10_transport.py" worker --release "$JSCC_RELEASE" --simulation "$JSCC_SIMULATION" --output "$output" --index "$index" --limit "$JSCC_LIMIT"'''+pilot+'''
'
'''
    script_path=DATA/(stage+'.sh');script_path.write_bytes(script.encode());remote=CPU_BASE+'/'+stage+'.sh'
    with connection('maty') as c:
        with c.open_sftp() as s:s.put(str(script_path),remote)
        if command(c,'sha256sum '+q(remote)).split()[0]!=digest(script_path):raise ValueError('Submission script transfer differs')
        command(c,'bash -n '+q(remote))
        write(intent,dict(stage=stage,script_sha256=digest(script_path),started_epoch=time.time(),nodes=nodes,tasks=tasks))
        sbatch=f'sbatch --parsable -p cnmix -N {nodes} -n {tasks} --ntasks-per-node=56 --cpus-per-task=1 --time={minutes} --job-name=JSCC5e10_{stage} --chdir=/tmp '
        if tasks==1:sbatch=sbatch.replace('--ntasks-per-node=56','--ntasks-per-node=1')
        job=command(c,sbatch+' --output='+q(CPU_BASE+'/logs/'+stage+'.%j.out')+' --error='+q(CPU_BASE+'/logs/'+stage+'.%j.err')+' '+q(remote),timeout=60).split(';')[0].strip()
        if not job.isdigit():raise ValueError('Ambiguous submit outcome; do not retry')
    write(path,dict(job=int(job),stage=stage,host='maty',nodes=nodes,tasks=tasks,release=release,limit_seconds=limit,
        walltime_minutes=minutes,script_sha256=digest(script_path),submitted_epoch=time.time()))
    write(intent,dict(read(intent),registered_job=int(job),resolved=True));print('REGISTERED_CPU_JOB',stage,job,flush=True)

def accept_pilot():
    if (REPORT/'transport_pilot_acceptance.json').exists():return True
    record=read(REPORT/'transport_pilot_job.json')
    with connection('maty') as c:
        state,s=job_state(c,record)
        if state!='complete':return False
        dest=REPORT/'transport_pilot';dest.mkdir(exist_ok=True)
        with c.open_sftp() as f:
            for n in ('receipt.json','allocation.json','console.log','run.mac'):
                sha=command(c,'sha256sum '+q(CPU_BASE+'/transport_pilot/'+n)).split()[0]
                f.get(CPU_BASE+'/transport_pilot/'+n,str(dest/n))
                if digest(dest/n)!=sha:raise ValueError('Pilot proof fetch SHA differs')
        proof=read(dest/'receipt.json')
        if not proof['passed'] or not proof['pilot'] or proof['photons']!=100000 or proof['seed']!=PILOT_SEED or proof['allocation_job']!=str(record['job']):
            raise ValueError('Actual pilot identity differs')
        proof.update(accounting=s,slurm_completed=True);write(REPORT/'transport_pilot_acceptance.json',proof)
    print('JSCC_TRANSPORT_PILOT_ACTUALLY_PASSED',record['job'],flush=True);return True

def submit_collection():
    if (REPORT/'collection_job.json').exists():return
    transport=read(REPORT/'transport_job.json');d=read(REPORT/'transport_array_deployment.json')
    scheduler=REPORT/'transport_array_accounting.json'
    if not read(scheduler)['passed'] or read(scheduler)['parent_job']!=transport['job']:raise ValueError('All 1000 actual workers must successfully exit before collection')
    script='''#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 '''+q(d['release']+'/jscc_5e10_transport.py')+' collect --release '+q(d['release'])+' --simulation '+q(d['simulation'])+' --transport '+q(transport['output'])+' --output '+q(CPU_BASE+'/input')+' --job '+str(transport['job'])+' --accounting '+q(CPU_BASE+'/transport_array_accounting.json')+'\n'
    local=DATA/'collection.sh';local.write_bytes(script.encode());remote=CPU_BASE+'/collection.sh'
    intent=REPORT/'collection_submission_intent.json'
    if intent.exists():raise ValueError('Unresolved collection submission intent')
    with connection('maty') as c:
        with c.open_sftp() as s:
            s.put(str(local),remote);s.put(str(scheduler),CPU_BASE+'/transport_array_accounting.json')
        if command(c,'sha256sum '+q(CPU_BASE+'/transport_array_accounting.json')).split()[0]!=digest(scheduler):raise ValueError('Complete worker scheduler mapping SHA differs')
        command(c,'bash -n '+q(remote));write(intent,dict(stage='collection',script_sha256=digest(local)))
        job=command(c,'sbatch --parsable -p cnmix -N 1 -n 1 --cpus-per-task=2 --time=60 --chdir=/tmp --job-name=JSCC5e10_collection --output='+q(CPU_BASE+'/logs/collection.%j.out')+' --error='+q(CPU_BASE+'/logs/collection.%j.err')+' '+q(remote),timeout=60).split(';')[0].strip()
        if not job.isdigit():raise ValueError('Ambiguous collection submission')
    write(REPORT/'collection_job.json',dict(job=int(job),stage='collection',host='maty',source_transport_job=transport['job'],script_sha256=digest(local)))
    write(intent,dict(read(intent),registered_job=int(job),resolved=True))

def fetch_transport():
    if (REPORT/'transport_acceptance.json').exists():return True
    r=read(REPORT/'collection_job.json')
    with connection('maty') as c:
        state,acc=job_state(c,r)
        if state!='complete':return False
        with c.open_sftp() as s:
            for n in ('collection_package.json','input/collection.json'):
                dest=DATA/'input_collection.json' if n.startswith('input') else DATA/n
                sha=command(c,'sha256sum '+q(CPU_BASE+'/'+n)).split()[0];s.get(CPU_BASE+'/'+n,str(dest))
                if digest(dest)!=sha:raise ValueError('Collection proof fetch differs')
            pack=read(DATA/'collection_package.json');archive=DATA/'transport_input.tar.gz'
            if not archive.exists() or digest(archive)!=pack['sha256']:s.get(CPU_BASE+'/transport_input.tar.gz',str(archive))
            if digest(archive)!=pack['sha256']:raise ValueError('Full input archive transfer SHA differs')
        # A single independent archive in the new study; all worker originals remain on maty.
        root=DATA/'input'
        if not root.exists():
            root.mkdir()
            with tarfile.open(archive) as t:
                for m in t.getmembers():
                    if not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts:raise ValueError('Unsafe archive member')
                t.extractall(root)
        coll=validate_collection(read(root/'collection.json'));verify_files(root,coll['files'])
        verify_files(root/'source_registry',read(REPORT/'transport_freeze.json')['source_registry_sha256'])
        record=read(REPORT/'transport_job.json');state,transport_acc=job_state(c,record)
        if state!='complete':raise ValueError('All original independent workers must successfully exit')
        local_scheduler=read(root/'transport_array_accounting.json')
        if local_scheduler['roots']!=read(REPORT/'transport_array_accounting.json')['roots']:raise ValueError('Actual collected child allocation mapping differs')
        write(REPORT/'transport_acceptance.json',dict(coll,collection_sha256=digest(root/'collection.json'),archive_sha256=pack['sha256'],collection_accounting=acc,
            transport_accounting=transport_acc,strict_local_sha_passed=True))
    return True

def alive(pid):
    if os.name=='nt':
        h=ctypes.windll.kernel32.OpenProcess(0x1000,False,int(pid))
        if h:ctypes.windll.kernel32.CloseHandle(h);return True
        return False
    try:os.kill(int(pid),0);return True
    except OSError:return False

@contextlib.contextmanager
def advancing():
    DATA.mkdir(parents=True,exist_ok=True);p=DATA/'advance_registration.json'
    if p.exists():
        previous=read(p)
        if previous.get('status')=='running' and alive(previous['pid']):raise RuntimeError('Registered local advance process is still alive; do not run concurrently')
    write(p,dict(pid=os.getpid(),status='running',started_epoch=time.time()))
    code=0
    try:yield
    except BaseException:
        code=1;raise
    finally:write(p,dict(read(p),status='complete',exit_code=code,finished_epoch=time.time()))

def advance():
    with advancing():
        from jscc_5e10_reconstruction_workflow import prepare_science,advance_reconstruction
        prepare_science()
        if (REPORT/'transport_acceptance.json').exists():
            advance_reconstruction();return
        prepare_transport();deploy_transport();submit_cpu('transport_pilot')
        if not accept_pilot():print('WAITING_TRANSPORT_PILOT');return
        submit_cpu('transport')
        with connection('maty') as c:state,_=job_state(c,read(REPORT/'transport_job.json'))
        if state!='complete':print('WAITING_FULL_TRANSPORT');return
        submit_collection()
        if not fetch_transport():print('WAITING_COLLECTION');return
        advance_reconstruction()

def status():
    for name in ('transport_pilot','transport','collection','selection','selection_verification','validation','validation_verification','formal','formal_verification'):
        p=REPORT/(name+'_job.json')
        if not p.exists():continue
        r=read(p)
        with connection(r['host']) as c:
            state,acc=job_state(c,r);print(name,r['job'],state,read(REPORT/'transport_latest_status.json')['state_counts'] if r.get('array') else acc,flush=True)
            if r.get('array'):continue
            base=CPU_BASE if r['host']=='maty' else GPU_BASE
            for ext in ('out','err'):
                f=base+'/logs/'+name+'.'+str(r['job'])+'.'+ext
                print(command(c,'if test -f '+q(f)+'; then tail -n 8 '+q(f)+'; fi'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('advance','status'))
    a=p.parse_args();globals()[a.action]()
