"""Independent EHE prepare/deploy/pilots/transport/response/validation/formal/fetch lifecycle."""
import argparse,contextlib,math,os,shlex,sys,time
from pathlib import Path
import paramiko
from ehe_common import *

@contextlib.contextmanager
def connection(host):
    if host=='gpu':
        sys.path.insert(0,str(ROOT/'experiments/FOV120'))
        from reconstruction_ssh import connect
        c=connect()
    else:
        c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
        c.connect('192.168.11.1',username='maty',look_for_keys=True,allow_agent=True,timeout=20)
    try:yield c
    finally:c.close()

def command(c,text,timeout=120):
    _,out,err=c.exec_command(text,timeout=timeout);channel=out.channel;chunks=[];errors=[];began=time.monotonic()
    while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
        if channel.recv_ready():chunks.append(channel.recv(65536))
        if channel.recv_stderr_ready():errors.append(channel.recv_stderr(65536))
        if time.monotonic()-began>timeout:channel.close();raise TimeoutError('Remote command timed out')
        time.sleep(.05)
    result=b''.join(chunks).decode(errors='replace');message=b''.join(errors).decode(errors='replace')
    if channel.recv_exit_status():raise RuntimeError((result+message)[-6000:])
    return result

def q(v):return shlex.quote(str(v))
def base(host):return MATY_BASE if host=='maty' else GPU_BASE
def frozen(host):
    name='transport_repair_freeze.json' if host=='maty' else 'response_repair_freeze.json'
    return read(REPORT/(name if (REPORT/name).exists() else 'freeze.json'))
def release(host):return base(host)+'/releases/'+frozen(host)['release_key']
def python(host):return 'python3' if host=='maty' else GPU_PYTHON

def fetch_transport_geometry(c,pilot):
    """Hash the actual remote geometry evidence before checking fetched bytes."""
    import json
    paths={name+'/EHE_GeometryComparison.txt':base('maty')+'/geometry_evidence/'+name+'/EHE_GeometryComparison.txt' for name in RESPONSES}
    paths.update({'raw/'+name:pilot['output']+'/'+name for name in ('EHE_GeometrySummary.txt','EHE_CollimatorHoles.csv','EHE_DetectorGeometry.csv')})
    paths['union_audit.json']=pilot['output']+'/EHE_MultiUnionAudit.json'
    code='import json;from ehe_common import digest;print(json.dumps({n:digest(p) for n,p in '+repr(paths)+'.items()}))'
    expected=json.loads(command(c,'cd '+q(release('maty'))+' && python3 -c '+q(code)))
    target=REPORT/'geometry_evidence'
    with c.open_sftp() as s:
        for name,remote in paths.items():
            local=target/name;local.parent.mkdir(parents=True,exist_ok=True);s.get(remote,str(local))
    verify_files(target,expected)
    return expected

def put_tree(sftp,source,target):
    try:sftp.mkdir(target)
    except OSError:pass
    for p in sorted(source.iterdir()):
        dest=target+'/'+p.name
        if p.is_dir():put_tree(sftp,p,dest)
        else:
            try:
                with sftp.open(dest,'rb') as f:
                    import hashlib
                    h=hashlib.sha256()
                    for b in iter(lambda:f.read(8<<20),b''):h.update(b)
                if h.hexdigest()!=digest(p):raise ValueError('Existing deployed bytes differ: '+dest)
            except FileNotFoundError:sftp.put(str(p),dest)

def deploy(host):
    r=frozen(host);candidate='transport_payload' if host=='maty' else 'response_payload';payload=DATA/r.get('payload_dir',candidate if (REPORT/('transport_repair_freeze.json' if host=='maty' else 'response_repair_freeze.json')).exists() else 'payload');verify_files(payload,r['sha256'])
    local=DATA/f'release_manifest_{host}.json';write(local,r)
    with connection(host) as c:
        command(c,'mkdir -p '+q(base(host)+'/releases')+' '+q(base(host)+'/logs'))
        with c.open_sftp() as s:
            put_tree(s,payload,release(host));s.put(str(local),release(host)+'/release_manifest.json')
            if host=='maty':put_tree(s,DATA/'simulation',base(host)+'/simulation')
        check='from pathlib import Path;from ehe_common import *;r=Path('+repr(release(host))+');verify_files(r,read(r/"release_manifest.json")["sha256"]);print("ALL_RELEASE_SHA_PASS")'
        print(command(c,'cd '+q(release(host))+' && '+q(python(host))+' -c '+q(check)))
        if host=='maty':
            script='''set -euo pipefail
source /etc/profile.d/modules.sh
module load compilers/gcc/v12.2.0
module load tools/cmake/v3.25.2
export Geant4_DIR=/apps/soft/geant/geant4-v11.1.0/lib64/cmake/Geant4
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
cmake -S "$1/Geant4Code_EHE" -B "$1/build" -DWITH_GEANT4_UIVIS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$1/build" --parallel 4
python3 -c 'import hashlib,json,pathlib,sys;p=pathlib.Path(sys.argv[1]);b=p/"build/ehe_spect";(p/"binary_manifest.json").write_text(json.dumps({"sha256":hashlib.sha256(b.read_bytes()).hexdigest()}))' "$1"
'''
            with c.open_sftp() as s:
                with s.open(release(host)+'/build_geant4.sh','w') as f:f.write(script)
            print(command(c,'bash '+q(release(host)+'/build_geant4.sh')+' '+q(release(host))+' > '+q(base(host)+'/logs/build_geant4.log')+' 2>&1 && tail -n 4 '+q(base(host)+'/logs/build_geant4.log'),600))
            if 'src/ehe_G4MultiUnion_11_1.cc' in (payload/'Geant4Code_EHE/CMakeLists.txt').read_text():
                symbols=command(c,'nm -C '+q(release(host)+'/build/ehe_spect'))
                required=('G4MultiUnion::InsideWithExclusion','G4MultiUnion::InsideNoVoxels','G4MultiUnion::G4MultiUnion')
                for name in required:
                    if not any(' T '+name in line for line in symbols.splitlines()):raise ValueError('Version-pinned local MultiUnion implementation was not linked: '+name)
                write(REPORT/'transport_link_acceptance.json',dict(passed=True,release_key=r['release_key'],symbols=[l.strip() for l in symbols.splitlines() if any(' T '+n in l for n in required)]))
    write(REPORT/f'deployment_{host}.json',dict(passed=True,release_key=r['release_key'],root=release(host),files_verified=len(r['sha256'])))

def query(c,job):
    return command(c,'squeue -h -j '+q(job)+' -o "%i|%T|%M|%R"; sacct -n -P -j '+q(job)+' --format=JobID,State,ExitCode,MaxRSS,Elapsed,AllocTRES',90)

def completed(c,job):
    text=query(c,job)
    from ehe_slurm_status import stage_completed
    diagnostic_steps={}
    path=REPORT/f'diagnostic_step_{job}_acceptance.json'
    if path.exists():
        proof=read(path)
        if not proof['passed'] or str(proof['job'])!=str(job) or proof['kind']!='read_only_diagnostic_failure':
            raise ValueError('Invalid auxiliary diagnostic evidence')
        verify_files(REPORT,proof['files'])
        diagnostic_steps=proof['failed_auxiliary_steps']
    return stage_completed(text,job,diagnostic_steps),text

def submit(host,stage,script,minutes,array=None):
    path=REPORT/(stage+'_job.json')
    if path.exists():return read(path)
    name='ehe5e9_'+stage
    with connection(host) as c:
        active=command(c,'squeue -h -u "$USER" -o "%i|%j|%T"')
        if any(name==line.split('|')[1] for line in active.splitlines() if '|' in line):raise ValueError('Same-name active job exists; do not duplicate')
        if host=='gpu' and len(active.strip().splitlines())>=50:raise RuntimeError('Account 50 job limit; wait without cancelling other jobs')
        shell=base(host)+'/scripts/'+stage+'.sh';command(c,'mkdir -p '+q(base(host)+'/scripts'))
        with c.open_sftp() as s:
            with s.open(shell,'w') as f:f.write('#!/bin/bash\nset -euo pipefail\n'+script)
        flags=('--partition=cnmix -N 1 -n 1 -c 1 ' if host=='maty' else '--partition=gpu_5090 --account=scxi717 --qos=gpugpu -N 1 -n 1 -c 6 --gres=gpu:1 --exclude=wqd10nba06g6 ')
        cmd='sbatch --parsable '+flags+'--job-name='+name+' --time='+str(minutes)+' --output='+q(base(host)+'/logs/'+stage+'_%A_%a.log')
        if array:cmd+=' --array='+array
        job=command(c,cmd+' '+q(shell)).strip().split(';')[0]
        if not job.isdigit():raise ValueError('Invalid scheduler registration')
        record=dict(job=int(job),host=host,stage=stage,release_key=frozen(host)['release_key'],minutes=minutes,array=array,script=script,submitted_epoch=time.time(),workflow_sha256=digest(__file__))
        write(path,record);print('EHE_JOB_SUBMITTED',stage,job)
        return record

def env_cpu():return 'source /etc/profile.d/modules.sh\nmodule load compilers/gcc/v12.2.0\nsource /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh\n'
def env_gpu():return 'source /etc/profile.d/modules.sh\nmodule load cuda/12.9\nexport OMP_NUM_THREADS=6\n'

def pilots():
    for host in ('maty','gpu'):
        record=REPORT/f'deployment_{host}.json'
        if not record.exists() or read(record)['release_key']!=frozen(host)['release_key']:deploy(host)
    r=release('maty');b=base('maty')
    output=b+'/pilot_transport_'+frozen('maty')['release_key']
    script=env_cpu()+f'cd {q(r)}\npython3 run_ehe_worker.py --release {q(r)} --simulation {q(b+"/simulation")} --output {q(output)} --index 0 --pilot --photons 100000 --limit 1800\n'
    for response in RESPONSES:
        script+=f'python3 Geant4Code_EHE/validate_against_params.py --params-dir {q(r+"/params/"+response)} --geant-output-dir {q(output)} --tolerance-mm 0.0001 --output-dir {q(b+"/geometry_evidence/"+response)}\n'
    registration=submit('maty','transport_pilot',script,35)
    if 'output' not in registration:registration['output']=output;write(REPORT/'transport_pilot_job.json',registration)
    r=release('gpu');b=base('gpu')
    script=env_gpu()+f'cd {q(r)}\nmkdir -p bin\n'
    script+='nvcc -std=c++17 -O3 -lineinfo -arch=sm_89 engine/PEGen_RayTracing_CircularHole/PEGen_V4_Production.cu -o bin/PEGen_V4_Production\n'
    script+='nvcc -std=c++17 -O3 -lineinfo -arch=sm_89 engine/ScatterGen_RayTracing_CircularHole/scatter.cu engine/ScatterGen_RayTracing_CircularHole/ScatterGen_CircularHole.cpp -o bin/ScatterGen_CircularHole\n'
    if (REPORT/'response_repair_freeze.json').exists():
        script+='nvcc -std=c++17 -O3 -lineinfo -arch=sm_89 ehe_pe_chord_test.cu -o bin/ehe_pe_chord_test\n'
        script+=q(GPU_PYTHON)+f' validate_ehe_pe_v4.py --binary {q(r+"/bin/ehe_pe_chord_test")} --params {q(r+"/params/A440")} --output {q(b+"/pe_geometry_"+frozen("gpu")["release_key"])}\n'
    code='from ehe_common import *;write("gpu_binary_manifest.json",{"files":{n:digest(n) for n in ("bin/PEGen_V4_Production","bin/ScatterGen_CircularHole")}})'
    script+=q(GPU_PYTHON)+' -c '+q(code)+'\n'
    output=b+'/pilot_responses_'+frozen('gpu')['release_key']
    script+=q(GPU_PYTHON)+f' ehe_gpu_pipeline.py pilot --release {q(r)} --output {q(output)} --limit 3600\n'
    registration=submit('gpu','response_pilot',script,190)
    if 'output' not in registration:registration['output']=output;write(REPORT/'response_pilot_job.json',registration)

def repair_transport_pilot():
    """One bounded startup repair, with old job/source/output kept as evidence."""
    registration=REPORT/'transport_pilot_job.json';old=read(registration)
    with connection('maty') as c:
        state=query(c,old['job'])
        if not any(line.split('|')[:2]==[str(old['job']),'FAILED'] for line in state.splitlines()):raise ValueError('Failed pilot must fully exit before repair')
        if str(old['job']) in command(c,'squeue -h -u "$USER" -o %i').split():raise ValueError('Old pilot still active')
        log=command(c,'cat '+q(base('maty')+'/logs/transport_pilot_'+str(old['job'])+'_4294967294.log'))
        if 'Actual scontrol AllocTRES required' not in log:raise ValueError('This repair is only for the diagnosed CPU memory-accounting startup failure')
        write(REPORT/f'failed_transport_pilot_{old["job"]}_evidence.json',dict(accounting=state,log=log,zero_transport_started=True,
            reason='CPU partition UNLIMITED omits real Slurm memory TRES; GPU imaging policy stays strict'))
    import shutil,hashlib,json
    payload=DATA/'transport_payload';shutil.copytree(DATA/'payload',payload)
    for name in ('ehe_common.py','run_ehe_worker.py','ehe_5e9_workflow.py'):shutil.copy2(HERE/name,payload/name)
    files=hashes(payload);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    write(REPORT/'transport_repair_freeze.json',dict(study=STUDY,release_key=key,sha256=files,config_sha256=digest(payload/'config.json'),
        prior_pilot_job=old['job'],gpu_release_unchanged=frozen('gpu')['release_key']))
    registration.rename(REPORT/f'failed_transport_pilot_{old["job"]}_job.json')
    deploy('maty');pilots()

def repair_response_pilot():
    registration=REPORT/'response_pilot_job.json';old=read(registration)
    with connection('gpu') as c:
        state=query(c,old['job'])
        if not any(line.split('|')[:2]==[str(old['job']),'FAILED'] for line in state.splitlines()):raise ValueError('Failed response pilot must fully exit before repair')
        if str(old['job']) in command(c,'squeue -h -u "$USER" -o %i').split():raise ValueError('Old response still active')
        log=command(c,'cat '+q(base('gpu')+'/pilot_responses/A218/slab_0/pe.log'))
        if 'supports zero-hole collimators only' not in log:raise ValueError('This repair is only for the diagnosed EHE aperture feature gap')
        write(REPORT/f'failed_response_pilot_{old["job"]}_evidence.json',dict(accounting=state,log=log,complete_response_stages=0,
            reason='PE-v4 currently rejects finite holes; independent EHE finite-cylinder extension, baseline untouched'))
    import shutil,hashlib,json
    from ehe_pe_v4_overlay import overlay
    payload=DATA/'response_payload';shutil.copytree(DATA/'payload',payload)
    for name in ('ehe_common.py','ehe_5e9_workflow.py','ehe_pe_v4_overlay.py','ehe_pe_chord_test.cu','validate_ehe_pe_v4.py'):shutil.copy2(HERE/name,payload/name)
    source=ENGINE/'PEGen_RayTracing_CircularHole/PEGen_V4_Production.cu'
    overlay(source,payload/'engine/PEGen_RayTracing_CircularHole/PEGen_V4_Production.cu')
    files=hashes(payload);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    write(REPORT/'response_repair_freeze.json',dict(study=STUDY,release_key=key,sha256=files,config_sha256=digest(payload/'config.json'),
        prior_pilot_job=old['job'],transport_release_unchanged=frozen('maty')['release_key']))
    registration.rename(REPORT/f'failed_response_pilot_{old["job"]}_job.json')
    deploy('gpu');pilots()

def repair_geometry_pilot():
    registration=REPORT/'transport_pilot_job.json';old=read(registration)
    evidence=read(REPORT/f'stalled_geometry_{old["job"]}.json')
    if 'G4PVPlacement::CheckOverlaps' not in evidence['stack'] or not evidence['zero_transport_started']:
        raise ValueError('Diagnosed geometry-only startup evidence required')
    with connection('maty') as c:
        state=query(c,old['job'])
        if str(old['job']) in command(c,'squeue -h -u "$USER" -o %i').split():raise ValueError('Old own pilot must fully exit')
        if not any(line.split('|')[0]==str(old['job']) and line.split('|')[1].startswith('CANCELLED') for line in state.splitlines()):raise ValueError('Own stalled pilot exit not recorded')
        evidence['accounting']=state;write(REPORT/f'stalled_geometry_{old["job"]}.json',evidence)
    import shutil,hashlib,json
    from prepare_ehe_5e9 import bounded_geometry_init
    prior=frozen('maty');write(REPORT/f'transport_freeze_{prior["release_key"]}.json',prior)
    payload=DATA/'transport_payload_v2';shutil.copytree(DATA/prior.get('payload_dir','transport_payload'),payload)
    bounded_geometry_init(payload/'Geant4Code_EHE')
    for name in ('run_ehe_worker.py','ehe_5e9_workflow.py'):shutil.copy2(HERE/name,payload/name)
    files=hashes(payload);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    write(REPORT/'transport_repair_freeze.json',dict(study=STUDY,release_key=key,sha256=files,payload_dir=payload.name,
        config_sha256=digest(payload/'config.json'),prior_pilot_job=old['job'],prior_release_key=prior['release_key'],
        repair='Disable redundant random Boolean overlap sampling; retain exhaustive analytical geometry and per-element Params audit; separate beam timing'))
    registration.rename(REPORT/f'failed_transport_pilot_{old["job"]}_job.json')
    deploy('maty');pilots()

def repair_chord_pilot():
    registration=REPORT/'response_pilot_job.json';old=read(registration)
    with connection('gpu') as c:
        state=query(c,old['job'])
        if not any(line.split('|')[:2]==[str(old['job']),'FAILED'] for line in state.splitlines()):raise ValueError('Failed numerical pilot must fully exit')
        if str(old['job']) in command(c,'squeue -h -u "$USER" -o %i').split():raise ValueError('Old numerical pilot still active')
        numerical=get_json(c,base('gpu')+'/pe_geometry_'+old['release_key']+'/geometry_numerical.json')
        if numerical['passed']:raise ValueError('Only diagnosed failed aperture check may be repaired')
        write(REPORT/f'failed_response_pilot_{old["job"]}_evidence.json',dict(accounting=state,numerical=numerical,
            complete_response_stages=0,repair='CUDA device-only double interval comparisons replace unsupported host constexpr mixed-type min/max; all-hole oracle gate stays unchanged'))
    import shutil,hashlib,json
    from ehe_pe_v4_overlay import overlay
    prior=frozen('gpu');write(REPORT/f'response_freeze_{prior["release_key"]}.json',prior)
    payload=DATA/'response_payload_v2';shutil.copytree(DATA/prior.get('payload_dir','response_payload'),payload)
    for name in ('ehe_5e9_workflow.py','ehe_pe_v4_overlay.py','ehe_pe_chord_test.cu','validate_ehe_pe_v4.py','compare_ehe_5e9.py'):shutil.copy2(HERE/name,payload/name)
    overlay(ENGINE/'PEGen_RayTracing_CircularHole/PEGen_V4_Production.cu',payload/'engine/PEGen_RayTracing_CircularHole/PEGen_V4_Production.cu')
    files=hashes(payload);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    write(REPORT/'response_repair_freeze.json',dict(study=STUDY,release_key=key,sha256=files,payload_dir=payload.name,
        config_sha256=digest(payload/'config.json'),prior_pilot_job=old['job'],prior_release_key=prior['release_key'],
        repair='Native device double interval arithmetic; exact same 5798-ray exhaustive float64 aperture gate'))
    registration.rename(REPORT/f'failed_response_pilot_{old["job"]}_job.json')
    deploy('gpu');pilots()

def repair_union_pilot():
    registration=REPORT/'transport_pilot_job.json';old=read(registration)
    evidence=read(REPORT/f'stalled_transport_{old["job"]}.json')
    if 'G4MultiUnion::InsideWithExclusion' not in evidence['stack'] or not evidence['transport_not_completed']:
        raise ValueError('Actual diagnosed union-navigation failure required')
    with connection('maty') as c:
        state=query(c,old['job'])
        if str(old['job']) in command(c,'squeue -h -u "$USER" -o %i').split():raise ValueError('Old own pilot still active')
        if not any(l.split('|')[0]==str(old['job']) and l.split('|')[1].startswith('CANCELLED') for l in state.splitlines()):raise ValueError('Own stalled pilot exit not recorded')
        evidence['accounting']=state;write(REPORT/f'stalled_transport_{old["job"]}.json',evidence)
    import shutil,hashlib,json
    from prepare_ehe_5e9 import safe_multiunion
    prior=frozen('maty');write(REPORT/f'transport_freeze_{prior["release_key"]}.json',prior)
    payload=DATA/'transport_payload_v3';shutil.copytree(DATA/prior.get('payload_dir','transport_payload'),payload)
    safe_multiunion(payload/'Geant4Code_EHE')
    for name in ('run_ehe_worker.py','ehe_5e9_workflow.py'):shutil.copy2(HERE/name,payload/name)
    files=hashes(payload);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    write(REPORT/'transport_repair_freeze.json',dict(study=STUDY,release_key=key,sha256=files,payload_dir=payload.name,
        config_sha256=digest(payload/'config.json'),prior_pilot_job=old['job'],prior_release_key=prior['release_key'],
        upstream_source_sha256='98f517c04c37edaa29d9ac5747f1ebf41e545d20ef0cdacb9754942b0f5e529d',
        repair='Geant4 11.1.0 executable-local MultiUnion empty surface-list bound guard; dimensions/materials/physics unchanged; 11252 classification checks versus exhaustive oracle'))
    registration.rename(REPORT/f'failed_transport_pilot_{old["job"]}_job.json')
    deploy('maty');pilots()

def repair_union_link_pilot():
    registration=REPORT/'transport_pilot_job.json';old=read(registration)
    evidence=read(REPORT/f'unlinked_union_{old["job"]}.json')
    if not any('libG4geometry.so' in p['stack'] and 'DetectorConstruction::Construct' in p['stack'] for p in evidence['processes']):raise ValueError('Actual unlinked geometry stack required')
    if ' T G4MultiUnion::' in evidence['linked_symbols']:raise ValueError('This repair only applies to absent executable-local symbols')
    with connection('maty') as c:
        state=query(c,old['job'])
        if str(old['job']) in command(c,'squeue -h -u "$USER" -o %i').split():raise ValueError('Old own pilot must fully exit before link repair')
        if not any(l.split('|')[0]==str(old['job']) and l.split('|')[1].startswith('CANCELLED') for l in state.splitlines()):raise ValueError('Own diagnosed stalled pilot exit required')
        evidence['accounting']=state;write(REPORT/f'unlinked_union_{old["job"]}.json',evidence)
    import shutil,hashlib,json
    from prepare_ehe_5e9 import link_multiunion
    prior=frozen('maty');write(REPORT/f'transport_freeze_{prior["release_key"]}.json',prior)
    payload=DATA/'transport_payload_v4';shutil.copytree(DATA/prior['payload_dir'],payload)
    link_multiunion(payload/'Geant4Code_EHE')
    shutil.copy2(HERE/'ehe_GEANT4_LICENSE.txt',payload/'Geant4Code_EHE/LICENSE.Geant4')
    for name in ('prepare_ehe_5e9.py','test_ehe_5e9.py','ehe_5e9_workflow.py'):shutil.copy2(HERE/name,payload/name)
    files=hashes(payload);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    write(REPORT/'transport_repair_freeze.json',dict(study=STUDY,release_key=key,sha256=files,payload_dir=payload.name,
        config_sha256=digest(payload/'config.json'),prior_pilot_job=old['job'],prior_release_key=prior['release_key'],
        repair='Add already version-pinned local MultiUnion source to explicit CMake target; require linked text symbols before pilot; geometry/physics/sampler unchanged'))
    registration.rename(REPORT/f'failed_transport_pilot_{old["job"]}_job.json')
    deploy('maty');pilots()

def get_json(c,path):
    with c.open_sftp() as s:
        with s.open(path,'r') as f:
            import json
            return json.load(f)

def gpu_stage_accounting(c,stage,output):
    import re
    registration=read(REPORT/(stage+'_job.json'));done,text=completed(c,registration['job'])
    if not done:raise ValueError('Actual successful stage exit required')
    try:alloc=get_json(c,output+'/allocation.json')
    except FileNotFoundError:
        primary=[l.split('|') for l in text.splitlines() if l.split('|')[:3]==[str(registration['job']),'COMPLETED','0:0']]
        if len(primary)!=1:raise ValueError('Actual completed allocation identity required')
        alloc=dict(job=registration['job'],host_allocated_bytes=allocated_bytes('AllocTRES='+primary[0][-1]),
            imaging_allocation_certificate=True,denominator_kind='actual completed sacct AllocTRES',accounting=text)
    if not alloc['imaging_allocation_certificate'] or str(alloc['job'])!=str(registration['job']):raise ValueError('Actual GPU allocation identity differs')
    peak=0
    for line in text.splitlines():
        fields=line.split('|')
        if len(fields)<4:continue
        m=re.fullmatch(r'([0-9.]+)([KMGT]?)',fields[3])
        if m:peak=max(peak,int(float(m[1])*1024**(' KMGT'.index(m[2]) if m[2] else 0)))
    if peak<=0 or peak>.8*alloc['host_allocated_bytes']:raise ValueError('Actual Slurm MaxRSS margin unavailable/failed')
    proof=dict(passed=True,job=registration['job'],allocation=alloc,slurm_maxrss_bytes=peak,
        slurm_maxrss_fraction=peak/alloc['host_allocated_bytes'],accounting=text)
    write(REPORT/(stage+'_resource_acceptance.json'),proof)
    return proof

def collect(c):
    """Aggregate only actual 200 independent EHE worker observations, preserving all receipts."""
    local=DATA/'transport';local.mkdir(exist_ok=True);records=[]
    import numpy as np
    from run_ehe_worker import FILES,validate
    jobs=read(DATA/'simulation/jobs.json')['jobs']
    with c.open_sftp() as s:
        for j in jobs:
            folder=local/f'worker_{j["index"]:03d}';folder.mkdir(exist_ok=True);remote=base('maty')+'/transport/'+folder.name
            for name in FILES+['receipt.json','TransportSummary.json','TransportTiming.json','EHE_MultiUnionAudit.json','source.mac','allocation.json']:
                s.get(remote+'/'+name,str(folder/name))
            receipt=read(folder/'receipt.json');verify_files(folder,receipt['files'])
            if not receipt['passed'] or receipt['pilot'] or receipt['index']!=j['index'] or receipt['seed']!=j['seed'] or receipt['registered_macro_sha256']!=j['macro_sha256'] or receipt['release_key']!=frozen('maty')['release_key']:raise ValueError('Independent actual transport differs')
            validate(folder,25_000_000);records.append(receipt)
    if len({r['seed'] for r in records})!=200 or sum(r['photons'] for r in records)!=5_000_000_000:raise ValueError('Actual seed/dose identity failed')
    arrays={name[:-4]:np.stack([np.loadtxt(local/f'worker_{i:03d}'/name,delimiter=',',dtype=np.int64) for i in range(200)]) for name in FILES}
    arrays['primary_counts']=np.array([r['primary_counts'] for r in records],np.int64)
    # Preserve the original registry/macros alongside the actual normalized
    # Geant4 macros, so acceptance can prove content across CRLF/LF platforms.
    import shutil
    registry=local/'source_registry'
    if not registry.exists():shutil.copytree(DATA/'simulation',registry)
    if hashes(registry)!=hashes(DATA/'simulation'):raise ValueError('Frozen source registry changed during collection')
    np.savez_compressed(local/'worker_counts.npz',**arrays)
    for e in (218,440):np.save(local/f'projection_{e}.npy',arrays[f'CntStat_{e}'].reshape(20,10,2312).sum(axis=1).T)
    primary=arrays['primary_counts'].sum(axis=0).tolist()
    expected=read(DATA/'simulation/jobs.json')['expected_primary_energy_fraction']['218'];observed=primary[0]/sum(primary)
    if abs(observed-expected)>5*math.sqrt(expected*(1-expected)/sum(primary)):raise ValueError('HOLD: actual source energy mixture outside 5-sigma binomial bound')
    record=dict(passed=True,total_primary_photons=sum(primary),primary_counts=primary,workers=200,views=20,
        seeds=[r['seed'] for r in records],receipts=records,
        files={p.relative_to(local).as_posix():digest(p) for p in local.rglob('*') if p.is_file() and p.name!='collection.json'})
    write(local/'collection.json',record)
    with connection('gpu') as gpu:
        command(gpu,'mkdir -p '+q(base('gpu')+'/counts'))
        with gpu.open_sftp() as s:put_tree(s,local,base('gpu')+'/counts')
    write(REPORT/'transport_acceptance.json',dict(passed=True,primary_counts=primary,workers=200,views=20,actual_218_primary_fraction=observed,expected_218_primary_fraction=expected,
        collection_sha256=digest(local/'collection.json'),seed_first=records[0]['seed'],seed_last=records[-1]['seed']))

def advance():
    if (REPORT/'reconstruction_execution_freeze.json').exists():
        from ehe_reconstruction_workflow import advance_reconstruction
        return advance_reconstruction()
    if not (REPORT/'transport_pilot_job.json').exists():pilots();return
    with connection('maty') as c:
        pilot=read(REPORT/'transport_pilot_job.json');done,_=completed(c,pilot['job'])
        if done and not (REPORT/'transport_job.json').exists():
            proof=get_json(c,pilot.get('output',base('maty')+'/pilot_transport')+'/receipt.json')
            if not proof['passed'] or not proof['pilot'] or proof['photons']!=100000 or proof['release_key']!=frozen('maty')['release_key']:raise ValueError('Actual transport throughput required')
            code=f'''from pathlib import Path
from ehe_common import *
r=Path({release('maty')!r});p=Path({pilot['output']!r})
receipt=read(p/'receipt.json')
verify_files(r,read(r/'release_manifest.json')['sha256'])
verify_files(p,receipt['files'])
verify_files(r,{{'build/ehe_spect':receipt['binary_sha256']}})
audit=read(p/'EHE_MultiUnionAudit.json')
assert audit['passed'] and audit['points']==11252 and audit['holes']==1250
print('ACTUAL_TRANSPORT_PROBE_SHA_PASS')
'''
            print(command(c,'cd '+q(release('maty'))+' && python3 -c '+q(code),120))
            write(REPORT/'transport_pilot.json',proof)
            geometry_sha=fetch_transport_geometry(c,pilot)
            write(REPORT/'transport_pilot_sha_acceptance.json',dict(passed=True,job=pilot['job'],release_key=proof['release_key'],
                receipt_files=proof['files'],geometry_evidence_sha256=geometry_sha,remote_and_fetched_geometry_sha_match=True))
            timing=proof['phase_seconds']
            limit=max(600,math.ceil((timing['initialization_seconds']+timing['beam_seconds']*250)*1.8+300))
            r=release('maty');b=base('maty')
            script=env_cpu()+f'cd {q(r)}\npython3 run_ehe_worker.py --release {q(r)} --simulation {q(b+"/simulation")} --output {q(b+"/transport/worker_")}'+'$(printf "%03d" "$SLURM_ARRAY_TASK_ID")'+f' --limit {limit}\n'
            submit('maty','transport',script,math.ceil(limit/60)+5,'0-199%40')
        if (REPORT/'transport_job.json').exists() and not (REPORT/'transport_acceptance.json').exists():
            done,_=completed(c,read(REPORT/'transport_job.json')['job'])
            if done:collect(c)
    with connection('gpu') as c:
        pilot=read(REPORT/'response_pilot_job.json');done,_=completed(c,pilot['job'])
        if done and not (REPORT/'response_job.json').exists():
            proof=get_json(c,pilot.get('output',base('gpu')+'/pilot_responses')+'/response_summary.json')
            if not proof['passed'] or not proof['pilot']:raise ValueError('Actual response pilot required')
            write(REPORT/'response_pilot.json',proof)
            gpu_stage_accounting(c,'response_pilot',pilot['output'])
            numerical=get_json(c,base('gpu')+'/pe_geometry_'+pilot['release_key']+'/geometry_numerical.json')
            if not numerical['passed'] or numerical['holes']!=1250 or numerical['rays']!=5798:raise ValueError('Actual finite-aperture GPU gate required')
            write(REPORT/'pe_geometry_acceptance.json',numerical)
            worst=max(x['pe_resource']['elapsed_seconds']+x['scatter_resource']['elapsed_seconds'] for x in proof['stages'].values())
            limit=max(600,math.ceil(worst*10*1.8+300));r=release('gpu');b=base('gpu')
            script=env_gpu()+f'cd {q(r)}\n'+q(GPU_PYTHON)+f' ehe_gpu_pipeline.py responses --release {q(r)} --output {q(b+"/responses")} --limit {limit}\n'
            submit('gpu','response',script,math.ceil(limit*12/60)+30)
        if not (REPORT/'transport_acceptance.json').exists() or not (REPORT/'response_job.json').exists():return
        from ehe_conversion_workflow import repair_binding,advance_conversion,response_root
        if repair_binding() is not None:
            if not advance_conversion(c):return
        else:
            done,_=completed(c,read(REPORT/'response_job.json')['job'])
            if not done:return
            if not (REPORT/'response_resource_acceptance.json').exists():gpu_stage_accounting(c,'response',base('gpu')+'/responses')
        if not (REPORT/'physical_job.json').exists():
            r=release('gpu');b=base('gpu');script=env_gpu()+f'cd {q(r)}\n'+q(GPU_PYTHON)+f' ehe_gpu_pipeline.py physical --release {q(r)} --responses {q(response_root())} --counts {q(b+"/counts")} --output {q(b+"/physical")}\n'
            submit('gpu','physical',script,120);return
        done,_=completed(c,read(REPORT/'physical_job.json')['job'])
        if not done:return
        if not (REPORT/'physical_resource_acceptance.json').exists():gpu_stage_accounting(c,'physical',base('gpu')+'/physical')
        gate=get_json(c,base('gpu')+'/physical/physical_gate.json')
        write(REPORT/'physical_gate.json',gate)
        if not gate['passed']:raise ValueError('Physical HOLD: inspect evidence; never relax thresholds')
        if not (REPORT/'validation_job.json').exists():submit_reconstruction('validation',1800,90);return
        done,_=completed(c,read(REPORT/'validation_job.json')['job'])
        if not done:return
    if not (REPORT/'validation_summary.json').exists():fetch('validation')
    if not (REPORT/'formal_job.json').exists():
        proof=read(REPORT/'validation_summary.json');limit=math.ceil(max(proof['phase_seconds'].values())*20*1.8+300)
        submit_reconstruction('formal',limit,math.ceil(limit*2/60)+15);return
    with connection('gpu') as c:done,_=completed(c,read(REPORT/'formal_job.json')['job'])
    if done and not (REPORT/'formal_summary.json').exists():fetch('formal')

def submit_reconstruction(mode,limit,minutes):
    from ehe_conversion_workflow import response_root
    from ehe_reconstruction_workflow import execution_release,policy_argument,binding
    b=base('gpu');r=execution_release();iterations=10 if mode=='validation' else 200
    script=env_gpu()+f'cd {q(r)}\n'+q(GPU_PYTHON)+f' run_ehe_reconstruction.py --release {q(r)} --responses {q(response_root())} --counts {q(b+"/counts")} --physical {q(b+"/physical")} --output {q(b+"/results/"+mode)} --mode {mode} --iterations {iterations} --limit {limit}'
    if mode=='formal':script+=' --authority '+q(b+'/validation_authority.json')
    script+=policy_argument()
    record=submit('gpu',mode,script+'\n',minutes)
    execution=binding()
    if execution is not None:
        record.update(release_key=execution['release_key'],producer_release_key=execution['producer_release_key'],
                      physical_policy_sha256=execution['policy_sha256'],physical_calibration_passed=False)
        write(REPORT/(mode+'_job.json'),record)

def fetch(mode):
    from ehe_conversion_workflow import response_root
    from ehe_reconstruction_workflow import execution_release,policy_argument
    job=read(REPORT/(mode+'_job.json'))['job'];b=base('gpu');r=execution_release()
    with connection('gpu') as c:
        done,accounting=completed(c,job)
        if not done:raise ValueError('Stage must fully exit before strict fetch')
        prior=REPORT/(mode+'_acceptance_job.json')
        if prior.exists():
            done,_=completed(c,read(prior)['job'])
            if not done:return False
        # Freeze a separate read-only acceptance bundle. Never overwrite sources
        # in the release used by the running or completed simulation/response job.
        import hashlib,json,shutil
        check_sources={n:digest(HERE/n) for n in ('verify_ehe.py','ehe_common.py','ehe_execution_policy.py','reconstruction_output_policy.py')}
        sources=dict(check_sources,**{'ehe_acceptance_driver.py':digest(HERE/'ehe_acceptance_driver.py')})
        key=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:16]
        audit=DATA/'verification_releases'/key;audit.mkdir(parents=True,exist_ok=True)
        for name in sources:
            destination=audit/name
            if destination.exists() and digest(destination)!=sources[name]:raise ValueError('Immutable verifier changed')
            if not destination.exists():shutil.copy2(HERE/name,destination)
        manifest=dict(release_key=key,files=sources,purpose='Read-only complete output/operator acceptance; execution release unchanged')
        manifest_path=audit/'verification_manifest.json'
        if manifest_path.exists() and read(manifest_path)!=manifest:raise ValueError('Verifier manifest changed')
        if not manifest_path.exists():write(manifest_path,manifest)
        audit_remote=b+'/verification_releases/'+key
        command(c,'mkdir -p '+q(b+'/verification_releases'))
        with c.open_sftp() as s:
            try:s.stat(audit_remote+'/verification_manifest.json')
            except FileNotFoundError:put_tree(s,audit,audit_remote)
            else:
                if get_json(c,audit_remote+'/verification_manifest.json')!=manifest:raise ValueError('Remote immutable verifier manifest changed')
        write(REPORT/'verification_freeze.json',dict(**manifest,manifest_sha256=digest(manifest_path)))
        accounting_path=b+'/results/'+mode+'/accounting.txt'
        with c.open_sftp() as s:
            with s.open(accounting_path,'w') as f:f.write(accounting)
        verify=q(GPU_PYTHON)+f' -u ehe_acceptance_driver.py --verification-manifest {q(audit_remote+"/verification_manifest.json")} --result {q(b+"/results/"+mode)} --release {q(r)} --responses {q(response_root())} --counts {q(b+"/counts")} --physical {q(b+"/physical")} --accounting {q(accounting_path)} --output {q(b+"/"+mode+"_authority.json")} --evidence {q(b+"/read_only_acceptance_"+mode+"_"+key)}'
        verify+=policy_argument()
        registration=submit('gpu',mode+'_acceptance',env_gpu()+'cd '+q(audit_remote)+'\n'+verify+'\n',45)
        binding=dict(verification_release_key=key,verification_source_sha256=sources,compute_job=job)
        if any(k in registration and registration[k]!=v for k,v in binding.items()):raise ValueError('Registered acceptance source/result identity differs')
        registration.update(binding)
        write(REPORT/(mode+'_acceptance_job.json'),registration)
        done,acceptance_accounting=completed(c,registration['job'])
        if not done:return False
        print(command(c,'tail -n 8 '+q(b+'/logs/'+mode+'_acceptance_'+str(registration['job'])+'_4294967294.log')))
        receipt=get_json(c,b+'/read_only_acceptance_'+mode+'_'+key+'/receipt.json')
        if not receipt['passed'] or receipt['driver_sha256']!=sources['ehe_acceptance_driver.py']:raise ValueError('Read-only allocation receipt differs')
        write(REPORT/(mode+'_acceptance_receipt.json'),dict(**receipt,accounting=acceptance_accounting))
        proof=get_json(c,b+'/'+mode+'_authority.json');local=DATA/'results'/mode;local.mkdir(parents=True,exist_ok=True)
        if proof['verification_source_sha256']!=check_sources or proof['verification_manifest_sha256']!=digest(manifest_path) or proof['acceptance_driver_sha256']!=sources['ehe_acceptance_driver.py']:raise ValueError('Actual read-only verifier identity differs')
        with c.open_sftp() as s:
            for name,sha in proof['files'].items():
                path=local/name;path.parent.mkdir(parents=True,exist_ok=True);s.get(b+'/results/'+mode+'/'+name,str(path))
                if digest(path)!=sha:raise ValueError('Fetched formal byte identity differs: '+name)
            for name in ('physical_gate.json','physical_audit.csv'):
                dest=DATA/'physical'/name;dest.parent.mkdir(exist_ok=True);s.get(b+'/physical/'+name,str(dest))
                expected=proof['physical_gate_sha256'] if name=='physical_gate.json' else read(DATA/'physical/physical_gate.json')['files'][name]
                if digest(dest)!=expected:raise ValueError('Fetched physical evidence byte identity differs')
            for name in RESPONSES:
                dest=DATA/'factor_evidence'/name;dest.mkdir(parents=True,exist_ok=True)
                s.get(response_root()+'/'+name+'/factor_manifest.json',str(dest/'factor_manifest.json'))
                if digest(dest/'factor_manifest.json')!=proof['factor_sha256'][name]:raise ValueError('Fetched factor manifest identity differs')
                s.get(response_root()+'/'+name+'/S_active.float64',str(dest/'S_active.float64'))
                if digest(dest/'S_active.float64')!=read(dest/'factor_manifest.json')['files']['S_active.float64']:raise ValueError('Fetched sensitivity byte identity differs')
        # Keep original Slurm logs, submitted script bytes and the acceptance
        # allocation separately from the immutable reconstruction outputs.
        evidence=REPORT/(mode+'_'+str(job));evidence.mkdir(parents=True,exist_ok=True)
        remote_files={'compute_slurm.log':b+'/logs/'+mode+'_'+str(job)+'_4294967294.log',
            'acceptance_slurm.log':b+'/logs/'+mode+'_acceptance_'+str(registration['job'])+'_4294967294.log',
            'compute_submitted.sh':b+'/scripts/'+mode+'.sh',
            'acceptance_submitted.sh':b+'/scripts/'+mode+'_acceptance.sh',
            'acceptance_allocation.json':b+'/read_only_acceptance_'+mode+'_'+key+'/allocation.json',
            'acceptance_receipt.json':b+'/read_only_acceptance_'+mode+'_'+key+'/receipt.json'}
        code='from ehe_common import digest;import json;print(json.dumps({n:digest(p) for n,p in '+repr(remote_files)+'.items()}))'
        remote_sha=json.loads(command(c,'cd '+q(audit_remote)+' && '+q(GPU_PYTHON)+' -c '+q(code)))
        with c.open_sftp() as s:
            for name,remote in remote_files.items():
                s.get(remote,str(evidence/name))
                if digest(evidence/name)!=remote_sha[name]:raise ValueError('Exit/launch evidence SHA differs: '+name)
        for prefix,record in [('compute',read(REPORT/(mode+'_job.json'))),('acceptance',registration)]:
            expected=('#!/bin/bash\nset -euo pipefail\n'+record['script']).encode()
            if (evidence/(prefix+'_submitted.sh')).read_bytes()!=expected:raise ValueError('Actual submitted script differs')
        write(evidence/'fetch_receipt.json',dict(passed=True,compute_job=job,acceptance_job=registration['job'],
            files=remote_sha,compute_accounting=accounting,acceptance_accounting=acceptance_accounting,
            output_manifest_sha256=digest(local/'run_manifest.json'),physical_calibration_passed=proof['physical_calibration_passed']))
        write(REPORT/(mode+'_summary.json'),proof)
        print('EHE_STRICT_FETCH_PASS',mode,job)
        return True

def status():
    for path in sorted(REPORT.glob('*_job.json')):
        if path.name.startswith('failed_'):continue
        record=read(path)
        if record['stage']=='response_debug':continue  # exited diagnostic; preserved in the runbook
        with connection(record['host']) as c:
            print(record['stage'],query(c,record['job']).strip())
            if record['host']=='gpu':
                print(command(c,'sstat -n -P -j '+q(str(record['job'])+'.batch')+' --format=JobID,MaxRSS,AveRSS 2>/dev/null || true'))
            if record['stage'] in ('response','response_pilot'):
                output=record.get('output',base('gpu')+'/responses')
                code=f'''from pathlib import Path
import json
root=Path({output!r})
for folder in sorted(root.glob('*/slab_*')):
    receipt=folder/'receipt.json';progress=folder/'PE_progress.json'
    if receipt.exists():print('RESPONSE_CHECKPOINT',str(folder.relative_to(root)),'complete')
    elif progress.exists():print('RESPONSE_PROGRESS',str(folder.relative_to(root)),progress.read_text()[-1800:])
'''
                print(command(c,q(GPU_PYTHON)+' -c '+q(code)))
            if record['stage'] in ('validation','formal'):
                print(command(c,'if [ -f '+q(base('gpu')+'/results/'+record['stage']+'/progress.json')+' ]; then cat '+q(base('gpu')+'/results/'+record['stage']+'/progress.json')+'; fi'))
            if record['stage'] in ('response_conversion_probe','response_conversion'):
                output=record['output']
                print(command(c,'if [ -f '+q(output+'/progress.json')+' ]; then cat '+q(output+'/progress.json')+'; fi'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','deploy','pilots','repair-transport-pilot','repair-response-pilot','repair-geometry-pilot','repair-chord-pilot','repair-union-pilot','repair-union-link-pilot','repair-response-conversion','advance','status','fetch','compare']);p.add_argument('--host',choices=['maty','gpu'],default='maty');p.add_argument('--mode',choices=['validation','formal'],default='formal');a=p.parse_args()
    if a.action=='prepare':
        from prepare_ehe_5e9 import prepare
        prepare()
    elif a.action=='deploy':deploy(a.host)
    elif a.action=='pilots':pilots()
    elif a.action=='repair-transport-pilot':repair_transport_pilot()
    elif a.action=='repair-response-pilot':repair_response_pilot()
    elif a.action=='repair-geometry-pilot':repair_geometry_pilot()
    elif a.action=='repair-chord-pilot':repair_chord_pilot()
    elif a.action=='repair-union-pilot':repair_union_pilot()
    elif a.action=='repair-union-link-pilot':repair_union_link_pilot()
    elif a.action=='repair-response-conversion':
        from ehe_conversion_workflow import setup,advance_conversion
        setup()
        with connection('gpu') as c:advance_conversion(c)
    elif a.action=='advance':advance()
    elif a.action=='status':status()
    elif a.action=='fetch':fetch(a.mode)
    else:
        from compare_ehe_5e9 import compare
        compare()
