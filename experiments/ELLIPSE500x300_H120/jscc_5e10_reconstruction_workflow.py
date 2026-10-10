"""Fresh 5e10 pipeline: selection -> validation10 -> authority -> formal10000."""
import hashlib, json, math, os, re, shutil, tarfile, time
from pathlib import Path
from jscc_5e10_common import *
from jscc_5e10_workflow import connection,command,q,put_archive,job_state,accounting,completed

BASELINE=HERE/'generated/compton_energy_probability_v5_5e9_full10000/formal_payload'
FACTORS=GPU_PROJECT+'/generated/FactorsCalibrated'
NEW_SOURCES=('jscc_5e10_common.py','jscc_5e10_contract.py','jscc_5e10_runtime.py',
    'jscc_5e10_selection.py','jscc_5e10_compton_mlem.py','run_jscc_5e10.py',
    'verify_jscc_5e10.py','jscc_5e10_verify_stage.py','test_jscc_5e10.py','jscc_5e10_reconstruction_workflow.py')

def verify_prepared_controller_repair(name, original_sha):
    """Allow only the separately frozen local scheduler controller revision."""
    if name!='jscc_5e10_reconstruction_workflow.py':
        raise ValueError('A scientific source changed; a launch-only repair cannot authorize it: '+name)
    proof=read(REPORT/'launch_control_repair_freeze.json')
    verify_files(DATA/'launch_control_repair_payload',proof['sha256'])
    if (proof['old_controller_sha256']!=original_sha or
        proof['new_controller_sha256']!=digest(HERE/name) or
        proof['science_preparation_sha256']!=digest(REPORT/'science_preparation.json') or
        proof['kernel_freeze_sha256']!=digest(REPORT/'kernel_freeze.json')):
        raise ValueError('Frozen scheduler-only repair identity differs')

def baseline_payload(target):
    cfg=read(BASELINE/'contract.json')
    accepted=read(HERE/'reports/NEMA_Body_H60/compton_energy_probability_v5_5e9_full10000/formal_freeze.json')
    verify_files(BASELINE,accepted['sha256'])
    if read(BASELINE/'calibration/calibration_gate.json')['status']!='PASSED':raise ValueError('Prior matched response/S calibration is not passed')
    for n,sha in cfg['files'].items():
        # Only immutable scientific helpers and basis; no old data, outputs, or launch scripts.
        if n.endswith('.py') or n in ('whole_geometry.npz','config.json','continuous_energy_Sensi_full','transfer_training_summary.json','calibration/calibration_gate.json'):
            if digest(BASELINE/n)!=sha:raise ValueError('Delivered helper identity differs: '+n)
            f=target/n;f.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASELINE/n,f)
    shutil.copy2(BASELINE/'contract.json',target/'baseline_contract.json')
    for n in NEW_SOURCES:shutil.copy2(HERE/n,target/n)
    return cfg

def prepare_science():
    """Check the immutable scientific entry during transport queue/compute time."""
    proof_path=REPORT/'science_preparation.json'
    if proof_path.exists():
        proof=read(proof_path);verify_files(DATA/'science_payload',proof['sha256'])
        for n,sha in proof['new_sources_sha256'].items():
            if digest(HERE/n)!=sha:verify_prepared_controller_repair(n,sha)
        return
    target=DATA/'science_payload';target.mkdir(exist_ok=False);baseline_payload(target)
    import subprocess,sys
    env=dict(os.environ,JSCC_PROJECT_ROOT=str(target))
    tests=subprocess.run([sys.executable,'-X','utf8','-m','unittest','test_jscc_5e10','-v'],cwd=target,env=env,
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120,check=True).stdout
    (REPORT/'local_contract_tests.txt').write_bytes(tests)
    files=hashes(target);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    release=GPU_BASE+'/science_releases/'+key
    with connection('gpu') as c:
        command(c,'mkdir -p '+q(GPU_BASE+'/logs')+' '+q(GPU_BASE+'/allocations'))
        put_archive(c,target,release)
        check='import run_jscc_5e10,jscc_5e10_selection,verify_jscc_5e10;print("NEW_LINUX_ENTRIES_IMPORT_PASSED")'
        result=command(c,'cd '+q(release)+' && '+q(GPU_PYTHON)+' -c '+q(check),timeout=60)
        remote_tests=command(c,'cd '+q(release)+' && JSCC_PROJECT_ROOT='+q(release)+' '+q(GPU_PYTHON)+' -m unittest test_jscc_5e10 -v 2>&1',timeout=120)
        if '\nOK' not in remote_tests or 'skipped' in remote_tests:raise ValueError('Prepared immutable Linux contract tests failed: '+remote_tests)
        (REPORT/'linux_contract_tests.txt').write_bytes(remote_tests.encode())
        # The same launch generator that will be frozen into both real GPU stages.
        local=DATA/'prepared_8x4_validation.sh';local.write_bytes(launcher('validation',release,3600,10800,14220).encode())
        remote=release+'/prepared_8x4_validation.sh'
        with c.open_sftp() as s:s.put(str(local),remote)
        if command(c,'sha256sum '+q(remote)).split()[0]!=digest(local):raise ValueError('Prepared launch bytes differ')
        command(c,'bash -n '+q(remote))
        checks='from pathlib import Path;from jscc_5e10_common import verify_files;verify_files(Path("."),'+repr(files)+');print("SCIENCE_PREPARATION_SHA_PASSED")'
        result+='\n'+command(c,'cd '+q(release)+' && '+q(GPU_PYTHON)+' -c '+q(checks),timeout=120)
    write(proof_path,dict(passed=True,sha256=files,release_key=key,release=release,
        new_sources_sha256={n:digest(HERE/n) for n in NEW_SOURCES},baseline_contract_sha256=digest(BASELINE/'contract.json'),
        local_tests_sha256=digest(REPORT/'local_contract_tests.txt'),linux_tests_sha256=digest(REPORT/'linux_contract_tests.txt'),
        actual_linux_import_and_sha_evidence=result,launch_sha256=digest(local),launch_bash_syntax_passed=True,
        no_gpu_slurm_job_submitted=True,complete_input_gpu_validation_still_required=True))
    print('IMMUTABLE_SCIENCE_AND_8X4_LINUX_ENTRY_PREPARED',key,flush=True)

def freeze_kernel():
    if (REPORT/'kernel_freeze.json').exists():return
    transport=read(REPORT/'transport_acceptance.json')
    if not transport['strict_local_sha_passed']:raise ValueError('Fresh complete transport acceptance required')
    prepare_science();prepared=read(REPORT/'science_preparation.json');verify_files(DATA/'science_payload',prepared['sha256'])
    target=DATA/'kernel_payload';shutil.copytree(DATA/'science_payload',target);old=read(target/'baseline_contract.json')
    shutil.copy2(DATA/'input/collection.json',target/'transport_collection.json')
    files=hashes(target)
    cfg=dict(study=STUDY,files=files,baseline_contract_sha256=digest(BASELINE/'contract.json'),
             factor_payload_sha256=old['factor_payload_sha256'],factor_manifest_sha256=old['factor_manifest_sha256'],
             whole_geometry_sha256=old['whole_geometry_sha256'],calibration_release=old['calibration_release'],
             transport_collection_sha256=digest(DATA/'input/collection.json'),input_sha256=hashes(DATA/'input'),
             source_delivered_job=1669255,nodes=8,gpus_per_node=4,world_size=32,
             event_policy='legacy',total_primary_photons=TOTAL,channels=list(CHANNELS))
    write(target/'kernel_config.json',cfg);files=hashes(target)
    key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    write(REPORT/'kernel_freeze.json',dict(release_key=key,sha256=files,baseline_contract_sha256=cfg['baseline_contract_sha256'],
           total_primary_photons=TOTAL,channels=list(CHANNELS),nodes=8,gpus_per_node=4,world_size=32))

def remote_free_bytes(c):
    n=int(command(c,'df -B1 --output=avail '+q(GPU_PROJECT)+' | tail -n 1'))
    return n

def deploy_kernel():
    if (REPORT/'kernel_deployment.json').exists():return
    f=read(REPORT/'kernel_freeze.json');verify_files(DATA/'kernel_payload',f['sha256'])
    pack=read(DATA/'collection_package.json');archive=DATA/'transport_input.tar.gz'
    if digest(archive)!=pack['sha256']:raise ValueError('Fresh transport package changed')
    # Leave room for raw input, archive, selected rows, snapshots and reports.
    raw_bytes=sum(p.stat().st_size for p in (DATA/'input').rglob('*') if p.is_file())
    reserve=raw_bytes+archive.stat().st_size+(8<<30)
    release=GPU_BASE+'/kernel_releases/'+f['release_key']
    with connection('gpu') as c:
        free=remote_free_bytes(c)
        if free<reserve:raise OSError('New experiment needs '+str(reserve)+' bytes, actual scxi717 free '+str(free)+'. Preserve existing results and diagnose storage.')
        command(c,'mkdir -p '+q(GPU_BASE+'/logs')+' '+q(GPU_BASE+'/allocations'))
        put_archive(c,DATA/'kernel_payload',release)
        # Unregistered partial upload/deployment requires inspection; never overwrite it.
        command(c,'test ! -e '+q(GPU_BASE+'/input')+' && test ! -e '+q(GPU_BASE+'/transport_input.tar.gz'))
        with c.open_sftp() as s:s.put(str(archive),GPU_BASE+'/transport_input.tar.gz')
        if command(c,'sha256sum '+q(GPU_BASE+'/transport_input.tar.gz'),timeout=600).split()[0]!=pack['sha256']:raise ValueError('Fresh full input transfer differs')
        command(c,'mkdir '+q(GPU_BASE+'/input')+' && tar --no-same-owner -xzf '+q(GPU_BASE+'/transport_input.tar.gz')+' -C '+q(GPU_BASE+'/input'),timeout=600)
        test_code='from pathlib import Path;from jscc_5e10_common import *;p=Path(".");verify_files(p,read(p/"kernel_config.json")["files"]);validate_collection(read(p/"transport_collection.json"));import run_jscc_5e10,jscc_5e10_selection,verify_jscc_5e10;print("GPU_ENTRY_IMPORTS_PASSED")'
        print(command(c,'cd '+q(release)+' && '+q(GPU_PYTHON)+' -c '+q(test_code),timeout=60),flush=True)
        tests=command(c,'cd '+q(release)+' && JSCC_PROJECT_ROOT='+q(release)+' '+q(GPU_PYTHON)+' -m unittest test_jscc_5e10 -v 2>&1',timeout=120)
        if '\nOK' not in tests or 'skipped' in tests:raise ValueError('Frozen GPU CPU contract tests failed: '+tests)
        (REPORT/'remote_contract_tests.txt').write_bytes(tests.encode())
    write(REPORT/'kernel_deployment.json',dict(release=release,sha256=f['sha256'],input=GPU_BASE+'/input',
        input_archive_sha256=pack['sha256'],actual_free_bytes_before_deployment=free,required_bytes_with_headroom=reserve,
        tests_sha256=digest(REPORT/'remote_contract_tests.txt')))

def launcher(stage,release,phase_seconds,prepare_seconds,total_seconds):
    # Slurm has one parent task per node; torchrun launches four distinct local GPUs.
    script='''#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
[[ "$SLURM_NNODES" == 8 && "$SLURM_NTASKS" == 8 ]]
export JSCC_PROJECT_ROOT='''+q(release)+''' PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=bond0 TORCH_ELASTIC_WORKER_IDENTICAL=1
export JSCC_PHASE_SECONDS='''+str(phase_seconds)+''' JSCC_PREPARE_SECONDS='''+str(prepare_seconds)+'''
export JSCC_MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export JSCC_MASTER_PORT=$((40000+SLURM_JOB_ID%10000))
allocation='''+q(GPU_BASE+'/allocations/'+stage+'_')+'''"${SLURM_JOB_ID}.txt"
scontrol show job "$SLURM_JOB_ID" > "$allocation"
export JSCC_HOST_BYTES_NODE=$(cd "$JSCC_PROJECT_ROOT" && '''+q(GPU_PYTHON)+''' -c 'import sys;from jscc_5e10_common import host_allocated_bytes;print(host_allocated_bytes(open(sys.argv[1]).read(),8))' "$allocation")
srun --chdir=/tmp --label --kill-on-bad-exit=1 '''+q(GPU_PYTHON)+''' -c 'import torch;assert torch.cuda.device_count()==4;print("FOUR_VISIBLE_GPUS",flush=True)'
output='''+q(GPU_BASE+'/'+stage+'_')+'''"${SLURM_JOB_ID}"
'''
    if stage=='selection':
        args=[release+'/jscc_5e10_selection.py','--release',release,'--input',GPU_BASE+'/input',
              '--factors',FACTORS,'--output','$output','--allocation','$allocation']
    else:
        args=[release+'/run_jscc_5e10.py','--contract',release+'/contract.json','--input-root',GPU_BASE+'/input',
              '--factors',FACTORS,'--output','$output','--allocation','$allocation','--mode',stage]
        if stage=='formal':
            auth=read(REPORT/'formal_authority.json')
            args+=['--authority',auth['remote'],'--authority-sha256',auth['sha256']]
    invocation=' '.join('"'+x+'"' if x.startswith('$') else q(x) for x in args)
    script+='''timeout --signal=TERM --kill-after=60s '''+str(total_seconds)+'''s srun --chdir=/tmp --label --kill-on-bad-exit=1 bash -c '
  exec '''+q(GPU_PYTHON)+''' -m torch.distributed.run --nnodes=8 --nproc_per_node=4 --node_rank="$SLURM_PROCID" \
    --master_addr="$JSCC_MASTER_ADDR" --master_port="$JSCC_MASTER_PORT" --rdzv_backend=static \
    --rdzv_conf=timeout=600 --rdzv_id="${SLURM_JOB_ID}_three" --max_restarts=0 "${@:1}"
' bash '''+invocation+'\necho JSCC5E10_COMPUTATION_COMPLETED '+stage+'\n'
    return script

def submit_gpu(stage):
    path=REPORT/(stage+'_job.json')
    if path.exists():return True
    if stage=='selection':release=read(REPORT/'kernel_deployment.json')['release'];minutes=240;phase=3600;prep=10800
    elif stage=='validation':release=read(REPORT/'production_deployment.json')['release'];minutes=240;phase=3600;prep=10800
    elif stage=='formal':
        release=read(REPORT/'production_deployment.json')['release'];a=read(REPORT/'formal_authority.json')
        if not a['passed'] or a['contract_sha256']!=read(REPORT/'production_freeze.json')['sha256']['contract.json']:raise ValueError('Strict actual full-input authority required')
        phase=max(1800,math.ceil(a['estimated_max_phase_seconds']*1.5+300))
        prep=max(1800,math.ceil(a['measured_prepare_seconds']*1.5+300))
        minutes=max(120,math.ceil((a['estimated_total_seconds']*1.5+1800)/60))
        if phase>86400 or minutes>2880:raise ValueError('Actual throughput exceeds bounded 48-hour topology budget; diagnose before submitting')
    else:raise ValueError(stage)
    total=minutes*60-180;intent=REPORT/(stage+'_submission_intent.json')
    if intent.exists():raise ValueError('Unresolved '+stage+' submission intent; inspect scheduler, do not duplicate')
    script=launcher(stage,release,phase,prep,total);local=DATA/(stage+'.sh');local.write_bytes(script.encode())
    repair=read(REPORT/'launch_control_repair_freeze.json')
    remote=GPU_BASE+'/launch_control_releases/'+repair['release_key']+'/'+stage+'.sh'
    with connection('gpu') as c:
        queue=command(c,"squeue -h -u scxi717 -o '%i|%j|%T'")
        if len(queue.splitlines())>=50:
            write(REPORT/(stage+'_submission_wait.json'),dict(reason='account_50_job_limit',other_jobs_untouched=True));return False
        if any('|JSCC5e10_' in line for line in queue.splitlines()):raise ValueError('An existing study GPU job is still active; inspect registered stage')
        command(c,'test ! -e '+q(remote)+' && mkdir -p '+q(remote.rsplit('/',1)[0]))
        with c.open_sftp() as s:s.put(str(local),remote)
        if command(c,'sha256sum '+q(remote)).split()[0]!=digest(local):raise ValueError('Launch bytes differ')
        command(c,'bash -n '+q(remote))
        write(intent,dict(stage=stage,script_sha256=digest(local),started_epoch=time.time()))
        line=(f'sbatch --parsable -p gpu_5090 --qos=gpugpu -N 8 -n 8 --ntasks-per-node=1 --cpus-per-task=32 --gres=gpu:4 '
              f'--time={minutes} --exclude=wqd10nba06g6 --chdir=/tmp --job-name=JSCC5e10_{stage} '
              '--output='+q(GPU_BASE+'/logs/'+stage+'.%j.out')+' --error='+q(GPU_BASE+'/logs/'+stage+'.%j.err')+' '+q(remote))
        job=command(c,line,timeout=60).split(';')[0].strip()
        if not job.isdigit():raise ValueError('Ambiguous submit outcome; preserve intent and inspect queue')
    write(path,dict(job=int(job),stage=stage,host='gpu',release=release,output=GPU_BASE+'/'+stage+'_'+job,
        allocation=GPU_BASE+'/allocations/'+stage+'_'+job+'.txt',nodes=8,gpus_per_node=4,world_size=32,
        partition='gpu_5090',cpus_per_node=32,explicit_mem_parameter=False,walltime_minutes=minutes,phase_limit_seconds=phase,
        launch_control_repair_key=repair['release_key'],launch_control_repair_sha256=digest(REPORT/'launch_control_repair_freeze.json'),
        prepare_limit_seconds=prep,total_limit_seconds=total,script_sha256=digest(local),submitted_epoch=time.time()))
    write(intent,dict(read(intent),resolved=True,registered_job=int(job)))
    print('REGISTERED_8X4_GPU_JOB',stage,job,flush=True);return True

def freeze_verification(stage):
    file=REPORT/(stage+'_verification_freeze.json')
    if file.exists():return read(file)
    src=DATA/('kernel_payload' if stage=='selection' else 'production_payload')
    scientific_freeze=read(REPORT/('kernel_freeze.json' if stage=='selection' else 'production_freeze.json'))
    verify_files(src,scientific_freeze['sha256'])
    dst=DATA/(stage+'_verification_payload');shutil.copytree(src,dst)
    # A separate release identity, writing only separate acceptance receipts/archives.
    files=hashes(dst);write(dst/'verification_release.json',dict(sha256=files,scientific_release_sha256=scientific_freeze['sha256'],read_only=True))
    files=hashes(dst);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    r=dict(release_key=key,sha256=files,release=GPU_BASE+'/verification_releases/'+key)
    write(file,r);return r

def submit_verification(stage):
    path=REPORT/(stage+'_verification_job.json')
    if path.exists():return True
    job=read(REPORT/(stage+'_job.json'))
    with connection('gpu') as c:
        state,acc=job_state(c,job)
        if state!='complete':return False
    f=freeze_verification(stage);deployment=REPORT/(stage+'_verification_deployment.json')
    with connection('gpu') as c:
        if not deployment.exists():
            put_archive(c,DATA/(stage+'_verification_payload'),f['release'])
            write(deployment,dict(release=f['release'],sha256=f['sha256']))
        queue=command(c,"squeue -h -u scxi717 -o '%i|%j|%T'")
        if len(queue.splitlines())>=50:return False
        release=f['release'];output=GPU_BASE+'/acceptance_'+stage+'_'+str(job['job'])
        args=['--release',release,'--input',GPU_BASE+'/input','--factors',FACTORS,'--result',job['output'],
              '--allocation',job['allocation'],'--output',output,'--mode',stage]
        script='''#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load miniforge3/25.11.0-1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 JSCC_PROJECT_ROOT='''+q(release)+'''
cd '''+q(release)+'''
timeout --signal=TERM --kill-after=60s 6900s '''+q(GPU_PYTHON)+' -u '+q(release+'/jscc_5e10_verify_stage.py')+' '+' '.join(q(x) for x in args)+'\n'
        local=DATA/(stage+'_verification.sh');local.write_bytes(script.encode());remote=GPU_BASE+'/'+local.name
        with c.open_sftp() as s:s.put(str(local),remote)
        if command(c,'sha256sum '+q(remote)).split()[0]!=digest(local):raise ValueError('Verification script transfer differs')
        command(c,'bash -n '+q(remote));intent=REPORT/(stage+'_verification_submission_intent.json')
        if intent.exists():raise ValueError('Unresolved verification submission intent; inspect queue')
        write(intent,dict(stage=stage+'_verification',script_sha256=digest(local),source_job=job['job']))
        # One reserved GPU obtains real automatic memory TRES; verifier uses CPU only.
        line=(f'sbatch --parsable -p gpu_5090 --qos=gpugpu -N1 -n1 --cpus-per-task=8 --gres=gpu:1 --time=120 '
              '--exclude=wqd10nba06g6 --chdir=/tmp --job-name=JSCC5e10_'+stage+'_verify '
              '--output='+q(GPU_BASE+'/logs/'+stage+'_verification.%j.out')+' --error='+q(GPU_BASE+'/logs/'+stage+'_verification.%j.err')+' '+q(remote))
        number=command(c,line,timeout=60).split(';')[0].strip()
        if not number.isdigit():raise ValueError('Ambiguous verification submission')
    write(path,dict(job=int(number),stage=stage+'_verification',host='gpu',release=release,output=output,
        source_job=job['job'],source_accounting=acc,script_sha256=digest(local),read_only_verification=True,
        partition='gpu_5090',cpus_per_node=8,reserved_gpus=1,explicit_mem_parameter=False,
        launch_control_repair_sha256=digest(REPORT/'launch_control_repair_freeze.json')))
    write(intent,dict(read(intent),resolved=True,registered_job=int(number)))
    print('REGISTERED_READ_ONLY_ACCEPTANCE_JOB',stage,number,flush=True);return True

def slurm_peak_bytes(acc):
    values=[]
    for line in acc.splitlines():
        x=line.split('|')
        if len(x)<4 or not x[3]:continue
        m=re.fullmatch(r'([0-9.]+)([KMGT]?)',x[3])
        if not m:raise ValueError('Unrecognized Slurm MaxRSS '+x[3])
        values.append(float(m[1])*1024**(('KMGT'.index(m[2])+1) if m[2] else 1))
    if not values or max(values)<=0:raise ValueError('Actual exited GPU computation Slurm MaxRSS is missing; do not issue authority')
    return int(max(values))

def fetch_stage(stage):
    path=REPORT/(stage+'_acceptance.json')
    if path.exists():return True
    record=read(REPORT/(stage+'_verification_job.json'));source=read(REPORT/(stage+'_job.json'))
    with connection('gpu') as c:
        state,acc=job_state(c,record)
        if state!='complete':return False
        state,source_acc=job_state(c,source)
        if state!='complete':raise ValueError('Scientific allocation must have fully exited successfully')
        destination=DATA/(stage+'_accepted_'+str(source['job']));destination.mkdir(exist_ok=True)
        package_path=REPORT/(stage+'_package.json')
        with c.open_sftp() as s:
            for n,dest in [('package.json',package_path),('verification.json',REPORT/(stage+'_verification.json'))]:
                sha=command(c,'sha256sum '+q(record['output']+'/'+n)).split()[0];s.get(record['output']+'/'+n,str(dest))
                if digest(dest)!=sha:raise ValueError('Strict acceptance metadata transfer differs')
            package=read(package_path);archive=DATA/(stage+'_accepted_result.tar.gz')
            if archive.exists() and digest(archive)!=package['archive_sha256']:raise ValueError('Preserve unmatched partial result archive and diagnose transfer')
            if not archive.exists():
                partial=archive.with_suffix(archive.suffix+'.partial')
                if partial.exists():raise ValueError('Preserve previous interrupted transfer; diagnose and recover bounded missing bytes')
                s.get(record['output']+'/accepted_result.tar.gz',str(partial))
                if digest(partial)!=package['archive_sha256']:raise ValueError('Complete strict result archive SHA differs')
                os.replace(partial,archive)
        if not (destination/'verification.json').exists():
            with tarfile.open(archive) as t:
                members=t.getmembers()
                if {m.name for m in members}!=set(package['files']):raise ValueError('Archive includes missing/extra scientific members')
                for m in members:
                    if not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts:raise ValueError('Unsafe result archive')
                t.extractall(destination)
        verify_files(destination,package['files']);proof=read(destination/'verification.json')
        if not proof['passed'] or proof['mode']!=stage or proof['actual_primary_photons']!=TOTAL:raise ValueError('Current strict scientific acceptance differs')
        memory=host_allocated_bytes((destination/'allocation.txt').read_text(),8);maxrss=slurm_peak_bytes(source_acc)
        if maxrss>.8*memory:raise ValueError('Actual Slurm MaxRSS does not retain 20 percent node margin')
        allocation_sha=command(c,'sha256sum '+q(source['allocation'])).split()[0]
        if allocation_sha!=digest(destination/'allocation.txt'):raise ValueError('Scientific allocation remote/local SHA differs')
        logs=REPORT/(stage+'_logs');logs.mkdir(exist_ok=True)
        with c.open_sftp() as s:
            for name,r in ((stage,source),(stage+'_verification',record)):
                for ext in ('out','err'):
                    f=GPU_BASE+'/logs/'+name+'.'+str(r['job'])+'.'+ext;dest=logs/(name+'.'+ext)
                    sha=command(c,'sha256sum '+q(f)).split()[0];s.get(f,str(dest))
                    if digest(dest)!=sha:raise ValueError('Original log strict fetch differs')
        write(path,dict(passed=True,study=STUDY,mode=stage,job=source['job'],verification_job=record['job'],
             local_result=str(destination),remote_result=source['output'],package_sha256=digest(package_path),
             verification_sha256=digest(destination/'verification.json'),archive_sha256=package['archive_sha256'],
             source_accounting=source_acc,verification_accounting=acc,strict_files_sha256=package['files'],
             slurm_maxrss_bytes=maxrss,actual_allocated_bytes_node=memory,slurm_rss_fraction=maxrss/memory,
             log_sha256=hashes(logs),strict_fetch_passed=True))
    print('STRICT_FULL_STAGE_ACCEPTED',stage,source['job'],flush=True);return True

def freeze_production():
    if (REPORT/'production_freeze.json').exists():return
    accepted=read(REPORT/'selection_acceptance.json');selection=Path(accepted['local_result'])
    m=read(selection/'selection_manifest.json');old=read(DATA/'kernel_payload/baseline_contract.json')
    # Reject before allocating: CPU resident compact rows plus conservative old measured overhead.
    event_bytes=m['accepted_events']*78920*4
    minimum_expected_memory_per_node=event_bytes/8+4*(16<<30)
    expected_auto_allocation=32*15750*(1<<20)
    if minimum_expected_memory_per_node>.8*expected_auto_allocation:raise MemoryError('Actual accepted event set does not fit the conservative 8x4 memory budget')
    dst=DATA/'production_payload';shutil.copytree(DATA/'kernel_payload',dst)
    for name in ('selections','selected_rows'):shutil.copytree(selection/name,dst/name)
    shutil.copy2(selection/'selection_manifest.json',dst/'selection_manifest.json')
    files=hashes(dst);kernel=read(dst/'kernel_config.json')
    cfg=dict(kernel,files=files,model='continuous_energy',event_policy='legacy',nodes=8,gpus_per_node=4,world_size=32,
        iterations=10000,save_step=50,channels=list(CHANNELS),regularization='none',initial_density=1,
        joint_solver_enabled=False,cross_prediction_source='440_SinglePhoton_final',
        events_per_view=m['events_per_view'],accepted_events=m['accepted_events'],
        selection_manifest_sha256=digest(dst/'selection_manifest.json'),
        selection_acceptance_sha256=digest(REPORT/'selection_acceptance.json'),
        source_delivered_job=1669255,new_training=False,new_response=False,actual_event_rows_cpu_bytes=event_bytes,
        conservative_expected_memory_per_node=minimum_expected_memory_per_node)
    write(dst/'contract.json',cfg)
    from jscc_5e10_contract import load_contract
    load_contract(dst/'contract.json','validation',10,10)
    files=hashes(dst);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    write(REPORT/'production_freeze.json',dict(release_key=key,sha256=files,channels=list(CHANNELS),iterations=10000,save_step=50,
            accepted_events=m['accepted_events'],event_bytes_cpu=event_bytes,total_primary_photons=TOTAL,
            nodes=8,gpus_per_node=4,world_size=32,source_delivered_job=1669255))

def deploy_production():
    if (REPORT/'production_deployment.json').exists():return
    f=read(REPORT/'production_freeze.json');verify_files(DATA/'production_payload',f['sha256'])
    release=GPU_BASE+'/production_releases/'+f['release_key']
    with connection('gpu') as c:
        put_archive(c,DATA/'production_payload',release)
        check='from pathlib import Path;from jscc_5e10_contract import load_contract;load_contract(Path("contract.json"),"validation",10,10);print("THREE_ROUTE_CONTRACT_PASSED")'
        print(command(c,'cd '+q(release)+' && '+q(GPU_PYTHON)+' -c '+q(check),timeout=120),flush=True)
    write(REPORT/'production_deployment.json',dict(release=release,sha256=f['sha256']))

def issue_authority():
    if (REPORT/'formal_authority.json').exists():return
    accepted=read(REPORT/'validation_acceptance.json');result=Path(accepted['local_result'])
    proof=read(result/'verification.json');resources=proof['resources'];source=read(REPORT/'validation_job.json')
    times={k:max(r['phase_solve_seconds'][k] for r in resources) for k in PHASE_CHANNELS}
    prep=max(r['prepare_seconds'] for r in resources)
    scaled={k:v*1000 for k,v in times.items()}
    # Preparation and full-input SHA/startup run once; never multiply them by 1000.
    once=max(r['elapsed_seconds'] for r in resources)-sum(times.values())
    estimate=sum(scaled.values())+max(once,prep,0)
    authority=dict(passed=True,study=STUDY,contract_sha256=proof['contract_sha256'],
        result=source['output'],evidence_sha256=proof['result_files_sha256'],validation_job=source['job'],
        strict_acceptance_sha256=digest(REPORT/'validation_acceptance.json'),verification_sha256=digest(result/'verification.json'),
        measured_phase_seconds=times,measured_prepare_seconds=prep,estimated_phase_seconds=scaled,
        estimated_max_phase_seconds=max(scaled.values()),estimated_total_seconds=estimate,
        actual_primary_photons=TOTAL,world_size=32,nodes=8,gpus_per_node=4,
        timing_estimate_is_not_completion_guarantee=True)
    local=DATA/'formal_authority.json';write(local,authority);remote=GPU_BASE+'/formal_authority.json'
    with connection('gpu') as c:
        command(c,'test ! -e '+q(remote))
        with c.open_sftp() as s:s.put(str(local),remote)
        if command(c,'sha256sum '+q(remote)).split()[0]!=digest(local):raise ValueError('Authority transfer differs')
    write(REPORT/'formal_authority.json',dict(authority,remote=remote,sha256=digest(local)))
    print('ACTUAL_FULL_INPUT_VALIDATION_AUTHORITY_ISSUED',source['job'],flush=True)

def advance_reconstruction():
    if (REPORT/'delivery_summary.json').exists() and read(REPORT/'delivery_summary.json').get('passed'):
        print('EXPERIMENT_ALREADY_DELIVERED');return
    freeze_kernel();deploy_kernel()
    for stage in ('selection','validation','formal'):
        if stage=='validation':freeze_production();deploy_production()
        if stage=='formal':issue_authority()
        if not submit_gpu(stage):print('WAITING_ACCOUNT_SPACE',stage);return
        if not submit_verification(stage):print('WAITING_GPU_COMPUTATION',stage);return
        if not fetch_stage(stage):print('WAITING_READ_ONLY_ACCEPTANCE',stage);return
    write(REPORT/'numerical_delivery.json',dict(passed=True,study=STUDY,channels=list(CHANNELS),iterations=10000,save_step=50,
        formal_acceptance_sha256=digest(REPORT/'formal_acceptance.json'),scientific_visual_qa_completed=False,
        experiment_delivery_complete=False,automation_must_remain_active=True))
    print('NUMERICAL_RECONSTRUCTION_COMPLETE_REQUIRES_REPORT_VISUAL_QA',flush=True)

