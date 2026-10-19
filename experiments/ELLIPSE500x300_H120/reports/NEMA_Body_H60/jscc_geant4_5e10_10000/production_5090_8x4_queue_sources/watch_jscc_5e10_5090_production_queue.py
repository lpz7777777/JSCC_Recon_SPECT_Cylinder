"""Queue one 32-GPU resident reconstruction, held until strict validation delivery.

This is a manual-request workflow. It never resumes recurring automations, runs
selection again, changes existing releases, or retries failed scientific jobs.
"""
import argparse
import ast
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from jscc_5e10_common import *
from jscc_5e10_5090_8x2_trial import live_registration, replace_once
from jscc_5e10_5090_8x4_monitor_repair import submitted_topology_matches
from jscc_5e10_formal_queue_gate import measured_limits
import jscc_5e10_reconstruction_workflow as workflow
from jscc_5e10_workflow import connection, command, q, accounting, completed, job_state

PREFIX = 'production_5090_8x4_queue'
REGISTRATION = DATA / (PREFIX + '_registration.json')
NEW = ('jscc_5e10_formal_queue_gate.py', 'test_jscc_5e10_formal_queue_gate.py')
GATE = GPU_BASE + '/formal_release_gate.json'


@contextlib.contextmanager
def controlled():
    for p in DATA.glob('*registration.json'):
        if live_registration(p):
            raise RuntimeError('Registered controller is alive: ' + p.name)
    write(REGISTRATION, dict(pid=os.getpid(), status='running', started_epoch=time.time()))
    code = 1
    try:
        yield
        code = 0
    finally:
        write(REGISTRATION, dict(read(REGISTRATION), status='complete', exit_code=code, finished_epoch=time.time()))


def prepare():
    path = REPORT / 'production_freeze.json'
    if path.exists():
        f = read(path)
        if f.get('queue_workflow') != PREFIX:
            raise ValueError('A different production release already exists')
        verify_files(DATA/'production_payload', f['sha256'])
        return f
    accepted_path = REPORT/'selection_5090_8x2_monitor_repair_acceptance.json'
    accepted = read(accepted_path)
    if not (accepted['passed'] and accepted['strict_fetch_passed'] and
            accepted['job'] == 1686507 and accepted['verification_job'] == 1686546):
        raise ValueError('Previously strictly delivered selection required')
    source = Path(accepted['local_result'])
    m = read(source/'selection_manifest.json')
    if digest(source/'selection_manifest.json') != accepted['strict_files_sha256']['selection_manifest.json']:
        raise ValueError('Accepted selection manifest changed')
    # Reuse only the already accepted final 40 arrays. Do not rerun the old verifier.
    array_sha = {folder+'/'+n: sha for folder, key in
                 [('selections', 'selection_files'), ('selected_rows', 'selected_rows_files')]
                 for n, sha in m[key].items()}
    if len(array_sha) != 40 or any(accepted['strict_files_sha256'][n] != sha for n, sha in array_sha.items()):
        raise ValueError('Exactly 40 strictly delivered source arrays required')
    verify_files(source, array_sha)
    repaired = read(REPORT/'selection_5090_8x4_monitor_repair_freeze.json')
    kernel = DATA/'selection_5090_8x4_monitor_repair_kernel_payload'
    verify_files(kernel, repaired['sha256'])
    original = read(REPORT/'kernel_freeze.json')
    unchanged = {n: sha for n, sha in original['sha256'].items()
                 if n not in ('jscc_5e10_runtime.py', 'kernel_config.json')
                 and '__pycache__' not in n and not n.endswith('.pyc')}
    verify_files(kernel, unchanged)
    # Actual per-rank partition sizes, including all rounding, not just an average.
    # Use the actual frozen partition implementation for rounding at rank boundaries.
    source_text = (kernel/'run_energy_preflight_v5.py').read_text(encoding='utf-8')
    fn = next(n for n in ast.parse(source_text).body if isinstance(n, ast.FunctionDef) and n.name == 'partition_indices')
    namespace = {}
    import numpy as np
    namespace.update(np=np)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<frozen partition>', 'exec'), namespace)
    rank_events = [sum(len(namespace['partition_indices'](np.arange(n), r, 32)[0])
                       for n in m['events_per_view']) for r in range(32)]
    node_bytes = [sum(rank_events[i:i+4])*78920*4 + 4*(16<<30) for i in range(0,32,4)]
    expected_memory = 32*15750*(1<<20)
    if max(node_bytes) > .8*expected_memory:
        raise MemoryError('Resident response plus conservative overhead exceeds 80% node budget')
    dst = DATA/'production_payload'
    dst.mkdir(exist_ok=False)
    for name in repaired['sha256']:
        target = dst/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(kernel/name, target)
    for name in array_sha:
        target = dst/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, target)
    shutil.copy2(source/'selection_manifest.json', dst/'selection_manifest.json')
    for name in NEW:
        shutil.copy2(HERE/name, dst/name)
    files = hashes(dst)
    cfg = dict(read(dst/'kernel_config.json'), files=files, model='continuous_energy',
               event_policy='legacy', nodes=8, gpus_per_node=4, world_size=32,
               iterations=10000, save_step=50, channels=list(CHANNELS), regularization='none',
               initial_density=1, joint_solver_enabled=False, cross_prediction_source='440_SinglePhoton_final',
               events_per_view=m['events_per_view'], accepted_events=m['accepted_events'],
               selection_manifest_sha256=digest(dst/'selection_manifest.json'),
               selection_acceptance_sha256=digest(accepted_path), selected_source_job=1686507,
               selected_source_verification_job=1686546, response_storage_policy='original_CPU_resident_float32',
               actual_event_rows_cpu_bytes=m['accepted_events']*78920*4,
               conservative_expected_memory_per_node=max(node_bytes), new_training=False, new_response=False)
    write(dst/'contract.json', cfg)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', JSCC_PROJECT_ROOT=str(dst.resolve()),
               JSCC_UUID_REPAIR_KERNEL=str(dst.resolve()),
               JSCC_TRIAL_ORIGINAL_KERNEL=str((DATA/'kernel_payload').resolve()))
    tests = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-m', 'unittest',
                            'test_jscc_5e10', 'test_jscc_5e10_gpu_uuid', 'test_jscc_5e10_formal_queue_gate', '-v'],
                           cwd=dst, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    (REPORT/(PREFIX+'_local_tests.txt')).write_bytes(tests.stdout)
    if tests.returncode or b'\nOK' not in tests.stdout or b'skipped' in tests.stdout:
        raise ValueError('Production preflight tests failed: '+tests.stdout.decode(errors='replace'))
    verify_files(dst, unchanged)
    files = hashes(dst)
    key = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    f = dict(queue_workflow=PREFIX, release_key=key, sha256=files,
             release=GPU_BASE+'/production_releases/'+key, contract_sha256=digest(dst/'contract.json'),
             source_kernel_freeze_sha256=digest(REPORT/'kernel_freeze.json'),
             source_uuid_repair_freeze_sha256=digest(REPORT/'selection_5090_8x4_monitor_repair_freeze.json'),
             source_selection_acceptance_sha256=digest(accepted_path), source_selection_job=1686507,
             source_selection_verification_job=1686546, selected_arrays_sha256=array_sha,
             unchanged_original_source_sha256=unchanged, channels=list(CHANNELS), iterations=10000, save_step=50,
             nodes=8, gpus_per_node=4, world_size=32, total_primary_photons=TOTAL,
             expected_node_allocation_bytes=expected_memory, conservative_node_bytes=node_bytes,
             full_response_memory_not_yet_measured=True, persistent_response_disk_cache_required=False,
             local_tests_sha256=digest(REPORT/(PREFIX+'_local_tests.txt')),
             prior_records={p.name:digest(p) for p in REPORT.glob('*job.json')},
             automations_remain_paused=True,
             authorization='2026-10-16 user requests formal reconstruction queue; formal stays held until strict validation authority')
    write(path, f)
    return f


def deploy(f):
    p = REPORT/'production_deployment.json'
    if p.exists():
        if read(p)['sha256'] != f['sha256']: raise ValueError('Production deployment mismatch')
        return
    with connection('gpu') as c:
        queue = command(c, "squeue -h -u scxi717 -o '%i|%j|%T'")
        if len(queue.splitlines()) > 47:
            raise RuntimeError('Three queue slots required; other jobs remain untouched')
        if any('JSCC5e10' in line for line in queue.splitlines()):
            raise RuntimeError('A live study job already exists; inspect before new submission')
        acc = accounting(c, 1688717)
        if not completed(acc, 1688717): raise RuntimeError('UUID-repaired 32-GPU selection must have exited successfully')
        policy = command(c, 'scontrol show partition gpu_5090')
        if not all(x in policy for x in ['DefCpuPerGPU=8', 'DefMemPerCPU=15750', 'State=UP']):
            raise ValueError('Actual scheduler memory policy changed')
        free = workflow.remote_free_bytes(c)
        if free < (20<<30): raise OSError('At least 20GiB output/release headroom required')
        release = f['release']
        workflow.put_archive(c, DATA/'production_payload', release)
        check = 'from pathlib import Path;from jscc_5e10_contract import load_contract;load_contract(Path("contract.json"),"validation",10,10);print("PRODUCTION_CONTRACT_PASSED")'
        identity = command(c, 'cd '+q(release)+' && PYTHONDONTWRITEBYTECODE=1 '+q(GPU_PYTHON)+' -B -c '+q(check), timeout=180)
        original_release=read(REPORT/'kernel_deployment.json')['release']
        tests = command(c, 'cd '+q(release)+' && PYTHONDONTWRITEBYTECODE=1 JSCC_PROJECT_ROOT='+q(release)
                        +' JSCC_UUID_REPAIR_KERNEL='+q(release)+' JSCC_TRIAL_ORIGINAL_KERNEL='+q(original_release)+' '
                        +q(GPU_PYTHON)+' -B -m unittest test_jscc_5e10 test_jscc_5e10_gpu_uuid test_jscc_5e10_formal_queue_gate -v 2>&1', timeout=180)
        (REPORT/(PREFIX+'_linux_tests.txt')).write_text(identity+tests, encoding='utf-8')
        if '\nOK' not in tests or 'skipped' in tests: raise ValueError('Actual Linux tests failed')
    write(p, dict(release=release, sha256=f['sha256'], actual_free_bytes=free,
                  source_1688717_accounting=acc, partition_policy=policy,
                  linux_tests_sha256=digest(REPORT/(PREFIX+'_linux_tests.txt'))))


def submit_script(stage, script, release, minutes, dependencies, held, verifier=False, output=None):
    path = REPORT/(stage+'_job.json')
    if path.exists(): return read(path)
    intent = REPORT/(stage+'_submission_intent.json')
    if intent.exists(): raise RuntimeError('Unresolved submission intent: '+str(intent))
    local = REPORT/'launch_scripts'/(PREFIX+'_'+stage+'.sh')
    if local.exists():
        if local.read_bytes() != script.encode(): raise ValueError('Existing script differs')
    else: local.write_bytes(script.encode())
    remote = GPU_BASE+'/queued_production_launch/'+read(REPORT/'production_freeze.json')['release_key']+'/'+stage+'.sh'
    allocation = '--nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1' if verifier else '--nodes=8 --ntasks=8 --ntasks-per-node=1 --cpus-per-task=32 --gres=gpu:4'
    options = '-p gpu_5090 --qos=gpugpu '+allocation+' --time='+str(minutes)+' --exclude=wqd10nba06g6 --chdir=/tmp --job-name=JSCC5e10_'+stage
    if dependencies: options += ' --dependency=afterok:'+':'.join(map(str,dependencies))
    if held: options += ' --hold'
    options += ' --output='+q(GPU_BASE+'/logs/'+stage+'.%j.out')+' --error='+q(GPU_BASE+'/logs/'+stage+'.%j.err')+' '+q(remote)
    with connection('gpu') as c:
        existing = command(c, 'if test -e '+q(remote)+'; then sha256sum '+q(remote)+'; fi').strip()
        if existing:
            if existing.split()[0] != digest(local):
                raise ValueError('Preserve unmatched previously uploaded launch script')
        else:
            command(c, 'mkdir -p '+q(remote.rsplit('/',1)[0]))
            with c.open_sftp() as s: s.put(str(local), remote)
        if command(c,'sha256sum '+q(remote)).split()[0] != digest(local): raise ValueError('Launch transfer mismatch')
        command(c, 'bash -n '+q(remote))
        test = command(c, 'sbatch --test-only '+options+' 2>&1', timeout=180)
        write(intent, dict(stage=stage, script_sha256=digest(local), command='sbatch --parsable '+options,
                           dependencies=dependencies, held=held, resolved=False, started_epoch=time.time(), test_only=test))
        job = command(c, 'sbatch --parsable '+options, timeout=60).split(';')[0].strip()
        if not job.isdigit(): raise RuntimeError('Ambiguous submit; preserve intent and inspect scheduler')
        record = dict(job=int(job), stage=stage, host='gpu', release=release,
                      output=output or GPU_BASE+'/'+stage+'_'+job,
                      allocation=GPU_BASE+'/allocations/'+stage+'_'+job+'.txt',
                      nodes=1 if verifier else 8, gpus_per_node=1 if verifier else 4,
                      world_size=1 if verifier else 32, cpus_per_node=8 if verifier else 32,
                      partition='gpu_5090', explicit_mem_parameter=False, walltime_minutes=minutes,
                      dependencies=dependencies, held_pending_validation=held, script_sha256=digest(local),
                      script=remote, submitted_epoch=time.time(), queue_workflow=PREFIX)
        write(path, record)
        write(intent, dict(read(intent), resolved=True, registered_job=int(job)))
        snapshot = command(c, 'scontrol show job -o '+job)
        (REPORT/(PREFIX+'_'+stage+'_scontrol.txt')).write_text(snapshot,encoding='utf-8')
        if not verifier and not submitted_topology_matches(snapshot): raise ValueError('Actual 32GPU request differs; preserve job')
        if held and 'JobHeldUser' not in snapshot: raise ValueError('Formal job must remain held pending authority')
    print('REGISTERED',stage,job,flush=True)
    return record


def formal_script(release):
    # Compile the unchanged launcher with a future authority placeholder. The
    # fail-closed gate replaces only runtime limits and the SHA, never science.
    fn = next(n for n in ast.parse((HERE/'jscc_5e10_reconstruction_workflow.py').read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='launcher')
    env = dict(q=q, GPU_BASE=GPU_BASE, FACTORS=workflow.FACTORS, GPU_PYTHON=GPU_PYTHON,
               REPORT=REPORT, read=lambda _:dict(remote=GPU_BASE+'/formal_authority.json',sha256='FUTURE_AUTHORITY_SHA'))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<held-formal-launcher>','exec'),env)
    script = env['launcher']('formal',release,86400,10800,172620)
    gate_command = q(GPU_PYTHON)+' -B '+q(release+'/jscc_5e10_formal_queue_gate.py')+' --gate '+q(GATE)+' --contract '+q(release+'/contract.json')
    return replace_once(script, {
        'export JSCC_PHASE_SECONDS=86400 JSCC_PREPARE_SECONDS=10800':
            'gate_exports=$('+gate_command+')\neval "$gate_exports"',
        '172620s srun': '"${JSCC_TOTAL_SECONDS}s" srun',
        'FUTURE_AUTHORITY_SHA': '"$JSCC_AUTHORITY_SHA256"',
    })


def submit_queue():
    f = prepare(); deploy(f); release=f['release']
    validation = submit_script('validation',workflow.launcher('validation',release,3600,10800,14220),
                               release,240,[],False)
    verification_freeze = workflow.freeze_verification('validation')
    d = REPORT/'validation_verification_deployment.json'
    if not d.exists():
        with connection('gpu') as c:
            workflow.put_archive(c,DATA/'validation_verification_payload',verification_freeze['release'])
        write(d,dict(release=verification_freeze['release'],sha256=verification_freeze['sha256']))
    verification_release=verification_freeze['release']
    output=GPU_BASE+'/acceptance_validation_'+str(validation['job'])
    args=['--release',verification_release,'--input',GPU_BASE+'/input','--factors',workflow.FACTORS,
          '--result',validation['output'],'--allocation',validation['allocation'],'--output',output,'--mode','validation']
    script='#!/usr/bin/env bash\nset -euo pipefail\nsource /etc/profile.d/modules.sh\nmodule load miniforge3/25.11.0-1\n'
    script+='export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 JSCC_PROJECT_ROOT='+q(verification_release)+'\n'
    script+='cd '+q(verification_release)+'\ntimeout --signal=TERM --kill-after=60s 6900s '+q(GPU_PYTHON)+' -B -u '+q(verification_release+'/jscc_5e10_verify_stage.py')+' '+' '.join(q(x) for x in args)+'\n'
    verification=submit_script('validation_verification',script,verification_release,120,[validation['job']],False,True,output)
    formal=submit_script('formal',formal_script(release),release,2880,[verification['job']],True)
    # This is a queued request, not an authority or claim of successful validation.
    write(REPORT/(PREFIX+'_submission.json'),dict(passed=True,validation_job=validation['job'],
          validation_verification_job=verification['job'],formal_job=formal['job'],formal_held=True,
          formal_authority_issued=False,provisional_formal_walltime_minutes=2880,
          measured_walltime_update_required_before_release=True,automations_remain_paused=True,
          production_freeze_sha256=digest(REPORT/'production_freeze.json')))
    for name,sha in f['prior_records'].items():
        if digest(REPORT/name)!=sha:raise ValueError('Prior job registration changed: '+name)


def advance_release():
    formal=read(REPORT/'formal_job.json')
    if (REPORT/(PREFIX+'_release.json')).exists():return True
    try:
        with connection('gpu') as c:
            for stage in ('validation','validation_verification'):
                state,_=job_state(c,read(REPORT/(stage+'_job.json')))
                if state!='complete':return False
            snapshot=command(c,'scontrol show job -o '+str(formal['job']))
            if 'JobState=PENDING' not in snapshot or 'JobHeldUser' not in snapshot:
                raise RuntimeError('Formal job is not in its expected held state; inspect without altering it')
    except (TimeoutError, OSError) as error:
        # Only the read-only poll is retried. Never retry a partial transfer,
        # authority publication, actual submission, or release automatically.
        write(REPORT/(PREFIX+'_poll_error.json'),dict(checked_epoch=time.time(),
              error_type=type(error).__name__,error=str(error),read_only_poll=True,formal_remains_held=True))
        print('READ_ONLY_POLL_UNAVAILABLE',type(error).__name__,flush=True)
        return False
    if not workflow.fetch_stage('validation'):return False
    workflow.issue_authority()
    authority=read(REPORT/'formal_authority.json'); limits=measured_limits(authority)
    gate=dict(passed=True,strict_fetch_passed=True,formal_job=formal['job'],
              validation_job=read(REPORT/'validation_job.json')['job'],
              validation_verification_job=read(REPORT/'validation_verification_job.json')['job'],
              contract_sha256=authority['contract_sha256'],authority_file=authority['remote'],
              authority_sha256=authority['sha256'],strict_acceptance_sha256=digest(REPORT/'validation_acceptance.json'),
              limits=limits,automations_remain_paused=True)
    local=DATA/'formal_release_gate.json'
    if local.exists() and read(local)!=gate:raise ValueError('Previously prepared release gate differs')
    write(local,gate)
    with connection('gpu') as c:
        exists=command(c,'if test -e '+q(GATE)+'; then sha256sum '+q(GATE)+'; fi').strip()
        if exists:
            if exists.split()[0]!=digest(local):raise ValueError('Remote release gate differs')
        else:
            with c.open_sftp() as s:s.put(str(local),GATE)
        if command(c,'sha256sum '+q(GATE)).split()[0]!=digest(local):raise ValueError('Release gate transfer differs')
        command(c,'scontrol update JobId='+str(formal['job'])+' TimeLimit='+str(limits['minutes']))
        snapshot=command(c,'scontrol show job -o '+str(formal['job']))
        fields=dict(x.split('=',1) for x in snapshot.split() if '=' in x)
        days,_,clock=fields['TimeLimit'].rpartition('-')
        hours,minutes,seconds=map(int,clock.split(':'))
        actual_seconds=(int(days or 0)*24+hours)*3600+minutes*60+seconds
        if actual_seconds!=limits['minutes']*60 or 'JobHeldUser' not in snapshot:
            raise ValueError('Measured walltime/held state did not apply')
        write(REPORT/(PREFIX+'_release_intent.json'),dict(formal_job=formal['job'],gate_sha256=digest(local),limits=limits))
        command(c,'scontrol release '+str(formal['job']))
        after=command(c,'scontrol show job -o '+str(formal['job']))
    write(REPORT/(PREFIX+'_release.json'),dict(passed=True,formal_job=formal['job'],gate_sha256=digest(local),
          limits=limits,actual_scontrol=after,strict_acceptance_sha256=digest(REPORT/'validation_acceptance.json'),released_epoch=time.time()))
    print('FORMAL_RELEASED_AFTER_STRICT_VALIDATION',formal['job'],flush=True)
    return True


def watch():
    # One finite controller for this explicit submission; not a recurring automation.
    deadline=time.monotonic()+14*24*3600
    while time.monotonic()<deadline:
        if advance_release():return
        print('WAITING_REGISTERED_VALIDATION',time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
        time.sleep(300)
    raise TimeoutError('One-shot controller deadline reached; preserve held formal job')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('submit','release','watch'))
    a=p.parse_args()
    with controlled():
        if a.command=='submit':submit_queue()
        elif a.command=='release':print('READY',advance_release(),flush=True)
        else:watch()
