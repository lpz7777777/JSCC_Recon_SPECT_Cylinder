"""User-authorized parallel 4090 selection trial; preserve the primary job."""
import argparse
import ctypes
import hashlib
import json
import os
import shutil
import time

from jscc_5e10_common import *
from jscc_5e10_workflow import connection, command, q, job_state
from jscc_5e10_reconstruction_workflow import launcher

STAGE = 'selection_4090_trial'


def live_registration(path):
    if not path.exists():
        return False
    pid = int(read(path).get('pid', 0))
    if pid <= 0:
        return False
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if handle:
            kernel.CloseHandle(ctypes.c_void_p(handle))
        return bool(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def trial_launcher(release):
    original = launcher('selection', release, 3600, 10800, 14220)
    replacements = {
        'export OMP_NUM_THREADS=8 ': 'export OMP_NUM_THREADS=6 ',
        q(GPU_BASE + '/allocations/selection_'): q(GPU_BASE + '/allocations/' + STAGE + '_'),
        q(GPU_BASE + '/selection_'): q(GPU_BASE + '/' + STAGE + '_'),
        'echo JSCC5E10_COMPUTATION_COMPLETED selection\n':
            'echo JSCC5E10_COMPUTATION_COMPLETED ' + STAGE + '\n',
    }
    script = original
    for old, new in replacements.items():
        if script.count(old) != 1:
            raise ValueError('Unexpected original launch expression: ' + old)
        script = script.replace(old, new, 1)
    restored = script
    for old, new in reversed(list(replacements.items())):
        restored = restored.replace(new, old, 1)
    if restored != original:
        raise ValueError('Trial changed more than launch controls and output paths')
    return script


def submit():
    job_path = REPORT / (STAGE + '_job.json')
    if job_path.exists():
        print('ALREADY_REGISTERED', read(job_path)['job'], flush=True)
        return
    for path in (DATA / 'advance_registration.json', DATA / (STAGE + '_registration.json')):
        if live_registration(path):
            raise RuntimeError('Registered local controller is still alive: ' + str(path))
    registration = DATA / (STAGE + '_registration.json')
    write(registration, dict(pid=os.getpid(), status='running', started_epoch=time.time()))
    exit_code = 1
    try:
        intent = REPORT / (STAGE + '_submission_intent.json')
        if intent.exists():
            raise RuntimeError('Unresolved trial intent; inspect scheduler before any submission')
        primary = REPORT / 'selection_job.json'
        primary_sha = digest(primary)
        kernel = read(REPORT / 'kernel_freeze.json')
        verify_files(DATA / 'kernel_payload', kernel['sha256'])
        release = read(REPORT / 'kernel_deployment.json')['release']
        script = trial_launcher(release)
        payload = DATA / (STAGE + '_launch_payload')
        payload.mkdir(exist_ok=False)
        shutil.copy2(__file__, payload / Path(__file__).name)
        shutil.copy2(HERE / 'test_jscc_5e10_4090_trial.py', payload / 'test_jscc_5e10_4090_trial.py')
        (payload / 'selection.sh').write_bytes(script.encode())
        files = hashes(payload)
        key = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
        remote = GPU_BASE + '/trial_launch_releases/' + key
        freeze = dict(release_key=key, sha256=files, primary_job=read(primary)['job'],
            primary_job_record_sha256=primary_sha, kernel_freeze_sha256=digest(REPORT / 'kernel_freeze.json'),
            transport_acceptance_sha256=digest(REPORT / 'transport_acceptance.json'),
            local_launch_tests_sha256=digest(REPORT / (STAGE + '_tests.txt')),
            original_controller_sha256=digest(HERE / 'jscc_5e10_reconstruction_workflow.py'),
            scope='Full-input selection only; no transport, response generation, validation or reconstruction.',
            user_authorized_parallel_trial=True, primary_job_preserved=True,
            partition='gpu_4090', nodes=8, gpus_per_node=4, world_size=32, cpus_per_node=24,
            expected_auto_memory_mib_per_node=240000, explicit_mem_parameter=False,
            actual_resource_and_strict_selection_acceptance_required=True,
            frozen_runtime_torch_threads_per_rank=10)
        write(REPORT / (STAGE + '_freeze.json'), freeze)
        with connection('gpu') as c:
            before = command(c, 'scontrol show job -o ' + str(read(primary)['job']), timeout=45)
            (REPORT / (STAGE + '_primary_before.txt')).write_bytes(before.encode())
            queue = command(c, "squeue -h -u scxi717 -o '%i|%j|%T'", timeout=45)
            if len(queue.splitlines()) >= 50:
                raise RuntimeError('Account has 50 active jobs; wait without submitting')
            if any('|JSCC5e10Trial4090|' in line for line in queue.splitlines()):
                raise RuntimeError('An unregistered 4090 trial exists; inspect before submitting')
            policy = command(c, 'scontrol show partition gpu_4090', timeout=45)
            if 'DefCpuPerGPU=6' not in policy or 'DefMemPerCPU=10000' not in policy:
                raise ValueError('4090 partition policy changed; recompute launch budget')
            policy_path = REPORT / (STAGE + '_partition_policy.txt')
            policy_path.write_bytes(policy.encode())
            command(c, 'test ! -e ' + q(remote) + ' && mkdir -p ' + q(remote))
            with c.open_sftp() as s:
                for name in files:
                    s.put(str(payload / name), remote + '/' + name)
            for name, expected in files.items():
                if command(c, 'sha256sum ' + q(remote + '/' + name)).split()[0] != expected:
                    raise ValueError('Frozen trial launch transfer differs')
            command(c, 'bash -n ' + q(remote + '/selection.sh'))
            options = ('-p gpu_4090 --qos=gpugpu -N8 -n8 --ntasks-per-node=1 '
                '--cpus-per-task=24 --gres=gpu:4 --time=240 --exclude=wqd10nba06g6 --chdir=/tmp '
                '--job-name=JSCC5e10Trial4090 --output=' + q(GPU_BASE + '/logs/' + STAGE + '.%j.out') +
                ' --error=' + q(GPU_BASE + '/logs/' + STAGE + '.%j.err') + ' ' + q(remote + '/selection.sh'))
            test = command(c, 'sbatch --test-only ' + options + ' 2>&1', timeout=60)
            write(REPORT / (STAGE + '_launch_acceptance.json'), dict(passed=True,
                scope='Launch policy and immutable script only; not a compute/resource certificate.',
                freeze_sha256=digest(REPORT / (STAGE + '_freeze.json')),
                partition_policy_sha256=digest(policy_path), scheduler_test_only=test,
                primary_job_record_sha256=primary_sha, scientific_release_unchanged=True))
            write(intent, dict(stage=STAGE, script_sha256=files['selection.sh'], started_epoch=time.time(),
                primary_job=read(primary)['job'], primary_job_record_sha256=primary_sha))
            number = command(c, 'sbatch --parsable ' + options, timeout=60).split(';')[0].strip()
            if not number.isdigit():
                raise RuntimeError('Ambiguous trial submission; preserve intent and inspect scheduler')
            record = dict(job=int(number), stage=STAGE, host='gpu', release=release,
                output=GPU_BASE + '/' + STAGE + '_' + number,
                allocation=GPU_BASE + '/allocations/' + STAGE + '_' + number + '.txt',
                nodes=8, gpus_per_node=4, world_size=32, partition='gpu_4090', cpus_per_node=24,
                explicit_mem_parameter=False, walltime_minutes=240, total_limit_seconds=14220,
                trial_launch_release=remote, trial_freeze_sha256=digest(REPORT / (STAGE + '_freeze.json')),
                script_sha256=files['selection.sh'], primary_job=read(primary)['job'],
                primary_job_record_sha256=primary_sha, user_authorized_parallel_trial=True,
                production_registry_replaced=False, submitted_epoch=time.time())
            write(job_path, record)
            write(intent, dict(read(intent), resolved=True, registered_job=int(number)))
            snapshot = command(c, 'scontrol show job -o ' + number, timeout=45)
            (REPORT / (STAGE + '_submit_scontrol.txt')).write_bytes(snapshot.encode())
            after = command(c, 'scontrol show job -o ' + str(read(primary)['job']), timeout=45)
            (REPORT / (STAGE + '_primary_after.txt')).write_bytes(after.encode())
        if digest(primary) != primary_sha:
            raise ValueError('Primary registration changed during trial submission')
        print('REGISTERED_PARALLEL_4090_SELECTION_TRIAL', number, flush=True)
        exit_code = 0
    finally:
        write(registration, dict(read(registration), status='complete', exit_code=exit_code,
            finished_epoch=time.time()))


def status():
    path = REPORT / (STAGE + '_job.json')
    if not path.exists():
        print('NO_REGISTERED_4090_TRIAL', flush=True)
        return
    job = read(path)
    with connection('gpu') as c:
        state, accounting = job_state(c, job)
        print('4090_TRIAL', job['job'], state, accounting, flush=True)
        for suffix in ('out', 'err'):
            log = GPU_BASE + '/logs/' + STAGE + '.' + str(job['job']) + '.' + suffix
            print(command(c, 'if test -f ' + q(log) + '; then tail -n 12 ' + q(log) + '; fi', timeout=45), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit', 'status'))
    args = parser.parse_args()
    (submit if args.action == 'submit' else status)()
