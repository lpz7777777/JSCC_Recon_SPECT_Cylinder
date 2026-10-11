"""User-authorized UUID-only recovery of failed primary selection 1685272."""
import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from shlex import quote as q

from jscc_5e10_common import *
from jscc_5e10_5090_8x2_trial import live_registration, launcher, replace_once

STAGE = 'selection_5090_8x4_monitor_repair'
JOB_NAME = 'JSCC5e10Repair5090x4'
ORIGINAL = DATA / 'kernel_payload'
PAYLOAD = DATA / (STAGE + '_kernel_payload')
GUARDS = ('advance', 'selection_4090_trial', 'selection_5090_8x2_trial',
          'selection_5090_8x2_monitor_repair', 'selection_5090_8x2_monitor_repair_verification',
          'streaming_4090_8x3', STAGE)
PRESERVED = ('selection_job.json', 'selection_4090_trial_job.json',
             'selection_5090_8x2_trial_job.json', 'selection_5090_8x2_monitor_repair_job.json',
             'selection_5090_8x2_monitor_repair_verification_job.json', 'streaming_4090_8x3_freeze.json')
ADDED = ('jscc_5e10_gpu_uuid.py', 'jscc_5e10_5090_8x2_trial.py',
         'test_jscc_5e10_gpu_uuid.py', 'test_jscc_5e10_5090_8x4_monitor_repair.py',
         'jscc_5e10_5090_8x4_monitor_repair.py')


def runtime_source(root):
    return replace_once((Path(root) / 'jscc_5e10_runtime.py').read_text(encoding='utf-8'), {
        'import torch\n': 'import torch\nfrom jscc_5e10_gpu_uuid import canonical_gpu_uuid\n',
        "prop=torch.cuda.get_device_properties(device);self.uuid=str(getattr(prop,'uuid',''))":
            "prop=torch.cuda.get_device_properties(device);self.raw_uuid=str(getattr(prop,'uuid',''))\n"
            "        self.uuid=canonical_gpu_uuid(self.raw_uuid) if self.raw_uuid else ''",
        'self.sample();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()':
            "self.sample()\n"
            "        print('JSCC_GPU_UUID_IDENTITY',os.environ['RANK'],self.raw_uuid,self.uuid,torch.__version__,flush=True)\n"
            '        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()',
        'gpu_uuid=_monitor.uuid,': 'gpu_uuid=_monitor.uuid,cuda_property_uuid_raw=_monitor.raw_uuid,',
    })


def repair_launcher(release):
    return replace_once(launcher('selection', release, 3600, 10800, 14220), {
        q(GPU_BASE + '/allocations/selection_'): q(GPU_BASE + '/allocations/' + STAGE + '_'),
        q(GPU_BASE + '/selection_'): q(GPU_BASE + '/' + STAGE + '_'),
        'echo JSCC5E10_COMPUTATION_COMPLETED selection\n':
            'echo JSCC5E10_COMPUTATION_COMPLETED ' + STAGE + '\n',
    })


def remote(c, text, timeout=120):
    _, out, err = c.exec_command(text, timeout=timeout)
    stdout = out.read().decode(errors='replace')
    stderr = err.read().decode(errors='replace')
    status = out.channel.recv_exit_status()
    if status:
        raise RuntimeError((stdout + stderr)[-12000:])
    return stdout + stderr


def submitted_topology_matches(snapshot):
    fields = dict(t.split('=', 1) for t in snapshot.split() if '=' in t)
    # Pending jobs express an exact node request as either 8 or the range 8-8.
    nodes = fields.get('NumNodes', '').split('-')
    tres = dict(t.split('=', 1) for t in fields.get('ReqTRES', '').split(',') if '=' in t)
    return (fields.get('Partition') == 'gpu_5090' and nodes in (['8'], ['8', '8'])
            and fields.get('NumCPUs') == '256' and fields.get('NumTasks') == '8'
            and fields.get('CPUs/Task') == '32' and tres.get('gres/gpu') == '32'
            and fields.get('TresPerNode') == 'gres/gpu:4')


def controller_guard():
    busy = [n for n in GUARDS if live_registration(DATA / (n + '_registration.json'))]
    if busy:
        raise RuntimeError('Controller is alive: ' + ','.join(busy))


def manifest(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def prepare():
    proof = REPORT / (STAGE + '_freeze.json')
    if proof.exists():
        freeze = read(proof)
        verify_files(PAYLOAD, freeze['sha256'])
        return freeze
    old = read(REPORT / 'kernel_freeze.json')
    verify_files(ORIGINAL, old['sha256'])
    PAYLOAD.mkdir(exist_ok=False)
    original_members = {n: s for n, s in old['sha256'].items()
                        if '__pycache__' not in Path(n).parts and not n.endswith('.pyc')}
    for name in original_members:
        p = PAYLOAD / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ORIGINAL / name, p)
    (PAYLOAD / 'jscc_5e10_runtime.py').write_bytes(runtime_source(ORIGINAL).encode())
    for name in ADDED:
        shutil.copy2(HERE / name, PAYLOAD / name)
    cfg = read(ORIGINAL / 'kernel_config.json')
    files = manifest(PAYLOAD)
    files.pop('kernel_config.json')
    cfg['files'] = files
    write(PAYLOAD / 'kernel_config.json', cfg)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', JSCC_PROJECT_ROOT=str(PAYLOAD),
               JSCC_UUID_REPAIR_KERNEL=str(PAYLOAD), JSCC_REPAIR_ORIGINAL_KERNEL=str(ORIGINAL),
               JSCC_TRIAL_ORIGINAL_KERNEL=str(ORIGINAL))
    checks = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'unittest',
                             'test_jscc_5e10_gpu_uuid', 'test_jscc_5e10_5090_8x4_monitor_repair',
                             'test_jscc_5e10', '-v'], cwd=PAYLOAD, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    test_path = REPORT / (STAGE + '_local_tests.txt')
    test_path.write_bytes(checks.stdout)
    if checks.returncode or b'\nOK' not in checks.stdout or b'skipped' in checks.stdout:
        raise ValueError('Local repaired-source checks failed: ' + checks.stdout.decode(errors='replace'))
    files = manifest(PAYLOAD)
    unchanged = {n: s for n, s in original_members.items()
                 if n not in ('jscc_5e10_runtime.py', 'kernel_config.json')}
    verify_files(PAYLOAD, unchanged)
    key = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    release = GPU_BASE + '/primary_selection_repair_releases/' + key
    script = REPORT / 'launch_scripts' / (STAGE + '.sh')
    assert not script.exists()
    script.write_bytes(repair_launcher(release).encode())
    freeze = dict(release_key=key, release=release, sha256=files,
                  repaired_job=1685272, original_kernel_freeze_sha256=digest(REPORT / 'kernel_freeze.json'),
                  original_kernel_config_sha256=digest(ORIGINAL / 'kernel_config.json'),
                  source_changes={'jscc_5e10_runtime.py': {'before': original_members['jscc_5e10_runtime.py'],
                                  'after': files['jscc_5e10_runtime.py']}},
                  unchanged_original_members=unchanged, new_members=list(ADDED),
                  excluded_derived_files=[n for n in old['sha256'] if n not in original_members],
                  preserved_records={n: digest(REPORT / n) for n in PRESERVED},
                  script_sha256=digest(script), local_tests_sha256=digest(test_path),
                  nodes=8, gpus_per_node=4, world_size=32, cpus_per_node=32,
                  expected_auto_memory_mib_per_node=504000, explicit_mem_parameter=False,
                  phase_seconds=3600, prepare_seconds=10800, total_seconds=14220,
                  scope='UUID canonicalization and raw identity evidence only; original selection science and topology unchanged',
                  authorization='User requested bug fix and resubmission after primary 1685272 failed',
                  automations_remain_paused=True, validation_or_formal_submitted=False)
    write(proof, freeze)
    return freeze


def queue_guard(c):
    queue = remote(c, "squeue -h -u scxi717 -o '%i|%j|%T'", 60)
    if queue.strip():
        raise RuntimeError('Account has live jobs; inspect before new primary recovery: ' + queue)
    text = remote(c, 'sacct -j 1685272 -n -P --format=JobID,State,ExitCode', 60)
    rows = [r.split('|') for r in text.splitlines() if r.strip()]
    terminal = {'FAILED', 'COMPLETED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}
    if not rows or not any(r[:3] == ['1685272', 'FAILED', '15:0'] for r in rows) or any(r[1] not in terminal for r in rows):
        raise RuntimeError('Original job is not fully exited: ' + text)
    remote(c, 'test ! -e ' + q(GPU_BASE + '/selection_1685272'))
    return text


def submit():
    from ehe_5e9_workflow import connection
    job_path = REPORT / (STAGE + '_job.json')
    if job_path.exists():
        print('ALREADY_REGISTERED', read(job_path)['job'], flush=True)
        return
    controller_guard()
    registration = DATA / (STAGE + '_registration.json')
    intent = REPORT / (STAGE + '_submission_intent.json')
    if intent.exists():
        raise RuntimeError('Submission intent exists; recover scheduler identity instead of resubmitting')
    write(registration, dict(pid=os.getpid(), status='running', started_epoch=time.time()))
    exit_code = 1
    try:
        freeze = prepare()
        script = REPORT / 'launch_scripts' / (STAGE + '.sh')
        release = freeze['release']
        with connection('gpu') as c:
            prior = queue_guard(c)
            (REPORT / (STAGE + '_prior_failure_sacct.txt')).write_bytes(prior.encode())
            policy = remote(c, 'scontrol show partition gpu_5090')
            if 'DefCpuPerGPU=8' not in policy or 'DefMemPerCPU=15750' not in policy or 'State=UP' not in policy:
                raise ValueError('Actual 5090 scheduling policy changed')
            (REPORT / (STAGE + '_partition_policy.txt')).write_bytes(policy.encode())
            remote(c, 'test ! -e ' + q(release) + ' && mkdir -p ' + q(release))
            with c.open_sftp() as s:
                for name in freeze['sha256']:
                    remote(c, 'mkdir -p ' + q(release + '/' + str(Path(name).parent).replace('\\', '/')))
                    s.put(str(PAYLOAD / name), release + '/' + name)
                s.put(str(script), release + '/selection.sh')
            check = ('from pathlib import Path;from jscc_5e10_common import *;'
                     'p=Path(".");verify_files(p,read(p/"kernel_config.json")["files"]);'
                     'import jscc_5e10_selection;print("REPAIRED_32GPU_SOURCE_SHA_IMPORT_PASS")')
            proof = remote(c, 'cd ' + q(release) + ' && PYTHONDONTWRITEBYTECODE=1 ' + q(GPU_PYTHON) + ' -c ' + q(check), 180)
            old_release = read(REPORT / 'kernel_deployment.json')['release']
            tests = remote(c, 'cd ' + q(release) + ' && PYTHONDONTWRITEBYTECODE=1 JSCC_PROJECT_ROOT=' + q(release)
                           + ' JSCC_UUID_REPAIR_KERNEL=' + q(release) + ' JSCC_REPAIR_ORIGINAL_KERNEL=' + q(old_release)
                           + ' JSCC_TRIAL_ORIGINAL_KERNEL=' + q(old_release) + ' ' + q(GPU_PYTHON)
                           + ' -m unittest test_jscc_5e10_gpu_uuid test_jscc_5e10_5090_8x4_monitor_repair test_jscc_5e10 -v 2>&1', 180)
            if '\nOK' not in tests or 'skipped' in tests:
                raise ValueError('Actual Linux checks failed: ' + tests)
            (REPORT / (STAGE + '_linux_tests.txt')).write_bytes((proof + tests).encode())
            if remote(c, 'sha256sum ' + q(release + '/selection.sh')).split()[0] != digest(script):
                raise ValueError('Transferred launch script differs')
            remote(c, 'bash -n ' + q(release + '/selection.sh'))
            options = ('-p gpu_5090 --qos=gpugpu -N8 -n8 --ntasks-per-node=1 --cpus-per-task=32 '
                       '--gres=gpu:4 --time=240 --exclude=wqd10nba06g6 --chdir=/tmp --job-name=' + JOB_NAME
                       + ' --output=' + q(GPU_BASE + '/logs/' + STAGE + '.%j.out')
                       + ' --error=' + q(GPU_BASE + '/logs/' + STAGE + '.%j.err') + ' ' + q(release + '/selection.sh'))
            test_only = remote(c, 'sbatch --test-only ' + options + ' 2>&1', 60)
            for name, sha in freeze['preserved_records'].items():
                if digest(REPORT / name) != sha:
                    raise ValueError('Original record changed: ' + name)
            queue_guard(c)
            write(REPORT / (STAGE + '_launch_acceptance.json'), dict(passed=True,
                  freeze_sha256=digest(REPORT / (STAGE + '_freeze.json')), scheduler_test_only=test_only,
                  script_sha256=digest(script), linux_tests_sha256=digest(REPORT / (STAGE + '_linux_tests.txt')),
                  gpu_execution_verified=False, no_explicit_mem=True))
            write(intent, dict(stage=STAGE, job_name=JOB_NAME, command='sbatch --parsable ' + options,
                  script_sha256=digest(script), release=release, started_epoch=time.time(), resolved=False))
            number = remote(c, 'sbatch --parsable ' + options, 60).split(';')[0].strip()
            if not number.isdigit():
                raise RuntimeError('Ambiguous submission response; preserve intent and inspect queue: ' + number)
            record = dict(job=int(number), stage=STAGE, host='gpu', release=release,
                          output=GPU_BASE + '/' + STAGE + '_' + number,
                          allocation=GPU_BASE + '/allocations/' + STAGE + '_' + number + '.txt',
                          repaired_job=1685272, nodes=8, gpus_per_node=4, world_size=32, cpus_per_node=32,
                          partition='gpu_5090', explicit_mem_parameter=False, walltime_minutes=240,
                          phase_limit_seconds=3600, prepare_limit_seconds=10800, total_limit_seconds=14220,
                          freeze_sha256=digest(REPORT / (STAGE + '_freeze.json')), script_sha256=digest(script),
                          preserved_records=freeze['preserved_records'], submitted_epoch=time.time())
            write(job_path, record)
            write(intent, dict(read(intent), resolved=True, registered_job=int(number)))
            snapshot = remote(c, 'scontrol show job -o ' + number, 60)
            (REPORT / (STAGE + '_submit_scontrol.txt')).write_bytes(snapshot.encode())
            if not submitted_topology_matches(snapshot):
                raise ValueError('Actual submitted topology differs; preserve registered job and inspect')
            write(REPORT / (STAGE + '_submission_acceptance.json'), dict(passed=True, job=int(number),
                  job_record_sha256=digest(job_path), actual_scontrol_sha256=digest(REPORT / (STAGE + '_submit_scontrol.txt')),
                  preserved_records=freeze['preserved_records'], science_byte_unchanged=True,
                  gpu_execution_acceptance_pending=True, automations_remain_paused=True))
        print('REGISTERED_REPAIRED_5090_8X4_SELECTION', number, flush=True)
        exit_code = 0
    finally:
        write(registration, dict(read(registration), status='complete', exit_code=exit_code, finished_epoch=time.time()))


def status():
    from ehe_5e9_workflow import connection
    record = read(REPORT / (STAGE + '_job.json'))
    with connection('gpu') as c:
        account = remote(c, 'sacct -j ' + str(record['job']) + ' -n -P --format=JobID,State,ExitCode,Elapsed,MaxRSS', 60)
        queue = remote(c, "squeue -h -u scxi717 -o '%i|%j|%T|%M|%R'", 60)
    result = dict(job=record['job'], accounting=account, queue=queue, checked_epoch=time.time(), read_only=True)
    write(REPORT / (STAGE + '_latest_status.json'), result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('submit', 'status'))
    args = parser.parse_args()
    globals()[args.command]()
