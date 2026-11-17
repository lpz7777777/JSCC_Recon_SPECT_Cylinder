"""Submit only failed 1749085_7 with an immutable allocation adapter."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from ehe_dense_surface_workflow import (
    DATA, PAYLOAD, REPORT, REMOTE, GPU_PYTHON, connection, command, q,
    read, write, digest, verify_files,
)

NAME = 'EHE_dense256_retry7'
LOCAL = DATA / 'allocation_repair_7'
SOURCE = Path(__file__).with_name('ehe_dense_surface_allocation_repair.py')
FREEZE = REPORT / 'allocation_repair_freeze.json'
JOB = REPORT / 'allocation_repair_job.json'
INTENT = REPORT / 'allocation_repair_submission_intent.json'


def prepare():
    if FREEZE.exists():
        f = read(FREEZE)
        verify_files(LOCAL, f['files'])
        return f
    original = read(REPORT / 'formal_job.json')
    frozen = read(REPORT / 'freeze.json')
    if original['job'] != 1749085:
        raise ValueError('Repair is bounded to original 1749085_7')
    verify_files(PAYLOAD, frozen['files'])
    LOCAL.mkdir(exist_ok=False)
    adapter = LOCAL / SOURCE.name
    shutil.copyfile(SOURCE, adapter)
    manifest = {
        'original_job': 1749085, 'index': 7, 'energy_keV': 440, 'slab': 3,
        'original_release': REMOTE + '/releases/' + frozen['key'],
        'original_contract_sha256': frozen['contract_sha256'],
        'original_freeze_sha256': digest(REPORT / 'freeze.json'),
        'output': original['output'], 'files': {adapter.name: digest(adapter)},
        'scientific_runtime_unchanged': True, 'original_source_files': frozen['files'],
        'repair_scope': 'array-element-specific actual Slurm AllocTRES query and identity selection only',
        'successful_or_running_blocks_untouched': list(range(7)),
        'maximum_additional_GPUs': 1, 'maximum_study_concurrent_GPUs': 8,
        'no_separate_validation_job': True, 'no_timer': True,
        'authorization': '2026-10-29 human requested resubmission of the failed element and runtime estimate',
    }
    import hashlib
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    manifest['key'] = key
    write(LOCAL / 'repair_manifest.json', manifest)
    result = {'key': key, 'release': REMOTE + '/repairs/' + key,
              'files': {p.name: digest(p) for p in LOCAL.iterdir() if p.is_file()},
              'original_contract_sha256': frozen['contract_sha256']}
    write(FREEZE, result)
    return result


def deploy():
    f = prepare()
    if (REPORT / 'allocation_repair_deployment.json').exists():
        return f
    with connection('gpu') as c:
        command(c, 'mkdir -p ' + q(f['release']), 30)
        with c.open_sftp() as s:
            s.get_channel().settimeout(40)
            for name, expected in f['files'].items():
                target = f['release'] + '/' + name
                try:
                    with s.open(target, 'rb') as h:
                        import hashlib
                        actual = hashlib.sha256(h.read()).hexdigest()
                    if actual != expected:
                        raise ValueError('Preserve different remote repair source: ' + name)
                except FileNotFoundError:
                    with s.open(target, 'wx') as h:
                        h.write((LOCAL / name).read_bytes())
                    with s.open(target, 'rb') as h:
                        import hashlib
                        if hashlib.sha256(h.read()).hexdigest() != expected:
                            raise ValueError('Repair upload SHA differs')
        # Login-node CPU/identity check, no Slurm validation job and no matrix work.
        code = '''import hashlib,json,pathlib,sys,importlib.util
p=pathlib.Path(sys.argv[1]);expected=json.loads(sys.argv[2])
for n,h in expected.items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h
m=json.loads((p/'repair_manifest.json').read_text());old=pathlib.Path(m['original_release'])
assert hashlib.sha256((old/'contract.json').read_bytes()).hexdigest()==m['original_contract_sha256']
spec=importlib.util.spec_from_file_location('adapter',p/'ehe_dense_surface_allocation_repair.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
import subprocess
raw=subprocess.check_output(['scontrol','show','job','-o','1749085'],text=True)
record,fields=a.select_record(raw,'1749085','1749085','7');assert fields['JobState']=='FAILED'
sys.path.insert(0,str(old));import ehe_common
assert ehe_common.allocated_bytes(record)==94500*1024**2
for n,h in m['original_source_files'].items():assert hashlib.sha256((old/n).read_bytes()).hexdigest()==h,n
print('REPAIR_SHA_ARRAY_IDENTITY_AND_UNCHANGED_SCIENCE_PASS')
'''
        result = command(c, q(GPU_PYTHON) + ' -B -c ' + q(code) + ' ' + q(f['release']) + ' ' + q(json.dumps(f['files'])), 90)
        if 'REPAIR_SHA_ARRAY_IDENTITY_AND_UNCHANGED_SCIENCE_PASS' not in result:
            raise ValueError('Actual Linux repair check did not confirm success')
    write(REPORT / 'allocation_repair_deployment.json', {'passed': True, 'key': f['key'], 'result': result,
                                                       'source_sha256': digest(SOURCE)})
    return f


def submit():
    if JOB.exists():
        print(json.dumps(read(JOB), indent=2));return
    if INTENT.exists():
        raise RuntimeError('Unresolved repair sbatch intent; inspect Slurm, never resubmit blindly')
    f = deploy()
    original = read(REPORT / 'formal_job.json')
    script = '''#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.9
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1
export CUDA_CACHE_PATH="/tmp/ehe_dense_repair_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$CUDA_CACHE_PATH"
''' + q(GPU_PYTHON) + ' -B ' + q(f['release'] + '/' + SOURCE.name) + ' --manifest ' + q(f['release'] + '/repair_manifest.json') + '\n'
    local_script = LOCAL / 'formal_retry7.sh'
    local_script.write_text(script, encoding='utf8', newline='\n')
    remote_script = f['release'] + '/formal_retry7.sh'
    with connection('gpu') as c:
        queue = command(c, 'squeue -h -r -u "$USER" -o "%i|%j|%T|%R"', 35)
        if NAME in queue or '1749085_7|' in queue:
            raise RuntimeError('Target or repair already active; preserve without duplicate submit')
        if len(queue.strip().splitlines()) >= 40:
            raise RuntimeError('Account queue headroom unavailable; do not cancel other work')
        acct = command(c, 'sacct -j 1749085 --parsable2 -n -o JobID,State,ExitCode', 40)
        rows = {p[0]: p[1:3] for line in acct.splitlines() if len(p := line.split('|')) >= 3}
        if rows.get('1749085_7') != ['FAILED', '1:0']:
            raise RuntimeError('Original target is not confirmed fully exited FAILED/1:0')
        if any(rows.get('1749085_' + str(i), [None])[0] not in ('RUNNING', 'COMPLETED') for i in range(7)):
            raise RuntimeError('Other block state changed; inspect before bounded repair')
        code = '''import pathlib,shutil,sys
p=pathlib.Path(sys.argv[1]);assert not (p/'A440/slab_3').exists(),'Target block already exists; preserve it'
assert not (p/'completion.json').exists();assert not (p/'conversion_claim').exists()
free=shutil.disk_usage(p).free;assert free>=80*1024**3,free
print('FAILED_TARGET_ABSENT_STORAGE_PASS',free)
'''
        storage = command(c, q(GPU_PYTHON) + ' -B -c ' + q(code) + ' ' + q(original['output']), 35)
        if 'FAILED_TARGET_ABSENT_STORAGE_PASS' not in storage:
            raise RuntimeError('Remote target/storage preflight was not confirmed')
        with c.open_sftp() as s:
            try:
                with s.open(remote_script, 'rb') as h:
                    if h.read() != local_script.read_bytes():raise ValueError('Different retry script exists')
            except FileNotFoundError:
                with s.open(remote_script, 'wx') as h:h.write(local_script.read_bytes())
        request = {'original_job': 1749085, 'index': 7, 'key': f['key'], 'array': '7',
                   'release': f['release'], 'output': original['output'], 'GPUs': 1, 'CPU': 6,
                   'Slurm_minutes': 4320, 'remote_script': remote_script, 'script_sha256': digest(local_script),
                   'submitted_epoch': time.time(), 'queue_before': queue, 'accounting_before': acct,
                   'storage_preflight': storage, 'source_sha256': digest(__file__)}
        write(INTENT, request)
        cmd = ('sbatch --parsable --partition=gpu_5090 --account=scxi717 --qos=gpugpu '
               '-N 1 -n 1 -c 6 --gres=gpu:1 --array=7 --time=4320 --job-name=' + NAME +
               ' --comment=' + f['key'] + ' --output=' + q(REMOTE + '/logs/repair_%A_%a.log') + ' ' + q(remote_script))
        result = command(c, cmd, 60).strip()
        job = result.split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Uncertain repair sbatch outcome; intent retained, inspect Slurm')
        request.update(job=int(job), sbatch_result=result)
        write(JOB, request)
    print('BOUNDED_FORMAL_REPAIR_SUBMITTED', job, 'array=7 original=1749085_7', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'deploy', 'submit'])
    getattr(sys.modules[__name__], parser.parse_args().action)()
