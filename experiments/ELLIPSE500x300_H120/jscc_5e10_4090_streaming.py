"""User-authorized 8x3 RTX4090 pipeline, reusing the accepted selection."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from jscc_5e10_common import *
from jscc_5e10_workflow import connection, command, q, put_archive, job_state
from jscc_5e10_reconstruction_workflow import FACTORS, slurm_peak_bytes
from jscc_5e10_5090_8x2_trial import live_registration
from jscc_5e10_8x2_selection_verification import strict_get
from jscc_5e10_streaming_cache import POLICY

PREFIX = 'streaming_4090_8x3'
PAYLOAD = DATA / (PREFIX + '_payload')
GUARDS = ('advance', 'selection_4090_trial', 'selection_5090_8x2_trial',
          'selection_5090_8x2_monitor_repair', 'selection_5090_8x2_monitor_repair_verification', PREFIX)
SOURCES = ('jscc_5e10_streaming_cache.py', 'jscc_5e10_streaming_contract.py',
           'jscc_5e10_streaming_runtime.py', 'jscc_5e10_streaming_probe.py',
           'run_jscc_5e10_streaming.py', 'verify_jscc_5e10_streaming.py',
           'jscc_5e10_streaming_verify_stage.py', 'test_jscc_5e10_streaming.py',
           'jscc_5e10_4090_streaming.py', 'jscc_5e10_gpu_uuid.py')


def record_path(stage, kind='job'):
    return REPORT / (PREFIX + '_' + stage + '_' + kind + '.json')


def prepare():
    path = REPORT / (PREFIX + '_freeze.json')
    if path.exists():
        f = read(path)
        verify_files(PAYLOAD, f['sha256'])
        for name in SOURCES:
            if digest(HERE / name) != f['sha256'][name]:
                raise ValueError('Preserve frozen streaming source; bounded repair required: ' + name)
        return f
    accepted_path = REPORT / 'selection_5090_8x2_monitor_repair_acceptance.json'
    accepted = read(accepted_path)
    if not (accepted['passed'] and accepted['strict_fetch_passed'] and accepted['job'] == 1686507):
        raise ValueError('Actual complete strict accepted selection required')
    selection = Path(accepted['local_result'])
    verify_files(selection, accepted['strict_files_sha256'])
    manifest = read(selection / 'selection_manifest.json')
    original = read(REPORT / 'kernel_freeze.json')
    verify_files(DATA / 'kernel_payload', original['sha256'])
    shutil.copytree(DATA / 'kernel_payload', PAYLOAD)
    for name in SOURCES:
        shutil.copy2(HERE / name, PAYLOAD / name)
    for name in ('selections', 'selected_rows'):
        shutil.copytree(selection / name, PAYLOAD / name)
    shutil.copy2(selection / 'selection_manifest.json', PAYLOAD / 'selection_manifest.json')
    files = hashes(PAYLOAD)
    cfg = dict(read(PAYLOAD / 'kernel_config.json'), files=files, model='continuous_energy',
        event_policy='legacy', nodes=8, gpus_per_node=3, world_size=24, iterations=10000, save_step=50,
        channels=list(CHANNELS), regularization='none', initial_density=1, joint_solver_enabled=False,
        cross_prediction_source='440_SinglePhoton_final', events_per_view=manifest['events_per_view'],
        accepted_events=manifest['accepted_events'], response_storage_policy=POLICY,
        selection_manifest_sha256=digest(PAYLOAD / 'selection_manifest.json'),
        selection_acceptance_sha256=digest(accepted_path), selected_source_job=1686507,
        streaming_batch_events=32, disk_cache_location='node_local_/tmp',
        formal_reuses_accepted_validation_cache=True, original_5090_job_untouched=1685272,
        total_primary_photons=TOTAL)
    write(PAYLOAD / 'contract.json', cfg)
    from jscc_5e10_streaming_contract import load_contract
    load_contract(PAYLOAD / 'contract.json', 'validation', 10, 10)
    env = dict(os.environ, JSCC_PROJECT_ROOT=str(PAYLOAD))
    tests = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'unittest', 'test_jscc_5e10_streaming', '-v'],
        cwd=PAYLOAD, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    (REPORT / (PREFIX + '_local_tests.txt')).write_bytes(tests.stdout)
    if tests.returncode:
        raise ValueError('Streaming local numerical/damage-rejection tests failed')
    files = hashes(PAYLOAD)
    key = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    freeze = dict(release_key=key, sha256=files, release=GPU_BASE + '/streaming_releases/' + key,
        contract_sha256=digest(PAYLOAD / 'contract.json'), source_selection_acceptance_sha256=digest(accepted_path),
        source_selection_job=1686507, original_selection_registry_sha256=digest(REPORT / 'selection_job.json'),
        original_4090_trial_registry_sha256=digest(REPORT / 'selection_4090_trial_job.json'),
        local_tests_sha256=digest(REPORT / (PREFIX + '_local_tests.txt')),
        nodes=8, gpus_per_node=3, world_size=24, accepted_events=cfg['accepted_events'],
        response_storage_policy=POLICY, original_scientific_helpers_unchanged=True)
    write(path, freeze)
    return freeze


def deploy(f):
    path = REPORT / (PREFIX + '_deployment.json')
    if path.exists():
        return read(path)
    with connection('gpu') as c:
        free = int(command(c, 'df -B1 --output=avail ' + q(GPU_PROJECT) + ' | tail -n1'))
        if free < 8 << 30:
            raise OSError('Shared disk reserve is insufficient; no cache writes to shared storage')
        put_archive(c, PAYLOAD, f['release'])
        code = ('from pathlib import Path;from jscc_5e10_common import verify_files;'
                'verify_files(Path("."),' + repr(f['sha256']) + ');'
                'import run_jscc_5e10_streaming,verify_jscc_5e10_streaming;print("STREAMING_IMPORT_SHA_PASSED")')
        result = command(c, 'cd ' + q(f['release']) + ' && ' + q(GPU_PYTHON) + ' -c ' + q(code), timeout=120)
        tests = command(c, 'cd ' + q(f['release']) + ' && PYTHONDONTWRITEBYTECODE=1 JSCC_PROJECT_ROOT=' + q(f['release']) +
            ' ' + q(GPU_PYTHON) + ' -m unittest test_jscc_5e10_streaming -v 2>&1', timeout=120)
        if '\nOK' not in tests or 'skipped' in tests:
            raise ValueError('Actual Linux streaming tests failed: ' + tests)
        (REPORT / (PREFIX + '_linux_tests.txt')).write_bytes(tests.encode())
        write(path, dict(release=f['release'], sha256=f['sha256'], actual_linux_import_and_sha=result,
            linux_tests_sha256=digest(REPORT / (PREFIX + '_linux_tests.txt')), shared_free_bytes=free,
            all_response_cache_writes_node_local_only=True))
    return read(path)


def launcher(stage, release, phase, prep, total):
    name = PREFIX + '_' + stage
    args = [release + ('/jscc_5e10_streaming_probe.py' if stage == 'probe' else '/run_jscc_5e10_streaming.py'),
            '--contract', release + '/contract.json', '--factors', FACTORS,
            '--output', '$output', '--allocation', '$allocation']
    authority_env = ''
    if stage != 'probe':
        args += ['--input-root', GPU_BASE + '/input', '--mode', stage]
    if stage == 'formal':
        authority = read(record_path('formal', 'authority'))
        args += ['--authority', authority['remote'], '--authority-sha256', authority['sha256']]
        authority_env = 'export JSCC_VALIDATION_RESULT=' + q(authority['result']) + '\n'
    invocation = ' '.join('"' + x + '"' if x.startswith('$') else q(x) for x in args)
    return '''#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
[[ "$SLURM_NNODES" == 8 && "$SLURM_NTASKS" == 8 ]]
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 JSCC_PROJECT_ROOT=''' + q(release) + '''
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=bond0 TORCH_ELASTIC_WORKER_IDENTICAL=1
export JSCC_PHASE_SECONDS=''' + str(phase) + ' JSCC_PREPARE_SECONDS=' + str(prep) + '''
export JSCC_MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export JSCC_MASTER_PORT=$((40000+SLURM_JOB_ID%10000))
allocation=''' + q(GPU_BASE + '/allocations/' + name + '_') + '''"${SLURM_JOB_ID}.txt"
scontrol show job "$SLURM_JOB_ID" > "$allocation"
export JSCC_HOST_BYTES_NODE=$(cd "$JSCC_PROJECT_ROOT" && ''' + q(GPU_PYTHON) + ''' -c 'import sys;from jscc_5e10_common import host_allocated_bytes;print(host_allocated_bytes(open(sys.argv[1]).read(),8))' "$allocation")
output=''' + q(GPU_BASE + '/' + name + '_') + '''"${SLURM_JOB_ID}"
''' + authority_env + '''timeout --signal=TERM --kill-after=60s ''' + str(total) + '''s srun --chdir=/tmp --label --kill-on-bad-exit=1 bash -c '
  exec ''' + q(GPU_PYTHON) + ''' -m torch.distributed.run --nnodes=8 --nproc_per_node=3 --node_rank="$SLURM_PROCID" \
    --master_addr="$JSCC_MASTER_ADDR" --master_port="$JSCC_MASTER_PORT" --rdzv_backend=static \
    --rdzv_conf=timeout=600 --rdzv_id="${SLURM_JOB_ID}_streaming" --max_restarts=0 "${@:1}"
' bash ''' + invocation + '\necho JSCC5E10_STREAMING_COMPLETED ' + stage + '\n'


def submit(stage, deployment):
    if record_path(stage).exists():
        return True
    minutes, phase, prep = (30, 600, 1200) if stage == 'probe' else (240, 3600, 10800)
    hosts = None
    if stage == 'formal':
        a = read(record_path('formal', 'authority'))
        phase = max(1800, math.ceil(a['estimated_max_phase_seconds'] * 1.5 + 300))
        prep = max(1800, math.ceil(a['measured_prepare_seconds'] * 1.5 + 300))
        minutes = max(120, math.ceil((a['estimated_total_seconds'] * 1.5 + 1800) / 60))
        if phase > 172800 or minutes > 5760:
            raise ValueError('Measured streaming throughput exceeds bounded four-day budget; diagnose before formal')
        hosts = a['cache_nodes']
    total = minutes * 60 - 180
    script = launcher(stage, deployment['release'], phase, prep, total)
    local = DATA / (PREFIX + '_' + stage + '.sh')
    local.write_bytes(script.encode())
    remote = deployment['release'] + '/launch_' + stage + '.sh'
    intent = record_path(stage, 'submission_intent')
    if intent.exists():
        raise ValueError('Unresolved streaming submit intent; inspect scheduler')
    with connection('gpu') as c:
        queue = command(c, "squeue -h -u scxi717 -o '%i|%j|%T'")
        if len(queue.splitlines()) >= 50:
            return False
        # The original 5090 job is explicitly allowed to remain queued/running.
        if any('|JSCC5e10_stream_' in x for x in queue.splitlines()):
            raise ValueError('Existing streaming computation is still active')
        command(c, 'test ! -e ' + q(remote))
        with c.open_sftp() as s:
            s.put(str(local), remote)
        if command(c, 'sha256sum ' + q(remote)).split()[0] != digest(local):
            raise ValueError('Streaming launch SHA differs')
        command(c, 'bash -n ' + q(remote))
        line = ('sbatch --parsable -p gpu_4090 --qos=gpugpu -N8 -n8 --ntasks-per-node=1 '
                '--cpus-per-task=18 --gres=gpu:3 --time=' + str(minutes) +
                ' --exclude=wqd10nba06g6 --chdir=/tmp --job-name=JSCC5e10_stream_' + stage +
                ' --output=' + q(GPU_BASE + '/logs/' + PREFIX + '_' + stage + '.%j.out') +
                ' --error=' + q(GPU_BASE + '/logs/' + PREFIX + '_' + stage + '.%j.err') +
                (' --nodelist=' + q(','.join(hosts)) if hosts else '') + ' ' + q(remote))
        check = command(c, line.replace('sbatch --parsable', 'sbatch --test-only'), timeout=60)
        write(intent, dict(stage=stage, script_sha256=digest(local), command=line, test_only=check, started_epoch=time.time()))
        number = command(c, line, timeout=60).split(';')[0].strip()
        if not number.isdigit():
            raise ValueError('Ambiguous streaming submission outcome')
        control = command(c, 'scontrol show job ' + number, timeout=60)
    write(record_path(stage), dict(job=int(number), stage=PREFIX + '_' + stage, mode=stage, host='gpu',
        release=deployment['release'], output=GPU_BASE + '/' + PREFIX + '_' + stage + '_' + number,
        allocation=GPU_BASE + '/allocations/' + PREFIX + '_' + stage + '_' + number + '.txt',
        nodes=8, gpus_per_node=3, world_size=24, partition='gpu_4090', cpus_per_node=18,
        explicit_mem_parameter=False, walltime_minutes=minutes, phase_limit_seconds=phase,
        prepare_limit_seconds=prep, total_limit_seconds=total, script_sha256=digest(local),
        submitted_epoch=time.time(), cache_nodes=hosts, submit_scontrol=control))
    write(intent, dict(read(intent), resolved=True, registered_job=int(number)))
    print('REGISTERED_STREAMING_8X3', stage, number, flush=True)
    return True


def accept_probe():
    path = record_path('probe', 'acceptance')
    if path.exists():
        return True
    source = read(record_path('probe'))
    with connection('gpu') as c:
        state, acc = job_state(c, source)
        if state != 'complete':
            return False
        folder = REPORT / (PREFIX + '_probe_evidence')
        folder.mkdir(exist_ok=True)
        strict_get(c, source['output'] + '/probe.json', folder / 'probe.json')
        strict_get(c, source['allocation'], folder / 'allocation.txt')
        for ext in ('out', 'err'):
            strict_get(c, GPU_BASE + '/logs/' + PREFIX + '_probe.' + str(source['job']) + '.' + ext, folder / ('original.' + ext))
        proof = read(folder / 'probe.json')
        from jscc_5e10_streaming_contract import verify_topology
        memory = verify_topology(proof['resources'], folder / 'allocation.txt')
        rss = slurm_peak_bytes(acc)
        if not proof['passed'] or proof['contract_sha256'] != digest(PAYLOAD / 'contract.json') or rss > .8 * memory:
            raise ValueError('Actual streaming storage/resource probe not accepted; preserve evidence')
        write(path, dict(passed=True, job=source['job'], accounting=acc, proof_sha256=digest(folder / 'probe.json'),
            evidence_sha256=hashes(folder), node_storage_budget=proof['node_storage_budget'],
            nodes=sorted({r['node'] for r in proof['resources']}), actual_allocated_bytes_node=memory,
            slurm_maxrss_bytes=rss, complete_input_validation_still_required=True))
    return True


def submit_verification(stage):
    path = record_path(stage, 'verification_job')
    if path.exists():
        return True
    source = read(record_path(stage))
    with connection('gpu') as c:
        state, acc = job_state(c, source)
        if state != 'complete':
            return False
    fpath = record_path(stage, 'verification_freeze')
    payload = DATA / (PREFIX + '_' + stage + '_verification_payload')
    if not fpath.exists():
        original = read(REPORT / (PREFIX + '_freeze.json'))
        verify_files(PAYLOAD, original['sha256'])
        shutil.copytree(PAYLOAD, payload)
        write(payload / 'verification_release.json', dict(sha256=hashes(payload), read_only=True,
            scientific_release_sha256=original['sha256']))
        files = hashes(payload)
        key = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
        write(fpath, dict(release_key=key, sha256=files, release=GPU_BASE + '/verification_releases/' + key))
    f = read(fpath)
    output = GPU_BASE + '/acceptance_' + PREFIX + '_' + stage + '_' + str(source['job'])
    intent = record_path(stage, 'verification_submission_intent')
    if intent.exists():
        raise ValueError('Unresolved streaming verification intent')
    with connection('gpu') as c:
        if not record_path(stage, 'verification_deployment').exists():
            put_archive(c, payload, f['release'])
            write(record_path(stage, 'verification_deployment'), dict(release=f['release'], sha256=f['sha256']))
        args = ['--release', f['release'], '--input', GPU_BASE + '/input', '--factors', FACTORS,
                '--result', source['output'], '--allocation', source['allocation'], '--output', output, '--mode', stage]
        script = '#!/usr/bin/env bash\nset -euo pipefail\nsource /etc/profile.d/modules.sh\nmodule load miniforge3/25.11.0-1\n'
        script += 'export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1 JSCC_PROJECT_ROOT=' + q(f['release']) + '\n'
        script += 'cd ' + q(f['release']) + '\ntimeout --signal=TERM --kill-after=60s 6900s ' + q(GPU_PYTHON) + ' -u ' + q(f['release'] + '/jscc_5e10_streaming_verify_stage.py') + ' ' + ' '.join(q(x) for x in args) + '\n'
        local = DATA / (PREFIX + '_' + stage + '_verification.sh')
        local.write_bytes(script.encode())
        remote = f['release'] + '/launch_verification.sh'
        command(c, 'test ! -e ' + q(remote))
        with c.open_sftp() as s:
            s.put(str(local), remote)
        if command(c, 'sha256sum ' + q(remote)).split()[0] != digest(local):
            raise ValueError('Independent verifier script differs')
        command(c, 'bash -n ' + q(remote))
        write(intent, dict(source_job=source['job'], script_sha256=digest(local)))
        line = 'sbatch --parsable -p gpu_4090 --qos=gpugpu -N1 -n1 --cpus-per-task=6 --gres=gpu:1 --time=120 --exclude=wqd10nba06g6 --chdir=/tmp --job-name=JSCC5e10_stream_verify'
        line += ' --output=' + q(GPU_BASE + '/logs/' + PREFIX + '_' + stage + '_verification.%j.out')
        line += ' --error=' + q(GPU_BASE + '/logs/' + PREFIX + '_' + stage + '_verification.%j.err') + ' ' + q(remote)
        number = command(c, line, timeout=60).split(';')[0].strip()
        if not number.isdigit():
            raise ValueError('Ambiguous streaming verification submit')
    write(path, dict(job=int(number), stage=PREFIX + '_' + stage + '_verification', host='gpu', mode=stage,
        release=f['release'], output=output, source_job=source['job'], source_accounting=acc,
        script_sha256=digest(local), read_only_verification=True, reserved_gpus=1, explicit_mem_parameter=False))
    write(intent, dict(read(intent), resolved=True, registered_job=int(number)))
    return True


def fetch(stage):
    path = record_path(stage, 'acceptance')
    if path.exists():
        return True
    record = read(record_path(stage, 'verification_job'))
    source = read(record_path(stage))
    with connection('gpu') as c:
        state, acc = job_state(c, record)
        if state != 'complete':
            return False
        state, source_acc = job_state(c, source)
        if state != 'complete':
            raise ValueError('Scientific job must have exited successfully')
        evidence = REPORT / (PREFIX + '_' + stage + '_evidence')
        evidence.mkdir(exist_ok=True)
        strict_get(c, record['output'] + '/package.json', evidence / 'package.json')
        package = read(evidence / 'package.json')
        archive = DATA / (PREFIX + '_' + stage + '_accepted.tar.gz')
        if not archive.exists():
            partial = archive.with_suffix('.partial')
            if partial.exists():
                raise ValueError('Preserve interrupted strict archive transfer')
            strict_get(c, record['output'] + '/accepted_result.tar.gz', partial)
            os.replace(partial, archive)
        if digest(archive) != package['archive_sha256']:
            raise ValueError('Strict complete archive SHA differs')
        destination = DATA / (PREFIX + '_' + stage + '_accepted_' + str(source['job']))
        if not destination.exists():
            destination.mkdir()
            with tarfile.open(archive) as t:
                members = t.getmembers()
                if len(members) != len(package['files']) or {x.name for x in members} != set(package['files']):
                    raise ValueError('Strict archive member closure differs')
                if any(not x.isfile() or Path(x.name).is_absolute() or '..' in Path(x.name).parts for x in members):
                    raise ValueError('Unsafe strict archive member')
                t.extractall(destination)
        verify_files(destination, package['files'])
        proof = read(destination / 'verification.json')
        if not proof['passed'] or proof['mode'] != stage or proof['actual_primary_photons'] != TOTAL or proof['contract_sha256'] != digest(PAYLOAD / 'contract.json'):
            raise ValueError('Actual current streaming full-input strict identity differs')
        memory = host_allocated_bytes((destination / 'allocation.txt').read_text(), 8)
        rss = slurm_peak_bytes(source_acc)
        if rss > .8 * memory:
            raise ValueError('Actual exited Slurm memory margin fails')
        for item in (source, record):
            for ext in ('out', 'err'):
                strict_get(c, GPU_BASE + '/logs/' + item['stage'] + '.' + str(item['job']) + '.' + ext,
                           evidence / (item['stage'] + '.' + ext))
        write(path, dict(passed=True, strict_fetch_passed=True, study=STUDY, mode=stage, job=source['job'],
            verification_job=record['job'], local_result=str(destination), remote_result=source['output'],
            strict_files_sha256=package['files'], archive_sha256=package['archive_sha256'],
            verification_sha256=digest(destination / 'verification.json'), evidence_sha256=hashes(evidence),
            source_accounting=source_acc, verification_accounting=acc, slurm_maxrss_bytes=rss,
            actual_allocated_bytes_node=memory, world_size=24, nodes=8, gpus_per_node=3))
    return True


def issue_authority():
    path = record_path('formal', 'authority')
    if path.exists():
        return
    accepted = read(record_path('validation', 'acceptance'))
    proof = read(Path(accepted['local_result']) / 'verification.json')
    resources = proof['resources']
    times = {k: max(r['phase_solve_seconds'][k] for r in resources) for k in PHASE_CHANNELS}
    prepare_seconds = max(r['prepare_seconds'] for r in resources)
    once = max(r['elapsed_seconds'] for r in resources) - sum(times.values())
    scaled = {k: v * 1000 for k, v in times.items()}
    authority = dict(passed=True, study=STUDY, contract_sha256=proof['contract_sha256'],
        result=accepted['remote_result'], evidence_sha256=proof['result_files_sha256'], validation_job=accepted['job'],
        strict_acceptance_sha256=digest(record_path('validation', 'acceptance')),
        measured_phase_seconds=times, measured_prepare_seconds=prepare_seconds, estimated_phase_seconds=scaled,
        estimated_max_phase_seconds=max(scaled.values()), estimated_total_seconds=sum(scaled.values()) + max(once, prepare_seconds, 0),
        actual_primary_photons=TOTAL, world_size=24, nodes=8, gpus_per_node=3,
        cache_nodes=sorted({r['node'] for r in resources}), response_storage_policy=POLICY,
        formal_reuses_exact_accepted_validation_cache=True, timing_estimate_is_not_completion_guarantee=True)
    local = DATA / (PREFIX + '_formal_authority.json')
    write(local, authority)
    remote = GPU_BASE + '/' + local.name
    with connection('gpu') as c:
        command(c, 'test ! -e ' + q(remote))
        with c.open_sftp() as s:
            s.put(str(local), remote)
        if command(c, 'sha256sum ' + q(remote)).split()[0] != digest(local):
            raise ValueError('Streaming formal authority transfer differs')
    write(path, dict(authority, remote=remote, sha256=digest(local)))


def status():
    with connection('gpu') as c:
        for stage in ('probe', 'validation', 'formal'):
            for kind in ('job', 'verification_job'):
                path = record_path(stage, kind)
                if path.exists():
                    record = read(path)
                    print(record['stage'], record['job'], job_state(c, record)[0], flush=True)
        primary = read(REPORT / 'selection_job.json')
        print('ORIGINAL_5090_UNTOUCHED', primary['job'], command(c, 'sacct -j ' + str(primary['job']) + ' -n -P --format=JobID,State,ExitCode'), flush=True)


def advance():
    deployment = deploy(prepare())
    if not submit('probe', deployment) or not accept_probe():
        print('WAITING_ACTUAL_24GPU_STORAGE_PROBE', flush=True)
        return
    for stage in ('validation', 'formal'):
        if stage == 'formal':
            issue_authority()
        if not submit(stage, deployment) or not submit_verification(stage) or not fetch(stage):
            print('WAITING_STREAMING_STAGE', stage, flush=True)
            return
    write(REPORT / (PREFIX + '_numerical_delivery.json'), dict(passed=True, channels=list(CHANNELS),
        formal_acceptance_sha256=digest(record_path('formal', 'acceptance')), iterations=10000,
        science_visual_qa_and_report_still_required=True, experiment_delivery_complete=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'advance', 'status'))
    action = parser.parse_args().action
    busy = [name for name in GUARDS if live_registration(DATA / (name + '_registration.json'))]
    if busy:
        raise RuntimeError('Existing controller PID is alive: ' + ', '.join(busy))
    if action == 'status':
        status()
    else:
        registration = DATA / (PREFIX + '_registration.json')
        write(registration, dict(pid=os.getpid(), action=action, status='running', started_epoch=time.time()))
        code = 1
        try:
            (prepare if action == 'prepare' else advance)()
            code = 0
        finally:
            write(registration, dict(read(registration), status='complete', exit_code=code, finished_epoch=time.time()))
