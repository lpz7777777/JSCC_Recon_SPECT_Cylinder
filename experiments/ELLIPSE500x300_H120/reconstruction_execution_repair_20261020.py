"""One bounded manual repair of the failed Derenzo440 and held NEMA5e10 chains.

Only launch I/O and measured execution bounds change. No simulation, selection,
scientific payload, original registration, or original pending job is modified.
"""
import argparse
import contextlib
import json
import os
import shutil
import time
from pathlib import Path
from paramiko import SSHException

import derenzo_campaign_workflow as dw
import jscc_5e10_reconstruction_workflow as nw
from ehe_5e9_workflow import connection, command, q
from jscc_5e10_common import HERE, GPU_PROJECT, GPU_PYTHON, read, write, digest, hashes
from reconstruction_execution_gate_20261020 import measured, check, allocation, MAX_MINUTES
from resume_reconstruction_controllers_20261019 import live_registration

TAG = 'execution_repair_20261020'
ROOT = HERE / 'reports' / TAG
DATA = HERE / 'generated' / TAG
REMOTE = GPU_PROJECT + '/generated/' + TAG
ORIGINAL = dict(derenzo=(dw.REPORT, dw.DATA, dw.GPU_BASE), nema=(nw.REPORT, nw.DATA, nw.GPU_BASE))
REPORTS = {k: r / TAG for k, (r, _, _) in ORIGINAL.items()}
DATAS = {k: d / TAG for k, (_, d, _) in ORIGINAL.items()}
REMOTES = {k: b + '/' + TAG for k, (_, _, b) in ORIGINAL.items()}
REGS = [DATA / 'controller_registration.json'] + [d / (TAG + '_registration.json') for _, d, _ in ORIGINAL.values()]
GATE_NAME = 'reconstruction_execution_gate_20261020.py'
TEST_NAME = 'test_reconstruction_execution_gate_20261020.py'


class ReadOnlyPollUnavailable(Exception):
    pass


def read_only_poll(function):
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (TimeoutError, ConnectionError) as error:
            raise ReadOnlyPollUnavailable(str(error)) from error
        except SSHException as error:
            if str(error) != 'Timeout opening channel.':
                raise
            raise ReadOnlyPollUnavailable(str(error)) from error
    return wrapped


def immutable(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Preserve different existing file: ' + str(path))
    else:
        with path.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())


def upload(c, local, remote):
    existing = command(c, 'if test -e ' + q(remote) + '; then sha256sum ' + q(remote) + '; fi').strip()
    if existing:
        if existing.split()[0] != digest(local):
            raise ValueError('Preserve different remote file: ' + remote)
    else:
        with c.open_sftp() as s:
            s.put(str(local), remote)
    if command(c, 'sha256sum ' + q(remote)).split()[0] != digest(local):
        raise ValueError('Strict execution file transfer differs')


@contextlib.contextmanager
def namespace(kind):
    module = dw if kind == 'derenzo' else nw
    old = module.REPORT, module.DATA, module.GPU_BASE
    module.REPORT, module.DATA, module.GPU_BASE = REPORTS[kind], DATAS[kind], REMOTES[kind]
    try:
        yield module
    finally:
        module.REPORT, module.DATA, module.GPU_BASE = old


def no_other_controller():
    paths = list(ORIGINAL['nema'][1].glob('*registration.json'))
    paths += list(ORIGINAL['derenzo'][1].glob('*registration.json')) + REGS[:1]
    alive = [str(p) for p in paths if live_registration(p) and read(p).get('pid') != os.getpid()]
    if alive:
        raise RuntimeError('Another actual controller is alive: ' + str(alive))


def tmp_environment(release, root):
    return ('#!/usr/bin/env bash\n'
            'umask 077\n'
            'export TMPDIR=' + q(root + '/tmp/job_') + '"${SLURM_JOB_ID}/node_${SLURMD_NODENAME:-$HOSTNAME}"\n'
            'export TMP="$TMPDIR" TEMP="$TMPDIR" PYTHONDONTWRITEBYTECODE=1\n'
            'mkdir -p -- "$TMPDIR"\n'
            'cd ' + q(root) + '\n' + q(GPU_PYTHON) + ' -B ' + q(release + '/' + GATE_NAME)
            + ' --tmp-root ' + q(root) + '\n')


def repair_script(script, kind, release):
    env = release + '/' + kind + '_tmp.sh'
    script = script.replace('set -euo pipefail\n', 'set -euo pipefail\nsource ' + q(env) + '\n', 1)
    script = script.replace('--chdir=/tmp', '--chdir=' + q(REMOTES[kind]))
    # Each remote parent task gets its own node-specific temporary path before torchrun.
    script = script.replace("bash -c 'exec ", "bash -c 'source " + q(env) + '; exec ')
    script = script.replace("bash -c '\n  exec ", "bash -c 'source " + q(env) + '\n  exec ')
    if kind == 'nema':
        old = 'srun --chdir=' + q(REMOTES[kind]) + ' --label --kill-on-bad-exit=1 ' + q(GPU_PYTHON)
        script = script.replace(old, 'srun --chdir=' + q(REMOTES[kind])
                                + ' --label --kill-on-bad-exit=1 bash -c \'source ' + q(env)
                                + '; exec "$@"\' bash ' + q(GPU_PYTHON), 1)
    return script


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    for kind, (r, d, b) in ORIGINAL.items():
        REPORTS[kind].mkdir(parents=True, exist_ok=True); DATAS[kind].mkdir(parents=True, exist_ok=True)
        (DATAS[kind] / 'launch_scripts').mkdir(exist_ok=True)
    for name in ('gpu_deployment.json', 'gpu_freeze.json'):
        immutable(REPORTS['derenzo'] / name, (ORIGINAL['derenzo'][0] / name).read_bytes())
    for name in ('validation_plan.json', 'validation_plan_deployment.json', 'selection_job.json', 'selection_acceptance.json'):
        immutable(REPORTS['derenzo'] / 'JSCC_440' / name,
                  (ORIGINAL['derenzo'][0] / 'JSCC_440' / name).read_bytes())
    (DATAS['derenzo'] / 'JSCC_440').mkdir(exist_ok=True)
    # Only small evidence and execution sources are frozen here, never response arrays.
    names = (GATE_NAME, TEST_NAME, Path(__file__).name)
    identity = {n: digest(HERE / n) for n in names}
    key = __import__('hashlib').sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    release = REMOTE + '/releases/' + key
    payload = DATA / ('release_' + key)
    for name in names:
        immutable(payload / name, (HERE / name).read_bytes())
    for kind in ORIGINAL:
        immutable(payload / (kind + '_tmp.sh'), tmp_environment(release, REMOTES[kind]).encode())
    frozen = dict(release=release, files=hashes(payload), source_identity=identity,
                  scientific_payloads_unchanged=True, policy_max_minutes=MAX_MINUTES)
    immutable(ROOT / 'freeze.json', (json.dumps(frozen, indent=2) + '\n').encode())
    with connection('gpu') as c:
        for root in [release] + list(REMOTES.values()):
            command(c, 'mkdir -p ' + ' '.join(q(root + '/' + x) for x in ('', 'scripts', 'logs', 'allocations', 'JSCC_440')))
        for name in frozen['files']:
            upload(c, payload / name, release + '/' + name)
        for kind in ORIGINAL:
            command(c, 'bash -n ' + q(release + '/' + kind + '_tmp.sh'))
        tests = command(c, 'cd ' + q(release) + ' && TMPDIR=' + q(release) + ' ' + q(GPU_PYTHON)
                        + ' -B -m unittest -v ' + TEST_NAME[:-3] + ' 2>&1', timeout=120)
        if '\nOK' not in tests or 'skipped' in tests:
            raise RuntimeError('Actual Linux execution-repair tests did not all pass')
        test_path = ROOT / 'linux_tests.txt'
        if not test_path.exists():
            immutable(test_path, tests.encode())
        if not (ROOT / 'baseline.json').exists():
            baseline = command(c, 'sacct -j 1713939 -n -P --format=JobID,State,ExitCode,Elapsed,MaxRSS', timeout=60)
            rows = [x.split('|') for x in baseline.splitlines() if x.strip()]
            if not rows or any(x[1] not in ('FAILED', 'COMPLETED') for x in rows):
                raise RuntimeError('Failed original job has not completely exited')
            snapshots = {str(j): command(c, 'scontrol show job -o ' + str(j)) for j in (1713940, 1713943, 1704572)}
            if any('JobState=PENDING' not in text for text in snapshots.values()):
                raise RuntimeError('Original chain state changed; inspect before replacement')
            old = ORIGINAL['derenzo'][2]
            absent = command(c, 'test ! -e ' + q(old + '/JSCC_440/validation')
                             + ' && test ! -e ' + q(old + '/JSCC_440/formal') + ' && echo ORIGINAL_SCIENCE_NOT_STARTED')
            policy = command(c, 'scontrol show partition gpu_5090')
            qos = command(c, 'sacctmgr -n -P show qos gpugpu format=Name,MaxWall,MaxTRESPJ')
            free = int(command(c, 'df -B1 --output=avail ' + q(REMOTE) + ' | tail -n1').strip())
            if free < 100 << 30:
                raise OSError('Project quota lacks bounded output and 20 percent margin')
            logdir = ROOT / 'original_failed_logs'; logdir.mkdir(exist_ok=True)
            for ext in ('out', 'err'):
                remote = old + '/logs/JSCC_440_validation.1713939.' + ext
                sha = command(c, 'sha256sum ' + q(remote)).split()[0]
                dw.download(c, remote, logdir / ('1713939.' + ext), sha)
            if 'No usable temporary directory' not in (logdir / '1713939.err').read_text():
                raise RuntimeError('Original failure is not the reviewed temporary-directory bug')
            evidence = {}
            for kind, files in dict(derenzo=['gpu_freeze.json', 'JSCC_440/selection_acceptance.json',
                                           'JSCC_440/validation_job.json', 'JSCC_440/controller_failure.json'],
                                    nema=['production_freeze.json', 'validation_acceptance.json',
                                          'formal_authority.json', 'formal_job.json']).items():
                evidence[kind] = {n: digest(ORIGINAL[kind][0] / n) for n in files}
            write(ROOT / 'baseline.json', dict(accounting=baseline, old_pending_jobs=snapshots,
                  untouched_evidence_sha256=evidence, original_science_absent=absent,
                  partition=policy, qos=qos, actual_project_free_bytes=free,
                  original_failed_logs_sha256=hashes(logdir), old_jobs_remain_untouched=True))
    return frozen


def submit(kind, stage, script, output, allocation_path, nodes, gpus, minutes, dependency=None,
           held=False, extra=None):
    folder = REPORTS[kind] / ('JSCC_440' if kind == 'derenzo' else '')
    path = folder / (stage + '_job.json')
    if path.exists():
        return read(path)
    intent = folder / (stage + '_submission_intent.json')
    if intent.exists():
        raise RuntimeError('Unresolved submission intent; inspect scheduler, never repeat sbatch')
    local = DATAS[kind] / 'launch_scripts' / (stage + '.sh')
    immutable(local, script.encode()); remote = REMOTES[kind] + '/scripts/' + local.name
    log = REMOTES[kind] + '/logs/' + ('JSCC_440_' if kind == 'derenzo' else '') + stage
    options = ('-p gpu_5090 --qos=gpugpu -N' + str(nodes) + ' -n' + str(nodes)
               + ' --ntasks-per-node=1 --cpus-per-task=' + str(gpus * 8)
               + ' --gres=gpu:' + str(gpus) + ' --time=' + str(minutes)
               + ' --exclude=wqd10nba06g6 --chdir=' + q(REMOTES[kind])
               + ' --job-name=R20_' + kind + '_' + stage
               + ' --output=' + q(log + '.%j.out') + ' --error=' + q(log + '.%j.err'))
    if dependency:
        options += ' --dependency=afterok:' + str(dependency)
    if held:
        options += ' --hold'
    options += ' ' + q(remote)
    with connection('gpu') as c:
        upload(c, local, remote); command(c, 'bash -n ' + q(remote))
        test = command(c, 'sbatch --test-only ' + options + ' 2>&1', timeout=90)
        write(intent, dict(command='sbatch --parsable ' + options, resolved=False,
                           script_sha256=digest(local), scheduler_test_only=test, epoch=time.time()))
        number = command(c, 'sbatch --parsable ' + options, timeout=60).split(';')[0].strip()
        if not number.isdigit():
            raise RuntimeError('Ambiguous actual submission; preserve intent')
        record = dict(job=int(number), stage=stage, host='gpu', output=output.replace('{job}', number),
                      allocation=allocation_path.replace('{job}', number) if allocation_path else None,
                      nodes=nodes, gpus_per_node=gpus, walltime_minutes=minutes, dependency=dependency,
                      held=held, script_sha256=digest(local), submitted_epoch=time.time(), **(extra or {}))
        write(path, record); write(intent, dict(read(intent), resolved=True, registered_job=int(number)))
        snapshot = command(c, 'scontrol show job -o ' + number)
        immutable(folder / (stage + '_submitted.txt'), snapshot.encode())
    print('REPAIR_REGISTERED', kind, stage, number, flush=True)
    return record


def derenzo_job(stage, release, source=None, dependency=None):
    report = REPORTS['derenzo'] / 'JSCC_440'
    plan = read(report / 'validation_plan.json'); pd = read(report / 'validation_plan_deployment.json')
    with namespace('derenzo'):
        script, output, _ = dw.gpu_script('JSCC_440', stage, plan, pd['remote'], pd['sha256'], source)
    script = script.replace(REMOTES['derenzo'] + '/JSCC_440/input', ORIGINAL['derenzo'][2] + '/JSCC_440/input')
    if stage == 'formal':
        oldgate = read(REPORTS['derenzo'] / 'gpu_deployment.json')['release'] + '/derenzo_campaign_gate.py'
        start = 'gate_exports=$(' + q(GPU_PYTHON) + ' -B ' + q(oldgate)
        oldline = next(line for line in script.splitlines() if line.startswith(start))
        script = script.replace(oldline, 'gate_exports=$(' + q(GPU_PYTHON) + ' -B '
                                + q(release + '/' + GATE_NAME) + ' --gate '
                                + q(REMOTES['derenzo'] + '/formal_gate.json') + ')')
    script = repair_script(script, 'derenzo', release)
    verify = stage.endswith('_verify')
    return submit('derenzo', stage, script, output,
                  REMOTES['derenzo'] + '/allocations/JSCC_440_' + stage + '_{job}.txt',
                  1 if verify else plan['nodes'], 1 if verify else plan['gpus_per_node'],
                  180 if verify else (MAX_MINUTES if stage == 'formal' else 240),
                  dependency, stage == 'formal', extra=dict(dataset='JSCC_440', plan=plan,
                  plan_remote=pd['remote'], plan_sha256=pd['sha256'], plan_stage='validation'))


def gate_record(kind, formal, authority_file, authority_sha, science, science_sha, proof_remote, proof_sha, limits):
    original_report = ORIGINAL[kind][0]
    acceptance = (original_report / 'validation_acceptance.json' if kind == 'nema'
                  else REPORTS[kind] / 'JSCC_440/validation_acceptance.json')
    dest = DATAS[kind] / 'strict_validation_acceptance.json'
    immutable(dest, acceptance.read_bytes())
    a = read(original_report / 'formal_authority.json') if kind == 'nema' else read(DATAS[kind] / 'JSCC_440/authority.json')
    g = dict(passed=True, strict_fetch_passed=True, kind=kind, formal_job=formal['job'],
             validation_job=a['validation_job'], policy_max_minutes=MAX_MINUTES, limits=limits,
             authority_file=authority_file, authority_sha256=authority_sha, science_file=science,
             science_sha256=science_sha, validation_proof_file=proof_remote, validation_proof_sha256=proof_sha,
             strict_acceptance_file=REMOTES[kind] + '/strict_validation_acceptance.json',
             strict_acceptance_sha256=digest(dest))
    gate = DATAS[kind] / 'formal_gate.json'; immutable(gate, (json.dumps(g, indent=2) + '\n').encode())
    with connection('gpu') as c:
        upload(c, dest, g['strict_acceptance_file']); upload(c, gate, REMOTES[kind] + '/formal_gate.json')
    return g


def release_formal(kind, record, g, release):
    folder = REPORTS[kind] / ('JSCC_440' if kind == 'derenzo' else '')
    marker = folder / 'formal_release.json'
    if marker.exists():
        return
    with connection('gpu') as c:
        before = command(c, 'scontrol show job -o ' + str(record['job']))
        if 'JobState=PENDING' not in before or 'JobHeldUser' not in before:
            raise RuntimeError('Ambiguous release state; inspect release intent without resubmitting')
        command(c, 'scontrol update JobId=' + str(record['job']) + ' TimeLimit=' + str(g['limits']['minutes']))
        current = command(c, 'scontrol show job -o ' + str(record['job']))
        allocation(current, record['job'], record['nodes'], g['limits']['minutes'])
        code = ('import sys;from ' + GATE_NAME[:-3] + ' import check;'
                'check(sys.argv[1],int(sys.argv[2]));print("STRICT_REPAIRED_FORMAL_GATE_PASS")')
        proof = command(c, 'cd ' + q(release) + ' && ' + q(GPU_PYTHON) + ' -B -c ' + q(code)
                        + ' ' + q(REMOTES[kind] + '/formal_gate.json') + ' ' + str(record['job']), timeout=180)
        write(folder / 'formal_release_intent.json', dict(job=record['job'], gate=g, pre_release=current, proof=proof))
        command(c, 'scontrol release ' + str(record['job']))
        after = command(c, 'scontrol show job -o ' + str(record['job']))
    write(marker, dict(job=record['job'], released_epoch=time.time(), gate=g, scontrol=after))
    print('REPAIR_FORMAL_RELEASED', kind, record['job'], g['limits'], flush=True)


def nema_jobs(release):
    oldreport, olddata, oldbase = ORIGINAL['nema']
    auth = read(oldreport / 'formal_authority.json')
    accepted = read(oldreport / 'validation_acceptance.json')
    proof = read(Path(accepted['local_result']) / 'verification.json')
    if digest(olddata / 'formal_authority.json') != auth['sha256']:
        raise ValueError('Original NEMA authority changed')
    bound = measured('nema', proof)
    with namespace('nema'):
        # launcher reads only the original immutable authority metadata.
        immutable(REPORTS['nema'] / 'formal_authority.json', (oldreport / 'formal_authority.json').read_bytes())
        script = nw.launcher('formal', read(oldreport / 'production_deployment.json')['release'],
                             bound['phase'], bound['prepare'], bound['total'])
    script = script.replace(REMOTES['nema'] + '/input', oldbase + '/input')
    script = repair_script(script, 'nema', release)
    gatecmd = q(GPU_PYTHON) + ' -B ' + q(release + '/' + GATE_NAME) + ' --gate ' + q(REMOTES['nema'] + '/formal_gate.json')
    script = script.replace('source /etc/profile.d/modules.sh', 'gate_exports=$(' + gatecmd + ')\neval "$gate_exports"\nsource /etc/profile.d/modules.sh', 1)
    formal = submit('nema', 'formal', script, REMOTES['nema'] + '/formal_{job}',
                    REMOTES['nema'] + '/allocations/formal_{job}.txt', 8, 4, bound['minutes'], held=True)
    vjob = read(oldreport / 'validation_verification_job.json')
    g = gate_record('nema', formal, auth['remote'], auth['sha256'],
                    read(oldreport / 'production_deployment.json')['release'] + '/contract.json', auth['contract_sha256'],
                    vjob['output'] + '/verification.json', auth['verification_sha256'], bound)
    release_formal('nema', formal, g, release)
    vr = read(oldreport / 'validation_verification_deployment.json')['release']
    output = REMOTES['nema'] + '/acceptance_formal_' + str(formal['job'])
    args = ['--release', vr, '--input', oldbase + '/input', '--factors', nw.FACTORS,
            '--result', formal['output'], '--allocation', formal['allocation'], '--output', output, '--mode', 'formal']
    script = ('#!/usr/bin/env bash\nset -euo pipefail\nsource /etc/profile.d/modules.sh\nmodule load miniforge3/25.11.0-1\n'
              'export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 JSCC_PROJECT_ROOT=' + q(vr) + '\ncd ' + q(vr)
              + '\ntimeout --signal=TERM --kill-after=60s 6900s ' + q(GPU_PYTHON) + ' -B -u '
              + q(vr + '/jscc_5e10_verify_stage.py') + ' ' + ' '.join(map(q, args)) + '\n')
    submit('nema', 'formal_verification', repair_script(script, 'nema', release), output, None,
           1, 1, 120, formal['job'], extra=dict(source_job=formal['job'], release=vr, read_only_verification=True))


def derenzo_advance(release):
    validation = derenzo_job('validation', release)
    verifier = derenzo_job('validation_verify', release, validation, validation['job'])
    formal = derenzo_job('formal', release, dependency=verifier['job'])
    if not (REPORTS['derenzo'] / 'JSCC_440/formal_release.json').exists():
        with namespace('derenzo'):
            if not dw.fetch_gpu('JSCC_440', 'validation'):
                return False
        accepted_path = REPORTS['derenzo'] / 'JSCC_440/validation_acceptance.json'
        accepted = read(accepted_path); proof = read(Path(accepted['local_result']) / 'verification.json')
        bound = measured('derenzo', proof)
        authority = dict(passed=True, strict_fetch_passed=True, study=dw.STUDY, dataset='JSCC_440',
                         formal_job=formal['job'], plan_sha256=formal['plan_sha256'],
                         collection_sha256=proof['collection_sha256'], release_sha256=proof['release_sha256'],
                         validation_result=validation['output'], validation_files_sha256=proof['result_files_sha256'],
                         validation_job=validation['job'], strict_acceptance_sha256=digest(accepted_path), limits=bound)
        path = DATAS['derenzo'] / 'JSCC_440/authority.json'
        immutable(path, (json.dumps(authority, indent=2) + '\n').encode())
        remote = REMOTES['derenzo'] + '/JSCC_440/authority.json'
        with connection('gpu') as c:
            upload(c, path, remote)
        g = gate_record('derenzo', formal, remote, digest(path), formal['plan_remote'], formal['plan_sha256'],
                        verifier['output'] + '/verification.json', accepted['verification_sha256'], bound)
        release_formal('derenzo', formal, g, release)
    derenzo_job('formal_verify', release, formal, formal['job'])
    with namespace('derenzo'):
        return dw.fetch_gpu('JSCC_440', 'formal')


def run(watch):
    no_other_controller(); DATA.mkdir(parents=True, exist_ok=True)
    dw.source_complete = read_only_poll(dw.source_complete)
    nw.job_state = read_only_poll(nw.job_state)
    lock = DATA / 'exclusive.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    started = time.time(); code = 1
    for path in REGS:
        write(path, dict(pid=os.getpid(), status='running', started_epoch=started,
                         command=__file__, independent_manual_repair=True))
    try:
        frozen = prepare(); release = frozen['release']; nema_jobs(release)
        recovery = ORIGINAL['derenzo'][0] / 'recovery_20261019'
        deadlines = dict(nema=read(recovery / 'nema_production_5090_8x4_queue_registration.json')['started_epoch'] + 14 * 86400,
                         derenzo=read(recovery / 'derenzo_controller_registration.json')['started_epoch'] + 21 * 86400)
        states = dict(nema='waiting', derenzo='waiting')
        while True:
            for kind in ORIGINAL:
                if states[kind] != 'waiting':
                    continue
                try:
                    if time.time() >= deadlines[kind]:
                        raise TimeoutError('Original bounded controller deadline reached: ' + kind)
                    if kind == 'derenzo':
                        done = derenzo_advance(release)
                    else:
                        with namespace('nema'):
                            done = nw.fetch_stage('formal')
                    if done:
                        states[kind] = 'numerically_accepted'
                except ReadOnlyPollUnavailable as error:
                    write(REPORTS[kind] / 'read_only_poll_wait.json', dict(epoch=time.time(),
                          error=str(error), jobs_unchanged=True))
                except Exception as error:
                    states[kind] = 'needs_attention'
                    write(REPORTS[kind] / 'controller_failure.json', dict(error_type=type(error).__name__,
                          error=str(error), epoch=time.time(), all_jobs_and_outputs_preserved=True,
                          no_automatic_scientific_resubmission=True))
                    print('REPAIR_CHAIN_NEEDS_ATTENTION', kind, str(error), flush=True)
            write(ROOT / 'latest_progress.json', dict(checked_epoch=time.time(), numerical_acceptance=states,
                  scientific_visual_qa_completed=False, recurring_automations_remain_paused=True))
            print('REPAIR_PROGRESS', states, flush=True)
            if all(x != 'waiting' for x in states.values()) or not watch:
                break
            time.sleep(300)
        code = 1 if 'needs_attention' in states.values() else 0
    except Exception as error:
        write(ROOT / 'controller_failure.json', dict(error_type=type(error).__name__, error=str(error),
              epoch=time.time(), all_jobs_and_outputs_preserved=True, no_automatic_scientific_resubmission=True))
        raise
    finally:
        for path in REGS:
            write(path, dict(read(path), status='complete', exit_code=code, finished_epoch=time.time()))
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    run(parser.parse_args().watch)
