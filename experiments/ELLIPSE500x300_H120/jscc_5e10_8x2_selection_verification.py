"""Freeze/submit/fetch separate read-only acceptance of the completed 16-rank trial."""
import argparse
import hashlib
import os
import re
import shutil
import tarfile
import time
from shlex import quote as q
from jscc_5e10_common import *
from jscc_5e10_5090_8x2_trial import live_registration, replace_once

SOURCE = 'selection_5090_8x2_monitor_repair'
STAGE = SOURCE + '_verification'
GUARDS = ('advance', 'selection_4090_trial', 'selection_5090_8x2_trial', SOURCE, STAGE)


def peak_rss(accounting):
    values = []
    for line in accounting.splitlines():
        fields = line.split('|')
        if len(fields) < 4 or not fields[3]:
            continue
        m = re.fullmatch(r'([0-9.]+)([KMGT]?)', fields[3])
        if not m:
            raise ValueError('Unrecognized actual Slurm MaxRSS')
        values.append(float(m[1]) * 1024 ** ('KMGT'.index(m[2]) + 1 if m[2] else 1))
    if not values or max(values) <= 0:
        raise ValueError('Actual exited selection MaxRSS missing')
    return int(max(values))


def strict_get(c, remote, local):
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    from jscc_5e10_workflow import command
    before = command(c, 'sha256sum ' + q(remote)).split()[0]
    if local.exists():
        if digest(local) != before:
            raise ValueError('Preserve differing prior local evidence: ' + str(local))
    else:
        partial = local.with_name(local.name + '.partial')
        if partial.exists():
            raise ValueError('Preserve interrupted fetch for diagnosis: ' + str(partial))
        with c.open_sftp() as s:
            s.get(remote, str(partial))
        if digest(partial) != before:
            raise ValueError('Fetched original bytes SHA differs')
        os.replace(partial, local)
    after = command(c, 'sha256sum ' + q(remote)).split()[0]
    if before != after:
        raise ValueError('Remote evidence changed during fetch')
    return before


def source_completion(c, source, accounting):
    folder = REPORT / (SOURCE + '_completion_' + str(source['job']))
    folder.mkdir(exist_ok=True)
    from jscc_5e10_workflow import command
    files = {}
    for remote, name in ((source['output'] + '/selection_manifest.json', 'selection_manifest.json'),
                         (source['allocation'], 'allocation.txt')):
        files[name] = strict_get(c, remote, folder / name)
    for ext in ('out', 'err'):
        files['original.' + ext + '.txt'] = strict_get(c, GPU_BASE + '/logs/' + SOURCE + '.' + str(source['job']) + '.' + ext,
                                                       folder / ('original.' + ext + '.txt'))
    # Final accounting is an observation, not an immutable remote file.
    (folder / 'sacct.txt').write_bytes(accounting.encode())
    files['sacct.txt'] = digest(folder / 'sacct.txt')
    manifest = read(folder / 'selection_manifest.json')
    cfg_path = DATA / (SOURCE + '_kernel_payload') / 'kernel_config.json'
    if (not manifest['passed'] or manifest['study'] != STUDY or manifest['actual_primary_photons'] != TOTAL or
            manifest['source_or_truth_used_in_selection'] is not False or manifest['kernel_config_sha256'] != digest(cfg_path) or
            manifest['input_collection_sha256'] != read(cfg_path)['transport_collection_sha256']):
        raise ValueError('Completed trial scientific/input identity differs')
    allocation = (folder / 'allocation.txt').read_text()
    # This is the actual frozen 16-rank contract, not the production 32-rank verifier.
    namespace = {}
    exec(compile((DATA / (SOURCE + '_kernel_payload') / 'jscc_5e10_contract.py').read_text(), '<frozen-16-rank-contract>', 'exec'), namespace)
    memory = namespace['verify_topology'](manifest['resources'], folder / 'allocation.txt')
    rss = peak_rss(accounting)
    if rss > .8 * memory:
        raise ValueError('Actual exited source MaxRSS leaves less than 20% node margin')
    resources = manifest['resources']
    nodes = sorted({r['node'] for r in resources})
    metrics = dict(actual_allocated_bytes_node=memory, slurm_maxrss_bytes=rss, slurm_rss_fraction=rss / memory,
                   max_node_aggregate_rss_fraction=max(sum(r['host_peak_rss_bytes'] for r in resources if r['node'] == n) / memory for n in nodes),
                   max_gpu_used_fraction=max(r['measured_gpu_used_peak_bytes'] / r['total_device_bytes'] for r in resources),
                   max_gpu_reserved_fraction=max(r['peak_reserved_bytes'] / r['total_device_bytes'] for r in resources))
    write(folder / 'completion_acceptance.json', dict(passed=True, job=source['job'], source_job_sha256=digest(REPORT / (SOURCE + '_job.json')),
          source_freeze_sha256=digest(REPORT / (SOURCE + '_freeze.json')), files_sha256=files, source_accounting=accounting,
          resources=metrics, events_per_view=manifest['events_per_view'], accepted_events=manifest['accepted_events'],
          original_accepted=manifest['original_accepted'], removed=manifest['removed'],
          scope='Successful exit, original logs/manifest/allocation strict SHA and frozen 16-rank resource checks only; complete-row independent acceptance pending.',
          all_row_acceptance=False, full_compton_response_memory_certificate=False))
    return folder


def freeze():
    path = REPORT / (STAGE + '_freeze.json')
    if path.exists():
        return read(path)
    source = DATA / (SOURCE + '_kernel_payload')
    scientific = read(REPORT / (SOURCE + '_freeze.json'))
    verify_files(source, scientific['sha256'])
    payload = DATA / (STAGE + '_payload')
    shutil.copytree(source, payload)
    for name in ('jscc_5e10_selection_parts.py', 'test_jscc_5e10_selection_parts.py'):
        shutil.copy2(HERE / name, payload / name)
    shutil.copy2(REPORT / (SOURCE + '_job.json'), payload / 'verified_source_job.json')
    shutil.copy2(REPORT / (SOURCE + '_freeze.json'), payload / 'verified_source_freeze.json')
    text = (source / 'jscc_5e10_verify_stage.py').read_text()
    additions = """        from jscc_5e10_selection_parts import verify_parts
        source=read(a.release/'verified_source_job.json')
        source_freeze=read(a.release/'verified_source_freeze.json')
        if (str(a.result)!=source['output'] or str(a.allocation)!=source['allocation'] or
                source['world_size']!=16 or source['nodes']!=8 or source['gpus_per_node']!=2 or
                manifest['kernel_config_sha256']!=source_freeze['sha256']['kernel_config.json'] or
                manifest['source_or_truth_used_in_selection'] is not False):
            raise ValueError('Independent selection source identity differs')
        verify_files(Path(source['release']),source_freeze['sha256'])
        parts=verify_parts(a.result,manifest,coll)
"""
    text = replace_once(text, {
        '        verify_selection(a.input,a.result,manifest)\n': additions + '        verify_selection(a.input,a.result,manifest)\n',
        "        members={'selection_manifest.json':digest(a.result/'selection_manifest.json')}\n":
            "        proof.update(executed_partition_closure=parts,source_job=source['job'],world_size=16)\n        members={'selection_manifest.json':digest(a.result/'selection_manifest.json'),**parts['files']}\n",
    })
    (payload / 'jscc_5e10_verify_stage.py').write_bytes(text.encode())
    unchanged = {n: sha for n, sha in scientific['sha256'].items() if n != 'jscc_5e10_verify_stage.py'}
    verify_files(payload, unchanged)
    files = hashes(payload)
    write(payload / 'verification_release.json', dict(sha256=files, read_only=True, scientific_release_sha256=scientific['sha256'],
          scientific_freeze_sha256=digest(REPORT / (SOURCE + '_freeze.json')), unchanged_members=unchanged,
          selection_only=True, world_size=16, image_reconstruction_authorized_by_this_release=False))
    files = hashes(payload)
    key = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    result = dict(release_key=key, release=GPU_BASE + '/verification_releases/' + key, sha256=files,
                  source_freeze_sha256=digest(REPORT / (SOURCE + '_freeze.json')), unchanged_members=unchanged,
                  isolated_read_only=True, world_size=16)
    write(path, result)
    return result


def submit():
    from jscc_5e10_workflow import connection, command, job_state, put_archive
    path = REPORT / (STAGE + '_job.json')
    if path.exists():
        print('EXISTING_VERIFICATION', read(path)['job'], flush=True)
        return
    intent = REPORT / (STAGE + '_submission_intent.json')
    if intent.exists():
        raise ValueError('Unresolved submission intent; inspect scheduler without duplicate submission')
    source = read(REPORT / (SOURCE + '_job.json'))
    preserved = {name: digest(REPORT / name) for name in ('selection_job.json', 'selection_4090_trial_job.json')}
    with connection('gpu') as c:
        state, acc = job_state(c, source)
        if state != 'complete':
            print('WAIT_SOURCE_COMPLETE', flush=True)
            return
        completion = source_completion(c, source, acc)
        f = freeze()
        deployment = REPORT / (STAGE + '_deployment.json')
        if not deployment.exists():
            put_archive(c, DATA / (STAGE + '_payload'), f['release'])
            check = 'from pathlib import Path;from jscc_5e10_common import *;verify_files(Path("."),read("verification_release.json")["sha256"]);import jscc_5e10_verify_stage;print("READ_ONLY_VERIFICATION_SHA_IMPORT_PASS")'
            checks = command(c, 'cd ' + q(f['release']) + ' && PYTHONDONTWRITEBYTECODE=1 JSCC_PROJECT_ROOT=' + q(f['release']) + ' ' + q(GPU_PYTHON) + ' -c ' + q(check), timeout=120)
            tests = command(c, 'cd ' + q(f['release']) + ' && PYTHONDONTWRITEBYTECODE=1 JSCC_PROJECT_ROOT=' + q(f['release']) + ' ' + q(GPU_PYTHON) + ' -m unittest test_jscc_5e10_selection_parts -v 2>&1', timeout=120)
            if '\nOK' not in tests or 'skipped' in tests:
                raise ValueError('Actual Linux independent evidence checks failed: ' + tests)
            testfile = REPORT / (STAGE + '_linux_tests.txt')
            testfile.write_bytes((checks + '\n' + tests).encode())
            write(deployment, dict(passed=True, release=f['release'], sha256=f['sha256'], linux_tests_sha256=digest(testfile),
                                   source_completion_sha256=digest(completion / 'completion_acceptance.json')))
        queue = command(c, "squeue -h -u scxi717 -o '%i|%j|%T'")
        if len(queue.splitlines()) >= 50:
            print('WAIT_ACCOUNT_JOB_SLOT', flush=True)
            return
        output = GPU_BASE + '/acceptance_' + SOURCE + '_' + str(source['job'])
        args = ['--release', f['release'], '--input', GPU_BASE + '/input', '--factors', GPU_PROJECT + '/generated/FactorsCalibrated',
                '--result', source['output'], '--allocation', source['allocation'], '--output', output, '--mode', 'selection']
        script = '#!/usr/bin/env bash\nset -euo pipefail\nsource /etc/profile.d/modules.sh\nmodule load miniforge3/25.11.0-1\n'
        script += 'export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 JSCC_PROJECT_ROOT=' + q(f['release']) + '\ncd ' + q(f['release']) + '\n'
        script += 'timeout --signal=TERM --kill-after=60s 6900s ' + q(GPU_PYTHON) + ' -u jscc_5e10_verify_stage.py ' + ' '.join(q(x) for x in args) + '\n'
        local = REPORT / 'launch_scripts' / (STAGE + '.sh')
        local.write_bytes(script.encode())
        remote = f['release'] + '/verify.sh'
        with c.open_sftp() as s:
            s.put(str(local), remote)
        if command(c, 'sha256sum ' + q(remote)).split()[0] != digest(local):
            raise ValueError('Read-only launch transfer differs')
        command(c, 'bash -n ' + q(remote))
        options = ('-p gpu_5090 --qos=gpugpu -N1 -n1 --cpus-per-task=8 --gres=gpu:1 --time=120 '
                   '--exclude=wqd10nba06g6 --chdir=/tmp --job-name=JSCC5e10_x2_readverify --output=' +
                   q(GPU_BASE + '/logs/' + STAGE + '.%j.out') + ' --error=' + q(GPU_BASE + '/logs/' + STAGE + '.%j.err') + ' ' + q(remote))
        test = command(c, 'sbatch --test-only ' + options + ' 2>&1', timeout=60)
        write(intent, dict(stage=STAGE, source_job=source['job'], script_sha256=digest(local), preserved_records=preserved,
                           scheduler_test_only=test, started_epoch=time.time()))
        number = command(c, 'sbatch --parsable ' + options, timeout=60).split(';')[0].strip()
        if not number.isdigit():
            raise ValueError('Ambiguous read-only verification submission')
        write(path, dict(job=int(number), stage=STAGE, host='gpu', release=f['release'], output=output,
                          source_job=source['job'], source_accounting=acc, script_sha256=digest(local),
                          verification_freeze_sha256=digest(REPORT / (STAGE + '_freeze.json')), read_only_verification=True,
                          nodes=1, reserved_gpus=1, cpus_per_node=8, explicit_mem_parameter=False,
                          source_world_size=16, mode='selection', preserved_records=preserved))
        write(intent, dict(read(intent), resolved=True, registered_job=int(number)))
        if {n: digest(REPORT / n) for n in preserved} != preserved:
            raise ValueError('Original production/trial registration changed')
    print('REGISTERED_INDEPENDENT_16RANK_READ_ONLY_VERIFICATION', number, flush=True)


def status():
    from jscc_5e10_workflow import connection, job_state
    path = REPORT / (STAGE + '_job.json')
    if not path.exists():
        print('NO_REGISTERED_VERIFICATION', flush=True)
        return
    with connection('gpu') as c:
        state, acc = job_state(c, read(path))
        print(state, acc, flush=True)


def fetch():
    from jscc_5e10_workflow import connection, job_state
    acceptance = REPORT / (SOURCE + '_acceptance.json')
    if acceptance.exists():
        print('EXISTING_STRICT_SELECTION_ACCEPTANCE', flush=True)
        return
    source = read(REPORT / (SOURCE + '_job.json'))
    record = read(REPORT / (STAGE + '_job.json'))
    with connection('gpu') as c:
        state, acc = job_state(c, record)
        if state != 'complete':
            print('WAIT_VERIFICATION_COMPLETE', flush=True)
            return
        state, source_acc = job_state(c, source)
        if state != 'complete':
            raise ValueError('Successful scientific full exit required')
        folder = REPORT / (STAGE + '_evidence')
        folder.mkdir(exist_ok=True)
        for name in ('package.json', 'verification.json'):
            strict_get(c, record['output'] + '/' + name, folder / name)
        package = read(folder / 'package.json')
        archive = DATA / (STAGE + '_accepted_result.tar.gz')
        if strict_get(c, record['output'] + '/accepted_result.tar.gz', archive) != package['archive_sha256'] or archive.stat().st_size != package['archive_bytes']:
            raise ValueError('Actual accepted result archive identity differs')
        destination = DATA / (SOURCE + '_accepted_' + str(source['job']))
        if not destination.exists():
            destination.mkdir()
            with tarfile.open(archive) as t:
                members = t.getmembers()
                if len(members) != len(package['files']) or {m.name for m in members} != set(package['files']):
                    raise ValueError('Result package member set differs')
                if any(not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members):
                    raise ValueError('Unsafe scientific result archive')
                t.extractall(destination)
        verify_files(destination, package['files'])
        proof = read(destination / 'verification.json')
        if (not proof['passed'] or proof['mode'] != 'selection' or proof['source_job'] != source['job'] or proof['world_size'] != 16 or
                proof['actual_primary_photons'] != TOTAL or not proof['all_selected_rows_exact'] or not proof['executed_partition_closure']['passed'] or
                proof['verification_release_sha256'] != read(REPORT / (STAGE + '_freeze.json'))['sha256']['verification_release.json']):
            raise ValueError('Independent 16-rank acceptance identity differs')
        strict_get(c, source['allocation'], folder / 'source_allocation.txt')
        if digest(folder / 'source_allocation.txt') != digest(destination / 'allocation.txt'):
            raise ValueError('Actual source allocation remote/archive differs')
        memory = host_allocated_bytes((destination / 'allocation.txt').read_text(), 8)
        rss = peak_rss(source_acc)
        if rss > .8 * memory:
            raise ValueError('Exited source Slurm resource certificate fails')
        for item in (source, record):
            for ext in ('out', 'err'):
                strict_get(c, GPU_BASE + '/logs/' + item['stage'] + '.' + str(item['job']) + '.' + ext,
                           folder / (item['stage'] + '.' + ext + '.txt'))
        write(acceptance, dict(passed=True, study=STUDY, mode='selection', job=source['job'], verification_job=record['job'],
              local_result=str(destination), remote_result=source['output'], strict_fetch_passed=True,
              package_sha256=digest(folder / 'package.json'), archive_sha256=package['archive_sha256'],
              verification_sha256=digest(destination / 'verification.json'), strict_files_sha256=package['files'],
              source_accounting=source_acc, verification_accounting=acc, evidence_sha256=hashes(folder),
              slurm_maxrss_bytes=rss, actual_allocated_bytes_node=memory, slurm_rss_fraction=rss / memory,
              world_size=16, production_registry_replaced=False, full_compton_response_memory_certificate=False))
    print('STRICT_INDEPENDENT_16RANK_SELECTION_ACCEPTED', source['job'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit', 'status', 'fetch'))
    action = parser.parse_args().action
    busy = [name for name in GUARDS if live_registration(DATA / (name + '_registration.json'))]
    if busy:
        raise RuntimeError('An existing controller PID is alive: ' + ', '.join(busy))
    if action == 'status':
        status()
    else:
        registration = DATA / (STAGE + '_registration.json')
        write(registration, dict(pid=os.getpid(), action=action, status='running', started_epoch=time.time()))
        code = 1
        try:
            (submit if action == 'submit' else fetch)()
            code = 0
        finally:
            write(registration, dict(read(registration), status='complete', exit_code=code, finished_epoch=time.time()))
