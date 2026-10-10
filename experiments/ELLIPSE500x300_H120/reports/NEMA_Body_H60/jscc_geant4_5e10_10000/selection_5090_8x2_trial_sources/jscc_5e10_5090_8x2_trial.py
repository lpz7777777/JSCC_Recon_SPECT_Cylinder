"""Isolated, user-authorized full-input 8x2 selection trial; no production switch."""
import argparse
import ast
import ctypes
import hashlib
import json
import os
import re
import shutil
import time
from shlex import quote as q

from jscc_5e10_common import *

STAGE = 'selection_5090_8x2_trial'
JOB_NAME = 'JSCC5e10Trial5090x2'
PRESERVED = ('selection_job.json', 'selection_4090_trial_job.json')


def launcher(*args):
    # Extract the original generator without importing local SSH/controller dependencies on Linux.
    tree = ast.parse((HERE / 'jscc_5e10_reconstruction_workflow.py').read_text())
    function = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'launcher'][0]
    namespace = dict(globals(), FACTORS=GPU_PROJECT + '/generated/FactorsCalibrated')
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<original-launch-generator>', 'exec'), namespace)
    return namespace['launcher'](*args)


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


def replace_once(text, replacements):
    original = text
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError('Unknown frozen layout: ' + old)
        text = text.replace(old, new, 1)
    restored = text
    for old, new in reversed(list(replacements.items())):
        restored = restored.replace(new, old, 1)
    if restored != original:
        raise ValueError('Unbounded topology change')
    return text


def topology_sources(root):
    """Only execution identity/resource checks and the final all-row sum change."""
    runtime = (root / 'jscc_5e10_runtime.py').read_text()
    runtime = replace_once(runtime, {
        'world!=32 or not 0<=local<4 or torch.cuda.device_count()!=4':
            'world!=16 or not 0<=local<2 or torch.cuda.device_count()!=2',
        'Frozen 8-node x4-GPU topology required': 'Frozen selection-only 8-node x2-GPU topology required',
        'result=[None]*32;dist.all_gather_object(result,record)':
            'result=[None]*16;dist.all_gather_object(result,record)',
    })
    contract = (root / 'jscc_5e10_contract.py').read_text()
    prefix, function = contract.split('def verify_topology(resources,allocation):', 1)
    function = replace_once(function, {
        'len(resources)!=32 or sorted(r[\'rank\'] for r in resources)!=list(range(32))':
            'len(resources)!=16 or sorted(r[\'rank\'] for r in resources)!=list(range(16))',
        'All 32 rank identities required': 'All 16 trial rank identities required',
        'gres/gpu=32': 'gres/gpu=16',
        'Actual allocation must contain 32 GPUs': 'Actual trial allocation must contain 16 GPUs',
        'list(range(4)) or len({r[\'gpu_uuid\'] for r in group})!=4':
            'list(range(2)) or len({r[\'gpu_uuid\'] for r in group})!=2',
        'Four different actual GPUs/local ranks per node required':
            'Two different actual GPUs/local ranks per node required',
    })
    contract = prefix + 'def verify_topology(resources,allocation):' + function
    selection = replace_once((root / 'jscc_5e10_selection.py').read_text(), {
        'for r in range(32))!=coll[\'raw_list_rows\']':
            'for r in range(world))!=coll[\'raw_list_rows\']',
    })
    return {'jscc_5e10_runtime.py': runtime, 'jscc_5e10_contract.py': contract,
            'jscc_5e10_selection.py': selection}


def trial_launcher(release):
    return replace_once(launcher('selection', release, 3600, 10800, 14220), {
        'torch.cuda.device_count()==4;print("FOUR_VISIBLE_GPUS"':
            'torch.cuda.device_count()==2;print("TWO_VISIBLE_GPUS"',
        '--nnodes=8 --nproc_per_node=4': '--nnodes=8 --nproc_per_node=2',
        q(GPU_BASE + '/allocations/selection_'): q(GPU_BASE + '/allocations/' + STAGE + '_'),
        q(GPU_BASE + '/selection_'): q(GPU_BASE + '/' + STAGE + '_'),
        'echo JSCC5E10_COMPUTATION_COMPLETED selection\n':
            'echo JSCC5E10_COMPUTATION_COMPLETED ' + STAGE + '\n',
    })


def preserved_fields(text):
    fields = dict(re.findall(r'(\S+?)=(\S*)', text))
    return {name: fields.get(name) for name in ('JobId', 'JobName', 'Partition', 'NumNodes',
        'NumCPUs', 'NumTasks', 'CPUs/Task', 'ReqTRES', 'Command', 'TimeLimit', 'ExcNodeList')}


def submit():
    from jscc_5e10_workflow import connection, command
    record_path = REPORT / (STAGE + '_job.json')
    if record_path.exists():
        print('ALREADY_REGISTERED', read(record_path)['job'], flush=True)
        return
    for name in ('advance_registration.json', 'selection_4090_trial_registration.json', STAGE + '_registration.json'):
        if live_registration(DATA / name):
            raise RuntimeError('A registered local controller is alive: ' + name)
    registration = DATA / (STAGE + '_registration.json')
    write(registration, dict(pid=os.getpid(), status='running', started_epoch=time.time()))
    exit_code = 1
    try:
        intent = REPORT / (STAGE + '_submission_intent.json')
        if intent.exists():
            raise RuntimeError('Unresolved submission intent; inspect scheduler before any retry')
        preserved = {name: {'record': read(REPORT / name), 'sha256': digest(REPORT / name)} for name in PRESERVED}
        original = read(REPORT / 'kernel_freeze.json')
        source = DATA / 'kernel_payload'
        verify_files(source, original['sha256'])
        payload = DATA / (STAGE + '_kernel_payload')
        shutil.copytree(source, payload)
        changed = topology_sources(source)
        for name, text in changed.items():
            (payload / name).write_bytes(text.encode())
        shutil.copy2(HERE / 'test_jscc_5e10_5090_8x2_trial.py', payload / 'test_jscc_5e10_5090_8x2_trial.py')
        shutil.copy2(__file__, payload / Path(__file__).name)
        cfg = read(payload / 'kernel_config.json')
        files = hashes(payload)
        files.pop('kernel_config.json')
        cfg.update(files=files, nodes=8, gpus_per_node=2, world_size=16, selection_trial_only=True,
            original_kernel_freeze_sha256=digest(REPORT / 'kernel_freeze.json'))
        write(payload / 'kernel_config.json', cfg)
        files = hashes(payload)
        unchanged = {name: sha for name, sha in original['sha256'].items() if name not in (*changed, 'kernel_config.json')}
        verify_files(payload, unchanged)
        key = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
        release = GPU_BASE + '/trial_selection_releases/' + key
        script = trial_launcher(release)
        script_path = REPORT / 'launch_scripts' / (STAGE + '.sh')
        script_path.parent.mkdir(exist_ok=True)
        script_path.write_bytes(script.encode())
        freeze_path = REPORT / (STAGE + '_freeze.json')
        write(freeze_path, dict(release_key=key, release=release, sha256=files,
            original_kernel_freeze_sha256=digest(REPORT / 'kernel_freeze.json'),
            source_changes={name: dict(before=digest(source / name), after=digest(payload / name)) for name in changed},
            unchanged_original_members=unchanged, preserved_jobs={name: item['sha256'] for name, item in preserved.items()},
            local_tests_sha256=digest(REPORT / (STAGE + '_tests.txt')),
            transport_acceptance_sha256=digest(REPORT / 'transport_acceptance.json'),
            script_sha256=digest(script_path), scope='Full-input selection only; production runtime/contract remain 8x4.',
            user_authorized_parallel_trial=True, nodes=8, gpus_per_node=2, world_size=16,
            partition='gpu_5090', cpus_per_node=16, expected_auto_memory_mib_per_node=252000,
            explicit_mem_parameter=False, frozen_runtime_torch_threads_per_rank=10,
            strict_all_row_and_resource_acceptance_required=True,
            selection_memory_not_complete_compton_response_memory=True))
        with connection('gpu') as c:
            before = {}
            for name, item in preserved.items():
                value = command(c, 'scontrol show job -o ' + str(item['record']['job']), timeout=45)
                before[name] = value
                (REPORT / (STAGE + '_' + name.removesuffix('.json') + '_before.txt')).write_bytes(value.encode())
            queue = command(c, "squeue -h -u scxi717 -o '%i|%j|%T'", timeout=45)
            if len(queue.splitlines()) >= 50 or any('|' + JOB_NAME + '|' in line for line in queue.splitlines()):
                raise RuntimeError('Account full or unregistered trial; preserve state and inspect')
            policy = command(c, 'scontrol show partition gpu_5090', timeout=45)
            if 'DefCpuPerGPU=8' not in policy or 'DefMemPerCPU=15750' not in policy:
                raise ValueError('5090 policy changed; recompute resource budget')
            policy_path = REPORT / (STAGE + '_partition_policy.txt')
            policy_path.write_bytes(policy.encode())
            command(c, 'test ! -e ' + q(release) + ' && mkdir -p ' + q(release))
            with c.open_sftp() as s:
                for name in files:
                    directory = release + '/' + str(Path(name).parent).replace('\\', '/')
                    command(c, 'mkdir -p ' + q(directory))
                    s.put(str(payload / name), release + '/' + name)
                s.put(str(script_path), release + '/selection.sh')
            check = ('from pathlib import Path;from jscc_5e10_common import *;'
                'p=Path(".");verify_files(p,read(p/"kernel_config.json")["files"]);'
                'import jscc_5e10_selection;print("TRIAL_SOURCE_SHA_IMPORT_PASS")')
            proof = command(c, 'cd ' + q(release) + ' && ' + q(GPU_PYTHON) + ' -c ' + q(check), timeout=120)
            tests = command(c, 'cd ' + q(release) + ' && JSCC_PROJECT_ROOT=' + q(release) +
                ' JSCC_TRIAL_ORIGINAL_KERNEL=' + q(read(REPORT / 'kernel_deployment.json')['release']) + ' ' + q(GPU_PYTHON) +
                ' -m unittest test_jscc_5e10_5090_8x2_trial -v 2>&1', timeout=120)
            if '\nOK' not in tests or 'skipped' in tests:
                raise ValueError('Actual Linux trial checks failed: ' + tests)
            test_path = REPORT / (STAGE + '_linux_tests.txt')
            test_path.write_bytes((proof + '\n' + tests).encode())
            if command(c, 'sha256sum ' + q(release + '/selection.sh')).split()[0] != digest(script_path):
                raise ValueError('Trial script transfer differs')
            command(c, 'bash -n ' + q(release + '/selection.sh'))
            options = ('-p gpu_5090 --qos=gpugpu -N8 -n8 --ntasks-per-node=1 --cpus-per-task=16 '
                '--gres=gpu:2 --time=240 --exclude=wqd10nba06g6 --chdir=/tmp --job-name=' + JOB_NAME +
                ' --output=' + q(GPU_BASE + '/logs/' + STAGE + '.%j.out') +
                ' --error=' + q(GPU_BASE + '/logs/' + STAGE + '.%j.err') + ' ' + q(release + '/selection.sh'))
            test = command(c, 'sbatch --test-only ' + options + ' 2>&1', timeout=60)
            write(REPORT / (STAGE + '_launch_acceptance.json'), dict(passed=True,
                scope='Source SHA, actual Linux CPU topology tests, bash and scheduler test-only; not actual GPU execution.',
                freeze_sha256=digest(freeze_path), linux_tests_sha256=digest(test_path),
                partition_policy_sha256=digest(policy_path), scheduler_test_only=test, script_sha256=digest(script_path)))
            write(intent, dict(stage=STAGE, script_sha256=digest(script_path), started_epoch=time.time(),
                preserved_jobs={name: item['sha256'] for name, item in preserved.items()}))
            number = command(c, 'sbatch --parsable ' + options, timeout=60).split(';')[0].strip()
            if not number.isdigit():
                raise RuntimeError('Ambiguous submission; preserve intent and inspect scheduler')
            write(record_path, dict(job=int(number), stage=STAGE, host='gpu', release=release,
                output=GPU_BASE + '/' + STAGE + '_' + number,
                allocation=GPU_BASE + '/allocations/' + STAGE + '_' + number + '.txt',
                nodes=8, gpus_per_node=2, world_size=16, partition='gpu_5090', cpus_per_node=16,
                explicit_mem_parameter=False, walltime_minutes=240, total_limit_seconds=14220,
                trial_freeze_sha256=digest(freeze_path), script_sha256=digest(script_path),
                user_authorized_parallel_trial=True, production_registry_replaced=False,
                preserved_jobs={name: item['sha256'] for name, item in preserved.items()}, submitted_epoch=time.time()))
            write(intent, dict(read(intent), resolved=True, registered_job=int(number)))
            snapshot = command(c, 'scontrol show job -o ' + number, timeout=45)
            (REPORT / (STAGE + '_submit_scontrol.txt')).write_bytes(snapshot.encode())
            after = {}
            for name, item in preserved.items():
                value = command(c, 'scontrol show job -o ' + str(item['record']['job']), timeout=45)
                after[name] = value
                (REPORT / (STAGE + '_' + name.removesuffix('.json') + '_after.txt')).write_bytes(value.encode())
                if digest(REPORT / name) != item['sha256'] or preserved_fields(value) != preserved_fields(before[name]):
                    raise ValueError('An original job registration/request changed')
            write(REPORT / (STAGE + '_submission_acceptance.json'), dict(passed=True, job=int(number),
                scope='Actual submission and preservation only; GPU execution/selection acceptance pending.',
                job_record_sha256=digest(record_path), freeze_sha256=digest(freeze_path),
                actual_scontrol_sha256=digest(REPORT / (STAGE + '_submit_scontrol.txt')),
                preserved_records={name: item['sha256'] for name, item in preserved.items()},
                preserved_scheduler_fields={name: preserved_fields(after[name]) for name in preserved},
                full_input_and_original_scientific_members_preserved=True))
        print('REGISTERED_5090_8X2_SELECTION_TRIAL', number, flush=True)
        exit_code = 0
    finally:
        write(registration, dict(read(registration), status='complete', exit_code=exit_code, finished_epoch=time.time()))


def status():
    from jscc_5e10_workflow import connection, command, job_state
    path = REPORT / (STAGE + '_job.json')
    if not path.exists():
        print('NO_REGISTERED_5090_8X2_TRIAL', flush=True)
        return
    job = read(path)
    with connection('gpu') as c:
        state, accounting = job_state(c, job)
        print('5090_8X2_TRIAL', job['job'], state, accounting, flush=True)
        for suffix in ('out', 'err'):
            log = GPU_BASE + '/logs/' + STAGE + '.' + str(job['job']) + '.' + suffix
            print(command(c, 'if test -f ' + q(log) + '; then tail -n 12 ' + q(log) + '; fi', timeout=45), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit', 'status'))
    args = parser.parse_args()
    (submit if args.action == 'submit' else status)()
