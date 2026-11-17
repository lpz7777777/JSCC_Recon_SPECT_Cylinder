"""Explicitly authorized additional 4090 formal run; preserve the 5090 job.

The execution fork changes only GPU admission and the matrix-source release
reference. All mathematical code, inputs and original study values are retained.
No simulation, matrix transfer or separate Slurm validation job is performed.
"""
from pathlib import Path
import argparse
import ast
import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import tarfile
import time
import traceback

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path[:0] = [str(PARENT), str(PARENT/'repair_20261116')]
import psutil
import handoff
from common import read, write, sha, now
from prepare import DATA as PARENT_DATA, REPORT as PARENT_REPORT
from workflow import connect, command, extraction_program, REMOTE_GPU as MATRIX_ROOT, GPU_PY
from repair_workflow import verify as verify_parent

DATA = PARENT_DATA/HERE.name
REPORT = PARENT_REPORT/HERE.name
REMOTE = MATRIX_ROOT+'/'+HERE.name
NAME = 'NEMA5e9_CartHE4090'
CPUS = 8
MEM_MIB = 80000
PARTITION = 'gpu_4090'
PARENT_KEY = '95022158ffedcdd7'


def verify_files(root, manifest):
    for name, digest in manifest['files'].items():
        if sha(root/name) != digest:
            raise ValueError('Frozen file differs: '+name)


def prepare():
    verify_parent()
    DATA.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    frozen = REPORT/'freeze.json'
    if frozen.exists():
        manifest = read(frozen)
        verify_files(DATA/'payload', manifest)
        return manifest
    parent = read(PARENT_REPORT/'freeze.json')
    assert parent['key'] == PARENT_KEY
    source = PARENT_DATA/'payload'
    verify_files(source, parent)
    assert read(PARENT_REPORT/'matrix_strict_transfer.json')['passed']
    target = DATA/'payload'
    target.mkdir(exist_ok=False)
    for name in parent['files']:
        dest = target/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, dest)
    original = (source/'run_reconstruction.py').read_text(encoding='utf8')
    replacements = [
        ("'5090' not in torch.cuda.get_device_name()", "'4090' not in torch.cuda.get_device_name()"),
        ("Expected one visible ordinary 5090 per rank", "Expected one visible ordinary 4090 per rank"),
        ("matrix(a.matrices, channel, release['key'])", "matrix(a.matrices, channel, cfg['matrix_source_release'])"),
        ("matrix(a.matrices, 'A440', release['key'])", "matrix(a.matrices, 'A440', cfg['matrix_source_release'])"),
        ("return np.memmap(p, dtype='<f4', mode='r', shape=(10496, 507000))", "return BoundedReadOnlyMatrix(np.memmap(p, dtype='<f4', mode='r', shape=(10496, 507000)))"),
        ("cfg['response_cache_bytes']/world+(34<<30)", "cfg['response_cache_bytes']/world+auxiliary_budget_bytes(geo, world)"),
        ("from adapters import OrderedTransferPool, ViewBlocks, current_cuda_budget", "from adapters import OrderedTransferPool, ViewBlocks, current_cuda_budget\nfrom bounded_matrix import BoundedReadOnlyMatrix, auxiliary_budget_bytes"),
    ]
    changed = original
    for before, after in replacements:
        assert changed.count(before) == 1, before
        changed = changed.replace(before, after)
    restored = changed
    for before, after in replacements:
        assert restored.count(after) == 1
        restored = restored.replace(after, before)
    assert restored == original
    assert ast.dump(ast.parse(restored), include_attributes=False) == ast.dump(ast.parse(original), include_attributes=False)
    compile(changed, 'run_reconstruction.py', 'exec')
    (target/'run_reconstruction.py').write_text(changed, encoding='utf8', newline='\n')
    old_cfg = read(source/'study.json')
    cfg = dict(old_cfg, matrix_source_release=PARENT_KEY,
               execution_parent_release=PARENT_KEY, execution_gpu='RTX 4090',
               execution_partition=PARTITION, execution_allocated_cpus_per_node=CPUS,
               execution_expected_memory_MiB_per_node=MEM_MIB,
               execution_memory_policy='Platform default; no explicit memory request',
               execution_matrix_storage='Read-only mmap; drop disjoint copied pages after every indexing operation')
    assert all(cfg[k] == v for k, v in old_cfg.items())
    write(target/'study.json', cfg, replace=True)
    shutil.copy2(HERE/'bounded_matrix.py',target/'bounded_matrix.py')
    hashes = {name: sha(target/name) for name in parent['files']}
    hashes['bounded_matrix.py']=sha(target/'bounded_matrix.py')
    changed_files = sorted(name for name, h in hashes.items() if name not in parent['files'] or h != parent['files'][name])
    assert changed_files == ['bounded_matrix.py','run_reconstruction.py','study.json']
    needed = cfg['response_cache_bytes']/8+(3<<30)+2*1312*507000*4+(512<<20)+200*227040*4+(2<<30)
    assert needed <= .8*(MEM_MIB<<20)
    body = dict(files=hashes, study=cfg, scientific_parent_release=PARENT_KEY,
                scientific_parent_manifest_sha256=sha(PARENT_REPORT/'freeze.json'),
                matrix_source_release=PARENT_KEY, execution_only=True)
    body['key'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    write(target/'release_manifest.json', body)
    write(frozen, body)
    audit = dict(time=now(), passed=True, parent=PARENT_KEY, execution=body['key'],
                 changed_files=changed_files, exact_allowed_replacements=replacements,
                 reverse_patch_restores_entire_original_source=True,
                 normalized_entire_program_AST_equal=True,
                 all_original_study_values_unchanged=True,
                 unchanged_frozen_files=len(hashes)-3,
                 host_predicted_peak_bytes=needed, allocated_host_bytes=MEM_MIB<<20,
                 host_fraction=needed/(MEM_MIB<<20), actual_resource_guard=.8,
                 gpu_cache='Original device-budget cache adapts to actual VRAM; actual 80% GPU guard unchanged',
                 GPU_memory_prediction_is_not_measurement=True,
                 matrix_strict_transfer_sha256=sha(PARENT_REPORT/'matrix_strict_transfer.json'),
                 parent_5090_registration_sha256=sha(PARENT_REPORT/'formal_registration.json'),
                 authorization='User explicitly requested an additional identical 8x1 formal job in the 4090 queue; retain original 5090 job')
    write(REPORT/'execution_equivalence.json', audit)
    archive = DATA/('release_'+body['key']+'.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        for name in sorted(hashes):
            tar.add(target/name, arcname=name, recursive=False)
        tar.add(target/'release_manifest.json', arcname='release_manifest.json', recursive=False)
    return body


def launch_files(release):
    files = handoff.launch_files('CART_4090_EXECUTION_RELEASE')
    files = {name: text.replace(MATRIX_ROOT, REMOTE).replace('CART_4090_EXECUTION_RELEASE',release) for name, text in files.items()}
    assert 'export CART_RELEASE='+shlex.quote(release)+'\n' in files['launch.sh']
    assert REMOTE+'/'+HERE.name not in files['launch.sh']
    needle = "'--matrices',os.environ['CART_REMOTE']+'/matrices'"
    assert files['node.py'].count(needle) == 1
    files['node.py'] = files['node.py'].replace(needle, "'--matrices',os.environ['CART_MATRICES']")
    marker = 'export CART_HOST_BYTES\n'
    assert files['launch.sh'].count(marker) == 1
    files['launch.sh'] = files['launch.sh'].replace(marker,
        marker+'export CART_MATRICES='+shlex.quote(MATRIX_ROOT+'/matrices')+'\n')
    compile(files['node.py'], 'node.py', 'exec')
    return files


def submit():
    if (REPORT/'formal_registration.json').exists():
        return read(REPORT/'formal_registration.json')
    if (REPORT/'formal_submission_intent.json').exists():
        raise RuntimeError('Uncertain previous submission; inspect Slurm, never repeat')
    manifest = prepare()
    audit = read(REPORT/'execution_equivalence.json')
    assert sha(PARENT_REPORT/'formal_registration.json') == audit['parent_5090_registration_sha256']
    assert read(REPORT/'local_cpu_regression.json')['passed']
    archive = DATA/('release_'+manifest['key']+'.tar.gz')
    release = REMOTE+'/releases/'+manifest['key']
    with connect() as client:
        client.get_transport().set_keepalive(30)
        quota = command(client, 'timeout 30s /usr/lpp/mmfs/bin/mmlsquota -u scxi717 hpc1', 40)
        rows = [line.split() for line in quota.splitlines() if re.match(r'^hpc1\s+run01\s+USR\s', line)]
        assert len(rows) == 1
        used, soft, hard, doubt = [int(v)*1024 for v in rows[0][3:7]]
        available = min(v for v in (soft, hard) if v > 0)-used-doubt
        partition = command(client, 'scontrol show partition '+PARTITION, 40)
        if available < 16<<30 or 'State=UP' not in partition:
            raise RuntimeError('Actual partition/quota gate failed')
        queue = command(client, "squeue -u scxi717 -h -o '%i|%j|%T|%R'", 40)
        if any(line.split('|')[1] == NAME for line in queue.splitlines() if '|' in line):
            raise RuntimeError('Matching unregistered 4090 job exists; inspect')
        metadata = '''from pathlib import Path
import json
records=RECORDS;root=Path(MATRIX_ROOT)
for row in records:
 p=Path(row['target']);m=json.loads((p.parent/'completed.json').read_text())
 assert p.resolve().parent.parent==root.resolve()/'matrices'
 assert p.stat().st_size==row['bytes'] and m['sha256']==row['sha256'] and m['release']==PARENT_KEY
print('SHARED_MATRIX_METADATA_PASS')'''.replace('RECORDS', repr(read(PARENT_REPORT/'matrix_strict_transfer.json')['files'])).replace('MATRIX_ROOT', repr(MATRIX_ROOT)).replace('PARENT_KEY', repr(PARENT_KEY))
        proof = handoff.rpc(client, metadata, 60)
        handoff.rpc(client, 'from pathlib import Path\nfor p in '+repr([REMOTE,REMOTE+'/logs',REMOTE+'/allocations',REMOTE+'/launcher'])+':Path(p).mkdir(parents=True,exist_ok=True)', 60)
        remote_archive = REMOTE+'/'+archive.name
        with client.open_sftp() as sftp:
            try:
                sftp.stat(remote_archive)
            except FileNotFoundError:
                sftp.put(str(archive), remote_archive)
        deployment = handoff.rpc(client, extraction_program(remote_archive, release, sha(archive)), 180)
        bounded_tests=command(client,shlex.quote(GPU_PY)+' -B '+shlex.quote(release+'/bounded_matrix.py')+' --self-test',120)
        write(REPORT/'linux_bounded_io_regression.json',dict(time=now(),passed=True,output=json.loads(bounded_tests),release=manifest['key'],no_Slurm_validation_job=True))
        files = launch_files(release)
        for name, text in files.items():
            p = REMOTE+'/launcher/'+name
            handoff.rpc(client, 'from pathlib import Path;p=Path('+repr(p)+');raw='+repr(text)+'\nif p.exists():assert p.read_text()==raw\nelse:p.write_text(raw)', 60)
        argv = ['sbatch','--parsable','-p',PARTITION,'--qos=gpugpu','-N','8','-n','8',
                '--ntasks-per-node=1','--cpus-per-task='+str(CPUS),
                '--gres=gpu:1','--time=4320','--job-name='+NAME,'--chdir='+REMOTE,
                '--output='+REMOTE+'/logs/%j.out','--error='+REMOTE+'/logs/%j.err',REMOTE+'/launcher/launch.sh']
        dry = command(client, shlex.join([argv[0], '--test-only']+argv[1:]), 60)
        intent = dict(time=now(),argv=argv,test_only=dry,parent_5090_job='1822462',
                      parent_5090_preserved=True, execution_freeze=manifest['key'],
                      source_freeze=PARENT_KEY, available_quota_bytes=available,
                      required_new_disk_bytes=16<<30, partition=partition,
                      shared_matrix_metadata=proof, deployment=deployment,
                      launcher_sha256={n:hashlib.sha256(t.encode()).hexdigest() for n,t in files.items()},
                      auto_retry=False)
        write(REPORT/'formal_submission_intent.json', intent)
        value = command(client, shlex.join(argv), 60).strip()
        if not re.fullmatch(r'\d+(?:;\S+)?', value):
            raise RuntimeError('Ambiguous sbatch result; do not repeat')
        job = value.split(';')[0]
        registration = dict(time=now(),job=job,release=release,source_freeze=manifest['key'],
            scientific_parent_freeze=PARENT_KEY,result=REMOTE+'/formal_'+job,
            nodes=8,gpus_per_node=1,total_GPUs=8,GPU='RTX4090',partition=PARTITION,
            allocated_cpus_per_node=CPUS,compute_threads_per_rank=8,expected_memory_MiB_per_node=MEM_MIB,memory_policy='platform default; actual allocation checked in formal job',
            iterations=10000,save_step=50,routes=['440_SC','440_JSCC','218_SC'],
            shared_matrices=MATRIX_ROOT+'/matrices',parent_5090_job='1822462',parent_5090_preserved=True)
        write(REPORT/'formal_registration.json', registration)
        detail = command(client, 'scontrol show job -dd '+job, 40)
        (REPORT/'formal_slurm.txt').write_text(detail, encoding='utf8')
        if 'NumNodes=8' not in detail or 'gres/gpu=8' not in detail or 'NumCPUs=64' not in detail:
            raise RuntimeError('Registered allocation differs; inspect existing job')
    assert sha(PARENT_REPORT/'formal_registration.json') == audit['parent_5090_registration_sha256']
    return registration


def configure_handoff():
    # Changes only globals in this independent Python process, never old files.
    handoff.REPORT = REPORT
    handoff.DATA = DATA
    handoff.REMOTE_GPU = REMOTE


def status():
    configure_handoff()
    result = handoff.formal_status()
    write(REPORT/'formal_latest_status.json', result, replace=True)
    return result


def watch():
    if not (REPORT/'formal_registration.json').exists():
        raise RuntimeError('Submit explicitly before starting a monitor')
    regfile = DATA/'controller_registration.json'
    if regfile.exists():
        raise RuntimeError('Monitor already registered; inspect actual identity')
    lock = DATA/'controller.lock'
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode());os.close(fd)
    old = read(PARENT_DATA/'completion_transfer_r6_registration_20261117.json')
    deadline = datetime.datetime.fromisoformat(old['deadline_utc']).timestamp()
    process = psutil.Process()
    state = dict(pid=process.pid,create_time=process.create_time(),cmdline=process.cmdline(),
                 status='ACTIVE',started=now(),deadline_utc=old['deadline_utc'],
                 job=read(REPORT/'formal_registration.json')['job'],
                 finite_monitor=True,periodic_chat_automation=False,cancellation=False,
                 scientific_resubmission=False,script_sha256=sha(Path(__file__)))
    write(regfile, state)
    try:
        while time.time() < deadline:
            if sha(Path(__file__)) != state['script_sha256']:
                raise ValueError('Monitor source changed')
            try:
                current = status()
            except Exception as error:
                state.update(last_check=now(),stage='NETWORK_WAIT',error=repr(error))
                write(regfile,state,replace=True);time.sleep(60);continue
            state.update(last_check=now(),stage='FORMAL_'+current['decision'],
                         progress=current['details'].get('progress.json'))
            write(regfile,state,replace=True)
            if current['decision']=='FAILED':
                raise RuntimeError('Formal job failed; preserve successful files and stop, no automatic re-compute')
            if current['decision']=='SUCCESS':
                configure_handoff()
                state.update(status='STRICTLY_DELIVERED_VISUAL_QA_PENDING',completed=now(),delivery=handoff.fetch())
                write(regfile,state,replace=True);return
            time.sleep(120)
        raise TimeoutError('Original finite deadline reached; no job cancelled')
    except BaseException:
        state.update(status='STOPPED_FAILURE',stopped=now(),traceback=traceback.format_exc())
        write(regfile,state,replace=True);raise
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()):
            lock.unlink()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','submit','status','watch'])
    action=parser.parse_args().action
    if action=='watch':watch()
    else:print(json.dumps({'prepare':prepare,'submit':submit,'status':status}[action](),indent=2))
