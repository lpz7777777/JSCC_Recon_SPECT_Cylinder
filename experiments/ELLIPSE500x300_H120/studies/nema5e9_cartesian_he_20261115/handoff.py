"""Single bounded matrix transfer and formal submission, with immutable intent.

Completed matrix files stream directly server -> GPU through RAM. Local disk
never receives the ~64GB response. Existing target files require exact SHA.
Any uncertain submit/partial transfer stops; no automatic scientific rerun.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import time
from common import sha, read, write, now
from prepare import CODE, DATA, REPORT
from workflow import (package, connect, command, server, matrix_status, gpu_preflight,
    extraction_program, REMOTE_GPU, REMOTE_GPU_BASE, GPU_PY, SERVER, REMOTE_SERVER)


def rpc(client, code, timeout=120):
    return command(client, shlex.quote(GPU_PY)+' -B -c '+shlex.quote(code), timeout)


def checksum(client, file, timeout=900):
    code = '''import hashlib
from pathlib import Path
p=Path(FILE)
if not p.exists():print('ABSENT')
else:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 print(h.hexdigest())
'''.replace('FILE', repr(file))
    return rpc(client, code, timeout).strip()


def transfer_one(client, source, target, expected, size):
    existing = checksum(client, target)
    if existing != 'ABSENT':
        if existing != expected:
            raise ValueError('Existing destination differs; overwrite forbidden')
        return dict(source=source, target=target, bytes=size, sha256=expected, reused_verified=True)
    partial = target+'.partial'
    rpc(client, 'from pathlib import Path;p=Path('+repr(partial)+');assert not p.exists(),"Partial transfer requires explicit inspection";p.parent.mkdir(parents=True,exist_ok=True)')
    ssh = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', SERVER,
        'cat -- '+shlex.quote(source)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    h, count = hashlib.sha256(), 0
    started = time.monotonic()
    try:
        with client.open_sftp() as sftp:
            sftp.get_channel().settimeout(120)
            with sftp.open(partial, 'wx', bufsize=1<<20) as out:
                out.set_pipelined(True)
                while True:
                    block = ssh.stdout.read(4<<20)
                    if not block:
                        break
                    if count+len(block) > size:
                        raise ValueError('Source grew beyond frozen matrix size')
                    out.write(block)
                    h.update(block)
                    count += len(block)
                    if time.monotonic()-started > 8*3600:
                        raise TimeoutError('Bounded matrix transfer exceeded eight hours')
            if ssh.wait(timeout=60) or count != size or h.hexdigest() != expected:
                raise ValueError('Source transfer size/SHA/exit failed')
        if checksum(client, partial) != expected:
            raise ValueError('Destination SHA failed')
        rpc(client, 'from pathlib import Path;import os;p=Path('+repr(partial)+');q=Path('+repr(target)+');assert not q.exists();os.rename(p,q)')
    finally:
        if ssh.poll() is None:
            ssh.terminate()
            ssh.wait(timeout=30)
    return dict(source=source, target=target, bytes=count, sha256=expected,
                elapsed_seconds=time.monotonic()-started, reused_verified=False)


def launch_files(release):
    launch = '''#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NCCL_SOCKET_IFNAME=bond0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CART_RELEASE=__RELEASE__
export CART_REMOTE=__REMOTE__
export CART_INPUT=__INPUT__
export CART_PY=__PY__
scontrol show job -dd "$SLURM_JOB_ID" > "$CART_REMOTE/allocations/${SLURM_JOB_ID}.txt"
CART_HOST_BYTES=$("$CART_PY" -B "$CART_RELEASE/verify_first_scatter.py" --allocation-only "$CART_REMOTE/allocations/${SLURM_JOB_ID}.txt" --nodes 8)
export CART_HOST_BYTES
export MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((35000+SLURM_JOB_ID%20000))
timeout --signal=TERM --kill-after=60s 258600s srun --label --kill-on-bad-exit=1 --chdir=/tmp "$CART_PY" -B __NODE__
'''.replace('__RELEASE__', shlex.quote(release)).replace('__REMOTE__', shlex.quote(REMOTE_GPU)).replace(
        '__INPUT__', shlex.quote(REMOTE_GPU_BASE+'/generated')).replace('__PY__', shlex.quote(GPU_PY)).replace(
        '__NODE__', shlex.quote(REMOTE_GPU+'/launcher/node.py'))
    node = '''import os
from pathlib import Path
job=os.environ['SLURM_JOB_ID'];root=Path(os.environ['CART_RELEASE'])
cache=Path('/tmp')/('cartesian_'+job+'_'+os.environ['SLURM_PROCID']);cache.mkdir(exist_ok=True)
for key,leaf in [('TMPDIR','tmp'),('TORCHINDUCTOR_CACHE_DIR','torch'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda')]:
 p=cache/leaf;p.mkdir(exist_ok=True);os.environ[key]=str(p)
args=[os.environ['CART_PY'],'-B','-m','torch.distributed.run','--nnodes=8','--nproc_per_node=1',
 '--node_rank='+os.environ['SLURM_PROCID'],'--master_addr='+os.environ['MASTER_ADDR'],
 '--master_port='+os.environ['MASTER_PORT'],'--rdzv_backend=static','--rdzv_id='+job+'_cartesian','--max_restarts=0',
 str(root/'run_reconstruction.py'),'--input-root',os.environ['CART_INPUT'],
 '--matrices',os.environ['CART_REMOTE']+'/matrices','--output',os.environ['CART_REMOTE']+'/formal_'+job,
 '--host-bytes',os.environ['CART_HOST_BYTES']]
os.execv(args[0],args)
'''
    return {'launch.sh': launch, 'node.py': node}


def transfer_and_submit():
    reg = REPORT/'formal_registration.json'
    if reg.exists():
        return read(reg)
    if (REPORT/'formal_submission_intent.json').exists():
        raise RuntimeError('Ambiguous sbatch; inspect accounting before any retry')
    snap = matrix_status()
    if snap['alive'] or 'completed.json' not in snap:
        raise RuntimeError('Matrix process must have completed and exited before handoff')
    capacity = gpu_preflight()
    if not capacity['passed']:
        raise RuntimeError('GPU connection/quota/resource preflight not ready')
    f, archive = package()
    complete = snap['completed.json']
    if complete['release'] != f['key']:
        raise ValueError('Matrix and reconstruction releases differ')
    release = REMOTE_GPU+'/releases/'+f['key']
    started = time.monotonic()
    with connect() as c:
        c.get_transport().set_keepalive(30)
        rpc(c, 'from pathlib import Path\nfor p in '+repr([REMOTE_GPU+'/logs', REMOTE_GPU+'/allocations', REMOTE_GPU+'/launcher'])+':Path(p).mkdir(parents=True,exist_ok=True)')
        tar = REMOTE_GPU+'/'+archive.name
        if checksum(c, tar) == 'ABSENT':
            with c.open_sftp() as sftp:
                sftp.put(str(archive), tar)
        rpc(c, extraction_program(tar, release, sha(archive)))
        tests = command(c, shlex.quote(GPU_PY)+' -B '+shlex.quote(release+'/test_cartesian.py'), 180)
        (REPORT/'gpu_login_cpu_tests.txt').write_text(tests, encoding='utf8')
        transfers = []
        for channel in ['A440', 'A218', 'C440to218']:
            record = complete['channels'][channel]
            source = snap['registration']['output']+'/'+channel+'/B_detector_zyx.float32'
            target = REMOTE_GPU+'/matrices/'+channel+'/B_detector_zyx.float32'
            transfer = transfer_one(c, source, target, record['sha256'], record['bytes'])
            transfers.append(transfer)
            text = json.dumps(record, indent=2)+'\n'
            rpc(c, 'from pathlib import Path;p=Path('+repr(REMOTE_GPU+'/matrices/'+channel+'/completed.json')+');raw='+repr(text)+'\nif p.exists():assert p.read_text()==raw\nelse:p.write_text(raw)')
            write(REPORT/'matrix_transfer_progress.json', dict(time=now(), files=transfers), replace=True)
        write(REPORT/'matrix_strict_transfer.json', dict(time=now(), passed=True, files=transfers,
            elapsed_seconds=time.monotonic()-started, release=f['key']))
        launchers = launch_files(release)
        for name, text in launchers.items():
            compile(text, name, 'exec') if name.endswith('.py') else None
            rpc(c, 'from pathlib import Path;p=Path('+repr(REMOTE_GPU+'/launcher/'+name)+');raw='+repr(text)+'\nif p.exists():assert p.read_text()==raw\nelse:p.write_text(raw)')
        write(REPORT/'execution_launcher.json', dict(release=f['key'], files={n:hashlib.sha256(t.encode()).hexdigest() for n,t in launchers.items()}))
        queue = command(c, "squeue -u scxi717 -h -o '%i|%j|%T'", 40)
        if 'NEMA5e9_CartHE' in queue:
            raise RuntimeError('Matching job already exists without local registration; inspect')
        args = ['sbatch', '--parsable', '-p', 'gpu_5090', '--qos=gpugpu', '-N', '8', '-n', '8',
                '--ntasks-per-node=1', '--cpus-per-task=8', '--gres=gpu:1', '--time=4320',
                '--job-name=NEMA5e9_CartHE', '--chdir='+REMOTE_GPU,
                '--output='+REMOTE_GPU+'/logs/%j.out', '--error='+REMOTE_GPU+'/logs/%j.err', REMOTE_GPU+'/launcher/launch.sh']
        dry = command(c, ' '.join(shlex.quote(s) for s in [args[0], '--test-only']+args[1:]), 45)
        write(REPORT/'formal_submission_intent.json', dict(time=now(), argv=args,
            test_only=dry, source_freeze=f['key'], auto_retry=False))
        value = command(c, ' '.join(shlex.quote(s) for s in args), 45).strip()
        if not re.fullmatch(r'\d+(?:;\S+)?', value):
            raise RuntimeError('Ambiguous sbatch output')
        job = value.split(';')[0]
        result = dict(job=job, time=now(), release=release, source_freeze=f['key'],
            result=REMOTE_GPU+'/formal_'+job, nodes=8, gpus_per_node=1, total_GPUs=8,
            routes=['440_SC', '440_JSCC', '218_SC'], iterations=10000, save_step=50)
        write(reg, result)
        actual = command(c, 'scontrol show job -dd '+job, 30)
        (REPORT/'formal_slurm.txt').write_text(actual, encoding='utf8')
        if 'NumNodes=8' not in actual or 'gres/gpu=8' not in actual:
            raise RuntimeError('Submitted allocation mismatch; inspect registered job')
        return result


def formal_status():
    reg = read(REPORT/'formal_registration.json')
    with connect() as c:
        account = command(c, 'sacct -n -P -j '+reg['job']+' --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS', 45)
        detail = json.loads(rpc(c, '''from pathlib import Path
import json
p=Path(RESULT);d={}
for n in ['started.json','progress.json','completed.json']:
 q=p/n
 if q.exists():d[n]=json.loads(q.read_text())
print(json.dumps(d))
'''.replace('RESULT', repr(reg['result']))))
    records = [line.split('|') for line in account.splitlines() if line.strip()]
    required = {reg['job']+suffix for suffix in ['', '.batch', '.extern', '.0']}
    success = required <= {r[0] for r in records} and all(r[1]=='COMPLETED' and r[2]=='0:0' for r in records)
    failed = any(r[1].split()[0] in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'] for r in records)
    return dict(time=now(), job=reg['job'], accounting=account, details=detail,
                decision='SUCCESS' if success else ('FAILED' if failed else 'WAIT'))


def fetch():
    if (REPORT/'strict_delivery.json').exists():
        return read(REPORT/'strict_delivery.json')
    state = formal_status()
    if state['decision'] != 'SUCCESS':
        raise RuntimeError('Full root/batch/extern/step success required')
    reg = read(REPORT/'formal_registration.json')
    archive_path = REMOTE_GPU+'/delivery_'+reg['job']+'.tar.gz'
    program = '''from pathlib import Path
import json,hashlib,tarfile,os
r=Path(RESULT);archive=Path(ARCHIVE)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
complete=json.loads((r/'completed.json').read_text());assert complete['total_checkpoints']==600
for route in ['440_SC','440_JSCC','218_SC']:
 d=r/route;m=json.loads((d/'completed.json').read_text());assert m['frames']==200
 assert len(list((d/'checkpoints').glob('*.float32')))==200
 for n,h in m['files'].items():assert sha(d/n)==h,n
manifest={str(p.relative_to(r)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in r.rglob('*') if p.is_file()}
receipt=dict(files=manifest,completed=complete)
m=r.parent/('delivery_'+JOB+'.json')
if m.exists():assert json.loads(m.read_text())==receipt
else:m.write_text(json.dumps(receipt,indent=2)+'\n')
if not archive.exists():
 tmp=archive.with_name(archive.name+'.partial')
 with tarfile.open(tmp,'w:gz',compresslevel=1) as t:
  t.add(r,arcname='result');t.add(m,arcname='manifest.json')
 os.rename(tmp,archive)
print(json.dumps(dict(bytes=archive.stat().st_size,sha256=sha(archive),manifest=receipt)))
'''.replace('RESULT', repr(reg['result'])).replace('ARCHIVE', repr(archive_path)).replace('JOB', repr(reg['job']))
    with connect() as c:
        receipt = json.loads(rpc(c, program, 1200))
        local = DATA/Path(archive_path).name
        if shutil.disk_usage(DATA).free < receipt['bytes']+sum(v['bytes'] for v in receipt['manifest']['files'].values())+2*(1<<30):
            raise OSError('Insufficient local delivery space')
        if not local.exists():
            partial = local.with_suffix(local.suffix+'.partial')
            with c.open_sftp() as sftp:
                sftp.get(archive_path, str(partial))
            if sha(partial) != receipt['sha256']:
                raise ValueError('Strict archive transfer SHA failed; partial retained')
            partial.rename(local)
    if sha(local) != receipt['sha256']:
        raise ValueError('Local archive changed')
    destination = DATA/('delivered_'+reg['job'])
    destination.mkdir(exist_ok=True)
    with tarfile.open(local) as t:
        for m in t:
            if m.isdir():
                continue
            path = Path(m.name)
            if not m.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe result archive')
            p = destination/path
            p.parent.mkdir(parents=True, exist_ok=True)
            data = t.extractfile(m).read()
            if p.exists():
                if p.read_bytes() != data:
                    raise ValueError('Existing delivered member differs')
            else:
                p.write_bytes(data)
    for n, v in receipt['manifest']['files'].items():
        p = destination/'result'/n
        if p.stat().st_size != v['bytes'] or sha(p) != v['sha256']:
            raise ValueError('Strict delivered member failed: '+n)
    result = dict(time=now(), passed=True, job=reg['job'], files=len(receipt['manifest']['files']),
        checkpoints=600, archive_sha256=receipt['sha256'], local_root=str(destination),
        successful_accounting=state['accounting'], visual_QA_passed=False)
    write(REPORT/'archive_manifest.json', receipt)
    write(REPORT/'strict_delivery.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['submit', 'status', 'fetch'])
    a = p.parse_args().action
    print(json.dumps({'submit':transfer_and_submit, 'status':formal_status, 'fetch':fetch}[a](), indent=2))
