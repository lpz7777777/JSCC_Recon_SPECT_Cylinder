"""One authorized 8x1 formal; identical frozen payload, single-file delivery.

Avoid slow shared-FS per-file deployment. Only small source/calibration files
are unpacked in each job-private tmpdir; event responses remain in RAM/VRAM.
"""
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import prejune_8x1_workflow as w
from ehe_5e9_workflow import connection, command, q


def pipe_command(c, cmd, data, limit=300):
    stdin, out, err = c.exec_command(cmd, timeout=60)
    stdin.write(data); stdin.flush(); stdin.channel.shutdown_write()
    channel=out.channel; output=[]; errors=[]; start=time.monotonic()
    while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
        if channel.recv_ready():
            block=channel.recv(65536); output.append(block)
            print(block.decode(errors='replace'),end='',flush=True)
        if channel.recv_stderr_ready():errors.append(channel.recv_stderr(65536))
        if time.monotonic()-start>limit:
            channel.close(); raise TimeoutError('Remote pipe timed out; inspect before retry')
        time.sleep(.05)
    stdout=b''.join(output).decode(); stderr=b''.join(errors).decode()
    if channel.recv_exit_status():raise RuntimeError(stdout+stderr)
    return stdout,stderr


def prepare():
    f=w.prepare()
    archive=w.DATA/'frozen_payload_v2.zip'
    if not archive.exists():
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for name in f['files']:
                info=zipfile.ZipInfo(name,date_time=(2026,10,26,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED
                z.writestr(info,(w.PAYLOAD/name).read_bytes())
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())==set(f['files'])
        assert all(hashlib.sha256(z.read(n)).hexdigest()==v for n,v in f['files'].items())
    sha=w.digest(archive)
    remote=w.REMOTE+'/releases/'+f['key']+'.zip'
    unpack="""import hashlib,json,pathlib,sys,zipfile
archive=pathlib.Path(sys.argv[1]);target=pathlib.Path(sys.argv[2])
assert hashlib.sha256(archive.read_bytes()).hexdigest()==sys.argv[3]
with zipfile.ZipFile(archive) as z:
 for name in z.namelist():
  p=pathlib.PurePosixPath(name)
  assert not p.is_absolute() and '..' not in p.parts
 z.extractall(target)
contract=json.loads((target/'contract.json').read_text())
for name,sha in contract['release_files'].items():
 assert hashlib.sha256((target/name).read_bytes()).hexdigest()==sha,name
assert hashlib.sha256((target/'contract.json').read_bytes()).hexdigest()==sys.argv[4]
print('NODE_SOURCE_UNPACK_SHA_OK',flush=True)
"""
    node_script='''set -euo pipefail
export TMPDIR="/tmp/nema_prejune8x1_${SLURM_JOB_ID}_${SLURM_PROCID}"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR" TEMP="$TMPDIR" CUDA_CACHE_PATH="$TMPDIR/cuda" TRITON_CACHE_DIR="$TMPDIR/triton" TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
'''+w.GPU_PYTHON+' -B -c '+q(unpack)+' '+q(remote)+' "$TMPDIR/release" '+q(sha)+' '+q(f['contract_sha256'])+'\n'+\
        'exec '+w.GPU_PYTHON+' -B -m torch.distributed.run --nnodes=8 --nproc_per_node=1 --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --rdzv_backend=static --max_restarts=0 "$TMPDIR/release/run_prejune_compton.py" --release "$TMPDIR/release" --base '+q(w.GPU_PROJECT)+' --output '+q(w.REMOTE)+'/formal_"${SLURM_JOB_ID}"\n'
    memory_parser="import re,sys; s=sys.stdin.read(); n=int(re.search(r'\\bNumNodes=(\\d+)',s)[1]); assert n==8; t=re.search(r'\\bAllocTRES=([^\\s]+)',s)[1]; m=re.search(r'(?:^|,)mem=([0-9.]+)([KMGT])(?:,|$)',t); print(int(float(m[1])*1024**('KMGT'.index(m[2])+1)/n))"
    script='''#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NCCL_SOCKET_IFNAME=bond0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
'''+ 'export JSCC_HOST_BYTES_NODE=$(scontrol show job "$SLURM_JOB_ID" | '+w.GPU_PYTHON+' -B -c '+q(memory_parser)+')\n'+\
        'scontrol show job "$SLURM_JOB_ID"\n'+\
        'export MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)\n'+\
        'export MASTER_PORT=$((40000+SLURM_JOB_ID%10000))\n'+\
        'timeout --signal=TERM --kill-after=60s 258600s srun --label --kill-on-bad-exit=1 bash -c '+q(node_script)+'\n'
    launch=w.DATA/'formal_archive.sh'
    if launch.exists():assert launch.read_text('utf8')==script
    else:launch.write_text(script,encoding='utf8',newline='\n')
    delivery=dict(archive=remote,archive_sha256=sha,archive_bytes=archive.stat().st_size,
        freeze_key=f['key'],contract_sha256=f['contract_sha256'],files=f['files'],
        scientific_payload_changed=False,execution='identical sources unpacked in job-private tmpdir; original shared projection/matrix inputs; RAM/VRAM response cache',
        launcher_sha256=w.digest(launch))
    marker=w.REPORT/'archive_delivery_freeze.json'
    if marker.exists():assert w.read(marker)==delivery
    else:w.write(marker,delivery)
    return f,archive,launch,delivery


def submit():
    f,archive,launch,d=prepare()
    if (w.REPORT/'formal_job.json').exists():return w.read(w.REPORT/'formal_job.json')
    if (w.REPORT/'submission_intent.json').exists():raise RuntimeError('Unresolved submit intent; no retry')
    with connection('gpu') as c:
        queue=command(c,"squeue -h -u scxi717 -o '%i|%j|%T|%R'",30)
        if 'NEMA5e9_preJune8x1' in queue:raise RuntimeError('Existing 8x1; inspect registration')
        (w.REPORT/'original_job_before_archive.txt').write_text(command(c,'scontrol show job 1739454',30),encoding='utf8')
        code="""import hashlib,os,pathlib,sys
p=pathlib.Path(%r);expected=%r
data=sys.stdin.buffer.read();assert hashlib.sha256(data).hexdigest()==expected
print('ARCHIVE_RECEIVED',len(data),flush=True)
s=os.statvfs(p.parent);assert s.f_bavail*s.f_frsize>16*1024**3
if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
else:
 with p.open('xb') as out:out.write(data);out.flush();os.fsync(out.fileno())
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
print('ARCHIVE_PERSISTENT_SHA_OK',flush=True)
""" % (d['archive'],d['archive_sha256'])
        proof,_=pipe_command(c,w.GPU_PYTHON+' -B -c '+q(code),archive.read_bytes())
        compatibility="""import hashlib,pathlib,sys,zipfile,json,ast
p=pathlib.Path(%r);assert hashlib.sha256(p.read_bytes()).hexdigest()==%r
print('COMPAT_ARCHIVE_SHA_OK',flush=True)
with zipfile.ZipFile(p) as z:
 assert set(z.namelist())==set(%r)
 for name in z.namelist():
  if name.endswith('.py'):compile(z.read(name).decode('utf8'),name,'exec')
 assert hashlib.sha256(z.read('contract.json')).hexdigest()==%r
print('COMPAT_SOURCE_COMPILE_OK',flush=True)
sys.path.insert(0,str(p))
import run_prejune_compton as runner
print('COMPAT_TORCH_RUNNER_IMPORTED',flush=True)
assert runner.sha(p)==%r
print(json.dumps(dict(python=sys.version,torch=runner.torch.__version__,sha_passed=True,import_passed=True,all_python_sources_compile=True,gpu_test_performed=False)))
""" % (d['archive'],d['archive_sha256'],f['files'],f['contract_sha256'],d['archive_sha256'])
        check,_=pipe_command(c,'env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '+w.GPU_PYTHON+' -B -c '+q(compatibility),b'',300)
        w.write(w.REPORT/'archive_import_check.json',json.loads(check.strip().splitlines()[-1]))
        pipe_command(c,'bash -n',launch.read_bytes(),30)
        opts='-p gpu_5090 --qos=gpugpu -N 8 -n 8 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1 --time=3-00:00:00 --job-name=NEMA5e9_preJune8x1 --chdir='+q(w.GPU_PROJECT)+' --output='+q(w.REMOTE+'/logs/formal.%j.out')+' --error='+q(w.REMOTE+'/logs/formal.%j.err')
        out,err=pipe_command(c,'sbatch --test-only '+opts,launch.read_bytes(),60)
        w.write(w.REPORT/'archive_deployment.json',dict(proof=proof,scheduler_test_only=out+err,queue_before=queue,delivery=d,import_check_sha256=w.digest(w.REPORT/'archive_import_check.json')))
        w.write(w.REPORT/'submission_intent.json',dict(command='sbatch --parsable '+opts,stdin_script=str(launch),script_sha256=w.digest(launch),time=time.time(),resolved=False,original_job_unchanged=1739454))
        reply,_=pipe_command(c,'sbatch --parsable '+opts,launch.read_bytes(),60)
        job=reply.strip().split(';')[0]
        if not job.isdigit():raise RuntimeError('Ambiguous submission; do not retry: '+reply)
        rec=dict(job=int(job),study=w.TAG,stage='formal',release_archive=d['archive'],release_archive_sha256=d['archive_sha256'],contract_sha256=f['contract_sha256'],output=w.REMOTE+'/formal_'+job,
            nodes=8,gpus_per_node=1,physical_gpus=8,logical_ranks=16,channel='440 keV pure Compton',iterations=10000,save_every=50,subsets=8,retained_parent_job=1739454,new_transport=False,separate_validation_jobs=0,four_gpu_submitted=False)
        w.write(w.REPORT/'formal_job.json',rec)
        status=command(c,'scontrol show job '+job+'; squeue -h -u scxi717 -o "%i|%j|%T|%R"',30)
        (w.REPORT/'submitted_status.txt').write_text(status,encoding='utf8')
        print(status,flush=True)
        return rec


if __name__=='__main__':print(json.dumps(submit(),ensure_ascii=False,indent=2))
