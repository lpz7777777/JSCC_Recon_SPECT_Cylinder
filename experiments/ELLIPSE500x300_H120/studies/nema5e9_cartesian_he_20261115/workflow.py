"""Explicit new-study actions. Never cancel jobs or resume an old controller."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
from common import sha, read, write, now
from prepare import CODE, EXP, ROOT, DATA, REPORT, SERVER, SERVER_ROOT

REMOTE_SERVER = SERVER_ROOT+'/experiments/ELLIPSE500x300_H120/generated/'+DATA.name
REMOTE_GPU_BASE = '/data/run01/scxi717/lpz/'+ROOT.name+'/experiments/ELLIPSE500x300_H120'
REMOTE_GPU = REMOTE_GPU_BASE+'/generated/'+DATA.name
GPU_PY = '/data/home/scxi717/.conda/envs/torch/bin/python'
SERVER_PY = SERVER_ROOT+'/.venv/bin/python'
SERVER_DEPS = REMOTE_SERVER+'/runtime_deps'
sys.path[:0] = [str(EXP), str(ROOT/'experiments/FOV120')]
from reconstruction_ssh import connect
from ehe_5e9_workflow import command


def server(code, timeout=120):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', SERVER,
        'env PYTHONPATH='+shlex.quote(SERVER_DEPS)+' '+SERVER_PY+' -B -c '+shlex.quote(code)], capture_output=True, text=True,
        encoding='utf8', errors='replace', timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout


def package():
    f = read(REPORT/'freeze.json')
    payload = DATA/'payload'
    for name, h in f['files'].items():
        assert sha(payload/name) == h, name
    archive = DATA/('release_'+f['key']+'.tar.gz')
    if not archive.exists():
        with tarfile.open(archive, 'w:gz') as t:
            for n in sorted(f['files']):
                t.add(payload/n, arcname=n, recursive=False)
            t.add(payload/'release_manifest.json', arcname='release_manifest.json', recursive=False)
    return f, archive


def extraction_program(archive, target, expected):
    return '''from pathlib import Path
import tarfile,hashlib,json
a=Path(ARCHIVE);r=Path(TARGET)
assert hashlib.sha256(a.read_bytes()).hexdigest()==EXPECTED
with tarfile.open(a) as t:
 for m in t:
  if not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts:raise ValueError('Unsafe archive')
  p=r/m.name;raw=t.extractfile(m).read();p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==raw,str(p)
  else:
   with p.open('xb') as stream:stream.write(raw)
f=json.loads((r/'release_manifest.json').read_text())
for n,h in f['files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
print('RELEASE_SHA_PASS',f['key'])
'''.replace('ARCHIVE', repr(archive)).replace('TARGET', repr(target)).replace('EXPECTED', repr(expected))


def deploy_server():
    if (REPORT/'server_deployment.json').exists():
        return read(REPORT/'server_deployment.json')
    f, archive = package()
    release = REMOTE_SERVER+'/releases/'+f['key']
    remote_tar = REMOTE_SERVER+'/'+archive.name
    server('from pathlib import Path;Path('+repr(REMOTE_SERVER)+').mkdir(parents=True,exist_ok=True)')
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', str(archive), SERVER+':'+remote_tar], check=True, timeout=180)
    proof = server(extraction_program(remote_tar, release, sha(archive)))
    tests = server('import subprocess;subprocess.run(['+repr(SERVER_PY)+",'-B',"+
        repr(release+'/test_cartesian.py')+'],check=True)', 180)
    (REPORT/'linux_tests_stdout.txt').write_text(tests, encoding='utf8')
    result = dict(time=now(), release=release, key=f['key'], proof=proof, linux_tests_passed=True,
                  archive_sha256=sha(archive), host=SERVER)
    write(REPORT/'server_deployment.json', result)
    return result


def start_matrix():
    registration = REPORT/'matrix_registration.json'
    if registration.exists():
        print(json.dumps(read(registration), indent=2))
        return
    deployed = deploy_server()
    if not (REPORT/'server_runtime_dependencies.json').exists():
        wheels = list((DATA/'runtime_wheels').glob('psutil-7.0.0-*.whl'))
        if len(wheels) != 1:
            raise ValueError('Supply the pinned Linux psutil 7.0.0 wheel in runtime_wheels')
        wheel = wheels[0]
        remote_wheel = REMOTE_SERVER+'/'+wheel.name
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', str(wheel), SERVER+':'+remote_wheel], check=True, timeout=120)
        proof = server('''import subprocess,sys,json,hashlib
from pathlib import Path
p=Path(DEPS)
wheel=Path(WHEEL)
assert hashlib.sha256(wheel.read_bytes()).hexdigest()==WHEEL_SHA
if not p.exists():
 subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(p),str(wheel)],check=True,stdout=sys.stderr)
sys.path.insert(0,str(p));import importlib;importlib.invalidate_caches();import psutil
assert psutil.__version__=='7.0.0'
print(json.dumps(dict(version=psutil.__version__,path=str(p),files={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and '__pycache__' not in str(f)})))
'''.replace('DEPS', repr(SERVER_DEPS)).replace('WHEEL_SHA', repr(sha(wheel))).replace('WHEEL', repr(remote_wheel)), 180)
        write(REPORT/'server_runtime_dependencies.json', json.loads(proof))
    out = REMOTE_SERVER+'/matrices_'+deployed['key']
    # Remote exclusive intent is the authority if SSH returns ambiguously.
    program = '''from pathlib import Path
import os,subprocess,json,time,psutil
root=Path(ROOT);release=Path(RELEASE);out=Path(OUTPUT)
reg=root/'matrix_registration.json';intent=root/'matrix_launch_intent.json'
if reg.exists():print(reg.read_text());raise SystemExit(0)
if intent.exists():raise RuntimeError('Ambiguous matrix intent; inspect PID before any retry')
args=[PYTHON,'-B',str(release/'build_matrix.py'),'--output',str(out),'--cuda','4']
with intent.open('x') as f:json.dump(dict(time=time.time(),argv=args),f)
stdout=(root/'matrix.stdout.log').open('xb');stderr=(root/'matrix.stderr.log').open('xb')
p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,cwd=release)
proc=psutil.Process(p.pid)
record=dict(pid=p.pid,create_time=proc.create_time(),cmdline=proc.cmdline(),release=str(release),output=str(out),gpu_index=4,host=HOST)
with reg.open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
'''.replace('ROOT', repr(REMOTE_SERVER)).replace('RELEASE', repr(deployed['release'])).replace(
        'OUTPUT', repr(out)).replace('PYTHON', repr(SERVER_PY)).replace('HOST', repr(SERVER))
    reg = json.loads(server(program))
    write(registration, reg)
    print(json.dumps(reg, ensure_ascii=False, indent=2))


def matrix_status():
    reg = read(REPORT/'matrix_registration.json')
    code = '''import json,psutil
from pathlib import Path
r=REG;p=Path(r['output']);d=dict(registration=r,alive=False)
if psutil.pid_exists(r['pid']):
 try:
  q=psutil.Process(r['pid']);d['alive']=abs(q.create_time()-r['create_time'])<.1 and q.cmdline()==r['cmdline'];d['process_status']=q.status()
 except (psutil.NoSuchProcess,psutil.ZombieProcess):d['process_status']='exited'
for n in ['started.json','progress.json','completed.json']:
 f=p/n
 if f.exists():d[n]=json.loads(f.read_text())
for n in ['matrix.stdout.log','matrix.stderr.log']:
 f=Path(BASE)/n
 if f.exists():
  with f.open('rb') as t:t.seek(max(0,f.stat().st_size-5000));d[n]=t.read().decode(errors='replace')
print(json.dumps(d))
'''.replace('REG', repr(reg)).replace('BASE', repr(REMOTE_SERVER))
    snap = json.loads(server(code))
    snap['time'] = now()
    write(REPORT/'matrix_latest_status.json', snap, replace=True)
    return snap


def gpu_preflight():
    """Read only; exact actual directory quota and active jobs before any write."""
    try:
        with connect() as c:
            account = command(c, "squeue -u scxi717 -h -o '%i|%j|%T|%D|%b'", 40)
            quota = command(c, 'timeout 30s /usr/lpp/mmfs/bin/mmlsquota -u scxi717 hpc1', 40)
            partition = command(c, 'scontrol show partition gpu_5090', 30)
        lines = [line.split() for line in quota.splitlines() if re.match(r'^hpc1\s+run01\s+USR\s', line)]
        if len(lines) != 1:
            raise ValueError('Cannot establish actual run01 user quota')
        used, soft, hard, doubt = [int(v)*1024 for v in lines[0][3:7]]
        limit = min(v for v in [soft, hard] if v > 0)
        free = limit-used-doubt
        required = 80<<30  # three 19.83GiB responses + results, with reserve
        result = dict(time=now(), passed=free >= required and 'State=UP' in partition,
            queue=account, partition=partition, quota=quota, available_bytes=free,
            required_bytes=required, no_jobs_submitted=True)
    except Exception as e:
        result = dict(time=now(), passed=False, error=repr(e), no_jobs_submitted=True)
    write(REPORT/'gpu_preflight_latest.json', result, replace=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['deploy-server', 'start-matrix', 'status', 'gpu-preflight'])
    action = p.parse_args().action
    if action == 'deploy-server':
        print(json.dumps(deploy_server(), indent=2))
    elif action == 'start-matrix':
        start_matrix()
    elif action == 'status':
        print(json.dumps(matrix_status(), indent=2))
    else:
        print(json.dumps(gpu_preflight(), indent=2))


if __name__ == '__main__':
    main()
