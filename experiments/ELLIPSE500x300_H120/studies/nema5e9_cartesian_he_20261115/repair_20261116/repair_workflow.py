"""New recovery registration; never rewrite the failed registration or freeze."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import read, write, sha, now
from prepare import DATA, REPORT
from workflow import server, SERVER, SERVER_PY, SERVER_DEPS, REMOTE_SERVER

MANIFEST = REPORT/'repair_release_20261116.json'
REG = REPORT/'matrix_repair_registration_20261116.json'
BINDING = REPORT/'completion_repair_binding_20261116.json'
LOCAL_REG = DATA/'completion_repair_registration_20261116.json'


def verify():
    b = read(BINDING)
    for name, expected in b['files'].items():
        if sha(name) != expected:
            raise ValueError('Recovery binding changed: '+name)
    return b


def status():
    reg = read(REG)
    code = '''from pathlib import Path
import psutil,json
r=REG;p=Path(r['output']);d=dict(registration=r,alive=False)
try:
 q=psutil.Process(r['pid']);d['alive']=abs(q.create_time()-r['create_time'])<.1 and q.cmdline()==r['cmdline'];d['process_status']=q.status()
except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
for n in ['started.json','progress.json','completed.json']:
 f=p/n
 if f.exists():d[n]=json.loads(f.read_text())
for n in ['matrix.stdout.log','matrix.stderr.log']:
 f=Path(r['repair_directory'])/n
 if f.exists():
  with f.open('rb') as s:s.seek(max(0,f.stat().st_size-3000));d[n]=s.read().decode(errors='replace')
print(json.dumps(d))
'''.replace('REG',repr(reg))
    d=json.loads(server(code))
    d['time']=now()
    write(REPORT/'matrix_repair_latest_status_20261116.json',d,replace=True)
    return d


def start():
    if REG.exists():
        raise RuntimeError('Recovery already registered; use status, never duplicate')
    oldlocal=read(DATA/'completion_once_registration.json')
    try:
        p=psutil.Process(oldlocal['pid'])
        if abs(p.create_time()-oldlocal['create_time'])<.1 and p.cmdline()==oldlocal['cmdline']:
            raise RuntimeError('Original controller still alive')
    except psutil.NoSuchProcess:
        pass
    original_binding=read(REPORT/'completion_source_manifest.json')
    for name, expected in original_binding['files'].items():
        if sha(name)!=expected:
            raise ValueError('Original source changed: '+name)
    if not MANIFEST.exists():
        original=read(REPORT/'matrix_registration.json')
        proof=json.loads(server('''from pathlib import Path
import hashlib,json,numpy as np
r=REG;p=Path(r['output']);root=Path(r['release'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (p/'completed.json').exists()
stderr=p.parent/'matrix.stderr.log'
assert 'FileNotFoundError' in stderr.read_text() and 'SysMat_withScatter' in stderr.read_text()
print(json.dumps(dict(original_stderr_sha256=sha(stderr),physics={c:np.fromfile(root/'parameters'/c/'Params_Physics.dat','<f4').tolist() for c in ['A440','A218','C440to218']})))
'''.replace('REG',repr(original))))
        plan=dict(scientific_release=read(REPORT/'freeze.json')['key'], original_release=original['release'],
            original_output=original['output'], original_registration=original,
            output=REMOTE_SERVER+'/matrices_repair_20261116', cuda=1,
            original_stderr_sha256=proof['original_stderr_sha256'], physics=proof['physics'],
            change='C uses Scatter_SysMat (combined disabled), as original production conversion; no physics change',
            reuse=['A440 complete','A218 complete','C440to218 slab_00 raw'],
            new_scientific_work='Only missing C slabs 02..28',
            local_controller_deadline_utc=oldlocal['deadline_utc'])
        files={p.name:sha(p) for p in sorted(HERE.glob('*.py'))}
        key=hashlib.sha256(json.dumps(dict(files=files,plan=plan),sort_keys=True).encode()).hexdigest()[:16]
        write(MANIFEST,dict(key=key,files=files,plan=plan,time=now()))
    m=read(MANIFEST)
    for name,h in m['files'].items():
        if sha(HERE/name)!=h:raise ValueError('Repair source changed')
    test=subprocess.run([sys.executable,'-B','-X','utf8',str(HERE/'test_repair.py')],capture_output=True,text=True,encoding='utf8',timeout=60)
    (REPORT/'repair_local_tests_20261116.txt').write_text(test.stdout+test.stderr,encoding='utf8')
    if test.returncode:raise RuntimeError('Local recovery regression failed')
    remote=REMOTE_SERVER+'/repairs/'+m['key']
    # Immutable uploads, each checked before execution. No old files changed.
    mapping={name:(HERE/name).read_text(encoding='utf8') for name in m['files']}
    mapping['repair_release_20261116.json']=MANIFEST.read_text(encoding='utf8')
    code='''from pathlib import Path
import json,hashlib,subprocess,sys
r=Path(ROOT);r.mkdir(parents=True,exist_ok=True)
for n,s in FILES.items():
 p=r/n;raw=s.encode('utf8')
 if p.exists():assert p.read_bytes()==raw,n
 else:
  with p.open('xb') as f:f.write(raw)
m=json.loads((r/'repair_release_20261116.json').read_text())
for n,h in m['files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
t=subprocess.run([sys.executable,'-B',str(r/'test_repair.py')],capture_output=True,text=True)
print(json.dumps(dict(returncode=t.returncode,stdout=t.stdout,stderr=t.stderr)))
'''.replace('ROOT',repr(remote)).replace('FILES',repr(mapping))
    tests=json.loads(server(code,120))
    write(REPORT/'repair_linux_tests_20261116.json',tests)
    if tests['returncode']:raise RuntimeError('Linux recovery regression failed')
    code='''from pathlib import Path
import psutil,json,subprocess,sys,time,os
r=Path(ROOT);reg=r/'registration.json';intent=r/'intent.json'
if reg.exists():print(reg.read_text());raise SystemExit(0)
if intent.exists():raise RuntimeError('Ambiguous recovery intent; no blind retry')
m=json.loads((r/'repair_release_20261116.json').read_text());old=m['plan']['original_registration']
for q in psutil.process_iter(['pid','create_time','cmdline']):
 if q.pid==os.getpid():continue
 a=q.info['cmdline'] or []
 if (q.pid==old['pid'] and abs(q.info['create_time']-old['create_time'])<.1) or any(x in a for x in [old['release']+'/build_matrix.py',old['release']+'/bin/pe',old['release']+'/bin/scatter',str(r/'repair_matrix.py')]):raise RuntimeError('Matrix process still alive')
args=[sys.executable,'-B',str(r/'repair_matrix.py'),'--plan',str(r/'repair_release_20261116.json')]
with intent.open('x') as f:json.dump(dict(time=time.time(),argv=args),f)
with (r/'matrix.stdout.log').open('xb') as out,(r/'matrix.stderr.log').open('xb') as err:
 p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,cwd=r)
q=psutil.Process(p.pid)
d=dict(pid=q.pid,create_time=q.create_time(),cmdline=q.cmdline(),output=m['plan']['output'],release=old['release'],repair_directory=str(r),execution_repair=m['key'],gpu_index=m['plan']['cuda'],host=HOST)
with reg.open('x') as f:json.dump(d,f,indent=2)
print(json.dumps(d))
'''.replace('ROOT',repr(remote)).replace('HOST',repr(SERVER))
    record=json.loads(server(code))
    write(REG,record)
    paths=list(original_binding['files'])+[str(HERE/name) for name in m['files']]+[str(MANIFEST),str(REG)]
    write(BINDING,dict(files={str(Path(p).resolve()):sha(p) for p in paths},time=now(),
        original_controller_preserved=True,deadline_utc=m['plan']['local_controller_deadline_utc']))
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','status','verify']);a=p.parse_args()
    if a.action=='start':start()
    else:print(json.dumps(status() if a.action=='status' else verify(),indent=2))
