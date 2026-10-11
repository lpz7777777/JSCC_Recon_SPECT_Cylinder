"""Execute the user's exact /data/run01/scxi717/zxc deletion; no job changes."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / 'experiments/ELLIPSE500x300_H120'
OUT = H / 'reports/storage_cleanup_zxc_20261011'
REMOTE_REPORT = '/data/run01/scxi717/lpz/' + ROOT.name + '/experiments/ELLIPSE500x300_H120/reports/storage_cleanup_zxc_20261011'
PYTHON = '/data/home/scxi717/.conda/envs/torch/bin/python'
sys.path.insert(0, str(H))
from ehe_5e9_workflow import connection, command, q
from jscc_5e10_5090_8x2_trial import live_registration

REMOTE_CODE = r'''
import datetime,hashlib,json,os,shutil,stat,subprocess,sys,time
from pathlib import Path
target=Path('/data/run01/scxi717/zxc')
audit=Path(sys.argv[2]);mode=sys.argv[1]
project=audit.parent.parent
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run(args):return subprocess.check_output(args,text=True,timeout=120).strip()
def save(name,value):
 p=audit/name
 assert not p.exists(),str(p)
 data=json.dumps(value,ensure_ascii=False,indent=2).encode()
 with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 return hashlib.sha256(data).hexdigest()
def valid_target():
 assert os.getuid()==1721 and target.parent.resolve()==Path('/data/run01/scxi717')
 assert target.resolve()==target and not target.is_symlink()
 s=target.lstat()
 assert stat.S_ISDIR(s.st_mode) and s.st_uid==os.getuid()
 mounts=[]
 for line in Path('/proc/self/mountinfo').read_text().splitlines():
  mount=line.split()[4]
  assert not (mount==str(target) or mount.startswith(str(target)+'/')),('Nested mount',mount)
 return dict(path=str(target),resolved=str(target.resolve()),uid=s.st_uid,gid=s.st_gid,mode=oct(s.st_mode),device=s.st_dev,inode=s.st_ino)
def queue_guard():
 ids=run(['squeue','-h','-u','scxi717','-o','%i']).split()
 assert set(ids).issubset({'1685272'}),('Unexpected live job; inspect references first',ids)
 jobs=[]
 for job in ids:
  raw=run(['scontrol','show','job',job]);fields=dict(t.split('=',1) for t in raw.split() if '=' in t)
  expected=str(project/'generated/jscc_geant4_5e10_10000/launch_control_releases/ce90d678eb32757f/selection.sh')
  assert fields.get('Command')==expected
  script=Path(expected).read_bytes()
  assert b'zxc' not in script and b'SCSPECT_EXP' not in script
  assert str(target) not in raw
  jobs.append(dict(id=job,state=fields.get('JobState'),reason=fields.get('Reason'),command=fields['Command'],workdir=fields.get('WorkDir'),script_sha256=hashlib.sha256(script).hexdigest()))
 if not ids:return dict(ids=ids,accounting=run(['sacct','-j','1685272','-n','-P','--format=JobID,State,ExitCode']))
 return dict(ids=ids,jobs=jobs)
def protected():
 h=hashlib.sha256();n=0;total=0
 for base in (project/'generated/jscc_geant4_5e10_10000',project/'generated/FactorsCalibrated'):
  for folder,dirs,files in os.walk(base):
   dirs.sort()
   for name in sorted(files):
    p=Path(folder)/name;s=p.lstat()
    h.update(json.dumps([str(p.relative_to(project)),s.st_size,s.st_mtime_ns,s.st_mode],separators=(',',':')).encode())
    n+=1;total+=s.st_size
 return dict(files=n,bytes=total,metadata_sha256=h.hexdigest())
def disk():
 raw=run(['df','-B1','/data/run01/scxi717']);row=raw.splitlines()[-1].split()
 return dict(raw=raw,total_bytes=int(row[1]),used_bytes=int(row[2]),available_bytes=int(row[3]),use_percent=row[4])
audit.mkdir(parents=True,exist_ok=True)
if mode=='inspect':
 identity=valid_target();queue=queue_guard()
 size=int(run(['du','-x','-s','-B1',str(target)]).split()[0])
 children=[]
 for p in sorted(target.iterdir()):
  s=p.lstat();children.append(dict(name=p.name,mode=oct(s.st_mode),uid=s.st_uid,symlink=p.is_symlink()))
 result=dict(utc=now(),target=identity,allocated_bytes=size,children=children,disk=disk(),queue=queue,protected=protected(),read_only=True)
 digest=save('inspection.json',result)
 print(json.dumps(dict(file='inspection.json',sha256=digest,result=result)),flush=True)
elif mode=='apply':
 assert not (audit/'deletion_intent.json').exists(), 'Prior deletion attempt exists; do not repeat blindly'
 inspection=json.loads((audit/'inspection.json').read_bytes())
 identity=valid_target();assert identity==inspection['target']
 queue=queue_guard();before=protected();assert before==inspection['protected']
 disk_before=disk()
 intent=dict(utc=now(),authorization='User: 删除/data/run01/scxi717/zxc这个文件夹',target=identity,inspection_sha256=hashlib.sha256((audit/'inspection.json').read_bytes()).hexdigest(),queue=queue,disk_before=disk_before,protected_before=before,recursive=True,scope='Only the exact authorized directory; symlink destinations are not followed')
 save('deletion_intent.json',intent)
 print('Deleting authorized directory '+str(target),flush=True)
 began=time.monotonic();shutil.rmtree(target)
 assert not os.path.lexists(target)
 after=protected();assert before==after
 queue_after=queue_guard()
 disks=[]
 for attempt in range(4):
  value=disk();disks.append(dict(utc=now(),**value))
  if disk_before['available_bytes']<value['available_bytes']:break
  time.sleep(3)
 result=dict(utc=now(),target=str(target),exists=False,elapsed_seconds=time.monotonic()-began,allocated_bytes_before=inspection['allocated_bytes'],disk_before=disk_before,disk_samples_after=disks,available_bytes_increase=disks[-1]['available_bytes']-disk_before['available_bytes'],protected_before=before,protected_after=after,queue_before=queue,queue_after=queue_after,no_job_mutation=True)
 digest=save('deletion_receipt.json',result)
 print(json.dumps(dict(file='deletion_receipt.json',sha256=digest,result=result)),flush=True)
else:raise ValueError(mode)
'''

def main():
    mode = sys.argv[1]
    assert mode in ('inspect', 'apply')
    guards = ('advance', 'selection_4090_trial', 'selection_5090_8x2_trial', 'selection_5090_8x2_monitor_repair', 'selection_5090_8x2_monitor_repair_verification', 'streaming_4090_8x3')
    live = [n for n in guards if live_registration(H / 'generated/jscc_geant4_5e10_10000' / (n + '_registration.json'))]
    assert not live, live
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / ('execution_' + mode + '.json')
    assert not log.exists(), 'Preserve prior execution evidence'
    code_sha = hashlib.sha256(REMOTE_CODE.encode()).hexdigest()
    with connection('gpu') as c:
        result = command(c, PYTHON + ' -c ' + q(REMOTE_CODE) + ' ' + q(mode) + ' ' + q(REMOTE_REPORT), timeout=1800)
        names = ('inspection.json',) if mode == 'inspect' else ('deletion_intent.json', 'deletion_receipt.json')
        downloaded = []
        with c.open_sftp() as sftp:
            for name in names:
                remote = REMOTE_REPORT + '/' + name
                expected = command(c, 'sha256sum ' + q(remote)).split()[0]
                local = OUT / name
                assert not local.exists()
                sftp.get(remote, str(local))
                actual = hashlib.sha256(local.read_bytes()).hexdigest()
                assert actual == expected
                downloaded.append(dict(file=name,sha256=actual,bytes=local.stat().st_size))
    evidence = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),mode=mode,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),remote_code_sha256=code_sha,tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),controller_pids_live=live,result=result,strict_downloads=downloaded)
    log.write_bytes(json.dumps(evidence,ensure_ascii=False,indent=2).encode())
    print(result,flush=True)

if __name__ == '__main__':
    main()
