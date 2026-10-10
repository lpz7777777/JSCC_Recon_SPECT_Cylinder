import json,os,socket,subprocess,time
from pathlib import Path
m=json.loads(subprocess.check_output(['findmnt','-J','-o','TARGET,FSTYPE,SOURCE'],text=True))
found=[]
def visit(items):
 for x in items:
  if x['fstype'] in ('ext4','xfs','btrfs'):
   p=Path(x['target']);s=os.statvfs(p);found.append(dict(target=str(p),fstype=x['fstype'],source=x['source'],total_bytes=s.f_blocks*s.f_frsize,free_bytes=s.f_bavail*s.f_frsize,writable=os.access(p,os.W_OK)))
  visit(x.get('children',[]))
visit(m['filesystems'])
candidates=[]
for raw in ['/tmp','/scratch','/local','/local_scratch','/nvme','/mnt/nvme','/ssd',os.environ.get('SLURM_TMPDIR','')]:
 if raw and Path(raw).exists():
  p=Path(raw).resolve();s=os.statvfs(p);fs=subprocess.check_output(['findmnt','-T',str(p),'-n','-o','FSTYPE'],text=True).strip();candidates.append(dict(path=str(p),fstype=fs,free_bytes=s.f_bavail*s.f_frsize,writable=os.access(p,os.W_OK)))
print(json.dumps(dict(node=socket.gethostname(),epoch=time.time(),local_mounts=found,candidates=candidates,read_only=True)),flush=True)
