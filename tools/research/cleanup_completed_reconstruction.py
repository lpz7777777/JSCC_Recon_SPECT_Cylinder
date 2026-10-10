"""Evidence-first cleanup of cache and completed one-time orchestration only."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, tarfile, time
ROOT=Path(__file__).resolve().parents[2]; H=ROOT/'experiments/ELLIPSE500x300_H120'
OUT=H/'reports/dual_energy_review_20261010'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(name,d):(OUT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')

def local():
 receipt=OUT/'local_cleanup_acceptance.json'
 if receipt.exists():raise FileExistsError('Cleanup already registered; do not repeat')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 names=['ehe_5e10_finalize.py','ehe_forward_poisson_5e10_finalize.py','render_ehe_5e10_reviewed.py','review_ehe_forward_poisson_5e10_layout.py']
 candidates=[H/n for n in names]+list((ROOT/'tools/runners/historical/2026-07-compton-validation').glob('*.ps1'))
 tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
 allcodes=[]
 for folder,dirs,files in os.walk(ROOT):
  dirs[:]=[d for d in dirs if d not in ('.git','generated','reports','__pycache__')]
  for n in files:
   if Path(n).suffix in ('.py','.sh','.ps1'):allcodes.append(Path(folder)/n)
 removedset=set(candidates);proofs=[]
 for p in candidates:
  assert p.resolve().is_relative_to(ROOT.resolve()) and not p.is_symlink()
  callers=[]
  for other in allcodes:
   if other in removedset:continue
   txt=other.read_text(encoding='utf-8',errors='replace')
   if p.name in txt or (p.suffix=='.py' and ('import '+p.stem in txt or 'from '+p.stem in txt)):callers.append(other.relative_to(ROOT).as_posix())
  # This cleanup program names its targets for audit; it is not a production caller.
  callers=[x for x in callers if x!='tools/research/cleanup_completed_reconstruction.py']
  assert not callers,(p,callers)
  rel=p.relative_to(ROOT).as_posix();assert rel in tracked
  raw=subprocess.check_output(['git','show',commit+':'+rel],cwd=ROOT)
  assert p.read_bytes()==raw,(rel,'uncommitted byte difference')
  proofs.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p),remaining_callers=callers,restore_commit=commit,reason='Completed one-time postprocessing/layout or obsolete July controller; scientific producer, solver, plots and verifiers retained'))
 temporary=list((H/'generated').glob('*.py'))
 # These generated-root snippets are historical progress/docs/accounting helpers.
 temp_records=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in temporary]
 archive=H/'generated/historical_process_scripts_20261010.tar.gz'
 if archive.exists():raise FileExistsError(archive)
 with tarfile.open(archive,'w:gz') as tar:
  for p in temporary:tar.add(p,arcname=p.name,recursive=False)
 with tarfile.open(archive,'r:gz') as tar:
  for p in temporary:
   f=tar.extractfile(p.name);assert f is not None and hashlib.sha256(f.read()).hexdigest()==sha(p)
 plan=dict(tracked_scripts=proofs,generated_scripts=temp_records,retained_historical_code_archive=str(archive),archive_sha256=sha(archive),deletion_started=False)
 save('local_cleanup_plan.json',plan)
 # Verify local process registrations before touching generated caches/snippets.
 cmd='Get-CimInstance Win32_Process -Filter "Name=\'python.exe\'" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress'
 processes=subprocess.check_output(['powershell','-NoProfile','-Command',cmd],text=True)
 if processes.strip():
  records=json.loads(processes);records=records if isinstance(records,list) else [records]
  busy=[r for r in records if r.get('ProcessId')!=os.getpid() and any(x in (r.get('CommandLine') or '') for x in ('ehe_5e10_finalize','ehe_forward_poisson_5e10_finalize','ehe_5e10_workflow.py','ehe_forward_poisson_5e10_workflow.py','ehe_5e9_workflow.py advance'))]
  assert not busy,'Live production controller: cleanup forbidden'
 for p in candidates+temporary:
  assert p.resolve().is_relative_to(ROOT.resolve()) and not p.is_symlink();p.unlink()
 caches=[]
 for folder,dirs,files in os.walk(ROOT):
  dirs[:]=[d for d in dirs if d!='.git']
  for n in files:
   p=Path(folder)/n;rel=p.relative_to(ROOT).as_posix()
   if ('__pycache__' in p.parts or n.endswith('.pyc')) and rel not in tracked and not p.is_symlink():
    assert p.resolve().is_relative_to(ROOT.resolve());caches.append(dict(path=rel,bytes=p.stat().st_size));p.unlink()
 # Remove only empty cache directories. No recursive data deletion occurs locally.
 for folder,dirs,files in os.walk(ROOT,topdown=False):
  p=Path(folder)
  if p.name=='__pycache__' and not any(p.iterdir()):p.rmdir()
 save('local_cleanup_acceptance.json',dict(passed=True,tracked_scripts=proofs,generated_scripts=temp_records,cache_files=caches,
  retained_historical_code_archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
  removed_files=len(proofs)+len(temp_records)+len(caches),removed_bytes=sum(x['bytes'] for x in proofs+temp_records+caches),
  accepted_raw_data_or_figures_deleted=False,algorithm_changed=False,production_dependencies_changed=False,source_commit=commit))
 print(json.dumps({'tracked_scripts':len(proofs),'archived_generated_scripts':len(temp_records),'cache_files':len(caches),'raw_data_deleted':False}))

def remote():
 receipt=OUT/'remote_cleanup_acceptance.json'
 if receipt.exists():raise FileExistsError('Remote cleanup already registered')
 sys.path.insert(0,str(H))
 from ehe_5e9_workflow import connection,command,q
 proof=json.loads((OUT/'redundant_transfer_acceptance.json').read_text());assert proof['passed']
 # The retained local archive must still exist and match the audited SHA.
 assert sha(proof['retained_local_archive'])==proof['local_archive_sha256']
 code='''import pathlib,json,hashlib,subprocess,shutil,time
archive=pathlib.Path(%r);expected=%r
cache=pathlib.Path('/data/run01/scxi717/.cache/pip')
account=pathlib.Path('/data/run01/scxi717').resolve()
assert archive.resolve().is_relative_to(account) and archive.name=='factors_transfer.tar.zst' and not archive.is_symlink()
assert cache.resolve()==account/'.cache/pip' and not cache.is_symlink()
queue=subprocess.check_output(['squeue','-h','-u','scxi717'],text=True);assert not queue.strip(),'Account jobs appeared: stop cleanup'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
assert sha(archive)==expected
assert not subprocess.check_output(['squeue','-h','-u','scxi717'],text=True).strip(),'Account jobs appeared during verification: stop cleanup'
before=subprocess.check_output(['df','-B1',str(account)],text=True)
files=[]
if cache.exists():
 for p in cache.rglob('*'):
  assert not p.is_symlink(),str(p)
  if p.is_file():files.append(dict(path=str(p),bytes=p.stat().st_size,allocated_bytes=p.stat().st_blocks*512))
a=dict(path=str(archive),bytes=archive.stat().st_size,allocated_bytes=archive.stat().st_blocks*512,sha256=expected)
archive.unlink()
if cache.exists():shutil.rmtree(cache)
assert not archive.exists() and not cache.exists()
after=subprocess.check_output(['df','-B1',str(account)],text=True)
print(json.dumps(dict(passed=True,duplicate_transfer=a,pip_download_cache=files,before_df=before,after_df=after,queue_was_empty=True,
 scientific_outputs_deleted=False,installed_conda_environments_deleted=False,unique_raw_data_deleted=False)),flush=True)
'''%(proof['remote_archive'],proof['local_archive_sha256'])
 with connection('gpu') as c:d=json.loads(command(c,'/data/home/scxi717/.conda/envs/torch/bin/python -c '+q(code),1200))
 save('remote_cleanup_acceptance.json',d)
 print(json.dumps({'passed':d['passed'],'duplicate_bytes':d['duplicate_transfer']['bytes'],'cache_bytes':sum(r['bytes'] for r in d['pip_download_cache']),'after_df':d['after_df']}),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('local','remote'));args=a.parse_args();(local if args.mode=='local' else remote)()
