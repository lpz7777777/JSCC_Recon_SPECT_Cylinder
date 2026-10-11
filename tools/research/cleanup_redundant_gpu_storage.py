"""Remove only SHA-identical local-backed remote copies from the Oct 11 plan.

No simulations, job changes, recursive deletion, or immutable-release edits.
The fixed scope is legacy root Factors/List and selected transfer archives.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / 'experiments/ELLIPSE500x300_H120'
OUT = H / 'reports/storage_cleanup_20261011'
REMOTE = '/data/run01/scxi717/lpz/' + ROOT.name
REMOTE_REPORT = REMOTE + '/experiments/ELLIPSE500x300_H120/reports/storage_cleanup_20261011'
PYTHON = '/data/home/scxi717/.conda/envs/torch/bin/python'
sys.path.insert(0, str(H))
from ehe_5e9_workflow import connection, command, q
from jscc_5e10_5090_8x2_trial import live_registration

GUARDS = ('advance', 'selection_4090_trial', 'selection_5090_8x2_trial',
          'selection_5090_8x2_monitor_repair',
          'selection_5090_8x2_monitor_repair_verification', 'streaming_4090_8x3')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def controller_guard():
    busy = [n for n in GUARDS if live_registration(H / 'generated/jscc_geant4_5e10_10000' / (n + '_registration.json'))]
    if busy:
        raise RuntimeError('Active controller: ' + ','.join(busy))


def prepare():
    controller_guard()
    if (OUT / 'candidate_plan.json').exists():
        raise FileExistsError('Plan already registered; preserve it')
    inventory = read('file_inventory.json')
    candidates = []
    excluded = []
    for row in inventory['legacy']:
        p = ROOT / row['path']
        if p.is_file() and not p.is_symlink() and p.stat().st_size == row['bytes']:
            candidates.append(dict(row, backup_path=row['path'], reason='Legacy remote copy; canonical local file retained'))
        else:
            excluded.append(dict(row, reason='Local backup absent or different size; remote retained'))
    aliases = {
        'experiments/FOV120_20260924/experiments/FOV120/generated/' + n:
        'experiments/FOV120/generated/' + n
        for n in ('Factors_production.tar', 'qc_1e10_collected.tar.gz', 'qc_1e9_collected.tar.gz')
    }
    # Only standalone transfer archives, outside frozen releases and current inputs.
    allowed = {
        'experiments/ELLIPSE500x300_H120/generated/' + n
        for n in ('imaging_1e9.tar.gz', 'nema_h60_imaging_1e9.tar.gz',
                  'nema_h60_imaging_5e9.tar.gz', 'ehe_spect_5e10_200/transport_counts.tar.gz',
                  'compton_first_scatter_v2/imaging_payload.tar.gz')
    }
    for row in inventory['archives']:
        path = row['path']
        if path not in allowed and path not in aliases:
            continue
        backup = aliases.get(path, path)
        p = ROOT / backup
        if p.is_file() and not p.is_symlink() and p.stat().st_size == row['bytes']:
            candidates.append(dict(row, backup_path=backup, reason='Redundant transfer archive; local archive retained'))

    def local_hash(row):
        p = ROOT / row['backup_path']
        assert p.resolve().is_relative_to(ROOT.resolve()) and not p.is_symlink()
        s = p.stat()
        digest = sha(p)
        assert p.stat().st_mtime_ns == s.st_mtime_ns
        return dict(row, local_sha256=digest, local_mtime_ns=s.st_mtime_ns)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        candidates = list(pool.map(local_hash, candidates))
    plan = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                remote_root=REMOTE, local_root=str(ROOT), candidates=candidates,
                candidate_bytes=sum(r['bytes'] for r in candidates), excluded_legacy=excluded,
                protected='Current 5e10 inputs, calibrated matrices, results, releases, evidence, installed environments, other projects',
                original_5090_registry_sha256=sha(H / 'reports/NEMA_Body_H60/jscc_geant4_5e10_10000/selection_job.json'),
                original_5090_untouched=True, deletion_started=False)
    save('candidate_plan.json', plan)
    print(json.dumps(dict(files=len(candidates), bytes=plan['candidate_bytes'], excluded=len(excluded))), flush=True)


REMOTE_CODE = r'''
import concurrent.futures,datetime,hashlib,json,os,subprocess
from pathlib import Path
root=Path(ROOT).resolve();audit=Path(AUDIT)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
assert sha(audit/'candidate_plan.json')==PLAN_SHA
plan=json.loads((audit/'candidate_plan.json').read_text())
assert plan['remote_root']==str(root)
def queue_guard():
 ids=subprocess.check_output(['squeue','-h','-u','scxi717','-o','%i'],text=True).split()
 assert set(ids).issubset({'1685272'}),('New jobs appeared',ids)
 if ids:
  job=subprocess.check_output(['scontrol','show','job','1685272'],text=True)
  expected=str(root/'experiments/ELLIPSE500x300_H120/generated/jscc_geant4_5e10_10000/launch_control_releases/ce90d678eb32757f/selection.sh')
  fields=dict(token.split('=',1) for token in job.split() if '=' in token)
  assert fields['Command']==expected
 else:job=subprocess.check_output(['sacct','-j','1685272','-n','-P','--format=JobID,State,ExitCode'],text=True)
 return dict(ids=ids,original_job=job)
def target(row):
 rel=row['path'];p=root/rel
 assert not p.is_symlink() and p.resolve()==p and p.resolve().is_relative_to(root)
 assert rel.startswith(('Factors/','List/')) or rel in ARCHIVES
 assert p.is_file() and p.stat().st_size==row['bytes'] and p.stat().st_mtime_ns==row['mtime_ns']
 return p
def protected_stat():
 bases=[root/'experiments/ELLIPSE500x300_H120/generated/jscc_geant4_5e10_10000',root/'experiments/ELLIPSE500x300_H120/generated/FactorsCalibrated']
 h=hashlib.sha256();n=0;b=0
 for base in bases:
  for folder,ds,fs in os.walk(base):
   ds.sort()
   for name in sorted(fs):
    p=Path(folder)/name;s=p.lstat()
    h.update(json.dumps([str(p.relative_to(root)),s.st_size,s.st_mtime_ns,s.st_mode],separators=(',',':')).encode());n+=1;b+=s.st_size
 return dict(files=n,bytes=b,metadata_sha256=h.hexdigest())
queue=queue_guard()
if MODE=='prove':
 def verify(row):
  p=target(row);s=p.stat();digest=sha(p)
  assert p.stat().st_mtime_ns==s.st_mtime_ns
  return dict(path=row['path'],bytes=s.st_size,allocated_bytes=s.st_blocks*512,sha256=digest,identical=digest==row['local_sha256'])
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(verify,plan['candidates']))
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=PLAN_SHA,rows=rows,queue=queue,protected_before=protected_stat(),read_only=True)
 (audit/'remote_proof.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(dict(identical_files=sum(r['identical'] for r in rows),identical_bytes=sum(r['bytes'] for r in rows if r['identical']),different_files=sum(not r['identical'] for r in rows))))
elif MODE=='apply':
 receipt=audit/'cleanup_acceptance.json'
 assert not receipt.exists(),'Cleanup already registered'
 assert sha(audit/'remote_proof.json')==PROOF_SHA
 proof=json.loads((audit/'remote_proof.json').read_text());assert proof['plan_sha256']==PLAN_SHA
 lookup={r['path']:r for r in proof['rows']}
 selected=[r for r in plan['candidates'] if lookup[r['path']]['identical']]
 for row in selected:target(row)
 # Rehash immediately before each unlink; log every completed deletion durably.
 before=subprocess.check_output(['df','-B1',str(root)],text=True)
 protected=protected_stat();assert protected==proof['protected_before']
 journal=audit/'deletion_journal.jsonl';assert not journal.exists()
 deleted=[]
 with journal.open('x') as f:
  for i,row in enumerate(selected):
   if i%50==0:queue_guard()
   p=target(row);s=p.stat();assert sha(p)==row['local_sha256'];assert p.stat().st_mtime_ns==s.st_mtime_ns
   p.unlink();assert not p.exists()
   record=dict(path=row['path'],bytes=s.st_size,allocated_bytes=s.st_blocks*512,sha256=row['local_sha256'],backup_path=row['backup_path'])
   deleted.append(record);f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
 after=subprocess.check_output(['df','-B1',str(root)],text=True)
 final=protected_stat();assert final==protected
 result=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=PLAN_SHA,proof_sha256=PROOF_SHA,deleted=deleted,deleted_bytes=sum(r['bytes'] for r in deleted),deleted_allocated_bytes=sum(r['allocated_bytes'] for r in deleted),before_df=before,after_df=after,protected_before=protected,protected_after=final,queue_before=queue,queue_after=queue_guard(),local_backups_retained=True,job_changes=False,recursive_deletion=False,unique_scientific_files_deleted=False)
 receipt.write_text(json.dumps(result,indent=2))
 print(json.dumps(dict(passed=True,files=len(deleted),bytes=result['deleted_bytes'],after_df=after)))
'''


def remote(mode):
    controller_guard()
    plan = read('candidate_plan.json')
    for row in plan['candidates']:
        p = ROOT / row['backup_path']
        assert p.is_file() and not p.is_symlink()
        assert p.stat().st_size == row['bytes'] and p.stat().st_mtime_ns == row['local_mtime_ns']
        # Byte check on apply protects the retained backup at the deletion boundary.
        if mode == 'apply':
            assert sha(p) == row['local_sha256']
    if mode == 'apply':
        print('Retained local backup SHA recheck completed', flush=True)
    assert sha(H / 'reports/NEMA_Body_H60/jscc_geant4_5e10_10000/selection_job.json') == plan['original_5090_registry_sha256']
    archives = [r['path'] for r in plan['candidates'] if not r['path'].startswith(('Factors/', 'List/'))]
    preamble = '\n'.join(f'{name}={value!r}' for name, value in dict(
        ROOT=REMOTE, AUDIT=REMOTE_REPORT, PLAN_SHA=sha(OUT/'candidate_plan.json'),
        PROOF_SHA=sha(OUT/'remote_proof.json') if mode == 'apply' else None,
        MODE=mode, ARCHIVES=archives).items())
    code = preamble + '\n' + REMOTE_CODE
    save('execution_' + mode + '.json', dict(mode=mode,source_sha256=sha(__file__),code_sha256=hashlib.sha256(code.encode()).hexdigest(),plan_sha256=sha(OUT/'candidate_plan.json')))
    with connection('gpu') as c:
        command(c, 'mkdir -p ' + q(REMOTE_REPORT), 30)
        with c.open_sftp() as s:
            if mode == 'prove':
                s.put(str(OUT/'candidate_plan.json'), REMOTE_REPORT+'/candidate_plan.json')
            result = command(c, q(PYTHON) + ' -u -c ' + q(code), timeout=2400)
            name = 'remote_proof.json' if mode == 'prove' else 'cleanup_acceptance.json'
            s.get(REMOTE_REPORT+'/'+name, str(OUT/name))
            if mode == 'apply':
                s.get(REMOTE_REPORT+'/deletion_journal.jsonl', str(OUT/'deletion_journal.jsonl'))
            names = [name] + (['deletion_journal.jsonl'] if mode == 'apply' else [])
            check = "import pathlib,hashlib,json;root=pathlib.Path(%r);print(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in %r}))" % (REMOTE_REPORT, names)
            hashes = json.loads(command(c, q(PYTHON)+' -c '+q(check), 30))
            assert all(sha(OUT/n) == hashes[n] for n in names), 'Downloaded audit SHA mismatch'
            save('fetch_' + mode + '.json', dict(passed=True,files=hashes))
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'prove', 'apply'))
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else remote(args.mode)
