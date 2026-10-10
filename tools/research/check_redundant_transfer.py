"""Bind a redundant remote transfer archive to a retained local copy and members."""
from pathlib import Path
import hashlib, json, shlex, sys, time
ROOT=Path(__file__).resolve().parents[2]
H=ROOT/'experiments/ELLIPSE500x300_H120'
sys.path.insert(0,str(H))
from ehe_5e9_workflow import connection,command,q
from ehe_common import digest
OUT=H/'reports/dual_energy_review_20261010'
REMOTE='/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/generated'
archive=H/'generated/factors_transfer.tar.zst'
manifest=H/'generated/factors_transfer_files.json'
print('Checking retained local archive SHA',flush=True)
sha=digest(archive)
assert sha==(H/'generated/factors_transfer.tar.zst.sha256').read_text().strip()
source=json.loads(manifest.read_text())
code='''import pathlib,json,hashlib,time,os
root=pathlib.Path(%r)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
a=root/'factors_transfer.tar.zst';m=root/'factors_transfer_files.json'
records=json.loads(m.read_text());checked={};began=time.monotonic()
for name,r in records.items():
 p=root/name
 assert p.is_file() and not p.is_symlink() and p.stat().st_size==r['bytes'],name
 value=sha(p);assert value==r['sha256'],name
 checked[name]=value
print(json.dumps(dict(archive_bytes=a.stat().st_size,archive_sha256=sha(a),manifest_sha256=sha(m),members=checked,elapsed_seconds=time.monotonic()-began)),flush=True)
'''%REMOTE
print('Checking remote archive and complete extracted Factors/Sensitivity',flush=True)
with connection('gpu') as c:
    data=json.loads(command(c,'/data/home/scxi717/.conda/envs/torch/bin/python -c '+q(code),1200))
    assert data['archive_sha256']==sha and data['manifest_sha256']==digest(manifest)
    assert data['members']=={n:r['sha256'] for n,r in source.items()}
    extra=command(c,'timeout 60s du -x -B1 --max-depth=2 '+q(REMOTE+'/ehe_spect_5e9_200'),75)
proof={'passed':True,'retained_local_archive':str(archive),'local_archive_bytes':archive.stat().st_size,'local_archive_sha256':sha,
       'remote_archive':REMOTE+'/factors_transfer.tar.zst','remote_verified':data,
       'remote_complete_members_match_original_manifest':True,'remote_archive_deletion_safe_as_duplicate':True,
       'ehe_response_inventory':extra,'read_only':True}
(OUT/'redundant_transfer_acceptance.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps({'passed':True,'archive_bytes':data['archive_bytes'],'members':len(data['members']),'EHE_storage':extra}),flush=True)
