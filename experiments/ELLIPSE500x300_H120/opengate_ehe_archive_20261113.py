"""Seal completed stages into hash-closed scientific delivery bundles."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
from datetime import datetime,timezone


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.writing')
    with t.open('w') as f:
        json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    t.replace(p)


def accounting(root,stage,count=None):
    job=read(root/'control'/(stage+'_job.json'))['job']
    raw=subprocess.check_output(['sacct','-n','-P','-S','2026-11-13','-j',str(job),
         '--format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES'],text=True,timeout=45)
    rows={v[0]:v for line in raw.splitlines() if '|' in line for v in [line.split('|')]}
    if not rows or any(v[1]!='COMPLETED' or v[2]!='0:0' for v in rows.values()):
        raise ValueError('Stage has not fully exited successfully: '+stage)
    expected=[str(job)] if count is None else [f'{job}_{i}' for i in range(count)]
    for key in expected:
        for suffix in ('','.batch','.extern'):
            if key+suffix not in rows:raise ValueError('Missing accounting component: '+key+suffix)
    return dict(job=job,raw=raw)


def seal(root,kind):
    root=Path(root)
    proof=root/'control'/(('reconstruction' if kind=='reconstruction' else 'transport')+'_delivery_authority.json')
    if proof.exists():return read(proof)
    if kind=='reconstruction':
        final=read(root/'reconstruction_complete.json')
        if not final['passed']:raise ValueError('Reconstruction not complete')
        acct={'reconstruction':accounting(root,'reconstruction')}
        for energy in ('440','218'):
            d=root/'reconstruction'/energy;p=read(d/'complete.json')
            if p['iterations']!=200 or p['checkpoints']!=200 or not p['passed']:
                raise ValueError('Incomplete final reconstruction')
            for n,r in p['files'].items():
                f=d/n
                if f.stat().st_size!=r['bytes'] or sha(f)!=r['sha256']:
                    raise ValueError('Changed final reconstruction member: '+str(f))
        authority=dict(passed=True,actual_primary_gamma_total=5_000_000_000,
                       channels=['440','218'],checkpoints=400,accounting=acct)
    else:
        acct={s:accounting(root,s,n) for s,n in [('imaging',200),('response',88),('collect',None)]}
        complete=read(root/'collected/complete.json')
        if not complete['passed'] or complete['actual_imaging_primaries']!=5_000_000_000:
            raise ValueError('Full actual 5e9 collection required')
        peak=0.;worker_resources={}
        for k,n in [('imaging',200),('response',88)]:
            for i in range(n):
                file=root/'resources'/f'{k}_{i:04d}.json';p=read(file)
                if p['status']!='COMPLETED' or p['rss_fraction']>.8:
                    raise ValueError('Actual worker resource/exit check failed')
                snapshot=root/'resources'/f'{k}_{i:04d}.allocation.txt'
                text=snapshot.read_text()
                if f"ArrayJobId={acct[k]['job']}" not in text or not re.search(r'\bArrayTaskId='+str(i)+r'\b',text):
                    raise ValueError('Worker allocation identity mismatch')
                worker_resources[f'{k}/{i}']=dict(sha256=sha(file),job=p['job'])
                peak=max(peak,p['rss_fraction'])
        authority=dict(passed=True,actual_primary_gamma_total=5_000_000_000,
                       response_primaries=4_400_000_000,workers=288,accounting=acct,
                       resource_records=worker_resources,max_worker_rss_fraction=peak,
                       collected_sha256=sha(root/'collected/complete.json'))
    authority['sealed_utc']=datetime.now(timezone.utc).isoformat()
    authority['physical_calibration_passed']=False
    save(proof,authority)
    return authority


def bundle(root,kind):
    root=Path(root)
    target=root/'bundles';target.mkdir(exist_ok=True)
    receipt=target/(kind+'_bundle.json')
    if receipt.exists():return read(receipt)
    authority=seal(root,kind)
    members={}
    if kind=='bridge':
        for n in ('observations.npz','response_kernels.npz','complete.json'):
            members['collected/'+n]=root/'collected'/n
        members['control/transport_delivery_authority.json']=root/'control/transport_delivery_authority.json'
    else:
        directories=(['collected','workers','resources','preflight','logs','control','scripts','releases']
                     if kind=='transport' else ['collected','reconstruction','logs','control','releases'])
        for name in directories:
            folder=root/name
            if folder.exists():
                for p in sorted(folder.rglob('*')):
                    if p.is_file():
                        if p.name.endswith(('.writing','.part','.uploading')):
                            raise ValueError('Unfinished stage file remains: '+str(p))
                        members[p.relative_to(root).as_posix()]=p
        if kind=='reconstruction':
            for name in ('reconstruction_complete.json','reconstruction_allocation.json'):
                members[name]=root/name
    # Follow only this explicit source/input pointer; no arbitrary symlinks.
    if (root/'inputs').exists():
        for p in sorted((root/'inputs').iterdir()):
            if p.is_file():members['inputs/'+p.name]=p
    else:
        job=read(root/'control'/'reconstruction_job.json')
        inp=root/'releases'/job['release']/'inputs'
        for p in inp.iterdir():
            if p.is_file():members['inputs/'+p.name]=p
    records={n:dict(bytes=p.stat().st_size,sha256=sha(p)) for n,p in members.items()}
    manifest=target/(kind+'_manifest.json')
    save(manifest,dict(kind=kind,files=records,authority=authority,format_version=1))
    tarpath=target/(kind+'.tar')
    temp=tarpath.with_suffix('.tar.writing')
    with tarfile.open(temp,'w',dereference=True) as t:
        for name,p in members.items():t.add(p,arcname=name,recursive=False)
        t.add(manifest,arcname='_manifest.json',recursive=False)
    temp.replace(tarpath)
    result=dict(passed=True,kind=kind,path=str(tarpath),bytes=tarpath.stat().st_size,
                sha256=sha(tarpath),manifest_sha256=sha(manifest),members=len(records))
    save(receipt,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--kind',choices=['bridge','transport','reconstruction'],required=True)
    a=p.parse_args();print('CAMPAIGN_JSON '+json.dumps(bundle(a.root,a.kind)))
