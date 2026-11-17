"""Strict resumable delivery; never overwrites accepted scientific bytes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
from opengate_ehe_common_20261113 import *
from opengate_ehe_workflow_20261113 import release,GPU_PYTHON,submit
from campaign_20261102_io import execute,upload_files,python_json
from ehe_5e9_workflow import connection,q


def request_bundle(host,kind):
    root=CPU_ROOT if host=="maty" else GPU_ROOT
    helper=HERE/'opengate_ehe_archive_20261113.py'
    remote=root+'/control/archive_helpers/'+sha(helper)
    upload_files(host,remote,{helper.name:helper})
    output=execute(host,'python3 '+q(remote+'/'+helper.name)+' --root '+q(root)+' --kind '+kind,
                   timeout=1800,readonly=False)
    if 'CAMPAIGN_JSON ' not in output:raise ValueError('Remote bundle receipt missing')
    record=json.loads(output.rsplit('CAMPAIGN_JSON ',1)[1])
    if not record['passed']:raise ValueError('Unsealed remote bundle')
    save(REPORT/(host+'_'+kind+'_bundle.json'),record)
    return record


def download(host,record):
    folder=DATA/'archives';folder.mkdir(exist_ok=True)
    final=folder/(host+'_'+record['kind']+'.tar')
    if final.exists():
        if final.stat().st_size!=record['bytes'] or sha(final)!=record['sha256']:
            raise ValueError('Existing local immutable archive differs')
        return final
    partial=final.with_suffix('.tar.part')
    needed=record['bytes']-(partial.stat().st_size if partial.exists() else 0)
    if needed<0:raise ValueError('Partial archive exceeds remote length')
    if shutil.disk_usage(folder).free<needed/.8+16*(1<<30):
        raise OSError('Insufficient actual local free space with reserve')
    if host=='maty':
        # Reuse the established Maty stop-and-wait protocol. Bulk stdout and
        # SFTP both stall on this VPN; original interrupted prefixes are kept.
        from ac225_transfer_io_20261104 import download_cpu
        if record['bytes']>512*(1<<20):
            raise ValueError('CPU archive exceeds the bounded in-memory compression limit')
        ack=folder/(host+'_'+record['kind']+'_ack_'+str(time.time_ns())+'.tar.part')
        result=download_cpu(record['path'],ack,record['sha256'],maximum_seconds=1800)
        if result['bytes']!=record['bytes']:raise ValueError('Acknowledged transfer size differs')
        ack.replace(final)
        save(REPORT/(host+'_'+record['kind']+'_transfer_protocol.json'),dict(**result,
             retained_bulk_partial=str(partial) if partial.exists() else None))
        return final
    for attempt in range(3):
        start=partial.stat().st_size if partial.exists() else 0
        try:
            with connection(host) as client:
                channel=client.get_transport().open_session(window_size=8<<20,max_packet_size=128<<10)
                channel.settimeout(60)
                channel.exec_command('tail -c +'+str(start+1)+' '+q(record['path']))
                last=time.monotonic()
                with partial.open('ab') as f:
                    while True:
                        b=channel.recv(512<<10)
                        if not b:break
                        f.write(b)
                        if time.monotonic()-last>30:
                            print('STRICT_DELIVERY_BYTES',host,record['kind'],f.tell(),record['bytes'],flush=True)
                            last=time.monotonic()
                    f.flush();os.fsync(f.fileno())
                error=channel.recv_stderr(1<<20)
                code=channel.recv_exit_status();channel.close()
                if code:raise OSError('Remote transfer exit '+str(code)+': '+error.decode('utf8','replace'))
            if partial.stat().st_size!=record['bytes']:
                raise OSError('Archive transfer byte count is incomplete')
            break
        except Exception:
            if attempt==2:raise
            time.sleep(2)
    if sha(partial)!=record['sha256']:
        raise ValueError('Downloaded archive SHA mismatch; preserve the partial for diagnosis')
    partial.replace(final)
    return final


def extract(archive,record,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:') as t:
        m=t.getmember('_manifest.json')
        payload=t.extractfile(m).read()
        if hashlib.sha256(payload).hexdigest()!=record['manifest_sha256']:
            raise ValueError('Bundle manifest SHA mismatch')
        manifest=json.loads(payload)
        expected=manifest['files']
        names=[v.name for v in t.getmembers()]
        if len(names)!=len(set(names)) or set(names)!=set(expected)|{'_manifest.json'}:
            raise ValueError('Archive has missing/extra/duplicate members')
        for name,r in expected.items():
            member=t.getmember(name);target=destination/name
            if not member.isfile() or not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Unsafe bundle member')
            if member.size!=r['bytes']:raise ValueError('Archive member byte size differs')
            if target.exists():
                if target.stat().st_size!=r['bytes'] or sha(target)!=r['sha256']:
                    raise ValueError('Refuse to overwrite accepted scientific member: '+name)
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            temp=target.with_name(target.name+'.writing')
            h=hashlib.sha256()
            with t.extractfile(member) as src,temp.open('wb') as out:
                for b in iter(lambda:src.read(8<<20),b''):
                    h.update(b);out.write(b)
                out.flush();os.fsync(out.fileno())
            if h.hexdigest()!=r['sha256']:raise ValueError('Extracted member SHA differs: '+name)
            temp.replace(target)
    return manifest


def fetch(host,kind):
    receipt=REPORT/(host+'_'+kind+'_strict_fetch.json')
    if receipt.exists():return read(receipt)
    record=request_bundle(host,kind)
    archive=download(host,record)
    destination=DATA/('transport_delivery' if kind=='transport' else 'delivery')
    manifest=extract(archive,record,destination)
    proof=dict(passed=True,strict_fetch_passed=True,all_files_SHA_verified=True,
               host=host,kind=kind,created_utc=now(),archive=record,
               destination=str(destination),files=manifest['files'],authority=manifest['authority'])
    save(receipt,proof)
    return proof


def stage_gpu():
    if not (REPORT/'maty_bridge_strict_fetch.json').exists():
        raise ValueError('Strict actual input bridge required')
    proof=read(REPORT/'maty_bridge_strict_fetch.json')
    if not proof['passed'] or not proof['all_files_SHA_verified']:
        raise ValueError('Unverified input bridge')
    files={n:DATA/'delivery'/n for n in proof['files']}
    for n,r in proof['files'].items():
        if sha(files[n])!=r['sha256']:raise ValueError('Changed local verified bridge')
    upload_files('gpu',GPU_ROOT,files)
    code='''import hashlib,json,pathlib,sys
a=json.loads(sys.argv[1]);root=pathlib.Path(a['root'])
for n,r in a['files'].items():
 p=root/n
 if p.stat().st_size!=r['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError(n)
print('CAMPAIGN_JSON '+json.dumps({'passed':True,'files_verified':len(a['files'])}))'''
    result=python_json('gpu','python3',code,dict(root=GPU_ROOT,files=proof['files']),timeout=90)
    save(REPORT/'gpu_actual_input_verified.json',dict(**result,source_bridge_sha256=sha(REPORT/'maty_bridge_strict_fetch.json'),created_utc=now()))
    return result


def finalize_receipt():
    cpu=read(REPORT/'maty_transport_strict_fetch.json')
    gpu=read(REPORT/'gpu_reconstruction_strict_fetch.json')
    bridge=read(REPORT/'maty_bridge_strict_fetch.json')
    if not all(p['passed'] and p['all_files_SHA_verified'] for p in (cpu,gpu,bridge)):
        raise ValueError('Full raw transport and reconstruction delivery required')
    files=dict(bridge['files'])
    for n,r in gpu['files'].items():
        if n in files and files[n]!=r:raise ValueError('Different shared scientific input')
        files[n]=r
    for n,r in files.items():
        p=DATA/'delivery'/n
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:
            raise ValueError('Final local member differs: '+n)
    proof=dict(passed=True,strict_fetch_passed=True,all_files_SHA_verified=True,
               actual_imaging_primaries=5_000_000_000,checkpoints=400,files=files,
               completed_utc=now(),transport_receipt_sha256=sha(REPORT/'maty_transport_strict_fetch.json'),
               reconstruction_receipt_sha256=sha(REPORT/'gpu_reconstruction_strict_fetch.json'),
               physical_calibration_passed=False,visual_QA_performed=False)
    save(REPORT/'strict_delivery.json',proof)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fetch_bridge','fetch_transport','stage_gpu','fetch_reconstruction','finalize'])
    a=p.parse_args()
    if a.action.startswith('fetch_'):
        kind=a.action[len('fetch_'):];r=fetch('gpu' if kind=='reconstruction' else 'maty',kind)
        print('STRICT_FETCH_COMPLETE',kind,len(r['files']))
    elif a.action=='stage_gpu':print(stage_gpu())
    else:print('FINAL_DELIVERY',len(finalize_receipt()['files']))
