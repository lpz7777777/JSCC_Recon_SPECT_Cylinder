"""Bounded storage bootstrap release; preserve existing acquisition and scientific code."""
import hashlib,json,shutil,tarfile
from ehe_common import read,write,digest,hashes,verify_files,HERE
from ehe_5e10_workflow import DATA,REPORT,BASE,registered,pid_alive


def prepare():
    path=REPORT/'input_storage_freeze.json'
    if path.exists():
        f=read(path);verify_files(DATA/f['payload_dir'],f['sha256']);return f
    for n in ('controller.json','postprocess_registration.json'):
        p=DATA/n
        if p.exists() and read(p)['status']=='running' and pid_alive(read(p)['pid']):
            raise RuntimeError('Owned helper is alive; no parallel storage repair')
    if any(registered(s) for s in ('validation','validation_acceptance','formal','formal_acceptance')):
        raise ValueError('Cannot replace a started scientific release')
    old=read(REPORT/'freeze.json');source=DATA/old['payload_dir'];verify_files(source,old['sha256'])
    bundle=DATA/'transport_counts.tar.gz';receipt=read(DATA/'transport_counts.json')
    accepted=read(REPORT/'transport_acceptance.json');stop=read(REPORT/'input_storage_stop_acceptance.json')
    if not accepted['passed'] or not stop['fully_exited'] or digest(bundle)!=receipt['archive_sha256']:
        raise ValueError('Completed transport and fully exited old I/O process required')
    if stop['archive_sha256']!=receipt['archive_sha256']:raise ValueError('Uploaded archive differs')
    with tarfile.open(bundle,'r:gz') as t:
        members=t.getmembers();raw=sum(m.size for m in members)
        # Conservative filesystem blocks plus every possible parent directory.
        disk=sum(((m.size+4095)//4096)*4096 for m in members)+sum(len(m.name.split('/'))*4096 for m in members)
    payload=DATA/'archive_input_payload';shutil.copytree(source,payload)
    (payload/'release_manifest.json').unlink()
    config=read(payload/'config.json');config['archive_storage']=dict(
        archive_path=BASE+'/transport_counts.tar.gz',archive_bytes=bundle.stat().st_size,
        archive_sha256=receipt['archive_sha256'],collection_sha256=accepted['collection_sha256'],
        uncompressed_bytes=raw,disk_allocation_bytes=disk,archive_members=len(members),
        method='Exact archive to actual Slurm node-local disk; original complete-worker verifier and science unchanged')
    write(payload/'config.json',config)
    shutil.copy2(HERE/'ehe_5e10_archive_input.py',payload/'ehe_5e10_archive_input.py')
    unchanged={n:s for n,s in old['sha256'].items() if n!='config.json'}
    verify_files(payload,unchanged)
    files=hashes(payload);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    f={**old,'release_key':key,'payload_dir':payload.name,'root':BASE+'/releases/'+key,'sha256':files,
       'storage_predecessor_release_key':old['release_key'],'input_storage':'node_local_archive'}
    write(payload/'release_manifest.json',f);write(path,f)
    shutil.copy2(REPORT/'freeze.json',REPORT/'freeze_before_archive_storage.json')
    if (REPORT/'deployment_gpu.json').exists():
        shutil.copy2(REPORT/'deployment_gpu.json',REPORT/'deployment_gpu_before_archive_storage.json')
    write(REPORT/'freeze.json',f)
    write(REPORT/'input_storage_local_acceptance.json',dict(passed=True,release_key=key,
        original_scientific_files_sha256=unchanged,original_config_except_storage_unchanged=True,
        archive_storage=config['archive_storage'],stop_acceptance_sha256=digest(REPORT/'input_storage_stop_acceptance.json'),
        no_new_simulation=True,no_matrix_or_algorithm_change=True,no_gpu_stage_submitted=True))
    return f


if __name__=='__main__':
    from ehe_5e10_workflow import connection,deploy_gpu
    f=prepare()
    with connection('gpu') as c:deploy_gpu(c,f)
