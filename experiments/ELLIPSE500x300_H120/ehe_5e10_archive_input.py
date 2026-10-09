"""Stage the accepted acquisition on allocation-local disk; run frozen code unchanged."""
import argparse, os, runpy, shutil, socket, sys, tempfile, time
from pathlib import Path, PurePosixPath
from ehe_common import read, write, digest, verify_files
from ehe_5e10_transport import extract


def mount_identity(path, mount_text):
    path=PurePosixPath(path);matches=[]
    for line in mount_text.splitlines():
        fields=line.split()
        if len(fields)<3:continue
        mount=PurePosixPath(fields[1].replace('\\040',' '))
        if path==mount or mount in path.parents:
            matches.append((len(mount.parts),fields[0],str(mount),fields[2]))
    if not matches:raise ValueError('Scratch mount identity unavailable')
    _,device,mount,kind=max(matches)
    if kind not in ('ext4','xfs','btrfs') or not device.startswith('/dev/'):
        raise ValueError('Scratch must use node-local disk, not shared or RAM filesystem: '+kind)
    return dict(device=device,mount=mount,filesystem=kind)


def choose_scratch(required_bytes):
    mounts=Path('/proc/mounts').read_text();errors=[];seen=set()
    candidates=[os.environ.get('SLURM_TMPDIR'),'/tmp','/var/tmp','/scratch','/localscratch']
    for name in candidates:
        if not name:continue
        path=Path(name).resolve()
        if str(path) in seen:continue
        seen.add(str(path))
        if not path.is_dir() or not os.access(path,os.W_OK):continue
        try:
            identity=mount_identity(path,mounts);free=shutil.disk_usage(path).free
            if free<required_bytes:raise ValueError(f'{free} free bytes < {required_bytes} required')
            return path,dict(**identity,path=str(path),free_bytes=free,required_bytes=required_bytes)
        except ValueError as exc:errors.append(str(path)+': '+str(exc))
    raise RuntimeError('No sufficiently large node-local disk for exact full input: '+'; '.join(errors))


def launch(a):
    job=os.environ.get('SLURM_JOB_ID')
    if not job:raise RuntimeError('Archive bootstrap must execute inside its actual Slurm allocation')
    release=a.release.resolve();f=read(release/'release_manifest.json');verify_files(release,f['sha256'])
    storage=read(release/'config.json')['archive_storage'];program=a.program.resolve()
    if program.name not in ('run_ehe_5e10_reconstruction.py','verify_ehe_5e10.py'):
        raise ValueError('Only the two original scientific entry points are allowed')
    if digest(program)!=f['sha256'][program.name] or digest(Path(__file__))!=f['sha256'][Path(__file__).name]:
        raise ValueError('Original entry point or bootstrap execution identity differs')
    if program.parent!=release:
        v=read(program.parent/'verification_manifest.json');verify_files(program.parent,v['files'])
        if v['files'][program.name]!=f['sha256'][program.name]:raise ValueError('Independent verifier differs')
    args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    if any(x=='--counts' or x.startswith('--counts=') for x in args):
        raise ValueError('Shared/alternate counts override is forbidden')
    if '--release' not in args or Path(args[args.index('--release')+1]).resolve()!=release:
        raise ValueError('Original target must use the registered release')
    required=storage['disk_allocation_bytes']+storage['archive_bytes']+(128<<20)
    scratch,identity=choose_scratch(required);started=time.monotonic()
    record=dict(passed=False,status='staging',job=job,hostname=socket.gethostname(),
        release_key=f['release_key'],program=program.name,program_sha256=digest(program),
        bootstrap_sha256=digest(Path(__file__)),archive_sha256=storage['archive_sha256'],
        collection_sha256=storage['collection_sha256'],scratch=identity,
        full_worker_verification='Unchanged target verify_counts, not yet completed',
        scientific_code_unchanged=True,shared_partial_counts_reused=False)
    write(a.receipt,record)
    try:
        with tempfile.TemporaryDirectory(prefix='ehe5e10_'+job+'_',dir=scratch) as tmp:
            local=Path(tmp);bundle=local/'transport_counts.tar.gz'
            shutil.copyfile(storage['archive_path'],bundle)
            if bundle.stat().st_size!=storage['archive_bytes'] or digest(bundle)!=storage['archive_sha256']:
                raise ValueError('Exact accepted archive transfer SHA differs')
            counts=local/'counts';extract(bundle,counts)
            if digest(counts/'collection.json')!=storage['collection_sha256']:
                raise ValueError('Exact accepted acquisition collection SHA differs')
            record.update(status='input_ready',stage_seconds=time.monotonic()-started,
                temporary_counts=str(counts));write(a.receipt,record)
            print('EHE_5E10_NODE_LOCAL_INPUT_READY',record['stage_seconds'],identity,flush=True)
            sys.path.insert(0,str(program.parent))
            sys.argv=[str(program),*args,'--counts',str(counts)]
            runpy.run_path(str(program),run_name='__main__')
            record.update(passed=True,status='complete',elapsed_seconds=time.monotonic()-started,
                full_worker_verification='Original target completed successfully, including all1000 worker verification')
            write(a.receipt,record)
            print('EHE_5E10_ARCHIVE_BOOTSTRAP_COMPLETE',program.name,flush=True)
    except BaseException as exc:
        record.update(status='failed',error=str(exc),elapsed_seconds=time.monotonic()-started)
        write(a.receipt,record);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True)
    p.add_argument('--program',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('arguments',nargs=argparse.REMAINDER);launch(p.parse_args())
