"""Bounded immutable JSCC Geant4 workers and exact full-worker collection."""
import argparse, math, os, re, shutil, subprocess, tarfile, time
from pathlib import Path
import numpy as np
from jscc_5e10_common import *
from ehe_common import bounded_process

def worker(release, simulation, output, index, limit, pilot=False):
    cfg=read(release/'transport_config.json');verify_files(release,read(release/'release_manifest.json')['sha256'])
    r=registry(simulation/'jobs.json');verify_files(simulation,cfg['source_registry_sha256'])
    j=r['jobs'][index];macro=simulation/j['macro']
    body=macro.read_bytes().replace(b'\r\n',b'\n')
    photons,seed=(100000,PILOT_SEED) if pilot else (PER_WORKER,j['seed'])
    if pilot:body=re.sub(rb'/run/beamOn\s+\d+',b'/run/beamOn 100000',body)
    if digest(release/'gamma01')!=BINARY_SHA or digest(release/'CrystalMatrix.txt')!=CRYSTAL_SHA:
        raise ValueError('Accepted executable/detector changed')
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(release/'CrystalMatrix.txt',output/'CrystalMatrix.txt')
    (output/'run.mac').write_bytes(body)
    meminfo=Path('/proc/meminfo').read_text();available=int(re.search(r'MemAvailable:\s+(\d+)',meminfo)[1])*1024
    actual=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID']],text=True)
    alloc=dict(job=os.environ['SLURM_JOB_ID'],scontrol=actual,proc_meminfo_at_start=meminfo,
        host_allocated_bytes=available,denominator_kind='physical MemAvailable CPU-only operational guard',
        imaging_allocation_certificate=False)
    write(output/'allocation.json',alloc)
    env=os.environ.copy();env['JSCC_RANDOM_SEED']=str(seed);started=time.time()
    try:
        usage=bounded_process([str(release/'gamma01'),'run.mac'],output,limit,output/'console.log',alloc,env)
        primary=np.loadtxt(output/'PrimaryCount.csv',delimiter=',',dtype=np.int64,ndmin=2)
        if primary.shape!=(1,3) or primary.sum()!=photons or primary[0,2]!=0:raise ValueError('Actual primary dose differs')
        counts={}
        for e in (218,440):
            x=np.loadtxt(output/f'CntStat_{e}.csv',delimiter=',',dtype=np.int64,ndmin=2)
            if x.shape!=(1,10496) or np.any(x<0):raise ValueError('Full 10496 detector observations required')
            counts[str(e)]=int(x.sum())
        members=('CntStat_218.csv','CntStat_440.csv','PrimaryCount.csv','List.csv','run.mac','CrystalMatrix.txt','allocation.json','console.log')
        write(output/'receipt.json',dict(passed=True,study=STUDY,pilot=pilot,index=index,view=j['view'],worker=j['worker'],
            seed=seed,photons=photons,primary_counts=primary[0].tolist(),counts=counts,resource=usage,
            binary_sha256=BINARY_SHA,crystal_sha256=CRYSTAL_SHA,registered_macro_sha256=digest(macro),
            actual_macro_sha256=digest(output/'run.mac'),allocation_job=os.environ['SLURM_JOB_ID'],
            started_epoch=started,finished_epoch=time.time(),files={n:digest(output/n) for n in members}))
        print('JSCC_TRANSPORT_WORKER_COMPLETE',index,photons,flush=True)
    except BaseException as e:
        write(output/'failure.json',dict(passed=False,error=str(e),index=index,pilot=pilot));raise

def collect(release,simulation,transport,output,job):
    cfg=read(release/'transport_config.json');r=registry(simulation/'jobs.json');verify_files(simulation,cfg['source_registry_sha256'])
    output.mkdir(parents=True,exist_ok=False)
    counts={e:np.zeros((WORKERS,10496),np.int64) for e in (218,440)};primary=np.zeros((WORKERS,3),np.int64);receipts=[]
    for i,j in enumerate(r['jobs']):
        f=transport/f'worker_{i:04d}';s=read(f/'receipt.json')
        if not s['passed'] or s['pilot'] or any(s[k]!=j[k] for k in ('index','view','worker','seed','photons')):
            raise ValueError('Missing/failed/mismatched independent worker '+str(i))
        if s['allocation_job']!=str(job) or s['binary_sha256']!=BINARY_SHA or s['crystal_sha256']!=CRYSTAL_SHA:
            raise ValueError('Actual execution identity differs')
        verify_files(f,s['files'])
        if (f/'run.mac').read_bytes()!=(simulation/j['macro']).read_bytes().replace(b'\r\n',b'\n'):
            raise ValueError('Registered source changed beyond CRLF-to-LF')
        primary[i]=np.loadtxt(f/'PrimaryCount.csv',delimiter=',',dtype=np.int64).reshape(3)
        if primary[i].tolist()!=s['primary_counts'] or primary[i].sum()!=PER_WORKER:raise ValueError('Worker primary count differs')
        for e in (218,440):
            counts[e][i]=np.loadtxt(f/f'CntStat_{e}.csv',delimiter=',',dtype=np.int64).reshape(10496)
            if int(counts[e][i].sum())!=s['counts'][str(e)]:raise ValueError('Worker window closure differs')
        receipts.append(s)
        if (i+1)%100==0:print('JSCC_COLLECT_WORKER_SHA',i+1,flush=True)
    p=primary.sum(0);fraction=r['expected_primary_energy_fraction']['218']
    if p.sum()!=TOTAL or p[2]!=0 or abs(p[0]/TOTAL-fraction)>5*math.sqrt(fraction*(1-fraction)/TOTAL):
        raise ValueError('Actual source mixture/dose fails original 5SE identity guard')
    shutil.copytree(simulation,output/'source_registry')
    for e in counts:
        np.savetxt(output/f'projection_{e}.csv',counts[e].reshape(20,50,10496).sum(1),delimiter=',',fmt='%d')
    np.savez_compressed(output/'worker_counts.npz',primary_counts=primary,counts218=counts[218],counts440=counts[440])
    write(output/'worker_receipts.json',receipts)
    (output/'List').mkdir();list_rows=0
    for v in range(1,21):
        with (output/'List'/f'{v}.csv').open('wb') as sink:
            for i in range((v-1)*50,v*50):
                with (transport/f'worker_{i:04d}'/'List.csv').open('rb') as f:
                    for b in iter(lambda:f.read(8<<20),b''):sink.write(b);list_rows+=b.count(b'\n')
            sink.flush();os.fsync(sink.fileno())
    files=hashes(output)
    c=dict(passed=True,study=STUDY,total_primary_photons=TOTAL,job=int(job),primary_counts=p.tolist(),
        worker_indices=list(range(WORKERS)),seeds=list(range(SEED_BASE,SEED_BASE+WORKERS)),views=list(range(1,21)),
        source_registry_sha256=cfg['source_registry_sha256'],binary_sha256=BINARY_SHA,crystal_sha256=CRYSTAL_SHA,
        window_counts={str(e):int(x.sum()) for e,x in counts.items()},raw_list_rows=list_rows,files=files,
        event_policy='legacy',source_solid_angle='4pi',dose_multiplier=1,
        worker_elapsed_seconds_min_max=[min(s['resource']['elapsed_seconds'] for s in receipts),max(s['resource']['elapsed_seconds'] for s in receipts)])
    validate_collection(c);write(output/'collection.json',c)
    archive=output.parent/'transport_input.tar.gz'
    if archive.exists():raise FileExistsError('Never overwrite an existing input archive')
    with tarfile.open(str(archive)+'.writing','w:gz',compresslevel=1) as t:
        for name in sorted(hashes(output)):t.add(output/name,arcname=name,recursive=False)
    os.replace(str(archive)+'.writing',archive)
    write(output.parent/'collection_package.json',dict(passed=True,sha256=digest(archive),bytes=archive.stat().st_size,
        collection_sha256=digest(output/'collection.json'),archive=str(archive)))
    print('JSCC_FULL_TRANSPORT_COLLECTED',TOTAL,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('worker','collect'))
    for n in ('release','simulation','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--transport',type=Path);p.add_argument('--job',type=int);p.add_argument('--index',type=int)
    p.add_argument('--limit',type=int,default=5400);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if a.action=='worker':worker(a.release,a.simulation,a.output,a.index,a.limit,a.pilot)
    else:collect(a.release,a.simulation,a.transport,a.output,a.job)
