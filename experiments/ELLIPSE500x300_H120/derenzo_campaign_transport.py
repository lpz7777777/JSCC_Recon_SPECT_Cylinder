"""Run immutable accepted binaries and collect every independent CPU worker."""
import argparse
import os
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
import numpy as np
from derenzo_campaign_common import *
from ehe_common import bounded_process
from jscc_5e10_array import array_accounting


def worker(release,dataset,output,index,pilot=False):
    verify_files(release,read(release/'release_manifest.json')['sha256'])
    cfg=read(release/'campaign.json');info=dataset_info(dataset);spec=cfg['datasets'][dataset]
    registry=validate_registry(read(release/'sources'/dataset/'jobs.json'),dataset)
    verify_files(release/'sources'/dataset,spec['source_registry_sha256'])
    job=registry['jobs'][index]
    if not pilot and int(os.environ.get('SLURM_ARRAY_TASK_ID','-1'))!=index:raise ValueError('Actual array index differs')
    photons=PILOT_PHOTONS if pilot else PER_WORKER
    seed=info['seed_base']-1 if pilot else job['seed']
    binary=Path(spec['binary_path'])
    if digest(binary)!=spec['binary_sha256']:raise ValueError('Accepted binary changed')
    body=(release/'sources'/dataset/job['macro']).read_bytes()
    if pilot:body=re.sub(rb'/run/beamOn \d+',('/run/beamOn '+str(photons)).encode(),body)
    output.mkdir(parents=True,exist_ok=False)
    (output/'source.mac').write_bytes(body)
    if info['system']=='JSCC':
        crystal=Path(spec['crystal_path'])
        if digest(crystal)!=spec['crystal_sha256']:raise ValueError('Detector crystal input changed')
        shutil.copy2(crystal,output/'CrystalMatrix.txt')
    actual=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID']],text=True)
    meminfo=Path('/proc/meminfo').read_text();available=int(re.search(r'MemAvailable:\s+(\d+)',meminfo)[1])*1024
    allocation=dict(job=os.environ['SLURM_JOB_ID'],scontrol=actual,host_allocated_bytes=available,
                    denominator_kind='physical MemAvailable CPU operational guard',imaging_allocation_certificate=False)
    write(output/'allocation.json',allocation)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env['JSCC_RANDOM_SEED' if info['system']=='JSCC' else 'EHE_RANDOM_SEED']=str(seed)
    started=time.time()
    usage=bounded_process([str(binary),'source.mac'],output,1800 if pilot else spec['worker_limit_seconds'],
                          output/'transport.log',allocation,env)
    primary=(np.loadtxt(output/'PrimaryCount.csv',delimiter=',',dtype=np.int64).reshape(3).tolist()
             if info['system']=='JSCC' else read(output/'TransportSummary.json')['primary_counts'])
    expected=[photons,0,0] if info['energy_keV']==218 else [0,photons,0]
    if primary!=expected:raise ValueError('Actual emitted monoenergy dose differs')
    counts={}
    members=['source.mac','allocation.json','transport.log']
    for e in (218,440):
        p=output/f'CntStat_{e}.csv';v=np.loadtxt(p,delimiter=',',dtype=np.int64,ndmin=2)
        if v.shape!=(1,info['bins']) or np.any(v<0):raise ValueError('Actual complete detector count vector differs')
        counts[str(e)]=int(v.sum());members.append(p.name)
    if info['system']=='JSCC':members+=['PrimaryCount.csv','CrystalMatrix.txt','List.csv']
    else:
        summary=read(output/'TransportSummary.json')
        if summary['primary_events']!=photons or summary['detector_bins']!=2312:raise ValueError('EHE actual event receipt differs')
        audit=read(output/'EHE_MultiUnionAudit.json')
        if audit.get('passed') is not True or (audit['points'],audit['holes'])!=(11252,1250):raise ValueError('EHE actual geometry audit failed')
        verify_files(output,spec['geometry_evidence_sha256'])
        for e in (218,440):
            tagged=[]
            for source in (218,440):
                name=f'CntStat_{e}_from{source}.csv';members.append(name)
                tagged.append(np.loadtxt(output/name,delimiter=',',dtype=np.int64).reshape(info['bins']))
            total=np.loadtxt(output/f'CntStat_{e}.csv',delimiter=',',dtype=np.int64).reshape(info['bins'])
            if not np.array_equal(total,tagged[0]+tagged[1]):raise ValueError('EHE primary-tagged window closure failed')
            if np.any(tagged[1 if info['energy_keV']==218 else 0]):raise ValueError('Wrong primary energy in tagged data')
        members+=['TransportSummary.json','TransportTiming.json','EHE_MultiUnionAudit.json']+list(spec['geometry_evidence_sha256'])
    write(output/'receipt.json',dict(passed=True,study=STUDY,dataset=dataset,pilot=pilot,index=index,
          view=job['view'],worker=job['worker'],seed=seed,photons=photons,primary_counts=primary,
          counts=counts,binary_sha256=spec['binary_sha256'],registered_macro_sha256=job['macro_sha256'],
          actual_macro_sha256=digest(output/'source.mac'),allocation_job=os.environ['SLURM_JOB_ID'],
          array_job=os.environ.get('SLURM_ARRAY_JOB_ID'),array_task=int(os.environ.get('SLURM_ARRAY_TASK_ID',-1)),
          resource=usage,started_epoch=started,finished_epoch=time.time(),files={n:digest(output/n) for n in members}))
    print('DERENZO_ACTUAL_TRANSPORT_COMPLETE',dataset,index,photons,flush=True)


def collect(release,dataset,transport,output,parent):
    cfg=read(release/'campaign.json');spec=cfg['datasets'][dataset];info=dataset_info(dataset)
    verify_files(release,read(release/'release_manifest.json')['sha256'])
    registry=validate_registry(read(release/'sources'/dataset/'jobs.json'),dataset)
    text=subprocess.check_output(['sacct','-j',str(parent),'-n','-P','--format=JobID%40,JobIDRaw%40,State,ExitCode,NNodes,NCPUS,MaxRSS,Elapsed,AllocTRES%200,NodeList%120'],text=True)
    scheduler=array_accounting(text,parent,workers=WORKERS)
    if not scheduler['passed']:raise ValueError('Every actual worker and step must exit successfully')
    output.mkdir(parents=True,exist_ok=False)
    counts={e:np.zeros((20,info['bins']),np.int64) for e in (218,440)}
    primary=np.zeros(3,np.int64);receipts=[];rows=0
    if info['system']=='JSCC':(output/'List').mkdir()
    for i,j in enumerate(registry['jobs']):
        folder=transport/f'worker_{i:04d}';r=read(folder/'receipt.json');root=scheduler['roots'][str(i)]
        if (not r['passed'] or r['pilot'] or r['dataset']!=dataset or r['study']!=STUDY
                or any(r[k]!=j[k] for k in ['index','view','worker','seed','photons'])
                or str(r['array_job'])!=str(parent) or r['array_task']!=i
                or r['allocation_job']!=root['allocation_job'] or r['binary_sha256']!=spec['binary_sha256']):
            raise ValueError('Actual worker identity differs')
        alloc=read(folder/'allocation.json')
        for key,value in {'JobId':root['allocation_job'],'ArrayJobId':str(parent),'ArrayTaskId':str(i),
                          'NumNodes':'1','NumCPUs':'1','NumTasks':'1','CPUs/Task':'1'}.items():
            if not re.search(r'(?<!\S)'+re.escape(key)+'='+re.escape(value)+r'(?=\s|$)',alloc['scontrol']):raise ValueError('CPU allocation differs: '+key)
        verify_files(folder,r['files'])
        if (folder/'source.mac').read_bytes()!=(release/'sources'/dataset/j['macro']).read_bytes():raise ValueError('Actual source commands differ')
        expected=[PER_WORKER,0,0] if info['energy_keV']==218 else [0,PER_WORKER,0]
        if r['primary_counts']!=expected:raise ValueError('Worker monoenergy dose differs')
        primary+=r['primary_counts']
        for e in (218,440):
            v=np.loadtxt(folder/f'CntStat_{e}.csv',delimiter=',',dtype=np.int64).reshape(info['bins'])
            if int(v.sum())!=r['counts'][str(e)]:raise ValueError('All-bin receipt closure failed')
            counts[e][j['view']-1]+=v
        if info['system']=='JSCC':
            with (folder/'List.csv').open('rb') as src,(output/'List'/f"{j['view']}.csv").open('ab') as dst:
                for b in iter(lambda:src.read(8<<20),b''):dst.write(b);rows+=b.count(b'\n')
                dst.flush();os.fsync(dst.fileno())
        receipts.append(r)
        if (i+1)%20==0:print('DERENZO_COLLECTED',dataset,i+1,flush=True)
    for e,v in counts.items():np.savetxt(output/f'projection_{e}.csv',v,delimiter=',',fmt='%d')
    shutil.copytree(release/'sources'/dataset,output/'source_registry')
    shutil.copy2(release/'designs'/f'derenzo_{info["energy_keV"]}keV_geometry.json',output/'geometry.json')
    write(output/'worker_receipts.json',receipts);write(output/'worker_accounting.json',scheduler)
    c=dict(passed=True,study=STUDY,dataset=dataset,data_kind='actual_Geant4_transport',total_primary_photons=TOTAL,
           primary_counts=primary.tolist(),bins=info['bins'],workers=WORKERS,views=20,
           seed_base=info['seed_base'],source_solid_angle='4pi',object_material='vacuum',
           source_geometry_sha256=digest(output/'geometry.json'),binary_sha256=spec['binary_sha256'],
           transport_array_job=parent,raw_list_rows=rows,window_counts={str(e):int(v.sum()) for e,v in counts.items()},
           files=hashes(output))
    validate_collection(c,dataset);write(output/'collection.json',c)
    members=hashes(output);archive=output.parent/'transport_input.tar.gz'
    if archive.exists() or archive.with_suffix('.gz.writing').exists():raise FileExistsError('Preserve previous transport package')
    with tarfile.open(str(archive)+'.writing','w:gz',compresslevel=1) as t:
        for n in sorted(members):t.add(output/n,arcname=n,recursive=False)
    os.replace(str(archive)+'.writing',archive)
    write(output.parent/'package.json',dict(passed=True,dataset=dataset,files=members,
          archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,collection_sha256=digest(output/'collection.json')))
    print('DERENZO_COMPLETE_5E9_COLLECTED',dataset,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['worker','collect'])
    for n in ['release','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--index',type=int,default=0)
    p.add_argument('--pilot',action='store_true');p.add_argument('--transport',type=Path);p.add_argument('--parent',type=int)
    a=p.parse_args()
    if a.action=='worker':worker(a.release,a.dataset,a.output,a.index,a.pilot)
    else:collect(a.release,a.dataset,a.transport,a.output,a.parent)
