"""Eight dense-matrix EHE outputs, with two new actual Derenzo acquisitions."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
from campaign_20261102_io import execute, python_json, upload_files, sftp, mkdirs
from ehe_5e9_workflow import q
from derenzo_campaign_common import HERE, GPU_PROJECT, CPU_PROJECT, GPU_PYTHON, read, write, digest, hashes, verify_files
from derenzo_campaign_workflow import cpu_environment, extract_archive
from campaign_reliable_io_20261104 import python_json as reliable_python_json,cpu_member,accounting

TAG='ehe_dense_reruns_20261102'
DATA=HERE/'generated'/TAG
REPORT=HERE/'reports/EHE'/TAG
GPU=GPU_PROJECT+'/generated/'+TAG
CPU=CPU_PROJECT+'/generated/'+TAG
OLD=HERE/'generated/derenzo_equal_regions_mono_5e9_20261016'
TRANSPORT_TAG='derenzo_ehe_actual5e10_20261102'
DATASETS=('EHE_440','EHE_218')
SEEDS={'EHE_440':71100101,'EHE_218':71200101}
TOTAL,WORKERS,PER_WORKER,PER_VIEW=50_000_000_000,500,100_000_000,25


def transport_prepare():
    marker=REPORT/'transport_freeze.json';payload=DATA/'transport_payload'
    if marker.exists():
        f=read(marker);verify_files(payload,f['files']);return f
    DATA.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    old_freeze=read(HERE/'reports/Derenzo/derenzo_equal_regions_mono_5e9_20261016/transport_freeze.json')
    verify_files(OLD/'transport_payload',old_freeze['sha256'])
    payload.mkdir(exist_ok=False)
    source=(OLD/'transport_payload/derenzo_campaign_common.py').read_text('utf8')
    changes={"STUDY = 'derenzo_equal_regions_mono_5e9_20261016'":'STUDY = '+repr(TRANSPORT_TAG),
        'TOTAL, VIEWS, WORKERS, PER_VIEW, PER_WORKER = 5_000_000_000, 20, 200, 10, 25_000_000':
            'TOTAL, VIEWS, WORKERS, PER_VIEW, PER_WORKER = 50_000_000_000, 20, 500, 25, 100_000_000',
        "DATASETS = ('JSCC_440', 'JSCC_218', 'EHE_440', 'EHE_218')":'DATASETS = '+repr(DATASETS),
        'SEED_BASE = dict(zip(DATASETS, (46100101, 46200101, 46300101, 46400101)))':'SEED_BASE = '+repr(SEEDS)}
    common=source
    for old,new in changes.items():
        if common.count(old)!=1:raise ValueError('Source configuration anchor differs')
        common=common.replace(old,new)
    functions=lambda text:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    if functions(source)!=functions(common):raise ValueError('Original source geometry/dose checks changed')
    (payload/'derenzo_campaign_common.py').write_text(common,encoding='utf8',newline='\n')
    loader=importlib.util.spec_from_file_location('new_ehe_source',payload/'derenzo_campaign_common.py')
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    transport=(OLD/'transport_payload/derenzo_campaign_transport.py').read_text('utf8')
    old="['scontrol','show','job',os.environ['SLURM_JOB_ID']]"
    if transport.count(old)!=1:raise ValueError('Original allocation query differs')
    transport=transport.replace(old,"['scontrol','show','job','-o',os.environ['SLURM_ARRAY_JOB_ID']+'_'+os.environ['SLURM_ARRAY_TASK_ID']]")
    (payload/'derenzo_campaign_transport.py').write_text(transport,encoding='utf8',newline='\n')
    for name in ['ehe_common.py','jscc_5e10_array.py']:shutil.copyfile(OLD/'transport_payload'/name,payload/name)
    cfg=read(OLD/'transport_payload/campaign.json');cfg.update(study=TRANSPORT_TAG,total_primary_photons_per_dataset=TOTAL)
    cfg['datasets']={d:cfg['datasets'][d] for d in DATASETS}
    for dataset in DATASETS:
        energy=int(dataset.split('_')[1]);geo=OLD/'transport_payload/designs'/f'derenzo_{energy}keV_geometry.json'
        (payload/'designs').mkdir(exist_ok=True);shutil.copyfile(geo,payload/'designs'/geo.name)
        design=read(geo);source_root=payload/'sources'/dataset;(source_root/'macros').mkdir(parents=True)
        for view in range(1,21):
            text=module.macro_text(design,view,PER_WORKER)
            original=(OLD/'transport_payload/sources'/dataset/'macros'/f'v{view:02d}.mac').read_text('utf8')
            if text!=original.replace('/run/beamOn 25000000','/run/beamOn 100000000'):raise ValueError('Geometry or emission density changed')
            (source_root/'macros'/f'v{view:02d}.mac').write_text(text,encoding='utf8',newline='\n')
        jobs=[dict(index=i,view=i//PER_VIEW+1,worker=i%PER_VIEW,photons=PER_WORKER,seed=SEEDS[dataset]+i,
                   energy_keV=energy,macro=f'macros/v{i//PER_VIEW+1:02d}.mac',
                   macro_sha256=digest(source_root/'macros'/f'v{i//PER_VIEW+1:02d}.mac')) for i in range(WORKERS)]
        registry=dict(study=TRANSPORT_TAG,dataset=dataset,total_primary_photons=TOTAL,views=20,
                      workers_per_view=PER_VIEW,jobs=jobs,geometry_sha256=digest(geo))
        module.validate_registry(registry,dataset);write(source_root/'jobs.json',registry)
        cfg['datasets'][dataset].update(module.dataset_info(dataset),source_registry_sha256=hashes(source_root),worker_limit_seconds=28200)
    write(payload/'campaign.json',cfg)
    manifest=dict(sha256=hashes(payload));key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    manifest['release_key']=key;write(payload/'release_manifest.json',manifest)
    f=dict(key=key,release=CPU+'/releases/'+key,files=hashes(payload),actual_primaries_per_energy=TOTAL,
           workers_per_energy=WORKERS,photons_per_worker=PER_WORKER,views=20,new_seeds=SEEDS,
           source_geometry_AST_identical=True,original_binary_and_material_geometry_unchanged=True,no_pilot_or_validation_jobs=True)
    write(marker,f);return f


def deploy_transport():
    f=transport_prepare()
    if (REPORT/'transport_deployment.json').exists():return f
    upload_files('maty',f['release'],{n:DATA/'transport_payload'/n for n in f['files']})
    code='''import sys,json,pathlib,hashlib,shutil
c=json.loads(sys.argv[1]);r=pathlib.Path(c['release']);sys.path.insert(0,str(r))
from derenzo_campaign_common import *
verify_files(r,read(r/'release_manifest.json')['sha256']);cfg=read(r/'campaign.json')
for d,s in cfg['datasets'].items():
 validate_registry(read(r/'sources'/d/'jobs.json'),d)
 assert digest(s['binary_path'])==s['binary_sha256']
 assert b'/ellipse/addRod' in pathlib.Path(s['binary_path']).read_bytes()
for n in ['scripts','logs']:(pathlib.Path(c['remote'])/n).mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(c['remote']).free>20*(1<<30)
print('CAMPAIGN_JSON '+json.dumps(dict(passed=True,total=TOTAL,workers=WORKERS,seeds=SEED_BASE,
 binary={d:s['binary_sha256'] for d,s in cfg['datasets'].items()},free=shutil.disk_usage(c['remote']).free)))
'''
    proof=python_json('maty','python3',code,dict(f,remote=CPU),timeout=180)
    write(REPORT/'transport_deployment.json',proof);return f


def submit_registered(host,label,body,options,remote_root,output,release):
    path=REPORT/'jobs'/(label+'.json');intent=path.with_name(label+'_submission_intent.json')
    if path.exists():return read(path)
    if intent.exists():raise RuntimeError('Unresolved scheduler intent: '+label)
    script=DATA/'scripts'/(label+'.sh');script.parent.mkdir(exist_ok=True)
    script.write_text(body,encoding='utf8',newline='\n');upload_files(host,remote_root+'/scripts',{script.name:script})
    execute(host,'bash -n '+q(remote_root+'/scripts/'+script.name),readonly=True)
    cmd='sbatch --parsable '+options+' --job-name='+label+' --output='+q(remote_root+'/logs/'+label+'_%A_%a.out')+' --error='+q(remote_root+'/logs/'+label+'_%A_%a.err')+' '+q(remote_root+'/scripts/'+script.name)
    request=dict(command=cmd,script_sha256=digest(script),output=output,release=release,host=host,label=label,submitted_epoch=time.time())
    write(intent,request);raw=execute(host,cmd,timeout=60).strip();number=raw.split(';')[0]
    if not number.isdigit():raise RuntimeError('Uncertain submission; preserve intent')
    request.update(job=int(number),sbatch_result=raw);write(path,request)
    print('FORMAL_REGISTERED',host,label,number,flush=True);return request


def submit_transport():
    f=deploy_transport()
    queue=execute('maty','squeue -h -r -u "$USER" -o "%i"',readonly=True).splitlines()
    required=sum(0 if (REPORT/'jobs'/('ED5e10_'+d+'_MC.json')).exists() else WORKERS+1 for d in DATASETS)
    if len(queue)+required>2000:raise RuntimeError('CPU association lacks actual submission headroom; do not submit repeated rejects')
    for dataset in DATASETS:
        out=CPU+'/'+dataset+'/transport';program=f['release']+'/derenzo_campaign_transport.py'
        args='python3 -B '+q(program)+' worker --release '+q(f['release'])+' --dataset '+dataset+' --output '+q(out)+'/worker_$(printf "%04d" "$SLURM_ARRAY_TASK_ID") --index "$SLURM_ARRAY_TASK_ID"'
        array=submit_registered('maty','ED5e10_'+dataset+'_MC',cpu_environment()+'srun --chdir=/tmp --kill-on-bad-exit=1 '+args+'\n',
            '-p cnmix -N1 -n1 --ntasks-per-node=1 -c1 --chdir=/tmp --time=480 --array=0-499%200',CPU,out,f['release'])
        out=CPU+'/'+dataset+'/input'
        args='python3 -B '+q(program)+' collect --release '+q(f['release'])+' --dataset '+dataset+' --transport '+q(array['output'])+' --output '+q(out)+' --parent '+str(array['job'])
        submit_registered('maty','ED5e10_'+dataset+'_collect',cpu_environment()+args+'\n',
            '-p cnmix -N1 -n1 --ntasks-per-node=1 -c1 --chdir=/tmp --time=180 --dependency=afterok:'+str(array['job']),CPU,out,f['release'])


def existing_inputs():
    return {0:('NEMA','5e9',HERE/'generated/ehe_spect_5e9_200/transport',[440,218],'npy'),
            1:('NEMA','5e10',HERE/'generated/ehe_spect_5e10_200/counts',[440,218],'npy'),
            2:('Derenzo','5e9',OLD/'EHE_440/input',[440],'csv'),
            3:('Derenzo','5e9',OLD/'EHE_218/input',[218],'csv')}


def compact_input(index, source):
    target=DATA/'compact_inputs'/str(index)
    marker=target/'collection.json'
    if marker.exists():
        c=read(marker);verify_files(target,c['files'])
        if c['source_collection_sha256']!=digest(source/'collection.json'):raise ValueError('Original acquisition identity changed')
        return target
    target.mkdir(parents=True,exist_ok=False)
    original=read(source/'collection.json')
    shutil.copyfile(source/'collection.json',target/'source_collection.json')
    for name in original['files']:
        if name.startswith('projection_') and '/' not in name:
            if digest(source/name)!=original['files'][name]:raise ValueError('Actual observed projection changed')
            shutil.copyfile(source/name,target/name)
    c={k:original[k] for k in ['passed','study','dataset','data_kind','total_primary_photons','primary_counts','workers','views','bins','source_solid_angle','source_geometry_sha256'] if k in original}
    c.update(data_kind='actual_Geant4_transport',source_collection_sha256=digest(source/'collection.json'),
        source_collection_path=str(source/'collection.json'),derivation='Exact observed projection bytes; no resampling/scaling/Poisson generation',files=hashes(target))
    write(marker,c);return target


def prepare():
    marker=REPORT/'reconstruction_compact_freeze.json';payload=DATA/'reconstruction_compact_payload'
    if marker.exists():
        f=read(marker);verify_files(payload,f['files']);return f
    DATA.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    snapshot=read(HERE/'reports/Derenzo/derenzo_jscc_mono_5e10_20261030/combined_status_20261102_154549.json')['matrix']
    matrix_root=read(HERE/'reports/EHE/ehe_dense_surface_20261029/formal_job.json')['output']+'/factors'
    factors={}
    for energy in (218,440):
        m=snapshot[f'factors/A{energy}/factor_manifest.json'];expected=snapshot['completion.json']['factors'][str(energy)]
        if not m['passed'] or m['surface_samples_per_visible_face']!=65536:raise ValueError('Complete new dense factors required')
        factors['A'+str(energy)]=dict(root=matrix_root+'/A'+str(energy),manifest_sha256=expected,
            files=m['files'],geometry_sha256=m['files']['whole_geometry.npz'])
    inventory=read(HERE/'reports/rerun_dense_ehe_jscc_20261102/inventory_gpu.json')
    cross=inventory['datasets']['ehe_spect_5e9_200']['responses_conversion_b02df58a9d3fd48c/C440to218/factor_manifest.json']
    factors['C440to218']=dict(root=GPU_PROJECT+'/generated/ehe_spect_5e9_200/responses_conversion_b02df58a9d3fd48c/C440to218',
        manifest_sha256='c73c0ec4fc9382aceb55228410a852da20b68a8be8bcf8675eaf9fa6c6a9db4a',files=cross['files'],geometry_sha256=cross['files']['whole_geometry.npz'])
    conditions={}
    for index,(phantom,dose,local,energies,fmt) in existing_inputs().items():
        coll=read(local/'collection.json')
        if coll['total_primary_photons']!=int(float(dose)):raise ValueError('Original acquired dose differs')
        compact=compact_input(index,local)
        conditions[str(index)]=dict(index=index,phantom=phantom,dose=dose,energies_keV=energies,input_root=GPU+'/compact_inputs/'+str(index),
            projection_format=fmt,actual_primary_photons=coll['total_primary_photons'],collection_sha256=digest(compact/'collection.json'),
            source_collection_sha256=digest(local/'collection.json'),
            requires_transport=False,data_kind='actual_Geant4_transport',original_source=str(local))
    for index,energy in [(4,440),(5,218)]:
        conditions[str(index)]=dict(index=index,phantom='Derenzo',dose='5e10',energies_keV=[energy],input_root=GPU+'/EHE_'+str(energy)+'/input',
            projection_format='csv',actual_primary_photons=TOTAL,requires_transport=True,data_kind='actual_Geant4_transport')
    contract=dict(study=TAG,conditions=conditions,factors=factors,iterations=200,save_step=1,initial_density=1,
        regularization='none',cross_energy_image_sum=False,physical_calibration_passed=False,
        cross_response_changed=False,NEMA218_background='Original frozen C440to218 applied to this new EHE final440 image',
        mono_Derenzo218_background='zero',full_input_prevalidation_passed=False,separate_validation_job=False,
        authorization='2026-11-02 human requests NEMA/Derenzo 5e9/5e10 with new dense EHE matrices')
    payload.mkdir(exist_ok=False)
    # Original operator/update loop copied by SHA, with a new orchestration only.
    parent=OLD/'gpu_payload'
    for name in ['ehe_common.py','torch_active_operator.py','single_checkpoint_mlem.py','whole_geometry.npz']:
        source=OLD/'transport_payload'/name if name=='ehe_common.py' else parent/name
        shutil.copyfile(source,payload/name)
    shutil.copyfile(HERE/'ehe_dense_rerun_runtime_20261102.py',payload/'ehe_dense_rerun_runtime_20261102.py')
    for f in factors.values():
        if digest(payload/'whole_geometry.npz')!=f['geometry_sha256']:raise ValueError('Different source reconstruction geometry')
    write(payload/'contract.json',contract)
    manifest=dict(sha256=hashes(payload));key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    manifest['release_key']=key;write(payload/'release_manifest.json',manifest)
    for path in payload.glob('*.py'):compile(path.read_text('utf8'),str(path),'exec')
    f=dict(key=key,release=GPU+'/releases/'+key,files=hashes(payload),matrix_parent_job=1749085,
           unchanged_helpers={n:digest(payload/n) for n in ['torch_active_operator.py','single_checkpoint_mlem.py']})
    write(marker,f);return f


def deploy():
    f=prepare()
    if (REPORT/'reconstruction_compact_deployment.json').exists():return f
    upload_files('gpu',f['release'],{n:DATA/'reconstruction_compact_payload'/n for n in f['files']})
    # New isolated input copies preserve the original large archives and studies.
    archive=DATA/'compact_actual_inputs.tar.gz'
    if not archive.exists():
        with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
            for index,(_,_,local,_,_) in existing_inputs().items():
                source=compact_input(index,local);coll=read(source/'collection.json');verify_files(source,coll['files'])
                for name in sorted(set(coll['files'])|{'collection.json'}):tar.add(source/name,arcname=str(index)+'/'+name,recursive=False)
    with sftp('gpu') as session:
        mkdirs(session,GPU+'/scripts');mkdirs(session,GPU+'/logs')
        remote=GPU+'/compact_actual_inputs.tar.gz'
        try:session.stat(remote)
        except FileNotFoundError:
            temporary=remote+'.uploading';session.put(str(archive),temporary);session.rename(temporary,remote)
    code='''import sys,json,pathlib,tarfile,hashlib,shutil
c=json.loads(sys.argv[1]);r=pathlib.Path(c['release']);sys.path.insert(0,str(r))
from ehe_common import *
verify_files(r,read(r/'release_manifest.json')['sha256']);cfg=read(r/'contract.json');root=pathlib.Path(c['remote'])
assert digest(root/'compact_actual_inputs.tar.gz')==c['archive_sha256']
target=root/'compact_inputs'
if not target.exists():
 with tarfile.open(root/'compact_actual_inputs.tar.gz') as tar:
  for m in tar.getmembers():assert m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts
  target.mkdir();tar.extractall(target)
for i in range(4):
 s=cfg['conditions'][str(i)];p=pathlib.Path(s['input_root']);assert digest(p/'collection.json')==s['collection_sha256']
 verify_files(p,read(p/'collection.json')['files'])
 assert digest(p/'source_collection.json')==s['source_collection_sha256']
 original=read(p/'source_collection.json')
 for n,h in original['files'].items():
  if n.startswith('projection_') and '/' not in n:assert digest(p/n)==h
for n,s in cfg['factors'].items():
 p=pathlib.Path(s['root']);assert digest(p/'factor_manifest.json')==s['manifest_sha256'];verify_files(p,s['files'])
assert shutil.disk_usage(root).free>30*(1<<30)
print('CAMPAIGN_JSON '+json.dumps(dict(passed=True,full_matrix_and_existing_input_SHA_passed=True,
 physical_calibration_passed=False,full_input_GPU_prevalidation_passed=False,free=shutil.disk_usage(root).free)))
'''
    proof=python_json('gpu',GPU_PYTHON,code,dict(f,remote=GPU,archive_sha256=digest(archive)),timeout=900)
    write(REPORT/'reconstruction_compact_deployment.json',proof);return f


def submit_reconstruction(indices=(0,1,2,3),label='EHEdense_existing'):
    f=deploy();body='#!/usr/bin/env bash\nset -euo pipefail\nsource /etc/profile.d/modules.sh\nmodule load cuda/12.9\n'
    body+='export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1\n'
    body+='index="$SLURM_ARRAY_TASK_ID"\n'+q(GPU_PYTHON)+' -B '+q(f['release']+'/ehe_dense_rerun_runtime_20261102.py')+' --release '+q(f['release'])+' --index "$index" --output '+q(GPU)+'/results/condition_${index}\n'
    return submit_registered('gpu',label,body,'-p gpu_5090 --qos=gpugpu -N1 -n1 -c8 --gres=gpu:1 --time=360 --array='+','.join(map(str,indices)),GPU,GPU+'/results',f['release'])


def transport_status():
    jobs={d:dict(collection=read(REPORT/'jobs'/('ED5e10_'+d+'_collect.json')),
                 transport=read(REPORT/'jobs'/('ED5e10_'+d+'_MC.json'))) for d in DATASETS}
    code='''import sys,json,subprocess,pathlib
c=json.loads(sys.argv[1]);out={}
queue=subprocess.check_output(['squeue','-h','-r','-u','maty','-o','%i|%T'],text=True)
for d,b in c.items():
 j=b['collection']
 text=subprocess.check_output(['sacct','-j',str(j['job']),'-nP','-o','JobID,State,ExitCode'],text=True)
 root=[s.split('|') for s in text.splitlines() if s.split('|')[0]==str(j['job'])]
 mc=subprocess.check_output(['sacct','-j',str(b['transport']['job']),'-nP','-o','JobID%40,State,ExitCode'],text=True)
 failed=[s.split('|') for s in mc.splitlines() if s.split('|')[1].split()[0] in ['FAILED','CANCELLED','OUT_OF_MEMORY','TIMEOUT','NODE_FAIL','PREEMPTED','BOOT_FAIL']]
 counts={}
 for s in mc.splitlines():
  row=s.split('|')
  if '.' not in row[0] and '[' not in row[0] and row[1] not in ['RUNNING','PENDING']:counts[row[1]]=counts.get(row[1],0)+1
 for line in queue.splitlines():
  row=line.split('|')
  if row[0].startswith(str(b['transport']['job'])+'_'):counts[row[1]]=counts.get(row[1],0)+1
 out[d]=dict(accounting=text,complete=bool(root and root[0][1:3]==['COMPLETED','0:0']),root=root,
             MC_state_counts=counts,MC_failures=failed)
print('CAMPAIGN_JSON '+json.dumps(out))
'''
    return reliable_python_json('maty','python3',code,jobs)


def transfer_completed(dataset):
    proof_path=REPORT/dataset/'input_delivery.json'
    if proof_path.exists():return read(proof_path)
    local=DATA/dataset;local.mkdir(exist_ok=True)
    # All original root/batch/extern/srun records must have exited successfully.
    records=accounting('maty',[read(REPORT/'jobs'/('ED5e10_'+dataset+'_'+n+'.json'))['job']
        for n in ('MC','collect')],'python3')
    mc=str(read(REPORT/'jobs'/('ED5e10_'+dataset+'_MC.json'))['job'])
    collector=str(read(REPORT/'jobs'/('ED5e10_'+dataset+'_collect.json'))['job'])
    required={mc+'_'+str(i)+s for i in range(500) for s in ('','.batch','.extern','.0')}
    required|={collector+s for s in ('','.batch','.extern')}
    byid={row[0]:row[1:3] for row in records}
    if not required<=set(byid) or any(value!=['COMPLETED','0:0'] for value in byid.values()):
        raise ValueError('Full actual transport/collection accounting not successful')
    write(REPORT/dataset/'transfer_accounting_r6.json',dict(passed=True,rows=records))
    code='''import sys,json,pathlib,hashlib
p=pathlib.Path(json.loads(sys.argv[1])['root'])/'package.json'
print('CAMPAIGN_JSON '+json.dumps(dict(package=json.loads(p.read_text()),package_file_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
'''
    meta=reliable_python_json('maty','python3',code,dict(root=CPU+'/'+dataset))
    package=meta['package']
    if not package['passed'] or package['dataset']!=dataset:raise ValueError('Original package identity differs')
    write(REPORT/dataset/'transport_package_r6.json',meta)
    archive=local/'transport_input.tar.gz'
    cpu_member(CPU+'/'+dataset+'/transport_input.tar.gz',archive,package['archive_sha256'])
    if digest(archive)!=package['archive_sha256']:raise ValueError('Full archived input SHA differs')
    extract_archive(archive,local/'input',package['files']);verify_files(local/'input',package['files'])
    collection=read(local/'input/collection.json')
    if (collection['dataset']!=dataset or collection['total_primary_photons']!=TOTAL or collection['workers']!=WORKERS
            or collection['data_kind']!='actual_Geant4_transport' or collection['source_solid_angle']!='4pi'):
        raise ValueError('Actual complete independent EHE5e10 dose differs')
    upload_files('gpu',GPU+'/'+dataset,{'transport_input.tar.gz':archive})
    code='''import sys,json,pathlib,tarfile
c=json.loads(sys.argv[1]);sys.path.insert(0,c['release']);from ehe_common import *
p=pathlib.Path(c['folder']);assert digest(p/'transport_input.tar.gz')==c['archive_sha256']
if not (p/'input').exists():
 with tarfile.open(p/'transport_input.tar.gz') as t:
  members=t.getmembers()
  assert {m.name for m in members}==set(c['files']) and len(members)==len(c['files'])
  for m in members:assert m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts
  (p/'input').mkdir();t.extractall(p/'input')
verify_files(p/'input',c['files']);v=read(p/'input/collection.json');assert v['total_primary_photons']==50000000000
proof=dict(passed=True,actual_primary_photons=v['total_primary_photons'],collection_sha256=digest(p/'input/collection.json'),strict_all_member_SHA=True)
write(p/'gpu_input_ready.json',proof);print('CAMPAIGN_JSON '+json.dumps(proof))
'''
    proof=python_json('gpu',GPU_PYTHON,code,dict(package,release=read(REPORT/'reconstruction_compact_freeze.json')['release'],folder=GPU+'/'+dataset),timeout=180,readonly=False)
    proof.update(strict_local_fetch=True,CPU_package_sha256=hashlib.sha256(json.dumps(package,sort_keys=True).encode()).hexdigest(),
        CPU_package_file_sha256=meta['package_file_sha256'],archive_sha256=package['archive_sha256'],
        complete_archive_and_all27_members_verified=True,all_source_root_batch_extern_srun_success=True,
        transport_backend='acknowledged512byte lossless SSH replies; original partial preserved')
    write(proof_path,proof);return proof


def advance():
    state=transport_status();write(REPORT/'transport_latest_status.json',dict(checked_epoch=time.time(),datasets=state))
    for dataset,s in state.items():
        failed=[r for r in s['root'] if r[1] in ['FAILED','CANCELLED','OUT_OF_MEMORY','TIMEOUT','NODE_FAIL']]
        failed+=s['MC_failures']
        if failed:raise RuntimeError('Preserve failed MC/collection without automatic resubmission: '+str(failed))
        if s['complete']:transfer_completed(dataset)
    if all((REPORT/d/'input_delivery.json').exists() for d in DATASETS):
        submit_reconstruction((4,5),'EHEdense_Derenzo5e10');return True
    return False


def run():
    import psutil
    reg=DATA/'one_shot_registration.json'
    if reg.exists():
        prior=read(reg)
        if prior.get('status')=='running' and psutil.pid_exists(prior['pid']) and abs(psutil.Process(prior['pid']).create_time()-prior['process_create_time'])<1:
            raise RuntimeError('This exact finite handoff is already alive')
    record=dict(pid=os.getpid(),process_create_time=psutil.Process().create_time(),status='running',started_epoch=time.time(),deadline_seconds=7*86400,
        scope='Only new EHE Derenzo5e10 MC/collection strict handoff to the two dense formal reconstructions',no_recurring_timer=True)
    write(reg,record)
    try:
        failures=0
        while time.time()-record['started_epoch']<record['deadline_seconds']:
            try:
                done=advance();failures=0
            except (__import__('paramiko').SSHException,TimeoutError) as error:
                failures+=1
                if failures>=3:raise
                done=False;write(REPORT/'transient_connection.json',dict(error=str(error),consecutive=failures,checked_epoch=time.time()))
            if done:
                write(reg,dict(record,status='handoff_complete',finished_epoch=time.time()));return
            time.sleep(180)
        raise TimeoutError('Finite handoff deadline reached; all jobs/results retained')
    except BaseException as error:
        write(reg,dict(record,status='failed',type=type(error).__name__,message=str(error),finished_epoch=time.time()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['transport_prepare','deploy_transport','submit_transport','prepare','deploy','submit_reconstruction','advance','run'])
    args=parser.parse_args()
    if args.action!='run' and (DATA/'one_shot_registration.json').exists():
        import psutil
        prior=read(DATA/'one_shot_registration.json')
        if prior.get('status')=='running' and psutil.pid_exists(prior['pid']) and abs(psutil.Process(prior['pid']).create_time()-prior['process_create_time'])<1:
            raise RuntimeError('This finite handoff is alive; do not parallelize advance/transfer/submit')
    globals()[args.action]()
