"""Manual-request Derenzo Geant4 -> strict transfer -> validated reconstruction."""
import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from derenzo_campaign_common import *
from ehe_5e9_workflow import connection, command, q
from jscc_5e10_5090_8x2_trial import live_registration

REGISTRATION=DATA/'controller_registration.json'
TRANSPORT_SOURCES=('derenzo_campaign_common.py','derenzo_campaign_transport.py','ehe_common.py','jscc_5e10_array.py')


@contextlib.contextmanager
def controlled():
    if live_registration(REGISTRATION):raise RuntimeError('This independent campaign controller is already alive')
    write(REGISTRATION,dict(pid=os.getpid(),status='running',started_epoch=time.time()))
    code=1
    try:yield;code=0
    finally:write(REGISTRATION,dict(read(REGISTRATION),status='complete',exit_code=code,finished_epoch=time.time()))


def prepare_transport():
    proof=REPORT/'transport_freeze.json';payload=DATA/'transport_payload'
    if proof.exists():
        f=read(proof);verify_files(payload,f['sha256']);return f
    payload.mkdir(parents=True,exist_ok=False)
    old=HERE/'reports/NEMA_Body_H60/jscc_geant4_5e10_10000'
    jscc_release=read(old/'transport_array_deployment.json')['release']
    ehe=read(HERE/'generated/ehe_spect_5e10_200/transport_payload/config.json')
    cfg=dict(study=STUDY,total_primary_photons_per_dataset=TOTAL,datasets={})
    geometry_checks={}
    for energy in (218,440):
        design_path=DESIGN/'data'/f'derenzo_{energy}keV_geometry.json';d=read(design_path)
        if d['rod_count']!=(114 if energy==218 else 53) or not d['all_regions_equal_area']:
            raise ValueError('Referenced equal-region design differs')
        if d['height_mm']!=60 or d['region_side_mm']!=115 or d['region_centroid_radius_mm']!=90:
            raise ValueError('Exact referenced geometry required')
        for rod in d['rods']:
            if not math.isclose(rod['volume_mm3'],math.pi*(rod['diameter_mm']/2)**2*60,rel_tol=1e-12):raise ValueError('Cylinder volume differs')
        geometry_checks[str(energy)]=dict(rods=d['rod_count'],geometry_sha256=digest(design_path),
                  diameters_mm=d['diameters_mm'],source_density='uniform within all hot rods; zero elsewhere')
        dest=payload/'designs'/design_path.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(design_path,dest)
        viewdir=DATA/'macros'/str(energy);viewdir.mkdir(parents=True)
        for view in range(1,21):
            (viewdir/f'Derenzo_{energy}keV_v{view:02d}.mac').write_bytes(macro_text(d,view,TOTAL//20).encode())
    for dataset in DATASETS:
        info=dataset_info(dataset);energy=info['energy_keV'];d=read(payload/'designs'/f'derenzo_{energy}keV_geometry.json')
        source=payload/'sources'/dataset;(source/'macros').mkdir(parents=True)
        jobs=[]
        for v in range(1,21):(source/'macros'/f'v{v:02d}.mac').write_bytes(macro_text(d,v,PER_WORKER).encode())
        for i in range(WORKERS):
            macro=f'macros/v{i//PER_VIEW+1:02d}.mac'
            jobs.append(dict(index=i,view=i//PER_VIEW+1,worker=i%PER_VIEW,photons=PER_WORKER,
                             seed=info['seed_base']+i,energy_keV=energy,macro=macro,macro_sha256=digest(source/macro)))
        registry=dict(study=STUDY,dataset=dataset,total_primary_photons=TOTAL,views=20,
                      workers_per_view=PER_VIEW,jobs=jobs,geometry_sha256=geometry_checks[str(energy)]['geometry_sha256'])
        validate_registry(registry,dataset);write(source/'jobs.json',registry)
        spec=dict(info,source_registry_sha256=hashes(source),worker_limit_seconds=14040)
        if info['system']=='JSCC':
            spec.update(binary_path=jscc_release+'/gamma01',binary_sha256=read(old/'transport_array_freeze.json')['binary_sha256'],
                        crystal_path=jscc_release+'/CrystalMatrix.txt',crystal_sha256=read(old/'transport_array_freeze.json')['crystal_sha256'])
        else:spec.update({k:ehe[k] for k in ['binary_path','binary_sha256','geometry_evidence_sha256']})
        cfg['datasets'][dataset]=spec
    for name in TRANSPORT_SOURCES:shutil.copy2(HERE/name,payload/name)
    write(payload/'campaign.json',cfg)
    files=hashes(payload);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    write(payload/'release_manifest.json',dict(release_key=key,sha256=files))
    f=dict(study=STUDY,release_key=key,root=CPU_BASE+'/releases/'+key,sha256=hashes(payload),
           design_image_sha256=digest(DESIGN/'figures/derenzo_440_218_design.png'),geometry_checks=geometry_checks,
           total_primary_photons_per_dataset=TOTAL,total_across_four_datasets=4*TOTAL,
           workers_per_dataset=WORKERS,photons_per_worker=PER_WORKER,pilot_photons_excluded=PILOT_PHOTONS,
           original_nema_jobs_untouched=True,material='vacuum; analytic emission cylinders; no material envelope',
           grid_policy='Reuse existing full 120mm matrices; 6mm radial basis; not a sub-grid resolution measurement',
           output_channels={k:list(v) for k,v in CHANNELS.items()},recurring_automations_remain_paused=True)
    write(proof,f);return f


def put_payload(c,payload,remote):
    files=hashes(payload);archive=DATA/('deploy_'+hashlib.sha256(remote.encode()).hexdigest()[:16]+'.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=1) as t:
        for n in sorted(files):t.add(payload/n,arcname=n,recursive=False)
    command(c,'test ! -e '+q(remote)+' && mkdir -p '+q(remote))
    with c.open_sftp() as s:s.put(str(archive),remote+'/payload.tar.gz')
    if command(c,'sha256sum '+q(remote+'/payload.tar.gz')).split()[0]!=digest(archive):raise ValueError('Deployment archive differs')
    command(c,'tar --no-same-owner -xzf '+q(remote+'/payload.tar.gz')+' -C '+q(remote),timeout=300)


def deploy_transport(f):
    p=REPORT/'transport_deployment.json'
    if p.exists():return read(p)
    cfg=read(DATA/'transport_payload/campaign.json')
    with connection('maty') as c:
        code='from pathlib import Path;import hashlib;items='+repr({d:{k:s[k] for k in ['binary_path','binary_sha256']} for d,s in cfg['datasets'].items()})+';assert all(hashlib.sha256(Path(s["binary_path"]).read_bytes()).hexdigest()==s["binary_sha256"] and b"/ellipse/addRod" in Path(s["binary_path"]).read_bytes() for s in items.values());print("ACCEPTED_BINARIES_AND_ANALYTIC_ROD_COMMAND_PASS")'
        identity=command(c,'python3 -c '+q(code),timeout=60)
        put_payload(c,DATA/'transport_payload',f['root'])
        check='from pathlib import Path;from derenzo_campaign_common import *;verify_files(Path("."),read("release_manifest.json")["sha256"]);print("ALL_TRANSPORT_RELEASE_SHA_PASS")'
        identity+=command(c,'cd '+q(f['root'])+' && PYTHONDONTWRITEBYTECODE=1 python3 -B -c '+q(check),timeout=120)
        command(c,'mkdir -p '+q(CPU_BASE+'/logs')+' '+q(CPU_BASE+'/scripts'))
    result=dict(release=f['root'],freeze_sha256=digest(REPORT/'transport_freeze.json'),identity=identity)
    write(p,result);return result


def submit_cpu(c,dataset,stage,script,options,output):
    path=REPORT/dataset/(stage+'_job.json')
    if path.exists():return read(path)
    intent=REPORT/dataset/(stage+'_submission_intent.json')
    if intent.exists():raise RuntimeError('Unresolved actual submission intent; inspect scheduler: '+str(intent))
    local=DATA/'launch_scripts'/(dataset+'_'+stage+'.sh');local.parent.mkdir(exist_ok=True)
    if local.exists() and local.read_bytes()!=script.encode():raise ValueError('Preserve existing launch script')
    local.write_bytes(script.encode());remote=CPU_BASE+'/scripts/'+local.name
    exists=command(c,'if test -e '+q(remote)+'; then sha256sum '+q(remote)+'; fi').strip()
    if exists:
        if exists.split()[0]!=digest(local):raise ValueError('Existing remote script differs')
    else:
        with c.open_sftp() as s:s.put(str(local),remote)
    command(c,'bash -n '+q(remote))
    sbatch='sbatch --parsable -p cnmix -N1 -n1 --ntasks-per-node=1 --cpus-per-task=1 --chdir=/tmp '+options
    sbatch+=' --job-name=DR5e9_'+dataset+'_'+stage+' --output='+q(CPU_BASE+'/logs/'+dataset+'_'+stage+'.%A_%a.out')+' --error='+q(CPU_BASE+'/logs/'+dataset+'_'+stage+'.%A_%a.err')+' '+q(remote)
    write(intent,dict(command=sbatch,script_sha256=digest(local),resolved=False,started_epoch=time.time()))
    number=command(c,sbatch,timeout=60).split(';')[0].strip()
    if not number.isdigit():raise RuntimeError('Ambiguous actual submission; never repeat sbatch')
    record=dict(job=int(number),dataset=dataset,stage=stage,host='maty',output=output,
                script_sha256=digest(local),release=read(REPORT/'transport_deployment.json')['release'],submitted_epoch=time.time())
    write(path,record);write(intent,dict(read(intent),registered_job=int(number),resolved=True))
    print('REGISTERED_CPU',dataset,stage,number,flush=True);return record


def cpu_environment():
    return '#!/usr/bin/env bash\nset -euo pipefail\nsource /etc/profile.d/modules.sh\nmodule load compilers/gcc/v12.2.0\nsource /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n'


def submit_pilots():
    f=prepare_transport();deploy_transport(f)
    with connection('maty') as c:
        for dataset in DATASETS:
            output=CPU_BASE+'/'+dataset+'/pilot'
            args='python3 -B '+q(f['root']+'/derenzo_campaign_transport.py')+' worker --release '+q(f['root'])+' --dataset '+dataset+' --output '+q(output)+' --pilot'
            submit_cpu(c,dataset,'pilot',cpu_environment()+'srun --chdir=/tmp --kill-on-bad-exit=1 '+args+'\n','--time=35',output)


def source_complete(c,record):
    text=command(c,'sacct -j '+str(record['job'])+' -n -P --format=JobID,State,ExitCode',timeout=60)
    rows=[x.split('|') for x in text.splitlines() if x.strip()]
    failures=[r for r in rows if r[1].split()[0] in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED']]
    if failures:raise RuntimeError('Preserve failed scientific job without retry: '+str(failures))
    return bool(rows and any(r[:3]==[str(record['job']),'COMPLETED','0:0'] for r in rows)
                and all(r[1:3]==['COMPLETED','0:0'] for r in rows)),text


def accept_pilot(c,dataset):
    path=REPORT/dataset/'pilot_acceptance.json'
    if path.exists():return True
    record=read(REPORT/dataset/'pilot_job.json');done,acc=source_complete(c,record)
    if not done:return False
    target=REPORT/dataset/'pilot';target.mkdir(exist_ok=True)
    with c.open_sftp() as s:
        for n in ['receipt.json','source.mac','transport.log','allocation.json']:
            expected=command(c,'sha256sum '+q(record['output']+'/'+n)).split()[0]
            s.get(record['output']+'/'+n,str(target/n))
            if digest(target/n)!=expected:raise ValueError('Pilot strict metadata fetch differs')
    r=read(target/'receipt.json');info=dataset_info(dataset)
    expected=[PILOT_PHOTONS,0,0] if info['energy_keV']==218 else [0,PILOT_PHOTONS,0]
    if not r['passed'] or not r['pilot'] or r['dataset']!=dataset or r['primary_counts']!=expected:raise ValueError('Actual pilot monoenergy dose differs')
    write(path,dict(passed=True,job=record['job'],receipt=r,accounting=acc,strict_small_fetch=True,pilot_excluded_from_formal_dose=True))
    return True


def submit_production_transport():
    if all((REPORT/d/'collection_job.json').exists() for d in DATASETS):return
    f=read(REPORT/'transport_freeze.json')
    with connection('maty') as c:
        for dataset in DATASETS:
            if not accept_pilot(c,dataset):continue
            output=CPU_BASE+'/'+dataset+'/transport'
            args='python3 -B '+q(f['root']+'/derenzo_campaign_transport.py')+' worker --release '+q(f['root'])+' --dataset '+dataset+' --output '+q(output)+'/worker_$(printf "%04d" "$SLURM_ARRAY_TASK_ID") --index "$SLURM_ARRAY_TASK_ID"'
            array=submit_cpu(c,dataset,'transport',cpu_environment()+'srun --chdir=/tmp --kill-on-bad-exit=1 '+args+'\n','--time=240 --array=0-199%200',output)
            output=CPU_BASE+'/'+dataset+'/input'
            args='python3 -B '+q(f['root']+'/derenzo_campaign_transport.py')+' collect --release '+q(f['root'])+' --dataset '+dataset+' --transport '+q(array['output'])+' --output '+q(output)+' --parent '+str(array['job'])
            submit_cpu(c,dataset,'collection',cpu_environment()+args+'\n','--time=120 --dependency=afterok:'+str(array['job']),output)


def download(c,remote,target,expected):
    """Resume only an immutable SHA-bound archive, checking any retained prefix."""
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if digest(target)!=expected:raise ValueError('Preserve mismatched previous artifact: '+str(target))
        return
    partial=target.with_name(target.name+'.partial')
    with c.open_sftp() as s:
        size=s.stat(remote).st_size;offset=partial.stat().st_size if partial.exists() else 0
        if offset>size:raise ValueError('Preserve oversized partial archive')
        if offset:
            code='import hashlib;f=open('+repr(remote)+',"rb");h=hashlib.sha256();left='+str(offset)+'\nwhile left:\n b=f.read(min(left,8<<20));assert b;h.update(b);left-=len(b)\nprint(h.hexdigest())'
            if command(c,'python3 -c '+q(code),timeout=180).strip()!=digest(partial):raise ValueError('Partial archive prefix differs')
        with s.open(remote,'rb') as src,partial.open('ab') as dst:
            src.seek(offset)
            while True:
                b=src.read(1<<20)
                if not b:break
                dst.write(b)
            dst.flush();os.fsync(dst.fileno())
    if digest(partial)!=expected:raise ValueError('Strict archive SHA differs; preserve partial')
    os.replace(partial,target)


def metadata(c,remote,local):
    expected=command(c,'sha256sum '+q(remote)).split()[0]
    download(c,remote,local,expected)
    return read(local)


def extract_archive(archive,target,files):
    if target.exists():
        verify_files(target,files);return
    target.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive) as t:
        members=t.getmembers()
        if {m.name for m in members}!=set(files) or any(not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members):raise ValueError('Unexpected or unsafe archive members')
        t.extractall(target)
    verify_files(target,files)


def fetch_transport(dataset):
    proof=REPORT/dataset/'transport_acceptance.json'
    if proof.exists():return True
    r=read(REPORT/dataset/'collection_job.json')
    with connection('maty') as c:
        done,acc=source_complete(c,r)
        if not done:
            parent=read(REPORT/dataset/'transport_job.json')['job']
            raw=command(c,'sacct -X -j '+str(parent)+' -n -P --format=JobID,State,ExitCode',timeout=90)
            failed=[line for line in raw.splitlines() if line.strip() and line.split('|')[1].split()[0] in
                    ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED']]
            if failed:raise RuntimeError('Preserve successful CPU workers; failed array requires bounded repair: '+str(failed[:8]))
            return False
        package=metadata(c,CPU_BASE+'/'+dataset+'/package.json',REPORT/dataset/'transport_package.json')
        if not package['passed'] or package['dataset']!=dataset:raise ValueError('Collection package identity differs')
        if shutil.disk_usage(DATA).free<package['archive_bytes']*3+(2<<30):raise OSError('Insufficient local strict-fetch space')
        archive=DATA/dataset/'transport_input.tar.gz'
        download(c,CPU_BASE+'/'+dataset+'/transport_input.tar.gz',archive,package['archive_sha256'])
    target=DATA/dataset/'input';extract_archive(archive,target,package['files'])
    coll=validate_collection(read(target/'collection.json'),dataset)
    if digest(target/'collection.json')!=package['collection_sha256']:raise ValueError('Collection identity differs')
    write(proof,dict(passed=True,strict_fetch_passed=True,collection=coll,local_input=str(target.resolve()),
          package_sha256=digest(REPORT/dataset/'transport_package.json'),archive_sha256=package['archive_sha256'],
          collection_accounting=acc,collection_job=r['job']))
    print('DERENZO_STRICT_TRANSPORT_DELIVERED',dataset,flush=True);return True


def prepare_gpu():
    proof=REPORT/'gpu_freeze.json';payload=DATA/'gpu_payload'
    if proof.exists():
        f=read(proof);verify_files(payload,f['sha256']);return f
    old=HERE/'generated/jscc_geant4_5e10_10000/production_payload'
    old_freeze=read(HERE/'reports/NEMA_Body_H60/jscc_geant4_5e10_10000/production_freeze.json')
    selected={n:s for n,s in old_freeze['sha256'].items() if '/' not in n and
              (n.endswith('.py') or n in ['whole_geometry.npz','config.json','baseline_contract.json',
                                          'continuous_energy_Sensi_full','transfer_training_summary.json'])}
    verify_files(old,selected);payload.mkdir(parents=True,exist_ok=False)
    for n in selected:shutil.copy2(old/n,payload/n)
    new=('derenzo_campaign_common.py','derenzo_campaign_reconstruction.py','derenzo_campaign_verify.py',
         'derenzo_campaign_gate.py','test_derenzo_campaign.py')
    for n in new:shutil.copy2(HERE/n,payload/n)
    original=read(old/'kernel_config.json');ehe=read(REPORT/'ehe_factor_reuse_metadata.json')
    cfg=dict(study=STUDY,datasets={},original_scientific_sources_sha256=selected,
             whole_geometry_sha256=selected['whole_geometry.npz'],cross_energy_background_enabled=False,
             physical_calibration_claim=False,grid_radial_spacing_mm=6,grid_z_spacing_mm=3)
    for dataset in DATASETS:
        info=dataset_info(dataset);energy=info['energy_keV']
        if info['system']=='JSCC':
            name=f'{energy}keV_RotateNum20'
            factor_files={n[len(name)+1:]:s for n,s in original['factor_payload_sha256'].items() if n.startswith(name+'/')}
            spec=dict(info,factor_root=GPU_PROJECT+'/generated/FactorsCalibrated/'+name,
                      factor_manifest_sha256=original['factor_manifest_sha256'][name],factor_files=factor_files)
        else:
            meta=ehe[f'A{energy}']
            if meta['manifest']['files']['whole_geometry.npz']!=selected['whole_geometry.npz']:raise ValueError('EHE and JSCC reconstruction geometry must match')
            spec=dict(info,factor_root=meta['path'],factor_manifest_sha256=meta['manifest_sha256'],
                factor_files={n:s for n,s in meta['manifest']['files'].items() if n!='SysMat_cartesian'})
        cfg['datasets'][dataset]=spec
    write(payload/'gpu_campaign.json',cfg)
    files=hashes(payload);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    write(payload/'release_manifest.json',dict(release_key=key,sha256=files))
    f=dict(study=STUDY,release_key=key,root=GPU_BASE+'/releases/'+key,sha256=hashes(payload),
           original_source_sha256=selected,new_source_sha256={n:digest(payload/n) for n in new},
           methods={d:dataset_info(d) for d in DATASETS},transport_freeze_sha256=digest(REPORT/'transport_freeze.json'))
    write(proof,f);return f


def deploy_gpu():
    path=REPORT/'gpu_deployment.json'
    if path.exists():return read(path)
    f=prepare_gpu()
    with connection('gpu') as c:
        policy=command(c,'scontrol show partition gpu_5090')
        if not all(x in policy for x in ['DefCpuPerGPU=8','DefMemPerCPU=15750','State=UP']):raise ValueError('Actual 5090 resource policy changed')
        put_payload(c,DATA/'gpu_payload',f['root'])
        check='from pathlib import Path;from derenzo_campaign_common import *;verify_files(Path("."),read("release_manifest.json")["sha256"]);import derenzo_campaign_reconstruction,derenzo_campaign_verify;print("DERENZO_LINUX_ENTRY_AND_SHA_PASS")'
        prefix='cd '+q(f['root'])+' && PYTHONDONTWRITEBYTECODE=1 JSCC_PROJECT_ROOT='+q(f['root'])+' '+q(GPU_PYTHON)+' -B '
        identity=command(c,prefix+'-c '+q(check),timeout=240)
        tests=command(c,prefix+'-m unittest test_derenzo_campaign.DerenzoCampaignTests.test_joint_branch_matches_original_all_saved_values test_jscc_5e10 -v 2>&1',timeout=240)
        (REPORT/'gpu_linux_tests.txt').write_text(identity+tests,encoding='utf-8')
        if '\nOK' not in tests or 'skipped' in tests:raise ValueError('Original scientific helpers/joint-history tests failed')
        command(c,'mkdir -p '+q(GPU_BASE+'/logs')+' '+q(GPU_BASE+'/scripts')+' '+q(GPU_BASE+'/allocations')+' '+q(GPU_BASE+'/plans'))
    d=dict(release=f['root'],release_manifest_sha256=digest(DATA/'gpu_payload/release_manifest.json'),
           linux_tests_sha256=digest(REPORT/'gpu_linux_tests.txt'),partition_policy=policy)
    write(path,d);return d


def deploy_input(dataset):
    path=REPORT/dataset/'input_deployment.json'
    if path.exists():return True
    accepted=read(REPORT/dataset/'transport_acceptance.json');target=GPU_BASE+'/'+dataset+'/input'
    source=Path(accepted['local_input']);collection=read(source/'collection.json')
    with connection('gpu') as c:
        free=int(command(c,'df -B1 --output=avail '+q(GPU_BASE)+' | tail -n1').strip())
        needed=read(REPORT/dataset/'transport_package.json')['archive_bytes']*4+(20<<30)
        if free<needed:raise OSError('Account quota lacks bounded input/output headroom')
        put_payload(c,source,target)
        check='from pathlib import Path;from derenzo_campaign_common import *;p=Path('+repr(target)+');c=validate_collection(read(p/"collection.json"),'+repr(dataset)+');verify_files(p,c["files"]);print("STRICT_GPU_ALL_INPUT_SHA_PASS")'
        release=read(REPORT/'gpu_deployment.json')['release']
        identity=command(c,'cd '+q(release)+' && '+q(GPU_PYTHON)+' -B -c '+q(check),timeout=600)
    write(path,dict(passed=True,input=target,collection_sha256=digest(source/'collection.json'),identity=identity,
          transport_acceptance_sha256=digest(REPORT/dataset/'transport_acceptance.json')))
    return True


def make_plan(dataset,stage):
    path=REPORT/dataset/(stage+'_plan.json')
    if path.exists():return read(path)
    info=dataset_info(dataset);gpu=read(REPORT/'gpu_deployment.json');inp=read(REPORT/dataset/'input_deployment.json')
    gpus=4 if info['system']=='JSCC' else 1;nodes=1;events=0
    if dataset=='JSCC_440' and stage!='selection':
        accepted=read(REPORT/dataset/'selection_acceptance.json');m=read(Path(accepted['local_result'])/'selection_manifest.json');events=m['accepted_events']
        for nodes in [1,2,4,8]:
            world=nodes*gpus
            ranks=[sum(n*(r+1)//world-n*r//world for n in m['events_per_view']) for r in range(world)]
            node_bytes=[sum(ranks[i:i+gpus])*78920*4+gpus*(16<<30) for i in range(0,world,gpus)]
            if max(node_bytes)<=.8*(gpus*8*15750*(1<<20)):break
        else:raise MemoryError('Full actual selected responses exceed bounded resident topology')
    plan=dict(study=STUDY,dataset=dataset,nodes=nodes,gpus_per_node=gpus,world_size=nodes*gpus,
              accepted_events=events,collection_sha256=inp['collection_sha256'],release_sha256=gpu['release_manifest_sha256'],
              channels=info['channels'],iterations=info['iterations'],save_step=info['save_step'],
              response_storage='CPU resident lossless float32; no disk response cache')
    write(path,plan);remote=GPU_BASE+'/plans/'+dataset+'_'+stage+'.json'
    with connection('gpu') as c:
        command(c,'test ! -e '+q(remote))
        with c.open_sftp() as s:s.put(str(path),remote)
        if command(c,'sha256sum '+q(remote)).split()[0]!=digest(path):raise ValueError('Plan transfer differs')
    write(REPORT/dataset/(stage+'_plan_deployment.json'),dict(remote=remote,sha256=digest(path)))
    return plan


def gpu_script(dataset,stage,plan,plan_remote,plan_sha,source=None):
    release=read(REPORT/'gpu_deployment.json')['release'];mode=stage.replace('_verify','');verify=stage.endswith('_verify')
    args=['--release',release,'--dataset',dataset,'--input',GPU_BASE+'/'+dataset+'/input','--plan',plan_remote]
    source_output=source['output'] if source else None
    output=(GPU_BASE+'/'+dataset+'/acceptance_'+mode+'_'+str(source['job']) if verify else GPU_BASE+'/'+dataset+'/'+stage)
    script='#!/usr/bin/env bash\nset -euo pipefail\nsource /etc/profile.d/modules.sh\nmodule load cuda/12.9\nmodule load miniforge3/25.11.0-1\n'
    script+='export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 JSCC_PROJECT_ROOT='+q(release)+'\n'
    if verify:
        args+=['--result',source_output,'--allocation',source['allocation'],'--output',output,'--mode',mode]
        script+='timeout --signal=TERM --kill-after=60s 10500s '+q(GPU_PYTHON)+' -B -u '+q(release+'/derenzo_campaign_verify.py')+' '+' '.join(q(x) for x in args)+'\n'
        return script,output,None
    allocation=GPU_BASE+'/allocations/'+dataset+'_'+stage+'_"${SLURM_JOB_ID}.txt"'
    script+='allocation='+allocation+'\nscontrol show job "$SLURM_JOB_ID" > "$allocation"\n'
    script+='export NCCL_SOCKET_IFNAME=bond0\nexport MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)\nexport MASTER_PORT=$((45000+SLURM_JOB_ID%10000))\n'
    args+=['--output',output,'--mode',mode,'--plan-sha256',plan_sha]
    if dataset=='JSCC_440' and stage!='selection':args+=['--selection',read(REPORT/dataset/'selection_job.json')['output']]
    if stage=='formal':
        authority=GPU_BASE+'/'+dataset+'/authority.json';args+=['--authority',authority]
        gate=q(GPU_PYTHON)+' -B '+q(release+'/derenzo_campaign_gate.py')+' --authority '+q(authority)+' --plan '+q(plan_remote)
        script+='gate_exports=$('+gate+')\neval "$gate_exports"\n'
    else:script+='export DERENZO_PHASE_SECONDS=7200 DERENZO_PREPARE_SECONDS=10800 DERENZO_TOTAL_SECONDS=14220\n'
    invocation=' '.join(q(x) for x in args)+' --allocation "$allocation"'
    if stage=='formal':invocation+=' --authority-sha256 "$DERENZO_AUTHORITY_SHA256"'
    script+='timeout --signal=TERM --kill-after=60s "${DERENZO_TOTAL_SECONDS}s" srun --label --chdir=/tmp --kill-on-bad-exit=1 bash -c \'exec '+q(GPU_PYTHON)+' -B -m torch.distributed.run --nnodes='+str(plan['nodes'])+' --nproc_per_node='+str(plan['gpus_per_node'])+' --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --rdzv_backend=static --rdzv_id="${SLURM_JOB_ID}_derenzo" --max_restarts=0 "${@:1}"\' bash '+q(release+'/derenzo_campaign_reconstruction.py')+' '+invocation+'\n'
    return script,output,allocation


def submit_gpu(dataset,stage,plan_stage,dependency=None,source=None):
    path=REPORT/dataset/(stage+'_job.json')
    if path.exists():return read(path)
    intent=REPORT/dataset/(stage+'_submission_intent.json')
    if intent.exists():raise RuntimeError('Unresolved GPU submission intent; do not repeat sbatch')
    plan=make_plan(dataset,plan_stage);pd=read(REPORT/dataset/(plan_stage+'_plan_deployment.json'))
    script,output,_=gpu_script(dataset,stage,plan,pd['remote'],pd['sha256'],source)
    local=DATA/'launch_scripts'/(dataset+'_GPU_'+stage+'.sh')
    if local.exists() and local.read_bytes()!=script.encode():raise ValueError('Preserve different launch script')
    local.write_bytes(script.encode());remote=GPU_BASE+'/scripts/'+local.name
    verify=stage.endswith('_verify');nodes=1 if verify else plan['nodes'];gpus=1 if verify else plan['gpus_per_node']
    minutes=180 if verify else (2880 if stage=='formal' else 240)
    with connection('gpu') as c:
        queue=command(c,"squeue -h -u scxi717 -o '%i|%j|%T'",timeout=90)
        if len(queue.splitlines())>=50:return None
        command(c,'test ! -e '+q(remote))
        with c.open_sftp() as s:s.put(str(local),remote)
        if command(c,'sha256sum '+q(remote)).split()[0]!=digest(local):raise ValueError('GPU launch transfer differs')
        command(c,'bash -n '+q(remote))
        options='-p gpu_5090 --qos=gpugpu -N'+str(nodes)+' -n'+str(nodes)+' --ntasks-per-node=1 --cpus-per-task='+str(gpus*8)+' --gres=gpu:'+str(gpus)+' --time='+str(minutes)+' --exclude=wqd10nba06g6 --chdir=/tmp'
        if dependency:options+=' --dependency=afterok:'+str(dependency)
        if stage=='formal':options+=' --hold'
        options+=' --job-name=DR5e9_'+dataset+'_'+stage+' --output='+q(GPU_BASE+'/logs/'+dataset+'_'+stage+'.%j.out')+' --error='+q(GPU_BASE+'/logs/'+dataset+'_'+stage+'.%j.err')+' '+q(remote)
        write(intent,dict(script_sha256=digest(local),command='sbatch --parsable '+options,resolved=False,started_epoch=time.time()))
        number=command(c,'sbatch --parsable '+options,timeout=60).split(';')[0].strip()
        if not number.isdigit():raise RuntimeError('Ambiguous GPU submission; preserve intent')
        record=dict(job=int(number),stage=stage,dataset=dataset,host='gpu',output=output,
                    allocation=GPU_BASE+'/allocations/'+dataset+'_'+stage+'_'+number+'.txt',plan_stage=plan_stage,
                    plan=plan,plan_remote=pd['remote'],plan_sha256=pd['sha256'],script_sha256=digest(local),
                    nodes=nodes,gpus_per_node=gpus,dependency=dependency,held=stage=='formal',walltime_minutes=minutes,
                    submitted_epoch=time.time())
        write(path,record);write(intent,dict(read(intent),resolved=True,registered_job=int(number)))
        snapshot=command(c,'scontrol show job -o '+number,timeout=90)
        (REPORT/dataset/(stage+'_submitted_scontrol.txt')).write_text(snapshot,encoding='utf-8')
        fields=dict(x.split('=',1) for x in snapshot.split() if '=' in x)
        if fields.get('NumNodes') not in [str(nodes),f'{nodes}-{nodes}'] or fields.get('TresPerNode')!='gres/gpu:'+str(gpus):raise ValueError('Actual submitted GPU topology differs')
    print('REGISTERED_GPU',dataset,stage,number,flush=True);return record


def fetch_gpu(dataset,stage):
    proof_path=REPORT/dataset/(stage+'_acceptance.json')
    if proof_path.exists():return True
    source=read(REPORT/dataset/(stage+'_job.json'));verifier=read(REPORT/dataset/(stage+'_verify_job.json'))
    with connection('gpu') as c:
        done,acc=source_complete(c,source)
        if not done:return False
        done,vacc=source_complete(c,verifier)
        if not done:return False
        package=metadata(c,verifier['output']+'/package.json',REPORT/dataset/(stage+'_package.json'))
        archive=DATA/dataset/(stage+'_accepted.tar.gz')
        download(c,verifier['output']+'/accepted_result.tar.gz',archive,package['archive_sha256'])
        target=DATA/dataset/(stage+'_accepted');extract_archive(archive,target,package['files'])
        proof=read(target/'verification.json')
        if not proof['passed'] or proof['dataset']!=dataset or proof['mode']!=stage:raise ValueError('Strict result acceptance differs')
        source_acc=command(c,'sacct -j '+str(source['job'])+' -n -P --format=JobID,State,ExitCode,MaxRSS,Elapsed,AllocTRES',timeout=90)
        from jscc_5e10_reconstruction_workflow import slurm_peak_bytes
        from jscc_5e10_common import host_allocated_bytes
        memory=host_allocated_bytes((target/'allocation.txt').read_text(),source['plan']['nodes'])
        if slurm_peak_bytes(source_acc)>.8*memory:raise MemoryError('Actual exited Slurm MaxRSS fails margin')
        logs=REPORT/dataset/(stage+'_logs');logs.mkdir(exist_ok=True)
        for record in [source,verifier]:
            for ext in ['out','err']:
                remote=GPU_BASE+'/logs/'+dataset+'_'+record['stage']+'.'+str(record['job'])+'.'+ext
                expected=command(c,'sha256sum '+q(remote)).split()[0]
                download(c,remote,logs/(record['stage']+'.'+ext),expected)
    write(proof_path,dict(passed=True,strict_fetch_passed=True,dataset=dataset,stage=stage,source_job=source['job'],
          verification_job=verifier['job'],local_result=str(target.resolve()),remote_result=source['output'],
          archive_sha256=package['archive_sha256'],all_files_sha256=package['files'],source_accounting=source_acc,
          verification_accounting=vacc,verification_sha256=digest(target/'verification.json'),logs_sha256=hashes(logs)))
    print('DERENZO_STRICT_GPU_STAGE_ACCEPTED',dataset,stage,flush=True);return True


def release_formal(dataset):
    record_path=REPORT/dataset/'formal_release.json'
    if record_path.exists():return True
    accepted=read(REPORT/dataset/'validation_acceptance.json');proof=read(Path(accepted['local_result'])/'verification.json')
    formal=read(REPORT/dataset/'formal_job.json');source=read(REPORT/dataset/'validation_job.json')
    # Import the pure timing estimator without importing GPU-only modules locally.
    from derenzo_campaign_gate import limits_from_validation
    limits=limits_from_validation(proof)
    authority=dict(passed=True,strict_fetch_passed=True,study=STUDY,dataset=dataset,formal_job=formal['job'],
         plan_sha256=formal['plan_sha256'],collection_sha256=proof['collection_sha256'],release_sha256=proof['release_sha256'],
         validation_result=source['output'],validation_files_sha256=proof['result_files_sha256'],
         validation_job=source['job'],strict_acceptance_sha256=digest(REPORT/dataset/'validation_acceptance.json'),limits=limits)
    path=DATA/dataset/'authority.json'
    if path.exists() and read(path)!=authority:raise ValueError('Preserve different formal authority')
    write(path,authority);remote=GPU_BASE+'/'+dataset+'/authority.json'
    with connection('gpu') as c:
        snapshot=command(c,'scontrol show job -o '+str(formal['job']),timeout=90)
        if 'JobState=PENDING' not in snapshot or 'JobHeldUser' not in snapshot:raise RuntimeError('Formal job no longer in expected held state; inspect')
        exists=command(c,'if test -e '+q(remote)+'; then sha256sum '+q(remote)+'; fi').strip()
        if exists:
            if exists.split()[0]!=digest(path):raise ValueError('Remote authority differs')
        else:
            with c.open_sftp() as s:s.put(str(path),remote)
        if command(c,'sha256sum '+q(remote)).split()[0]!=digest(path):raise ValueError('Authority transfer differs')
        command(c,'scontrol update JobId='+str(formal['job'])+' TimeLimit='+str(limits['minutes']),timeout=90)
        checked=command(c,'scontrol show job -o '+str(formal['job']),timeout=90)
        fields=dict(x.split('=',1) for x in checked.split() if '=' in x)
        days,_,clock=fields['TimeLimit'].rpartition('-');h,m,s=map(int,clock.split(':'))
        if (int(days or 0)*24+h)*3600+m*60+s!=limits['minutes']*60:raise ValueError('Measured time limit did not apply')
        write(REPORT/dataset/'formal_release_intent.json',dict(job=formal['job'],authority_sha256=digest(path),limits=limits))
        command(c,'scontrol release '+str(formal['job']),timeout=90)
        after=command(c,'scontrol show job -o '+str(formal['job']),timeout=90)
    write(record_path,dict(passed=True,job=formal['job'],authority_sha256=digest(path),limits=limits,scontrol=after,released_epoch=time.time()))
    print('DERENZO_FORMAL_RELEASED',dataset,formal['job'],flush=True);return True


def advance_dataset(dataset):
    if (REPORT/dataset/'formal_acceptance.json').exists():return 'complete'
    if not fetch_transport(dataset):return 'waiting_transport'
    deploy_input(dataset)
    if dataset=='JSCC_440':
        selection=submit_gpu(dataset,'selection','selection')
        if selection is None:return 'waiting_queue_capacity'
        verifier=submit_gpu(dataset,'selection_verify','selection',selection['job'],selection)
        if verifier is None or not fetch_gpu(dataset,'selection'):return 'waiting_selection_acceptance'
    validation=submit_gpu(dataset,'validation','validation')
    if validation is None:return 'waiting_queue_capacity'
    verifier=submit_gpu(dataset,'validation_verify','validation',validation['job'],validation)
    if verifier is None:return 'waiting_queue_capacity'
    formal=submit_gpu(dataset,'formal','validation',verifier['job'])
    if formal is None:return 'waiting_queue_capacity'
    if not fetch_gpu(dataset,'validation'):return 'formal_queued_waiting_validation'
    release_formal(dataset)
    verifier=submit_gpu(dataset,'formal_verify','validation',formal['job'],formal)
    if verifier is None or not fetch_gpu(dataset,'formal'):return 'formal_submitted_waiting_complete_acceptance'
    return 'complete'


def advance():
    submit_production_transport();deploy_gpu();states={}
    for dataset in DATASETS:
        if not (REPORT/dataset/'collection_job.json').exists():states[dataset]='waiting_pilot';continue
        blocked=REPORT/dataset/'controller_failure.json'
        if blocked.exists():states[dataset]='needs_attention';continue
        try:states[dataset]=advance_dataset(dataset)
        except (TimeoutError,ConnectionError) as error:
            states[dataset]='waiting_connection'
            write(REPORT/dataset/'connection_wait.json',dict(checked_epoch=time.time(),error=str(error),no_job_resubmission=True))
        except Exception as error:
            write(blocked,dict(checked_epoch=time.time(),error_type=type(error).__name__,error=str(error),
                  preserve_all_jobs_and_outputs=True,automatic_scientific_resubmission=False))
            states[dataset]='needs_attention'
    write(REPORT/'latest_progress.json',dict(checked_epoch=time.time(),states=states,recurring_automations_paused=True))
    print(json.dumps(states),flush=True)
    if all(x in ['complete','needs_attention'] for x in states.values()) and any(x=='needs_attention' for x in states.values()):
        raise RuntimeError('Campaign needs attention; independent completed outputs and all failure evidence preserved')
    return all(x=='complete' for x in states.values())


def watch():
    deadline=time.monotonic()+21*86400
    while time.monotonic()<deadline:
        if advance():return
        time.sleep(300)
    raise TimeoutError('One-shot campaign controller reached its bounded deadline; preserve all jobs and outputs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pilots','transport','advance','watch','deploy-gpu'])
    a=p.parse_args()
    with controlled():
        if a.action=='pilots':submit_pilots()
        elif a.action=='transport':submit_production_transport()
        elif a.action=='deploy-gpu':deploy_gpu()
        elif a.action=='advance':advance()
        else:watch()
