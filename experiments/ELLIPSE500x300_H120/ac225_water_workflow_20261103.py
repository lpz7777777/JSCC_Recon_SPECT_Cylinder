"""Independent formal-only water NEMA correction study; explicit bounded actions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from ac225_corrections.io import digest,read,write,hashes,verify,immutable
from ac225_corrections.material import water_mu_mm_inv,NIST_ROWS,NIST_URL
from ac225_corrections.windows import energy_windows
from ac225_corrections.geant4_overlay import overlay
from campaign_20261102_io import execute,upload_files,python_json,sftp
from derenzo_campaign_common import HERE,ROOT,CPU_PROJECT,GPU_PROJECT,GPU_PYTHON
from ehe_5e9_workflow import q

TAG='ac225_water_corrections_20261103'
DATA=HERE/'generated'/TAG
REPORT=HERE/'reports/NEMA_Body_H60'/TAG
CPU=CPU_PROJECT+'/generated/'+TAG
GPU=GPU_PROJECT+'/generated/'+TAG
BRANCHES=('none','attenuation','attenuation_tew_raw','attenuation_tew_pool3')


def prepare():
    marker=REPORT/'freeze.json';payload=DATA/'payload'
    if marker.exists():
        f=read(marker);verify(payload,f['files']);return f
    DATA.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    if payload.exists():raise FileExistsError('Incomplete freeze retained; inspect before new freeze')
    payload.mkdir()
    package=HERE/'ac225_corrections'
    for n,p in hashes(package).items():
        target=payload/'ac225_corrections'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(package/n,target)
    for n in ('ehe_common.py','jscc_5e10_array.py','torch_active_operator.py','single_checkpoint_mlem.py','test_ac225_corrections.py','nema_body_h60_config.json'):
        shutil.copyfile(HERE/n,payload/n)
    old=HERE/'generated/jscc_geant4_5e10_10000/source_registry'
    expected=read(HERE/'reports/NEMA_Body_H60/jscc_geant4_5e10_10000/transport_array_freeze.json')['source_registry_sha256']
    verify(old,expected)
    (payload/'macros').mkdir()
    for v in range(1,21):shutil.copyfile(old/'macros'/f'NEMA_Body_H60_v{v:02d}.mac',payload/'macros'/f'v{v:02d}.mac')
    geometry=HERE/'generated/jscc_geant4_5e10_10000/production_payload/whole_geometry.npz'
    shutil.copyfile(geometry,payload/'whole_geometry.npz')
    parents={'JSCC':ROOT/'Geant4Sim/Geant4Code',
             'EHE':HERE/'generated/ehe_spect_5e9_200/transport_payload_v4/Geant4Code_EHE'}
    source_before={}
    for system,parent in parents.items():
        source_before[system]={n:h for n,h in hashes(parent).items() if n.startswith(('src/','include/')) or n in ('CMakeLists.txt','gamma01.cc','ehe_spect.cc','CrystalMatrix.txt')}
        overlay(parent,payload/system,system)
    acquisitions={};conditions={};(payload/'jobs').mkdir()
    for s,system in enumerate(('JSCC','EHE')):
        for k,(dose,total,workers) in enumerate((('5e9',5_000_000_000,20),('5e10',50_000_000_000,200))):
            dataset=system+'_'+dose;seed=80100101+s*100000+k*10000
            jobs=[dict(index=i,view=i//(workers//20)+1,worker=i%(workers//20),
                       photons=total//workers,seed=seed+i) for i in range(workers)]
            assert sum(j['photons'] for j in jobs)==total
            write(payload/'jobs'/f'{dataset}.json',dict(dataset=dataset,jobs=jobs))
            acquisitions[dataset]=dict(system=system,dose=dose,total_gamma=total,workers=workers,
                bins=10496 if system=='JSCC' else 2312,seed_first=seed,seed_last=seed+workers-1)
            for branch in BRANCHES:
                for energy in (440,218):
                    name=f'{dataset}_{branch}_{energy}'
                    conditions[name]=dict(dataset=dataset,branch=branch,energy_keV=energy,
                        iterations=10000 if system=='JSCC' else 200,save_step=50 if system=='JSCC' else 1,
                        final440_from_same_branch=(name.rsplit('_',1)[0]+'_440') if energy==218 else None)
    contract=dict(study=TAG,acquisitions=acquisitions,conditions=conditions,views=20,
        phantom_material='uniform_water_inner_cavity_H60',density_g_cm3=1,
        source_macros_sha256=hashes(payload/'macros'),source_parent_registry_sha256=digest(old/'jobs.json'),
        Geant4_parent_source_sha256=source_before,whole_geometry_sha256=digest(geometry),
        water_mu_mm_inv={str(e):water_mu_mm_inv(e) for e in (218,440)},
        mu_provenance=dict(url=NIST_URL,quantity='total mu/rho',table_rows_keV_cm2_g=NIST_ROWS,
                           interpolation='log-log',cm_inv_to_mm_inv_factor=.1),
        windows_keV=energy_windows(),initial_density=1,regularization='none',image_smoothing=False,
        cross_energy_image_sum=False,scatter_warmup_iterations=20,scatter_update_interval=20,
        matched_S_required=True,center_ray_threshold=.02,physical_calibration_passed=False,
        EHE_original_C_known_bias_held=True,separate_validation_job=False,
        resources=dict(GPUs_per_route=1,CPUs_per_route=8,maximum_concurrent_GPU_routes=4,maximum_fraction=.8),
        authorization='2026-11-03 human explicitly requested implementation and new actual water acquisitions/reconstructions')
    write(payload/'contract.json',contract)
    files=hashes(payload);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    write(payload/'release_manifest.json',dict(key=key,files=files))
    f=dict(key=key,files=hashes(payload),CPU_release=CPU+'/releases/'+key,GPU_release=GPU+'/releases/'+key)
    write(marker,f);return f


def deploy_cpu():
    f=prepare();marker=REPORT/'cpu_deployment.json'
    if marker.exists():return read(marker)
    # Hash on the remote filesystem instead of re-reading large macros via SFTP.
    check='''import sys,json,pathlib,hashlib
c=json.loads(sys.argv[1]);r=pathlib.Path(c['release']);missing=[]
for n,h in c['files'].items():
 p=r/n
 if not p.exists():missing.append(n);continue
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,n
print('CAMPAIGN_JSON '+json.dumps(missing))
'''
    missing=python_json('maty','python3',check,dict(release=f['CPU_release'],files=f['files']))
    if missing:upload_files('maty',f['CPU_release'],{n:DATA/'payload'/n for n in missing})
    # Build is preparation on the login CPU; no separate Slurm test job.
    script='''set -euo pipefail
source /etc/profile.d/modules.sh
module load compilers/gcc/v12.2.0
module load tools/cmake/v3.25.2
source /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh
export Geant4_DIR=/apps/soft/geant/geant4-v11.1.0/lib64/cmake/Geant4
export CC=/apps/compilers/gcc/v12.2.0/bin/gcc CXX=/apps/compilers/gcc/v12.2.0/bin/g++
for system in JSCC EHE; do
 cmake -S "$1/$system" -B "$1/build_gcc12_$system" -DWITH_GEANT4_UIVIS=OFF -DCMAKE_BUILD_TYPE=Release
 cmake --build "$1/build_gcc12_$system" --parallel 4
done
'''
    local=DATA/'build_gcc12.sh';immutable(local,script.encode());upload_files('maty',CPU,{'build_gcc12.sh':local})
    execute('maty','bash '+q(CPU+'/build_gcc12.sh')+' '+q(f['CPU_release'])+' > '+q(f['CPU_release']+'/build_gcc12.log')+' 2>&1',timeout=600)
    code='''import sys,pathlib,json,hashlib
r=pathlib.Path(json.loads(sys.argv[1])['release']);sys.path.insert(0,str(r))
from ac225_corrections.io import verify,read,write,digest
verify(r,read(r/'release_manifest.json')['files'])
b={s:dict(path=str(r/('build_gcc12_'+s)/name),sha256=digest(r/('build_gcc12_'+s)/name)) for s,name in [('JSCC','gamma01'),('EHE','ehe_spect')]}
write(r/'binary_manifest.json',b)
print('CAMPAIGN_JSON '+json.dumps(b))
'''
    binaries=python_json('maty','python3',code,dict(release=f['CPU_release']),readonly=False)
    result=dict(passed=True,release=f['CPU_release'],binaries=binaries,freeze_key=f['key'])
    write(marker,result);return result


def submit_registered(host,label,body,options,output,remote_root):
    path=REPORT/'jobs'/(label+'.json');intent=path.with_name(label+'_submission_intent.json')
    if path.exists():return read(path)
    if intent.exists():raise RuntimeError('Unresolved scheduler intent; inspect Slurm: '+label)
    local=DATA/'scripts'/(label+'.sh');immutable(local,body.encode())
    upload_files(host,remote_root+'/scripts',{local.name:local})
    execute(host,'bash -n '+q(remote_root+'/scripts/'+local.name),readonly=True)
    queue=execute(host,'squeue -h -u "$USER" -o "%i|%j|%T"',readonly=True)
    if any(line.split('|')[1]==label for line in queue.splitlines() if '|' in line):raise RuntimeError('Duplicate active formal name')
    cmd='sbatch --parsable '+options+' --job-name='+label+' --output='+q(remote_root+'/logs/'+label+'_%A_%a.out')+' --error='+q(remote_root+'/logs/'+label+'_%A_%a.err')+' '+q(remote_root+'/scripts/'+local.name)
    request=dict(command=cmd,output=output,script_sha256=digest(local),host=host,submitted_epoch=time.time())
    write(intent,request)
    raw=execute(host,cmd,timeout=60).strip();number=raw.split(';')[0]
    if not number.isdigit():raise ValueError('Uncertain scheduler result retained')
    request.update(job=int(number),sbatch_result=raw);write(path,request);return request


def submit_transport():
    f=prepare();deploy_cpu()
    proof=read(REPORT/'cpu_preparation_checks.json')
    if not proof['passed'] or proof['freeze_key']!=f['key']:raise ValueError('Compiled local Linux checks required')
    execute('maty','mkdir -p '+q(CPU+'/logs'),readonly=False)
    code='''import sys,json,os
p=json.loads(sys.argv[1])['path'];s=os.statvfs(p)
print('CAMPAIGN_JSON '+json.dumps(dict(available=s.f_bavail*s.f_frsize,total=s.f_blocks*s.f_frsize,writable=os.access(p,os.W_OK))))
'''
    quota=python_json('maty','python3',code,dict(path=CPU))
    if not quota['writable'] or quota['available']<100*(1<<30):raise OSError('CPU actual directory headroom insufficient')
    env='#!/usr/bin/env bash\nset -euo pipefail\nsource /etc/profile.d/modules.sh\nmodule load compilers/gcc/v12.2.0\nsource /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\ncd '+q(f['CPU_release'])+'\n'
    result={}
    for dataset,spec in read(DATA/'payload/contract.json')['acquisitions'].items():
        out=CPU+'/'+dataset+'/transport'
        body=env+'python3 -B -m ac225_corrections.transport worker --release '+q(f['CPU_release'])+' --dataset '+dataset+' --output '+q(out)+'/worker_$(printf "%04d" "$SLURM_ARRAY_TASK_ID") --index "$SLURM_ARRAY_TASK_ID"\n'
        opts=f"-p cnmix -N1 -n1 -c1 --time=1440 --array=0-{spec['workers']-1}%64"
        j=submit_registered('maty','ACW_'+dataset,body,opts,out,CPU)
        body=env+'python3 -B -m ac225_corrections.transport collect --release '+q(f['CPU_release'])+' --dataset '+dataset+' --root '+q(out)+' --output '+q(CPU+'/'+dataset+'/collection')+' --parent '+str(j['job'])+'\n'
        collect=submit_registered('maty','ACW_'+dataset+'_collect',body,'-p cnmix -N1 -n1 -c1 --time=180 --dependency=afterok:'+str(j['job']),CPU+'/'+dataset+'/collection',CPU)
        result[dataset]=dict(transport=j,collect=collect)
    write(REPORT/'transport_submitted.json',result);return result


def cpu_checks():
    f=prepare();deploy_cpu()
    # This preparation module was added after transport freeze: independently
    # hash-bound supplement, leaving the original freeze/source bytes intact.
    module=HERE/'ac225_corrections/cpu_checks.py'
    upload_files('maty',CPU+'/preparation',{'cpu_checks.py':module})
    old=read(HERE/'generated/derenzo_equal_regions_mono_5e9_20261016/transport_payload/campaign.json')['datasets']
    parents={s:dict(path=old[s+'_440']['binary_path'],sha256=old[s+'_440']['binary_sha256']) for s in ('JSCC','EHE')}
    code='''import sys,json,pathlib,importlib.util
c=json.loads(sys.argv[1]);r=pathlib.Path(c['release']);sys.path.insert(0,str(r))
from ac225_corrections.io import digest
assert digest(c['module'])==c['module_sha256']
spec=importlib.util.spec_from_file_location('ac225_corrections.cpu_checks',c['module'])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
result=m.run(r,c['parents'],pathlib.Path(c['output']))
print('CAMPAIGN_JSON '+json.dumps(dict(passed=result['passed'],checks_sha256=digest(pathlib.Path(c['output'])/'checks.json'))))
'''
    args=dict(release=f['CPU_release'],parents=parents,module=CPU+'/preparation/cpu_checks.py',module_sha256=digest(module),output=CPU+'/preparation/checks_'+f['key'])
    env='source /etc/profile.d/modules.sh\nmodule load compilers/gcc/v12.2.0\nsource /apps/soft/geant/geant4-v11.1.0/bin/geant4.sh\n'
    cmd='python3 -B -c '+q(code)+' '+q(json.dumps(args))
    raw=execute('maty','bash -c '+q(env+cmd),timeout=600)
    result=json.loads(raw.rsplit('CAMPAIGN_JSON ',1)[1]);result.update(freeze_key=f['key'],module_sha256=digest(module),parents=parents,output=args['output'])
    write(REPORT/'cpu_preparation_checks.json',result);return result


def status():
    result={}
    for host in ('maty','gpu'):
        ids=[str(read(p)['job']) for p in (REPORT/'jobs').glob('*.json') if not p.name.endswith('_submission_intent.json') and read(p)['host']==host]
        if ids:
            result[host]=dict(queue=execute(host,'squeue -h -r -j '+','.join(ids)+' -o "%i|%j|%T|%M|%R"',readonly=True),
                accounting=execute(host,'sacct -n -P -j '+','.join(ids)+' -o JobID,State,ExitCode,Elapsed,AllocTRES',readonly=True))
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','deploy_cpu','cpu_checks','submit_transport','status']);a=p.parse_args()
    result=globals()[a.action]()
    if a.action!='status':print(json.dumps(result,ensure_ascii=False,indent=2))
