"""One isolated formal array, four synthetic doses; no timer or validation job."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import time
from campaign_20261102_io import execute,python_json,upload_files,sftp,mkdirs
from ehe_5e9_workflow import q
from derenzo_campaign_common import HERE,GPU_PROJECT,GPU_PYTHON,read,write,digest,hashes,verify_files
from ehe_60view_operator_20261102 import local_checks

TAG='ehe_dense_nema_60view_poisson_20261102'
DATA=HERE/'generated'/TAG;REPORT=HERE/'reports/EHE'/TAG;GPU=GPU_PROJECT+'/generated/'+TAG


def prepare():
    REPORT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    if (REPORT/'freeze.json').exists():
        f=read(REPORT/'freeze.json');verify_files(DATA/'payload',f['files']);return f
    parent=HERE/'generated/ehe_dense_nema_low_noise_20261102/payload'
    pf=read(parent/'release_manifest.json');verify_files(parent,pf['sha256'])
    pc=read(parent/'contract.json');sm=read(parent/'source_native_manifest.json')
    check=local_checks(parent/'whole_geometry.npz');write(REPORT/'local_operator_regression.json',check)
    payload=DATA/'payload';payload.mkdir(exist_ok=False)
    for name in ['ehe_common.py','torch_active_operator.py','single_checkpoint_mlem.py','ehe_dense_rerun_runtime_20261102.py',
                 'whole_geometry.npz','source_native.npz','source_native_manifest.json']:
        shutil.copyfile(parent/name,payload/name)
    for name in ['ehe_nema_60view_runtime_20261102.py','ehe_60view_operator_20261102.py']:
        shutil.copyfile(HERE/name,payload/name)
    conditions={}
    for index,dose in enumerate([5_000_000_000,50_000_000_000,500_000_000_000,5_000_000_000_000]):
        conditions[str(index)]=dict(index=index,dose_label=['5e9','5e10','5e11','5e12'][index],expected_emitted_photons=dose,
            noise_seeds={n:20261102601+index*10+j for j,n in enumerate(['A440','C440to218','A218'])})
    c=dict(study=TAG,conditions=conditions,factors=pc['factors'],gamma_yields={'218':.114,'440':.259},
        physical_FOV_mm=[500,300,120],views=60,angles_deg=list(range(0,360,6)),per_view_emission_fraction=1/60,detector_bins=2312,full_points=132040,active_points=78920,
        iterations=200,save_step=1,initial_density=1,regularization='none',cross_energy_image_sum=False,
        generator='Same frozen native volume-weighted operator and original3mm truth projected into finite-volume basis',
        source_native_sha256=sm['source_native_sha256'],source_native_manifest_sha256=digest(payload/'source_native_manifest.json'),
        dose_definition='Total expected emittedgamma budget, notAc225decays or detected counts; originalyields/sourceintegrals split energies',
        background218='This dose own final440200 forwarded through originalC; generatedtrueC onlydiagnostic',
        noise='Independent PCG64 Poisson percomponent/view/bin, preregisteredunique seeds',
        data_kind='matched_native_matrix_forward_plus_independent_Poisson',transport_performed=False,
        physical_calibration_passed=False,separate_validation_job=False,full_input_prevalidation_passed=False,
        angular_interpolation='Same-ring periodic linear response interpolation; exact original20 positions at views0,3,...57; matching60-viewS',
        original20_source_release_sha256=digest(parent/'release_manifest.json'),
        authorization='2026-11-02 user requests60angles NEMA5e9/5e10/5e11/5e12 usingforward+Poisson')
    write(payload/'contract.json',c)
    for p in payload.glob('*.py'):compile(p.read_text('utf8'),str(p),'exec')
    manifest=dict(sha256=hashes(payload));key=hashlib.sha256(str(sorted(manifest['sha256'].items())).encode()).hexdigest()[:16]
    manifest['release_key']=key;write(payload/'release_manifest.json',manifest)
    f=dict(key=key,release=GPU+'/releases/'+key,files=hashes(payload),parent_dense_release_sha256=digest(parent/'release_manifest.json'),
        helpers_unchanged={n:digest(payload/n) for n in ['torch_active_operator.py','single_checkpoint_mlem.py','ehe_dense_rerun_runtime_20261102.py']},
        source_native_manifest_sha256=digest(payload/'source_native_manifest.json'),
        original_MLEM_unchanged=True,angular_operator_changed=True,no_actual_Geant4_observation_used=True)
    write(REPORT/'freeze.json',f);return f


def deploy():
    f=prepare()
    if (REPORT/'deployment.json').exists():return f
    upload_files('gpu',f['release'],{n:DATA/'payload'/n for n in f['files']})
    code='''import sys,json,pathlib,shutil
c=json.loads(sys.argv[1]);r=pathlib.Path(c['release']);sys.path.insert(0,str(r))
from ehe_common import *
verify_files(r,read(r/'release_manifest.json')['sha256']);cfg=read(r/'contract.json')
from ehe_60view_operator_20261102 import local_checks
operator_check=local_checks(r/'whole_geometry.npz')
for name,s in cfg['factors'].items():
 p=pathlib.Path(s['root']);assert digest(p/'factor_manifest.json')==s['manifest_sha256']
 assert digest(p/'whole_geometry.npz')==s['geometry_sha256']==digest(r/'whole_geometry.npz')
 assert (p/'SysMat_polar').stat().st_size==132040*2312*4
root=pathlib.Path(c['remote']);(root/'logs').mkdir(parents=True,exist_ok=True);(root/'scripts').mkdir(exist_ok=True)
assert shutil.disk_usage(root).free>8*(1<<30)
print('CAMPAIGN_JSON '+json.dumps(dict(passed=True,release_SHA_passed=True,matrices_already_accepted=True,operator_regression=operator_check,
 full_input_prevalidation_passed=False,free=shutil.disk_usage(root).free,no_validation_job=True)))
'''
    proof=python_json('gpu',GPU_PYTHON,code,dict(f,remote=GPU),timeout=180,readonly=False)
    write(REPORT/'deployment.json',proof);return f


def submit():
    if (REPORT/'formal_job.json').exists():return read(REPORT/'formal_job.json')
    if (REPORT/'submission_intent.json').exists():raise RuntimeError('Unresolved intent: inspect Slurm before any resubmit')
    f=deploy();body='#!/usr/bin/env bash\nset -euo pipefail\nsource /etc/profile.d/modules.sh\nmodule load cuda/12.9\n'
    body+='export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1\n'
    body+=q(GPU_PYTHON)+' -B '+q(f['release']+'/ehe_nema_60view_runtime_20261102.py')+' --release '+q(f['release'])+' --index "$SLURM_ARRAY_TASK_ID" --output '+q(GPU)+'/results/condition_${SLURM_ARRAY_TASK_ID}\n'
    script=DATA/'formal.sh';script.write_text(body,encoding='utf8',newline='\n');upload_files('gpu',GPU+'/scripts',{'formal.sh':script})
    execute('gpu','bash -n '+q(GPU+'/scripts/formal.sh'),readonly=True)
    options='-p gpu_4090 --qos=gpugpu -N1 -n1 -c8 --gres=gpu:1 --time=180 --array=0-3%4'
    command='sbatch --parsable '+options+' --job-name=EHE_NEMA60_Pois4 --output='+q(GPU+'/logs/formal_%A_%a.out')+' --error='+q(GPU+'/logs/formal_%A_%a.err')+' '+q(GPU+'/scripts/formal.sh')
    request=dict(command=command,script_sha256=digest(script),release=f['release'],output=GPU+'/results',submitted_epoch=time.time(),
        scope='Onlythisnewsyntheticstudy; preserves otheractivejobs andpriorcompletedstudies')
    write(REPORT/'submission_intent.json',request);raw=execute('gpu',command,timeout=60,readonly=False).strip();number=raw.split(';')[0]
    if not number.isdigit():raise RuntimeError('Uncertain sbatch; retain intent')
    request.update(job=int(number),sbatch_result=raw);write(REPORT/'formal_job.json',request)
    print('FORMAL_FOUR_DOSE_ARRAY',number,flush=True);return request


def status():
    j=read(REPORT/'formal_job.json')
    code='''import sys,json,pathlib,subprocess,time
c=json.loads(sys.argv[1]);out=dict(checked_epoch=time.time(),conditions={})
out['queue']=subprocess.check_output(['squeue','-h','-r','-j',str(c['job']),'-o','%i|%T|%M|%R'],text=True)
out['accounting']=subprocess.check_output(['sacct','-j',str(c['job']),'-nP','-o','JobID%40,State,ExitCode,Elapsed,Start,End,MaxRSS,AllocTRES'],text=True)
for i in range(4):
 p=pathlib.Path(c['output'])/f'condition_{i}';d={}
 for n in ['progress.json','completion.json','failure.json','run_manifest.json','collection.json']:
  f=p/n
  if f.exists():
   v=json.loads(f.read_text());d[n]=({k:v[k] for k in ['passed','job','completed_epoch','phase_solve_seconds','window_counts','expected_emitted_photons'] if k in v} if n in ['completion.json','run_manifest.json','collection.json'] else v)
 out['conditions'][str(i)]=d
print('CAMPAIGN_JSON '+json.dumps(out))
'''
    s=python_json('gpu',GPU_PYTHON,code,j,timeout=180)
    # The one-shot reader and an interactive read may finish together. Give
    # each status writer its own staging path; scientific output is untouched.
    staging=REPORT/f'.status_latest.{os.getpid()}.{time.time_ns()}.json'
    write(staging,s);os.replace(staging,REPORT/'status_latest.json')
    print(s['queue']);print(s['accounting']);print({k:v.get('progress.json',v.get('failure.json',{})) for k,v in s['conditions'].items()})
    return s


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','status']);a=p.parse_args()
    print({'prepare':prepare,'submit':submit,'status':status}[a.action]())
