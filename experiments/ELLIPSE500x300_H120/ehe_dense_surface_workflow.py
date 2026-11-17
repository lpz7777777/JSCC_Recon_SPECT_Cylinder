"""One authorized formal array for dense EHE A218/A440; no timers or reconstruction."""
import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from ehe_5e9_workflow import connection,command,q
from ehe_common import HERE,ROOT,GPU_BASE,GPU_PYTHON,digest,read,write,hashes,verify_files

TAG='ehe_dense_surface_20261029'
DATA=HERE/'generated'/TAG
PAYLOAD=DATA/'payload'
REPORT=HERE/'reports/EHE'/TAG
REMOTE=GPU_BASE.rsplit('/',1)[0]+'/'+TAG
PARENT=GPU_BASE+'/releases/74e129c4460163c5'
SOURCE=HERE/'generated/ehe_spect_5e9_200/response_payload_v2'
PARENT_REPORT=HERE/'reports/NEMA_Body_H60/ehe_spect_5e9_200'
JOB_NAME='EHE_dense256_218440'


def prepare():
    DATA.mkdir(exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    marker=REPORT/'freeze.json'
    if marker.exists():
        f=read(marker);verify_files(PAYLOAD,f['files']);return f
    if PAYLOAD.exists():raise FileExistsError('Incomplete local freeze retained')
    density=read(DATA/'surface_density_selection.json')
    assert density['all_selected_point_gates_passed'] and len(density['results'])==24
    original=read(PARENT_REPORT/'response_repair_freeze.json')
    binaries=read(DATA/'parent_gpu_binary_manifest.json')
    PAYLOAD.mkdir()
    old=['ehe_common.py','ehe_gpu_pipeline.py','whole_geometry.npz']
    old += ['params/'+name+'/'+file for name in ('A218','A440') for file in ('Params_Image.dat','Params_Detector.dat','Params_Collimator.dat','Params_Physics.dat')]
    for name in old:
        p=SOURCE/name
        if digest(p)!=original['sha256'][name]:raise ValueError('Parent frozen identity differs: '+name)
        dest=PAYLOAD/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    shutil.copyfile(HERE/'ehe_dense_surface_runtime.py',PAYLOAD/'ehe_dense_surface_runtime.py')
    shutil.copyfile(DATA/'surface_density_selection.json',PAYLOAD/'surface_density_selection.json')
    shutil.copyfile(ROOT/'tools/research/audit_ehe_crystal_geometry.py',PAYLOAD/'cpu_point_reference_source.py')
    files=hashes(PAYLOAD)
    contract={'study':TAG,'parent_release':PARENT,'parent_release_key':original['release_key'],
        'parent_freeze_sha256':digest(PARENT_REPORT/'response_repair_freeze.json'),'binary_files':binaries['files'],
        'files':files,'energies_keV':[218,440],'face_subdivisions':256,'surface_samples_per_visible_face':65536,
        'old_samples_per_visible_face':256,'sample_count_multiplier':256,'density_selection':density,
        'formal_array_blocks':8,'concurrent_GPU_limit':8,'array_mapping':'index//4: 218/440; index%4: native z slab0..3',
        'PE_hard_limit_seconds':240000,'Slurm_minutes':4320,'minimum_available_storage_bytes':80*1024**3,
        'expected_full_output_peak_upper_bound_bytes':50*1024**3,'new_transport':False,'new_reconstruction':False,
        'cross_energy_sum':False,'C440to218_changed':False,'geometry_grid_physics_windows_unchanged':True,
        'Scatter_formula_unchanged_but_recomputed_with_new_PE':True,'physical_calibration_passed':False,
        'authorization':'2026-10-29 human explicitly requested denser EHE surface integration and new 440/218 complete system matrices; original versions preserved; no separate validation jobs'}
    key=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()[:16];contract['key']=key
    write(PAYLOAD/'contract.json',contract)
    archive=DATA/(key+'.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PAYLOAD.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(PAYLOAD).as_posix())
    freeze={'key':key,'files':hashes(PAYLOAD),'archive_sha256':digest(archive),'archive':str(archive),'contract_sha256':digest(PAYLOAD/'contract.json')}
    write(marker,freeze)
    (REPORT/'README.md').write_text('# EHE 218/440 keV表面积分加密\n\n独立正式计算，保留74e129c4460163c5原生产版本。每个晶体可见面从16²=256点提高到256²=65536点；24个CPU代表点与512²参考比较通过选点门槛，但不是全矩阵物理认证。\n\n仅A218、A440，2312bin、85×85×40原Cartesian网格、132040完整Polar点/78920活动单元、20view；原孔/晶体/能窗/材料参数和网格不变。PE未展宽和能窗矩阵重新生成；Scatter使用本块新PE未展宽输入按原公式重算，不拼接旧Scatter；每套生成完整Cartesian/Polar及自身S。原准直器散射均匀板深度近似保持，physical_calibration_passed=false，不把加密当作物理修复完成。C440→218不在本轮范围，不修改。\n\n一个正式数组0–7，单元素1张5090、最多8张同时运行；实际GPU/独立CPU点值检查嵌在正式元素内部，无单独验证作业。最后完成的元素在同一正式作业内汇总两套矩阵；所有成功分块只读保留，不自动重投。无新输运、筛选或重建，无定时器。操作入口`ehe_dense_surface_workflow.py status/fetch`，原始数据在新generated目录，不覆盖旧结果。\n',encoding='utf8')
    return freeze


def deploy():
    f=prepare();archive=Path(f['archive']);release=REMOTE+'/releases/'+f['key'];remote_zip=release+'.zip'
    if (REPORT/'deployment.json').exists():return f,release
    code='''import hashlib,json,pathlib,sys,zipfile
archive=pathlib.Path(sys.argv[1]);target=pathlib.Path(sys.argv[2]);expected=sys.argv[3]
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
if not target.exists():
 with zipfile.ZipFile(archive) as z:
  for name in z.namelist():
   p=pathlib.PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts
  target.mkdir();z.extractall(target)
c=json.loads((target/'contract.json').read_text())
assert hashlib.sha256((target/'contract.json').read_bytes()).hexdigest()==sys.argv[4]
for n,h in c['files'].items():assert hashlib.sha256((target/n).read_bytes()).hexdigest()==h,n
parent=pathlib.Path(c['parent_release'])
for n,h in c['binary_files'].items():assert hashlib.sha256((parent/n).read_bytes()).hexdigest()==h,n
print('DENSE_RELEASE_AND_PARENT_BINARIES_SHA_PASS')
'''
    with connection('gpu') as c:
        command(c,'mkdir -p '+q(REMOTE+'/releases')+' '+q(REMOTE+'/logs'))
        with c.open_sftp() as s:
            try:s.stat(remote_zip)
            except FileNotFoundError:s.put(str(archive),remote_zip)
        text=command(c,q(GPU_PYTHON)+' -B -c '+q(code)+' '+q(remote_zip)+' '+q(release)+' '+q(f['archive_sha256'])+' '+q(f['contract_sha256']),120)
    write(REPORT/'deployment.json',{'key':f['key'],'release':release,'archive_sha256':f['archive_sha256'],'result':text})
    return f,release


def launcher(f,release):
    return '''#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.9
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1
export CUDA_CACHE_PATH="/tmp/ehe_dense_cuda_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$CUDA_CACHE_PATH"
'''+q(GPU_PYTHON)+' -B '+q(release+'/ehe_dense_surface_runtime.py')+' --release '+q(release)+' --output '+q(REMOTE)+'/formal_"${SLURM_ARRAY_JOB_ID}" --index "$SLURM_ARRAY_TASK_ID"\n'


def submit():
    if (REPORT/'formal_job.json').exists():return read(REPORT/'formal_job.json')
    if (REPORT/'submission_intent.json').exists():raise RuntimeError('Unresolved sbatch intent; inspect queue, never blindly resubmit')
    f,release=deploy();script=launcher(f,release);local=DATA/'formal.sh';local.write_text(script,encoding='utf8',newline='\n')
    for name in ('local_prepare_checks.json','linux_prepare_checks.json'):
        proof=read(REPORT/name)
        if not proof['passed'] or proof['freeze_key']!=f['key']:raise ValueError('Frozen local/Linux preparation checks required')
    with connection('gpu') as c:
        queue=command(c,'squeue -h -u "$USER" -o "%i|%j|%T|%R"',30)
        if JOB_NAME in queue:raise RuntimeError('Same formal job name already in queue')
        if len(queue.strip().splitlines())>40:raise RuntimeError('Account job limit headroom unavailable; preserve without cancelling other work')
        free=command(c,q(GPU_PYTHON)+' -B -c '+q('import shutil;print(shutil.disk_usage('+repr(REMOTE)+').free)'),30)
        if int(free.strip())<80*1024**3:raise OSError('Actual persistent directory storage insufficient')
        remote=REMOTE+'/formal_'+f['key']+'.sh'
        with c.open_sftp() as s:s.put(str(local),remote)
        request={'key':f['key'],'script_sha256':digest(local),'array':'0-7%8','GPU_partition':'gpu_5090','GPUs_per_element':1,
            'maximum_concurrent_GPUs':8,'Slurm_minutes':4320,'remote_script':remote,'source_sha256':digest(__file__),'submitted_epoch':time.time()}
        write(REPORT/'submission_intent.json',request)
        cmd='sbatch --parsable --partition=gpu_5090 --account=scxi717 --qos=gpugpu -N 1 -n 1 -c 6 --gres=gpu:1 --array=0-7%8 --time=4320 --job-name='+JOB_NAME+' --comment='+f['key']+' --output='+q(REMOTE+'/logs/formal_%A_%a.log')+' '+q(remote)
        result=command(c,cmd,60).strip();job=result.split(';')[0]
        if not job.isdigit():raise ValueError('Uncertain sbatch outcome, submission intent retained')
        request.update(job=int(job),output=REMOTE+'/formal_'+job,sbatch_result=result)
        write(REPORT/'formal_job.json',request)
    print('EHE_DENSE_FORMAL_ARRAY_SUBMITTED',job,flush=True);return request


def status():
    job=read(REPORT/'formal_job.json');j=str(job['job']);out=job['output']
    repair=read(REPORT/'allocation_repair_job.json') if (REPORT/'allocation_repair_job.json').exists() else None
    code='''import pathlib,json,sys
p=pathlib.Path(sys.argv[1]);d={'completed_blocks':[],'PE_progress':{},'completion':None}
for f in sorted(p.glob('A*/slab_*/receipt.json')):d['completed_blocks'].append(str(f.relative_to(p)))
for f in sorted(p.glob('A*/slab_*/PE_progress.json')):
 r=json.loads(f.read_text());d['PE_progress'][str(f.relative_to(p))]={k:r.get(k) for k in ('status','completed_rows','total_rows','elapsed_seconds')}
if (p/'completion.json').exists():d['completion']=json.loads((p/'completion.json').read_text())
for f in p.glob('conversion_claim/failure.json'):d['conversion_failure']=json.loads(f.read_text())
print(json.dumps(d))
'''
    with connection('gpu') as c:
        queue=command(c,'squeue -h -r -j '+j+' -o "%i|%j|%T|%M|%R"',30)
        acct=command(c,'sacct -j '+j+' --parsable2 -n -o JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES',40)
        progress=json.loads(command(c,q(GPU_PYTHON)+' -B -c '+q(code)+' '+q(out),30))
        retry=None
        if repair:
            r=str(repair['job'])
            retry={'job':repair['job'],'replaces':'1749085_7','queue':command(c,'squeue -h -r -j '+r+' -o "%i|%j|%T|%M|%R"',30),
                   'accounting':command(c,'sacct -j '+r+' --parsable2 -n -o JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES',40)}
    result={'job':job['job'],'queue':queue,'accounting':acct,'progress':progress,'checked_epoch':time.time()}
    if retry:result['bounded_repair']=retry
    write(REPORT/'latest_status.json',result);print(json.dumps(result,ensure_ascii=False,indent=2));return result


def fetch():
    state=status();job=read(REPORT/'formal_job.json')
    repair=state.get('bounded_repair')
    if state['queue'].strip() or (repair and repair['queue'].strip()) or not state['progress']['completion']:raise RuntimeError('Formal calculation not fully exited and complete')
    lines=[line.split('|') for line in state['accounting'].splitlines() if line.split('|')[0] in [str(job['job'])+'_'+str(i) for i in range(8)]]
    effective_jobs={str(i):str(job['job'])+'_'+str(i) for i in range(8)}
    if repair:
        old={row[0]:row[1:3] for row in lines}
        retry_rows=[row.split('|') for row in repair['accounting'].splitlines() if row.split('|')[0]==str(repair['job'])+'_7']
        if old.get(str(job['job'])+'_7')!=['FAILED','1:0'] or len(retry_rows)!=1 or retry_rows[0][1:3]!=['COMPLETED','0:0']:
            raise RuntimeError('Original failed element and unique successful bounded replacement required')
        effective_jobs['7']=str(repair['job'])+'_7'
        lines=[row for row in lines if row[0]!=str(job['job'])+'_7']+retry_rows
    if len(lines)!=8 or any(row[1:3]!=['COMPLETED','0:0'] for row in lines):raise RuntimeError('All eight effective formal roots must COMPLETED/0:0')
    remote=job['output'];target=DATA/('delivered_'+str(job['job']));target.mkdir(exist_ok=True)
    with connection('gpu') as c:
        with c.open_sftp() as s:
            def pull(name,expected=None):
                p=target/name;p.parent.mkdir(parents=True,exist_ok=True)
                if p.exists():
                    if expected is not None and digest(p)==expected:return
                    raise FileExistsError('Preserve existing unverified local file: '+str(p))
                partial=p.with_name(p.name+'.partial')
                if partial.exists():raise FileExistsError('Interrupted strict fetch preserved: '+str(partial))
                s.get(remote+'/'+name,str(partial))
                if expected is not None and digest(partial)!=expected:raise ValueError('Fetch SHA differs: '+name)
                partial.replace(p)
            completion=state['progress']['completion']
            for energy in (218,440):
                prefix='factors/A'+str(energy)+'/'
                pull(prefix+'factor_manifest.json',completion['factors'][str(energy)])
                manifest=read(target/prefix/'factor_manifest.json')
                for name,h in manifest['files'].items():pull(prefix+name,h)
            pull('completion.json')
    write(REPORT/'strict_fetch.json',{'passed':True,'job':job['job'],'all_factor_files_full_SHA_verified':True,'local':str(target),
        'physical_calibration_passed':False,'fetched_epoch':time.time(),'effective_formal_jobs_by_block':effective_jobs})
    print('TWO_DENSE_FACTORS_STRICTLY_DELIVERED',target)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','deploy','submit','status','fetch']);a=p.parse_args();globals()[a.action]()
