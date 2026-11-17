"""Independent GE-HEGP/existing-detector NEMA 5B and 50B study.

Only formal transport/collection jobs. The first full imaging worker contains
the zero-history Geant4 solid check. New response histories are shared between
the two new acquisitions. No previous observations or responses are relabelled.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from opengate_ehe_common_20261113 import HERE,read,save,save_array,sha,now
from opengate_he_geometry_20261114 import numerical_geometry_check,holes_mm,crystals_mm

TAG='opengate_he_nema_20261114'
DATA=HERE/'generated'/TAG
REPORT=HERE/'reports/NEMA_Body_H60'/TAG
CPU='/WORK/maty_work/lpz/'+TAG
PARENT=HERE/'generated/opengate_ehe_nema5e9_rotfix_20261113'
PARENT_REPORT=HERE/'reports/NEMA_Body_H60/opengate_ehe_nema5e9_rotfix_20261113'
LOCAL_ENV=HERE/'generated/opengate_ehe_nema5e9_20261113/local_env/Scripts/python.exe'
ENV='/WORK/maty_work/lpz/opengate_ehe_nema5e9_20261113/environment'
DOSES={'5e9':(5_000_000_000,5,616),'5e10':(50_000_000_000,25,617)}


def ast_functions(text):
    return {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
            for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}


def prepare():
    if (REPORT/'freeze.json').exists():return read(REPORT/'freeze.json')
    parent=read(PARENT_REPORT/'freeze.json');assert parent['release']=='f7231adc45ed8607'
    donor=PARENT/parent['payload_relative'];payload=DATA/'payload';payload.mkdir(parents=True,exist_ok=False)
    checks=numerical_geometry_check()
    for name in ('opengate_ehe_common_20261113.py','opengate_ehe_transport_20261113.py',
                 'opengate_ehe_projector_20261113.py','opengate_ehe_guard_20261113.py','cpu_python.sh'):
        assert sha(donor/name)==parent['files'][name]['sha256']
        shutil.copy2(donor/name,payload/name)
    # Retain sharing-safe accepted atomic I/O; only geometry helper bodies differ.
    common=PARENT/'delivery/local_execution/opengate_ehe_common_20261113.py'
    t=common.read_text(encoding='utf8');tree=ast.parse(t)
    for name in ('holes_mm','crystals_mm'):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        lines=t.splitlines(keepends=True)
        old=''.join(lines[node.lineno-1:node.end_lineno])
        t=t.replace(old,f'def {name}():\n    from opengate_he_geometry_20261114 import {name} as implementation\n    return implementation()\n')
        tree=ast.parse(t)
    (payload/'opengate_ehe_common_20261113.py').write_text(t,encoding='utf8',newline='\n')
    transport=payload/'opengate_ehe_transport_20261113.py';t=transport.read_text(encoding='utf8')
    node=next(n for n in ast.parse(t).body if isinstance(n,ast.FunctionDef) and n.name=='build_camera')
    old=''.join(t.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    t=t.replace(old,'def build_camera(sim):\n    from opengate_he_geometry_20261114 import build_camera as implementation\n    return implementation(sim)\n')
    t=t.replace('    sim.run()\n    events=int(stats.counts.events)',
        '    if kind=="imaging" and index==0:\n        from opengate_he_geometry_20261114 import actual_solid_check\n        actual_solid_check(sim,output)\n    sim.run()\n    events=int(stats.counts.events)')
    transport.write_text(t,encoding='utf8',newline='\n')
    old_ast=ast_functions((donor/transport.name).read_text(encoding='utf8'));new_ast=ast_functions(t)
    unchanged=[n for n in old_ast if old_ast[n]==new_ast[n]]
    assert set(old_ast)-set(unchanged)=={'build_camera','run'}
    projector=payload/'opengate_ehe_projector_20261113.py';t=projector.read_text(encoding='utf8')
    assert t.count('-55.5')==1 and t.count('35.4')==2
    t=t.replace('-55.5','-71.0').replace('35.4','36.95').replace('50.5+5=55.5mm','66+5=71mm')
    projector.write_text(t,encoding='utf8',newline='\n')
    cpu_source=PARENT/'delivery/local_execution/opengate_ehe_recon_cpu_20261113.py'
    t=cpu_source.read_text(encoding='utf8')
    assert t.count('!=5_000_000_000')==1
    t=t.replace('!=5_000_000_000','!=cfg["imaging_primary_gamma_total"]').replace('actual 5e9 input required','actual configured HE input required')
    (payload/cpu_source.name).write_text(t,encoding='utf8',newline='\n')
    collect=HERE/'opengate_ehe_collect_20261113.py';t=collect.read_text(encoding='utf8')
    t=t.replace('out=root/"workers"/kind/', 'out=(Path(cfg["HE_response_root"]) if kind=="response" else root)/"workers"/kind/')
    t=t.replace('emitted.sum()!=5_000_000_000','emitted.sum()!=cfg["imaging_primary_gamma_total"]').replace('do not close to 5e9','do not close to configured HE budget')
    (payload/collect.name).write_text(t,encoding='utf8',newline='\n')
    archive=(HERE/'opengate_ehe_archive_20261113.py').read_text(encoding='utf8').replace("'2026-11-13'","'2026-11-14'")
    (payload/'opengate_ehe_archive_20261113.py').write_text(archive,encoding='utf8',newline='\n')
    for name in ('opengate_he_geometry_20261114.py','opengate_he_archive_20261114.py'):
        shutil.copy2(HERE/name,payload/name)
    shutil.copy2(Path(__file__),payload/Path(__file__).name)
    gate=read(PARENT_REPORT/'released_remaining.json')['actual_primary_vertex_check']
    assert gate['actual_primary_vertex_rotation_verified']
    inherited=dict(passed=True,source_parent_release=parent['release'],source_rotation_check=gate,
        source_rotation_gate_receipt_sha256=sha(PARENT_REPORT/'released_remaining.json'),
        actual_primary_vertex_rotation_verified=True,inherited_imaging_primaries=0,
        inherited_response_primaries=0,unchanged_transport_functions=unchanged,
        reused_source_truth_sha256=sha(donor/'inputs/truth_3mm.npz'))
    save(payload/'inherited_science.json',inherited)
    cfg_base=read(donor/'inputs/config.json')
    for dose,(total,parts,seed) in DOSES.items():
        inp=payload/('inputs_'+dose);inp.mkdir()
        shutil.copy2(donor/'inputs/truth_3mm.npz',inp/'truth_3mm.npz')
        cfg=json.loads(json.dumps(cfg_base))
        for key in ('repair_of','reused_views','recomputed_views','reused_response_characterization_primaries'):
            cfg.pop(key,None)
        cfg.update(study=TAG,dose=dose,created_utc=now(),imaging_primary_gamma_total=total,
            HE_response_root=CPU+'/5e9',imaging_observations_inherited=False,
            source_parent_release=parent['release'],max_response_or_imaging_concurrency=160,
            collimator=dict(material='G4_Pb',size_mm=[330,66,165],center_mm=[0,331.5,0],
                hole_count=1787,hole_shape='hexagon',hole_diameter_across_flats_mm=4.,
                septal_thickness_mm=1.8,hole_pitch_mm=5.8,hole_material='G4_Galactic',
                entrance_y_mm=298.5,back_y_mm=364.5,commercial_GE_full_camera=False),
            HE_design='GE HEGP hole specification on existing segmented NaI detector',
            physical_calibration_passed=False)
        cfg['detector']['center_y_mm']=369.5
        cfg['reconstruction']['response']='new HE full-transport 2D CDR, 11 depths x 4 phases x 2 energies'
        cfg['reconstruction']['geometry_crystal_radius_mm']=369.5
        cfg['reconstruction']['geometry_Pb_entrance_to_crystal_mm']=71.
        rng=np.random.default_rng(seed*1_000_000+30001);jobs=[];emitted=np.zeros((20,2),dtype=np.int64)
        p218=cfg['imaging_primary_energy_fraction_expected']['218']
        for view in range(20):
            n218=int(rng.binomial(total//20,p218));emitted[view]=[n218,total//20-n218]
            for ei,energy in enumerate((218,440)):
                for part in range(parts):
                    i=len(jobs)
                    jobs.append(dict(index=i,kind='imaging',view=view,angle_deg=18*view,
                        energy_keV=energy,primaries=int(emitted[view,ei]//parts+(part<emitted[view,ei]%parts)),
                        seed=seed*1_000_000+130001+i,smear_seed=seed*1_000_000+430001+i,importance_weight=1.))
        responses=read(donor/'inputs/response_jobs.json')
        for j in responses:j.update(seed=616230001+j['index'],smear_seed=616530001+j['index'])
        cfg['emitted_primaries_by_view_energy']=emitted.tolist()
        assert sum(j['primaries'] for j in jobs)==total and len(jobs)==parts*40
        save(inp/'config.json',cfg);save(inp/'imaging_jobs.json',jobs);save(inp/'response_jobs.json',responses)
        # Image conversion is numerical, uses exactly the inherited truth, no MC.
        subprocess.run([str(LOCAL_ENV),str(transport),'source_images','--inputs',str(inp)],check=True)
        save(REPORT/dose/'inherited_science.json',inherited)
    all_jobs=[]
    for d in DOSES:all_jobs+=read(payload/f'inputs_{d}/imaging_jobs.json')
    all_jobs+=read(payload/'inputs_5e9/response_jobs.json')
    seeds=[j[k] for j in all_jobs for k in ('seed','smear_seed')]
    assert len(seeds)==len(set(seeds))
    np.savetxt(payload/'HE_holes_xz_mm.csv',holes_mm(),delimiter=',',header='x_mm,z_mm',comments='')
    np.savetxt(payload/'HE_crystals_xyz_mm.csv',crystals_mm(),delimiter=',',header='x_mm,y_mm,z_mm',comments='')
    save(payload/'geometry_checks.json',checks)
    provenance=dict(passed=True,unchanged_transport_functions=unchanged,original_transport_ast=old_ast,
        HE_transport_ast=new_ast,original_transport_sha256=sha(donor/transport.name),
        original_projector_sha256=sha(donor/projector.name),original_CPU_sha256=sha(cpu_source),
        original_source_rotation_gate_sha256=sha(PARENT_REPORT/'released_remaining.json'),
        change_scope=['HE camera geometry','HE projector distances','configured exact budget gate',
                      'zero-history inline geometry check','new acquisition seeds and HE responses'])
    save(payload/'scientific_provenance.json',provenance)
    records={p.relative_to(payload).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))
             for p in sorted(payload.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    identity=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()[:16]
    result=dict(passed=True,study=TAG,release=identity,files=records,payload_relative='payload',
        actual_imaging_budgets=[5_000_000_000,50_000_000_000],new_HE_response_primaries=4_400_000_000,
        independent_validation_jobs=False,reconstruction_CPU_parent_sha256=sha(cpu_source),
        geometry_checks=checks,created_utc=now())
    save(payload/'release.json',result);save(REPORT/'freeze.json',result)
    return dict(passed=True,release=identity,files=len(records),geometry_checks=checks)


def remote_release():return CPU+'/releases/'+read(REPORT/'freeze.json')['release']


def payload_path():return DATA/read(REPORT/'freeze.json')['payload_relative']


def deploy():
    from campaign_20261102_io import upload_files,python_json
    frozen=read(REPORT/'freeze.json');payload=payload_path()
    for n,r in frozen['files'].items():assert sha(payload/n)==r['sha256']
    upload_files('maty',remote_release(),{n:payload/n for n in list(frozen['files'])+['release.json']})
    source='''import os,pathlib,json,sys,hashlib,subprocess
a=json.loads(sys.argv[1]);p=pathlib.Path(a['release']);f=json.loads((p/'release.json').read_text())
for n,r in f['files'].items():
 q=p/n
 if q.stat().st_size!=r['bytes'] or hashlib.sha256(q.read_bytes()).hexdigest()!=r['sha256']:raise ValueError(n)
r=pathlib.Path(a['root']);(r/'control').mkdir(parents=True,exist_ok=True)
receipt=r/'control/output_capacity.json';reserve=r/'control/output_reservation.bin'
if receipt.exists():result=json.loads(receipt.read_text())
else:
 with reserve.open('xb') as h:os.posix_fallocate(h.fileno(),0,4*(1<<30));h.flush();os.fsync(h.fileno())
 st=reserve.stat()
 if st.st_blocks*512<st.st_size:raise ValueError('Real quota allocation failed')
 result=dict(passed=True,real_reserved_bytes=st.st_size,allocated_bytes=st.st_blocks*512,planned_output_bytes=3*(1<<30),fraction_limit=.8)
 receipt.write_text(json.dumps(result))
for dose in ('5e9','5e10'):
 d=r/dose;(d/'control').mkdir(parents=True,exist_ok=True);(d/'logs').mkdir(exist_ok=True)
 link=d/'inputs'
 if not link.exists():link.symlink_to(p/('inputs_'+dose))
 if link.resolve()!=p/('inputs_'+dose):raise ValueError('Different source inputs')
 link=d/'releases'
 if not link.exists():link.symlink_to(r/'releases')
 if link.resolve()!=r/'releases':raise ValueError('Different executed release pointer')
print('CAMPAIGN_JSON '+json.dumps(dict(result,active_jobs=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%T'],text=True,timeout=20))))'''
    result=python_json('maty','python3',source,dict(root=CPU,release=remote_release()),timeout=120,readonly=False)
    save(REPORT/'deployment.json',result);return result


def submit(dose,stage,dependency=None,hold=False):
    from campaign_20261102_io import upload_files,python_json
    from ehe_5e9_workflow import q
    receipt=REPORT/dose/(stage+'_job.json')
    if receipt.exists():return read(receipt)
    assert read(REPORT/'deployment.json')['passed']
    root=CPU+'/'+dose;r=remote_release();inp=r+'/inputs_'+dose
    body='#!/bin/bash\nset -euo pipefail\n'+f'cd {q(r)}\n'
    flags=['--partition=cnmix','-N','1','-n','1','-c','1','--time=72:00:00']
    if stage in ('imaging','response'):
        n=(200 if dose=='5e9' else 1000) if stage=='imaging' else 88
        concurrency=72 if dose=='5e9' and stage=='imaging' else (88 if stage=='response' else 160)
        body+='index="${SLURM_ARRAY_TASK_ID:?}"\nprintf -v index4 "%04d" "$index"\n'
        body+=f'mkdir -p {q(root+"/resources")}\nallocation={q(root+"/resources/"+stage+"_")}"$index4".allocation.txt\n'
        body+='scontrol show job "${SLURM_JOB_ID:?}" > "$allocation"\n'
        body+=(f'bash {q(r+"/cpu_python.sh")} {q(r+"/opengate_ehe_guard_20261113.py")} '
            f'--record {q(root+"/resources/"+stage+"_")}"$index4".json --allocation "$allocation" --seconds 250000 -- '
            f'{q(ENV+"/venv/bin/python")} {q(r+"/opengate_ehe_transport_20261113.py")} '
            f'run --inputs {q(inp)} --kind {stage} --index "$index" --output {q(root+"/workers/"+stage+"/")}"$index4"\n')
        flags += [f'--array=0-{n-1}%{concurrency}']
    elif stage=='collect':
        body+=f'bash {q(r+"/cpu_python.sh")} {q(r+"/opengate_ehe_collect_20261113.py")} --root {q(root)}\n'
    else:raise ValueError(stage)
    if dependency:flags+=['--dependency=afterok:'+dependency]
    if hold:flags+=['--hold']
    flags+=['--job-name=og_HE_'+dose+'_'+stage,'--output='+root+'/logs/'+stage+'_%A_%a.log']
    script=DATA/dose/(stage+'.sh');script.parent.mkdir(parents=True,exist_ok=True)
    script.write_text(body,encoding='ascii',newline='\n')
    remote=root+'/scripts/'+read(REPORT/'freeze.json')['release']
    upload_files('maty',remote,{script.name:script})
    plan=dict(root=root,study_root=CPU,stage=stage,dose=dose,release=read(REPORT/'freeze.json')['release'],
        flags=flags,script=remote+'/'+script.name,script_sha256=sha(script),created_utc=now())
    save(REPORT/dose/(stage+'_submission_intent.json'),plan)
    source='''import pathlib,json,sys,fcntl,hashlib,subprocess,time,os
a=json.loads(sys.argv[1]);r=pathlib.Path(a['root']);lock=(r/'control'/('submit_'+a['stage']+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=r/'control'/(a['stage']+'_job.json');intent=r/'control'/(a['stage']+'_intent.json')
def save(p,d):
 t=p.with_suffix('.writing')
 with t.open('w') as h:json.dump(d,h);h.flush();os.fsync(h.fileno())
 t.replace(p)
if p.exists():
 result=json.loads(p.read_text())
 if result['script_sha256']!=a['script_sha256']:raise ValueError('Different registered submit')
else:
 if intent.exists():raise ValueError('Unresolved submit intent; inspect scheduler before any retry')
 if hashlib.sha256(pathlib.Path(a['script']).read_bytes()).hexdigest()!=a['script_sha256']:raise ValueError('Changed script')
 save(intent,a)
 reserve=pathlib.Path(a['study_root'])/'control/output_reservation.bin'
 if reserve.exists():
  if reserve.stat().st_size!=4*(1<<30):raise ValueError('Capacity allocation changed')
  reserve.unlink();save(pathlib.Path(a['study_root'])/'control/capacity_released.json',dict(released_epoch=time.time(),bytes=4*(1<<30)))
 x=subprocess.run(['sbatch','--parsable']+a['flags']+[a['script']],capture_output=True,text=True,timeout=60)
 if x.returncode:raise RuntimeError(x.stderr)
 number=x.stdout.strip().split(';')[0]
 if not number.isdigit():raise ValueError('Uncertain sbatch result')
 result=dict(a,job=int(number),submitted_epoch=time.time());save(p,result)
print('CAMPAIGN_JSON '+json.dumps(result))'''
    result=python_json('maty','python3',source,plan,timeout=100,readonly=False)
    save(receipt,result);return result


def fetch(dose,kind):
    from campaign_20261102_io import execute
    from ehe_5e9_workflow import q
    import opengate_ehe_delivery_20261113 as delivery
    data=DATA/dose;report=REPORT/dose;receipt=report/('maty_'+kind+'_strict_fetch.json')
    if receipt.exists():return read(receipt)
    remote=remote_release();raw=execute('maty','python3 '+q(remote+'/opengate_he_archive_20261114.py')+
        ' --root '+q(CPU+'/'+dose)+' --kind '+kind,timeout=1800)
    record=json.loads(raw.rsplit('CAMPAIGN_JSON ',1)[1]);save(report/('maty_'+kind+'_bundle.json'),record)
    delivery.DATA,delivery.REPORT=data,report
    archive=delivery.download('maty',record)
    dest=data/('delivery' if kind=='bridge' else 'transport_delivery')
    manifest=delivery.extract(archive,record,dest)
    result=dict(passed=True,strict_fetch_passed=True,all_files_SHA_verified=True,host='maty',kind=kind,
        created_utc=now(),archive=record,destination=str(dest),files=manifest['files'],authority=manifest['authority'])
    save(receipt,result);return dict(passed=True,kind=kind,members=len(manifest['files']),dose=dose)


def reconstruct(dose):
    import psutil
    data=DATA/dose;report=REPORT/dose;root=data/'delivery';payload=payload_path()
    if (report/'local_cpu_exit.json').exists():raise FileExistsError('Do not repeat registered reconstruction')
    proof=read(report/'maty_bridge_strict_fetch.json');assert proof['passed']
    for n,r in proof['files'].items():assert sha(root/n)==r['sha256']
    assert proof['authority']['actual_primary_gamma_total']==DOSES[dose][0]
    if (root/'reconstruction').exists():raise FileExistsError('Parallel/partial reconstruction requires explicit bounded repair')
    release=data/'local_cpu_release';release.mkdir()
    for n in ('opengate_ehe_common_20261113.py','opengate_ehe_projector_20261113.py',
              'opengate_ehe_recon_cpu_20261113.py','opengate_he_geometry_20261114.py'):
        shutil.copy2(payload/n,release/n)
    frozen=dict(passed=True,release=read(REPORT/'freeze.json')['release'],threads=8,iterations=200,save_step=1,
        initial_density=1.,image_filter=None,regularization=None,actual_resume_used=False,
        geometry_crystal_radius_mm=369.5,Pb_entrance_to_crystal_center_mm=71.,
        response_kernel_sha256=sha(root/'collected/response_kernels.npz'),
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in release.iterdir()})
    save(release/'release.json',frozen);save(report/'reconstruction_freeze.json',frozen)
    execution=root/'local_execution';shutil.copytree(release,execution)
    cap=int(min(8*(1<<30),.8*psutil.virtual_memory().available))
    if cap<4*(1<<30) or shutil.disk_usage(root).free<4*(1<<30)/.8+16*(1<<30):raise MemoryError('Actual local resource margin')
    env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',OPENGATE_LOCAL_HOST_CAP_BYTES=str(cap))
    cmd=[str(LOCAL_ENV),'-u',str(release/'opengate_ehe_recon_cpu_20261113.py'),'--root',str(root),'--inputs',str(root/'inputs')]
    start=time.monotonic();peak=0;last=0
    with (execution/'stdout.log').open('xb') as out,(execution/'stderr.log').open('xb') as err:
        child=subprocess.Popen(cmd,env=env,stdout=out,stderr=err);proc=psutil.Process(child.pid)
        reg=dict(pid=child.pid,create_time=proc.create_time(),cmdline=proc.cmdline(),controller_pid=os.getpid(),status='RUNNING',started_utc=now())
        save(report/'local_cpu_registration.json',reg)
        try:
            while child.poll() is None:
                group=[proc]+proc.children(recursive=True);rss=0
                for p in group:
                    try:rss+=max(p.memory_info().rss,getattr(p.memory_info(),'peak_wset',0))
                    except psutil.NoSuchProcess:pass
                peak=max(peak,rss)
                if peak>.8*cap or psutil.virtual_memory().available<.2*cap:raise MemoryError('Actual 20% memory margin')
                if time.monotonic()-start>86400:raise TimeoutError('Finite execution limit')
                if time.monotonic()-last>30:print('HE_CPU_RUNNING',dose,round(time.monotonic()-start),peak,flush=True);last=time.monotonic()
                time.sleep(.5)
            if child.returncode or not read(root/'reconstruction_complete.json')['passed']:raise RuntimeError('HE reconstruction failure')
        except BaseException:
            if child.poll() is None:
                for p in proc.children(recursive=True)+[proc]:
                    try:p.terminate()
                    except psutil.NoSuchProcess:pass
                child.wait(timeout=30)
            save(report/'local_cpu_exit.json',dict(reg,status='FAILED',returncode=child.returncode,ended_utc=now()));raise
    result=dict(reg,status='COMPLETED',passed=True,returncode=0,elapsed_seconds=time.monotonic()-start,
        peak_family_rss_bytes=peak,ended_utc=now())
    save(report/'local_cpu_exit.json',result);save(execution/'exit.json',result);return result


def finalize(dose):
    # Independent accepted finite/nonnegative/domain/history/member checks.
    import opengate_ehe_nema5e10_20261113 as donor
    donor.DATA,donor.REPORT,donor.TOTAL=DATA/dose,REPORT/dose,DOSES[dose][0]
    result=donor.finalize()
    result.update(study=TAG,dose=dose,geometry='GE HEGP + existing segmented NaI',
        response_reused_from_EHE=False,response_kernel_sha256=sha(DATA/dose/'delivery/collected/response_kernels.npz'))
    save(REPORT/dose/'strict_delivery.json',result)
    return dict(passed=True,dose=dose,checkpoints=result['checkpoints'],members=len(result['files']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','deploy','submit','fetch','reconstruct','finalize'])
    p.add_argument('--dose',choices=list(DOSES));p.add_argument('--stage',choices=['imaging','response','collect'])
    p.add_argument('--dependency');p.add_argument('--hold',action='store_true');p.add_argument('--kind',choices=['bridge','transport'])
    a=p.parse_args()
    if a.action=='prepare':r=prepare()
    elif a.action=='deploy':r=deploy()
    elif a.action=='submit':r=submit(a.dose,a.stage,a.dependency,a.hold)
    elif a.action=='fetch':r=fetch(a.dose,a.kind)
    elif a.action=='reconstruct':r=reconstruct(a.dose)
    else:r=finalize(a.dose)
    print('CAMPAIGN_JSON '+json.dumps(r))
