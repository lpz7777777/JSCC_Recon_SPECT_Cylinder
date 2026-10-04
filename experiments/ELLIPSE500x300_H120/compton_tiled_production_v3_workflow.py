"""Bounded/resumable production of the independently tested .75mm A lattice."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import numpy as np
from compton_geometry_v3_workflow import HERE,ROOT,DATA,REPORT,REMOTE,SERVER,SERVER_ROOT,PYTHON,ssh,transfer,digest,write
from plan_compton_a_tiles_v3 import tile_spec


def make_plan():
    tiles=json.loads((REPORT/'A_tile_plan.json').read_text())
    regional=json.loads((REPORT/'regional_a_plan.json').read_text())
    validation=json.loads((REPORT/'regional_validation_frozen_measure_summary.json').read_text())
    faces=json.loads((REPORT/'tile_pilot_summary.json').read_text())
    if (validation['status']!='REGIONAL_REFINEMENT_PASSED_LOCAL_ONLY'
            or faces['status']!='TILE_INTERFACE_AND_COMPACTION_PASSED_ACCURACY_HOLD'
            or regional['geometry_sha256']!=tiles['geometry_sha256']):raise ValueError('Pre-production diagnostic gates not passed')
    selected={};allowed={tuple(x) for x in tiles['xy_tile_indices']}
    for c in regional['cases']:
        low,high=np.asarray(c['full_reference_bounds_mm'])
        first=np.floor((low-[-258,-258,-60])/12+1e-10).astype(int)
        last=np.ceil((high-[-258,-258,-60])/12-1e-10).astype(int)-1
        for x in range(first[0],last[0]+1):
            for y in range(first[1],last[1]+1):
                for z in range(first[2],last[2]+1):
                    if (x,y) not in allowed:raise ValueError('Frozen regional pilot outside required coverage')
                    spec=tile_spec(x,y,z);selected[spec['name']]=spec
    selected=[selected[k] for k in sorted(selected)]
    return dict(status='FROZEN_TILED_PHYSICAL_PRODUCTION_PLAN_ACCURACY_HOLD',
        geometry_sha256=tiles['geometry_sha256'],A_tile_plan_sha256=digest(REPORT/'A_tile_plan.json'),
        regional_plan_sha256=digest(REPORT/'regional_a_plan.json'),
        regional_validation_sha256=validation['evidence_sha256'],
        tile_interface_gate_sha256=faces['evidence_sha256'],
        total_tiles=tiles['total_tiles'],xy_tile_indices=tiles['xy_tile_indices'],pilot_specs=selected,
        pilot_tile_count=len(selected),pilot_controls=12,
        retained_bytes_per_tile=tiles['retained_combined_10496_row_bytes_per_tile'],
        complete_retained_bytes=tiles['retained_combined_all_tiles_bytes'],
        maximum_temporary_bytes=4*(tiles['four_raw_physical_11520_row_bytes_per_tile']+
            tiles['retained_combined_10496_row_bytes_per_tile'])+(4<<30),
        disk_reserve_fraction=.20,imaging_grid_unchanged=True,new_transport_photons=0,
        reconstruction_permitted=False,S2_generated=False,
        interpretation='All tiles covering12 frozen regional controls first; full field only after multi-worker/resource/interface audit. '
            'Local sampling convergence does not certify untested regions or authorize S2/imaging.')


def launch_pilot(repair=False):
    tag='tiled_field_pilot_repaired' if repair else 'tiled_field_pilot'
    if (REPORT/(tag+'_job.json')).exists():raise ValueError('Pilot already registered; do not repeat')
    previous=None
    if repair:
        previous=json.loads((REPORT/'tiled_field_pilot_job.json').read_text())
        for worker in previous['workers']:
            state=ssh(SERVER,f'if ps -p {worker["pid"]} -o args= | grep -q run_tiled_field; then echo ACTIVE; fi')
            if state:raise ValueError('Original pilot must finish/exit before repair')
        logs={}
        for w in previous['workers']:
            logs[str(w['shard'])]=ssh(SERVER,'cat '+REMOTE+f'/tiled_field_pilot_s{w["shard"]}.log')
        if any('Different production plan already registered' not in logs[str(i)] for i in (1,2,3)):
            raise ValueError('Repair limited to diagnosed plan-byte mismatch')
        for key,value in logs.items():(REPORT/f'tiled_field_pilot_original_s{key}.txt').write_text(value,encoding='utf-8')
    devices=(0,1,3,4)
    for d in devices:
        free,util=map(int,ssh(SERVER,f'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i {d}').split(','))
        if free<30000 or util>5:raise ValueError(f'Physical GPU{d} occupied; leave other projects untouched')
    job=json.loads((REPORT/'regional_validation_frozen_measure_job.json').read_text())
    actual=ssh(SERVER,'sha256sum '+job['output']+'/regional_validation.json').split()[0]
    plan=make_plan()
    if actual!=plan['regional_validation_sha256']:raise ValueError('Executed regional diagnostic differs')
    file=REPORT/'tiled_production_plan.json'
    if file.exists():
        if json.loads(file.read_text())!=plan:raise ValueError('Existing frozen production plan differs')
    else:write(file,plan)
    names=('generate_compton_tiled_field_v3.py','generate_compton_a_tile_pilot_v3.py',
        'generate_compton_a_column_v3.py','generate_compton_a_guard.py','plan_compton_a_tiles_v3.py',
        'compton_overlap_integrator.py','compton_cell_response_field.py','compton_boundary_quadrature.py',
        'geometry.py','compton_cartesian_patch.py','test_compton_tiled_production_v3.py')
    paths={n:HERE/n for n in names};paths['tiled_production_plan.json']=file
    paths['compton_event_response.py']=ROOT/'compton_event_response.py'
    paths['detector_csv.py']=ROOT/'detector_csv.py'
    hashes={n:digest(path) for n,path in paths.items()};key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/tiled_field_releases/'+key;ssh(SERVER,'mkdir -p -- '+shlex.quote(release))
    for n,path in paths.items():transfer(SERVER,path,release+'/'+n)
    if repair:
        script='''import hashlib,json,os,shutil
from pathlib import Path
root=Path(OUTPUT);want=Path(RELEASE)/'tiled_production_plan.json';old=root/'production_plan.json'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=h(old);expected=h(want)
if json.loads(old.read_text())!=json.loads(want.read_text()):raise ValueError('Semantic plan differs; not a line-ending repair')
archive=root/('production_plan_original_'+before+'.json')
if archive.exists():
    if h(archive)!=before:raise ValueError('Historical plan bytes differ')
else:shutil.copyfile(old,archive)
temp=root/'production_plan_exact.tmp'
with temp.open('xb') as f:f.write(want.read_bytes());f.flush();os.fsync(f.fileno())
os.replace(temp,old)
if h(old)!=expected:raise ValueError('Exact plan replacement failed')
print(json.dumps(dict(original_sha256=before,frozen_sha256=expected,semantic_identity_preserved=True,old_plan_archived=True,physical_tiles_unchanged=True)))
'''.replace('OUTPUT',repr(REMOTE+'/A440_tiled_field')).replace('RELEASE',repr(release))
        receipt=json.loads(ssh(SERVER,PYTHON+' -c '+shlex.quote(script)))
        write(REPORT/'tiled_plan_identity_repair.json',receipt)
    tests=ssh(SERVER,'cd '+shlex.quote(release)+' && '+PYTHON+' -m unittest test_compton_tiled_production_v3 -v 2>&1')
    if '\nOK' not in tests or 'skipped' in tests:raise ValueError('Tile continuation/integrity tests failed')
    (REPORT/(tag+'_tests.txt')).write_text(tests,encoding='utf-8')
    write(REPORT/(tag+'_deployment.json'),dict(release=release,code_sha256=hashes))
    output=REMOTE+'/A440_tiled_field';jobs=[]
    for shard,d in enumerate(devices):
        script=DATA/(f'run_{tag}_s{shard}.sh')
        script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+f'/{tag}_s{shard}.lock\nflock -n 9\n'+
            f'export CUDA_VISIBLE_DEVICES={d} PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4\ncd '+release+'\n'+
            'timeout --signal=TERM --kill-after=30s 45m '+PYTHON+' generate_compton_tiled_field_v3.py --root '+SERVER_ROOT+
            ' --plan '+release+'/tiled_production_plan.json --field '+REMOTE+'/A440_guard_field --output '+output+
            f' --phase pilot --shard {shard} --shards 4 --max-new-tiles {len(plan["pilot_specs"])} --cuda 0 --physical-gpu {d}\n',encoding='ascii',newline='\n')
        transfer(SERVER,script,release+'/'+script.name)
        pid=ssh(SERVER,'nohup bash '+release+'/'+script.name+' > '+REMOTE+f'/{tag}_s{shard}.log 2>&1 < /dev/null & echo $!')
        if not pid.isdigit():raise ValueError('Unknown pilot launch outcome; inspect before retry')
        jobs.append(dict(shard=shard,physical_gpu=d,pid=int(pid),launcher_name=script.name,launcher_sha256=digest(script)))
        write(REPORT/(tag+'_job.json'),dict(release=release,output=output,workers=jobs,
            total_expected_workers=4,total_pilot_tiles=len(plan['pilot_specs']),timeout_minutes=45,
            full_field_started=False,S2_generated=False,reconstruction_submitted=False))
    print(json.dumps(jobs,indent=2))


def status():
    for tag in ('tiled_field_pilot','tiled_field_pilot_repaired','tiled_field_probe','tiled_field_full'):
        path=REPORT/(tag+'_job.json')
        if not path.exists():continue
        j=json.loads(path.read_text())
        for w in j['workers']:
            pid=w['pid'];command=f'if ps -p {pid} -o args= | grep -q run_tiled_field; then echo ACTIVE; else echo EXITED; fi'
            command+='; tail -2 '+REMOTE+f'/{tag}_s{w["shard"]}.log'
            print(tag,w['shard'],ssh(SERVER,command))
    print(ssh(SERVER,'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'))


def fetch():
    tag='tiled_field_pilot_repaired' if (REPORT/'tiled_field_pilot_repaired_job.json').exists() else 'tiled_field_pilot'
    j=json.loads((REPORT/(tag+'_job.json')).read_text());results=[]
    for w in j['workers']:
        path=j['output']+f'/pilot_shard{w["shard"]}_complete.json'
        expected=ssh(SERVER,'sha256sum '+path).split()[0];target=DATA/f'tiled_pilot_shard{w["shard"]}.json'
        subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+path,str(target)],check=True)
        if digest(target)!=expected:raise ValueError('Pilot transfer differs')
        results.append(dict(evidence_sha256=expected,worker=w,result=json.loads(target.read_text())))
    names=[t['name'] for r in results for t in r['result']['tiles']]
    if (len(names)!=j['total_pilot_tiles'] or len(set(names))!=j['total_pilot_tiles']
            or not all(r['result']['actual_resources']['passed'] for r in results)):
        raise ValueError('Pilot tile/worker/resource completeness gate failed')
    write(REPORT/'tiled_field_pilot_summary.json',dict(status='MULTI_WORKER_TILES_STORED_INTERFACE_AUDIT_PENDING',
        workers=results,total_tiles=len(names),new_transport_photons=0,full_field_started=False,reconstruction_permitted=False,S2_generated=False))
    print(json.dumps(dict(tiles=len(names),seconds=[r['result']['elapsed_seconds'] for r in results],
        resources=[r['result']['actual_resources'] for r in results]),indent=2))


def validate_pilot():
    tag='tiled_field_validation'
    if (REPORT/(tag+'_job.json')).exists():raise ValueError('Tiled diagnostic already registered')
    summary=json.loads((REPORT/'tiled_field_pilot_summary.json').read_text())
    if summary['status']!='MULTI_WORKER_TILES_STORED_INTERFACE_AUDIT_PENDING':raise ValueError('Pilot storage/resources not passed')
    device=5
    free,util=map(int,ssh(SERVER,f'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i {device}').split(','))
    if free<30000 or util>5:raise ValueError('Diagnostic GPU occupied')
    names=('validate_compton_regional_a_v3.py','geometry.py','compton_cartesian_patch.py',
        'compton_tiled_a_field.py','test_compton_tiled_a_field_v3.py','plan_compton_a_tiles_v3.py',
        'compton_overlap_integrator.py','compton_cell_response_field.py','compton_boundary_quadrature.py',
        'generate_compton_a_guard.py','generate_compton_a_column_v3.py','generate_compton_a_tile_pilot_v3.py')
    paths={n:HERE/n for n in names}
    paths.update({n:ROOT/n for n in ('compton_event_response.py','detector_csv.py')})
    paths['geometry.npz']=HERE/'generated/Geometry/geometry.npz';paths['config.json']=HERE/'config.json'
    paths['regional_a_plan.json']=REPORT/'regional_a_plan.json'
    for n in ('measure.npz','measure_manifest.json'):paths[n]=DATA/'precise_measure'/n
    hashes={n:digest(p) for n,p in paths.items()};key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/'+tag+'_releases/'+key;ssh(SERVER,'mkdir -p -- '+shlex.quote(release))
    for n,path in paths.items():transfer(SERVER,path,release+'/'+n)
    tests=ssh(SERVER,'cd '+shlex.quote(release)+' && '+PYTHON+' -m unittest test_compton_tiled_a_field_v3 -v 2>&1')
    if '\nOK' not in tests or 'skipped' in tests:raise ValueError('Tiled exact-node/interpolation tests failed')
    (REPORT/(tag+'_tests.txt')).write_text(tests,encoding='utf-8')
    write(REPORT/(tag+'_deployment.json'),dict(release=release,code_sha256=hashes))
    from compton_geometry_v3_workflow import SERVER_STUDY,SERVER_BASE
    output=REMOTE+'/'+tag;script=DATA/('run_'+tag+'.sh')
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+'/'+tag+'.lock\nflock -n 9\n'+
        f'export CUDA_VISIBLE_DEVICES={device} PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8\ncd '+release+'\n'+
        'timeout --signal=TERM --kill-after=30s 15m '+PYTHON+' validate_compton_regional_a_v3.py --plan '+release+'/regional_a_plan.json'+
        ' --samples '+REMOTE+' --field '+REMOTE+'/A440_guard_field --geometry '+release+'/geometry.npz --config '+release+'/config.json'+
        ' --measure '+release+' --inputs '+SERVER_STUDY+'/analysis_inputs --factors '+SERVER_BASE+'/generated/FactorsCalibrated'+
        ' --scan '+REMOTE+'/R1_analysis --tiles '+REMOTE+'/A440_tiled_field --output '+output+'\n',encoding='ascii',newline='\n')
    transfer(SERVER,script,release+'/'+script.name)
    pid=ssh(SERVER,'nohup bash '+release+'/'+script.name+' > '+REMOTE+'/'+tag+'.log 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unknown pilot validation launch outcome')
    write(REPORT/(tag+'_job.json'),dict(pid=int(pid),release=release,physical_gpu=device,output=output,
        launcher_sha256=digest(script),timeout_minutes=15,new_transport_photons=0,reconstruction_submitted=False,S2_generated=False))
    print('TILED_FIELD_VALIDATION_PID',pid)


def fetch_validation():
    tag='tiled_field_validation';job=json.loads((REPORT/(tag+'_job.json')).read_text())
    for n in ('regional_validation.json','regional_cases.csv'):
        path=job['output']+'/'+n;expected=ssh(SERVER,'sha256sum '+path).split()[0]
        target=REPORT/(tag+('_cases.csv' if n.endswith('.csv') else '_evidence.json'))
        subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+path,str(target)],check=True)
        if digest(target)!=expected:raise ValueError('Tiled precision evidence transfer differs')
    result=json.loads((REPORT/(tag+'_evidence.json')).read_text())
    write(REPORT/(tag+'_summary.json'),dict(result,evidence_sha256=digest(REPORT/(tag+'_evidence.json')),job=job))
    print(json.dumps({k:v for k,v in result.items() if k not in ('events','by_control','tiled_shared_faces')},indent=2))


def launch_full(probe=True):
    tag='tiled_field_probe' if probe else 'tiled_field_full'
    if (REPORT/(tag+'_job.json')).exists():raise ValueError('Full-field invocation already registered')
    gate=json.loads((REPORT/'tiled_field_validation_summary.json').read_text())
    if (gate['status']!='REGIONAL_REFINEMENT_PASSED_LOCAL_ONLY' or gate['tiled_fine_entries_passed']!=168
            or gate['tiled_coarse_entries_matched']!=168 or not gate['tiled_shared_faces']):
        raise ValueError('Pilot physical-field/interface/precision gate not passed')
    if not probe:
        previous=json.loads((REPORT/'tiled_field_probe_summary.json').read_text())
        if previous['status']!='BOUNDED_FULL_LATTICE_THROUGHPUT_AND_RESOURCES_PASSED':
            raise ValueError('Complete-lattice probe not passed')
    # All earlier workers must have exited before a changed invocation/shard.
    for name in ('tiled_field_pilot','tiled_field_pilot_repaired','tiled_field_probe'):
        path=REPORT/(name+'_job.json')
        if path.exists():
            for w in json.loads(path.read_text())['workers']:
                if ssh(SERVER,f'if ps -p {w["pid"]} -o args= | grep -q run_tiled_field; then echo ACTIVE; fi'):
                    raise ValueError('Earlier physical tile worker still active')
    devices=(0,1,3,4)
    for d in devices:
        free,util=map(int,ssh(SERVER,f'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i {d}').split(','))
        if free<30000 or util>5:raise ValueError('Full-field GPU occupied')
    names=('generate_compton_tiled_field_v3.py','generate_compton_a_tile_pilot_v3.py',
        'generate_compton_a_column_v3.py','generate_compton_a_guard.py','plan_compton_a_tiles_v3.py',
        'compton_overlap_integrator.py','compton_cell_response_field.py','compton_boundary_quadrature.py',
        'geometry.py','compton_cartesian_patch.py','test_compton_tiled_production_v3.py')
    paths={n:HERE/n for n in names};paths['tiled_production_plan.json']=REPORT/'tiled_production_plan.json'
    paths.update({n:ROOT/n for n in ('compton_event_response.py','detector_csv.py')})
    hashes={n:digest(p) for n,p in paths.items()};key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/tiled_field_releases/'+key;ssh(SERVER,'mkdir -p -- '+shlex.quote(release))
    for n,path in paths.items():transfer(SERVER,path,release+'/'+n)
    tests=ssh(SERVER,'cd '+shlex.quote(release)+' && '+PYTHON+' -m unittest test_compton_tiled_production_v3 -v 2>&1')
    if '\nOK' not in tests or 'skipped' in tests:raise ValueError('Frozen continuation tests failed')
    (REPORT/(tag+'_tests.txt')).write_text(tests,encoding='utf-8')
    write(REPORT/(tag+'_deployment.json'),dict(release=release,code_sha256=hashes))
    output=REMOTE+'/A440_tiled_field';jobs=[];label='full_probe' if probe else 'full_batch1'
    for shard,d in enumerate(devices):
        script=DATA/(f'run_{tag}_s{shard}.sh');timeout='45m' if probe else '24h'
        maximum=2 if probe else 10720;stop=0 if probe else 72000
        script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+f'/{tag}_s{shard}.lock\nflock -n 9\n'+
            f'export CUDA_VISIBLE_DEVICES={d} PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4\ncd '+release+'\n'+
            'timeout --signal=TERM --kill-after=30s '+timeout+' '+PYTHON+' generate_compton_tiled_field_v3.py --root '+SERVER_ROOT+
            ' --plan '+release+'/tiled_production_plan.json --field '+REMOTE+'/A440_guard_field --output '+output+
            f' --phase full --shard {shard} --shards 4 --max-new-tiles {maximum} --cuda 0 --physical-gpu {d}'+
            f' --run-id {label} --stop-after-seconds {stop}\n',encoding='ascii',newline='\n')
        transfer(SERVER,script,release+'/'+script.name)
        pid=ssh(SERVER,'nohup bash '+release+'/'+script.name+' > '+REMOTE+f'/{tag}_s{shard}.log 2>&1 < /dev/null & echo $!')
        if not pid.isdigit():raise ValueError('Unknown full-field launch outcome; inspect before retry')
        jobs.append(dict(shard=shard,physical_gpu=d,pid=int(pid),launcher_name=script.name,launcher_sha256=digest(script)))
        write(REPORT/(tag+'_job.json'),dict(release=release,output=output,workers=jobs,run_id=label,
            total_expected_workers=4,maximum_new_tiles_per_worker=maximum,stop_between_tiles_seconds=stop,
            external_timeout_minutes=45 if probe else 1440,full_lattice_probe=probe,
            complete_field_production_started=not probe,total_required_tiles=10720,
            S2_generated=False,reconstruction_submitted=False))
    print(json.dumps(jobs,indent=2))


def fetch_probe():
    tag='tiled_field_probe';j=json.loads((REPORT/(tag+'_job.json')).read_text());results=[]
    for w in j['workers']:
        path=j['output']+f'/full_probe_shard{w["shard"]}_complete.json'
        expected=ssh(SERVER,'sha256sum '+path).split()[0];target=DATA/f'full_probe_shard{w["shard"]}.json'
        subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+path,str(target)],check=True)
        if digest(target)!=expected:raise ValueError('Full probe transfer differs')
        results.append(dict(evidence_sha256=expected,worker=w,result=json.loads(target.read_text())))
    if (sum(r['result']['new_tiles'] for r in results)!=8
            or not all(r['result']['actual_resources']['passed'] for r in results)):
        raise ValueError('Four-worker full-lattice throughput/resource probe not complete')
    write(REPORT/(tag+'_summary.json'),dict(status='BOUNDED_FULL_LATTICE_THROUGHPUT_AND_RESOURCES_PASSED',
        workers=results,new_tiles=8,new_transport_photons=0,reconstruction_permitted=False,S2_generated=False))
    print(json.dumps(dict(new_tiles=8,seconds=[r['result']['elapsed_seconds'] for r in results],
        resources=[r['result']['actual_resources'] for r in results]),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('pilot-launch','pilot-repair','status','fetch','validate','fetch-validation','full-probe','fetch-probe','full-launch'))
    a=p.parse_args();{'pilot-launch':launch_pilot,'pilot-repair':lambda:launch_pilot(repair=True),'status':status,'fetch':fetch,
        'validate':validate_pilot,'fetch-validation':fetch_validation,'full-probe':launch_full,
        'fetch-probe':fetch_probe,'full-launch':lambda:launch_full(probe=False)}[a.action]()
