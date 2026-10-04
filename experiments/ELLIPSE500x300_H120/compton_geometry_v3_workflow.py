"""Isolated, hash-checked stable geometry scan on existing immutable transport."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import tarfile
import subprocess
from first_scatter_pipeline import ssh,transfer,SERVER,SERVER_ROOT,SERVER_BASE,SERVER_STUDY

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=HERE/'generated/compton_response_geometry_v3'
REPORT=HERE/'reports/NEMA_Body_H60/compton_response_geometry_v3'
REMOTE=SERVER_BASE+'/generated/compton_response_geometry_v3'
PYTHON=SERVER_ROOT+'/.venv/bin/python'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def stage():
    paths={n:HERE/n for n in ('analyze_first_scatter.py','analyze_compton_geometry_v3.py',
        'validate_factors.py','test_compton_geometry_v3.py','test_compton_geometry_stability.py',
        'diagnose_compton_geometry_stability.py','compton_response_geometry_v3.json','geometry.py','config.json',
        'verify_compton_legacy_compatibility.py')}
    paths.update({n:ROOT/n for n in ('compton_event_response.py','detector_csv.py')})
    paths['geometry.npz']=HERE/'generated/Geometry/geometry.npz'
    hashes={n:digest(p) for n,p in paths.items()}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/releases/'+key
    DATA.mkdir(parents=True,exist_ok=True);archive=DATA/'scan_code.tar.gz'
    with tarfile.open(archive,'w:gz') as f:
        for n,p in paths.items():f.add(p,arcname=n)
    ssh(SERVER,'mkdir -p -- '+shlex.quote(release));transfer(SERVER,archive,release+'/code.tar.gz')
    ssh(SERVER,'tar --no-same-owner -xzf '+shlex.quote(release+'/code.tar.gz')+' -C '+shlex.quote(release))
    write(REPORT/'deployment.json',dict(release=release,code_sha256=hashes,archive_sha256=digest(archive)))
    print('STAGED',release)

def launch(device):
    if (REPORT/'scan_job.json').exists():raise ValueError('Scan already registered; inspect before any retry')
    release=json.loads((REPORT/'deployment.json').read_text())['release']
    active=ssh(SERVER,'pgrep -af '+shlex.quote('[a]nalyze_compton_geometry_v3.py')+' || true')
    if active:raise ValueError('A v3 scanner is active; do not duplicate it')
    free,util=map(int,ssh(SERVER,f'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i {device}').split(','))
    if free<30000 or util>5:raise ValueError('Requested device is occupied; do not touch its process')
    tests=ssh(SERVER,'cd '+shlex.quote(release)+' && CUDA_VISIBLE_DEVICES='+str(device)+' '+PYTHON+
        ' -m unittest test_compton_geometry_v3 -v 2>&1')
    if 'skipped' in tests or 'Ran 8 tests' not in tests:raise ValueError('Full CUDA tests not executed')
    (REPORT/'cuda_tests.txt').write_text(tests+'\n',encoding='utf-8');print(tests)
    compat=ssh(SERVER,'cd '+shlex.quote(release)+' && CUDA_VISIBLE_DEVICES='+str(device)+' '+PYTHON+
        ' verify_compton_legacy_compatibility.py --old-kernel '+SERVER_STUDY+'/releases/345b1974b85d06dd/compton_event_response.py'+
        ' --inputs '+SERVER_STUDY+'/analysis_inputs --factors '+SERVER_BASE+'/generated/FactorsCalibrated'+
        ' --geometry '+release+'/geometry.npz --config '+release+'/compton_response_geometry_v3.json'+
        ' --output '+REMOTE+'/legacy_kernel_compatibility.json')
    write(REPORT/'legacy_kernel_compatibility.json',json.loads(compat));print('LEGACY_KERNEL_COMPATIBILITY_PASSED')
    command=(PYTHON+' analyze_compton_geometry_v3.py --inputs '+SERVER_STUDY+'/analysis_inputs'+
        ' --factors '+SERVER_BASE+'/generated/FactorsCalibrated --geometry '+release+'/geometry.npz'+
        ' --output '+REMOTE+'/R1_analysis --study-config '+release+'/compton_response_geometry_v3.json'+
        ' --device cuda:0 --batch-size 32')
    script=DATA/'run_R1_scan.sh'
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+'/scan.lock\nflock -n 9\n'+
        'export CUDA_VISIBLE_DEVICES='+str(device)+' PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8\ncd '+release+'\n'+
        'timeout --signal=TERM --kill-after=30s 4h '+command+'\n'+
        'echo R1_SCAN_FINISHED > '+REMOTE+'/scan_finished.txt\n',encoding='ascii',newline='\n')
    transfer(SERVER,script,release+'/run_R1_scan.sh')
    pid=ssh(SERVER,'nohup bash '+shlex.quote(release+'/run_R1_scan.sh')+' > '+REMOTE+'/scan.log 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unknown launch outcome; inspect before retry')
    input_hash=ssh(SERVER,'sha256sum '+SERVER_STUDY+'/analysis_inputs/input_manifest.json').split()[0]
    write(REPORT/'scan_job.json',dict(pid=int(pid),server=SERVER,physical_gpu=device,release=release,
        input_manifest_sha256=input_hash,launcher_sha256=digest(script),new_transport_photons=0,timeout_hours=4))
    print('SCAN_PID',pid)

def status():
    tags=('regional_validation','regional_a_g0','regional_a_g1','regional_a_g2','tile_pilot','column_operator_cuda','column_operator_cpu','guard_column','overlap_benchmark_cuda','overlap_benchmark_batched',
          'guard_components','guard_sampling','overlap_benchmark_cached','overlap_benchmark','scan')
    jobs=[dict(tag=tag,**json.loads((REPORT/(tag+'_job.json')).read_text()))
          for tag in tags if (REPORT/(tag+'_job.json')).exists()]
    program='''import json,subprocess
jobs=json.loads(%r)
for j in jobs:
    p=subprocess.run(['ps','-p',str(j['pid']),'-o','pid,etime,stat,args'],capture_output=True,text=True)
    active=p.returncode==0
    result=dict(tag=j['tag'],pid=j['pid'],active=active,gpu=j['physical_gpu'])
    if active:
        result['process']=p.stdout.strip()
        q=subprocess.run(['tail','-n','4',%r+'/'+j['tag']+'.log'],capture_output=True,text=True)
        result['log_tail']=q.stdout.strip()
    print(json.dumps(result))
print(subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout)
'''%(json.dumps(jobs),REMOTE)
    print(ssh(SERVER,PYTHON+' -c '+shlex.quote(program)))

def guard(device):
    """One bounded original-binary halo, common-point gate first."""
    if (REPORT/'guard_job.json').exists():raise ValueError('Guard generation already registered')
    if ssh(SERVER,"pgrep -af '[g]enerate_compton_a_guard.py' || true"):
        raise ValueError('Guard producer already active')
    free,util=map(int,ssh(SERVER,f'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i {device}').split(','))
    if free<40000 or util>5:raise ValueError('Guard GPU is occupied')
    names=('generate_compton_a_guard.py','prepare_compton_overlap_measure.py','test_compton_a_guard.py',
        'compton_boundary_quadrature.py','geometry.py','config.json','compton_response_geometry_v3.json')
    paths={n:HERE/n for n in names};paths['geometry.npz']=HERE/'generated/Geometry/geometry.npz'
    hashes={n:digest(p) for n,p in paths.items()}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/guard_releases/'+key;archive=DATA/'guard_code.tar.gz'
    with tarfile.open(archive,'w:gz') as f:
        for n,path in paths.items():f.add(path,arcname=n)
    ssh(SERVER,'mkdir -p -- '+release);transfer(SERVER,archive,release+'/code.tar.gz')
    tests=ssh(SERVER,'tar --no-same-owner -xzf '+release+'/code.tar.gz -C '+release+' && cd '+release+
        ' && '+PYTHON+' -m unittest test_compton_a_guard -v 2>&1')
    if 'Ran 4 tests' not in tests or '\nOK' not in tests:raise ValueError('Guard tests not passed')
    (REPORT/'guard_tests.txt').write_text(tests+'\n',encoding='utf-8')
    write(REPORT/'guard_deployment.json',dict(release=release,code_sha256=hashes,archive_sha256=digest(archive)))
    script=DATA/'run_guard.sh';output=REMOTE+'/A440_guard'
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+'/guard.lock\nflock -n 9\n'+
        'export CUDA_VISIBLE_DEVICES='+str(device)+' PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8\ncd '+release+'\n'+
        'timeout --signal=TERM --kill-after=30s 4h '+PYTHON+' generate_compton_a_guard.py --root '+SERVER_ROOT+
        ' --output '+output+' --cuda 0\n'+
        'echo GUARD_GENERATION_FINISHED > '+REMOTE+'/guard_finished.txt\n',encoding='ascii',newline='\n')
    transfer(SERVER,script,release+'/run_guard.sh')
    pid=ssh(SERVER,'nohup bash '+release+'/run_guard.sh > '+REMOTE+'/guard.log 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unknown guard launch outcome; inspect before retry')
    write(REPORT/'guard_job.json',dict(pid=int(pid),physical_gpu=device,release=release,
        output=output,launcher_sha256=digest(script),timeout_hours=4,new_transport_photons=0,
        anchor_gate_required=True,production_imaging_permitted=False))
    print('GUARD_PID',pid)

def guard_field():
    """Wait on the registered producer using CPU only; no second GPU reservation."""
    if (REPORT/'guard_field_job.json').exists():raise ValueError('Guard field continuation already registered')
    job=json.loads((REPORT/'guard_job.json').read_text())
    names=('run_compton_guard_field_stage.py','build_compton_a_guard_field.py',
        'validate_compton_guard_field.py','generate_compton_a_guard.py','prepare_compton_overlap_measure.py',
        'test_compton_guard_field.py','test_compton_a_guard.py','compton_boundary_quadrature.py',
        'geometry.py','config.json')
    paths={n:HERE/n for n in names};paths['generated/Geometry/geometry.npz']=HERE/'generated/Geometry/geometry.npz'
    hashes={n:digest(p) for n,p in paths.items()}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/guard_field_releases/'+key;archive=DATA/'guard_field_code.tar.gz'
    with tarfile.open(archive,'w:gz') as f:
        for n,path in paths.items():f.add(path,arcname=n)
    ssh(SERVER,'mkdir -p -- '+release);transfer(SERVER,archive,release+'/code.tar.gz')
    tests=ssh(SERVER,'tar --no-same-owner -xzf '+release+'/code.tar.gz -C '+release+' && cd '+release+
        ' && '+PYTHON+' -m unittest test_compton_a_guard test_compton_guard_field -v 2>&1')
    if 'Ran 7 tests' not in tests or '\nOK' not in tests:raise ValueError('Guard field tests failed')
    (REPORT/'guard_field_tests.txt').write_text(tests+'\n',encoding='utf-8')
    write(REPORT/'guard_field_deployment.json',dict(release=release,code_sha256=hashes,archive_sha256=digest(archive)))
    script=DATA/'run_guard_field.sh';output=REMOTE+'/A440_guard_field'
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+'/guard_field.lock\nflock -n 9\n'+
        'export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8\ncd '+release+'\n'+
        'timeout --signal=TERM --kill-after=30s 2h '+PYTHON+' run_compton_guard_field_stage.py --root '+SERVER_ROOT+
        ' --guard '+job['output']+' --factors '+SERVER_BASE+'/generated/FactorsCalibrated/440keV_RotateNum20'+
        ' --geometry '+release+'/generated/Geometry/geometry.npz --config '+release+'/config.json'+
        ' --output '+output+' --producer-pid '+str(job['pid'])+' --wait-seconds 2400\n'+
        'echo GUARD_FIELD_DIAGNOSTICS_FINISHED > '+REMOTE+'/guard_field_finished.txt\n',encoding='ascii',newline='\n')
    transfer(SERVER,script,release+'/run_guard_field.sh')
    pid=ssh(SERVER,'nohup bash '+release+'/run_guard_field.sh > '+REMOTE+'/guard_field.log 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unknown field launch outcome')
    write(REPORT/'guard_field_job.json',dict(pid=int(pid),physical_gpu=None,release=release,
        output=output,depends_on_pid=job['pid'],launcher_sha256=digest(script),timeout_hours=2,
        new_transport_photons=0,diagnostic_only=True,reconstruction_submitted=False))
    print('GUARD_FIELD_PID',pid)

def guard_radial(device):
    if (REPORT/'guard_radial_job.json').exists():raise ValueError('Radial check already registered')
    job=json.loads((REPORT/'guard_job.json').read_text())
    field=json.loads((REPORT/'guard_field_job.json').read_text())
    names=('run_compton_radial_guard_stage.py','validate_compton_guard_field.py',
        'build_compton_a_guard_field.py','generate_compton_a_guard.py','compton_boundary_quadrature.py','geometry.py')
    paths={n:HERE/n for n in names};hashes={n:digest(p) for n,p in paths.items()}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/guard_radial_releases/'+key;archive=DATA/'guard_radial_code.tar.gz'
    with tarfile.open(archive,'w:gz') as f:
        for n,path in paths.items():f.add(path,arcname=n)
    ssh(SERVER,'mkdir -p -- '+release);transfer(SERVER,archive,release+'/code.tar.gz')
    ssh(SERVER,'tar --no-same-owner -xzf '+release+'/code.tar.gz -C '+release+
        ' && cd '+release+' && '+PYTHON+' -m py_compile '+' '.join(names))
    write(REPORT/'guard_radial_deployment.json',dict(release=release,code_sha256=hashes,archive_sha256=digest(archive)))
    script=DATA/'run_guard_radial.sh';output=REMOTE+'/A440_radial_check'
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+'/guard_radial.lock\nflock -n 9\n'+
        'export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8\ncd '+release+'\n'+
        'timeout --signal=TERM --kill-after=30s 2h '+PYTHON+' run_compton_radial_guard_stage.py --root '+SERVER_ROOT+
        ' --field '+field['output']+' --guard '+job['output']+' --output '+output+
        ' --producer-pid '+str(job['pid'])+' --gpu '+str(device)+'\n'+
        'echo RADIAL_INTERPOLATION_DIAGNOSTICS_FINISHED > '+REMOTE+'/guard_radial_finished.txt\n',encoding='ascii',newline='\n')
    transfer(SERVER,script,release+'/run_guard_radial.sh')
    pid=ssh(SERVER,'nohup bash '+release+'/run_guard_radial.sh > '+REMOTE+'/guard_radial.log 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unknown radial launch outcome')
    write(REPORT/'guard_radial_job.json',dict(pid=int(pid),physical_gpu=device,release=release,
        output=output,depends_on_guard_pid=job['pid'],depends_on_field_pid=field['pid'],
        launcher_sha256=digest(script),timeout_hours=2,new_transport_photons=0,
        waiting_device_reserved=False,diagnostic_only=True,reconstruction_submitted=False))
    print('GUARD_RADIAL_PID',pid)

def fetch_guard():
    """Small, hash-checked evidence only; never download multi-GB A fields."""
    jobs={n:json.loads((REPORT/('guard'+n+'_job.json')).read_text())
          for n in ('','_field','_radial') if (REPORT/('guard'+n+'_job.json')).exists()}
    if '' not in jobs:raise ValueError('No registered guard producer')
    files={'anchor_gate.json':jobs['']['output']+'/anchor/anchor_gate.json',
        'guard_progress.json':jobs['']['output']+'/progress.json',
        'guard_ready.json':jobs['']['output']+'/guard_ready.json'}
    if '_field' in jobs:
        base=jobs['_field']['output'];files.update({'field_manifest.json':base+'/field_manifest.json',
            'midpoint_gate.json':base+'/midpoint_gate.json','field_stage_complete.json':base+'/stage_complete.json',
            'measure_manifest.json':base+'/precise_measure/measure_manifest.json'})
    if '_radial' in jobs:files['radial_midpoint_gate.json']=jobs['_radial']['output']+'/radial_midpoint_gate.json'
    repaired=REPORT/'guard_measure_job.json'
    if repaired.exists():
        job=json.loads(repaired.read_text());jobs['_measure']=job;base=job['output']
        files.update({'midpoint_gate.json':base+'/midpoint_gate.json',
            'field_stage_complete.json':base+'/stage_complete.json',
            'measure_manifest.json':base+'/precise_measure/measure_manifest.json'})
    weighted=REPORT/'guard_weighted_job.json'
    if weighted.exists():
        job=json.loads(weighted.read_text());jobs['_weighted']=job
        files['weighted_interpolation.json']=job['output']+'/weighted_interpolation.json'
    target=DATA/'guard_evidence';target.mkdir(exist_ok=True);found={}
    for name,path in files.items():
        checksum=ssh(SERVER,'if test -f '+shlex.quote(path)+'; then sha256sum -- '+shlex.quote(path)+'; fi')
        if not checksum:continue
        expected=checksum.split()[0]
        subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+path,str(target/name)],check=True)
        if digest(target/name)!=expected:raise ValueError('Guard evidence transfer hash differs')
        found[name]=dict(sha256=expected,value=json.loads((target/name).read_text()))
    anchor=found.get('anchor_gate.json',{}).get('value',{})
    for name in ('anchor_gate.json','measure_manifest.json','midpoint_gate.json','radial_midpoint_gate.json','weighted_interpolation.json'):
        if name in found:write(REPORT/name,found[name]['value'])
    snapshot=ssh(SERVER,'date -u +%FT%TZ; ps -p '+','.join(str(j['pid']) for j in jobs.values())+
        ' -o pid,etime,stat,args; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i '+
        str(jobs['']['physical_gpu']))
    progress=found.get('guard_progress.json',{}).get('value',{})
    summary=dict(anchor_gate=anchor.get('status','PENDING'),
        completed_parts=progress.get('completed_parts',0),total_parts=9,
        guard_ready='guard_ready.json' in found,field_built='field_manifest.json' in found,
        midpoint_gate=found.get('midpoint_gate.json',{}).get('value',{}).get('status','PENDING'),
        radial_midpoint_gate=found.get('radial_midpoint_gate.json',{}).get('value',{}).get('status','PENDING'),
        precise_measure_gate=found.get('measure_manifest.json',{}).get('value',{}).get('status','PENDING'),
        evidence_sha256={n:v['sha256'] for n,v in found.items()},live_snapshot=snapshot,
        R2_boundary_gate='HOLD_PENDING_FULL_VALIDATION',S2_generated=False,reconstruction_submitted=False,
        new_transport_photons=0)
    write(REPORT/'guard_summary.json',summary);print(json.dumps(summary,indent=2))

def repair_guard_measure():
    if (REPORT/'guard_measure_job.json').exists():raise ValueError('CPU diagnostic repair already registered')
    old=json.loads((REPORT/'guard_field_job.json').read_text())
    if ssh(SERVER,'ps -p '+str(old['pid'])+' -o args= || true'):raise ValueError('Original CPU stage still active')
    guard=json.loads((REPORT/'guard_job.json').read_text())
    if ssh(SERVER,'test -f '+old['output']+'/field_manifest.json && echo READY')!='READY':
        raise ValueError('Immutable A field has not completed')
    names=('repair_compton_guard_measure.py','prepare_compton_overlap_measure.py','validate_compton_guard_field.py',
        'build_compton_a_guard_field.py','generate_compton_a_guard.py','compton_boundary_quadrature.py',
        'test_compton_a_guard.py','geometry.py','config.json')
    paths={n:HERE/n for n in names};paths['geometry.npz']=HERE/'generated/Geometry/geometry.npz'
    hashes={n:digest(p) for n,p in paths.items()}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/guard_measure_releases/'+key;archive=DATA/'guard_measure_code.tar.gz'
    with tarfile.open(archive,'w:gz') as f:
        for n,path in paths.items():f.add(path,arcname=n)
    ssh(SERVER,'mkdir -p -- '+release);transfer(SERVER,archive,release+'/code.tar.gz')
    tests=ssh(SERVER,'tar --no-same-owner -xzf '+release+'/code.tar.gz -C '+release+' && cd '+release+
        ' && '+PYTHON+' -m unittest test_compton_a_guard -v 2>&1')
    if 'Ran 4 tests' not in tests or '\nOK' not in tests:raise ValueError('Roundoff repair tests failed')
    (REPORT/'guard_measure_tests.txt').write_text(tests+'\n',encoding='utf-8')
    write(REPORT/'guard_measure_deployment.json',dict(release=release,code_sha256=hashes,archive_sha256=digest(archive)))
    script=DATA/'run_guard_measure_repair.sh';output=REMOTE+'/A440_guard_field_repair'
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+'/guard_measure.lock\nflock -n 9\n'+
        'export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8\ncd '+release+'\n'+
        'timeout --signal=TERM --kill-after=30s 30m '+PYTHON+' repair_compton_guard_measure.py --field '+old['output']+
        ' --guard '+guard['output']+' --geometry '+release+'/geometry.npz --config '+release+'/config.json --output '+output+'\n',
        encoding='ascii',newline='\n')
    transfer(SERVER,script,release+'/run_guard_measure_repair.sh')
    pid=ssh(SERVER,'nohup bash '+release+'/run_guard_measure_repair.sh > '+REMOTE+'/guard_measure.log 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unknown CPU repair launch outcome')
    write(REPORT/'guard_measure_job.json',dict(pid=int(pid),physical_gpu=None,release=release,output=output,
        reused_field=old['output'],failed_cpu_pid=old['pid'],launcher_sha256=digest(script),timeout_minutes=30,
        reason='Cross-platform libm coordinate differences up to 2.84e-14 mm; physical tolerance 1e-10 mm, input SHA unchanged',
        original_A_read_only=True,new_transport_photons=0,reconstruction_submitted=False))
    print('GUARD_MEASURE_REPAIR_PID',pid)

def guard_weighted(device):
    if (REPORT/'guard_weighted_job.json').exists():raise ValueError('Weighted diagnostic already registered')
    free,util=map(int,ssh(SERVER,f'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i {device}').split(','))
    if free<40000 or util>5:raise ValueError('Weighted diagnostic GPU occupied')
    names=('diagnose_compton_guard_weighting.py','build_compton_a_guard_field.py',
           'generate_compton_a_guard.py','compton_boundary_quadrature.py','geometry.py')
    paths={n:HERE/n for n in names};paths.update({n:ROOT/n for n in ('compton_event_response.py','detector_csv.py')})
    paths['geometry.npz']=HERE/'generated/Geometry/geometry.npz';hashes={n:digest(p) for n,p in paths.items()}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/guard_weighted_releases/'+key;archive=DATA/'guard_weighted_code.tar.gz'
    with tarfile.open(archive,'w:gz') as f:
        for n,path in paths.items():f.add(path,arcname=n)
    ssh(SERVER,'mkdir -p -- '+release);transfer(SERVER,archive,release+'/code.tar.gz')
    ssh(SERVER,'tar --no-same-owner -xzf '+release+'/code.tar.gz -C '+release+' && cd '+release+
        ' && '+PYTHON+' -m py_compile diagnose_compton_guard_weighting.py')
    write(REPORT/'guard_weighted_deployment.json',dict(release=release,code_sha256=hashes,archive_sha256=digest(archive)))
    script=DATA/'run_guard_weighted.sh';output=REMOTE+'/guard_weighted'
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+'/guard_weighted.lock\nflock -n 9\n'+
        'export CUDA_VISIBLE_DEVICES='+str(device)+' PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8\ncd '+release+'\n'+
        'timeout --signal=TERM --kill-after=30s 10m '+PYTHON+' diagnose_compton_guard_weighting.py --field '+REMOTE+'/A440_guard_field'+
        ' --guard '+REMOTE+'/A440_guard --radial '+REMOTE+'/A440_radial_check --inputs '+SERVER_STUDY+'/analysis_inputs'+
        ' --factors '+SERVER_BASE+'/generated/FactorsCalibrated --geometry '+release+'/geometry.npz --output '+output+'\n',
        encoding='ascii',newline='\n')
    transfer(SERVER,script,release+'/run_guard_weighted.sh')
    pid=ssh(SERVER,'nohup bash '+release+'/run_guard_weighted.sh > '+REMOTE+'/guard_weighted.log 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unknown weighted diagnostic launch outcome')
    write(REPORT/'guard_weighted_job.json',dict(pid=int(pid),physical_gpu=device,release=release,output=output,
        launcher_sha256=digest(script),timeout_minutes=10,diagnostic_only=True,new_transport_photons=0,reconstruction_submitted=False))
    print('GUARD_WEIGHTED_PID',pid)

def guard_integral(device, patch=False, refined=False, ultrafine=False):
    """Bounded whole-cell convergence or independent local A refinement."""
    tag='guard_patch_ultrafine' if ultrafine else ('guard_patch_refined' if refined else ('guard_patch' if patch else 'guard_integral'))
    script_name='run_compton_patch_guard.py' if patch else 'validate_compton_integral_guard.py'
    if (REPORT/(tag+'_job.json')).exists():raise ValueError('Integral diagnostic already registered')
    if ssh(SERVER,'pgrep -af '+shlex.quote('['+script_name[0]+']'+script_name[1:])+' || true'):
        raise ValueError('This integral diagnostic is already active')
    free,util=map(int,ssh(SERVER,f'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i {device}').split(','))
    if free<40000 or util>5:raise ValueError('Integral diagnostic GPU occupied')
    names=('run_compton_patch_guard.py','validate_compton_integral_guard.py','compton_cartesian_patch.py',
        'test_compton_cartesian_patch.py','build_compton_a_guard_field.py','generate_compton_a_guard.py',
        'compton_boundary_quadrature.py','geometry.py','config.json')
    paths={n:HERE/n for n in names};paths.update({n:ROOT/n for n in ('compton_event_response.py','detector_csv.py')})
    paths['geometry.npz']=HERE/'generated/Geometry/geometry.npz';hashes={n:digest(p) for n,p in paths.items()}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/'+tag+'_releases/'+key;archive=DATA/(tag+'_code.tar.gz')
    with tarfile.open(archive,'w:gz') as f:
        for n,path in paths.items():f.add(path,arcname=n)
    ssh(SERVER,'mkdir -p -- '+release);transfer(SERVER,archive,release+'/code.tar.gz')
    tests=ssh(SERVER,'tar --no-same-owner -xzf '+release+'/code.tar.gz -C '+release+' && cd '+release+
        ' && '+PYTHON+' -m unittest test_compton_cartesian_patch -v 2>&1')
    if 'Ran 4 tests' not in tests or '\nOK' not in tests:raise ValueError('Independent patch tests failed')
    (REPORT/(tag+'_tests.txt')).write_text(tests+'\n',encoding='utf-8')
    write(REPORT/(tag+'_deployment.json'),dict(release=release,code_sha256=hashes,archive_sha256=digest(archive)))
    output=REMOTE+('/A440_near_patch_ultrafine' if ultrafine else ('/A440_near_patch_refined' if refined else ('/A440_near_patch' if patch else '/guard_integral_validation')))
    command=PYTHON+' '+script_name+' --field '+REMOTE+'/A440_guard_field --geometry '+release+'/geometry.npz'+\
        ' --config '+release+'/config.json --measure '+REMOTE+'/A440_guard_field_repair/precise_measure'+\
        ' --inputs '+SERVER_STUDY+'/analysis_inputs --factors '+SERVER_BASE+'/generated/FactorsCalibrated --output '+output
    if patch:command+=' --root '+SERVER_ROOT+' --guard '+REMOTE+'/A440_guard --cuda 0'
    if refined:command+=' --previous-patches '+REMOTE+'/A440_near_patch'
    if ultrafine:command+=' --intermediate-patches '+REMOTE+'/A440_near_patch_refined'
    script=DATA/('run_'+tag+'.sh')
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+'/'+tag+'.lock\nflock -n 9\n'+
        'export CUDA_VISIBLE_DEVICES='+str(device)+' PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8\ncd '+release+'\n'+
        'timeout --signal=TERM --kill-after=30s 45m '+command+'\n',encoding='ascii',newline='\n')
    transfer(SERVER,script,release+'/run_'+tag+'.sh')
    pid=ssh(SERVER,'nohup bash '+release+'/run_'+tag+'.sh > '+REMOTE+'/'+tag+'.log 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unknown integral diagnostic launch outcome')
    write(REPORT/(tag+'_job.json'),dict(pid=int(pid),physical_gpu=device,release=release,output=output,
        launcher_sha256=digest(script),timeout_minutes=45,diagnostic_only=True,
        new_transport_photons=0,reconstruction_submitted=False))
    print(tag.upper()+'_PID',pid)


def fetch_integrals():
    import csv
    tags=['guard_integral','guard_patch']
    if (REPORT/'guard_patch_refined_job.json').exists():tags.append('guard_patch_refined')
    if (REPORT/'guard_patch_ultrafine_job.json').exists():tags.append('guard_patch_ultrafine')
    for tag in tags:
        job=json.loads((REPORT/(tag+'_job.json')).read_text())
        source=job['output']+('/integrals' if tag.startswith('guard_patch') else '')
        target=DATA/tag;target.mkdir(exist_ok=True)
        for name in ('integral_gate.json','integral_cases.csv'):
            path=source+'/'+name;expected=ssh(SERVER,'sha256sum '+path).split()[0]
            subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+path,str(target/name)],check=True)
            if digest(target/name)!=expected:raise ValueError('Integral diagnostic transfer differs')
        gate=json.loads((target/'integral_gate.json').read_text())
        if digest(target/'integral_cases.csv')!=gate['output_sha256']:raise ValueError('Case receipt differs')
        rows=list(csv.DictReader((target/'integral_cases.csv').open()))
        gate['cases_sha256']=digest(target/'integral_cases.csv')
        gate['evidence_sha256']=digest(target/'integral_gate.json')
        if tag.startswith('guard_patch'):
            gate['maximum_patch_refinement_relative_change']=max(float(r['patch_refinement_relative_change']) for r in rows)
            gate['maximum_guard_to_patch_relative_change']=max(abs(float(r['guard_to_patch_relative_change'])) for r in rows)
            gate['guard_to_patch_cases_over_1percent']=sum(abs(float(r['guard_to_patch_relative_change']))>.01 for r in rows)
        write(REPORT/(tag+'_summary.json'),gate)
        print(json.dumps({k:v for k,v in gate.items() if k not in ('events','predefined_cells')},indent=2))


def repair(device):
    old=json.loads((REPORT/'scan_job.json').read_text());new=json.loads((REPORT/'deployment.json').read_text())
    if old['release']==new['release']:raise ValueError('Repair requires a new frozen release')
    if ssh(SERVER,'ps -p '+str(old['pid'])+' -o args= || true'):
        raise ValueError('Previous launcher has not exited')
    if ssh(SERVER,'pgrep -af '+shlex.quote('[a]nalyze_compton_geometry_v3.py')+' || true'):
        raise ValueError('Previous scan remains active')
    # This startup failure occurred before output creation. Preserve evidence;
    # never retry into a partially produced output directory.
    ssh(SERVER,'test ! -d '+REMOTE+'/R1_analysis && cp -- '+REMOTE+'/scan.log '+REMOTE+'/scan.failed_'+str(old['pid'])+'.log')
    write(REPORT/('scan_job.failed_'+str(old['pid'])+'.json'),old)
    (REPORT/'scan_job.json').unlink()
    launch(device)

def boundary(device):
    if (REPORT/'boundary_job.json').exists():raise ValueError('Boundary validation already registered')
    free,util=map(int,ssh(SERVER,f'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i {device}').split(','))
    if free<30000 or util>5:raise ValueError('Boundary device is occupied')
    names=('validate_compton_boundary_v3.py','compton_boundary_quadrature.py','test_compton_boundary_v3.py',
           'torch_active_operator.py','geometry.py','config.json','compton_response_geometry_v3.json')
    paths={n:HERE/n for n in names};paths.update({n:ROOT/n for n in ('compton_event_response.py','detector_csv.py')})
    paths['geometry.npz']=HERE/'generated/Geometry/geometry.npz'
    hashes={n:digest(p) for n,p in paths.items()}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/boundary_releases/'+key;archive=DATA/'boundary_code.tar.gz'
    with tarfile.open(archive,'w:gz') as f:
        for n,path in paths.items():f.add(path,arcname=n)
    ssh(SERVER,'mkdir -p -- '+release);transfer(SERVER,archive,release+'/code.tar.gz')
    tests=ssh(SERVER,'tar --no-same-owner -xzf '+release+'/code.tar.gz -C '+release+' && cd '+release+
        ' && '+PYTHON+' -m unittest test_compton_boundary_v3 -v 2>&1')
    (REPORT/'boundary_tests.txt').write_text(tests+'\n',encoding='utf-8')
    write(REPORT/'boundary_deployment.json',dict(release=release,code_sha256=hashes,archive_sha256=digest(archive)))
    command=(PYTHON+' validate_compton_boundary_v3.py --inputs '+SERVER_STUDY+'/analysis_inputs'+
        ' --factors '+SERVER_BASE+'/generated/FactorsCalibrated --geometry '+release+'/geometry.npz'+
        ' --grid-config '+release+'/config.json --study-config '+release+'/compton_response_geometry_v3.json'+
        ' --output '+REMOTE+'/boundary_validation --device cuda:0')
    script=DATA/'run_boundary_gate.sh'
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+'/boundary.lock\nflock -n 9\n'+
        'export CUDA_VISIBLE_DEVICES='+str(device)+' PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8\ncd '+release+'\n'+
        'timeout --signal=TERM --kill-after=30s 2h '+command+'\n'+
        'echo BOUNDARY_FINISHED > '+REMOTE+'/boundary_finished.txt\n',encoding='ascii',newline='\n')
    transfer(SERVER,script,release+'/run_boundary_gate.sh')
    pid=ssh(SERVER,'nohup bash '+release+'/run_boundary_gate.sh > '+REMOTE+'/boundary.log 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unknown boundary launch outcome')
    write(REPORT/'boundary_job.json',dict(pid=int(pid),physical_gpu=device,release=release,
        launcher_sha256=digest(script),new_transport_photons=0,diagnostic_only=True,timeout_hours=2))
    print('BOUNDARY_PID',pid)

def fetch_boundary():
    target=DATA/'boundary_validation';target.mkdir(exist_ok=True)
    hashes={}
    for name in ('boundary_gate.json','response_cases.csv'):
        path=REMOTE+'/boundary_validation/'+name
        expected=ssh(SERVER,'sha256sum '+path).split()[0]
        subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+path,str(target/name)],check=True)
        if digest(target/name)!=expected:raise ValueError('Boundary evidence transfer differs')
        hashes[name]=expected
    gate=json.loads((target/'boundary_gate.json').read_text())
    summary={k:v for k,v in gate.items() if k not in ('volume_rows','transverse_missing')}
    summary['volume_max_relative_error']=max(abs(r['relative_error']) for r in gate['volume_rows'])
    summary['full_reference_unsupported_cells_per_layer']=len(gate['transverse_missing'])
    summary['evidence_sha256']=hashes
    write(REPORT/'boundary_summary.json',summary);print(json.dumps(summary))

def fetch_scan():
    ssh(SERVER,'test -f '+REMOTE+'/scan_finished.txt && tar -czf '+REMOTE+'/R1_analysis.tar.gz -C '+REMOTE+' R1_analysis')
    archive=DATA/'R1_analysis.tar.gz'
    expected=ssh(SERVER,'sha256sum '+REMOTE+'/R1_analysis.tar.gz').split()[0]
    subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+REMOTE+'/R1_analysis.tar.gz',str(archive)],check=True)
    if digest(archive)!=expected:raise ValueError('R1 archive differs')
    with tarfile.open(archive) as f:
        if not (DATA/'R1_analysis').exists():f.extractall(DATA,filter='data')
        else:
            for member in f.getmembers():
                if member.isfile() and hashlib.sha256(f.extractfile(member).read()).hexdigest()!=digest(DATA/member.name):
                    raise ValueError('Existing retrieved evidence differs; never overwrite it')
    gate=json.loads((DATA/'R1_analysis/validation_gate.json').read_text())
    old=HERE/'generated/compton_first_scatter_v2/analysis'
    import numpy as np
    a=np.load(old/'NEMA_ideal_event_identities.npy');b=np.load(DATA/'R1_analysis/NEMA_ideal_event_identities.npy')
    old_ids={tuple(map(int,r)) for r in a};new_ids={tuple(map(int,r)) for r in b}
    changed=dict(added=sorted(new_ids-old_ids),removed=sorted(old_ids-new_ids),common=len(old_ids&new_ids))
    write(REPORT/'event_identity_delta.json',changed)
    summary=dict(status=gate['status'],scans={k:{name:r[name] for name in ('uncut','kept','primary_counts')} for k,r in gate['scans'].items()},
        gates=gate['gates'],geometry_sha256=gate['geometry_sha256'],kernel_sha256=gate['kernel_sha256'],
        archive_sha256=expected,validation_gate_sha256=digest(DATA/'R1_analysis/validation_gate.json'),
        sensitivity_sha256=digest(DATA/'R1_analysis/ideal/Sensi_d'),
        sensitivity_provenance_sha256=digest(DATA/'R1_analysis/ideal/Sensi_d_provenance.json'),
        added_events=len(changed['added']),removed_events=len(changed['removed']),new_transport_photons=0)
    write(REPORT/'R1_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('gates','scans')}));print('NEMA',summary['scans']['NEMA_ideal'])

def spatial(device,repair=False):
    old=None
    if (REPORT/'spatial_job.json').exists():
        if not repair:raise ValueError('Spatial gate already registered')
        old=json.loads((REPORT/'spatial_job.json').read_text())
        if ssh(SERVER,'ps -p '+str(old['pid'])+' -o args= || true'):raise ValueError('Prior spatial gate has not exited')
    free,util=map(int,ssh(SERVER,f'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i {device}').split(','))
    if free<30000 or util>5:raise ValueError('Spatial device is occupied')
    names=('validate_compton_spatial_v3.py','analyze_first_scatter.py','validate_factors.py','geometry.py','config.json')
    paths={n:HERE/n for n in names};paths.update({n:ROOT/n for n in ('compton_event_response.py','detector_csv.py')})
    paths['geometry.npz']=HERE/'generated/Geometry/geometry.npz'
    hashes={n:digest(p) for n,p in paths.items()};key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/spatial_releases/'+key;archive=DATA/'spatial_code.tar.gz'
    if old and old['release']==release:raise ValueError('A frozen repair release is required')
    output=REMOTE+('/spatial_validation_v2' if repair else '/spatial_validation')
    with tarfile.open(archive,'w:gz') as f:
        for n,path in paths.items():f.add(path,arcname=n)
    ssh(SERVER,'mkdir -p -- '+release);transfer(SERVER,archive,release+'/code.tar.gz')
    ssh(SERVER,'tar --no-same-owner -xzf '+release+'/code.tar.gz -C '+release+' && test -f '+REMOTE+'/scan_finished.txt')
    write(REPORT/'spatial_deployment.json',dict(release=release,code_sha256=hashes,archive_sha256=digest(archive)))
    command=(PYTHON+' validate_compton_spatial_v3.py --inputs '+SERVER_STUDY+'/analysis_inputs --analysis '+REMOTE+'/R1_analysis'+
        ' --factors '+SERVER_BASE+'/generated/FactorsCalibrated --geometry '+release+'/geometry.npz --grid-config '+release+'/config.json'+
        ' --output '+output+' --device cuda:0')
    script=DATA/'run_spatial_gate.sh'
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+'/spatial.lock\nflock -n 9\n'+
        'export CUDA_VISIBLE_DEVICES='+str(device)+' PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8\ncd '+release+'\n'+
        'timeout --signal=TERM --kill-after=30s 1h '+command+'\n'+
        'echo SPATIAL_FINISHED > '+REMOTE+'/spatial_finished.txt\n',encoding='ascii',newline='\n')
    transfer(SERVER,script,release+'/run_spatial_gate.sh')
    pid=ssh(SERVER,'nohup bash '+release+'/run_spatial_gate.sh > '+REMOTE+'/spatial.log 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unknown spatial launch outcome')
    if old:
        write(REPORT/('spatial_job.superseded_'+str(old['pid'])+'.json'),old)
        if (REPORT/'spatial_summary.json').exists():
            invalid=json.loads((REPORT/'spatial_summary.json').read_text())
            invalid['superseded_reason']='Direct-source labels omitted the 20-view rotation average used by S; invalid spatial comparison, not scientific gate evidence'
            write(REPORT/'spatial_unrotated_diagnostic.json',invalid)
    write(REPORT/'spatial_job.json',dict(pid=int(pid),physical_gpu=device,release=release,launcher_sha256=digest(script),timeout_hours=1))
    job=json.loads((REPORT/'spatial_job.json').read_text());job['output']=output;write(REPORT/'spatial_job.json',job)
    print('SPATIAL_PID',pid)

def fetch_spatial():
    job=json.loads((REPORT/'spatial_job.json').read_text())
    path=job.get('output',REMOTE+'/spatial_validation')+'/spatial_gate.json';expected=ssh(SERVER,'sha256sum '+path).split()[0]
    target=DATA/'spatial_validation';target.mkdir(exist_ok=True)
    subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+path,str(target/'spatial_gate.json')],check=True)
    if digest(target/'spatial_gate.json')!=expected:raise ValueError('Spatial evidence transfer differs')
    value=json.loads((target/'spatial_gate.json').read_text());value['evidence_sha256']=expected
    write(REPORT/'spatial_summary.json',value);print(json.dumps(value))

def overlap_benchmark(device,cached=False,batched=False,cuda=False,column=False):
    tag=('column_operator_cuda' if cuda else 'column_operator_cpu') if column else ('overlap_benchmark_cuda' if cuda else ('overlap_benchmark_batched' if batched else ('overlap_benchmark_cached' if cached else 'overlap_benchmark')))
    if (REPORT/(tag+'_job.json')).exists():raise ValueError('Full-cell benchmark already registered')
    active=ssh(SERVER,"pgrep -af '[b]enchmark_compton_overlap_v3.py' || true")
    if active and not (cuda and all('overlap_benchmark_batched_releases/' in line for line in active.splitlines())):
        raise ValueError('A full-cell benchmark is already active')
    free,util=map(int,ssh(SERVER,f'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i {device}').split(','))
    if free<40000 or util>5:raise ValueError('Benchmark GPU is occupied')
    names=('benchmark_compton_overlap_v3.py','compton_overlap_integrator.py','compton_cell_response_field.py','compton_overlap_cuda.py','test_compton_overlap_cuda.py',
        'compton_overlap_assembly.py','test_compton_overlap_integrator.py','test_compton_overlap_assembly.py',
        'prepare_compton_overlap_measure.py','build_compton_a_guard_field.py','generate_compton_a_guard.py',
        'compton_cartesian_patch.py','compton_boundary_quadrature.py','geometry.py','config.json')
    paths={n:HERE/n for n in names}
    paths.update({n:ROOT/n for n in ('compton_event_response.py','detector_csv.py')})
    paths['geometry.npz']=HERE/'generated/Geometry/geometry.npz';hashes={n:digest(p) for n,p in paths.items()}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/'+tag+'_releases/'+key;archive=DATA/(tag+'_code.tar.gz')
    with tarfile.open(archive,'w:gz') as f:
        for n,path in paths.items():f.add(path,arcname=n)
    ssh(SERVER,'mkdir -p -- '+shlex.quote(release));transfer(SERVER,archive,release+'/code.tar.gz')
    tests=ssh(SERVER,'tar --no-same-owner -xzf '+shlex.quote(release+'/code.tar.gz')+' -C '+shlex.quote(release)+
        ' && cd '+shlex.quote(release)+' && CUDA_VISIBLE_DEVICES='+str(device)+' '+PYTHON+' -m unittest test_compton_overlap_integrator test_compton_overlap_assembly test_compton_overlap_cuda -v 2>&1')
    if 'Ran 12 tests' not in tests or '\nOK' not in tests:raise ValueError('Full-cell integration tests failed')
    (REPORT/(tag+'_tests.txt')).write_text(tests+'\n',encoding='utf-8')
    write(REPORT/(tag+'_deployment.json'),dict(release=release,code_sha256=hashes,archive_sha256=digest(archive)))
    output=REMOTE+'/'+tag
    command=PYTHON+' benchmark_compton_overlap_v3.py --field '+REMOTE+'/A440_guard_field'+\
        ' --patches '+REMOTE+'/A440_near_patch_ultrafine --geometry '+release+'/geometry.npz --config '+release+'/config.json'+\
        ' --measure '+REMOTE+'/A440_guard_field_repair/precise_measure --inputs '+SERVER_STUDY+'/analysis_inputs'+\
        ' --factors '+SERVER_BASE+'/generated/FactorsCalibrated --scan '+REMOTE+'/R1_analysis --output '+output
    if column:
        command+=' --column '+REMOTE+'/A440_near_column --views 6 16'
        if cuda:command+=' --device-integrate --event-count 8 --reference-benchmark '+REMOTE+'/column_operator_cpu'
        else:command+=' --event-count 2'
    elif cuda:command+=' --device-integrate --event-count 64 --views 1 6 --reference-benchmark '+REMOTE+'/overlap_benchmark_cached'
    elif batched:command+=' --event-count 64 --views 1 6 --reference-benchmark '+REMOTE+'/overlap_benchmark_cached'
    elif cached:command+=' --event-count 8 --views 1 6 11 16 --reference-benchmark '+REMOTE+'/overlap_benchmark'
    script=DATA/('run_'+tag+'.sh')
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+'/'+tag+'.lock\nflock -n 9\n'+
        'export CUDA_VISIBLE_DEVICES='+str(device)+' PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8\ncd '+release+'\n'+
        'timeout --signal=TERM --kill-after=30s 45m '+command+'\n',encoding='ascii',newline='\n')
    transfer(SERVER,script,release+'/run_'+tag+'.sh')
    pid=ssh(SERVER,'nohup bash '+shlex.quote(release+'/run_'+tag+'.sh')+' > '+shlex.quote(REMOTE+'/'+tag+'.log')+' 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unknown complete-cell benchmark launch outcome')
    write(REPORT/(tag+'_job.json'),dict(pid=int(pid),physical_gpu=device,release=release,output=output,
        launcher_sha256=digest(script),timeout_minutes=45,diagnostic_events=(16 if cuda else 4) if column else (128 if (batched or cuda) else (32 if cached else 4)),partial_cells=6880,
        diagnostic_only=True,new_transport_photons=0,reconstruction_submitted=False,S2_generated=False))
    print('OVERLAP_BENCHMARK_PID',pid)


def fetch_overlap_benchmark(cached=False,batched=False,cuda=False,column=False):
    tag=('column_operator_cuda' if cuda else 'column_operator_cpu') if column else ('overlap_benchmark_cuda' if cuda else ('overlap_benchmark_batched' if batched else ('overlap_benchmark_cached' if cached else 'overlap_benchmark')))
    job=json.loads((REPORT/(tag+'_job.json')).read_text());target=DATA/tag
    target.mkdir(exist_ok=True);source=job['output']+'/benchmark_gate.json'
    expected=ssh(SERVER,'sha256sum '+shlex.quote(source)).split()[0]
    subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+source,str(target/'benchmark_gate.json')],check=True)
    if digest(target/'benchmark_gate.json')!=expected:raise ValueError('Complete-cell evidence transfer differs')
    value=json.loads((target/'benchmark_gate.json').read_text())
    deployment=json.loads((REPORT/(tag+'_deployment.json')).read_text())
    if any(deployment['code_sha256'][n]!=s for n,s in value['source_sha256'].items()):
        raise ValueError('Executed complete-cell code differs')
    summary=dict(value,evidence_sha256=expected,job=job)
    write(REPORT/(tag+'_summary.json'),summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('cases','source_sha256')},indent=2))


def guard_components(device,sampling=False):
    tag='guard_sampling' if sampling else 'guard_components'
    entry='diagnose_compton_a_sampling_v3.py' if sampling else 'diagnose_compton_a_components_v3.py'
    if (REPORT/(tag+'_job.json')).exists():raise ValueError('Component diagnostic already registered')
    if ssh(SERVER,"pgrep -af "+shlex.quote('[d]'+entry[1:])+' || true'):
        raise ValueError('Component diagnostic is already running')
    free,util=map(int,ssh(SERVER,f'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i {device}').split(','))
    if free<30000 or util>5:raise ValueError('Diagnostic GPU is occupied')
    names=(entry,'build_compton_a_guard_field.py','generate_compton_a_guard.py',
        'compton_cartesian_patch.py','compton_boundary_quadrature.py','geometry.py','config.json')
    paths={n:HERE/n for n in names};paths.update({n:ROOT/n for n in ('compton_event_response.py','detector_csv.py')})
    paths['geometry.npz']=HERE/'generated/Geometry/geometry.npz'
    hashes={n:digest(p) for n,p in paths.items()};key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/'+tag+'_releases/'+key;archive=DATA/(tag+'_code.tar.gz')
    with tarfile.open(archive,'w:gz') as f:
        for n,path in paths.items():f.add(path,arcname=n)
    ssh(SERVER,'mkdir -p -- '+shlex.quote(release));transfer(SERVER,archive,release+'/code.tar.gz')
    ssh(SERVER,'tar --no-same-owner -xzf '+shlex.quote(release+'/code.tar.gz')+' -C '+shlex.quote(release)+
        ' && cd '+shlex.quote(release)+' && '+PYTHON+' -m py_compile '+entry)
    write(REPORT/(tag+'_deployment.json'),dict(release=release,code_sha256=hashes,archive_sha256=digest(archive)))
    output=REMOTE+('/A440_sampling_diagnosis' if sampling else '/A440_component_diagnosis')
    command=PYTHON+' '+entry
    if not sampling:command+=' --root '+SERVER_ROOT+' --guard '+REMOTE+'/A440_guard'
    command+=' --field '+REMOTE+'/A440_guard_field --patches '+REMOTE+'/A440_near_patch_ultrafine'+\
        ' --cases '+REMOTE+'/A440_near_patch_ultrafine/integrals/integral_cases.csv'+\
        ' --geometry '+release+'/geometry.npz --config '+release+'/config.json'+\
        ' --inputs '+SERVER_STUDY+'/analysis_inputs --factors '+SERVER_BASE+'/generated/FactorsCalibrated --output '+output
    script=DATA/('run_'+tag+'.sh')
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+'/'+tag+'.lock\nflock -n 9\n'+
        'export CUDA_VISIBLE_DEVICES='+str(device)+' PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8\ncd '+release+'\n'+
        'timeout --signal=TERM --kill-after=30s 30m '+command+'\n',encoding='ascii',newline='\n')
    transfer(SERVER,script,release+'/run_'+tag+'.sh')
    pid=ssh(SERVER,'nohup bash '+shlex.quote(release+'/run_'+tag+'.sh')+' > '+shlex.quote(REMOTE+'/'+tag+'.log')+' 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unknown component diagnostic launch outcome')
    write(REPORT/(tag+'_job.json'),dict(pid=int(pid),physical_gpu=device,release=release,output=output,
        launcher_sha256=digest(script),timeout_minutes=30,diagnostic_cases=84,new_transport_photons=0,
        new_matrix_points=0,reconstruction_submitted=False,S2_generated=False))
    print('COMPONENT_DIAGNOSTIC_PID',pid)


def fetch_guard_components(sampling=False):
    tag='guard_sampling' if sampling else 'guard_components';job=json.loads((REPORT/(tag+'_job.json')).read_text())
    kind='sampling' if sampling else 'component'
    target=DATA/tag;target.mkdir(exist_ok=True);hashes={}
    for name in (kind+'_gate.json',kind+'_cases.csv'):
        source=job['output']+'/'+name;expected=ssh(SERVER,'sha256sum '+shlex.quote(source)).split()[0]
        subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+source,str(target/name)],check=True)
        if digest(target/name)!=expected:raise ValueError('Component evidence transfer differs')
        hashes[name]=expected
    value=json.loads((target/(kind+'_gate.json')).read_text())
    deployment=json.loads((REPORT/(tag+'_deployment.json')).read_text())
    entry='diagnose_compton_a_sampling_v3.py' if sampling else 'diagnose_compton_a_components_v3.py'
    if value['code_sha256']!=deployment['code_sha256'][entry]:
        raise ValueError('Executed component code differs')
    if hashes[kind+'_cases.csv']!=value['output_sha256']:raise ValueError('Diagnostic CSV differs')
    write(REPORT/(tag+'_summary.json'),dict(value,evidence_sha256=hashes,job=job))
    print(json.dumps({k:v for k,v in value.items() if k!='inputs_sha256'},indent=2))


def verify_overlap_backends():
    tag='overlap_backend_validation'
    if (REPORT/(tag+'.json')).exists():raise ValueError('Backend verification already registered')
    paths={n:HERE/n for n in ('verify_compton_overlap_backends_v3.py','generate_compton_a_guard.py')}
    hashes={n:digest(p) for n,p in paths.items()};key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/'+tag+'_releases/'+key;ssh(SERVER,'mkdir -p -- '+shlex.quote(release))
    for n,path in paths.items():transfer(SERVER,path,release+'/'+n)
    write(REPORT/(tag+'_deployment.json'),dict(release=release,code_sha256=hashes))
    output=REMOTE+'/'+tag+'.json'
    print(ssh(SERVER,'cd '+shlex.quote(release)+' && timeout 5m '+PYTHON+' verify_compton_overlap_backends_v3.py'+
        ' --cpu '+REMOTE+'/overlap_benchmark_batched --device '+REMOTE+'/overlap_benchmark_cuda --output '+output))
    expected=ssh(SERVER,'sha256sum '+output).split()[0];target=DATA/(tag+'.json')
    subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+output,str(target)],check=True)
    if digest(target)!=expected:raise ValueError('Backend verification evidence differs')
    value=json.loads(target.read_text());write(REPORT/(tag+'.json'),dict(value,evidence_sha256=expected))


def guard_column(device):
    tag='guard_column'
    if (REPORT/(tag+'_job.json')).exists():raise ValueError('Column matrix already registered')
    if ssh(SERVER,"pgrep -af '[g]enerate_compton_a_column_v3.py' || true"):raise ValueError('Column is already active')
    free,util=map(int,ssh(SERVER,f'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i {device}').split(','))
    if free<40000 or util>5:raise ValueError('Column GPU is occupied')
    paths={n:HERE/n for n in ('generate_compton_a_column_v3.py','generate_compton_a_guard.py')}
    hashes={n:digest(p) for n,p in paths.items()};key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/'+tag+'_releases/'+key;ssh(SERVER,'mkdir -p -- '+shlex.quote(release))
    for n,path in paths.items():transfer(SERVER,path,release+'/'+n)
    ssh(SERVER,'cd '+shlex.quote(release)+' && '+PYTHON+' -m py_compile generate_compton_a_column_v3.py')
    write(REPORT/(tag+'_deployment.json'),dict(release=release,code_sha256=hashes))
    output=REMOTE+'/A440_near_column'
    script=DATA/('run_'+tag+'.sh')
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+'/'+tag+'.lock\nflock -n 9\n'+
        'export CUDA_VISIBLE_DEVICES='+str(device)+' PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8\ncd '+release+'\n'+
        'timeout --signal=TERM --kill-after=30s 45m '+PYTHON+' generate_compton_a_column_v3.py --root '+SERVER_ROOT+
        ' --patches '+REMOTE+'/A440_near_patch_ultrafine --output '+output+' --cuda 0\n',encoding='ascii',newline='\n')
    transfer(SERVER,script,release+'/run_'+tag+'.sh')
    pid=ssh(SERVER,'nohup bash '+release+'/run_'+tag+'.sh > '+REMOTE+'/'+tag+'.log 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unknown column launch outcome')
    write(REPORT/(tag+'_job.json'),dict(pid=int(pid),physical_gpu=device,release=release,output=output,
        launcher_sha256=digest(script),timeout_minutes=45,new_matrix_points=30107,new_transport_photons=0,
        physical_models_unchanged=True,computational_detector_rows=11520,
        reconstruction_submitted=False,S2_generated=False))
    print('COLUMN_PID',pid)


def fetch_guard_column():
    tag='guard_column';job=json.loads((REPORT/(tag+'_job.json')).read_text());path=job['output']+'/column_gate.json'
    target=DATA/(tag+'.json');expected=ssh(SERVER,'sha256sum '+path).split()[0]
    subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+path,str(target)],check=True)
    if digest(target)!=expected:raise ValueError('Column evidence transfer differs')
    value=json.loads(target.read_text());write(REPORT/(tag+'_summary.json'),dict(value,evidence_sha256=expected,job=job))
    print(json.dumps({k:v for k,v in value.items() if k not in ('part','common_point_regression')},indent=2))


def tile_pilot(device):
    tag='tile_pilot'
    if (REPORT/(tag+'_job.json')).exists():raise ValueError('A tile pilot is already registered')
    if ssh(SERVER,"pgrep -af '[g]enerate_compton_a_tile_pilot_v3.py' || true"):
        raise ValueError('Tile pilot already active')
    free,util=map(int,ssh(SERVER,f'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i {device}').split(','))
    if free<40000 or util>5:raise ValueError('Requested pilot GPU is occupied')
    names=('generate_compton_a_tile_pilot_v3.py','generate_compton_a_column_v3.py','generate_compton_a_guard.py',
        'plan_compton_a_tiles_v3.py','test_compton_a_tiles_v3.py','geometry.py',
        'compton_overlap_integrator.py','compton_boundary_quadrature.py')
    paths={n:HERE/n for n in names};paths['compton_event_response.py']=ROOT/'compton_event_response.py'
    paths['A_tile_plan.json']=REPORT/'A_tile_plan.json'
    hashes={n:digest(path) for n,path in paths.items()}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/'+tag+'_releases/'+key;ssh(SERVER,'mkdir -p -- '+shlex.quote(release))
    for n,path in paths.items():transfer(SERVER,path,release+'/'+n)
    tests=ssh(SERVER,'cd '+shlex.quote(release)+' && '+PYTHON+' -m unittest test_compton_a_tiles_v3 -v 2>&1')
    if 'Ran 5 tests' not in tests or '\nOK' not in tests:raise ValueError('Pilot tests did not pass')
    (REPORT/(tag+'_tests.txt')).write_text(tests,encoding='utf-8')
    write(REPORT/(tag+'_deployment.json'),dict(release=release,code_sha256=hashes))
    output=REMOTE+'/A440_tile_pilot';script=DATA/('run_'+tag+'.sh')
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+'/'+tag+'.lock\nflock -n 9\n'+
        'export CUDA_VISIBLE_DEVICES='+str(device)+' PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8\ncd '+release+'\n'+
        'timeout --signal=TERM --kill-after=30s 45m '+PYTHON+' generate_compton_a_tile_pilot_v3.py --root '+SERVER_ROOT+
        ' --plan '+release+'/A_tile_plan.json --field '+REMOTE+'/A440_guard_field --output '+output+' --cuda 0\n',
        encoding='ascii',newline='\n')
    transfer(SERVER,script,release+'/run_'+tag+'.sh')
    pid=ssh(SERVER,'nohup bash '+release+'/run_'+tag+'.sh > '+REMOTE+'/'+tag+'.log 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unknown tile pilot launch outcome; inspect before retry')
    write(REPORT/(tag+'_job.json'),dict(pid=int(pid),physical_gpu=device,release=release,output=output,
        launcher_sha256=digest(script),timeout_minutes=45,new_matrix_points=9826,new_transport_photons=0,
        physical_models_unchanged=True,computational_detector_rows=11520,
        reconstruction_submitted=False,S2_generated=False))
    print('TILE_PILOT_PID',pid)


def fetch_tile_pilot():
    tag='tile_pilot';job=json.loads((REPORT/(tag+'_job.json')).read_text());path=job['output']+'/tile_pilot_gate.json'
    target=DATA/(tag+'.json');expected=ssh(SERVER,'sha256sum '+path).split()[0]
    subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+path,str(target)],check=True)
    if digest(target)!=expected:raise ValueError('Tile pilot evidence transfer differs')
    value=json.loads(target.read_text());write(REPORT/(tag+'_summary.json'),dict(value,evidence_sha256=expected,job=job))
    print(json.dumps({k:v for k,v in value.items() if k not in ('receipts','original_common_points','shared_physical_face')},indent=2))


def verify_tile_reader():
    tag='tile_reader';pilot=json.loads((REPORT/'tile_pilot_summary.json').read_text())
    if pilot['status']!='TILE_INTERFACE_AND_COMPACTION_PASSED_ACCURACY_HOLD':raise ValueError('Tile pilot not validated')
    names=('compton_tiled_a_field.py','verify_compton_a_tile_reader_v3.py','test_compton_tiled_a_field_v3.py',
        'plan_compton_a_tiles_v3.py','generate_compton_a_guard.py','geometry.py',
        'compton_overlap_integrator.py','compton_boundary_quadrature.py')
    paths={n:HERE/n for n in names};paths['compton_event_response.py']=ROOT/'compton_event_response.py'
    hashes={n:digest(path) for n,path in paths.items()};key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/'+tag+'_releases/'+key;ssh(SERVER,'mkdir -p -- '+shlex.quote(release))
    for n,path in paths.items():transfer(SERVER,path,release+'/'+n)
    tests=ssh(SERVER,'cd '+shlex.quote(release)+' && '+PYTHON+' -m unittest test_compton_tiled_a_field_v3 -v 2>&1')
    if 'Ran 3 tests' not in tests or '\nOK' not in tests:raise ValueError('Compact reader tests failed')
    (REPORT/(tag+'_tests.txt')).write_text(tests,encoding='utf-8')
    write(REPORT/(tag+'_deployment.json'),dict(release=release,code_sha256=hashes))
    output=REMOTE+'/tile_reader_gate.json'
    ssh(SERVER,'cd '+shlex.quote(release)+' && timeout --signal=TERM --kill-after=10s 2m '+PYTHON+
        ' verify_compton_a_tile_reader_v3.py --pilot '+pilot['job']['output']+' --field '+REMOTE+'/A440_guard_field --output '+output)
    expected=ssh(SERVER,'sha256sum '+output).split()[0];target=DATA/'tile_reader_gate.json'
    subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+output,str(target)],check=True)
    if digest(target)!=expected:raise ValueError('Compact reader evidence transfer differs')
    value=json.loads(target.read_text());write(REPORT/(tag+'_summary.json'),dict(value,evidence_sha256=expected))
    print(json.dumps({k:v for k,v in value.items() if k!='records'},indent=2))


def regional_a(device,group):
    tag='regional_a_g'+str(group)
    if (REPORT/(tag+'_job.json')).exists():raise ValueError('Regional group already registered')
    plan=json.loads((REPORT/'regional_a_plan.json').read_text())
    if plan['source_sha256']!=digest(HERE/'plan_compton_regional_a_v3.py'):
        raise ValueError('Frozen regional plan generator differs')
    free,util=map(int,ssh(SERVER,f'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i {device}').split(','))
    if free<40000 or util>5:raise ValueError('Regional diagnostic GPU is occupied')
    if ssh(SERVER,"pgrep -af '[g]enerate_compton_regional_a_v3.py.*--group "+str(group)+" ' || true"):
        raise ValueError('Regional group is already active')
    names=('generate_compton_regional_a_v3.py','generate_compton_a_column_v3.py',
        'generate_compton_a_guard.py','plan_compton_regional_a_v3.py')
    paths={n:HERE/n for n in names};paths['regional_a_plan.json']=REPORT/'regional_a_plan.json'
    hashes={n:digest(path) for n,path in paths.items()};key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/'+tag+'_releases/'+key;ssh(SERVER,'mkdir -p -- '+shlex.quote(release))
    for n,path in paths.items():transfer(SERVER,path,release+'/'+n)
    ssh(SERVER,'cd '+shlex.quote(release)+' && '+PYTHON+' -m py_compile generate_compton_regional_a_v3.py')
    write(REPORT/(tag+'_deployment.json'),dict(release=release,code_sha256=hashes))
    output=REMOTE+'/regional_a_g'+str(group);script=DATA/('run_'+tag+'.sh')
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+'/'+tag+'.lock\nflock -n 9\n'+
        'export CUDA_VISIBLE_DEVICES='+str(device)+' PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8\ncd '+release+'\n'+
        'timeout --signal=TERM --kill-after=30s 45m '+PYTHON+' generate_compton_regional_a_v3.py --root '+SERVER_ROOT+
        ' --plan '+release+'/regional_a_plan.json --group '+str(group)+' --output '+output+' --cuda 0\n',encoding='ascii',newline='\n')
    transfer(SERVER,script,release+'/run_'+tag+'.sh')
    pid=ssh(SERVER,'nohup bash '+release+'/run_'+tag+'.sh > '+REMOTE+'/'+tag+'.log 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unknown regional group launch outcome; inspect before retry')
    points=sum(__import__('math').prod(s['shape']) for c in plan['cases'] if c['group']==group for s in c['parts'])
    write(REPORT/(tag+'_job.json'),dict(pid=int(pid),physical_gpu=device,release=release,output=output,group=group,
        launcher_sha256=digest(script),timeout_minutes=45,new_matrix_points=points,new_transport_photons=0,
        physical_models_unchanged=True,computational_detector_rows=11520,
        reconstruction_submitted=False,S2_generated=False))
    print('REGIONAL_A_PID',pid,'GROUP',group)


def fetch_regional_a(group):
    tag='regional_a_g'+str(group);job=json.loads((REPORT/(tag+'_job.json')).read_text());path=job['output']+'/regional_ready.json'
    expected=ssh(SERVER,'sha256sum '+path).split()[0];target=DATA/(tag+'.json')
    subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+path,str(target)],check=True)
    if digest(target)!=expected:raise ValueError('Regional evidence transfer differs')
    value=json.loads(target.read_text());write(REPORT/(tag+'_summary.json'),dict(value,evidence_sha256=expected,job=job))
    print(json.dumps({k:v for k,v in value.items() if k!='results'},indent=2))


def regional_validation(device, frozen_measure=False):
    tag='regional_validation_frozen_measure' if frozen_measure else 'regional_validation'
    if frozen_measure:
        previous=json.loads((REPORT/'regional_validation_job.json').read_text())
        if ssh(SERVER,'if ps -p '+str(previous['pid'])+' -o args= | grep -q run_regional_validation; then echo ACTIVE; fi'):
            raise ValueError('Previous regional diagnostic still running')
        log=ssh(SERVER,'cat '+REMOTE+'/regional_validation.log')
        if 'Frozen regional response/measure/kernel identity differs' not in log:
            raise ValueError('This repair is restricted to the identified measure reference failure')
        (REPORT/'regional_validation_identity_failure.txt').write_text(log,encoding='utf-8')
    if (REPORT/(tag+'_job.json')).exists():raise ValueError('Regional precision diagnostic already registered')
    for group in range(3):
        job=json.loads((REPORT/('regional_a_g'+str(group)+'_job.json')).read_text())
        ready=json.loads(ssh(SERVER,'cat '+job['output']+'/regional_ready.json'))
        if ready['status']!='REGIONAL_PHYSICAL_COMMON_POINTS_PASSED_ACCURACY_PENDING':
            raise ValueError('Regional physical production has not passed')
    free,util=map(int,ssh(SERVER,f'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i {device}').split(','))
    if free<30000 or util>5:raise ValueError('Diagnostic GPU is occupied')
    names=('validate_compton_regional_a_v3.py','geometry.py','compton_cartesian_patch.py',
        'test_compton_cartesian_patch.py','compton_boundary_quadrature.py','generate_compton_a_guard.py')
    paths={n:HERE/n for n in names};paths.update({n:ROOT/n for n in ('compton_event_response.py','detector_csv.py')})
    paths['geometry.npz']=HERE/'generated/Geometry/geometry.npz';paths['config.json']=HERE/'config.json'
    paths['regional_a_plan.json']=REPORT/'regional_a_plan.json'
    if frozen_measure:
        paths['measure.npz']=DATA/'precise_measure/measure.npz'
        paths['measure_manifest.json']=DATA/'precise_measure/measure_manifest.json'
        plan=json.loads((REPORT/'regional_a_plan.json').read_text())
        if digest(paths['measure.npz'])!=plan['measure_npz_sha256']:
            raise ValueError('Exact frozen plan measure differs')
    hashes={n:digest(path) for n,path in paths.items()};key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/'+tag+'_releases/'+key;ssh(SERVER,'mkdir -p -- '+shlex.quote(release))
    for n,path in paths.items():transfer(SERVER,path,release+'/'+n)
    tests=ssh(SERVER,'cd '+shlex.quote(release)+' && '+PYTHON+' -m unittest test_compton_cartesian_patch -v 2>&1')
    if '\nOK' not in tests or 'skipped' in tests:raise ValueError('Physical patch interpolation tests failed')
    (REPORT/(tag+'_tests.txt')).write_text(tests,encoding='utf-8')
    write(REPORT/(tag+'_deployment.json'),dict(release=release,code_sha256=hashes))
    output=REMOTE+'/'+tag;script=DATA/('run_'+tag+'.sh')
    measure_path=release if frozen_measure else REMOTE+'/A440_guard_field_repair/precise_measure'
    command=(PYTHON+' validate_compton_regional_a_v3.py --plan '+release+'/regional_a_plan.json --samples '+REMOTE+
        ' --field '+REMOTE+'/A440_guard_field --geometry '+release+'/geometry.npz --config '+release+'/config.json'+
        ' --measure '+measure_path+' --inputs '+SERVER_STUDY+'/analysis_inputs'+
        ' --factors '+SERVER_BASE+'/generated/FactorsCalibrated --scan '+REMOTE+'/R1_analysis --output '+output)
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec 9>'+REMOTE+'/'+tag+'.lock\nflock -n 9\n'+
        'export CUDA_VISIBLE_DEVICES='+str(device)+' PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8\ncd '+release+'\n'+
        'timeout --signal=TERM --kill-after=30s 10m '+command+'\n',encoding='ascii',newline='\n')
    transfer(SERVER,script,release+'/run_'+tag+'.sh')
    pid=ssh(SERVER,'nohup bash '+release+'/run_'+tag+'.sh > '+REMOTE+'/'+tag+'.log 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unknown regional precision diagnostic launch outcome')
    write(REPORT/(tag+'_job.json'),dict(pid=int(pid),physical_gpu=device,release=release,output=output,
        launcher_sha256=digest(script),timeout_minutes=10,control_cells=12,new_transport_photons=0,
        reconstruction_submitted=False,S2_generated=False))
    print('REGIONAL_VALIDATION_PID',pid)


def fetch_regional_validation():
    tag='regional_validation_frozen_measure' if (REPORT/'regional_validation_frozen_measure_job.json').exists() else 'regional_validation'
    job=json.loads((REPORT/(tag+'_job.json')).read_text())
    for remote,local in (('regional_validation.json',DATA/'regional_validation.json'),
            ('regional_cases.csv',REPORT/'regional_validation_cases.csv')):
        path=job['output']+'/'+remote;expected=ssh(SERVER,'sha256sum '+path).split()[0]
        subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+path,str(local)],check=True)
        if digest(local)!=expected:raise ValueError('Regional diagnostic evidence transfer differs')
    value=json.loads((DATA/'regional_validation.json').read_text())
    expected=digest(DATA/'regional_validation.json')
    if digest(REPORT/'regional_validation_cases.csv')!=value['csv_sha256']:raise ValueError('Regional case manifest differs')
    write(REPORT/(tag+'_summary.json'),dict(value,evidence_sha256=expected,job=job))
    print(json.dumps({k:v for k,v in value.items() if k not in ('events','by_control')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('stage','launch','status','repair','boundary','fetch-boundary','fetch-scan','spatial','fetch-spatial','repair-spatial','guard','guard-field','guard-radial','fetch-guard','repair-guard-measure','guard-weighted','guard-integral','guard-patch','guard-patch-refined','guard-patch-ultrafine','fetch-integrals','overlap-benchmark','fetch-overlap-benchmark','overlap-benchmark-cached','fetch-overlap-benchmark-cached','guard-components','fetch-guard-components','guard-sampling','fetch-guard-sampling','overlap-benchmark-batched','fetch-overlap-benchmark-batched','overlap-benchmark-cuda','fetch-overlap-benchmark-cuda','verify-overlap-backends','guard-column','fetch-guard-column','column-operator-cpu','fetch-column-operator-cpu','column-operator-cuda','fetch-column-operator-cuda','tile-pilot','fetch-tile-pilot','verify-tile-reader','regional-a','fetch-regional-a','regional-validation','regional-validation-repair','fetch-regional-validation'))
    p.add_argument('--gpu',type=int,default=3);p.add_argument('--group',type=int,choices=(0,1,2),default=0);a=p.parse_args()
    if a.action=='stage':stage()
    elif a.action=='launch':launch(a.gpu)
    elif a.action=='repair':repair(a.gpu)
    elif a.action=='boundary':boundary(a.gpu)
    elif a.action=='fetch-boundary':fetch_boundary()
    elif a.action=='fetch-scan':fetch_scan()
    elif a.action=='spatial':spatial(a.gpu)
    elif a.action=='repair-spatial':spatial(a.gpu,repair=True)
    elif a.action=='fetch-spatial':fetch_spatial()
    elif a.action=='guard':guard(a.gpu)
    elif a.action=='guard-field':guard_field()
    elif a.action=='guard-radial':guard_radial(a.gpu)
    elif a.action=='fetch-guard':fetch_guard()
    elif a.action=='repair-guard-measure':repair_guard_measure()
    elif a.action=='guard-weighted':guard_weighted(a.gpu)
    elif a.action=='guard-integral':guard_integral(a.gpu)
    elif a.action=='guard-patch':guard_integral(a.gpu,patch=True)
    elif a.action=='guard-patch-refined':guard_integral(a.gpu,patch=True,refined=True)
    elif a.action=='guard-patch-ultrafine':guard_integral(a.gpu,patch=True,refined=True,ultrafine=True)
    elif a.action=='fetch-integrals':fetch_integrals()
    elif a.action=='overlap-benchmark':overlap_benchmark(a.gpu)
    elif a.action=='fetch-overlap-benchmark':fetch_overlap_benchmark()
    elif a.action=='overlap-benchmark-cached':overlap_benchmark(a.gpu,cached=True)
    elif a.action=='fetch-overlap-benchmark-cached':fetch_overlap_benchmark(cached=True)
    elif a.action=='overlap-benchmark-batched':overlap_benchmark(a.gpu,batched=True)
    elif a.action=='fetch-overlap-benchmark-batched':fetch_overlap_benchmark(batched=True)
    elif a.action=='overlap-benchmark-cuda':overlap_benchmark(a.gpu,cuda=True)
    elif a.action=='fetch-overlap-benchmark-cuda':fetch_overlap_benchmark(cuda=True)
    elif a.action=='verify-overlap-backends':verify_overlap_backends()
    elif a.action=='guard-column':guard_column(a.gpu)
    elif a.action=='fetch-guard-column':fetch_guard_column()
    elif a.action=='column-operator-cpu':overlap_benchmark(a.gpu,column=True)
    elif a.action=='fetch-column-operator-cpu':fetch_overlap_benchmark(column=True)
    elif a.action=='column-operator-cuda':overlap_benchmark(a.gpu,column=True,cuda=True)
    elif a.action=='fetch-column-operator-cuda':fetch_overlap_benchmark(column=True,cuda=True)
    elif a.action=='tile-pilot':tile_pilot(a.gpu)
    elif a.action=='fetch-tile-pilot':fetch_tile_pilot()
    elif a.action=='verify-tile-reader':verify_tile_reader()
    elif a.action=='regional-a':regional_a(a.gpu,a.group)
    elif a.action=='fetch-regional-a':fetch_regional_a(a.group)
    elif a.action=='regional-validation':regional_validation(a.gpu)
    elif a.action=='regional-validation-repair':regional_validation(a.gpu,frozen_measure=True)
    elif a.action=='fetch-regional-validation':fetch_regional_validation()
    elif a.action=='guard-components':guard_components(a.gpu)
    elif a.action=='fetch-guard-components':fetch_guard_components()
    elif a.action=='guard-sampling':guard_components(a.gpu,sampling=True)
    elif a.action=='fetch-guard-sampling':fetch_guard_components(sampling=True)
    else:status()
