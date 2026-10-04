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
    job=json.loads((REPORT/'scan_job.json').read_text())
    print(ssh(SERVER,'ps -p '+str(job['pid'])+' -o pid,etime,stat,args; tail -n 8 '+REMOTE+'/scan.log; '+
        'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i '+str(job['physical_gpu'])))

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

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('stage','launch','status','repair','boundary','fetch-boundary','fetch-scan','spatial','fetch-spatial','repair-spatial','guard','guard-field','guard-radial','fetch-guard','repair-guard-measure','guard-weighted','guard-integral','guard-patch','guard-patch-refined','guard-patch-ultrafine','fetch-integrals'))
    p.add_argument('--gpu',type=int,default=3);a=p.parse_args()
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
    else:status()
