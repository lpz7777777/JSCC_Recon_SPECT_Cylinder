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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('stage','launch','status','repair','boundary','fetch-boundary','fetch-scan','spatial','fetch-spatial','repair-spatial'))
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
    else:status()
