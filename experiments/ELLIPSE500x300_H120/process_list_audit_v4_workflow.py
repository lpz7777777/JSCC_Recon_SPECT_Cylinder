"""Deploy and run one bounded diagnostic chain; no transport or reconstruction."""
from pathlib import Path
import hashlib
import json
import shlex
import sys
import tarfile
from first_scatter_pipeline import ssh,transfer,SERVER,SERVER_ROOT,SERVER_BASE,SERVER_STUDY

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REPORT=HERE/'reports/NEMA_Body_H60/process_list_global_audit_v4'
DATA=HERE/'generated/process_list_global_audit_v4'
REMOTE=SERVER_BASE+'/generated/process_list_global_audit_v4'
PYTHON=SERVER_ROOT+'/.venv/bin/python'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def deploy(record_name='deployment.json'):
    DATA.mkdir(exist_ok=True,parents=True)
    paths={n:HERE/n for n in ('process_list_global_audit_v4.py','test_process_list_global_audit_v4.py',
        'energy_probability_probe_v4.py','native_peak_responsibility_v4.py','geometry.py','config.json')}
    paths.update({n:ROOT/n for n in ('compton_event_response.py','detector_csv.py')})
    paths['geometry.npz']=HERE/'generated/Geometry/geometry.npz'
    hashes={n:digest(p) for n,p in paths.items()}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/releases/'+key
    archive=DATA/'audit_code.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name,path in paths.items():tar.add(path,arcname=name)
    ssh(SERVER,'mkdir -p -- '+shlex.quote(release))
    transfer(SERVER,archive,release+'/code.tar.gz')
    program='''import hashlib,json,pathlib,subprocess,sys,tarfile
release=pathlib.Path(sys.argv[1]);expected=json.loads(sys.argv[2])
with tarfile.open(release/'code.tar.gz') as t:t.extractall(release,filter='data')
for name,sha in expected.items():
 assert hashlib.sha256((release/name).read_bytes()).hexdigest()==sha,name
subprocess.run([sys.executable,'-m','unittest','test_process_list_global_audit_v4','-v'],cwd=release,check=True)
print('RELEASE_HASHES_AND_TESTS_PASSED')
'''
    print(ssh(SERVER,PYTHON+' -c '+shlex.quote(program)+' '+shlex.quote(release)+' '+shlex.quote(json.dumps(hashes))))
    record=dict(release=release,code_sha256=hashes,archive_sha256=digest(archive),remote=REMOTE)
    write(REPORT/record_name,record)
    return record

def launch(record):
    if (REPORT/'diagnostic_job.json').exists():raise ValueError('Already launched; inspect existing evidence')
    # One lock covers all phases. Never touch another project or GPU process.
    snapshot=ssh(SERVER,'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i 0')
    free,util=map(int,snapshot.split(','))
    if free<40000 or util>5:raise ValueError('GPU0 not idle; leave pending')
    release=record['release']
    config=dict(study='process_list_global_audit_v4',geometry_mode='stable_float64',q_delete_strictly_above=3,
        event_sets='unchanged R1 kept rows; each consumed file checked against frozen manifest',
        fine_spatial_bins=dict(radial_edges_mm=[0,81,159,219,255],azimuth_sectors=8,
                              axial_edges_mm=[-60,-45,-24,0,24,45,60],adequate_accepted_mass=400),
        hold_condition='abs(relative bias)>20% and >3 combined worker SE in an adequate bin',
        independent_workers=200,rotations_per_worker=20,rotations_are_independent=False,
        whole_cell_controls='fixed radii 0/60/120/180, four angles, z=-21/1.5/22.5 plus four far axial controls',
        sampling_orders=[2,4,8],sampling_events_per_view=16,new_transport_photons=0,new_fine_A_matrices=0,
        probe_time_limit_seconds=600,single_diagnostic_gpu_limit_seconds=7200,output_limit_bytes=5_000_000_000,
        frozen_release=release,code_sha256=record['code_sha256'])
    write(REPORT/'execution_config.json',config)
    transfer(SERVER,REPORT/'execution_config.json',release+'/execution_config.json')
    args=f'--inputs {SERVER_STUDY}/analysis_inputs --analysis {SERVER_BASE}/generated/compton_response_geometry_v3/R1_analysis --factors {SERVER_BASE}/generated/FactorsCalibrated --geometry {release}/geometry.npz --grid-config {release}/config.json'
    body=f'''#!/usr/bin/env bash
set -euo pipefail
exec 9>{REMOTE}/audit.lock
flock -n 9
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd {release}
timeout --signal=TERM --kill-after=30s 10m {PYTHON} process_list_global_audit_v4.py points {args} --device cpu --output {REMOTE}/points_{Path(release).name}
timeout --signal=TERM --kill-after=30s 10m {PYTHON} process_list_global_audit_v4.py spatial {args} --device cuda:0 --probe-events 2048 --output {REMOTE}/probe_{Path(release).name}
{PYTHON} -c 'import json;from pathlib import Path;p=Path("{REMOTE}/probe_{Path(release).name}");a=json.loads((p/"spatial_probe.json").read_text());r=json.loads((p/"execution.json").read_text());assert a["extrapolated_compute_seconds"]<7200;assert r["gpu_peak_reserved_bytes"]<.8*r["gpu_total_bytes"];print("PROBE_RESOURCE_GATE_PASSED")'
timeout --signal=TERM --kill-after=30s 2h {PYTHON} process_list_global_audit_v4.py spatial {args} --device cuda:0 --output {REMOTE}/spatial_{Path(release).name}
timeout --signal=TERM --kill-after=30s 2h {PYTHON} process_list_global_audit_v4.py sampling {args} --device cuda:0 --output {REMOTE}/sampling_{Path(release).name}
du -sb {REMOTE}
echo AUDIT_CHAIN_FINISHED
'''
    launcher=DATA/'run_diagnostics.sh'
    launcher.write_text(body,encoding='ascii',newline='\n')
    transfer(SERVER,launcher,release+'/run_diagnostics.sh')
    log=REMOTE+'/audit_'+Path(release).name+'.log'
    pid=ssh(SERVER,'nohup bash '+shlex.quote(release+'/run_diagnostics.sh')+' > '+shlex.quote(log)+' 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unexpected PID')
    write(REPORT/'diagnostic_job.json',dict(pid=int(pid),server=SERVER,release=release,log=log,
        launcher_sha256=digest(launcher),execution_config_sha256=digest(REPORT/'execution_config.json'),
        outputs={phase:REMOTE+'/'+phase+'_'+Path(release).name for phase in ('points','probe','spatial','sampling')}))
    print('BOUNDED_AUDIT_PID',pid)

def launch_extra(phase):
    record=json.loads((REPORT/'deployment.json').read_text())
    jobfile=REPORT/(phase+'_job.json')
    if jobfile.exists():raise ValueError('Existing phase receipt; inspect before retrying')
    source=HERE/('energy_probability_probe_v4.py' if phase=='energy' else 'native_peak_responsibility_v4.py')
    sha=digest(source);release=REMOTE+'/releases/'+phase+'_'+sha[:16]
    ssh(SERVER,'mkdir -p -- '+shlex.quote(release))
    for name in ('process_list_global_audit_v4.py','geometry.py','detector_csv.py','compton_event_response.py','geometry.npz'):
        ssh(SERVER,'cp -- '+shlex.quote(record['release']+'/'+name)+' '+shlex.quote(release+'/'+name))
    transfer(SERVER,source,release+'/'+source.name)
    output=REMOTE+'/'+phase+'_'+sha[:16];log=output+'.log'
    args=f'--inputs {SERVER_STUDY}/analysis_inputs --output {output}'
    if phase=='energy':
        points=json.loads((REPORT/'diagnostic_job.json').read_text())['outputs']['points']+'/point_residuals.csv'
        args+=f' --points {points} --detector {SERVER_BASE}/generated/FactorsCalibrated/440keV_RotateNum20/Detector.csv'
    else:
        free,util=map(int,ssh(SERVER,'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i 0').split(','))
        if free<40000 or util>5:raise ValueError('GPU0 busy; leave pending')
        transfer(SERVER,REPORT/'native_responsibility_probes.json',release+'/probes.json')
        images=HERE/'generated/compton_first_scatter_v2/RemoteResults/ideal'
        for c in ('440_ComptonOnly','440_SinglePlusCompton'):
            transfer(SERVER,images/f'Image_{c}_active.float32',release+f'/Image_{c}_active.float32')
        args+=f' --analysis {SERVER_BASE}/generated/compton_response_geometry_v3/R1_analysis --factors {SERVER_BASE}/generated/FactorsCalibrated --geometry {release}/geometry.npz --probes {release}/probes.json --images {release}'
    command='timeout --signal=TERM --kill-after=30s 10m '+PYTHON+' '+release+'/'+source.name+' '+args
    pid=ssh(SERVER,'nohup '+command+' > '+shlex.quote(log)+' 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unexpected PID')
    write(jobfile,dict(pid=int(pid),source_sha256=sha,release=release,log=log,output=output,
        budget_seconds=600,new_transport_photons=0,new_reconstruction=False))
    print(phase.upper()+'_AUDIT_PID',pid)

if __name__=='__main__':
    args=sys.argv[1:]
    if args==['start']:
        if (REPORT/'diagnostic_job.json').exists():raise ValueError('Existing chain receipt; do not redeploy or duplicate')
        launch(deploy())
    elif args==['deploy']:deploy('final_code_deployment.json')
    elif args in (['energy'],['responsibility']):launch_extra(args[0])
    else:raise SystemExit('Usage: process_list_audit_v4_workflow.py start|deploy|energy|responsibility')
