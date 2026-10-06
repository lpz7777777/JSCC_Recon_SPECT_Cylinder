"""Isolated, bounded existing-data legacy calibration on the agent-authenticated host."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
from first_scatter_pipeline import ssh, transfer, SERVER, SERVER_ROOT, SERVER_BASE, SERVER_STUDY
from process_list_global_audit_v4 import digest, write
from prepare_energy_5e9_v5 import STUDY, validate_transport

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
REPORT=HERE/'reports/NEMA_Body_H60'/STUDY
DATA=HERE/'generated'/STUDY
REMOTE=SERVER_BASE+'/generated/'+STUDY
PYTHON=SERVER_ROOT+'/.venv/bin/python'


def start():
    REPORT.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
    if (REPORT/'calibration_job.json').exists():raise ValueError('Already registered; inspect instead of duplicating')
    free,util=map(int,ssh(SERVER,'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i 0').split(','))
    if free<40000 or util>5:raise ValueError('Calibration GPU0 is busy')
    inputs=json.loads((HERE/'generated/nema_h60_imaging_5e9_files.json').read_text())
    validate_transport(json.loads((HERE/'generated/collections/NEMA_Body_H60_5e9.json').read_text()))
    archive=HERE/'generated/nema_h60_imaging_5e9.tar.gz'
    expected=(HERE/'generated/nema_h60_imaging_5e9.tar.gz.sha256').read_text().strip()
    if digest(archive)!=expected:raise ValueError('Original 5e9 archive changed')
    with tarfile.open(archive) as t:
        members=t.getmembers()
        if len({m.name for m in members})!=len(members) or set(m.name for m in members)!=set(inputs)|{'nema_h60_imaging_5e9_files.json'}:
            raise ValueError('Archive membership differs')
        for m in members:
            if not m.isfile() or '..' in Path(m.name).parts or Path(m.name).is_absolute():
                raise ValueError('Unsafe archive member')
            if m.name in inputs:
                with t.extractfile(m) as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
                if m.size!=inputs[m.name]['bytes'] or sha!=inputs[m.name]['sha256']:
                    raise ValueError('Original 5e9 member changed: '+m.name)
    paths={n:HERE/n for n in ('prepare_energy_5e9_v5.py','calibrate_energy_5e9_v5.py',
        'test_energy_5e9_v5.py','compton_energy_probability_v5.py','test_compton_energy_probability_v5.py',
        'process_list_global_audit_v4.py','geometry.py','config.json')}
    paths.update({n:ROOT/n for n in ('compton_event_response.py','detector_csv.py')})
    paths['geometry.npz']=HERE/'generated/Geometry/geometry.npz'
    paths['whole_geometry.npz']=HERE/'generated/process_list_global_audit_v4/WholeCellGeometry/geometry.npz'
    paths['transfer_training_summary.json']=HERE/'reports/NEMA_Body_H60/process_list_global_audit_v4/evidence/energy_energy_probability_summary.json'
    hashes={n:digest(p) for n,p in paths.items()}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/calibration_releases/'+key
    code=DATA/'calibration_code.tar.gz'
    with tarfile.open(code,'w:gz') as t:
        for name,path in paths.items():t.add(path,arcname=name)
    ssh(SERVER,'mkdir -p -- '+shlex.quote(release)+' '+shlex.quote(REMOTE+'/nema_base'))
    transfer(SERVER,archive,REMOTE+'/nema_input.tar.gz'); transfer(SERVER,code,release+'/code.tar.gz')
    script='''import hashlib,json,pathlib,subprocess,sys,tarfile
release=pathlib.Path(sys.argv[1]);base=pathlib.Path(sys.argv[2]);expected=json.loads(sys.argv[3]);inputs=json.loads(sys.argv[4])
with tarfile.open(release/'code.tar.gz') as t:t.extractall(release,filter='data')
with tarfile.open(base.parent/'nema_input.tar.gz') as t:t.extractall(base/'generated',filter='data')
for name,sha in expected.items():assert hashlib.sha256((release/name).read_bytes()).hexdigest()==sha,name
for name,item in inputs.items():assert hashlib.sha256((base/'generated'/name).read_bytes()).hexdigest()==item['sha256'],name
subprocess.run([sys.executable,'-m','unittest','test_energy_5e9_v5','test_compton_energy_probability_v5','-v'],cwd=release,check=True)
print('LEGACY_5E9_INPUTS_AND_CALIBRATION_CODE_VERIFIED')
'''
    print(ssh(SERVER,PYTHON+' -c '+shlex.quote(script)+' '+shlex.quote(release)+' '+
        shlex.quote(REMOTE+'/nema_base')+' '+shlex.quote(json.dumps(hashes))+' '+shlex.quote(json.dumps(inputs))),flush=True)
    outputs=dict(selections=REMOTE+'/selections_'+key,probe=REMOTE+'/calibration_probe_'+key,calibration=REMOTE+'/calibration_'+key)
    args=f'--inputs {SERVER_STUDY}/analysis_inputs --analysis {outputs["selections"]} --factors {SERVER_BASE}/generated/FactorsCalibrated --geometry {release}/geometry.npz --grid-config {release}/config.json --training-law {release}/transfer_training_summary.json'
    # Phase timeouts bound all GPU work; flock and receipts prohibit duplication.
    body=f'''#!/usr/bin/env bash
set -euo pipefail
exec 9>{REMOTE}/calibration.lock
flock -n 9
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd {release}
timeout --signal=TERM --kill-after=30s 2700s {PYTHON} prepare_energy_5e9_v5.py --base {REMOTE}/nema_base --inputs {SERVER_STUDY}/analysis_inputs --factors {SERVER_BASE}/generated/FactorsCalibrated --geometry {release}/geometry.npz --output {outputs['selections']}
timeout --signal=TERM --kill-after=30s 600s {PYTHON} calibrate_energy_5e9_v5.py {args} --probe-events 8192 --output {outputs['probe']}
{PYTHON} -c 'import json;from pathlib import Path;r=json.loads(Path("{outputs['probe']}/resource_probe.json").read_text());assert r["estimated_full_seconds"]<6600,r;print("LEGACY_FULL_CALIBRATION_TIME_GATE_PASSED")'
timeout --signal=TERM --kill-after=30s 7200s {PYTHON} calibrate_energy_5e9_v5.py {args} --output {outputs['calibration']}
echo LEGACY_5E9_CALIBRATION_CHAIN_FINISHED
'''
    launcher=DATA/'run_calibration.sh';launcher.write_text(body,encoding='ascii',newline='\n')
    transfer(SERVER,launcher,release+'/run_calibration.sh')
    write(REPORT/'calibration_deployment.json',dict(study=STUDY,release=release,code_sha256=hashes,
        launcher_sha256=digest(launcher),code_archive_sha256=digest(code),input_archive_sha256=digest(archive),
        original_inputs={n:i['sha256'] for n,i in inputs.items()},event_policy='legacy',new_photons=0,new_training=False,
        bounded_gpu_phase_seconds=dict(scan=2700,probe=600,calibration=7200)))
    log=REMOTE+'/calibration_'+key+'.log'
    pid=ssh(SERVER,'nohup bash '+shlex.quote(release+'/run_calibration.sh')+' > '+shlex.quote(log)+' 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unexpected launch PID; inspect before retry')
    write(REPORT/'calibration_job.json',dict(study=STUDY,pid=int(pid),server=SERVER,release=release,log=log,outputs=outputs,
        new_photons=0,new_training=False,formal_submitted=False))
    print('LEGACY_5E9_CALIBRATION_PID',pid,flush=True)


def status():
    j=json.loads((REPORT/'calibration_job.json').read_text())
    print(ssh(SERVER,'ps -p '+str(j['pid'])+' -o pid,etime,args; tail -n 12 -- '+shlex.quote(j['log'])+
        '; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader -i 0'))


def fetch():
    j=json.loads((REPORT/'calibration_job.json').read_text());local=DATA/'calibration_results'
    readiness='''import json,pathlib,subprocess,sys
j=json.loads(sys.argv[1])
active=subprocess.run(['ps','-p',str(j['pid']),'-o','args='],capture_output=True,text=True).stdout
if j['release'] in active:raise ValueError('Calibration chain is still active; no partial fetch')
gate=pathlib.Path(j['outputs']['calibration'])/'calibration_gate.json'
if not gate.is_file():raise ValueError('No complete calibration gate; inspect the bounded failure log')
print('CALIBRATION_EXIT_AND_GATE_CONFIRMED',json.loads(gate.read_text())['status'])
'''
    print(ssh(SERVER,PYTHON+' -c '+shlex.quote(readiness)+' '+shlex.quote(json.dumps(j))))
    local.mkdir(exist_ok=False)
    script='''import hashlib,json,pathlib,sys,tarfile
j=json.loads(sys.argv[1]);out=pathlib.Path(sys.argv[2]);paths={}
for phase,folder in j['outputs'].items():
 for p in pathlib.Path(folder).glob('*'):
  if p.is_file():paths[phase+'/'+p.name]=p
with tarfile.open(out,'w:gz') as t:
 for name,p in paths.items():t.add(p,arcname=name)
print(json.dumps(dict(sha256=hashlib.sha256(out.read_bytes()).hexdigest(),files={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()})))
'''
    archive=REMOTE+'/calibration_delivery.tar.gz'
    receipt=json.loads(ssh(SERVER,PYTHON+' -c '+shlex.quote(script)+' '+shlex.quote(json.dumps(j))+' '+shlex.quote(archive)))
    target=DATA/'calibration_delivery.tar.gz'
    subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+archive,str(target)],check=True)
    if digest(target)!=receipt['sha256']:raise ValueError('Calibration archive SHA differs')
    with tarfile.open(target) as t:t.extractall(local,filter='data')
    for name,sha in receipt['files'].items():
        if digest(local/name)!=sha:raise ValueError('Calibration output SHA differs: '+name)
    write(REPORT/'calibration_fetch.json',receipt)
    gate=json.loads((local/'calibration/calibration_gate.json').read_text())
    write(REPORT/'calibration_gate.json',gate)
    print('LEGACY_CALIBRATION_FETCHED',gate['status'],gate['failures'])
    if gate['status']!='PASSED' or gate['failures']:raise ValueError('Calibration holds; no formal submission')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('start','status','fetch'))
    globals()[p.parse_args().action]()
