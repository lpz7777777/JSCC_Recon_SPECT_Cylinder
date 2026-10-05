"""One bounded isolated candidate validation; no simulation/reconstruction jobs."""
from pathlib import Path
import hashlib
import json
import shlex
import sys
import tarfile
from first_scatter_pipeline import ssh,transfer,SERVER,SERVER_ROOT,SERVER_BASE,SERVER_STUDY

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
REPORT=HERE/'reports/NEMA_Body_H60/compton_energy_probability_v5'
DATA=HERE/'generated/compton_energy_probability_v5'
REMOTE=SERVER_BASE+'/generated/compton_energy_probability_v5'
PYTHON=SERVER_ROOT+'/.venv/bin/python'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def idle_program():
    # Utilization is sampled over an interval and can outlive our exited phase.
    # A bounded grace period preserves the check for another user's process.
    return '''import subprocess,time
for attempt in range(16):
 f,u=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits','-i','0'],text=True).split(','))
 if f>40000 and u<=5:break
 time.sleep(1)
else:raise RuntimeError('GPU0 remains busy after bounded idle grace period')
'''


def start():
    if (REPORT/'job.json').exists():raise ValueError('Existing job receipt; do not duplicate')
    REPORT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    paths={n:HERE/n for n in ('compton_energy_probability_v5.py','test_compton_energy_probability_v5.py',
        'validate_energy_candidate_v5.py','process_list_global_audit_v4.py','geometry.py','config.json')}
    paths.update({n:ROOT/n for n in ('compton_event_response.py','detector_csv.py')})
    paths['geometry.npz']=HERE/'generated/Geometry/geometry.npz'
    paths['whole_geometry.npz']=HERE/'generated/process_list_global_audit_v4/WholeCellGeometry/geometry.npz'
    paths['transfer_training_summary.json']=HERE/'reports/NEMA_Body_H60/process_list_global_audit_v4/evidence/energy_energy_probability_summary.json'
    hashes={n:digest(p) for n,p in paths.items()}
    charged=json.loads((REPORT/'gpu_budget_spent.json').read_text()) if (REPORT/'gpu_budget_spent.json').exists() else {'conservative_spent_seconds':0}
    remaining=int(7200-charged['conservative_spent_seconds']);full_allowance=remaining-600-120
    if full_allowance<600:raise ValueError('Insufficient remaining two-hour GPU budget; leave HOLD')
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/releases/'+key
    free,util=map(int,ssh(SERVER,'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i 0').split(','))
    if free<40000 or util>5:raise ValueError('GPU0 busy; leave pending')
    archive=DATA/'code.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for name,path in paths.items():t.add(path,arcname=name)
    ssh(SERVER,'mkdir -p -- '+shlex.quote(release));transfer(SERVER,archive,release+'/code.tar.gz')
    program='''import hashlib,json,pathlib,subprocess,sys,tarfile
release=pathlib.Path(sys.argv[1]);expected=json.loads(sys.argv[2])
with tarfile.open(release/'code.tar.gz') as t:t.extractall(release,filter='data')
for name,sha in expected.items():assert hashlib.sha256((release/name).read_bytes()).hexdigest()==sha,name
subprocess.run([sys.executable,'-m','unittest','test_compton_energy_probability_v5','-v'],cwd=release,check=True)
print('CANDIDATE_FROZEN_HASHES_AND_TESTS_PASSED')
'''
    print(ssh(SERVER,PYTHON+' -c '+shlex.quote(program)+' '+shlex.quote(release)+' '+shlex.quote(json.dumps(hashes))))
    points=json.loads((HERE/'reports/NEMA_Body_H60/process_list_global_audit_v4/diagnostic_job.json').read_text())['outputs']['points']+'/point_residuals.csv'
    config=dict(study='compton_energy_probability_v5',code_sha256=hashes,
        candidate='continuous layer-specific 128 equal-mass residual bins; physical-width truncation; analytic Gaussian convolution in measured E1',
        fixed_selection='R1 stable_float64 kept rows; no candidate rescreening',
        position_model='original uniform-crystal variance, unchanged; Gaussian first-order energy propagation',
        preserved_proxy='B first-crystal and original measured-energy KN factor',
        sparse_angles='continuous quantile interpolation; nearest endpoint outside trained anchor range; explicitly flagged',
        normalization='unconditioned energy density in forward proxy; selected-domain density only in diagnostic scoring',
        sensitivity_contract='accepted uniform-transport reference measure; full-circle normalized posterior rows integrated using actual N',
        quadrature=dict(density='analytic vs GL16',selection_mass_order=8,reference_order=16,point_sample_per_source=32,relative_target=.01),
        probe_events=8192,batch=32,node_chunk=16,probe_seconds=600,phase_gpu_seconds=full_allowance,
        original_gpu_budget_seconds=7200,conservative_spent_gpu_seconds=charged['conservative_spent_seconds'],
        remaining_gpu_seconds=remaining,
        cuda_execution='float64 geometry/transfer/normalization; float32 Gaussian evaluation, analytic reference gate',
        cuda_allocator='expandable_segments:True; release unused cache above 60%; runtime peak hard limit 75%',
        numerical_forward=dict(endpoint_bins_order=16,interior_bins_order=2,
            all_point_density_relative_target=.01,full_grid_probe_TV_L2_target=.001),
        memory_fraction_limit=.8,output_limit_bytes=5_000_000_000,
        q_domain_sample_per_source=32,q_scan_nodes=[256,512],q_refinement_mass_target=.001,
        spatial_bins=192,spatial_adequacy=400,spatial_hold_bias=.2,spatial_hold_standard_errors=3,
        joint_categories=dict(layer_pairs=12,E1_edges_MeV=[.05,.125,.20,'ComptonEdge-1keV'],
            E2_edges_MeV=[.05,.25,'infinity'],adequacy=50,hold_bias=.3,hold_standard_errors=3),
        new_transport_photons=0,new_fine_A_matrices=0,new_reconstruction=False)
    write(REPORT/'config.json',config)
    transfer(SERVER,REPORT/'config.json',release+'/execution_config.json')
    args=f'--inputs {SERVER_STUDY}/analysis_inputs --analysis {SERVER_BASE}/generated/compton_response_geometry_v3/R1_analysis --factors {SERVER_BASE}/generated/FactorsCalibrated --geometry {release}/geometry.npz --grid-config {release}/config.json --training-law {release}/transfer_training_summary.json --points {points}'
    outputs={phase:REMOTE+'/'+phase+'_'+key for phase in ('points','probe','q','spatial')}
    idle=idle_program()
    body=f'''#!/usr/bin/env bash
set -euo pipefail
exec 9>{REMOTE}/candidate.lock
flock -n 9
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export TORCHINDUCTOR_CACHE_DIR={REMOTE}/kernel_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd {release}
timeout --signal=TERM --kill-after=30s 10m {PYTHON} validate_energy_candidate_v5.py points {args} --device cpu --output {outputs['points']}
{PYTHON} -c 'import json;from pathlib import Path;g=json.loads(Path("{outputs['points']}/point_gate.json").read_text());assert g["quadrature"]["passed"] and g["quadrature"]["hybrid"]["passed"],g["quadrature"];print("CONTINUOUS_BIN_QUADRATURE_GATE_PASSED")'
{PYTHON} -c {shlex.quote(idle)}
timeout --signal=TERM --kill-after=30s 10m {PYTHON} validate_energy_candidate_v5.py spatial {args} --device cuda:0 --energy-backend tail16_mid2 --gaussian-float32 --probe-events 8192 --output {outputs['probe']}
{PYTHON} -c 'import json;from pathlib import Path;p=Path("{outputs['probe']}");r=json.loads((p/"resource_probe.json").read_text());e=json.loads((p/"execution.json").read_text());assert r["estimated_full_seconds"]<{full_allowance};assert e["gpu_peak_reserved_bytes"]<.75*e["gpu_total_bytes"];print("CANDIDATE_PROBE_RESOURCE_GATE_PASSED")'
{PYTHON} -c 'import json;from pathlib import Path;e=json.loads(Path("{outputs['probe']}/execution.json").read_text());mem=int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal:")))*1024;p=Path("/sys/fs/cgroup/memory.max");v=p.read_text().strip() if p.exists() else "max";limit=min(mem,int(v)) if v.isdigit() else mem;assert e["host_peak_rss_bytes"]<.8*limit;print("ACTUAL_HOST_OR_CGROUP_RAM_GATE_PASSED")'
{PYTHON} -c {shlex.quote(idle)}
timeout --signal=TERM --kill-after=30s 2m {PYTHON} validate_energy_candidate_v5.py q {args} --device cuda:0 --q-per-source 32 --output {outputs['q']}
{PYTHON} -c {shlex.quote(idle)}
timeout --signal=TERM --kill-after=30s {full_allowance}s {PYTHON} validate_energy_candidate_v5.py spatial {args} --device cuda:0 --energy-backend tail16_mid2 --gaussian-float32 --output {outputs['spatial']}
du -sb {REMOTE}
echo ENERGY_CANDIDATE_CHAIN_FINISHED
'''
    launcher=DATA/'run_candidate.sh';launcher.write_text(body,encoding='ascii',newline='\n')
    transfer(SERVER,launcher,release+'/run_candidate.sh')
    write(REPORT/'deployment.json',dict(release=release,code_sha256=hashes,archive_sha256=digest(archive),
        launcher_sha256=digest(launcher),execution_config_sha256=digest(REPORT/'config.json')))
    log=REMOTE+'/candidate_'+key+'.log'
    pid=ssh(SERVER,'nohup bash '+shlex.quote(release+'/run_candidate.sh')+' > '+shlex.quote(log)+' 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unexpected PID')
    write(REPORT/'job.json',dict(pid=int(pid),server=SERVER,release=release,log=log,outputs=outputs,
        new_photons=0,new_fine_A_matrices=0,new_reconstruction=False))
    print('ENERGY_CANDIDATE_BOUNDED_PID',pid)


def resume_after_idle_exit():
    if (REPORT/'job_stopped_node_truncation_652867.json').exists():
        raise ValueError('Original discrete-node release is superseded; legacy resume is disabled')
    j=json.loads((REPORT/'job.json').read_text());deployment=json.loads((REPORT/'deployment.json').read_text())
    if (REPORT/'resume.json').exists():raise ValueError('Existing resume receipt; inspect before further action')
    code='''import json,pathlib,sys
j=json.loads(sys.argv[1]);assert not pathlib.Path('/proc/'+str(j['pid'])+'/cmdline').exists(),'Old task is still present'
for phase in ('q','spatial'):assert not pathlib.Path(j['outputs'][phase]).exists(),'Remaining phase already exists'
for phase in ('points','probe'):assert (pathlib.Path(j['outputs'][phase])/'execution.json').exists(),'Completed phase missing'
p=pathlib.Path(j['outputs']['probe']);r=json.loads((p/'resource_probe.json').read_text());e=json.loads((p/'execution.json').read_text())
assert r['estimated_full_seconds']<7200 and e['gpu_peak_reserved_bytes']<.8*e['gpu_total_bytes']
mem=int(next(line.split()[1] for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:')))*1024
cg=pathlib.Path('/sys/fs/cgroup/memory.max');value=cg.read_text().strip() if cg.exists() else 'max'
if value.isdigit():mem=min(mem,int(value))
assert e['host_peak_rss_bytes']<.8*mem,'RAM probe exceeds actual host/cgroup limit'
print('PRIOR_PHASES_AND_RAM_GATE_VERIFIED',r['estimated_full_seconds'],e['host_peak_rss_bytes']/mem)
'''
    print(ssh(SERVER,PYTHON+' -c '+shlex.quote(code)+' '+shlex.quote(json.dumps(j))))
    for name in ('compton_energy_probability_v5.py','validate_energy_candidate_v5.py','test_compton_energy_probability_v5.py'):
        if digest(HERE/name)!=deployment['code_sha256'][name]:raise ValueError('Candidate source changed; resume forbidden')
    release=j['release'];o=j['outputs']
    points=json.loads((HERE/'reports/NEMA_Body_H60/process_list_global_audit_v4/diagnostic_job.json').read_text())['outputs']['points']+'/point_residuals.csv'
    args=f'--inputs {SERVER_STUDY}/analysis_inputs --analysis {SERVER_BASE}/generated/compton_response_geometry_v3/R1_analysis --factors {SERVER_BASE}/generated/FactorsCalibrated --geometry {release}/geometry.npz --grid-config {release}/config.json --training-law {release}/transfer_training_summary.json --points {points}'
    body=f'''#!/usr/bin/env bash
set -euo pipefail
exec 9>{REMOTE}/candidate.lock
flock -n 9
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd {release}
{PYTHON} -c {shlex.quote(idle_program())}
timeout --signal=TERM --kill-after=30s 2h {PYTHON} validate_energy_candidate_v5.py q {args} --device cuda:0 --q-per-source 32 --output {o['q']}
{PYTHON} -c {shlex.quote(idle_program())}
timeout --signal=TERM --kill-after=30s 2h {PYTHON} validate_energy_candidate_v5.py spatial {args} --device cuda:0 --output {o['spatial']}
echo ENERGY_CANDIDATE_CHAIN_FINISHED
'''
    key=hashlib.sha256(body.encode()).hexdigest()[:16];script=DATA/('resume_'+key+'.sh')
    script.write_text(body,encoding='ascii',newline='\n');remote_script=release+'/'+script.name
    transfer(SERVER,script,remote_script);log=REMOTE+'/resume_'+key+'.log'
    pid=ssh(SERVER,'nohup bash '+shlex.quote(remote_script)+' > '+shlex.quote(log)+' 2>&1 < /dev/null & echo $!')
    if not pid.isdigit():raise ValueError('Unexpected PID')
    write(REPORT/f'job_failed_idle_gate_{j["pid"]}.json',j)
    j.update(pid=int(pid),log=log,launcher_sha256=digest(script),reused_completed_phases=['points','probe'])
    write(REPORT/'job.json',j);write(REPORT/'resume.json',dict(new_pid=int(pid),old_pid_exited=True,
        scientific_code_unchanged=True,launcher_sha256=digest(script),reason='Bounded GPU telemetry grace period'))
    print('CANDIDATE_REMAINING_PHASES_PID',pid)


def replace_stopped_candidate():
    """Archive a proven exited candidate; never reuse its scientific outputs."""
    receipt=json.loads((REPORT/'job_stopped_node_truncation_652867.json').read_text())
    j=json.loads((REPORT/'job.json').read_text())
    if j['pid']!=receipt['job']['pid'] or not receipt['stop']['all_exited']:
        raise ValueError('Stopped-candidate identity mismatch')
    code='''import json,sys;from pathlib import Path
j=json.loads(sys.argv[1]);p=Path('/proc/'+str(j['pid'])+'/stat')
if p.exists():
 s=p.read_text();assert s[s.rfind(')')+2:].split()[0]=='Z','Previous candidate still active'
print('PREVIOUS_CANDIDATE_EXIT_VERIFIED')
'''
    print(ssh(SERVER,PYTHON+' -c '+shlex.quote(code)+' '+shlex.quote(json.dumps(j))))
    destination=REPORT/('superseded_'+Path(j['release']).name)
    destination.mkdir(exist_ok=False)
    for name in ('job.json','deployment.json','config.json','resume.json','collection.json','validation_gate.json'):
        path=REPORT/name
        if path.exists():path.rename(destination/name)
    evidence=REPORT/'evidence'
    if evidence.exists():evidence.rename(destination/'evidence')
    write(destination/'superseded.json',dict(reason='Discrete physical-support node count creates density jumps',
        scientific_outputs_rejected=True,remote_outputs_preserved=True,paired_reconstruction_submitted=False))
    start()


def replace_numerical_hold():
    j=json.loads((REPORT/'job.json').read_text())
    code='''import json,sys;from pathlib import Path
j=json.loads(sys.argv[1]);p=Path('/proc/'+str(j['pid'])+'/stat')
if p.exists():
 s=p.read_text();assert s[s.rfind(')')+2:].split()[0]=='Z','Previous candidate still active'
g=json.loads((Path(j['outputs']['points'])/'point_gate.json').read_text())
assert not g['quadrature']['passed'],'Only a failed numerical gate may be replaced here'
assert not Path(j['outputs']['probe']).exists(),'GPU phase unexpectedly started'
print('NUMERICAL_HOLD_AND_EXIT_VERIFIED')
'''
    print(ssh(SERVER,PYTHON+' -c '+shlex.quote(code)+' '+shlex.quote(json.dumps(j))))
    destination=REPORT/('superseded_'+Path(j['release']).name);destination.mkdir(exist_ok=False)
    for name in ('job.json','deployment.json','config.json','collection.json','validation_gate.json'):
        path=REPORT/name
        if path.exists():path.rename(destination/name)
    evidence=REPORT/'evidence'
    if evidence.exists():evidence.rename(destination/'evidence')
    write(destination/'superseded.json',dict(reason='GL2 vs GL4 density numerical error exceeds fixed 1% gate',
        scientific_outputs_rejected=True,remote_outputs_preserved=True,paired_reconstruction_submitted=False))
    start()


def replace_resource_hold():
    j=json.loads((REPORT/'job.json').read_text())
    code='''import json,sys;from pathlib import Path
j=json.loads(sys.argv[1]);p=Path('/proc/'+str(j['pid'])+'/stat')
if p.exists():
 s=p.read_text();assert s[s.rfind(')')+2:].split()[0]=='Z','Previous candidate still active'
r=json.loads((Path(j['outputs']['probe'])/'resource_probe.json').read_text())
assert r['estimated_full_seconds']>7200,'Expected an explicit resource hold'
for phase in ('q','spatial'):assert not Path(j['outputs'][phase]).exists(),'Later phase unexpectedly started'
print('RESOURCE_HOLD_AND_EXIT_VERIFIED')
'''
    print(ssh(SERVER,PYTHON+' -c '+shlex.quote(code)+' '+shlex.quote(json.dumps(j))))
    destination=REPORT/('superseded_'+Path(j['release']).name);destination.mkdir(exist_ok=False)
    for name in ('job.json','deployment.json','config.json','collection.json','validation_gate.json'):
        path=REPORT/name
        if path.exists():path.rename(destination/name)
    evidence=REPORT/'evidence'
    if evidence.exists():evidence.rename(destination/'evidence')
    write(destination/'superseded.json',dict(reason='Eager exact GPU implementation exceeded two-hour throughput budget',
        scientific_model_unchanged=True,remote_outputs_preserved=True,paired_reconstruction_submitted=False))
    start()


def replace_memory_hold():
    receipt=json.loads((REPORT/'job_stopped_memory_752181.json').read_text())
    j=json.loads((REPORT/'job.json').read_text())
    if j['pid']!=receipt['job']['pid'] or not receipt['stop']['all_exited']:raise ValueError('Memory stop identity mismatch')
    code='''import json,sys;from pathlib import Path
j=json.loads(sys.argv[1]);p=Path('/proc/'+str(j['pid'])+'/stat')
if p.exists():
 s=p.read_text();assert s[s.rfind(')')+2:].split()[0]=='Z','Previous candidate still active'
print('MEMORY_STOP_AND_EXIT_VERIFIED')
'''
    print(ssh(SERVER,PYTHON+' -c '+shlex.quote(code)+' '+shlex.quote(json.dumps(j))))
    destination=REPORT/('superseded_'+Path(j['release']).name);destination.mkdir(exist_ok=False)
    for name in ('job.json','deployment.json','config.json','collection.json','validation_gate.json','progress_snapshot.json'):
        path=REPORT/name
        if path.exists():path.rename(destination/name)
    evidence=REPORT/'evidence'
    if evidence.exists():evidence.rename(destination/'evidence')
    write(destination/'superseded.json',dict(reason='Full-loop GPU allocator reserved memory exceeded safe headroom',
        partial_sensitivity_rejected=True,remote_outputs_preserved=True,paired_reconstruction_submitted=False))
    start()


if __name__=='__main__':
    if sys.argv[1:]==['resume']:resume_after_idle_exit()
    elif sys.argv[1:]==['replace-stopped']:replace_stopped_candidate()
    elif sys.argv[1:]==['replace-numerical-hold']:replace_numerical_hold()
    elif sys.argv[1:]==['replace-resource-hold']:replace_resource_hold()
    elif sys.argv[1:]==['replace-memory-hold']:replace_memory_hold()
    elif not sys.argv[1:]:start()
    else:raise SystemExit('Usage: energy_candidate_v5_workflow.py [resume|replace-stopped|replace-numerical-hold|replace-resource-hold|replace-memory-hold]')
