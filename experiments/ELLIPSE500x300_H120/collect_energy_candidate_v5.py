"""Hash-checked collection and hard gate; never submits a reconstruction."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import sys
from first_scatter_pipeline import ssh,SERVER,SERVER_ROOT

HERE=Path(__file__).parent
REPORT=HERE/'reports/NEMA_Body_H60/compton_energy_probability_v5'
DATA=HERE/'generated/compton_energy_probability_v5'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def collect():
    j=json.loads((REPORT/'job.json').read_text())
    program='''from pathlib import Path
import hashlib,json,sys
j=json.loads(sys.argv[1]);result={'pid':j['pid'],'running':Path('/proc/'+str(j['pid'])+'/cmdline').exists(),'phases':{}}
for phase,name in j['outputs'].items():
 p=Path(name)
 if not (p/'execution.json').exists():continue
 files={}
 for f in p.iterdir():
  if f.is_file() and (f.suffix in ('.json','.csv','.npz') or f.name.endswith(('_Sensi_full','_Sensi_detector'))):
   files[f.name]={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
 result['phases'][phase]=files
log=Path(j['log']);result['last_log_lines']=log.read_text().splitlines()[-5:] if log.exists() else []
mem=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:')))*1024
cg=Path('/sys/fs/cgroup/memory.max');value=cg.read_text().strip() if cg.exists() else 'max'
if value.isdigit():mem=min(mem,int(value))
result['native_host_or_cgroup_limit_bytes']=mem
print(json.dumps(result))
'''
    state=json.loads(ssh(SERVER,SERVER_ROOT+'/.venv/bin/python -c '+shlex.quote(program)+' '+shlex.quote(json.dumps(j))))
    deployment=json.loads((REPORT/'deployment.json').read_text())
    for phase,files in state['phases'].items():
        local=DATA/'validation'/phase;local.mkdir(parents=True,exist_ok=True)
        for name,r in files.items():
            target=local/name
            if target.exists() and digest(target)==r['sha256']:continue
            subprocess.run(['scp','-q','-o','BatchMode=yes',SERVER+':'+j['outputs'][phase]+'/'+name,str(target)],check=True)
            if digest(target)!=r['sha256']:raise ValueError('Fetched SHA mismatch: '+phase+'/'+name)
        execution=json.loads((local/'execution.json').read_text())
        for key,name in [('source_sha256','validate_energy_candidate_v5.py'),('candidate_sha256','compton_energy_probability_v5.py')]:
            if execution[key]!=deployment['code_sha256'][name]:raise ValueError('Collected phase belongs to another release: '+phase)
    write(REPORT/'collection.json',state)
    evidence=REPORT/'evidence';evidence.mkdir(exist_ok=True)
    for phase,names in {'points':['point_gate.json'],'probe':['execution.json','resource_probe.json','fusion_gate.json'],
                        'q':['q_selected_probability.json','execution.json'],
                        'spatial':['spatial_gate.json','joint_gate.json','execution.json']}.items():
        if phase not in state['phases']:continue
        for name in names:
            source=DATA/'validation'/phase/name
            if source.exists():(evidence/(phase+'_'+name)).write_bytes(source.read_bytes())
    if set(state['phases'])!=set(('points','probe','q','spatial')):
        status='RUNNING' if state['running'] else 'INCOMPLETE_HOLD'
        reasons=[]
        if not state['running']:
            if 'points' in state['phases']:
                pg=json.loads((DATA/'validation/points/point_gate.json').read_text())
                if not pg['quadrature']['passed']:reasons.append('Numerical density/reference convergence')
                if 'hybrid' in pg['quadrature'] and not pg['quadrature']['hybrid']['passed']:reasons.append('Numerical forward density convergence')
            if 'probe' in state['phases']:
                rg=json.loads((DATA/'validation/probe/resource_probe.json').read_text())
                if rg['estimated_full_seconds']>=7200:reasons.append('Full calculation exceeds fixed two-hour budget')
        write(REPORT/'validation_gate.json',dict(status=status,paired_reconstruction_submitted=False,
            failures=reasons,completed_phases=list(state['phases']),new_photons=0,new_fine_A_matrices=0))
        print(status,'completed',list(state['phases']));print('\n'.join(state['last_log_lines']));return
    local=DATA/'validation'
    p=json.loads((local/'points/point_gate.json').read_text())
    q=json.loads((local/'q/q_selected_probability.json').read_text())
    s=json.loads((local/'spatial/spatial_gate.json').read_text())
    joint=json.loads((local/'spatial/joint_gate.json').read_text())
    resource=json.loads((local/'spatial/execution.json').read_text())
    failures=[]
    if p['scored_events']!=10270 or p['dropped_events']!=0:failures.append('Point scoring identity')
    if not p['quadrature']['passed']:failures.append('Energy quadrature convergence')
    if not p['quadrature']['hybrid']['passed']:failures.append('Numerical forward density convergence')
    if any(r['mean_gain_nats']<=0 for r in p['reports']):failures.append('Independent energy likelihood')
    if not q['membership_matches'] or not q['all_refinements_pass']:failures.append('Fixed q normalization')
    if s['original_S_relative_L2']>1e-5:failures.append('Original S regression')
    if s['candidate_hold_bins']:failures.append('Independent spatial efficiency')
    if not 0<=s['candidate_min_cell_detection_probability']<=s['candidate_max_cell_detection_probability']<=1:
        failures.append('Valid undetected probability')
    if abs(s['ellipse']['candidate']['relative_bias'])>.05:failures.append('Independent ellipse efficiency')
    if joint['candidate_hold_categories']:failures.append('Independent joint measurement categories')
    if resource['gpu_peak_reserved_bytes']>.8*resource['gpu_total_bytes']:failures.append('GPU reserve')
    if resource['host_peak_rss_bytes']>.8*state['native_host_or_cgroup_limit_bytes']:failures.append('RAM reserve')
    for field,value in [('train_events',159919),('validation_events',159923),
                        ('actual_training_photons',1_000_000_000),('actual_validation_photons',1_000_000_000)]:
        if s[field]!=value:failures.append('Frozen count '+field)
    write(REPORT/'validation_gate.json',dict(status='HOLD' if failures else 'DIAGNOSTIC_GATES_PASSED',failures=failures,
        paired_reconstruction_submitted=False,production_pilot_completed=False,
        calibrated_surrogate_not_unconditional_physical_certificate=True,
        next_step='Diagnose failed categories; no imaging' if failures else 'Whole-cell basis regression and complete-event pilot before pairing',
        source_config_sha256=digest(REPORT/'config.json'),collection_sha256=digest(REPORT/'collection.json')))
    print('FINAL_GATE','HOLD' if failures else 'DIAGNOSTIC_GATES_PASSED',failures)


if __name__=='__main__':collect()
