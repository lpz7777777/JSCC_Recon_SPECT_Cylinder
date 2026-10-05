"""Freeze, deploy and submit only original regression + two complete-event pilots."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import sys
import tarfile
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/'experiments/FOV120')]
from reconstruction_ssh import connect
from first_scatter_imaging import command,REMOTE as BASE
from verify_first_scatter import digest

DATA=HERE/'generated/compton_energy_probability_v5'
REPORT=HERE/'reports/NEMA_Body_H60/compton_energy_probability_v5'
REMOTE=BASE+'/generated/compton_energy_probability_v5'
OLD=BASE+'/generated/compton_first_scatter_v2'
PYTHON='/data/home/scxi717/.conda/envs/torch/bin/python'


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def freeze():
    gate=json.loads((REPORT/'validation_gate.json').read_text())
    if gate['status']!='DIAGNOSTIC_GATES_PASSED' or gate['failures']:raise ValueError('Diagnostic gate is on HOLD')
    r1=HERE/'generated/compton_response_geometry_v3/R1_preflight'
    cfg=json.loads((r1/'R1.json').read_text())
    if digest(ROOT/'compton_event_response.py')!=cfg['kernel_sha256']:raise ValueError('Shared kernel changed')
    output=DATA/'preflight_payload';output.mkdir(exist_ok=False)
    names=('run_reconstruction.py','torch_active_operator.py','validate_factors.py','geometry.py','config.json',
        'verify_first_scatter.py','compton_geometry_run_contract.py','test_compton_geometry_run_contract.py',
        'run_energy_preflight_v5.py','verify_energy_preflight_v5.py','test_energy_preflight_v5.py',
        'compton_energy_probability_v5.py','test_compton_energy_probability_v5.py','reconstruct_energy_preflight_v5.sh')
    paths={n:HERE/n for n in names}
    paths.update({n:ROOT/n for n in ('compton_event_response.py','compton_sparse_ops.py','process_list_plane_sparse.py','detector_csv.py')})
    paths.update({'geometry.npz':HERE/'generated/Geometry/geometry.npz',
        'whole_geometry.npz':HERE/'generated/process_list_global_audit_v4/WholeCellGeometry/geometry.npz',
        'diagnostic_gate.json':REPORT/'validation_gate.json',
        'transfer_training_summary.json':HERE/'reports/NEMA_Body_H60/process_list_global_audit_v4/evidence/energy_energy_probability_summary.json',
        'angular_Sensi_full':DATA/'validation/spatial/legacy_Sensi_full',
        'continuous_energy_Sensi_full':DATA/'validation/spatial/candidate_Sensi_full'})
    paths.update({'legacy_regression/'+n:r1/n for n in ('R1.json','validation_gate.json','spatial_gate.json')})
    for view in range(1,21):
        paths[f'selections/{view}.npy']=HERE/f'generated/compton_response_geometry_v3/R1_analysis/NEMA_ideal_v{view:02d}_kept_rows.npy'
    counts=[]
    for v in range(1,21):
        indices=np.load(paths[f'selections/{v}.npy'])
        if indices.ndim!=1 or indices.dtype.kind not in 'iu' or np.any(indices[1:]<=indices[:-1]) or np.any(indices<0):
            raise ValueError('Frozen R1 selections invalid')
        counts.append(len(indices))
    if counts!=cfg['kept_per_view'] or sum(counts)!=91225:raise ValueError('Frozen R1 per-view closure differs')
    collection=json.loads((REPORT/'collection.json').read_text())
    for name in ('legacy_Sensi_full','candidate_Sensi_full'):
        path=DATA/'validation/spatial'/name
        expected=collection['phases']['spatial'][name]
        expected=expected['sha256'] if isinstance(expected,dict) else expected
        if digest(path)!=expected:raise ValueError('Received matched S differs from diagnostic collection')
    for name,path in paths.items():
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    hashes={n:digest(output/n) for n in paths}
    contract=dict(study='compton_energy_probability_v5_preflight',files=hashes,
        input_sha256=cfg['baseline_input_sha256'],factor_manifest_sha256=cfg['baseline_factor_manifest_sha256'],
        whole_geometry_sha256=hashes['whole_geometry.npz'],original_geometry_sha256=hashes['geometry.npz'],
        events_per_view=counts,accepted_events=91225,selection='frozen stable_float64 q<=3 on full 132040 grid',
        diagnostic_collection_sha256=digest(REPORT/'collection.json'),
        formal_submission_permitted=False,new_photons=0,paired_iterations_not_authorized_by_this_entry=True)
    write(output/'contract.json',contract)
    hashes['contract.json']=digest(output/'contract.json')
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    write(REPORT/'preflight_freeze.json',dict(release_key=key,sha256=hashes,
        contract_sha256=hashes['contract.json'],formal_submission_permitted=False))
    print('ENERGY_V5_PREFLIGHT_FROZEN',key)


def deploy():
    payload=DATA/'preflight_payload';record=json.loads((REPORT/'preflight_freeze.json').read_text())
    for n,sha in record['sha256'].items():
        if digest(payload/n)!=sha:raise ValueError('Frozen local payload changed')
    original=json.loads((REPORT/'deployment.json').read_text())['code_sha256']
    for n in ('compton_energy_probability_v5.py','compton_event_response.py','geometry.npz',
        'whole_geometry.npz','transfer_training_summary.json'):
        if record['sha256'][n]!=original[n]:raise ValueError('Pilot candidate differs from independently validated candidate')
    archive_r1=HERE/'generated/compton_response_geometry_v3/R1_analysis.tar.gz'
    original_r1=json.loads((HERE/'reports/NEMA_Body_H60/compton_response_geometry_v3/R1_summary.json').read_text())
    if digest(archive_r1)!=original_r1['archive_sha256']:raise ValueError('Original R1 selection archive differs')
    with tarfile.open(archive_r1) as f:
        for v in range(1,21):
            member=f.getmember(f'R1_analysis/NEMA_ideal_v{v:02d}_kept_rows.npy')
            if hashlib.sha256(f.extractfile(member).read()).hexdigest()!=record['sha256'][f'selections/{v}.npy']:
                raise ValueError('Frozen event membership differs from validated R1')
    release=REMOTE+'/preflight_releases/'+record['release_key']
    archive=DATA/'preflight_payload.tar.gz'
    with tarfile.open(archive,'w:gz') as f:
        for n in sorted(record['sha256']):f.add(payload/n,arcname=n)
    with connect() as c:
        command(c,'mkdir -p -- '+shlex.quote(REMOTE+'/logs')+' '+shlex.quote(release))
        with c.open_sftp() as s:s.put(str(archive),release+'/payload.tar.gz')
        if command(c,'sha256sum -- '+shlex.quote(release+'/payload.tar.gz')).split()[0]!=digest(archive):
            raise ValueError('Frozen archive transfer differs')
        command(c,'tar --no-same-owner -xzf '+shlex.quote(release+'/payload.tar.gz')+' -C '+shlex.quote(release))
        cfg=json.loads((payload/'contract.json').read_text());r1=json.loads((payload/'legacy_regression/R1.json').read_text())
        remote_checks={release+'/'+n:s for n,s in record['sha256'].items()}
        remote_checks.update({OLD+'/recon_inputs/ideal/'+n:s for n,s in cfg['input_sha256'].items()})
        remote_checks.update({BASE+'/generated/FactorsCalibrated/'+n+'/factor_manifest.json':s for n,s in cfg['factor_manifest_sha256'].items()})
        remote_checks[OLD+'/analysis/ideal/Sensi_d']=r1['baseline_sensi_d_sha256']
        script='import hashlib,json; checks=json.loads('+repr(json.dumps(remote_checks))+'); '+\
            '[(None if hashlib.sha256(open(p,"rb").read()).hexdigest()==s else (_ for _ in ()).throw(ValueError(p))) for p,s in checks.items()]; print("FROZEN_TRANSFER_AND_INPUTS_VERIFIED",len(checks))'
        print(command(c,PYTHON+' -c '+shlex.quote(script)))
        tests=command(c,'cd '+shlex.quote(release)+' && JSCC_PROJECT_ROOT='+shlex.quote(release)+' '+PYTHON+
            ' -m unittest test_compton_geometry_run_contract test_compton_energy_probability_v5 test_energy_preflight_v5 -v 2>&1')
        if 'Ran 17 tests' not in tests or '\nOK' not in tests:raise ValueError('Frozen remote tests failed: '+tests)
        (REPORT/'preflight_tests.txt').write_text(tests+'\n')
        command(c,'bash -n '+shlex.quote(release+'/reconstruct_energy_preflight_v5.sh'))
        command(c,'test -r '+shlex.quote(OLD+'/formal_ideal_1660254/run_manifest.json'))
    write(REPORT/'preflight_deployment.json',dict(release=release,sha256=record['sha256'],
        tests_passed=17,tests_sha256=digest(REPORT/'preflight_tests.txt'),reused_input_root=OLD+'/recon_inputs/ideal',
        formal_submission_permitted=False))
    print('ENERGY_V5_PREFLIGHT_DEPLOYED',release)


def submit(nodes):
    if nodes not in (4,8):raise ValueError('Four/eight distinct one-GPU nodes only')
    if (REPORT/'preflight_job.json').exists():raise ValueError('Preflight already registered; inspect before any repair')
    record=json.loads((REPORT/'preflight_deployment.json').read_text());release=record['release']
    with connect() as c:
        queue=command(c,"squeue -u scxi717 -h -o '%i %j %T'")
        if 'Energy_v5_preflight' in queue:raise ValueError('Duplicate active pilot refused')
        if len(queue.splitlines())>=50:
            write(REPORT/'preflight_wait.json',dict(reason='account_50_job_limit',other_jobs_untouched=True));return
        text=(f'sbatch --parsable -N {nodes} --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=6 '+
            '-p gpu_4090,gpu_5090 --qos=gpugpu --time=01:30:00 --chdir=/tmp --job-name=Energy_v5_preflight '+
            '--exclude=wqd10nba06g6 --export='+shlex.quote('ALL,ENERGY_V5_RELEASE='+release)+
            ' --output='+shlex.quote(REMOTE+'/logs/preflight.%j.out')+' --error='+shlex.quote(REMOTE+'/logs/preflight.%j.err')+
            ' '+shlex.quote(release+'/reconstruct_energy_preflight_v5.sh'))
        job=command(c,text).split(';')[0]
        if not job.isdigit():raise ValueError('Ambiguous submission; inspect queue before retry')
    write(REPORT/'preflight_job.json',dict(job=int(job),nodes=nodes,gpus_per_node=1,release=release,
        submitted_utc=datetime.now(timezone.utc).isoformat(),phases=['legacy regression 50','angular whole-cell 10','continuous_energy whole-cell 10'],
        nccl_interface='bond0',host_memory_policy='Slurm GPU-count default; no explicit mem option permitted',walltime_minutes=90,
        phase_hard_timeout_minutes=25,formal_submission_permitted=False))
    print('ENERGY_V5_PREFLIGHT_JOB',job)


def status():
    job=str(json.loads((REPORT/'preflight_job.json').read_text())['job'])
    with connect() as c:
        print(command(c,f"squeue -j {job} -h -o '%i %T %M %D %R'"))
        print(command(c,f'sacct -j {job} -n -P --format=JobIDRaw,State,ExitCode,MaxRSS,AllocTRES'))
        for suffix in ('out','err'):
            path=shlex.quote(REMOTE+'/logs/preflight.'+job+'.'+suffix)
            print(command(c,'if test -f '+path+'; then tail -n 12 '+path+'; fi'))


def fetch():
    job=str(json.loads((REPORT/'preflight_job.json').read_text())['job'])
    target=DATA/'preflight_results'/job;target.mkdir(parents=True,exist_ok=True)
    with connect() as c:
        accounting=command(c,f'sacct -j {job} -n -P --format=JobIDRaw,State,ExitCode,MaxRSS,AllocTRES')
        overall=[r.split('|') for r in accounting.splitlines() if r.split('|')[0]==job]
        if len(overall)!=1 or overall[0][1:3]!=['COMPLETED','0:0']:raise ValueError('Preflight running or failed')
        phases=[]
        for phase in ('regression','angular','continuous_energy'):
            remote=REMOTE+'/preflight_'+phase+'_'+job;folder=target/phase;folder.mkdir(exist_ok=True)
            with c.open_sftp() as s:
                for name in ('verification.json','run_manifest.json'):
                    sha=command(c,'sha256sum -- '+shlex.quote(remote+'/'+name)).split()[0]
                    s.get(remote+'/'+name,str(folder/name))
                    if digest(folder/name)!=sha:raise ValueError('Evidence transfer differs')
            v=json.loads((folder/'verification.json').read_text())
            if not v['passed'] or (v.get('model',v.get('mode'))!=phase):raise ValueError('Phase not actually verified')
            phases.append(dict(phase=phase,verification_sha256=digest(folder/'verification.json'),
                run_manifest_sha256=digest(folder/'run_manifest.json'),accepted_events=v['accepted_events'],
                outputs=v['outputs'],resources=v['resources']))
        rss=[]
        for line in accounting.splitlines():
            m=re.fullmatch(r'([0-9.]+)([KMGT])',line.split('|')[3])
            if m:rss.append(float(m[1])*1024**('KMGT'.index(m[2])+1))
        minimum=min(r['host_allocated_bytes'] for p in phases for r in p['resources'])
        if not rss or max(rss)>.8*minimum:raise ValueError('Actual Slurm MaxRSS missing or margin fails')
    write(REPORT/'preflight_summary.json',dict(status='PASSED',job=int(job),phases=phases,
        slurm_accounting=accounting,slurm_peak_rss_bytes=max(rss),minimum_granted_host_bytes=minimum,
        formal_submission_permitted=False,paired_imaging_completed=False))
    print('ENERGY_V5_PREFLIGHT_VERIFIED',job)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('freeze','deploy','submit','status','fetch'))
    p.add_argument('--nodes',type=int,default=4);a=p.parse_args()
    if a.mode=='submit':submit(a.nodes)
    else:globals()[a.mode]()
