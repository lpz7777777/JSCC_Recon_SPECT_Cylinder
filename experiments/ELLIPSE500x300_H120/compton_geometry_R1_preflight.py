"""Freeze and submit only R0 50-iteration regression plus R1 10-iteration pilot.

R2 A accuracy is still on HOLD. This entry deliberately cannot submit formal
imaging; it reuses immutable ideal List/CntStat and does not rerun transport.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import sys
import tarfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/'experiments/FOV120')]
from reconstruction_ssh import connect
from first_scatter_imaging import command
from compton_geometry_v3_workflow import DATA,REPORT,digest,write

REMOTE_ROOT='/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor'
BASE=REMOTE_ROOT+'/experiments/ELLIPSE500x300_H120'
REMOTE=BASE+'/generated/compton_response_geometry_v3'
OLD=BASE+'/generated/compton_first_scatter_v2'


def freeze():
    output=DATA/'R1_preflight';output.mkdir(exist_ok=False)
    gate=DATA/'R1_analysis/validation_gate.json';spatial=DATA/'spatial_validation/spatial_gate.json'
    for path in (gate,spatial):
        if json.loads(path.read_text())['status']!='PASSED':raise ValueError('R1 independent gate not passed')
    old=json.loads((HERE/'generated/compton_first_scatter_v2/configs/ideal.json').read_text())
    scan=json.loads(gate.read_text())['scans']['NEMA_ideal']
    cfg=dict(old,study='compton_response_geometry_v3',variant='R1_stable_point',
        kernel_sha256=digest(ROOT/'compton_event_response.py'),validation_gate_sha256=digest(gate),
        validation_evidence_sha256={'validation_gate.json':digest(gate),'spatial_gate.json':digest(spatial)},
        baseline_accepted_compton_events=old['kept_compton_events'],baseline_sensi_d_sha256=old['sensi_d_sha256'],
        baseline_result='formal_ideal_1660254',baseline_per_view=old['kept_per_view'],
        kept_compton_events=scan['kept'],removed_compton_events=scan['uncut']-scan['kept'],
        kept_per_view=[r['kept'] for r in scan['records']],
        scan_manifest_sha256=digest(gate),sensi_d_sha256=digest(DATA/'R1_analysis/ideal/Sensi_d'))
    if cfg['baseline_accepted_compton_events']!=91231 or cfg['kept_compton_events']!=91225:
        raise ValueError('Frozen R0/R1 event closure differs')
    write(output/'R1.json',cfg);shutil.copy2(gate,output/'validation_gate.json');shutil.copy2(spatial,output/'spatial_gate.json')
    for name in ('Sensi_d','Sensi_d_provenance.json'):shutil.copy2(DATA/'R1_analysis/ideal'/name,output/name)
    print('R1_PREFLIGHT_FROZEN_NO_FORMAL')


def deploy():
    configs=DATA/'R1_preflight';cfg=json.loads((configs/'R1.json').read_text())
    from compton_geometry_run_contract import validate_run
    validate_run(cfg,regression=False,pilot=True,dry_run=False,iterations=10,save_step=10,
        dataset='NEMA_Body_H60',level='1e9',channels='compton-jscc',sensitivity=configs/'Sensi_d',
        geometry_sha=digest(HERE/'generated/Geometry/geometry.npz'),kernel_sha=digest(ROOT/'compton_event_response.py'),
        digest=digest,config_directory=configs)
    names=('run_reconstruction.py','torch_active_operator.py','validate_factors.py','geometry.py','config.json',
        'verify_first_scatter.py','compton_geometry_run_contract.py','test_compton_geometry_run_contract.py',
        'reconstruct_compton_geometry_preflight.sh')
    paths={n:HERE/n for n in names}
    paths.update({n:ROOT/n for n in ('compton_event_response.py','compton_sparse_ops.py','process_list_plane_sparse.py','detector_csv.py')})
    paths.update({n:configs/n for n in ('R1.json','validation_gate.json','spatial_gate.json','Sensi_d','Sensi_d_provenance.json')})
    paths['geometry.npz']=HERE/'generated/Geometry/geometry.npz'
    hashes={n:digest(p) for n,p in paths.items()}
    key=__import__('hashlib').sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/preflight_releases/'+key
    with connect() as client:
        command(client,'mkdir -p -- '+shlex.quote(release)+' '+shlex.quote(REMOTE+'/logs'))
        with client.open_sftp() as sftp:
            for name,path in paths.items():
                sftp.put(str(path),release+'/'+name)
                if command(client,'sha256sum '+shlex.quote(release+'/'+name)).split()[0]!=hashes[name]:
                    raise ValueError('Frozen preflight transfer differs')
        # Reuse existing per-file-verified inputs, do not copy or overwrite them.
        for relative,sha in cfg['baseline_input_sha256'].items():
            if command(client,'sha256sum '+shlex.quote(OLD+'/recon_inputs/ideal/'+relative)).split()[0]!=sha:
                raise ValueError('Original ideal transport input changed')
        if command(client,'sha256sum '+shlex.quote(OLD+'/analysis/ideal/Sensi_d')).split()[0]!=cfg['baseline_sensi_d_sha256']:
            raise ValueError('Historical q3 sensitivity changed')
        command(client,'test -r '+shlex.quote(OLD+'/formal_ideal_1660254/run_manifest.json'))
        for name,sha in cfg['baseline_factor_manifest_sha256'].items():
            if command(client,'sha256sum '+shlex.quote(BASE+'/generated/FactorsCalibrated/'+name+'/factor_manifest.json')).split()[0]!=sha:
                raise ValueError('Original Factors changed')
        tests=command(client,'cd '+shlex.quote(release)+' && /data/home/scxi717/.conda/envs/torch/bin/python -m unittest test_compton_geometry_run_contract -v 2>&1')
        if 'Ran 1 test' not in tests or '\nOK' not in tests:raise ValueError('Remote regression contract failed')
        (REPORT/'R1_preflight_tests.txt').write_text(tests+'\n')
        command(client,'bash -n '+shlex.quote(release+'/reconstruct_compton_geometry_preflight.sh'))
    write(REPORT/'R1_preflight_deployment.json',dict(release=release,code_sha256=hashes,
        reused_input_root=OLD+'/recon_inputs/ideal',formal_submission_permitted=False))
    print('R1_PREFLIGHT_DEPLOYED',release)


def submit(nodes):
    if nodes not in (4,8):raise ValueError('Only 4/8 distinct one-GPU nodes')
    if (REPORT/'R1_preflight_job.json').exists():raise ValueError('Preflight already registered')
    release=json.loads((REPORT/'R1_preflight_deployment.json').read_text())['release']
    with connect() as client:
        queue=command(client,"squeue -u scxi717 -h -o '%i %j %T'")
        if 'Compton_v3_preflight' in queue:raise ValueError('Duplicate preflight refused')
        if len(queue.splitlines())>=50:
            write(REPORT/'R1_preflight_wait.json',dict(reason='account_50_job_limit',other_jobs_untouched=True));print('ACCOUNT_LIMIT_WAIT');return
        command_text=(f'sbatch --parsable -N {nodes} --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=6 '+
            '-p gpu_4090,gpu_5090 --qos=gpugpu --time=01:30:00 --chdir=/tmp --job-name=Compton_v3_preflight '+
            '--exclude=wqd10nba06g6 --export='+shlex.quote('ALL,GEOMETRY_V3_RELEASE='+release)+
            ' --output='+shlex.quote(REMOTE+'/logs/preflight.%j.out')+
            ' --error='+shlex.quote(REMOTE+'/logs/preflight.%j.err')+' '+shlex.quote(release+'/reconstruct_compton_geometry_preflight.sh'))
        job=command(client,command_text).split(';')[0]
        if not job.isdigit():raise ValueError('Ambiguous submit outcome; inspect before retry')
    write(REPORT/'R1_preflight_job.json',dict(job=int(job),nodes=nodes,gpus_per_node=1,release=release,
        submitted_utc=datetime.now(timezone.utc).isoformat(),phases=['R0 legacy geometry with q3:50','R1 stable geometry with q3:10'],
        formal_submission_permitted=False,nccl_interface='bond0',walltime_minutes=90))
    print('R1_PREFLIGHT_JOB',job)


def status():
    if not (REPORT/'R1_preflight_job.json').exists():print('NO_REGISTERED_PREFLIGHT');return
    job=str(json.loads((REPORT/'R1_preflight_job.json').read_text())['job'])
    with connect() as client:
        print(command(client,f"squeue -j {job} -h -o '%i %T %M %D %R'; sacct -X -j {job} -n -P --format=JobIDRaw,State,ExitCode"))
        for suffix in ('out','err'):
            path=shlex.quote(REMOTE+'/logs/preflight.'+job+'.'+suffix)
            print(command(client,'if test -f '+path+'; then tail -n 6 '+path+'; fi'))


def fetch():
    job=str(json.loads((REPORT/'R1_preflight_job.json').read_text())['job'])
    target=DATA/'R1_preflight_results'/job;target.mkdir(parents=True,exist_ok=True)
    with connect() as client:
        accounting=command(client,f'sacct -j {job} -n -P --format=JobIDRaw,State,ExitCode,MaxRSS,ReqMem,AllocTRES')
        overall=[r.split('|') for r in accounting.splitlines() if r.split('|')[0]==job]
        if len(overall)!=1 or overall[0][1:3]!=['COMPLETED','0:0']:
            raise ValueError('Preflight still running or unsuccessful')
        summaries=[]
        for phase in ('regression','pilot'):
            remote=REMOTE+'/preflight_'+phase+'_'+job;folder=target/phase;folder.mkdir(exist_ok=True)
            with client.open_sftp() as sftp:
                for name in ('verification.json','run_manifest.json'):
                    sha=command(client,'sha256sum '+shlex.quote(remote+'/'+name)).split()[0]
                    sftp.get(remote+'/'+name,str(folder/name))
                    if digest(folder/name)!=sha:raise ValueError('Preflight evidence transfer differs')
            v=json.loads((folder/'verification.json').read_text())
            if not v['passed'] or v['mode']!=phase:raise ValueError('Numerical preflight did not pass')
            summaries.append(dict(phase=phase,verification_sha256=digest(folder/'verification.json'),
                run_manifest_sha256=digest(folder/'run_manifest.json'),accepted_events=v['accepted_events'],
                outputs=v['outputs'],resources=v['resources']))
        import re
        rss=[]
        for line in accounting.splitlines():
            value=line.split('|')[3];match=re.fullmatch(r'([0-9.]+)([KMGT])',value)
            if match:rss.append(float(match[1])*1024**('KMGT'.index(match[2])+1))
        if not rss:raise ValueError('Slurm actual MaxRSS missing; not yet accepted')
        minimum=min(r['host_allocated_bytes'] for s in summaries for r in s['resources'])
        if max(rss)/minimum>.8:raise ValueError('Slurm actual memory margin failed')
    write(REPORT/'R1_preflight_summary.json',dict(status='PASSED',job=int(job),phases=summaries,
        slurm_accounting=accounting,slurm_peak_rss_bytes=max(rss),minimum_granted_host_bytes=minimum,
        formal_submission_permitted=False,paired_imaging_completed=False))
    print('R1_PREFLIGHT_VERIFIED',job)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('freeze','deploy','submit','status','fetch'))
    p.add_argument('--nodes',type=int,default=4);a=p.parse_args()
    if a.mode=='freeze':freeze()
    elif a.mode=='deploy':deploy()
    elif a.mode=='submit':submit(a.nodes)
    elif a.mode=='fetch':fetch()
    else:status()
