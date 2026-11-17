"""Independent HE Derenzo acquisition -> strict delivery -> mono MLEM.

Only four formal imaging arrays and their collectors. No response rerun,
separate validation job, GPU job, or recurring automation.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from opengate_ehe_common_20261113 import read, save, sha, now

STUDY = Path(__file__).resolve().parent
TAG = STUDY.name
ROOT = HERE.parents[1]
DATA = HERE / 'generated' / TAG
REPORT = HERE / 'reports/Derenzo' / TAG
CPU = '/WORK/maty_work/lpz/' + TAG
ENV = '/WORK/maty_work/lpz/opengate_ehe_nema5e9_20261113/environment'
LOCAL = HERE / 'generated/opengate_ehe_nema5e9_20261113/local_env/Scripts/python.exe'
DONOR = HERE / 'generated/opengate_he_nema_20261114'
DONOR_REPORT = HERE / 'reports/NEMA_Body_H60/opengate_he_nema_20261114'
ROUTES = {f'{d}_{e}': dict(dose=d, energy=e, total=n, parts=p, seed=620000000 + k * 1000000)
          for k, (d, e, n, p) in enumerate([
              ('5e9', 218, 5_000_000_000, 5), ('5e9', 440, 5_000_000_000, 5),
              ('5e10', 218, 50_000_000_000, 50), ('5e10', 440, 50_000_000_000, 50)])}


def files(path):
    return {p.relative_to(path).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(path.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name != 'release.json'}


def verify(root, records):
    for name, r in records.items():
        p = Path(root) / name
        if p.stat().st_size != r['bytes'] or sha(p) != r['sha256']:
            raise ValueError('Changed immutable member: ' + str(p))


def prepare():
    if (REPORT / 'freeze.json').exists():
        frozen = read(REPORT / 'freeze.json'); verify(DATA / 'payload', frozen['files']); return frozen
    parent = read(DONOR_REPORT / 'freeze.json')
    assert parent['release'] == '46d438089710a6ce'
    source = DONOR / parent['payload_relative']
    payload = DATA / 'payload'
    if payload.exists():
        payload.rename(DATA / ('unfrozen_prepare_' + str(time.time_ns())))
    payload.mkdir(parents=True, exist_ok=False)
    copied = ['cpu_python.sh', 'opengate_ehe_common_20261113.py', 'opengate_ehe_transport_20261113.py',
              'opengate_ehe_projector_20261113.py', 'opengate_ehe_guard_20261113.py',
              'opengate_he_geometry_20261114.py', 'opengate_ehe_archive_20261113.py',
              'opengate_ehe_recon_cpu_20261113.py']
    for name in copied:
        verify(source, {name: parent['files'][name]}); shutil.copy2(source / name, payload / name)
    kernel = DONOR / '5e9/delivery/collected/response_kernels.npz'
    strict = read(DONOR_REPORT / '5e9/strict_delivery.json')
    bridge = read(DONOR_REPORT / '5e9/maty_bridge_strict_fetch.json')
    assert strict['passed'] and strict['all_files_SHA_verified'] and bridge['passed']
    assert sha(kernel) == bridge['authority']['response_kernel_sha256'] == strict['response_kernel_sha256']
    provenance = dict(passed=True, source_release=parent['release'], response_new_primaries=0,
        response_kernel_sha256=sha(kernel), source_strict_delivery_sha256=sha(DONOR_REPORT / '5e9/strict_delivery.json'),
        source_bridge_receipt_sha256=sha(DONOR_REPORT / '5e9/maty_bridge_strict_fetch.json'),
        source_authority=bridge['authority'], physical_calibration_passed=False)
    # Change the source only. Detector, physics, digitization and smearing stay frozen.
    p = payload / 'opengate_ehe_transport_20261113.py'; t = p.read_text(encoding='utf8')
    start = t.index('    src=sim.add_source(', t.index('def make_simulation'))
    end = t.index('    hits=sim.add_actor(', start)
    t = t[:start] + '''    if job["kind"] != "imaging":
        raise ValueError("HE response is already accepted; never rerun here")
    from rod_source import add_sources
    add_sources(sim, read(Path(inputs)/"geometry.json"), job)
''' + t[end:]
    needle = '    sim,stats=make_simulation(inputs,job,output)'
    assert t.count(needle) == 2  # run and the inherited, unused self_check
    t = t.replace(needle, '''    if index == 0:
        from rod_source import source_check
        source_check(inputs,output/"source_check")
''' + needle, 1)
    p.write_text(t, encoding='utf8', newline='\n')
    # Original mono MLEM, actual per-view exposure; no unrelated other-energy image/background.
    p = payload / 'opengate_ehe_recon_cpu_20261113.py'; t = p.read_text(encoding='utf8')
    assert t.count('for energy in (440,218):') == 1 and t.count('        if energy==218:') == 1
    t = t.replace('for energy in (440,218):', 'for energy in (cfg["mono_energy_keV"],):')
    t = t.replace('        if energy==218:', '        if energy==218 and not cfg["independent_mono_acquisition"]:')
    t = t.replace('for e in ("440","218")', 'for e in (str(cfg["mono_energy_keV"]),)')
    p.write_text(t, encoding='utf8', newline='\n')
    for n in ('rod_source.py', 'collect.py', 'archive.py'):
        shutil.copy2(STUDY / n, payload / n)
    all_seeds = set(); qa = {}
    for route, spec in ROUTES.items():
        inp = payload / ('inputs_' + route); inp.mkdir()
        gpath = ROOT / f'Analysis_Plots/research/derenzo_218_440_design_20261015_equal_regions/data/derenzo_{spec["energy"]}keV_geometry.json'
        g = read(gpath); rods = g['rods']
        probabilities = np.array([r['volume_mm3'] for r in rods]); volume = float(probabilities.sum())
        probabilities /= volume
        assert len(rods) == (114 if spec['energy'] == 218 else 53)
        assert abs(volume - g['total_hot_volume_mm3']) < 1e-6
        assert all(abs(r['z_mm']) < 1e-9 and r['z_min_mm'] == -30 and r['z_max_mm'] == 30 for r in rods)
        assert all(np.hypot(r['x_mm'], r['y_mm']) + r['diameter_mm']/2 <= 135.000001 for r in rods)
        rng = np.random.default_rng(spec['seed'] + 90000)
        jobs = []; emitted = np.zeros((20, 2), dtype=np.int64)
        for view in range(20):
            for part in range(spec['parts']):
                i = len(jobs); n = spec['total'] // (20 * spec['parts'])
                j = dict(index=i, kind='imaging', view=view, angle_deg=18*view, energy_keV=spec['energy'],
                         primaries=n, seed=spec['seed']+100001+i, smear_seed=spec['seed']+400001+i,
                         rod_primary_counts=rng.multinomial(n, probabilities).tolist(), importance_weight=1.)
                assert not ({j['seed'], j['smear_seed']} & all_seeds)
                all_seeds.update([j['seed'], j['smear_seed']]); jobs.append(j)
                emitted[view, (218, 440).index(spec['energy'])] += n
        cfg = read(source / 'inputs_5e9/config.json')
        for key in ('gamma_yields', 'imaging_primary_energy_fraction_expected', 'source_axes_mm',
                    'source_origin_sha256', 'source_manifest_sha256', 'HE_response_root', 'source_parent_release',
                    'old_geometry_sha256', 'baseline_metadata_sha256'):
            cfg.pop(key, None)
        cfg.update(study=TAG, route=route, dose=spec['dose'], created_utc=now(),
            mono_energy_keV=spec['energy'], independent_mono_acquisition=True,
            imaging_primary_gamma_total=spec['total'], emitted_primaries_by_view_energy=emitted.tolist(),
            activity_integrals_mm3={str(e): volume if e == spec['energy'] else 0. for e in (218, 440)},
            primary_count_definition='actual monoenergetic gamma emissions per detector and energy; not Ac225 decays',
            source_model='analytic uniform cylinders; multinomial rod allocation proportional to exact volume',
            source_geometry_sha256=sha(gpath), source_rotation='GenericSource cylinders attached to rotated vacuum Box300x300x120',
            source_frame_max_corner_radius_mm=float(np.hypot(150,150)),
            source_frame_min_collimator_clearance_mm=float(298.5-np.hypot(150,150)),
            response_kernel_sha256=sha(kernel), response_source_release=parent['release'],
            max_response_or_imaging_concurrency=160, imaging_observations_inherited=False)
        cfg['reconstruction']['response'] = 'byte-identical accepted HE response, 11 depths x 4 phases x 2 energies'
        cfg['reconstruction']['crosstalk'] = 'zero background: independent monoenergetic acquisition'
        shutil.copy2(gpath, inp / 'geometry.json'); shutil.copy2(kernel, inp / 'response_kernels.npz')
        save(inp / 'response_provenance.json', provenance); save(inp / 'config.json', cfg); save(inp / 'imaging_jobs.json', jobs)
        totals = np.sum([j['rod_primary_counts'] for j in jobs], axis=0)
        expected = spec['total'] * probabilities
        z = (totals - expected) / np.sqrt(expected * (1-probabilities))
        assert abs(z).max() < 8 and int(totals.sum()) == spec['total']
        qa[route] = dict(workers=len(jobs), primary_total=int(emitted.sum()), rods=len(rods),
                        hot_volume_mm3=volume, max_abs_rod_allocation_z=float(abs(z).max()),
                        each_worker_exact_primaries=50_000_000, geometry_sha256=sha(gpath))
    def asts(p):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(p.read_text(encoding='utf8')).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    a, b = asts(source/'opengate_ehe_transport_20261113.py'), asts(payload/'opengate_ehe_transport_20261113.py')
    changed = [n for n in a if a[n] != b[n]]
    assert set(changed) == {'make_simulation', 'run'}
    assert a['process_singles'] == b['process_singles']
    save(payload / 'local_checks.json', dict(passed=True, routes=qa, source_seed_count=len(all_seeds),
        transport_changed_functions=changed, digitization_and_smearing_AST_identical=True,
        projector_byte_identical=True, parent_release=parent['release'], separate_validation_jobs=False))
    record = files(payload)
    release = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:16]
    frozen = dict(passed=True, release=release, files=record, created_utc=now(), parent_release=parent['release'])
    save(payload / 'release.json', frozen); save(REPORT / 'freeze.json', frozen)
    return dict(release=release, routes=qa, new_gamma_total=sum(x['total'] for x in ROUTES.values()))


def remote_release():
    return CPU + '/releases/' + read(REPORT / 'freeze.json')['release']


def deploy():
    from campaign_20261102_io import upload_files, python_json
    frozen = read(REPORT / 'freeze.json'); verify(DATA / 'payload', frozen['files'])
    upload_files('maty', remote_release(), {**{n: DATA/'payload'/n for n in frozen['files']}, 'release.json': DATA/'payload/release.json'})
    src = '''import os,pathlib,json,hashlib,subprocess,sys
a=json.loads(sys.argv[1]);r=pathlib.Path(a['release']);f=json.loads((r/'release.json').read_text())
for n,x in f['files'].items():
 p=r/n
 assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],n
root=pathlib.Path(a['root']);s=os.statvfs(root)
q=subprocess.run(['lfs','quota','-u','maty',str(root)],capture_output=True,text=True,timeout=30)
if q.returncode:raise RuntimeError('Cannot establish account quota')
if s.f_bavail*s.f_frsize<32*(1<<30):raise RuntimeError('Physical storage margin')
# Actual bounded allocation also tests account quota, not merely whole-volume availability.
reservation=root/'capacity_reservation.bin'
if not reservation.exists():
 with reservation.open('xb') as h:os.posix_fallocate(h.fileno(),0,8*(1<<30));h.flush();os.fsync(h.fileno())
if reservation.stat().st_size!=8*(1<<30) or reservation.stat().st_blocks*512<8*(1<<30):raise RuntimeError('Quota reservation failed')
for name in a['routes']:
 p=root/name;p.mkdir(exist_ok=True)
 for d in ('control','resources','logs','scripts','workers','collected'): (p/d).mkdir(exist_ok=True)
 link=p/'inputs';target=r/('inputs_'+name)
 if link.exists():assert link.resolve()==target.resolve()
 else:link.symlink_to(target)
print('CAMPAIGN_JSON '+json.dumps(dict(passed=True,release=f['release'],quota=q.stdout,available_bytes=s.f_bavail*s.f_frsize,reserved_bytes=8*(1<<30))))'''
    result = python_json('maty', 'python3', src, dict(root=CPU, release=remote_release(), routes=list(ROUTES)), timeout=120, readonly=False)
    save(REPORT / 'deployment.json', result); return result


def submit(route, stage, dependency=None):
    from campaign_20261102_io import upload_files, python_json
    from ehe_5e9_workflow import q
    receipt = REPORT / route / (stage + '_job.json')
    if receipt.exists(): return read(receipt)
    assert read(REPORT / 'deployment.json')['passed']
    root = CPU + '/' + route; rel = remote_release(); spec = ROUTES[route]
    body = '#!/bin/bash\nset -euo pipefail\ncd ' + q(rel) + '\n'
    flags = ['--partition=cnmix', '-N','1','-n','1','-c','1','--time=72:00:00']
    if stage == 'imaging':
        n = spec['parts'] * 20
        body += 'index="${SLURM_ARRAY_TASK_ID:?}"\nprintf -v index4 "%04d" "$index"\n'
        body += f'allocation={q(root+"/resources/imaging_")}"$index4".allocation.txt\n'
        body += 'scontrol show job "${SLURM_JOB_ID:?}" > "$allocation"\n'
        body += (f'bash {q(rel+"/cpu_python.sh")} {q(rel+"/opengate_ehe_guard_20261113.py")} '
                 f'--record {q(root+"/resources/imaging_")}"$index4".json --allocation "$allocation" --seconds 250000 -- '
                 f'{q(ENV+"/venv/bin/python")} {q(rel+"/opengate_ehe_transport_20261113.py")} '
                 f'run --inputs {q(root+"/inputs")} --kind imaging --index "$index" --output {q(root+"/workers/imaging/")}"$index4"\n')
        flags += [f'--array=0-{n-1}%80']
    elif stage == 'collect':
        body += f'bash {q(rel+"/cpu_python.sh")} {q(rel+"/collect.py")} --root {q(root)}\n'
    else: raise ValueError(stage)
    if dependency: flags += ['--dependency=afterok:' + str(dependency)]
    flags += ['--job-name=og_HE_DR_' + route + '_' + stage, '--output=' + root + '/logs/' + stage + '_%A_%a.log']
    script = DATA / route / (stage + '.sh'); script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(body, encoding='ascii', newline='\n')
    remote = root + '/scripts/' + read(REPORT/'freeze.json')['release']
    upload_files('maty', remote, {script.name: script})
    plan = dict(root=root, stage=stage, route=route, flags=flags, script=remote+'/'+script.name,
                script_sha256=sha(script), release=read(REPORT/'freeze.json')['release'], created_utc=now())
    save(REPORT / route / (stage + '_submission_intent.json'), plan)
    source = '''import pathlib,json,sys,fcntl,hashlib,subprocess,time,os
a=json.loads(sys.argv[1]);r=pathlib.Path(a['root']);lock=(r/'control'/('submit_'+a['stage']+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=r/'control'/(a['stage']+'_job.json');intent=r/'control'/(a['stage']+'_intent.json')
def save(p,d):
 t=p.with_suffix('.writing')
 with t.open('w') as h:json.dump(d,h);h.flush();os.fsync(h.fileno())
 t.replace(p)
if p.exists():
 result=json.loads(p.read_text());assert result['script_sha256']==a['script_sha256']
else:
 if intent.exists():raise ValueError('Unresolved intent; inspect scheduler, never blind resubmit')
 assert hashlib.sha256(pathlib.Path(a['script']).read_bytes()).hexdigest()==a['script_sha256']
 save(intent,a)
 x=subprocess.run(['sbatch','--parsable']+a['flags']+[a['script']],capture_output=True,text=True,timeout=60)
 if x.returncode:raise RuntimeError(x.stderr)
 number=x.stdout.strip().split(';')[0]
 if not number.isdigit():raise ValueError('Uncertain sbatch')
 result=dict(a,job=int(number),submitted_epoch=time.time());save(p,result)
print('CAMPAIGN_JSON '+json.dumps(result))'''
    result = python_json('maty','python3',source,plan,timeout=100,readonly=False)
    save(receipt,result); return result


def submit_all():
    result = {}
    for route, spec in ROUTES.items():
        dependency = None if spec['dose'] == '5e9' else read(REPORT / f'5e9_{spec["energy"]}' / 'collect_job.json')['job']
        imaging = submit(route, 'imaging', dependency)
        collection = submit(route, 'collect', imaging['job'])
        result[route] = dict(imaging=imaging['job'], collect=collection['job'], dependency=dependency)
        save(REPORT / 'submitted.json', result)
    save(REPORT / 'submitted.json', result); return result


def status():
    from campaign_20261102_io import python_json
    ids = {}
    for route in ROUTES:
        ps = {s: REPORT/route/(s+'_job.json') for s in ('imaging','collect')}
        if all(p.exists() for p in ps.values()):
            ids[route] = {s:read(p)['job'] for s,p in ps.items()}
    source = '''import json,sys,subprocess,pathlib,collections
a=json.loads(sys.argv[1]);out={}
for route,jobs in a['jobs'].items():
 root=pathlib.Path(a['root'])/route
 raw=subprocess.check_output(['sacct','-n','-P','-S','2026-11-16','-j',str(jobs['imaging'])+','+str(jobs['collect']),'--format=JobID,State,ExitCode,Elapsed'],text=True,timeout=45)
 lines=[l.split('|') for l in raw.splitlines() if '|' in l]
 states=collections.Counter(x[1] for x in lines if '.' not in x[0])
 errors=[x for x in lines if x[1] in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL') or (x[2]!='0:0' and x[1]=='COMPLETED')]
 complete=list((root/'workers/imaging').glob('*/complete.json'))
 collected=(root/'collected/complete.json').exists()
 out[route]=dict(states=dict(states),complete_workers=len(complete),collected=collected,errors=errors,accounting=raw,
  fully_exited_successfully=bool(lines) and all(x[1]=='COMPLETED' and x[2]=='0:0' for x in lines) and collected)
print('CAMPAIGN_JSON '+json.dumps(out))'''
    result = python_json('maty','python3',source,dict(root=CPU,jobs=ids),timeout=240,readonly=True)
    save(REPORT / 'status_latest.json', dict(created_utc=now(), routes=result))
    return result


def fetch(route, kind):
    from campaign_20261102_io import execute
    from ehe_5e9_workflow import q
    import opengate_ehe_delivery_20261113 as delivery
    receipt = REPORT / route / (kind + '_strict_fetch.json')
    if receipt.exists(): return read(receipt)
    raw = execute('maty', 'python3 ' + q(remote_release()+'/archive.py')+' --root '+q(CPU+'/'+route)+' --kind '+kind, timeout=1800)
    record = json.loads(raw.rsplit('CAMPAIGN_JSON ',1)[1]); save(REPORT/route/(kind+'_bundle.json'),record)
    delivery.DATA, delivery.REPORT = DATA/route, REPORT/route
    archive = delivery.download('maty', record)
    dest = DATA/route/('delivery' if kind=='bridge' else 'transport_delivery')
    manifest = delivery.extract(archive,record,dest)
    result = dict(passed=True, all_files_SHA_verified=True, destination=str(dest),
                  files=manifest['files'], authority=manifest['authority'], archive=record, created_utc=now())
    save(receipt,result); return result


def reconstruct(route):
    import psutil
    root = DATA/route/'delivery'; payload = DATA/'payload'; report = REPORT/route
    if (report/'local_exit.json').exists():
        prior=read(report/'local_exit.json')
        if prior.get('passed'): return prior
        raise RuntimeError('Prior failed reconstruction requires bounded repair')
    proof=read(report/'bridge_strict_fetch.json');assert proof['passed'];verify(root,proof['files'])
    assert proof['authority']['actual_primary_gamma_total']==ROUTES[route]['total']
    if (root/'reconstruction').exists(): raise FileExistsError('Do not overwrite partial/successful scientific output')
    exe=DATA/route/'local_execution';exe.mkdir(exist_ok=False)
    for name in ('opengate_ehe_common_20261113.py','opengate_ehe_projector_20261113.py',
                 'opengate_ehe_recon_cpu_20261113.py','opengate_he_geometry_20261114.py'):
        shutil.copy2(payload/name,exe/name)
    cap=min(8*(1<<30),int(.8*psutil.virtual_memory().available))
    if cap<4*(1<<30) or shutil.disk_usage(root).free<24*(1<<30):raise MemoryError('Local 20% memory/disk margin')
    env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',OPENGATE_LOCAL_HOST_CAP_BYTES=str(cap))
    cmd=[str(LOCAL),'-u',str(exe/'opengate_ehe_recon_cpu_20261113.py'),'--root',str(root),'--inputs',str(root/'inputs')]
    save(report/'local_execution_freeze.json',dict(files=files(exe),command=cmd,threads=8,cap_bytes=cap))
    start=time.monotonic();peak=0
    with (exe/'stdout.log').open('xb') as out,(exe/'stderr.log').open('xb') as err:
        child=subprocess.Popen(cmd,env=env,stdout=out,stderr=err);process=psutil.Process(child.pid)
        reg=dict(pid=child.pid,create_time=process.create_time(),cmdline=process.cmdline(),status='RUNNING',started_utc=now())
        save(report/'local_registration.json',reg)
        try:
            while child.poll() is None:
                peak=max(peak,process.memory_info().rss)
                if peak>.8*cap or psutil.virtual_memory().available<.2*cap:raise MemoryError('Actual local host margin')
                if time.monotonic()-start>86400:raise TimeoutError('Finite local reconstruction timeout')
                time.sleep(1)
            if child.returncode:raise RuntimeError('Scientific reconstruction exit '+str(child.returncode))
        except BaseException:
            if child.poll() is None:child.terminate();child.wait(timeout=30)
            save(report/'local_exit.json',dict(reg,passed=False,returncode=child.returncode,ended_utc=now()));raise
    assert read(root/'reconstruction_complete.json')['passed']
    result=dict(reg,passed=True,status='COMPLETED',returncode=0,elapsed_seconds=time.monotonic()-start,
                peak_rss_bytes=peak,ended_utc=now())
    save(report/'local_exit.json',result);return result


def finalize(route):
    root=DATA/route/'delivery';energy=str(ROUTES[route]['energy']);target=root/'reconstruction'/energy
    assert read(REPORT/route/'local_exit.json')['passed']
    complete=read(target/'complete.json');assert complete['passed'] and complete['checkpoints']==200
    verify(target,complete['files'])
    with np.load(target/'sensitivity.npz') as f: fraction=f['fraction_xyz'].copy();s=f['sensitivity_xyz'].copy()
    assert np.isfinite(s).all() and np.all(s[fraction>0]>0)
    with np.load(target/'fixed_background.npz') as f:assert np.count_nonzero(f['background'])==0
    for i in range(1,201):
        with np.load(target/f'iteration_{i:04d}.npz') as f:a=f['density_xyz']
        assert np.isfinite(a).all() and a.min()>=0 and np.count_nonzero(a[fraction==0])==0
    with np.load(target/'final.npz') as f:np.testing.assert_array_equal(a,f['density_xyz'])
    history=read(target/'history.json');assert [x['iteration'] for x in history]==list(range(1,201))
    assert read(REPORT/route/'transport_strict_fetch.json')['passed']
    result=dict(passed=True,checkpoints=200,energy_keV=int(energy),files=complete['files'],
                actual_primary_gamma_total=ROUTES[route]['total'],source_release=read(REPORT/'freeze.json')['release'],
                response_kernel_sha256=sha(root/'collected/response_kernels.npz'),zero_fixed_background=True,
                numerical_checks=complete['numerical_checks'],physical_calibration_passed=False,completed_utc=now())
    save(REPORT/route/'strict_delivery.json',result);return dict(route=route,passed=True,checkpoints=200)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','deploy','submit','status','fetch','reconstruct','finalize'])
    p.add_argument('--route',choices=list(ROUTES));p.add_argument('--kind',choices=['bridge','transport'],default='bridge')
    a=p.parse_args()
    if a.action=='prepare':r=prepare()
    elif a.action=='deploy':r=deploy()
    elif a.action=='submit':r=submit_all()
    elif a.action=='status':
        r=status();r={k:{n:v for n,v in x.items() if n!='accounting'} for k,x in r.items()}
    elif a.action=='fetch':r=fetch(a.route,a.kind);r={k:v for k,v in r.items() if k not in ('files','authority')}
    elif a.action=='reconstruct':r=reconstruct(a.route)
    else:r=finalize(a.route)
    print(json.dumps(r,indent=2,ensure_ascii=False))
