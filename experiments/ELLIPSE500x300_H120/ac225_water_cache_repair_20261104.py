"""Bounded formal-only replacement of failed/unstarted r4 routes, preserving science."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import psutil
from ac225_corrections.io import read, write, digest, immutable
from ac225_water_workflow_20261103 import HERE, DATA, REPORT, GPU, GPU_PYTHON, submit_registered
from campaign_20261102_io import execute, upload_files, python_json as mutation_json
from campaign_reliable_io_20261104 import python_json, accounting, ReadOnlyUnavailable
from ehe_5e9_workflow import q

R = REPORT / 'cache_repair_20261104'
KEPT = (0, 2, 3, 4)
REPLACED = (1, *range(5, 16))
OLD440, OLD218 = 1772388, 1772389


def controller_absent():
    current = read(REPORT / 'one_shot_current.json')
    try:
        p = psutil.Process(current['pid'])
        if p.create_time() == current['create_time'] and p.cmdline() == current['cmdline']:
            raise RuntimeError('An identified water controller is alive; no parallel mutation')
    except psutil.NoSuchProcess:
        pass


def conditions():
    return read(DATA / 'repair_payload_20261104/contract.json')['conditions']


def route_map():
    result = {}
    cfgs = conditions()
    for energy in (440, 218):
        selected = [(name, cfg) for name, cfg in cfgs.items() if cfg['energy_keV'] == energy]
        if len(selected) != 16:
            raise ValueError('Expected sixteen conditions per energy')
        for index, (name, cfg) in enumerate(selected):
            old = energy == 440 and index in KEPT
            result[name] = dict(index=index, energy_keV=energy,
                execution='original_r4' if old else 'cache_r5',
                output=GPU + ('/formal/' if old else '/formal_cache_r5/') + name,
                input=GPU + '/inputs/' + cfg['dataset'], iterations=cfg['iterations'], save_step=cfg['save_step'])
    for name, route in result.items():
        if route['energy_keV'] == 218:
            parent = cfgs[name]['final440_from_same_branch']
            route.update(parent440=result[parent]['output'], parent440_condition=parent)
    return result


def freeze():
    marker = R / 'freeze.json'
    if marker.exists():
        f = read(marker)
        for path, expected in f['local_files'].items():
            if digest(path) != expected: raise ValueError('Frozen execution file changed')
        return f
    old = read(REPORT / 'repair_20261104/freeze.json')
    launcher = HERE / 'ac225_formal_cache_launcher_20261104.py'
    if digest(old['archive']) != old['archive_sha256']:
        raise ValueError('Original r4 archive differs')
    plan = dict(schema=1, routes=route_map(), launcher_sha256=digest(launcher),
        archive=old['GPU_release'] + '/bundle.tar.gz', archive_sha256=old['archive_sha256'],
        scientific_release_key=old['key'], original_bootstrap_sha256=old['bootstrap_sha256'],
        side_root=old['output'], cache_evidence=GPU + '/cache_evidence_r5',
        planned_compiler_cache_bytes=1 << 30, unchanged_scientific_members=old['files'],
        original_jobs_preserved=[OLD440, OLD218], retained440_indices=list(KEPT),
        replaced440_indices=list(REPLACED), replacement218_indices=list(range(16)),
        maximum_concurrent_GPU_routes=4, no_standalone_validation_jobs=True,
        authorization='2026-11-04 user: fix cache paths, resubmit failed branches and adjust218 dependencies')
    key = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]
    root = GPU + '/formal_cache_releases/' + key
    path = DATA / 'formal_cache_r5_plan.json'
    immutable(path, (json.dumps(plan, indent=2) + '\n').encode())
    f = dict(key=key, release=root, plan=str(path), plan_sha256=digest(path),
        launcher=str(launcher), launcher_sha256=digest(launcher),
        local_files={str(path):digest(path), str(launcher):digest(launcher)},
        scientific_archive_sha256=old['archive_sha256'], scientific_source_unchanged=True)
    write(marker, f)
    return f


def inventory():
    source = '''import sys,json,pathlib,subprocess
c=json.loads(sys.argv[1]);base=pathlib.Path(c['base']);out={}
for name,route in c['routes'].items():
 p=base/'formal'/name
 out[name]=dict(exists=p.exists(),checkpoint_count=len(list((p/'checkpoints').glob('*.npz'))) if p.exists() else 0,
  files=[x.name for x in p.iterdir()] if p.exists() else [],
  progress=json.loads((p/'progress.json').read_text()) if (p/'progress.json').is_file() else None)
p=subprocess.run(['squeue','-h','-r','-j','1772388,1772389','-o','%i|%T'],capture_output=True,text=True,check=True,timeout=30)
print('CAMPAIGN_JSON '+json.dumps(dict(outputs=out,queue=p.stdout)))'''
    rows = accounting('gpu', [OLD440, OLD218], GPU_PYTHON)
    result = python_json('gpu', GPU_PYTHON, source, dict(base=GPU, routes=route_map()))
    result.update(accounting=rows, checked_epoch=time.time())
    write(R / 'inventory_before_cancellation.json', result)
    return result


def cancel_unstarted():
    marker = R / 'cancelled_unstarted.json'
    if marker.exists(): return read(marker)
    controller_absent()
    intent = R / 'cancellation_intent.json'
    targets = [f'{OLD440}_{i}' for i in REPLACED if i != 1] + [f'{OLD218}_{i}' for i in range(16)]
    if not intent.exists():
        inv = inventory()
        rows = {r[0]:r[1:3] for r in inv['accounting']}
        if any(rows.get(n) != ['PENDING', '0:0'] for n in targets):
            raise ValueError('Only confirmed pending elements may be replaced')
        if rows.get(f'{OLD440}_1') != ['FAILED','1:0'] or rows.get(f'{OLD440}_1.batch') != ['FAILED','1:0']:
            raise ValueError('Known failed element or batch identity changed')
        if rows.get(f'{OLD440}_1.extern') != ['COMPLETED','0:0']:
            raise ValueError('Failed branch has not completely exited')
        for name, route in route_map().items():
            state = inv['outputs'][name]
            if route['execution'] == 'cache_r5':
                if route['energy_keV'] == 440 and route['index'] == 1:
                    if state['checkpoint_count'] or state['progress']:
                        raise ValueError('Failed branch unexpectedly contains successful science')
                elif state['exists']:
                    raise ValueError('Pending route has an existing output; preserve and inspect')
        command = 'scancel ' + ' '.join(targets)
        write(intent, dict(targets=targets, command=command, checked_epoch=time.time(),
              reason='Replace unstarted shared-home-cache execution; no running task cancelled'))
        execute('gpu', command, readonly=False)
    elif read(intent)['targets'] != targets:
        raise ValueError('Different unresolved cancellation intent')
    # Never repeat an uncertain scancel. Reconciliation is read-only.
    rows = accounting('gpu', [OLD440, OLD218], GPU_PYTHON)
    by_id = {r[0]:r[1:3] for r in rows}
    if any(n not in by_id or not by_id[n][0].startswith('CANCELLED') for n in targets):
        raise ReadOnlyUnavailable('Pending cancellation not yet fully reconciled; no submission')
    for row in rows:
        if row[0].split('.')[0] in targets and row[1] in ('RUNNING','COMPLETING','CONFIGURING','PENDING'):
            raise ValueError('Cancelled array element still has an active step')
    result = dict(targets=targets, accounting=rows, running440_retained=list(KEPT),
                  failed440_original_preserved=True, checked_epoch=time.time())
    write(marker, result)
    return result


def deploy():
    f = freeze()
    if (R / 'deployment.json').exists(): return read(R / 'deployment.json')
    linux = read(R / 'linux_checks.json')
    if not read(R / 'local_checks.json')['passed'] or not linux['passed']:
        raise ValueError('Local and actual Linux execution checks required')
    if linux['files']['ac225_formal_cache_launcher_20261104.py'] != f['launcher_sha256']:
        raise ValueError('Actual Linux cache checks used another launcher')
    upload_files('gpu', f['release'], {'launcher.py':Path(f['launcher']), 'plan.json':Path(f['plan'])})
    source = '''import json,sys,pathlib,hashlib,os
c=json.loads(sys.argv[1]);r=pathlib.Path(c['root']);b=pathlib.Path(c['base'])
for n,h in c['files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
plan=json.loads((r/'plan.json').read_text());assert hashlib.sha256(pathlib.Path(plan['archive']).read_bytes()).hexdigest()==plan['archive_sha256']
for route in plan['routes'].values():
 if route['execution']=='cache_r5':assert not pathlib.Path(route['output']).exists(),route['output']
s=os.statvfs(b);available=s.f_bavail*s.f_frsize;assert os.access(b,os.W_OK) and available>=40*(1<<30)
(b/'cache_evidence_r5').mkdir(exist_ok=True)
print('CAMPAIGN_JSON '+json.dumps(dict(passed=True,available_bytes=available,total_bytes=s.f_blocks*s.f_frsize,
 planned_output_bound_bytes=32*(1<<30),reserve_fraction=.2,original_archive_SHA_verified=True)))'''
    proof = mutation_json('gpu', GPU_PYTHON, source, dict(root=f['release'], base=GPU,
            files={'launcher.py':f['launcher_sha256'], 'plan.json':f['plan_sha256']}), readonly=False)
    write(R / 'deployment.json', proof)
    return proof


def submit():
    marker = R / 'submitted.json'
    if marker.exists(): return read(marker)
    controller_absent(); f = freeze(); deploy(); cancel_unstarted()
    # One original slot must leave before any replacement can start.
    # r8 may then increase replacement throttle while maintaining <=4 GPUs.
    common = '-p gpu_5090 --account=scxi717 --qos=gpugpu -N1 -n1 -c8 --gres=gpu:1 --time=4320 '
    records = {}
    for energy in (440,218):
        body = '#!/usr/bin/env bash\nset -euo pipefail\nsource /etc/profile.d/modules.sh\nmodule load cuda/12.9\nexport OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n'
        body += q(GPU_PYTHON) + ' -B ' + q(f['release']+'/launcher.py') + ' ' + q(f['release']+'/plan.json') + ' ' + q(f['plan_sha256']) + ' ' + str(energy) + '\n'
        if energy == 440:
            options = common + '--array=1,5-15%1 --dependency=afterany:1772388_0'
        else:
            deps = [f'{OLD440}_{i}' for i in KEPT] + [str(records['440']['job'])]
            options = common + '--array=0-15%4 --dependency=afterok:' + ':'.join(deps)
        records[str(energy)] = submit_registered('gpu','ACW_formal'+str(energy)+'_cache_r5',body,
            options,GPU+'/formal_cache_r5',GPU)
    routes = route_map()
    for route in routes.values():
        job = OLD440 if route['execution']=='original_r4' else records[str(route['energy_keV'])]['job']
        route['job_root'] = str(job) + '_' + str(route['index'])
    result = dict(jobs=records, routes=routes, execution_freeze=f['key'], plan_sha256=f['plan_sha256'],
                  maximum_concurrent_GPU_routes=4, original_failed_and_cancelled_jobs_preserved=True,
                  dependency218='successful retained440 elements AND entire replacement440 array',
                  submitted_epoch=time.time())
    write(marker,result)
    write(REPORT/'formal_effective_routes_r5.json',result)
    write(REPORT/'formal_current.json',dict(effective_routes='formal_effective_routes_r5.json',
        repair='cache_repair_20261104/submitted.json', original='formal_submitted.json'))
    return result


def effective_snapshot():
    effective = read(REPORT/'formal_effective_routes_r5.json')
    expected = {r['job_root'] for r in effective['routes'].values()}
    ids = sorted({n.split('_')[0] for n in expected})
    rows = accounting('gpu', ids, GPU_PYTHON)
    by_id = {r[0]:r[1:3] for r in rows}
    if not expected <= set(by_id):
        raise ReadOnlyUnavailable('Incomplete effective formal accounting; admission deferred')
    effective_rows = [r for r in rows if r[0].split('.')[0] in expected]
    bad = [r for r in effective_rows if r[1].split()[0] in
           ('FAILED','OUT_OF_MEMORY','TIMEOUT','CANCELLED','NODE_FAIL','BOOT_FAIL','PREEMPTED')]
    active_old = sum(by_id[f'{OLD440}_{i}'][0] not in ('COMPLETED',) for i in KEPT)
    counts = Counter(by_id[n][0] for n in expected)
    return dict(checked_epoch=time.time(), rows=effective_rows, failures=bad,
                active_original440=active_old, root_state_counts=dict(counts),
                all_completed=all(by_id[n]==['COMPLETED','0:0'] for n in expected))


def tune_throttle(state):
    if state['failures']: raise RuntimeError('New effective science failed; preserve and stop: '+str(state['failures']))
    desired = 4 - state['active_original440']
    if desired < 1: return
    job = read(R/'submitted.json')['jobs']['440']['job']
    if all(r[1:3]==['COMPLETED','0:0'] for r in state['rows'] if r[0].startswith(str(job)+'_')):
        return
    # A fresh read controls every idempotent scheduler limit adjustment.
    source = r'''import json,sys,subprocess,re
j=str(json.loads(sys.argv[1])['job']);p=subprocess.run(['scontrol','show','job','-o',j],capture_output=True,text=True,timeout=30)
matches=re.findall(r'ArrayTaskThrottle=(\d+)',p.stdout)
print('CAMPAIGN_JSON '+json.dumps(dict(returncode=p.returncode,throttles=sorted(set(map(int,matches))),text=p.stdout,error=p.stderr)))'''
    before = python_json('gpu',GPU_PYTHON,source,dict(job=job))
    if before['returncode'] != 0:
        # Finished array may already have left scontrol; accounting remains authority.
        if all(r[1:3]==['COMPLETED','0:0'] for r in state['rows'] if r[0].startswith(str(job)+'_')): return
        raise ReadOnlyUnavailable('Cannot verify replacement array throttle')
    if before['throttles'] == [desired]: return
    if before['throttles'] and max(before['throttles']) > desired:
        raise RuntimeError('Unexpected scheduler throttle larger than allowed')
    command=f'scontrol update JobId={job} ArrayTaskThrottle={desired}'
    write(R/'throttle_intent.json',dict(command=command,checked_epoch=time.time(),state=state))
    execute('gpu',command,readonly=False)
    after=python_json('gpu',GPU_PYTHON,source,dict(job=job))
    if after['throttles'] != [desired]: raise ValueError('Updated array concurrency not verified')
    write(R/f'throttle_{desired}_verified.json',dict(command=command,after=after,checked_epoch=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','deploy','submit','effective_snapshot'])
    a=p.parse_args();print(json.dumps(globals()[a.action](),indent=2))
