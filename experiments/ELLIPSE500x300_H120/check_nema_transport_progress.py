"""Audit finished transport workers while other array tasks are still running."""
import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path

from deploy_nema_simulation import REMOTE, remote

CODE = r'''
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
base=Path(sys.argv[1]); manifest_path=base/'jobs.json'
manifest=json.loads(manifest_path.read_text()); jobs=manifest['jobs']
def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''): sha.update(block)
    return sha.hexdigest()
for macro in manifest['macros']:
    path=base/macro['path']
    if digest(path)!=macro['sha256'] or path.stat().st_size!=macro['bytes']:
        raise ValueError('Prepared macro changed')
verified=[]; unfinished=[]; failed=[]; totals=np.zeros(3,dtype=np.int64); times=[]; hashes=set()
for job in jobs:
    folder=base/'workers'/f"{job['index']:05d}"; record_path=folder/'worker.json'
    if not record_path.exists(): unfinished.append(job['index']); continue
    try: row=json.loads(record_path.read_text())
    except json.JSONDecodeError: unfinished.append(job['index']); continue
    if row['status']!='complete':
        (failed if row['exit_code'] else unfinished).append(job['index']); continue
    if any(row[key]!=job[key] for key in ('index','seed','view','worker','photons','macro_sha256','level','dataset')):
        raise ValueError(f'Worker provenance mismatch: {folder}')
    names={'CntStat_218.csv','CntStat_440.csv','List.csv','PrimaryCount.csv'}
    if set(row['output_sha256'])!=names: raise ValueError('Missing output hash')
    for name,expected in row['output_sha256'].items():
        if digest(folder/name)!=expected: raise ValueError(f'Output hash mismatch: {folder/name}')
    primary=np.loadtxt(folder/'PrimaryCount.csv',delimiter=',',dtype=np.int64,ndmin=2)
    if (primary.shape!=(1,3) or primary[0,2]!=0 or np.any(primary<0) or
        primary.sum()!=job['photons'] or primary[0].tolist()!=row['primary_counts']):
        raise ValueError(f'Primary closure failed: {folder}')
    for energy in (218,440):
        counts=np.loadtxt(folder/f'CntStat_{energy}.csv',delimiter=',',dtype=np.int64,ndmin=2)
        if counts.shape!=(1,10496) or np.any(counts<0): raise ValueError('Invalid detector counts')
    hashes.add((row['executable_sha256'],row['crystal_sha256']))
    verified.append(job['index']); totals+=primary[0]; times.append(row['elapsed_seconds'])
total=int(totals.sum()); expected=manifest['expected_primary_energy_fraction']['218']
fraction=float(totals[0]/total) if total else None
if total and abs(fraction-expected)>6*math.sqrt(expected*(1-expected)/total):
    raise ValueError('Primary mixture does not match frozen source')
if len(hashes)>1: raise ValueError('Workers used inconsistent executable/crystal files')
print(json.dumps({'level':manifest['level'],'jobs_sha256':digest(manifest_path),
    'stage':'partial_transport_output_audit','verified_completed_workers':len(verified),
    'verified_worker_indices':verified,'unfinished_workers':len(unfinished),
    'failed_worker_indices':failed,'verified_primary_counts_218_440_other':totals.tolist(),
    'verified_primary_total':total,'expected_primary_total':manifest['total_primary_photons'],
    'verified_views':sorted({jobs[i]['view'] for i in verified}),
    'observed_218_fraction':fraction,'expected_218_fraction':expected,
    'completed_worker_elapsed_seconds_min_max':[min(times),max(times)] if times else [],
    'worker_binary_geometry_signatures':sorted(hashes),
    'complete_transport_gate':len(verified)==len(jobs) and total==manifest['total_primary_photons']},indent=2))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level",choices=("1e9","5e9","1e10"),required=True)
    args=parser.parse_args()
    folder=f"{REMOTE}/generated/NEMA_Body_H60/Simulation_{args.level}"
    report=json.loads(remote("python3 -c "+shlex.quote(CODE)+" "+shlex.quote(folder)))
    report["checked_utc"]=datetime.now(timezone.utc).isoformat()
    out=Path(__file__).resolve().parent/f"reports/NEMA_Body_H60/{args.level}/transport_progress.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    if report["failed_worker_indices"]: raise SystemExit("Transport worker failures need investigation")


if __name__=="__main__": main()
