"""Verify point workers and export compact detector-count responses (no List duplication)."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from workflow import digest, load_jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path(__file__).parent/'generated/Simulation/jobs.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();jobs=[j for j in load_jobs(a.manifest) if j['role']=='point']
    if a.output.exists() or a.output.with_suffix('.json').exists(): raise FileExistsError(a.output)
    records=[];missing=[];counts={e:[] for e in (218,440)}
    for j in jobs:
        folder=a.manifest.parent/'workers'/f"{j['index']:05d}"
        path=folder/'worker.json'
        if not path.exists(): missing.append(j['index']);continue
        r=json.loads(path.read_text())
        if r['status']!='complete': raise ValueError(f"Failed point worker: {j['index']}")
        if any(r[k]!=v for k,v in j.items()) or r['simulated_photons']!=j['photons']:
            raise ValueError('Worker/manifest mismatch')
        if sum(r['primary_counts'])!=j['photons'] or r['primary_counts'][2]!=0:
            raise ValueError('Primary count mismatch')
        if r['primary_counts'][0 if j['mono_keV']==440 else 1]!=0:
            raise ValueError('Wrong primary energy')
        for name,checksum in r['output_sha256'].items():
            if digest(folder/name)!=checksum: raise ValueError(f'Changed output: {folder/name}')
        for energy in counts:
            data=np.loadtxt(folder/f'CntStat_{energy}.csv',delimiter=',',dtype=np.int64).reshape(-1)
            if len(data)!=10496 or np.any(data<0): raise ValueError('Invalid detector counts')
            counts[energy].append(data)
        records.append(r)
    if missing and not a.allow_partial: raise ValueError(f'Missing workers: {missing}')
    if not records: raise ValueError('No complete workers')
    if len({r['seed'] for r in records})!=len(records): raise ValueError('Duplicate seeds')
    for key in ('executable_sha256','crystal_sha256'):
        if len({r[key] for r in records})!=1: raise ValueError('Mixed detector/executable')
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),expected_workers=len(jobs),
                complete_workers=len(records),missing_indices=missing,records=records,
                actual_photons=sum(sum(r['primary_counts']) for r in records),manifest_sha256=digest(a.manifest))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,counts218=np.stack(counts[218]),counts440=np.stack(counts[440]))
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__': main()
