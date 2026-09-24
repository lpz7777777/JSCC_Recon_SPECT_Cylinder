"""Read FOV120 Geant4 progress on the maty supercomputer."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

DEFAULT_ROOT='/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/FOV120_20260924'
CODE='''
import collections,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]); generated=root/'experiments/FOV120/generated'
report={'root':str(root),'smoke':[],'workers':{},'scheduler':[]}
for folder in ('smoke','workers'):
 records=[]
 for path in sorted((generated/'Simulation'/folder).glob('*/worker.json')):
  try:records.append(json.loads(path.read_text()))
  except json.JSONDecodeError:continue
 if folder=='smoke':
  report['smoke']=[{k:r.get(k) for k in ('index','dataset','status','primary_counts','elapsed_seconds','photons_per_second')} for r in records]
 else:
  report['workers']={'finished_records':len(records),'status_counts':dict(collections.Counter(r['status'] for r in records)),
   'completed_primary_photons':sum(sum(r.get('primary_counts',[])) for r in records if r['status']=='complete')}
queue=subprocess.run(['squeue','-u','maty','-h','-o','%i|%j|%T|%M|%R'],text=True,capture_output=True)
report['scheduler']=[line for line in queue.stdout.splitlines() if '|FOV120_' in line]
for suffix in ('out','err'):
 paths=sorted(generated.glob('build_smoke.*.'+suffix),key=lambda p:p.stat().st_mtime)
 if paths:
  path=paths[-1]
  report['latest_build_'+suffix]={'file':path.name,'tail':path.read_text(errors='replace')[-4000:]}
print(json.dumps(report))
'''

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='maty@192.168.11.1')
    parser.add_argument('--root',default=DEFAULT_ROOT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',args.host,
        shlex.join(['python3','-c',CODE,args.root])],capture_output=True,text=True,check=True)
    report=json.loads(result.stdout)
    report['captured_at_utc']=datetime.now(timezone.utc).isoformat()
    text=json.dumps(report,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding='utf-8')
    print(text)
