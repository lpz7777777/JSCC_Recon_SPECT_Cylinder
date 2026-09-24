"""Read the isolated remote matrix run without modifying the compute host."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from datetime import datetime, timezone

CODE = '''
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
engine=root/'Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main'
report={'root':str(root),'responses':{}}
for name in ('JSCC_218keV','JSCC_440keV','JSCC_440keV_to_218keVwin'):
 folder=engine/'runs'/(name+'_pe_v4_FOV120')
 record={}
 progress=folder/'PE_progress.json'
 if progress.exists():
  try: record['pe_progress']=json.loads(progress.read_text())
  except json.JSONDecodeError: record['pe_progress']='being updated'
 record['matrices']={p.name:p.stat().st_size for p in folder.glob('*.sysmat')}
 for filename in ('PE_console.log','Scatter_console.log'):
  path=folder/filename
  if path.exists():
   with path.open('rb') as f:
    f.seek(max(0,path.stat().st_size-3000));text=f.read().decode(errors='replace')
   record[filename]=text.splitlines()[-5:]
 report['responses'][name]=record
path=root/'matrix_pipeline.log'
report['pipeline_log_tail']=path.read_text(errors='replace')[-4000:] if path.exists() else None
path=root/'experiments/FOV120/generated/factor_conversion_status.json'
report['factor_conversion']=json.loads(path.read_text()) if path.exists() else None
print(json.dumps(report))
'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='65114_lipeize')
    parser.add_argument('--root',default='/home/lipeize/JSCC_FOV120_20260924')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    completed=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',args.host,
        shlex.join(['python3','-c',CODE,args.root])],text=True,capture_output=True,check=True)
    report=json.loads(completed.stdout)
    report['captured_at_utc']=datetime.now(timezone.utc).isoformat()
    text=json.dumps(report,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding='utf-8')
    print(text)
