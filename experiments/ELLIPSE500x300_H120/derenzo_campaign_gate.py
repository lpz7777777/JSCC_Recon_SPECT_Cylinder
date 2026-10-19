"""Release only the current held formal job after strictly delivered validation."""
import argparse
import math
import os
from pathlib import Path
from derenzo_campaign_common import *


def limits_from_validation(proof):
    info=dataset_info(proof['dataset']);records=proof['resources']
    phase={c:max(r['phase_solve_seconds'][c] for r in records) for c in info['channels']}
    if any(not math.isfinite(x) or x<=0 for x in phase.values()):raise ValueError('Measured positive solver times required')
    preparation=max(r['prepare_seconds'] for r in records)
    once=max(r['elapsed_seconds'] for r in records)-sum(phase.values())
    scaled={c:t*(info['iterations']/10) for c,t in phase.items()}
    phase_limit=max(1800,math.ceil(max(scaled.values())*1.5+300))
    prep_limit=max(1800,math.ceil(preparation*1.5+300))
    minutes=max(120,math.ceil(((sum(scaled.values())+max(once,preparation,0))*1.5+1800)/60))
    if phase_limit>86400 or minutes>2880:raise ValueError('Measured solve exceeds bounded 48-hour plan')
    return dict(phase=phase_limit,prepare=prep_limit,minutes=minutes,total=minutes*60-180,
                measured_phase_seconds=phase,measured_prepare_seconds=preparation)


def check(authority,plan,job):
    a=read(authority);p=read(plan)
    if (a.get('passed') is not True or a.get('strict_fetch_passed') is not True or a['formal_job']!=int(job)
            or a['dataset']!=p['dataset'] or a['plan_sha256']!=digest(plan)
            or a['collection_sha256']!=p['collection_sha256'] or a['release_sha256']!=p['release_sha256']):
        raise ValueError('Strict authority for this exact formal job/monoenergy input required')
    verify_files(Path(a['validation_result']),a['validation_files_sha256'])
    return a


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--authority',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    authority=check(a.authority,a.plan,os.environ['SLURM_JOB_ID']);limits=authority['limits']
    for key in ['phase','prepare','minutes','total']:
        if not isinstance(limits[key],int) or limits[key]<=0:raise ValueError('Invalid measured execution bound')
    if limits['minutes']>2880 or limits['phase']>86400 or limits['total']!=limits['minutes']*60-180:raise ValueError('Invalid bounded execution plan')
    print('export DERENZO_PHASE_SECONDS='+str(limits['phase']))
    print('export DERENZO_PREPARE_SECONDS='+str(limits['prepare']))
    print('export DERENZO_TOTAL_SECONDS='+str(limits['total']))
    print('export DERENZO_AUTHORITY_SHA256='+digest(a.authority))
