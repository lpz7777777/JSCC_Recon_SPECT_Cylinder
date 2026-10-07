"""One independently seeded EHE transport; receipts close primary and window identities."""
import argparse,os,time
from pathlib import Path
import numpy as np
from ehe_common import digest,read,write,allocation,bounded_process,verify_files

FILES=['CntStat_218.csv','CntStat_440.csv']+[f'CntStat_{w}_from{p}.csv' for w in (218,440) for p in (218,440)]

def validate(folder,photons):
    counts={}
    for name in FILES:
        a=np.loadtxt(folder/name,delimiter=',',dtype=np.int64,ndmin=2)
        if a.shape!=(1,2312) or np.any(a<0):raise ValueError('Invalid independent worker counts: '+name)
        counts[name]=a[0]
    for w in (218,440):
        if not np.array_equal(counts[f'CntStat_{w}.csv'],counts[f'CntStat_{w}_from218.csv']+counts[f'CntStat_{w}_from440.csv']):raise ValueError('Tagged window closure failed')
    summary=read(folder/'TransportSummary.json')
    if summary['primary_events']!=photons or sum(summary['primary_counts'])!=photons or summary['primary_counts'][2]!=0 or summary['detector_bins']!=2312:raise ValueError('Actual primary identity failed')
    return summary,{k:int(a.sum()) for k,a in counts.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--simulation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--index',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')));p.add_argument('--pilot',action='store_true');p.add_argument('--photons',type=int,default=100000);p.add_argument('--limit',type=int,required=True)
    a=p.parse_args();r=a.release.resolve();s=a.simulation.resolve();out=a.output.resolve()
    freeze=read(r/'release_manifest.json');verify_files(r,freeze['sha256'])
    if digest(s/'jobs.json')!=read(r/'config.json')['simulation_manifest_sha256']:raise ValueError('Frozen worker registry changed')
    jobs=read(s/'jobs.json');job=jobs['jobs'][a.index]
    if job['index']!=a.index or digest(s/job['macro'])!=job['macro_sha256']:raise ValueError('Macro identity changed')
    if len(jobs['jobs'])!=200 or len({j['seed'] for j in jobs['jobs']})!=200:raise ValueError('Worker/seed closure failed')
    binary=r/'build/ehe_spect';binary_manifest=read(r/'binary_manifest.json')
    if digest(binary)!=binary_manifest['sha256']:raise ValueError('Binary changed')
    out.mkdir(parents=True,exist_ok=False);photons=a.photons if a.pilot else job['photons']
    macro=(s/job['macro']).read_text()
    if a.pilot:macro=macro.replace('/run/beamOn 25000000',f'/run/beamOn {photons}')
    (out/'source.mac').write_text(macro,encoding='ascii')
    alloc=allocation(unlimited_transport=True);write(out/'allocation.json',alloc)
    env=os.environ.copy();env['EHE_RANDOM_SEED']=str(job['seed']+1000000 if a.pilot else job['seed'])
    began=time.time()
    try:
        usage=bounded_process([str(binary),str(out/'source.mac')],out,a.limit,out/'transport.log',alloc,env)
        summary,totals=validate(out,photons)
        timing=read(out/'TransportTiming.json')
        if timing['beam_seconds']<=0 or timing['initialization_seconds']<0:raise ValueError('Actual phase timing required')
        geometry_audit=read(out/'EHE_MultiUnionAudit.json')
        if not geometry_audit['passed'] or geometry_audit['points']!=11252 or geometry_audit['holes']!=1250:raise ValueError('Actual union classification audit required')
        write(out/'receipt.json',dict(passed=True,pilot=a.pilot,index=a.index,view=job['view'],worker=job['worker'],seed=int(env['EHE_RANDOM_SEED']),
              photons=photons,primary_counts=summary['primary_counts'],counts=totals,resource=usage,phase_seconds=timing,
              binary_sha256=digest(binary),release_key=freeze['release_key'],release_manifest_sha256=digest(r/'release_manifest.json'),
              allocation_denominator_kind=alloc['denominator_kind'],registered_macro_sha256=job['macro_sha256'],actual_macro_sha256=digest(out/'source.mac'),
              files={name:digest(out/name) for name in FILES+['TransportSummary.json','TransportTiming.json','EHE_MultiUnionAudit.json','source.mac','allocation.json']},started_epoch=began))
    except BaseException as e:
        write(out/'failure.json',dict(passed=False,error=str(e),index=a.index,photons=photons));raise

if __name__=='__main__':main()
