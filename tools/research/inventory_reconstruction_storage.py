"""Read-only storage inventory for the accepted dual-energy studies."""
from pathlib import Path
import argparse, collections, datetime, json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / 'experiments/ELLIPSE500x300_H120'
OUT = H / 'reports/dual_energy_review_20261010'

def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def local():
    started = time.monotonic()
    totals = collections.defaultdict(lambda: [0, 0])
    large = []
    caches = []
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d != '.git']
        for name in files:
            p = Path(folder) / name
            if p.is_symlink(): continue
            try: size = p.stat().st_size
            except OSError: continue
            rel = p.relative_to(ROOT).as_posix()
            parts = rel.split('/')
            key = '/'.join(parts[:2])
            if rel.startswith('experiments/ELLIPSE500x300_H120/'):
                key = '/'.join(parts[:5 if parts[3] == 'NEMA_Body_H60' else 4]) if len(parts)>3 else '/'.join(parts[:3])
            totals[key][0] += size; totals[key][1] += 1
            if size >= 100_000_000: large.append({'path': rel, 'bytes': size})
            if '__pycache__' in parts or name.endswith('.pyc'): caches.append({'path': rel, 'bytes': size})
    controllers=[]
    for p in (H/'generated').glob('*/**/*registration.json'):
        try:
            d=json.loads(p.read_text(encoding='utf-8'))
            if isinstance(d,dict) and any(k in d for k in ('pid','state','status')): controllers.append({'path':p.relative_to(ROOT).as_posix(),'record':d})
        except (ValueError, OSError): pass
    for pattern in ('*/controller.json','*/bounded_advance.json'):
        for p in (H/'generated').glob(pattern):
            controllers.append({'path':p.relative_to(ROOT).as_posix(),'record':json.loads(p.read_text(encoding='utf-8'))})
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'workspace':str(ROOT),
            'elapsed_seconds':time.monotonic()-started,'groups':[{'path':k,'bytes':v[0],'files':v[1]} for k,v in sorted(totals.items(),key=lambda x:-x[1][0])],
            'large_files':sorted(large,key=lambda x:-x['bytes']),'cache_files':caches,'controllers':controllers,
            'read_only':True}
    save('local_storage_inventory.json',result)
    print(json.dumps({'top_groups':result['groups'][:25],'large_file_count':len(large),'cache_files':len(caches),'cache_bytes':sum(x['bytes'] for x in caches),'controllers':controllers},ensure_ascii=False))

def remote():
    sys.path.insert(0,str(H))
    from ehe_5e9_workflow import connection
    root='/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor'
    commands=[
        ['filesystem','df -h /data/run01/scxi717 /data/home/scxi717'],
        ['quota','quota -s'],
        ['jobs','squeue -u scxi717 -o "%.18i %.9P %.40j %.8T %.10M %.6D %R"'],
        ['run01','timeout 90s du -x -B1 --max-depth=2 /data/run01/scxi717'],
        ['project','timeout 90s du -x -B1 --max-depth=2 '+root],
        ['ellipse_generated','timeout 90s du -x -B1 --max-depth=1 '+root+'/experiments/ELLIPSE500x300_H120/generated'],
        ['home','timeout 90s du -x -B1 --max-depth=2 /data/home/scxi717'],
    ]
    results=[]
    with connection('gpu') as c:
        for label,cmd in commands:
            began=time.monotonic(); _,out,err=c.exec_command(cmd,timeout=105)
            stdout=out.read().decode(errors='replace'); stderr=err.read().decode(errors='replace'); rc=out.channel.recv_exit_status()
            row={'label':label,'command':cmd,'exit_code':rc,'elapsed_seconds':time.monotonic()-began,'stdout':stdout,'stderr':stderr}
            results.append(row); save('remote_storage_inventory.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'results':results})
            print(json.dumps(row,ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('local','remote'));args=ap.parse_args()
    (local if args.mode=='local' else remote)()
