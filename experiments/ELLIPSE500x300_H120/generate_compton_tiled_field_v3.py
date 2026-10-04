"""Resumable physical A tiles; no S2 or reconstruction permission.

All 11520 physical rows are calculated. Only verified selected combined rows
are persisted. A failed/incomplete tile is preserved and never overwritten.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import threading
import subprocess
import os
import numpy as np
from generate_compton_a_guard import digest,write,axes,run_one,ENGINE_REL,SOURCE_RUN,NDET,PE_HASH,SCATTER_HASH
from generate_compton_a_tile_pilot_v3 import extract_selected,remove_verified_raw,KINDS,PARENT_KINDS
from generate_compton_a_column_v3 import comparison
from plan_compton_a_tiles_v3 import tile_spec


class ResourceSampler:
    def __init__(self,physical_gpu):
        self.physical_gpu=physical_gpu;self.stop=threading.Event();self.samples=[];self.errors=[]
        self.thread=threading.Thread(target=self.loop,daemon=True)
    def loop(self):
        while not self.stop.is_set():
            try:
                table=subprocess.run(['ps','-eo','pid=,ppid=,rss='],capture_output=True,text=True,check=True,timeout=8)
                processes=[tuple(map(int,line.split())) for line in table.stdout.splitlines() if line.strip()]
                tree={os.getpid()}
                while True:
                    children={pid for pid,ppid,rss in processes if ppid in tree}
                    if children.issubset(tree):break
                    tree.update(children)
                rss=sum(kib*1024 for pid,ppid,kib in processes if pid in tree)
                ram=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemTotal:'))
                result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits','-i',str(self.physical_gpu)],capture_output=True,text=True,check=True,timeout=8)
                used,total,util=map(int,result.stdout.strip().split(','))
                self.samples.append(dict(rss_bytes=rss,ram_total_bytes=ram,
                    gpu_used_mib=used,gpu_total_mib=total,gpu_utilization=util))
            except Exception as e:self.errors.append(type(e).__name__+': '+str(e))
            self.stop.wait(2)
    def finish(self):
        self.stop.set();self.thread.join(10)
        if not self.samples or self.errors:raise ValueError('Actual resource sampling incomplete: '+str(self.errors))
        gpu=max(x['gpu_used_mib']/x['gpu_total_mib'] for x in self.samples)
        ram=max(x['rss_bytes']/x['ram_total_bytes'] for x in self.samples)
        return dict(samples=len(self.samples),maximum_gpu_used_fraction=gpu,maximum_process_tree_rss_fraction=ram,
            maximum_process_tree_rss_bytes=max(x['rss_bytes'] for x in self.samples),
            ram_basis='65114 physical server RAM; not a Slurm grant',passed=gpu<=.8 and ram<=.8)


def specs_for_plan(plan,phase):
    if phase=='pilot':return plan['pilot_specs']
    if phase!='full':raise ValueError('Unknown tiled field phase')
    return [tile_spec(x,y,z) for x,y in plan['xy_tile_indices'] for z in range(10)]


def verify_resume(folder,spec,plan_sha,selected):
    path=folder/'compact_receipt.json'
    if not path.exists():raise ValueError('Incomplete tile preserved; explicit repair required: '+str(folder))
    r=json.loads(path.read_text());compact=folder/'A_selected.float32'
    mapping=__import__('hashlib').sha256(np.asarray(selected,dtype='<i8').tobytes()).hexdigest()
    if (r['status']!='PHYSICAL_TILE_STORED_ACCURACY_HOLD' or r['spec']!=spec
            or r['plan_sha256']!=plan_sha or r['compact']['raw_selected_mapping_sha256']!=mapping
            or compact.stat().st_size!=r['compact']['bytes'] or digest(compact)!=r['compact']['sha256']
            or not r['compact']['all_selected_rows_bitwise_verified']
            or r['pe_binary_sha256']!=PE_HASH or r['scatter_binary_sha256']!=SCATTER_HASH):
        raise ValueError('Existing tile identity/integrity differs; do not overwrite')
    return r


def original_anchors(folder,receipt,source):
    original_axes=(np.arange(85)*6-252,np.arange(85)*6-252,np.arange(40)*3-58.5)
    current=axes(receipt['spec']);ii=[];jj=[]
    for x,y in zip(current,original_axes):
        shared,left,right=np.intersect1d(x,y,return_indices=True)
        if not len(shared):raise ValueError('Required tile has no original physical anchor')
        ii.append(left);jj.append(right)
    ix,iy,iz=ii;jx,jy,jz=jj;checks={}
    for kind,parent_kind in zip(KINDS,PARENT_KINDS):
        name=next(n for n in receipt['matrices'] if n.startswith(kind))
        values=np.memmap(folder/name,mode='r',dtype='<f4',shape=(NDET,17,17,17))
        suffix='_v4' if parent_kind.startswith('PE_') else ''
        old=np.memmap(source/f'{parent_kind}_shift_0.000000_0.000000_0.000000{suffix}.sysmat',
            mode='r',dtype='<f4',shape=(NDET,40,85,85))
        check=comparison(np.array(values[:,iz[:,None,None],iy[None,:,None],ix[None,None,:]],copy=True),
            np.array(old[:,jz[:,None,None],jy[None,:,None],jx[None,None,:]],copy=True))
        if not check['passed'] or (kind.startswith('pe') and not check['bitwise_equal']):
            raise ValueError('Original physical anchors failed; preserve raw outputs')
        checks[kind]=check
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','plan','field','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--phase',choices=('pilot','full'),required=True)
    p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,required=True)
    p.add_argument('--cuda',type=int,default=0);p.add_argument('--max-new-tiles',type=int,default=3)
    p.add_argument('--physical-gpu',type=int,required=True)
    p.add_argument('--run-id',default='',help='Immutable invocation label; separate progress/completion receipts')
    p.add_argument('--stop-after-seconds',type=float,default=0,help='Stop between tiles; do not interrupt a physical matrix')
    a=p.parse_args();started=time.monotonic()
    if not 0<=a.shard<a.shards or a.max_new_tiles<1:raise ValueError('Invalid bounded shard')
    if a.run_id and not all(x.isalnum() or x in '_-' for x in a.run_id):raise ValueError('Invalid run receipt label')
    label=a.run_id or a.phase
    plan=json.loads(a.plan.read_text());plan_sha=digest(a.plan)
    if plan['status']!='FROZEN_TILED_PHYSICAL_PRODUCTION_PLAN_ACCURACY_HOLD':raise ValueError('Invalid production plan')
    specs=specs_for_plan(plan,a.phase)[a.shard::a.shards]
    if len(specs)!=len({s['name'] for s in specs}):raise ValueError('Duplicate tile assignment')
    fm=json.loads((a.field/'field_manifest.json').read_text());fg=a.field/'field_geometry.npz'
    if digest(fg)!=fm['field_geometry_sha256'] or fm['baseline_geometry_sha256']!=plan['geometry_sha256']:
        raise ValueError('Frozen geometry/crystal map changed')
    selected=np.load(fg)['selected_raw_detectors'];engine=a.root/ENGINE_REL;source=engine/SOURCE_RUN
    pe=engine/'PEGen_RayTracing_CircularHole/PEGen_V4_Production'
    scatter=engine/'ScatterGen_RayTracing_CircularHole/ScatterGen_CircularHole_detector_local'
    if digest(pe)!=PE_HASH or digest(scatter)!=SCATTER_HASH:raise ValueError('Physical model changed')
    original=json.loads((source/'ELLIPSE_inputs.json').read_text())
    for name,sha in original['parameters'].items():
        if digest(source/name)!=sha:raise ValueError('Physical parameters changed')
    rawdet=np.fromfile(source/'Params_Detector.dat',dtype='<f4')[1:].reshape(NDET,12)
    if len(selected)!=10496 or not np.array_equal(selected,np.flatnonzero(rawdet[:,11]==1)):
        raise ValueError('Crystal row mapping changed')
    a.output.mkdir(parents=True,exist_ok=True)
    # All shard workers use the same immutable plan; no worker overwrites it.
    stored=a.output/'production_plan.json'
    import fcntl
    with (a.output/'plan.lock').open('a') as planlock:
        fcntl.flock(planlock,fcntl.LOCK_EX)
        if stored.exists():
            if digest(stored)!=plan_sha:raise ValueError('Different production plan already registered')
        else:
            # Exact release bytes (including line endings) are the identity.
            with stored.open('xb') as f:f.write(a.plan.read_bytes())
    summaries=[];new=0;resumed=0
    sampler=ResourceSampler(a.physical_gpu);sampler.thread.start()
    for spec in specs:
        if new>=a.max_new_tiles:break
        if a.stop_after_seconds>0 and time.monotonic()-started>=a.stop_after_seconds:break
        # An OS lock prevents duplicates even if launch/shard boundaries change.
        with (a.output/(spec['name']+'.lock')).open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('Another worker owns this tile')
            folder=a.output/spec['name']
            if folder.exists():
                r=verify_resume(folder,spec,plan_sha,selected);resumed+=1
            else:
                usage=shutil.disk_usage(a.output)
                # Keep the entire remaining compact-field allowance plus 20%
                # filesystem headroom, and concurrent temporary raw allowance.
                completed=len(list(a.output.glob('tile_*/compact_receipt.json')))
                remaining=max(0,plan['total_tiles']-completed)*plan['retained_bytes_per_tile']
                required=remaining+plan['maximum_temporary_bytes']+.20*usage.total
                if usage.free<required:raise ValueError('Disk reserve insufficient; no new tile started')
                receipt=run_one(spec,a.output,source,pe,scatter,a.cuda)
                anchors=original_anchors(folder,receipt,source)
                name=next(n for n in receipt['matrices'] if n.startswith('SysMat_withScatter'))
                raw=np.memmap(folder/name,mode='r',dtype='<f4',shape=(NDET,17,17,17))
                compact=extract_selected(raw,folder/'A_selected.float32',selected);del raw
                r=dict(status='PHYSICAL_TILE_STORED_ACCURACY_HOLD',spec=spec,plan_sha256=plan_sha,
                    raw_production_receipt=receipt,compact=compact,original_common_points=anchors,
                    pe_binary_sha256=PE_HASH,scatter_binary_sha256=SCATTER_HASH,
                    original_params_sha256=original['parameters'],source_sha256=digest(Path(__file__)),
                    raw_outputs_retained=True,cleanup_scope='four new verified tile outputs only',
                    new_transport_photons=0,reconstruction_permitted=False,S2_generated=False)
                # Store verified receipt before cleanup, so interrupted cleanup
                # cannot cause the physical matrix to be recomputed/overwritten.
                write(folder/'compact_receipt.json',r);remove_verified_raw(folder,receipt)
                r['raw_outputs_retained']=False;write(folder/'compact_receipt.json',r);new+=1
        summaries.append(dict(name=spec['name'],compact_sha256=r['compact']['sha256'],
            elapsed_physical_seconds=r['raw_production_receipt']['elapsed_seconds']))
        write(a.output/f'{label}_shard{a.shard}_progress.json',dict(new_tiles=new,resumed_tiles=resumed,
            assigned_tiles=len(specs),completed_tiles=len(summaries),elapsed_seconds=time.monotonic()-started,
            latest=spec['name'],phase=a.phase,shard=a.shard,shards=a.shards,plan_sha256=plan_sha,
            source_sha256=digest(Path(__file__)),reconstruction_permitted=False,S2_generated=False))
    resources=sampler.finish()
    write(a.output/f'{label}_shard{a.shard}_complete.json',dict(status='BOUNDED_SHARD_COMPLETE_ACCURACY_HOLD',
        phase=a.phase,shard=a.shard,shards=a.shards,assigned_tiles=len(specs),new_tiles=new,resumed_tiles=resumed,
        completed_tiles=len(summaries),tiles=summaries,elapsed_seconds=time.monotonic()-started,
        plan_sha256=plan_sha,source_sha256=digest(Path(__file__)),new_transport_photons=0,
        actual_resources=resources,reconstruction_permitted=False,S2_generated=False))
    if not resources['passed']:raise ValueError('Actual GPU/physical-server RAM exceeds80%; stop before more tiles')


if __name__=='__main__':main()
