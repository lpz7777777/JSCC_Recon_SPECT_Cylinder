"""Immutable PE-v4/Scatter EHE responses, own density Factors and direct-source audit."""
import argparse,math,os,shutil,time
from pathlib import Path
import numpy as np
from ehe_common import *

def stencil(x,y):
    fx=np.clip((x+252)/6,0,84);fy=np.clip((y+252)/6,0,84)
    ix=np.minimum(fx.astype(int),83);iy=np.minimum(fy.astype(int),83)
    return ix,iy,(fx-ix).astype('f4'),(fy-iy).astype('f4')

def build(release,output,pilot,limit):
    freeze=read(release/'release_manifest.json');verify_files(release,freeze['sha256'])
    alloc=allocation();output.mkdir(parents=True,exist_ok=True);write(output/'allocation.json',alloc)
    completed={};timing={}
    for response in RESPONSES:
        target=output/response;target.mkdir(exist_ok=True)
        for slab in range(1 if pilot else 4):
            folder=target/f'slab_{slab}';receipt=folder/'receipt.json'
            if receipt.exists():
                old=read(receipt);verify_files(folder,old['files']);completed[f'{response}/{slab}']=old;continue
            if folder.exists():raise ValueError('Partial response stage retained; diagnose before selecting a new release')
            folder.mkdir();shutil.copytree(release/'params'/response,folder,dirs_exist_ok=True)
            image=np.fromfile(folder/'Params_Image.dat','<f4');image[2]=1 if pilot else 10;image[10]=-58.5 if pilot else -45+30*slab
            image.tofile(folder/'Params_Image.dat')
            pe=release/'bin/PEGen_V4_Production';scatter=release/'bin/ScatterGen_CircularHole'
            binary=read(release/'gpu_binary_manifest.json');verify_files(release,binary['files'])
            use=bounded_process([str(pe),'--cuda','0','--face-subdiv','16','--rows-per-chunk','4','--samples-per-launch','32',
                '--output-unwindowed','PE_unwindowed.sysmat','--output-windowed','PE_windowed.sysmat',
                '--manifest','PE_manifest.json','--progress','PE_progress.json','--log','PE_progress.tsv'],folder,limit,folder/'pe.log',alloc,gpu=True)
            use_scatter=bounded_process([str(scatter),'-PE','PE_unwindowed.sysmat','-cuda','0'],folder,limit,folder/'scatter.log',alloc,gpu=True)
            candidates=list(folder.glob('Scatter_SysMat_shift_*.sysmat'))
            if len(candidates)!=1:raise ValueError('One exact Scatter output required')
            shape=(2312,int(image[2]),85,85);count=int(np.prod(shape));nbytes=count*4
            for file in ('PE_unwindowed.sysmat','PE_windowed.sysmat',candidates[0].name):
                if (folder/file).stat().st_size!=nbytes:raise ValueError('Complete matrix shape failed')
            # Cross-window response is Scatter-only: PE window generator does not model a 218 window for 440 photons.
            dest=np.memmap(folder/'response.sysmat','<f4',mode='w+',shape=(count,))
            raw=np.memmap(candidates[0],'<f4',mode='r',shape=(count,));pw=np.memmap(folder/'PE_windowed.sysmat','<f4',mode='r',shape=(count,))
            for start in range(0,count,1<<20):
                a=raw[start:start+(1<<20)].copy()
                if response!='C440to218':a+=pw[start:start+(1<<20)]
                if np.any(~np.isfinite(a)) or np.any(a<0):raise ValueError('Nonfinite/negative full response')
                dest[start:start+len(a)]=a
            dest.flush();del dest,raw,pw
            record=dict(passed=True,pilot=pilot,shape=shape,response=response,slab=slab,
                        pe_resource=use,scatter_resource=use_scatter,files=hashes(folder))
            write(receipt,record);completed[f'{response}/{slab}']=record
            timing[f'{response}/{slab}']=use['elapsed_seconds']+use_scatter['elapsed_seconds']
        if not pilot:convert(release,target,response)
    write(output/'response_summary.json',dict(passed=True,pilot=pilot,release_key=freeze['release_key'],stages=completed,timing_seconds=timing))

def convert(release,target,response):
    receipt=target/'factor_manifest.json'
    if receipt.exists():verify_files(target,read(receipt)['files']);return
    g=np.load(release/'whole_geometry.npz');coords=g['coordinates_mm'];vol=g['cell_volume_mm3'];active=g['active_indices'];inverse=g['inverse_rotation']
    if len(coords)!=132040 or len(active)!=78920 or np.any(~np.isin(g['ellipse_fraction'],[0,1])):raise ValueError('Exact whole basis required')
    out=np.memmap(target/'SysMat_polar','<f4',mode='w+',shape=(132040,2312))
    rawall=np.memmap(target/'SysMat_cartesian','<f4',mode='w+',shape=(2312,40,85,85))
    ix,iy,tx,ty=stencil(coords[:3301,0],coords[:3301,1])
    for slab in range(4):
        folder=target/f'slab_{slab}';old=read(folder/'receipt.json');verify_files(folder,old['files'])
        raw=np.memmap(folder/'response.sysmat','<f4',mode='r',shape=(2312,10,85,85))
        rawall[:,slab*10:(slab+1)*10]=raw
        for k in range(10):
            z=slab*10+k
            for start in range(0,2312,64):
                layer=raw[start:start+64,k]
                top=layer[:,iy,ix]*(1-tx)+layer[:,iy,ix+1]*tx
                bot=layer[:,iy+1,ix]*(1-tx)+layer[:,iy+1,ix+1]*tx
                out[z*3301:(z+1)*3301,start:start+64]=((top*(1-ty)+bot*ty).T*vol[z*3301:(z+1)*3301,None])
    out.flush();rawall.flush()
    sums=np.asarray(out.sum(axis=1,dtype=np.float64));sens=np.zeros(78920,np.float64)
    for v in range(20):sens+=sums[inverse[active,v]]/20
    if np.any(~np.isfinite(sens)) or np.any(sens<=0):raise ValueError('HOLD: activity-domain sensitivity not finite positive')
    array_write(target/'S_active.float64',sens);array_write(target/'S_full.float64',sums)
    del out,rawall
    for name in ('whole_geometry.npz',):shutil.copy2(release/name,target/name)
    for p in (release/'params'/response).glob('Params_*.dat'):shutil.copy2(p,target/p.name)
    write(receipt,dict(passed=True,response=response,bins=2312,full_points=132040,active_points=78920,
        density_equation='B=A diag(volume_mm3), full cell volume exactly once; own S=sum_views(B)/20',
        sensitivity_min=float(sens.min()),sensitivity_max=float(sens.max()),files={n:digest(target/n) for n in
        ['SysMat_cartesian','SysMat_polar','S_active.float64','S_full.float64','whole_geometry.npz']+[p.name for p in (release/'params'/response).glob('Params_*.dat')]}))

def source_grid(truth,e,view):
    """Exact voxel mass scatter to Cartesian response interpolation; no fitted gain."""
    density=truth[f'activity_{e}_zyx'];k,j,i=np.nonzero(density)
    x=truth['x_mm'][i];y=truth['y_mm'][j];angle=view*2*np.pi/20
    xr=x*np.cos(angle)+y*np.sin(angle);yr=y*np.cos(angle)-x*np.sin(angle)
    if np.any(abs(xr)>252) or np.any(abs(yr)>252):raise ValueError('Truth outside response grid')
    ix,iy,tx,ty=stencil(xr,yr);mass=density[k,j,i].astype(float)*27
    grid=np.zeros((40,85,85),np.float64)
    for dx,dy,w in ((0,0,(1-tx)*(1-ty)),(1,0,tx*(1-ty)),(0,1,(1-tx)*ty),(1,1,tx*ty)):
        np.add.at(grid,(k,iy+dy,ix+dx),mass*w)
    if not np.isclose(grid.sum(),mass.sum(),rtol=1e-7):raise ValueError('True source integral failed')
    return grid/grid.sum()

def physical(release,response_root,counts_root,output):
    import torch
    truth=np.load(release/'truth_3mm.npz');collection=read(counts_root/'collection.json')
    verify_files(counts_root,collection['files']);workers=np.load(counts_root/'worker_counts.npz')
    diagnostics=[];hold_count=0;unknown=0;alloc=allocation()
    for response,e,window in (('A218',218,218),('A440',440,440),('C440to218',440,218)):
        factor=read(response_root/response/'factor_manifest.json');verify_files(response_root/response,factor['files'])
        raw=np.memmap(response_root/response/'SysMat_cartesian','<f4',mode='r',shape=(2312,40*85*85))
        matrix=torch.as_tensor(np.array(raw),device='cuda');prediction=[];observations=[];ses=[]
        primary=workers['primary_counts'][:,0 if e==218 else 1].reshape(20,10)
        data=workers[f'CntStat_{window}_from{e}'].reshape(20,10,2312)
        for view in range(20):
            weights=torch.as_tensor(source_grid(truth,e,view).reshape(-1),dtype=torch.float32,device='cuda')
            predicted=(matrix@weights).cpu().numpy()*primary[view].sum()
            o=data[view].sum(axis=0);se=data[view].std(axis=0,ddof=1)*math.sqrt(10)
            prediction.append(predicted);observations.append(o);ses.append(se)
            # View sum has correlated detector bins: estimate its worker variance directly.
            values=[('view_total',float(predicted.sum()),float(o.sum()),float(data[view].sum(axis=1).std(ddof=1)*math.sqrt(10)))]
            for b in range(2312):values.append((f'bin_{b}',float(predicted[b]),float(o[b]),float(se[b])))
            for label,p,o,s in values:
                adequate=o>=100;rel,h=physical_gate(p,o,s,adequate);hold_count+=int(h);unknown+=int(not adequate)
                diagnostics.append(dict(response=response,view=view+1,scope=label,predicted=p,observed=o,standard_error=s,
                    relative_bias=float(rel),adequate=adequate,status='HOLD' if h else 'PASSED' if adequate else 'UNDETERMINED'))
        # Independent-worker global totals, with view-specific means removed.
        p=float(np.sum(prediction));o=float(np.sum(observations));s=math.sqrt(sum(float(data[v].sum(axis=1).var(ddof=1))*10 for v in range(20)))
        rel,h=physical_gate(p,o,s,o>=100);hold_count+=int(h)
        diagnostics.append(dict(response=response,scope='global',predicted=p,observed=o,standard_error=s,relative_bias=float(rel),adequate=o>=100,status='HOLD' if h else 'PASSED'))
        del matrix;torch.cuda.empty_cache()
    output.mkdir(parents=True,exist_ok=False)
    import csv
    with (output/'physical_audit.csv').open('w',newline='') as f:
        keys=['response','view','scope','predicted','observed','standard_error','relative_bias','adequate','status'];w=csv.DictWriter(f,keys);w.writeheader();w.writerows(diagnostics)
    write(output/'physical_gate.json',dict(passed=hold_count==0,hold_count=hold_count,undetermined_bins=unknown,
        adequate_rule='at least 100 observed counts and 10 independent workers/view',
        rule='HOLD iff adequate and absolute relative bias >10% and absolute difference >3 worker-derived SE',
        source_method='unfitted 3mm true voxel mass; bilinear point-response interpolation, clockwise source rotation',
        source_sha256=digest(release/'truth_3mm.npz'),collection_sha256=digest(counts_root/'collection.json'),resource=resources(alloc),files=hashes(output)))
    if hold_count:raise ValueError(f'Physical response HOLD: {hold_count} diagnostics; inspect CSV before any reconstruction')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','responses','physical']);p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=3600);p.add_argument('--responses',type=Path);p.add_argument('--counts',type=Path)
    a=p.parse_args()
    if a.stage=='physical':physical(a.release,a.responses,a.counts,a.output)
    else:build(a.release,a.output,a.stage=='pilot',a.limit)
