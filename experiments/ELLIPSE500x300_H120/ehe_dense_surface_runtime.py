"""Formal-only dense PE -> dependent Scatter -> two complete Factors. Python 3.9+."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from ehe_common import allocation, array_write, bounded_process, digest, hashes, read, resources, verify_files, write
from ehe_gpu_pipeline import stencil

SHAPE=(2312,10,85,85)
RAW_BYTES=int(np.prod(SHAPE))*4


def positive(a):
    if np.any(~np.isfinite(a)) or np.any(a<0):raise ValueError('Nonfinite/negative response')


def sum_original(pe,scatter):
    """Frozen float32 addition order, not double accumulation or rescaling."""
    a=np.asarray(scatter,dtype='<f4').copy();a+=np.asarray(pe,dtype='<f4');positive(a);return a


def atomic_array(path,a):
    if path.exists() or path.with_name(path.name+'.writing').exists():raise FileExistsError(str(path))
    expected=hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
    array_write(path,a)
    if digest(path)!=expected:raise ValueError('Sequential output SHA mismatch')
    return expected


def inputs(release):
    contract=read(release/'contract.json');verify_files(release,contract['files'])
    parent=Path(contract['parent_release']);verify_files(parent,contract['binary_files'])
    if contract['energies_keV']!=[218,440] or contract['face_subdivisions']!=256:raise ValueError('Scope changed')
    return contract,parent


def gpu_identity():
    import torch
    if torch.cuda.device_count()!=1:raise ValueError('Exactly one allocated CUDA GPU per formal array element required')
    torch.cuda.init();prop=torch.cuda.get_device_properties(0)
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
    ids=[line.split(',')[1].strip() for line in text.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
    if not ids and hasattr(prop,'uuid'):ids=[str(prop.uuid)]
    if len(set(ids))!=1:raise ValueError('Cannot bind actual allocated GPU UUID')
    return {'name':prop.name,'uuid':ids[0],'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'total_bytes':prop.total_memory}


def parameters(release,energy,folder,layers=10,shift=-45):
    folder.mkdir(parents=True,exist_ok=False)
    for p in (release/'params'/('A'+str(energy))).glob('Params_*.dat'):shutil.copy2(p,folder/p.name)
    im=np.fromfile(folder/'Params_Image.dat','<f4');im[2]=layers;im[10]=shift;im.tofile(folder/'Params_Image.dat')


def pe_args(binary,n):
    return [str(binary),'--cuda','0','--face-subdiv',str(n),'--rows-per-chunk','4','--samples-per-launch','32',
        '--output-unwindowed','PE_unwindowed.sysmat','--output-windowed','PE_windowed.sysmat',
        '--manifest','PE_manifest.json','--progress','PE_progress.json','--log','PE_progress.tsv']


def embedded_checks(release,parent,folder,energy,alloc,contract):
    """Same formal job, real GPU scalar outputs against the independent CPU reference."""
    result=[]
    for k,case in enumerate(contract['density_selection']['results']):
        if case['energy_keV']!=energy:continue
        check=folder/('point_'+str(k));parameters(release,energy,check,layers=1,shift=case['source_mm'][2])
        im=np.fromfile(check/'Params_Image.dat','<f4');im[:3]=1;im[8:11]=case['source_mm'];im.tofile(check/'Params_Image.dat')
        argv=pe_args(parent/'bin/PEGen_V4_Production',256)+['--detector-start',str(case['detector']-1),'--detector-count','1']
        use=bounded_process(argv,check,300,check/'pe.log',alloc,gpu=True)
        a=np.fromfile(check/'PE_windowed.sysmat','<f4');assert a.shape==(1,)
        ref=case['PE256'];error=abs(float(a[0])-ref)
        if error>max(1e-35,.002*abs(ref)):raise ValueError('Dense actual GPU vs independent CPU point mismatch: '+str(case))
        result.append({'case':case,'actual_GPU_PE256':float(a[0]),'relative_error':error/max(abs(ref),1e-300),'resource':use,'files':hashes(check)})
    write(folder/'point_checks.json',{'passed':True,'full_matrix_physical_validation':False,'cases':result})


def tile(release,output,index):
    contract,parent=inputs(release)
    if not 0<=index<8:raise ValueError('Only eight formal blocks allowed')
    energy=(218,440)[index//4];slab=index%4
    alloc=allocation();gpu=gpu_identity();output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output).free<contract['minimum_available_storage_bytes']:raise OSError('Persistent storage headroom no longer sufficient')
    folder=output/('A'+str(energy))/('slab_'+str(slab))
    if (folder/'receipt.json').exists():
        old=read(folder/'receipt.json');verify_files(folder,old['files'])
        if old['contract_sha256']!=digest(release/'contract.json'):raise ValueError('Completed block identity changed')
        print('SUCCESSFUL_BLOCK_REUSED_READ_ONLY',index,flush=True)
        maybe_convert(release,output,alloc,contract);return
    if folder.exists():raise FileExistsError('Incomplete block preserved; no blind restart: '+str(folder))
    parameters(release,energy,folder,shift=-45+30*slab)
    write(folder/'allocation.json',alloc);write(folder/'gpu.json',gpu)
    checks=folder/'embedded_point_checks';checks.mkdir()
    embedded_checks(release,parent,checks,energy,alloc,contract)
    pe=bounded_process(pe_args(parent/'bin/PEGen_V4_Production',256),folder,contract['PE_hard_limit_seconds'],folder/'pe.log',alloc,gpu=True)
    meta=read(folder/'PE_manifest.json')
    if meta['face_subdivisions']!=256 or meta['detector_count']!=2312 or meta['voxel_count']!=10*85*85:raise ValueError('Incomplete/wrong dense PE manifest')
    scatter_env={k:v for k,v in os.environ.items() if not k.startswith(('SCATTER_','DETECTOR_LOCAL_SCATTER_','COLLIMATOR_SCATTER_'))}
    write(folder/'controlled_scatter_environment.json',{'inherited_scatter_overrides_removed':True,'configuration':'unchanged frozen executable defaults; complete detector range, local and collimator terms included'})
    scatter=bounded_process([str(parent/'bin/ScatterGen_CircularHole'),'-PE','PE_unwindowed.sysmat','-cuda','0'],folder,3600,folder/'scatter.log',alloc,env=scatter_env,gpu=True)
    candidates=list(folder.glob('Scatter_SysMat_shift_*.sysmat'))
    if len(candidates)!=1:raise ValueError('One complete dependent Scatter required')
    for p in [folder/'PE_unwindowed.sysmat',folder/'PE_windowed.sysmat',candidates[0]]:
        if p.stat().st_size!=RAW_BYTES:raise ValueError('Wrong complete block byte length')
    a=np.fromfile(folder/'PE_windowed.sysmat','<f4');b=np.fromfile(candidates[0],'<f4')
    total=sum_original(a,b);del a,b
    atomic_array(folder/'response.sysmat',total);del total
    usage=resources(alloc)
    files={p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file()}
    write(folder/'receipt.json',{'passed':True,'energy_keV':energy,'slab':slab,'shape':list(SHAPE),'face_subdivisions':256,
        'surface_samples_per_visible_face':65536,'contract_sha256':digest(release/'contract.json'),'files':files,
        'PE_resource':pe,'dependent_Scatter_resource':scatter,'postprocess_resource':usage,'gpu':gpu,
        'scatter_input':'this block new dense PE_unwindowed, never the old PE','physical_model_passed':False})
    print('DENSE_FULL_BLOCK_COMPLETE',energy,slab,flush=True)
    maybe_convert(release,output,alloc,contract)


def convert_energy(release,output,energy,alloc):
    name='A'+str(energy);target=output/'factors'/name
    if (target/'factor_manifest.json').exists():
        old=read(target/'factor_manifest.json');verify_files(target,old['files']);return old
    if target.exists():raise FileExistsError('Partial Factors retained; diagnose before bounded recovery')
    target.mkdir(parents=True)
    with np.load(release/'whole_geometry.npz') as g:
        coords=g['coordinates_mm'];vol=g['cell_volume_mm3'];active=g['active_indices'];inverse=g['inverse_rotation']
    assert coords.shape==(132040,3) and len(active)==78920 and np.all(vol>0)
    cart=np.empty((2312,40,85,85),dtype='<f4');polar=np.empty((132040,2312),dtype='<f4')
    ix,iy,tx,ty=stencil(coords[:3301,0],coords[:3301,1]);source={}
    for slab in range(4):
        folder=output/name/('slab_'+str(slab));receipt=read(folder/'receipt.json');verify_files(folder,receipt['files'])
        if not receipt['passed'] or receipt['face_subdivisions']!=256:raise ValueError('Wrong block receipt')
        source[str(slab)]=digest(folder/'receipt.json');raw=np.fromfile(folder/'response.sysmat','<f4').reshape(SHAPE);positive(raw)
        cart[:,slab*10:(slab+1)*10]=raw
        for k in range(10):
            z=slab*10+k
            for start in range(0,2312,64):
                layer=raw[start:start+64,k]
                top=layer[:,iy,ix]*(1-tx)+layer[:,iy,ix+1]*tx
                bot=layer[:,iy+1,ix]*(1-tx)+layer[:,iy+1,ix+1]*tx
                polar[z*3301:(z+1)*3301,start:start+64]=((top*(1-ty)+bot*ty).T*vol[z*3301:(z+1)*3301,None])
        del raw;resources(alloc)
    positive(cart);positive(polar)
    full=polar.sum(axis=1,dtype=np.float64);s=np.zeros(len(active),dtype=np.float64)
    for view in range(20):s+=full[inverse[active,view]]/20
    if np.any(~np.isfinite(s)) or np.any(s<=0):raise ValueError('New matched S invalid')
    # Complete active matrix in each of the 20 views, independent bilinear/transpose scalar identity.
    rng=np.random.default_rng(20261029+energy);x=rng.random(len(active));y=rng.random(2312);errors=[]
    for view in range(20):
        b=polar[inverse[active,view]].astype(np.float64)
        lhs=float(np.dot(x,b@y));rhs=float(np.dot(x@b,y));errors.append(abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-300));del b
    if max(errors)>1e-10:raise ValueError('All-view full-input adjoint closure failed')
    for p,a in [(target/'SysMat_cartesian',cart),(target/'SysMat_polar',polar),(target/'S_full.float64',full),(target/'S_active.float64',s)]:atomic_array(p,a)
    del cart,polar
    shutil.copy2(release/'whole_geometry.npz',target/'whole_geometry.npz')
    for p in (release/'params'/name).glob('Params_*.dat'):shutil.copy2(p,target/p.name)
    result={'passed':True,'response':name,'bins':2312,'cartesian_shape':[2312,40,85,85],'full_points':132040,'active_points':78920,
        'face_subdivisions':256,'surface_samples_per_visible_face':65536,'views':20,
        'density_equation':'B=A diag(original cell volume mm3), exactly once; S=sum_views(B^T 1)/20',
        'all20_complete_active_view_adjoint_relative_errors':errors,'source_receipt_SHA':source,'resource':resources(alloc),
        'files':hashes(target),'physical_calibration_passed':False,'known_scatter_approximations_retained':True}
    write(target/'factor_manifest.json',result);print('DENSE_FACTORS_COMPLETE',name,flush=True);return result


def maybe_convert(release,output,alloc,contract):
    members=[output/('A'+str(e))/('slab_'+str(s))/'receipt.json' for e in (218,440) for s in range(4)]
    if not all(p.is_file() for p in members):return
    claim=output/'conversion_claim'
    try:claim.mkdir()
    except FileExistsError:return
    write(claim/'owner.json',{'SLURM_JOB_ID':os.environ.get('SLURM_JOB_ID'),'array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID')})
    try:
        factors={str(e):convert_energy(release,output,e,alloc) for e in (218,440)}
        write(output/'completion.json',{'passed':True,'study':contract['study'],'contract_sha256':digest(release/'contract.json'),
            'factors':{str(e):digest(output/'factors'/('A'+str(e))/'factor_manifest.json') for e in (218,440)},
            'new_Geant4_transport':False,'new_reconstruction':False,'physical_calibration_passed':False,
            'array_job_id':os.environ.get('SLURM_ARRAY_JOB_ID'),'completed_epoch':time.time()})
    except BaseException as error:
        write(claim/'failure.json',{'type':type(error).__name__,'message':str(error),'successful_blocks_preserved':True});raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--index',type=int,required=True)
    a=p.parse_args();tile(a.release,a.output,a.index)


if __name__=='__main__':main()
