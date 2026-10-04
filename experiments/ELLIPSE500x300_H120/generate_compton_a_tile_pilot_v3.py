"""Two adjacent A tiles: full physical calculation, verified selected storage.

Only newly created pilot raw outputs can be removed, after public/face/extraction
checks pass. Old matrices, calibration and transport remain read-only.
No field accuracy certificate, sensitivity or reconstruction is produced.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from generate_compton_a_guard import digest,write,axes,run_one,ENGINE_REL,SOURCE_RUN,NDET,PE_HASH,SCATTER_HASH
from generate_compton_a_column_v3 import comparison

KINDS=('pe.sysmat','pe_windowed.sysmat','Scatter_SysMat','SysMat_withScatter')
PARENT_KINDS=('PE_SysMat','PE_Windowed_SysMat','Scatter_SysMat','SysMat_withScatter')


def matrix(folder,receipt,kind):
    name=next(n for n in receipt['matrices'] if n.startswith(kind))
    path=folder/name
    if digest(path)!=receipt['matrices'][name]['sha256']:raise ValueError('Raw pilot response changed')
    return np.memmap(path,mode='r',dtype='<f4',shape=(NDET,*reversed(receipt['spec']['shape'])))


def extract_selected(source,target,selected,row_chunk=64):
    shape=(len(selected),*source.shape[1:])
    out=np.memmap(target,mode='w+',dtype='<f4',shape=shape)
    for start in range(0,len(selected),row_chunk):
        values=np.array(source[selected[start:start+row_chunk]],copy=True)
        if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid compact response')
        out[start:start+len(values)]=values
    out.flush();del out
    check=np.memmap(target,mode='r',dtype='<f4',shape=shape)
    for start in range(0,len(selected),row_chunk):
        if not np.array_equal(check[start:start+row_chunk],source[selected[start:start+row_chunk]]):
            raise ValueError('Compact extraction changed a physical response')
    return dict(bytes=target.stat().st_size,sha256=digest(target),shape_selected_zyx=list(shape),
        finite_nonnegative=True,all_selected_rows_bitwise_verified=True,
        calibration_applied=False,raw_selected_mapping_sha256=__import__('hashlib').sha256(
            np.asarray(selected,dtype='<i8').tobytes()).hexdigest())


def remove_verified_raw(folder,receipt):
    root=Path(folder).resolve()
    for name,item in receipt['matrices'].items():
        path=(root/name).resolve()
        if path.parent!=root or not path.is_file() or digest(path)!=item['sha256']:
            raise ValueError('Unsafe or unverified raw-output cleanup target')
    # All checks precede the first unlink. Only the four recorded new files.
    for name in receipt['matrices']:(root/name).unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','plan','field','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cuda',type=int,default=0);a=p.parse_args();started=time.monotonic()
    plan=json.loads(a.plan.read_text());specs=plan['pilot_tiles']
    if (len(specs)!=2 or plan['status']!='GEOMETRY_AND_STORAGE_PLAN_ONLY_ACCURACY_HOLD'
            or any(s['shape']!=[17,17,17] or s['spacing']!=[.75]*3 for s in specs)):
        raise ValueError('Frozen bounded pilot plan differs')
    fm=json.loads((a.field/'field_manifest.json').read_text())
    if fm['baseline_geometry_sha256']!=plan['geometry_sha256']:raise ValueError('Plan/field geometry differs')
    fg=a.field/'field_geometry.npz'
    if digest(fg)!=fm['field_geometry_sha256']:raise ValueError('Crystal map changed')
    selected=np.load(fg)['selected_raw_detectors'];engine=a.root/ENGINE_REL;source=engine/SOURCE_RUN
    pe=engine/'PEGen_RayTracing_CircularHole/PEGen_V4_Production'
    scatter=engine/'ScatterGen_RayTracing_CircularHole/ScatterGen_CircularHole_detector_local'
    if digest(pe)!=PE_HASH or digest(scatter)!=SCATTER_HASH:raise ValueError('Original physical models differ')
    original=json.loads((source/'ELLIPSE_inputs.json').read_text())
    for name,sha in original['parameters'].items():
        if digest(source/name)!=sha:raise ValueError('Original physical parameters changed')
    rawdet=np.fromfile(source/'Params_Detector.dat',dtype='<f4')[1:].reshape(NDET,12)
    if len(selected)!=10496 or not np.array_equal(selected,np.flatnonzero(rawdet[:,11]==1)):
        raise ValueError('Selected crystal mapping differs')
    if shutil.disk_usage(a.output.parent).free<plan['four_raw_physical_11520_row_bytes_per_tile']*3+(10<<30):
        raise ValueError('Insufficient bounded pilot disk allowance')
    a.output.mkdir(exist_ok=False,parents=True)
    receipts=[]
    for spec in specs:
        receipts.append(run_one(spec,a.output,source,pe,scatter,a.cuda))
        write(a.output/'progress.json',dict(completed_physical_tiles=len(receipts),total=2))
    common={};faces={}
    for kind,parent_kind in zip(KINDS,PARENT_KINDS):
        matrices=[matrix(a.output/r['spec']['name'],r,kind) for r in receipts]
        # Full 11520 rows, not only scintillator targets, on the shared y face.
        face=comparison(np.array(matrices[1][:,:,0,:],copy=True),np.array(matrices[0][:,:,-1,:],copy=True))
        faces[kind]=face
        if not face['passed'] or (kind.startswith('pe') and not face['bitwise_equal']):
            raise ValueError('Shared physical tile face inconsistent; preserve all raw files')
        suffix='_v4' if parent_kind.startswith('PE_') else ''
        old=np.memmap(source/f'{parent_kind}_shift_0.000000_0.000000_0.000000{suffix}.sysmat',
            mode='r',dtype='<f4',shape=(NDET,40,85,85))
        reference_axes=(np.arange(85)*6-252,np.arange(85)*6-252,np.arange(40)*3-58.5)
        common[kind]={}
        for r,values in zip(receipts,matrices):
            xyz=axes(r['spec']);ii=[];jj=[]
            for current,baseline in zip(xyz,reference_axes):
                shared,left,right=np.intersect1d(current,baseline,return_indices=True)
                if not len(shared):raise ValueError('No original common anchor in pilot tile')
                ii.append(left);jj.append(right)
            ix,iy,iz=ii;jx,jy,jz=jj
            check=comparison(np.array(values[:,iz[:,None,None],iy[None,:,None],ix[None,None,:]],copy=True),
                np.array(old[:,jz[:,None,None],jy[None,:,None],jx[None,None,:]],copy=True))
            common[kind][r['spec']['name']]=check
            if not check['passed'] or (kind.startswith('pe') and not check['bitwise_equal']):
                raise ValueError('Original common points differ; preserve all raw files')
    extraction=[]
    for r in receipts:
        folder=a.output/r['spec']['name'];values=matrix(folder,r,'SysMat_withScatter')
        extraction.append(extract_selected(values,folder/'A_selected.float32',selected))
    # Only this pilot's four temporary components are removed after every gate.
    for r,x in zip(receipts,extraction):
        folder=a.output/r['spec']['name'];remove_verified_raw(folder,r)
        write(folder/'compact_receipt.json',dict(spec=r['spec'],raw_production_receipt=r,compact=x,
            original_raw_files_retained=False,cleanup_scope='four new verified pilot outputs only'))
    result=dict(status='TILE_INTERFACE_AND_COMPACTION_PASSED_ACCURACY_HOLD',
        plan_sha256=digest(a.plan),field_geometry_sha256=digest(fg),
        geometry_sha256=plan['geometry_sha256'],source_sha256=digest(Path(__file__)),
        pe_binary_sha256=PE_HASH,scatter_binary_sha256=SCATTER_HASH,
        original_params_sha256=original['parameters'],receipts=receipts,
        compact_extraction=extraction,shared_physical_face=faces,original_common_points=common,
        physical_matrix_points=2*17**3,physical_detector_rows=NDET,retained_selected_rows=len(selected),
        raw_production_bytes=sum(x['bytes'] for r in receipts for x in r['matrices'].values()),
        retained_response_bytes=sum(x['bytes'] for x in extraction),elapsed_seconds=time.monotonic()-started,
        original_data_read_only=True,calibration_refitted=False,new_transport_photons=0,
        reconstruction_permitted=False,S2_generated=False,
        interpretation='Two adjacent .75 mm response tiles only. All physical couplings calculated; '
            'compact selected combined rows are bitwise identical to raw rows. Shared PE faces are bitwise '
            'identical; Scatter/combined faces use the recorded numerical tolerance. A future tiled '
            'reader must choose one deterministic shared-vertex owner. Global interpolation accuracy '
            'and regional independent .375 mm checks remain required before production.')
    write(a.output/'tile_pilot_gate.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('receipts','original_common_points')},indent=2),flush=True)


if __name__=='__main__':main()
