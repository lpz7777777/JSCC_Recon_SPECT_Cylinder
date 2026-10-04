"""Build a sampled A halo while preserving every original B/volume node.

Output layout [detector, z, xy]; density basis is removed exactly once. This
does not itself authorize R2 imaging: independent interpolation checks follow.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from geometry import grid
from generate_compton_a_guard import digest, ENGINE_REL, SOURCE_RUN, NDET
from compton_boundary_quadrature import GuardedPolarResponseField


def interpolate_cartesian(values, xy):
    p=(xy+258)/6
    if np.any(p < -1e-9) or np.any(p > 86+1e-9):raise ValueError('Halo point outside Cartesian box')
    p=np.clip(p,0,86);lo=np.minimum(np.floor(p).astype(int),85);t=p-lo
    ix,iy=lo.T;tx,ty=t.T
    return ((values[...,iy,ix]*(1-tx)+values[...,iy,ix+1]*tx)*(1-ty)+
            (values[...,iy+1,ix]*(1-tx)+values[...,iy+1,ix+1]*tx)*ty)


def combined(guard, name):
    part=guard/name;receipt=json.loads((part/'complete.json').read_text())
    files=[f for f in receipt['matrices'] if f.startswith('SysMat_withScatter')]
    if len(files)!=1:raise ValueError('Guard combined response not unique')
    path=part/files[0]
    if digest(path)!=receipt['matrices'][files[0]]['sha256']:raise ValueError('Guard data changed')
    shape=(NDET,*reversed(receipt['spec']['shape']))
    return np.memmap(path,mode='r',dtype='<f4',shape=shape)


def build(root, guard, factors, geometry, config, output):
    ready_path=guard/'guard_ready.json';ready=json.loads(ready_path.read_text())
    if ready['status']!='READY_FOR_INTERPOLATION_VALIDATION':raise ValueError('Halo incomplete')
    if ready['parts'][0]['anchor']['status']!='PASSED':raise ValueError('Common-point regression failed')
    output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(config.read_text());geo=np.load(geometry);n=cfg['points_per_layer']
    coords,cells,_=grid(cfg);original=coords[:n,:2]
    angle=np.arange(140)*2*np.pi/140;ring=258*np.column_stack((np.cos(angle),np.sin(angle)))
    xy=np.vstack((original,ring));z=np.r_[-60.,np.arange(40)*3-58.5,60.]
    field=GuardedPolarResponseField(xy,n,z)
    # All complete reference-cell quadrature locations lie inside this support.
    for c in range(n):
        radius=cells[c,1];start,end=cells[c,2:]
        angles=np.linspace(start,end,5)
        points=np.column_stack((radius*np.cos(angles),radius*np.sin(angles),np.full(5,60.)))
        for view in range(20):
            phi=view*np.pi/10;c0,s0=np.cos(phi),np.sin(phi)
            rotated=points.copy();rotated[:,0]=points[:,0]*c0+points[:,1]*s0
            rotated[:,1]=points[:,1]*c0-points[:,0]*s0
            field.cache(rotated)
    source=root/ENGINE_REL/SOURCE_RUN
    manifest=json.loads((factors/'factor_manifest.json').read_text())
    raw_path=source/'SysMat_withScatter_shift_0.000000_0.000000_0.000000.sysmat'
    if digest(raw_path)!=manifest['input_sha256']['matrix']:raise ValueError('Original A440 source differs')
    det=np.fromfile(source/'Params_Detector.dat',dtype='<f4')[1:].reshape(NDET,12)
    selected=np.flatnonzero(det[:,11]==1)
    if len(selected)!=10496:raise ValueError('Wrong selected detector map')
    scales={m['layer_mm']:m['scale'] for m in manifest['calibration']['metrics']}
    scale=np.array([scales[float(v+270)] for v in det[selected,1]])
    old=np.memmap(factors/'SysMat_polar',dtype='<f4',mode='r',shape=(132040,10496))
    raw=np.memmap(raw_path,dtype='<f4',mode='r',shape=(NDET,40,85,85))
    halos={name:combined(guard,name) for name in ('z_minus','z_plus','x_minus','x_plus','y_minus','y_plus')}
    dest=np.memmap(output/'A_field.float64',dtype='<f8',mode='w+',shape=(10496,42,len(xy)))
    start=time.monotonic()
    for offset in range(0,10496,64):
        sub=selected[offset:offset+64];size=len(sub)
        cart=np.empty((size,40,87,87),dtype=np.float64)
        cart[:,:,1:-1,1:-1]=raw[sub]
        cart[:,:,:,0]=halos['x_minus'][sub,:,:,0];cart[:,:,:,-1]=halos['x_plus'][sub,:,:,0]
        cart[:,:,0,1:-1]=halos['y_minus'][sub,:,0,:];cart[:,:,-1,1:-1]=halos['y_plus'][sub,:,0,:]
        dest[offset:offset+size,1:-1,n:]=interpolate_cartesian(cart,ring)*scale[offset:offset+size,None,None]
        dest[offset:offset+size,1:-1,:n]=(np.asarray(old[:,offset:offset+size],dtype=float).T/
            geo['cell_volume_mm3'][None,:]).reshape(size,40,n)
        for name,layer in (('z_minus',0),('z_plus',-1)):
            dest[offset:offset+size,layer,:]=interpolate_cartesian(halos[name][sub,0].astype(float),xy)*scale[offset:offset+size,None]
        block=dest[offset:offset+size]
        if not np.isfinite(block).all() or np.any(block<0):raise ValueError('Invalid guarded A')
        if offset%512==0:print(json.dumps(dict(detectors=offset+size,elapsed=time.monotonic()-start)),flush=True)
    dest.flush();del dest
    np.savez_compressed(output/'field_geometry.npz',xy_mm=xy,z_mm=z,original_points=n,selected_raw_detectors=selected)
    result=dict(status='BUILT_NOT_YET_VALIDATED',layout='detector_z_xy',shape=[10496,42,len(xy)],dtype='<f8',
        A_field_sha256=digest(output/'A_field.float64'),field_geometry_sha256=digest(output/'field_geometry.npz'),
        guard_ready_sha256=digest(ready_path),baseline_geometry_sha256=digest(geometry),
        baseline_factor_manifest_sha256=digest(factors/'factor_manifest.json'),
        baseline_B_sha256=digest(factors/'SysMat_polar'),original_cartesian_sha256=digest(raw_path),
        original_nodes='exact float64(Bfloat32)/full_cell_volume; no interior A replacement',
        interior_interpolation='original Delaunay stencils retained inside original hull',
        support_radius_mm=255,axial_support_mm=[-60,60],calibration_scales=scale.tolist(),
        elapsed_seconds=time.monotonic()-start,new_transport_photons=0)
    (output/'field_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','guard','factors','geometry','config','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();build(a.root,a.guard,a.factors,a.geometry,a.config,a.output)
