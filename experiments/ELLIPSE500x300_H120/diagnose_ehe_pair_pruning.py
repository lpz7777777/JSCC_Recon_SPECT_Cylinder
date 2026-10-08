"""CPU read-only reproduction of frozen crystal-pair support tests, not a response run."""
import math
import sys
import time
sys.dont_write_bytecode=True
import numpy as np
from ehe_common import DATA,REPORT,digest,read,write


def support_tests(detector,image,physics,dtype):
    d=detector.astype(dtype);im=image.astype(dtype);ph=physics.astype(dtype)
    assert im[7]==0 and im[8]==0 and im[9]==0
    cast=dtype
    pi=cast(np.pi)
    ex=cast(.5)*(im[:3]-cast(1))*im[3:6]
    radius=np.sqrt(ex[0]*ex[0]+ex[1]*ex[1]+ex[2]*ex[2])
    n=len(d);pairs=0;rejected=0;ambiguous=0;minimum_allowed_margin=float('inf')
    closest_rejected_margin=float('-inf');maximum_f32_f64_margin=None
    target_radius=np.sqrt((cast(.5)*d[:,3])**2+(cast(.5)*d[:,4])**2+(cast(.5)*d[:,5])**2)[None,:]
    for start in range(0,n,64):
        sd=d[start:start+64]
        dx=d[None,:,0]-sd[:,None,0]
        dy=d[None,:,1]-sd[:,None,1]
        dz=d[None,:,2]-sd[:,None,2]
        distance=np.sqrt(dx*dx+dy*dy+dz*dz)
        valid=distance>0
        dist=np.where(valid,distance,cast(1))
        ox=dx/dist;oy=dy/dist;oz=dz/dist
        ix=sd[:,0];iy=sd[:,1]+im[11];iz=sd[:,2]-im[10]
        incoming=np.sqrt(ix*ix+iy*iy+iz*iz)
        ix/=incoming;iy/=incoming;iz/=incoming
        cosine=np.clip(ix[:,None]*ox+iy[:,None]*oy+iz[:,None]*oz,cast(-1),cast(1))
        angle=np.arccos(cosine)
        half=np.arcsin(np.minimum(cast(1),radius/incoming))[:,None]
        half=half+np.arcsin(np.minimum(cast(1),target_radius/dist))
        half=np.where(distance>target_radius,half,pi)+cast(1e-5)
        tmin=np.where(incoming[:,None]>radius,np.maximum(cast(0),angle-half),cast(0))
        tmax=np.where(incoming[:,None]>radius,np.minimum(pi,angle+half),pi)
        energy=ph[7]
        maximum_energy=energy/(cast(1)+(energy/cast(511))*(cast(1)-np.cos(tmin)))
        minimum_energy=energy/(cast(1)+(energy/cast(511))*(cast(1)-np.cos(tmax)))
        resolution=d[None,:,9]
        lower=ph[5] if ph[4]>0 else (cast(1)-resolution/cast(2))*energy
        upper=ph[6] if ph[4]>0 else (cast(1)+resolution/cast(2))*energy
        support_upper=maximum_energy*(cast(1)+cast(5)*resolution*np.sqrt(energy/maximum_energy)/cast(2.35482))
        support_lower=minimum_energy*(cast(1)-cast(5)*resolution*np.sqrt(energy/minimum_energy)/cast(2.35482))
        margin=np.minimum(support_upper+cast(.001)-lower,upper-(support_lower-cast(.001)))
        prune=valid&(margin<=0)
        pairs+=int(valid.sum());rejected+=int(prune.sum());ambiguous+=int((valid&(abs(margin)<.001)).sum())
        if np.any(valid&~prune):minimum_allowed_margin=min(minimum_allowed_margin,float(margin[valid&~prune].min()))
        if np.any(prune):closest_rejected_margin=max(closest_rejected_margin,float(margin[prune].max()))
    return dict(ordered_off_diagonal_pairs=pairs,rejected_by_support_test=rejected,
        retained_by_support_test=pairs-rejected,within_1e_minus3_keV_of_decision=ambiguous,
        minimum_retained_support_margin_keV=minimum_allowed_margin if math.isfinite(minimum_allowed_margin) else None,
        closest_rejected_support_margin_keV=closest_rejected_margin if math.isfinite(closest_rejected_margin) else None,
        fov_bound_radius_mm=float(radius))


def audit():
    began=time.monotonic()
    freeze=read(REPORT/'response_repair_freeze.json');payload=DATA/freeze['payload_dir']
    source='engine/ScatterGen_RayTracing_CircularHole/scatter.cu'
    assert digest(payload/source)==freeze['sha256'][source]
    original_gate=digest(REPORT/'physical_gate.json')
    assert not read(REPORT/'physical_gate.json')['passed']
    results={};bindings={}
    for name in ['A218','A440','C440to218']:
        p=payload/'params'/name
        for file in ['Params_Detector.dat','Params_Image.dat','Params_Physics.dat']:
            key=f'params/{name}/{file}';assert digest(p/file)==freeze['sha256'][key];bindings[key]=digest(p/file)
        detector=np.fromfile(p/'Params_Detector.dat','<f4')[1:].reshape(2312,12)
        assert np.all(detector[:,11]==1) and len(np.unique(detector[:,1]))==1
        image=np.fromfile(p/'Params_Image.dat','<f4');physics=np.fromfile(p/'Params_Physics.dat','<f4')
        slabs=[]
        for slab in range(4):
            im=image.copy();im[2]=10;im[10]=-45+30*slab
            receipt=REPORT/f'response_preservation_1672966/{name}/slab_{slab}/receipt.json'
            saved=read(receipt)
            import hashlib
            derived=hashlib.sha256(im.tobytes()).hexdigest()
            assert derived==saved['files']['Params_Image.dat']
            assert digest(p/'Params_Detector.dat')==saved['files']['Params_Detector.dat']
            assert digest(p/'Params_Physics.dat')==saved['files']['Params_Physics.dat']
            f32=support_tests(detector,im,physics,np.float32)
            f64=support_tests(detector,im,physics,np.float64)
            assert f32['ordered_off_diagonal_pairs']==2312*2311==f64['ordered_off_diagonal_pairs']
            slabs.append(dict(slab=slab,registered_image_sha256=derived,saved_receipt_sha256=digest(receipt),
                cpu_float32=f32,cpu_float64=f64,aggregate_pruned_counts_equal=f32['rejected_by_support_test']==f64['rejected_by_support_test']))
        results[name]=slabs
        print(name,[(s['cpu_float32']['rejected_by_support_test'],s['cpu_float32']['minimum_retained_support_margin_keV']) for s in slabs],flush=True)
    write(REPORT/'physical_pair_pruning_read_only.json',dict(passed=True,
        scope='CPU reproduction of the frozen crystal-pair pruning support arithmetic for all ordered detector pairs and registered slabs; not a GPU response or a physical PASS',
        scientific_status='HOLD',physical_gate_pass_claimed=False,science_job=1677211,
        source_code_sha256=digest(payload/source),producer_release_key=freeze['release_key'],
        parameter_sha256=bindings,channels=results,elapsed_seconds=time.monotonic()-began,
        code_sha256=digest(__file__),original_physical_gate_sha256=original_gate,
        no_response_calculated=True,no_production_input_modified=True,no_observed_count_fit=True,
        limitation='CPU NumPy float32 and float64 support arithmetic is not a capture of GPU runtime flags. Matching counts do not imply bitwise agreement. It audits pair eligibility only, not surface/position quadrature, material attenuation, Gaussian collimator cutoff, multiple-interaction histories, or measured pathway fractions.'))
    assert digest(REPORT/'physical_gate.json')==original_gate


if __name__=='__main__':audit()
