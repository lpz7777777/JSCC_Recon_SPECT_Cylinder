"""GPU bucket/chord results versus an independent all-1250-hole float64 oracle."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from ehe_common import *

def oracle(pair,holes,half,center):
    s=pair[:3].astype(float);d=pair[3:].astype(float)-s
    local=s-np.array([0,center,0]);lo=0.;hi=1.
    for axis in range(3):
        if abs(d[axis])<1e-12:
            if abs(local[axis])>half[axis]:return 0.
        else:
            a=(-half[axis]-local[axis])/d[axis];b=(half[axis]-local[axis])/d[axis]
            lo=max(lo,min(a,b));hi=min(hi,max(a,b))
    if hi<=lo:return 0.
    a=np.full(len(holes),lo);b=np.full(len(holes),hi)
    if abs(d[1])<1e-12:
        b[(s[1]<holes[:,1])|(s[1]>holes[:,2])]=a[(s[1]<holes[:,1])|(s[1]>holes[:,2])]
    else:
        first=(holes[:,1]-s[1])/d[1];last=(holes[:,2]-s[1])/d[1]
        a=np.maximum(a,np.minimum(first,last));b=np.minimum(b,np.maximum(first,last))
    dx=s[0]-holes[:,0];dz=s[2]-holes[:,3];aa=d[0]**2+d[2]**2;bb=2*(dx*d[0]+dz*d[2]);cc=dx**2+dz**2-holes[:,4]**2
    if aa<1e-20:b[cc>0]=a[cc>0]
    else:
        disc=bb**2-4*aa*cc;root=np.sqrt(np.maximum(disc,0));a=np.maximum(a,(-bb-root)/(2*aa));b=np.minimum(b,(-bb+root)/(2*aa));b[disc<=0]=a[disc<=0]
    return max(0,hi-lo-np.maximum(b-a,0).sum())*np.linalg.norm(d)

def validate(binary,params,output):
    output.mkdir(parents=True,exist_ok=False)
    c=np.fromfile(params/'Params_Collimator.dat','<f4');im=np.fromfile(params/'Params_Image.dat','<f4')
    holes=c[100:].reshape(1250,9)[:,:5].astype(float);holes[:,1:3]+=float(im[11]);half=c[11:14].astype(float)/2;center=float(im[11]+c[14])
    rng=np.random.default_rng(500200);s=rng.uniform([-252,-252,-60],[252,252,60],(2048,3));e=rng.uniform([-136,349,-68],[136,359,68],(2048,3));pairs=[np.c_[s,e]]
    # Every aperture axis plus near-tangent and oblique rays, no nearest-hole truncation.
    for delta in (0,1.249,1.251):
        x=holes[:,0]+delta;z=holes[:,3]
        pairs.append(np.c_[x,np.zeros(1250),z,x,np.full(1250,354),z])
    data=np.vstack(pairs).astype('<f4');array_write(output/'rays.float32',data)
    expected=np.array([oracle(pair,holes,half,center) for pair in data]);array_write(output/'expected.float64',expected.astype('<f8'))
    subprocess.run([str(binary.resolve()),str((output/'rays.float32').resolve()),str((output/'actual.float32').resolve())],cwd=params,check=True)
    device_config=np.fromfile(output/'actual.float32.config','<f4')
    if device_config.shape!=(6258,) or not np.array_equal(device_config[:6250].reshape(1250,5),holes.astype('f4')):
        raise ValueError('GPU finite-hole parameter transfer differs')
    if device_config[6250]!=1250 or device_config[6253]!=1.25 or device_config[6254]!=1250:
        raise ValueError('GPU complete aperture index differs: '+repr(device_config[-8:].tolist()))
    if not np.isclose(device_config[6255],50.5/354,atol=1e-7):raise ValueError('GPU exact aperture axis interval failed')
    actual=np.fromfile(output/'actual.float32','<f4')
    if actual.shape!=expected.shape or np.any(~np.isfinite(actual)) or np.any(actual<0):raise ValueError('Invalid GPU chord result')
    error=np.max(abs(actual-expected));l2=np.linalg.norm(actual-expected)/np.linalg.norm(expected)
    passed=error<=1e-4 and l2<=1e-5
    write(output/'geometry_numerical.json',dict(passed=bool(passed),rays=len(data),holes=1250,max_absolute_chord_error_mm=float(error),relative_l2=float(l2),
        oracle='independent exhaustive float64 finite-cylinder union subtracted from clipped material box',files=hashes(output)))
    if not passed:raise ValueError(f'GPU hole chord audit failed: error_mm={error}, L2={l2}')
    print('EHE_HOLE_CHORD_PASS',len(data),float(error),float(l2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--params',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();validate(a.binary,a.params,a.output)
