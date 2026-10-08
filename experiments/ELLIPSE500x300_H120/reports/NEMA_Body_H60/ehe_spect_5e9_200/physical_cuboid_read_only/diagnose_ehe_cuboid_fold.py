"""CPU-only sensitivity diagnostic using existing full Cartesian matrices; no new response."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def sum_all_bins(path,expected):
    """Consume every detector row once, binding the consumed bytes to the accepted SHA."""
    total=np.zeros((40,85,85),np.float64)
    h=hashlib.sha256();rows=0
    row_bytes=40*85*85*4
    with path.open('rb') as f:
        while True:
            b=f.read(32*row_bytes)
            if not b:break
            assert len(b)%row_bytes==0
            a=np.frombuffer(b,dtype='<f4').reshape(-1,40,85,85)
            assert np.all(np.isfinite(a)) and np.all(a>=0)
            total+=a.sum(axis=0,dtype=np.float64)
            rows+=a.shape[0];h.update(b)
    assert rows==2312 and h.hexdigest()==expected
    return total


def interpolate_xy(response,k,x,y):
    assert np.all(abs(x)<=252) and np.all(abs(y)<=252)
    fx=(x+252)/6;fy=(y+252)/6
    ix=np.minimum(fx.astype(int),83);iy=np.minimum(fy.astype(int),83)
    # Keep the original XY float32 interpolation fractions in the center diagnostic.
    tx=(fx-ix).astype(np.float32);ty=(fy-iy).astype(np.float32)
    return ((response[k,iy,ix]*(1-tx)+response[k,iy,ix+1]*tx)*(1-ty)
            +(response[k,iy+1,ix]*(1-tx)+response[k,iy+1,ix+1]*tx)*ty)


def run(binding,output):
    began=time.monotonic()
    assert not output.exists(),'Existing diagnostic output retained; do not repeat/overwrite'
    truth_path=Path(binding['truth_path'])
    assert digest(truth_path)==binding['truth_sha256']
    truth=np.load(truth_path)
    workers_path=Path(binding['workers_path'])
    assert digest(workers_path)==binding['workers_sha256']
    workers=np.load(workers_path)
    results={};consumed={}
    for name,e in [('A218',218),('A440',440),('C440to218',440)]:
        response_began=time.monotonic()
        folder=Path(binding['response_root'])/name
        assert digest(folder/'factor_manifest.json')==binding['factor_manifest_sha256'][name]
        manifest=json.loads((folder/'factor_manifest.json').read_text())
        matrix_sum=sum_all_bins(folder/'SysMat_cartesian',manifest['files']['SysMat_cartesian'])
        consumed[name]=manifest['files']['SysMat_cartesian']
        print('FULL_MATRIX_CONSUMED',name,'seconds',time.monotonic()-response_began,flush=True)
        k,j,i=np.nonzero(truth[f'activity_{e}_zyx'])
        assert k.min()>0 and k.max()<39
        assert np.array_equal(truth['z_mm'],np.arange(40)*3-58.5)
        x=truth['x_mm'][i];y=truth['y_mm'][j]
        mass=truth[f'activity_{e}_zyx'][k,j,i].astype(float)*27
        mass/=mass.sum()
        primaries=workers['primary_counts'][:,0 if e==218 else 1].reshape(20,10).sum(axis=1)
        # Exact integral of piecewise linear z interpolation over a uniform 3mm voxel.
        z_average=matrix_sum.copy()
        z_average[1:-1]=.125*matrix_sum[:-2]+.75*matrix_sum[1:-1]+.125*matrix_sum[2:]
        rows=[]
        for view in range(20):
            angle=2*np.pi*view/20;c=math.cos(angle);s=math.sin(angle)
            xr=x*c+y*s;yr=y*c-x*s
            center=float(np.dot(mass,interpolate_xy(matrix_sum,k,xr,yr))*primaries[view])
            old=binding['original_view_predicted'][name][view]
            assert math.isclose(center,old,rel_tol=1e-5,abs_tol=1e-3),'CPU sum differs from original unfitted center prediction'
            averages={}
            for n in (4,8):
                nodes,weights=np.polynomial.legendre.leggauss(n)
                average=0.
                for a,wa in zip(nodes,weights):
                    for b,wb in zip(nodes,weights):
                        xs=xr+1.5*(a*c+b*s);ys=yr+1.5*(b*c-a*s)
                        value=interpolate_xy(z_average,k,xs,ys)
                        average+=float(np.dot(mass,value))*wa*wb/4
                averages[str(n)]=average*primaries[view]
            rows.append(dict(view=view+1,original_center_predicted=old,cpu_center_predicted=center,
                cuboid_quadrature4_predicted=averages['4'],cuboid_quadrature8_predicted=averages['8'],
                quadrature8_over_center=averages['8']/center,
                quadrature8_minus4_over8=(averages['8']-averages['4'])/averages['8']))
        center=sum(r['cpu_center_predicted'] for r in rows)
        q4=sum(r['cuboid_quadrature4_predicted'] for r in rows)
        q8=sum(r['cuboid_quadrature8_predicted'] for r in rows)
        observed=binding['original_global_observed'][name]
        results[name]=dict(cpu_center_total=center,cuboid_quadrature4_total=q4,cuboid_quadrature8_total=q8,
            cuboid8_over_center=q8/center,quadrature8_minus4_over8=(q8-q4)/q8,
            actual_observed=observed,cuboid8_signed_bias=(q8-observed)/observed,views=rows,
            elapsed_seconds=time.monotonic()-response_began)
        print(name,json.dumps({k:v for k,v in results[name].items() if k!='views'}),flush=True)
    output.mkdir(parents=True)
    proof=dict(passed=True,scope='CPU read-only cuboid-averaging sensitivity of the existing interpolated full response; original scientific HOLD unchanged',
        scientific_status='HOLD',physical_gate_pass_claimed=False,job=1677211,rows_per_response=2312,views=20,
        channels=results,consumed_matrix_sha256=consumed,binding_sha256=digest(binding_path),code_sha256=digest(__file__),
        elapsed_seconds=time.monotonic()-began,cpu_readonly_rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        imaging_allocation_certificate=False,no_gpu_used=True,no_slurm_job_submitted=True,no_response_generated=True,
        no_observed_count_fit=True,no_production_input_modified=True,
        limitation='4x4 and 8x8 XY Gauss quadrature plus exact piecewise-linear z averaging test only the frozen interpolation model. Convergence of these two quadratures is diagnostic, not a rigorous integration error bound or a physics calibration PASS.')
    target=output/'cuboid_fold.json'
    with target.open('wb') as f:
        f.write((json.dumps(proof,indent=2,allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())
    print('READ_ONLY_DIAGNOSTIC_COMPLETE',digest(target),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();binding_path=a.binding;run(json.loads(binding_path.read_text()),a.output)
