"""Finite-volume representation of the registered 3mm NEMA truth.

The independent Poisson study intentionally generates and reconstructs using
the SAME native basis. No measured projection or fitted intensity enters it.
"""
from pathlib import Path
import time
import numpy as np
from scipy.sparse import coo_matrix
from geometry import grid
from ehe_common import read,write,digest

HERE=Path(__file__).resolve().parent


def average(truth,cells,order):
    nodes,weights=np.polynomial.legendre.leggauss(order)
    result={e:np.zeros((40,len(cells)),np.float64) for e in (218,440)}
    nx=len(truth['x_mm']);ny=len(truth['y_mm']);x0=truth['x_mm'][0]-1.5;y0=truth['y_mm'][0]-1.5
    w=(weights[:,None]*weights[None,:]/4).reshape(-1)
    for start in range(0,len(cells),32):
        c=cells[start:start+32];r0,r1,t0,t1=c.T
        rr=np.sqrt((r0[:,None]**2+r1[:,None]**2)/2+(r1[:,None]**2-r0[:,None]**2)/2*nodes)
        tt=(t0[:,None]+t1[:,None])/2+(t1[:,None]-t0[:,None])/2*nodes
        xx=rr[:,:,None]*np.cos(tt[:,None,:]);yy=rr[:,:,None]*np.sin(tt[:,None,:])
        ix=np.floor((xx-x0)/3).astype(int);iy=np.floor((yy-y0)/3).astype(int)
        valid=(ix>=0)&(ix<nx)&(iy>=0)&(iy<ny)
        row=np.broadcast_to(np.arange(len(c))[:,None,None],ix.shape).reshape(-1)
        pix=(iy*nx+ix).reshape(-1);valid=valid.reshape(-1)
        ww=np.broadcast_to(w,(len(c),len(w))).reshape(-1)
        mat=coo_matrix((ww[valid],(row[valid],pix[valid])),shape=(len(c),nx*ny)).tocsr()
        for e in result:result[e][:,start:start+len(c)]=(mat@truth[f'activity_{e}_zyx'].reshape(40,-1).T).T
    return {e:r.reshape(-1) for e,r in result.items()}


def prepare(output,geometry_path):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    marker=output/'source_native_manifest.json'
    if marker.exists():
        m=read(marker)
        if digest(output/'source_native.npz')!=m['source_native_sha256']:raise ValueError('Existing source changed')
        return m
    original=HERE/'generated/NEMA_Body_H60/truth_3mm.npz'
    truth=np.load(original);g=np.load(geometry_path)
    coords,cells,_=grid(read(HERE/'config.json'));assert np.array_equal(coords,g['coordinates_mm'])
    vol=g['cell_volume_mm3']*g['ellipse_fraction'];integrals={str(e):float(truth[f'activity_{e}_zyx'].astype(float).sum()*27) for e in (218,440)}
    convergence=[];previous=None;began=time.monotonic()
    for order in [32,64,128,256]:
        current=average(truth,cells,order)
        native={str(e):float(current[e]@vol) for e in current}
        errors={str(e):float(np.linalg.norm(current[e]-previous[e])/np.linalg.norm(current[e])) for e in current} if previous is not None else None
        row=dict(order=order,native_integral_mm3=native,relative_integral_error={e:native[e]/integrals[e]-1 for e in native},relative_L2_vs_previous=errors)
        convergence.append(row);print('SOURCE_CELL_QUADRATURE',row,flush=True)
        if errors and max(errors.values())<.002 and max(abs(v) for v in row['relative_integral_error'].values())<.001:break
        previous=current
    if errors is None or max(errors.values())>=.004 or max(abs(v) for v in row['relative_integral_error'].values())>=.001:
        raise ValueError('Source finite-volume quadrature insufficient; retain convergence proof')
    for e in current:
        assert current[e].min()>=0 and current[e].max()<=10+1e-9
        assert np.all(g['ellipse_fraction'][current[e]>1e-12]==1)
    np.savez_compressed(output/'source_native.npz',**{f'activity_{e}_full':current[e] for e in current},
        **{k:g[k] for k in ['active_indices','coordinates_mm','cell_volume_mm3','ellipse_fraction']})
    m=dict(passed=True,truth_3mm_sha256=digest(original),geometry_sha256=digest(geometry_path),
        source_native_sha256=digest(output/'source_native.npz'),code_sha256=digest(__file__),
        original_integral_mm3=integrals,native_integral_mm3=native,quadrature=convergence,
        source_basis='Native finite-volume average of original3mm piecewise-constant truth, Gauss in r²/theta, original40z layers',
        expected_gamma_split='original3mm integrals times original .114/.259 yields; never fitted observed counts',
        source_normalization='Native source normalized to probability mass for each energy; no gain fitted to data',
        matched_generation_reconstruction_basis=True,physical_calibration_passed=False,
        finite_volume_approximation_disclosed=True,elapsed_seconds=time.monotonic()-began)
    write(marker,m);return m


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True)
    a=p.parse_args();prepare(a.output,a.geometry)
