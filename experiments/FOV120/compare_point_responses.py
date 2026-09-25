"""Compare independent monoenergetic Geant4 point counts with calibrated Factors.

Point response is interpolated in x/y (Delaunay) and z after removing density
cell volumes. No 1/20 factor: each point task emits all photons at one view.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np
from scipy.spatial import Delaunay
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from detector_csv import load_detector_coordinates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--points',type=Path,required=True)
    p.add_argument('--factors',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    data=np.load(a.points);meta=json.loads(a.points.with_suffix('.json').read_text())
    ref=a.factors/'440keV_RotateNum20'
    coords=np.loadtxt(ref/'coor_polar_full.csv',delimiter=',');zs=np.unique(coords[:,2]);nxy=len(coords)//len(zs)
    vol=np.fromfile(ref/'polar_cell_volume_mm3.float64',dtype='<f8')
    tri=Delaunay(coords[:nxy,:2]);detector=load_detector_coordinates(ref/'Detector.csv')
    layers=np.unique(np.round(detector[:,1],4))
    if not np.allclose(layers,[200,230,260,290]): raise ValueError('Detector normal-coordinate mismatch')
    matrices={name:np.memmap(a.factors/folder/'SysMat_polar',mode='r',dtype='<f4',shape=(len(coords),10496))
              for name,folder in [('218','218keV_RotateNum20'),('440','440keV_RotateNum20'),('cross','440keV_to218win_RotateNum20')]}
    rows=[]
    for i,r in enumerate(meta['records']):
        match=re.fullmatch(r'Point_E(218|440)_r(\d+)_a(\d+)_z([+-]\d+)',r['dataset'])
        if not match: raise ValueError(r['dataset'])
        e,rad,angle,z=map(int,match.groups());xy=rad*np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
        simplex=int(tri.find_simplex(xy))
        if simplex<0 or z<zs[0] or z>zs[-1]:raise ValueError('Point outside interpolation support')
        w=tri.transform[simplex,:2]@(xy-tri.transform[simplex,2]);w=np.r_[w,1-w.sum()]
        if w.min() < -1e-10:raise ValueError('Negative interpolation weight')
        upper=int(np.searchsorted(zs,z));lower=max(0,upper-1);upper=min(upper,len(zs)-1)
        wz=0 if upper==lower else (z-zs[lower])/(zs[upper]-zs[lower])
        indices=np.r_[tri.simplices[simplex]+lower*nxy,tri.simplices[simplex]+upper*nxy]
        weights=np.r_[w*(1-wz),w*wz]
        for channel,window in ([('218',218)] if e==218 else [('440',440),('cross',218)]):
            response=np.sum(np.asarray(matrices[channel][indices],dtype=np.float64)*(weights/vol[indices])[:,None],axis=0)
            predicted=response*r['photons'];observed=data[f'counts{window}'][i]
            per_layer=[]
            for layer in layers:
                selection=np.isclose(detector[:,1],layer);obs=int(observed[selection].sum());pred=float(predicted[selection].sum())
                per_layer.append(dict(normal_mm=float(layer),observed=obs,predicted=pred,ratio=obs/pred,
                                      poisson_relative_se=float(1/np.sqrt(obs)) if obs else None))
            rows.append(dict(index=r['index'],channel=channel,source_keV=e,r_mm=rad,azimuth_deg=angle,z_mm=z,
                             photons=r['photons'],observed=int(observed.sum()),predicted=float(predicted.sum()),
                             ratio=float(observed.sum()/predicted.sum()),
                             observed_efficiency=float(observed.sum()/r['photons']),predicted_efficiency=float(response.sum()),
                             poisson_debiased_relative_l2=float(np.sqrt(max(0,np.sum((observed-predicted)**2-observed))/np.sum(predicted**2))),
                             layers=per_layer))
    a.output.write_text(json.dumps(dict(complete_workers=meta['complete_workers'],expected_workers=meta['expected_workers'],
                                        missing_indices=meta['missing_indices'],actual_photons=meta['actual_photons'],
                                        approximation='Calibrated B/volume, linear z and barycentric xy interpolation; one view, no /20',rows=rows),indent=2)+'\n')
    for channel in matrices:
        ratios=[r['ratio'] for r in rows if r['channel']==channel]
        print(channel,'MC/model min, median, max',np.min(ratios),np.median(ratios),np.max(ratios))


if __name__=='__main__':main()
