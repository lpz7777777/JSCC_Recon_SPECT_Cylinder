"""Independently check all saved ROI measurements via sparse interpolation.

Reads completed histories only. Does not generate images or sum energies.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix

ROOT=Path(__file__).resolve().parents[2];H=ROOT/'experiments/ELLIPSE500x300_H120'
sys.path.insert(0,str(H))
from nema_roi_policy import build_masks
from reconstruction_output_policy import EHE_CHANNELS
REPORT=H/'reports/nema_interior_roi_20261010'


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):d.update(block)
    return d.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def main():
    started=time.monotonic();proof=read(REPORT/'scientific_acceptance.json')
    for mapping in ('code_sha256','histories_sha256','authorities_sha256'):
        for name,expected in proof[mapping].items():
            if sha(ROOT/name)!=expected:raise ValueError('Bound source changed: '+name)
    for name,expected in proof['outputs_sha256'].items():
        if sha(REPORT/name)!=expected:raise ValueError('Derived output changed: '+name)
    other_sources={
        'truth_sha256':H/'generated/NEMA_Body_H60/truth_3mm.npz',
        'geometry_sha256':H/'generated/ehe_spect_5e9_200/payload/whole_geometry.npz',
        'phantom_manifest_sha256':H/'reports/NEMA_Body_H60/manifest.json',
        'source_manifest_sha256':H/'reports/dual_energy_review_20261010/scientific_sources.json',
        'source_scale_record_sha256':H/'reports/NEMA_Body_H60/ehe_forward_poisson_5e10_200/comparison/comparison_report.json',
        'source_config_sha256':H/'nema_body_h60_config.json',
        'original_interpolator_sha256':H/'analyze_nema_result.py'}
    for key,path in other_sources.items():
        if sha(path)!=proof[key]:raise ValueError('Bound phantom/display source changed: '+str(path))
    t=np.load(H/'generated/NEMA_Body_H60/truth_3mm.npz');g=np.load(H/'generated/ehe_spect_5e9_200/payload/whole_geometry.npz')
    meta=read(H/'reports/NEMA_Body_H60/manifest.json');m=build_masks(t,meta,read(H/'nema_body_h60_config.json'))
    union=m['background'].copy()
    for mask in m['spheres'].values():union|=mask
    pooled=np.flatnonzero(union);iz,iy,ix=np.unravel_index(pooled,union.shape)
    query=np.column_stack((t['x_mm'][ix],t['y_mm'][iy]));tri=Delaunay(g['coordinates_mm'][:3301,:2]);simplex=tri.find_simplex(query)
    if np.any(simplex<0):raise ValueError('ROI outside interpolation domain')
    xy=tri.simplices[simplex];transform=tri.transform[simplex]
    bary=np.einsum('nij,nj->ni',transform[:,:2,:],query-transform[:,2,:])
    weights=np.column_stack((bary,1-bary.sum(axis=1)))
    full_ids=iz[:,None]*3301+xy
    inverse=np.full(132040,78920,dtype=np.int64);inverse[g['active_indices']]=np.arange(78920)
    operator=coo_matrix((weights.ravel(),(np.repeat(np.arange(len(pooled)),3),inverse[full_ids].ravel())),shape=(len(pooled),78921)).tocsr()
    bg=np.searchsorted(pooled,np.flatnonzero(m['background']))
    sphere_index={d:np.searchsorted(pooled,np.flatnonzero(mask)) for d,mask in m['spheres'].items()}
    actual=rows(REPORT/'sphere_iteration_metrics.csv');native=rows(REPORT/'common_background_iteration_metrics.csv')
    sphere_lookup={(r['system'],r['channel'],int(r['iteration']),int(r['diameter_mm'])):r for r in actual}
    background_lookup={(r['system'],r['channel'],int(r['iteration'])):r for r in native}
    if len(sphere_lookup)!=5832 or len(background_lookup)!=972:raise ValueError('Duplicate or missing metric rows')
    maximum=0.;verified=0
    def close(value,text):
        nonlocal maximum
        if value is None:
            if text!='':raise ValueError('Undefined metric was replaced by a value')
        else:
            rel=abs(value-float(text))/max(1.,abs(value));maximum=max(maximum,rel)
            if rel>1e-6:raise ValueError('Independent sparse-fold measurement differs')
    for route in proof['route_coverage']:
        system,channel=route['system'],route['channel'];end=route['iterations_end'];step=route['save_step']
        if 'Plus218' in channel:raise ValueError('Cross-energy sum route in current analysis')
        matches=[ROOT/n for n in proof['histories_sha256'] if channel+'_history.float32' in n and
            ('compton_energy_probability_v5_5e9_full10000' in n if system=='JSCC' else
             {'EHE':'ehe_spect_5e9_200','EHE matrix+Poisson':'ehe_forward_poisson_5e9_200','EHE Geant4 5e10':'ehe_spect_5e10_200','EHE matrix+Poisson 5e10':'ehe_forward_poisson_5e10_200'}[system] in n)]
        if len(matches)!=1:raise ValueError('Source route is ambiguous')
        hist=np.memmap(matches[0],dtype='<f4',mode='r',shape=(end//step,78920));energy=218 if channel==EHE_CHANNELS[1] else 440
        for iteration in range(0,end+1,step):
            density=np.ones(78920,dtype='<f4') if iteration==0 else hist[iteration//step-1]
            sampled=np.asarray(operator@np.r_[density,np.float32(0)],dtype=np.float32).astype(np.float64)
            b=sampled[bg];mean=float(b.mean());sd=float(b.std(ddof=1));nr=background_lookup[(system,channel,iteration)]
            close(mean,nr['background_mean']);close(sd,nr['background_std']);close(sd/mean,nr['background_cv'])
            for s in meta['spheres']:
                d=int(s['diameter_mm']);h=float(sampled[sphere_index[d]].mean());r=sphere_lookup[(system,channel,iteration,d)]
                hot=s['hot_energy_keV']==energy;crc=(h/mean-1)/(9 if hot else -1);cnr=(h-mean)/sd if sd>0 else None
                close(h,r['sphere_mean']);close(mean,r['background_mean']);close(sd,r['background_std']);close(crc,r['crc']);close(cnr,r['cnr'])
                if int(r['sphere_voxels'])!=len(sphere_index[d]) or int(r['background_voxels'])!=len(bg):raise ValueError('ROI count changed')
                verified+=1
        print('SPARSE_ROI_QA_COMPLETE',system,channel,flush=True)
    if proof['sum_histories_read']!=0 or proof['cross_energy_sum_images_computed']!=0 or len(proof['histories_sha256'])!=12:
        raise ValueError('Separate channel scope failed')
    output=dict(passed=True,all_iteration_rows_checked=972,all_sphere_rows_checked=verified,
        independent_method='One sparse Cartesian-query operator; independent ROI reductions from original active histories; no full-image export',
        maximum_scaled_absolute_difference=maximum,tolerance=1e-6,source_and_derived_SHA_passed=True,
        scientific_acceptance_sha256=sha(REPORT/'scientific_acceptance.json'),verifier_sha256=sha(__file__),
        elapsed_seconds=time.monotonic()-started,reconstruction_run=False,transport_run=False,cross_energy_sum_images_computed=0)
    (REPORT/'independent_numeric_acceptance.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('INDEPENDENT_ROI_ACCEPTANCE_PASSED',verified,maximum,flush=True)


if __name__=='__main__':main()
