"""Specified finite measurement-space Poisson experiment, not a JSCC simulation.

All C1/C2/energy bins, including zero counts, enter the forward probability and
its exact column-sum sensitivity. No observed List rows generate the model.
"""
from pathlib import Path
import json
import csv
import hashlib
import numpy as np
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def run():
    out=HERE/'reports/NEMA_Body_H60/process_list_global_audit_v4/identifiability_toy'
    out.mkdir(exist_ok=False)
    geo=np.load(HERE/'generated/Geometry/geometry.npz')
    xyz=geo['coordinates_mm']
    controls=np.flatnonzero((np.hypot(xyz[:,0],xyz[:,1])<=30)&(abs(xyz[:,2])<=10.5))
    cells=controls[np.linspace(0,len(controls)-1,64,dtype=int)]
    source=xyz[cells];volume=geo['cell_volume_mm3'][cells]
    E0=.440;mass=.511
    edges=np.linspace(.05,2*E0**2/(mass+2*E0)-.001,33)
    response=[];labels=[]
    for c1,p1 in enumerate([(-60,300,-30),(60,300,-30),(-60,300,30),(60,300,30)]):
        p1=np.array(p1,dtype=float);incoming=p1-source
        incoming/=np.linalg.norm(incoming,axis=1)[:,None]
        # Explicit toy first-hit efficiency, not the project's photopeak A.
        A=np.exp(-((source[:,0]-p1[0])/100)**2)/(np.sum((source-p1)**2,axis=1)/300**2)
        for c2,(dx,dz) in enumerate([(70,20),(-70,20),(70,-20),(-70,-20)]):
            p2=p1+np.array([dx,30,dz]);ray=(p2-p1)/np.linalg.norm(p2-p1)
            cosine=incoming@ray;beta=np.arccos(np.clip(cosine,-1,1))
            epsilon=1/(1+E0/mass*(1-cosine));transfer=E0*(1-epsilon)
            sd=.13/2.355*np.sqrt(.511*transfer)
            KN=epsilon**2*(epsilon+1/epsilon-np.sin(beta)**2)
            binprob=np.diff(ndtr((edges[:,None]-transfer[None])/sd[None]),axis=0)
            for k,row in enumerate(binprob*A[None]*KN[None]):
                response.append(row*volume);labels.append([c1,c2,k])
    R=np.array(response);R*=1e-5/R.sum(0).max()
    S=R.sum(0)
    norm=R/np.linalg.norm(R,axis=0)[None]
    gram=norm.T@norm;off=gram[np.triu_indices(64,1)]
    rng=np.random.default_rng(51005002);records=[];summary=[]
    for level in (120,1200):
        truth=np.ones(64)*(level/S.sum());prediction=R@truth
        for name,y in [('noiseless',prediction),('poisson',rng.poisson(prediction))]:
            image=np.ones(64)
            for iteration in range(1,2001):
                image*=R.T@(y/np.maximum(R@image,1e-300))/S
                if iteration in (1,50,100,500,1000,2000):
                    mean=float(image@volume/volume.sum())
                    records.append(dict(expected_events=level,mode=name,iteration=iteration,
                        max_over_volume_weighted_mean=float(image.max()/mean),
                        relative_L2_to_uniform_truth=float(np.linalg.norm(image-truth)/np.linalg.norm(truth)),
                        image_mass_in_max_cell=float((image*volume).max()/(image@volume))))
            summary.append(dict(expected_events=level,observed_events=float(y.sum()),mode=name,
                max_over_uniform_truth=float((image/truth).max()),uniform_truth_nrmse=float(np.linalg.norm(image-truth)/np.linalg.norm(truth)),
                Poisson_deviance=float(2*np.sum(np.where(y>0,y*np.log(np.maximum(y,1e-300)/np.maximum(R@image,1e-300)),0)-y+R@image))))
    with (out/'toy_iteration_metrics.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    np.savez_compressed(out/'measurement_space_model.npz',response_density=R,sensitivity_density=S,
        source_centres=source,source_volumes=volume,source_indices=cells,energy_bin_edges=edges,measurement_labels=labels)
    manifest=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        columns=64,measurement_bins=len(R),all_zero_count_bins_retained=True,seed=51005002,
        likelihood='Poisson in discrete C1,C2,E1 bins',S='Exact column sum of the generating matrix',
        off_diagonal_column_cosines_p05_p50_p95=np.quantile(off,[.05,.5,.95]).tolist(),results=summary,
        limitations='Explicit toy transport and geometry probabilities; not a validation of actual Geant4 K*A, NEMA, or historical FOV120',
        interpretation='Demonstrates possible finite-count ML concentration on correlated columns; cannot attribute actual peaks quantitatively')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for level in (120,1200):
        for name in ('noiseless','poisson'):
            values=[r for r in records if r['expected_events']==level and r['mode']==name]
            ax.plot([r['iteration'] for r in values],[r['max_over_volume_weighted_mean'] for r in values],label=f'{name}, expected {level}')
    ax.set(xlabel='MLEM iteration',ylabel='maximum / volume-weighted mean',title='Specified Poisson toy: same generating and reconstruction model')
    ax.grid(alpha=.25);ax.legend();fig.savefig(out/'statistical_concentration.png',dpi=160);plt.close(fig)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':run()
