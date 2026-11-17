"""Accepted HE mono results only: native 4mm/72mm MIPs, no fit or filtering."""
from pathlib import Path
import argparse
import copy
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import workflow as w

OUT=w.ROOT/'Analysis_Plots/research/ac225_results_meeting_he_derenzo_20261116'
OLD=w.ROOT/'Analysis_Plots/research/ac225_results_meeting_20261116'
COMPARISON=w.ROOT/'Analysis_Plots/results/Derenzo/summary_20261112'


def native_mip(route,iteration):
    root=w.DATA/route/'delivery/reconstruction'/str(w.ROUTES[route]['energy'])
    proof=w.read(w.REPORT/route/'strict_delivery.json');assert proof['passed']
    path=root/f'iteration_{iteration:04d}.npz'
    assert w.sha(path)==proof['files'][path.name]['sha256']
    with np.load(root/'final.npz') as f:
        axes=[f[n].copy() for n in ('x_mm','y_mm','z_mm')]
    with np.load(path) as f:a=f['density_xyz'].copy()
    assert a.shape==(126,126,34) and np.isfinite(a).all() and a.min()>=0
    keep=abs(axes[2])<36
    assert keep.sum()==18 and all(np.allclose(np.diff(x),4) for x in axes)
    mip=a[:,:,keep].max(axis=2).T
    return mip,dict(path=str(path),sha256=w.sha(path),z_centers_mm=axes[2][keep].tolist(),
                    extent_mm=[-252,252,-252,252],raw_peak=float(a.max()),mip_peak=float(mip.max()))


def panel(mip,path):
    fig=plt.figure(figsize=(5.04,3.04),dpi=200,facecolor='white');ax=fig.add_axes([0,0,1,1])
    ax.imshow(mip,extent=[-252,252,-252,252],origin='lower',cmap='gray_r',vmin=0,vmax=2,interpolation='none')
    ax.set(xlim=(-252,252),ylim=(-152,152),aspect='equal')
    ax.add_patch(Ellipse((0,0),500,300,fill=False,edgecolor='#D4DDE2',linewidth=.55));ax.axis('off')
    fig.savefig(path,dpi=200,pad_inches=0);plt.close(fig)


def route_figures(route):
    for name in ('assets','data','figures'):(OUT/name).mkdir(parents=True,exist_ok=True)
    spec=w.ROUTES[route];energy=spec['energy'];dose=spec['dose']
    oldhash=w.read(COMPARISON/'data/artifact_sha256.json')['data/display_arrays.npz']
    assert w.sha(COMPARISON/'data/display_arrays.npz')==oldhash
    with np.load(COMPARISON/'data/display_arrays.npz') as f:old={n:f[n].copy() for n in f.files}
    selected=[10,20,40,60,100,200];mips={};metadata={};panels=[]
    for it in selected:
        mip,record=native_mip(route,it);mips[it]=mip;metadata[str(it)]=record
        p=OUT/'assets'/f'Derenzo_{dose}_HE_{energy}_{it}.png';panel(mip,p)
        panels.append(dict(key=f'{dose}_{energy}_HE_mip_{it}',family='Derenzo',path=str(p),sha256=w.sha(p),
            shape=list(mip.shape),vmax=2,peak=float(mip.max()),saturated_pixels=int((mip>2).sum()),extent_mm=[-252,252,-252,252]))
    methods=['SC','JSCC','HE'] if energy==440 else ['SC','HE']
    fig,axs=plt.subplots(len(methods),4,figsize=(15,3.15*len(methods)),squeeze=False)
    for row,method in enumerate(methods):
        its=[20,60,200] if method=='HE' else [500,2000,9950 if method=='JSCC' and dose=='5e10' else 10000]
        arrays=[old[f'truth_{energy}']]+[mips[i] if method=='HE' else old[f'{dose}_{energy}_{method}_mip_{i}'] for i in its]
        for col,(ax,a) in enumerate(zip(axs[row],arrays)):
            extent=[-252,252,-252,252] if method=='HE' and col else [-252,252,-150,150]
            ax.imshow(a,extent=extent,origin='lower',cmap='gray_r',vmin=0,vmax=2,interpolation='none')
            ax.set(xlim=(-252,252),ylim=(-152,152),aspect='equal');ax.axis('off')
            ax.add_patch(Ellipse((0,0),500,300,fill=False,edgecolor='#ccd3d8',linewidth=.6))
            ax.set_title('Truth' if col==0 else f'{its[col-1]:,} iterations',fontsize=13)
            if col==0:ax.text(-.08,.5,method,transform=ax.transAxes,rotation=90,va='center',ha='center',fontsize=18,fontweight='bold')
    fig.suptitle(f'Derenzo {dose}, {energy} keV | central 72 mm MIP',fontsize=21)
    fig.text(.5,.02,'Uniform scale 0-2 relative to true hot-rod density; no smoothing or gain fit. HE: 4 mm Cartesian grid.',ha='center',fontsize=11)
    fig.subplots_adjust(left=.05,right=.99,top=.91,bottom=.07,wspace=.03,hspace=.21)
    figure=OUT/'figures'/f'Derenzo_{dose}_{energy}_SC_JSCC_HE.png';fig.savefig(figure,dpi=160,facecolor='white');plt.close(fig)
    np.savez_compressed(OUT/'data'/f'{route}_display_arrays.npz',**{f'HE_mip_{i}':a for i,a in mips.items()})
    result=dict(passed=True,route=route,source_release=w.read(w.REPORT/'freeze.json')['release'],selected=metadata,
                panels=panels,figure=str(figure),figure_sha256=w.sha(figure),strict_delivery_sha256=w.sha(w.REPORT/route/'strict_delivery.json'),
                matched_same_phantom_geometry=True,geometry_sha256=w.sha(w.DATA/'payload'/('inputs_'+route)/'geometry.json'),
                common_scale=[0,2],gain_fit=False,smoothing=False,physical_calibration_passed=False,visual_QA_passed=False)
    w.save(OUT/'data'/f'{route}_visual_data.json',result);return result


def presentation_data():
    d=copy.deepcopy(w.read(OLD/'data/presentation_data.json'))
    for p,h in d['sources'].items():assert w.sha(Path(p))==h
    # Retain all existing NEMA, SC, JSCC, truth panels and native CNR-BV data.
    removed={r['path'] for r in d['panel_records'] if r['family']=='Derenzo' and '_EHE_' in r['key']}
    d['panel_records']=[r for r in d['panel_records'] if r['path'] not in removed]
    changed=[]
    for page in d['image_slides']:
        if page['family']!='Derenzo':continue
        route=f'{page["dose"]}_{page["energy"]}'
        result=route_figures(route)
        row=next(r for r in page['rows'] if r['method']=='EHE');row['method']='HE';row['images']=[]
        for iteration in (20,60,200):
            p=next(x for x in result['panels'] if x['key'].endswith('_'+str(iteration)))
            row['images'].append(dict(iteration=iteration,path=p['path']));d['panel_records'].append(p)
        for rec in result['selected'].values():d['sources'][rec['path']]=rec['sha256']
        proof=w.REPORT/route/'strict_delivery.json';d['sources'][str(proof)]=w.sha(proof)
        changed.append(dict(route=route,iterations=[20,60,200],HE_input_primary_total=w.ROUTES[route]['total']))
    d['HE_Derenzo_replacement']=changed
    w.save(OUT/'data/presentation_data.json',d)
    return dict(passed=True,routes=changed,panel_count=len(d['panel_records']),out=str(OUT))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--route',choices=list(w.ROUTES));a=p.parse_args()
    r=route_figures(a.route) if a.route else presentation_data()
    print(json.dumps({k:v for k,v in r.items() if k not in ('selected','panels')},ensure_ascii=False,indent=2))
