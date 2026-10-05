"""Read-only 3D native-cell spike catalogue; no interpolation or filtering."""
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REPORT=HERE/'reports/NEMA_Body_H60/process_list_global_audit_v4'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def table(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def catalogue():
    out=REPORT/'native_spikes';out.mkdir(exist_ok=False)
    geo=np.load(HERE/'generated/Geometry/geometry.npz')
    active=geo['active_indices'];coordinates=geo['coordinates_mm'][active]
    volume=(geo['cell_volume_mm3']*geo['ellipse_fraction'])[active]
    truth_path=HERE/'generated/NEMA_Body_H60/truth_3mm.npz'
    truth=np.load(truth_path)
    source=json.loads((HERE/'reports/NEMA_Body_H60/manifest.json').read_text())
    assert digest(truth_path)==source['truth_sha256']
    x,y,z=(truth[k+'_mm'] for k in 'xyz')
    ix=np.clip(np.rint((coordinates[:,0]-x[0])/3).astype(int),0,len(x)-1)
    iy=np.clip(np.rint((coordinates[:,1]-y[0])/3).astype(int),0,len(y)-1)
    iz=np.clip(np.rint((coordinates[:,2]-z[0])/3).astype(int),0,len(z)-1)
    background=(truth['body_fraction_zyx']-truth['lung_fraction_zyx']-
        sum(truth[f'sphere_{d}_fraction_zyx'] for d in (10,13,17,22,28,37)))[iz,iy,ix]>=.99
    background &= abs(coordinates[:,2])<=24
    radius=np.hypot(coordinates[:,0]/250,coordinates[:,1]/150)
    sets=[];provenance={'truth_sha256':digest(truth_path),'files':{}}
    for group in ('legacy','ideal'):
        result=HERE/'generated/compton_first_scatter_v2/RemoteResults'/group
        manifest=json.loads((result/'run_manifest.json').read_text())
        verified=json.loads((result/'verification.json').read_text())
        assert verified['passed'] and verified['iterations']==2000
        assert manifest['geometry_sha256']==digest(HERE/'generated/Geometry/geometry.npz')
        for channel in ('440_ComptonOnly','440_SinglePlusCompton'):
            file=result/f'Image_{channel}_history.float32'
            assert file.stat().st_size==40*len(active)*4
            sha=digest(file);expected=next(r['sha256']['history'] for r in verified['outputs'] if r['channel']==channel)
            assert sha==expected
            provenance['files'][str(file.relative_to(ROOT))]=sha
            sets.append((group+' NEMA1e9',channel,np.memmap(file,'<f4',mode='r',shape=(40,len(active))),
                np.arange(50,2001,50),coordinates,volume,active,radius,background,geo['ellipse_fraction'][active]))
    fov=ROOT/'experiments/FOV120/generated'
    result=fov/'Results/Uniform_1e9_1626560'
    oldprovenance=json.loads((result/'VisualReport/provenance.json').read_text())
    coor=fov/'VisualizationInputs/Factors/440keV_RotateNum20/coor_polar_full.csv'
    coords=np.loadtxt(coor,delimiter=',')
    volumes=np.fromfile(coor.with_name('polar_cell_volume_mm3.float64'),'<f8')
    rr=np.hypot(coords[:,0],coords[:,1])/150
    bg=(rr<=.8)&(abs(coords[:,2])<=24)
    for channel in ('440_ComptonOnly','440_SinglePlusCompton'):
        file=result/f'Image_{channel}_Iter_10000_200.selected'
        assert file.stat().st_size==6*51240*4
        sha=digest(file)
        assert sha==next(v for k,v in oldprovenance['hashes'].items() if k.endswith(file.name))
        provenance['files'][str(file.relative_to(ROOT))]=sha
        sets.append(('circle Uniform1e9',channel,np.memmap(file,'<f4',mode='r',shape=(6,51240)),
            np.array(oldprovenance['selected_iterations']),coords,volumes,np.arange(51240),rr,bg,np.ones(51240)))
    metrics=[];peaks=[]
    for label,channel,history,iterations,coords,volumes,indices,rr,bg,fraction in sets:
        tree=cKDTree(coords)
        regions={'full':np.ones(len(coords),bool),'internal_core':(rr<=.8)&(abs(coords[:,2])<=24),
                 'internal_background':(rr<=.8)&bg,'axial_ends':abs(coords[:,2])>45,'lateral_edge':rr>.9}
        fixed_bg=float(history[-1][bg].astype(float)@volumes[bg]/volumes[bg].sum())
        for values,iteration in zip(history,iterations):
            assert np.isfinite(values).all() and np.all(values>=0)
            integral=float(values.astype(float)@volumes)
            for region,mask in regions.items():
                ids=np.flatnonzero(mask);j=int(ids[np.argmax(values[ids])])
                metrics.append(dict(dataset=label,channel=channel,iteration=int(iteration),region=region,
                    max_density=float(values[j]),max_over_fixed_final_native_background=float(values[j]/fixed_bg),
                    x_mm=float(coords[j,0]),y_mm=float(coords[j,1]),z_mm=float(coords[j,2]),
                    full_cell_index=int(indices[j]),effective_cell_volume_mm3=float(volumes[j]),fraction=float(fraction[j]),
                    region_integral_fraction=float(values[mask].astype(float)@volumes[mask]/integral),
                    p999=float(np.quantile(values[mask],.999))))
                if iteration==iterations[-1]:
                    selected=[]
                    for k in ids[np.argsort(values[ids])[::-1]]:
                        neighbours=tree.query_ball_point(coords[k],9)
                        if values[k]<values[neighbours].max():continue
                        if any(np.linalg.norm(coords[k]-coords[q])<9 for q in selected):continue
                        selected.append(int(k))
                        peaks.append(dict(dataset=label,channel=channel,iteration=int(iteration),region=region,
                            rank=len(selected),full_cell_index=int(indices[k]),x_mm=float(coords[k,0]),y_mm=float(coords[k,1]),
                            z_mm=float(coords[k,2]),density=float(values[k]),peak_over_fixed_native_background=float(values[k]/fixed_bg),
                            effective_cell_volume_mm3=float(volumes[k]),fraction=float(fraction[k]),
                            local_peak_radius_mm=9))
                        if len(selected)==3:break
    table(out/'native_region_iterations.csv',metrics);table(out/'native_local_peaks.csv',peaks)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for col,label in enumerate(('circle Uniform1e9','legacy NEMA1e9','ideal NEMA1e9')):
        for row,channel in enumerate(('440_ComptonOnly','440_SinglePlusCompton')):
            for region in ('full','internal_core','internal_background','axial_ends'):
                data=[r for r in metrics if r['dataset']==label and r['channel']==channel and r['region']==region]
                axes[row,col].plot([r['iteration'] for r in data],[r['max_over_fixed_final_native_background'] for r in data],label=region)
            axes[row,col].set(title=label+'\n'+channel,yscale='log',xlabel='saved iteration',ylabel='native peak / fixed final native background')
            axes[row,col].grid(alpha=.25)
            axes[row,col].legend(fontsize=7)
    fig.savefig(out/'native_spike_curves.png',dpi=150);plt.close(fig)
    provenance.update(source_sha256=digest(__file__),native_data=True,smoothing=False,cropping=False,
        NEMA_label_method='Nearest existing 3mm truth sample at native centre; diagnostic labels only, not ROI recovery measurement',
        scale='One native volume-weighted final background mean per historical dataset/channel; no absolute cross-experiment comparison',
        peak_definition='Physical 9mm neighbourhood local maxima; at most three per predefined region',
        interpretation='Historical kernels/events/geometries differ; catalogue is not a new paired reconstruction')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('NATIVE_METRICS',len(metrics),'PEAKS',len(peaks))

if __name__=='__main__':catalogue()
