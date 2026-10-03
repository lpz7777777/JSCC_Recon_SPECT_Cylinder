"""Plot rejected-event operator diagnostics; these are not reconstructed images."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

HERE=Path(__file__).resolve().parent
DATA=HERE/'generated/response_mismatch_cut3_v1/scan'
OUT=HERE/'reports/NEMA_Body_H60/response_mismatch_cut3_v1'


def main():
    g=np.load(HERE/'generated/Geometry/geometry.npz')
    active=g['active_indices']; c=g['coordinates_mm'][active].reshape(40,-1,3)
    z=c[:,0,2]; v=(g['cell_volume_mm3']*g['ellipse_fraction'])[active].reshape(40,-1)
    titles={'removed_backprojection_active.npy':('Sum of removed-event normalized ellipse responses','removed_backprojection.png'),
            'removed_truth_score_active.npy':('Removed-event truth score / baseline sensitivity (production floor)','removed_truth_score.png')}
    angle=np.linspace(0,2*np.pi,300)
    summaries={}
    for source,(title,name) in titles.items():
        values=np.load(DATA/source).reshape(40,-1)
        if not np.isfinite(values).all() or np.any(values<0): raise ValueError('Invalid diagnostic array')
        positive=values[values>0]
        top=float(values.max()); lo=max(top*1e-6,float(np.quantile(positive,.01)))
        norm=LogNorm(vmin=lo,vmax=top)
        fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
        for ax,layer in zip(axes[:2],[20,39]):
            im=ax.scatter(c[layer,:,0],c[layer,:,1],c=np.maximum(values[layer],lo),s=9,cmap='magma',norm=norm)
            ax.plot(250*np.cos(angle),150*np.sin(angle),color='gray',lw=.8)
            ax.set(xlim=(-255,255),ylim=(-155,155),aspect='equal',title=f'z={z[layer]:+.1f} mm',xlabel='x (mm)',ylabel='y (mm)')
        fig.colorbar(im,ax=axes[:2],label='Diagnostic value (log color scale)',shrink=.8)
        axes[2].semilogy(z,np.maximum(values.max(1),1e-20),'o-',label='Layer maximum')
        axes[2].axvline(-30,color='gray',ls=':'); axes[2].axvline(30,color='gray',ls=':')
        axes[2].set(xlabel='z (mm)',ylabel='Layer maximum',title='Full 120-mm axial profile')
        axes[2].grid(alpha=.2)
        fig.suptitle(title+'\nAll 1,168 rejected events; operator diagnostic, no reconstruction',fontsize=11)
        fig.savefig(OUT/name,dpi=160); plt.close(fig)
        peak=np.unravel_index(values.argmax(),values.shape)
        summaries[name]={'maximum':top,'peak_xyz_mm':c[peak].tolist(),
                         'display':'native polar centers, log colors; no smoothing; full-z statistics'}
    ratio=np.load(DATA/'sensitivity_new_old_ratio.npy')
    summaries['sensitivity_new_old_ratio']={k:float(val) for k,val in
        [('min',ratio.min()),('median',np.median(ratio)),('max',ratio.max())]}
    with (DATA/'rejected_events.csv').open() as stream: rows=list(csv.DictReader(stream))
    counts={str(view):sum(int(r['view'])==view for r in rows) for view in range(1,21)}
    pairs={}
    for r in rows:
        key=r['first_y_mm']+'->'+r['second_y_mm']; pairs[key]=pairs.get(key,0)+1
    summaries['removed_per_view']=counts; summaries['removed_layer_pairs']=pairs
    summaries['q_percentiles']={str(q):float(np.quantile([float(r['q']) for r in rows],q)) for q in (.0,.5,.9,.99,1)}
    for name,key in [('first_energy_keV','e1_MeV'),('second_energy_keV','e2_MeV')]:
        energy=np.array([float(r[key])*1000 for r in rows])
        summaries[name]={str(q):float(np.quantile(energy,q)) for q in (0,.1,.5,.9,1)}
    energy=np.array([(float(r['e1_MeV'])+float(r['e2_MeV']))*1000 for r in rows])
    edges=np.arange(350,np.ceil(energy.max()/25)*25+25,25)
    counts,_=np.histogram(energy,edges)
    summaries['sum_energy_keV']={'percentiles':{str(q):float(np.quantile(energy,q)) for q in (0,.1,.5,.9,1)},
        'bin_edges_keV':edges.tolist(),'bin_counts':counts.tolist()}
    baseline=HERE/'generated/RemoteResults/NEMA_Body_H60_5e9_1644876/Image_440_ComptonOnly_history.float32'
    old=np.memmap(baseline,'<f4',mode='r',shape=(200,len(active)))
    peak=int(old[-1].argmax())
    summaries['removed_score_at_baseline_compton_final_peak']={
        'xyz_mm':g['coordinates_mm'][active[peak]].tolist(),
        'score':float(np.load(DATA/'removed_truth_score_active.npy')[peak]),
        'interpretation':'Response diagnostic at true density with production floor; not a reconstructed density or predicted improvement'}
    (OUT/'diagnostics.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__': main()
