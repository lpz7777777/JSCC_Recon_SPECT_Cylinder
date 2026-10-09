"""Accepted EHE Geant4 5e10/5e9, model-Poisson 5e9 and JSCC trajectories."""
import math, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ehe_common import HERE, CHANNELS, read, write, digest, hashes, verify_files
from ehe_5e10_workflow import DATA, REPORT
from compare_ehe_5e9 import table
from analyze_nema_result import xy_interpolator, weighted_mean
import compare_ehe_jscc_full_trajectories as prior
import plot_ehe_forward_poisson as model_reference

SYSTEM='EHE Geant4 5e10'
OUT=REPORT/'comparison'
NODES=(0,10,20,50,100,150,200)
SYSTEMS=(SYSTEM,'EHE matrix+Poisson','EHE','JSCC')


def acquisition_budget():
    collection=read(DATA/'counts/collection.json')
    meta=read(HERE/'reports/NEMA_Body_H60/manifest.json')
    primary={str(e):collection['primary_counts'][i] for i,e in enumerate((218,440))}
    density={e:primary[e]/meta['relative_activity_integral_mm3'][e] for e in primary}
    if sum(primary.values())!=50000000000:raise ValueError('Actual full-sphere 5e10 primary dose required')
    return collection,primary,density


def scientific_data():
    accepted=read(REPORT/'formal_summary.json')
    if not accepted['passed'] or accepted['iterations']!=200 or accepted['frames_per_channel']!=20:raise ValueError('Actual strict formal200 acceptance required')
    result=DATA/'results/formal';verify_files(result,accepted['files'])
    f=read(REPORT/'freeze.json');payload=DATA/f['payload_dir'];verify_files(payload,f['sha256'])
    collection,primary,density=acquisition_budget()
    if accepted['counts_sha256']!=digest(DATA/'counts/collection.json'):raise ValueError('Actual observation identity differs')
    nr,sr,reference_routes,selected,source,views,scales,_,inputs,_,_=model_reference.scientific_data()
    truth=np.load(payload/'truth_3mm.npz');meta=read(HERE/'reports/NEMA_Body_H60/manifest.json')
    g=np.load(payload/'whole_geometry.npz');coords=g['coordinates_mm'];active=g['active_indices'];vol=g['cell_volume_mm3'][active]
    if digest(payload/'truth_3mm.npz')!=meta['truth_sha256']:raise ValueError('Existing authoritative 3D truth differs')
    scales[SYSTEM]={**density,'sum':sum(density.values())}
    x,y,z=[truth[k+'_mm'] for k in 'xyz'];interp=xy_interpolator(coords[:3301,:2],x,y)
    def image(v):
        full=np.zeros(132040,'<f4');full[active]=v
        return interp(full.reshape(40,3301))
    bgfraction=np.maximum(truth['body_fraction_zyx']-truth['lung_fraction_zyx']-
        sum(truth[f'sphere_{d}_fraction_zyx'] for d in prior.DIAMETERS),0)
    bg=(bgfraction>=.99)&(abs(z[:,None,None])<=25.5);xx,yy=np.meshgrid(x,y);masks={}
    for sphere in meta['spheres']:
        d=int(sphere['diameter_mm']);cx,cy,cz=sphere['center_mm']
        masks[d]=(((xx-cx)**2+(yy-cy)**2<=(d/2+25)**2)[None]&(abs(z[:,None,None]-cz)<=d/2+3)&(bgfraction>=.99))
    leak=abs(coords[active,2])>30;native=[];spheres=[]
    for c in CHANNELS:
        h=np.fromfile(result/f'Image_{c}_history.float32','<f4').reshape(20,78920);e=prior.energy(c)
        for index in range(21):
            iteration=index*10;v=np.full(78920,prior.initialization(c),'<f4') if index==0 else h[index-1]
            im=image(v);b=im[bg];mean=float(b.mean());sd=float(b.std(ddof=1));integral=float(v.astype(float)@vol);peak=int(v.argmax());order=np.argsort(v)
            budget=50000000000 if e=='sum' else primary[e]
            native.append(dict(system=SYSTEM,channel=c,iteration=iteration,max_density=float(v[peak]),peak_background_ratio=float(v[peak]/mean),
                peak_x_mm='' if not iteration else float(coords[active[peak],0]),peak_y_mm='' if not iteration else float(coords[active[peak],1]),peak_z_mm='' if not iteration else float(coords[active[peak],2]),
                p99_density=float(np.quantile(v.astype(np.float64),.99)),p999_density=float(np.quantile(v.astype(np.float64),.999)),volume_weighted_p999_density=float(np.interp(.999,np.cumsum(vol[order])/vol.sum(),v[order])),
                background_mean=mean,background_cv=sd/mean,total_integral=integral,integral_recovery=integral/budget,source_z_leakage=float(v[leak].astype(float)@vol[leak]/integral)))
            for sphere in meta['spheres']:
                own=str(sphere['hot_energy_keV']);d=int(sphere['diameter_mm'])
                if e!='sum' and e!=own:continue
                local=im[masks[d]];lm=float(local.mean());ls=float(local.std(ddof=1));hot=weighted_mean(im,truth[f'sphere_{d}_fraction_zyx'])
                nominal=9 if e!='sum' else 10*density[own]/sum(density.values())-1
                spheres.append(dict(system=SYSTEM,channel=c,iteration=iteration,energy_keV=int(own),diameter_mm=d,hot_mean=hot,local_background_mean=lm,
                    crc=(hot/lm-1)/nominal,cnr='' if iteration==0 else (hot-lm)/ls))
            if iteration in NODES:selected[SYSTEM,c,iteration]=im/scales[SYSTEM][e]
    nr.extend(native);sr.extend(spheres)
    labels={CHANNELS[1]:'218 corrected single',CHANNELS[0]:'440 single',CHANNELS[2]:'dual single sum'}
    routes=tuple((SYSTEM,c,labels[c],labels[c]) for c in (CHANNELS[1],CHANNELS[0],CHANNELS[2]))+reference_routes
    for e in ('218','440'):source[SYSTEM,e]=truth[f'activity_{e}_zyx']
    source[SYSTEM,'sum']=sum(density[e]*source[SYSTEM,e] for e in ('218','440'))/sum(density.values())
    inputs.update(new_transport_formal_summary=digest(REPORT/'formal_summary.json'),new_transport_collection=digest(DATA/'counts/collection.json'),
        new_source_angular_audit=digest(REPORT/'source_angular_audit.json'),new_plotting_source=digest(__file__),
        new_transport_histories={c:digest(result/f'Image_{c}_history.float32') for c in CHANNELS})
    return nr,sr,routes,selected,source,views,scales,collection,inputs,native,spheres


def gallery(kind,routes,selected,source,views):
    selector,extent,units=views[kind];fig,axes=plt.subplots(len(routes),8,figsize=(24,2.4*len(routes)+1.4),squeeze=False,layout='constrained')
    for row,(system,c,label,_) in enumerate(routes):
        nodes=prior.NODES['JSCC'] if system=='JSCC' else NODES
        for col,im in enumerate([source[system,prior.energy(c)]]+[selected[system,c,i] for i in nodes]):
            ax=axes[row,col];color=ax.imshow(selector(im),origin='lower',extent=extent,cmap='gray_r',vmin=0,vmax=10,interpolation='nearest')
            ax.set(aspect='equal',title='H60 3D truth' if col==0 else '0 (uniform init)' if nodes[col-1]==0 else str(nodes[col-1])+' iterations')
            ax.set_xticks([]);ax.set_yticks([])
            if col==0:ax.set_ylabel(system+'\n'+label+'\n'+units,fontsize=9)
    fig.colorbar(color,ax=axes,shrink=.4,label='gamma density / actual emitted-source background density, fixed 0-10')
    heading='EHE full-4pi Geant4 5e10; '+kind if len(routes)==3 else 'Geant4 5e10 / Geant4 5e9 / matrix-Poisson 5e9 / JSCC; '+kind
    fig.suptitle(heading+'\nEHE 0-200 / JSCC 0-10000: separate iteration ranges; crop0, no smoothing, no fitted gain\nDifferent dose/model/background budgets; same column is not equal convergence',fontsize=12)
    return fig


def sphere_dashboard(rows,routes,key):
    fig,axes=plt.subplots(math.ceil(len(routes)/3),3,figsize=(19,4.25*math.ceil(len(routes)/3)),layout='constrained')
    values=[float(v[key]) for v in rows if v[key]!=''];low,high=min(values),max(values);pad=max((high-low)*.06,.01)
    for ax,(system,c,label,_) in zip(axes.flat,routes):
        rr=[v for v in rows if v['system']==system and v['channel']==c]
        for d in sorted({int(v['diameter_mm']) for v in rr}):
            r=sorted([v for v in rr if int(v['diameter_mm'])==d],key=lambda v:int(v['iteration']))
            ax.plot([int(v['iteration']) for v in r],[float(v[key]) if v[key]!='' else np.nan for v in r],color=prior.COLORS[d],label=str(d)+' mm')
        ax.set(title=system+' / '+label,xlabel='actual iterations',ylabel=key.upper(),xlim=(0,10000 if system=='JSCC' else 200),ylim=(low-pad,high+pad));ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=3)
    fig.suptitle('Actual 3D sphere ROI '+key.upper()+'; all saved frames; independent iteration ranges and different doses; init CNR undefined')
    return fig


def native_dashboard(rows,metrics,routes):
    fig,axes=plt.subplots(len(metrics),len(SYSTEMS),figsize=(26,len(metrics)*2.7+1),layout='constrained',squeeze=False)
    for row,(key,label,units) in enumerate(metrics):
        values=[float(v[key]) for v in rows if v[key]!=''];low,high=min(values),max(values);pad=max((high-low)*.06,.01)
        for col,system in enumerate(SYSTEMS):
            ax=axes[row,col]
            for i,(_,c,caption,_) in enumerate([r for r in routes if r[0]==system]):
                rr=sorted([v for v in rows if v['system']==system and v['channel']==c],key=lambda v:int(v['iteration']))
                ax.plot([int(v['iteration']) for v in rr],[float(v[key]) if v[key]!='' else np.nan for v in rr],label=caption,color=prior.ROUTE_COLORS[i])
            ax.set(title=system+' / '+key,xlabel='actual iterations',ylabel='fraction' if units=='%' else units,xlim=(0,10000 if system=='JSCC' else 200),ylim=(low-pad,high+pad));ax.grid(alpha=.2)
            if row==0:ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Complete native 120mm metrics; physical density units, different doses, shared vertical scales; separate iteration ranges')
    return fig


def main():
    nr,sr,routes,selected,source,views,scales,collection,inputs,native,spheres=scientific_data()
    OUT.mkdir(parents=True,exist_ok=False);plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':9,'axes.unicode_minus':False})
    table(OUT/'transport_native_metrics.csv',native);table(OUT/'transport_sphere_metrics.csv',spheres)
    table(OUT/'reference_native_metrics.csv',[v for v in nr if v['system']!=SYSTEM]);table(OUT/'reference_sphere_metrics.csv',[v for v in sr if v['system']!=SYSTEM])
    def save(fig,name):fig.savefig(OUT/name,dpi=120);plt.close(fig)
    for kind in views:save(gallery(kind,routes[:3],selected,source,views),'transport_'+kind+'.png')
    save(gallery('mip72',routes,selected,source,views),'overall_mip72_15_routes.png')
    for key in ('crc','cnr'):save(sphere_dashboard(sr,routes,key),key+'_full_trajectories.png')
    save(native_dashboard(nr,prior.METRICS[:7],routes),'density_noise_curves.png')
    save(native_dashboard(nr,prior.METRICS[7:],routes),'integral_position_curves.png')
    fixed=np.load(DATA/'results/formal/fixed_cross_background.npy');workers=np.load(DATA/'counts/worker_counts.npz')
    sums={name:workers[name].reshape(20,50,2312).sum(axis=(1,2)) for name in workers.files if name.startswith('CntStat_')}
    table(OUT/'counts_by_view.csv',[dict(view=i+1,actual218=int(sums['CntStat_218'][i]),actual440=int(sums['CntStat_440'][i]),
        actual218from218=int(sums['CntStat_218_from218'][i]),actual218from440=int(sums['CntStat_218_from440'][i]),
        actual440from218=int(sums['CntStat_440_from218'][i]),actual440from440=int(sums['CntStat_440_from440'][i]),
        estimated_fixed_cross_from_own440200=float(fixed[:,i].astype(float).sum())) for i in range(20)])
    peaks=[]
    for c in CHANNELS:
        for d in sorted({v['diameter_mm'] for v in spheres if v['channel']==c}):
            rr=[v for v in spheres if v['channel']==c and v['diameter_mm']==d and v['iteration']>0];peak=max(rr,key=lambda v:v['cnr']);last=next(v for v in rr if v['iteration']==200)
            peaks.append(dict(channel=c,diameter_mm=d,peak_cnr_iteration=peak['iteration'],peak_cnr=peak['cnr'],final200_cnr=last['cnr'],final200_crc=last['crc']))
    table(OUT/'transport_cnr_peaks.csv',peaks)
    report=dict(passed=True,inputs_sha256=inputs,density_scales=scales,actual_source_dose=50000000000,
        primary_counts=collection['primary_counts'],dose_kind='Actual full-4pi transported primary photons; no hemisphere multiplier',
        windows=collection['window_counts'],tagged_counts=collection['tagged_counts'],data_kind='Geant4_transport',
        actual_cross_fraction=collection['tagged_counts']['CntStat_218_from440']/collection['window_counts']['218'],
        fixed_cross_background_total=float(fixed.astype(float).sum()),fixed_cross_background_source='This 5e10 acquisition own final440 single200',
        physical_calibration_claim=False,nodes={s:prior.NODES['JSCC'] if s=='JSCC' else NODES for s in SYSTEMS},
        transport_native_rows=len(native),transport_sphere_rows=len(spheres),routes=[dict(system=s,channel=c,label=l) for s,c,l,_ in routes],
        source_roi='Existing H60 3D truth and authoritative fractional 3D sphere masks',crop=0,smoothing_sigma=0,fitted_gain=False,
        display_range=[0,10],colormap='gray_r',mip_z_mm=[-36,36],metrics_z_mm=[-60,60],
        display_clipping=[dict(system=s,channel=c,iteration=i,display_grid_pixels_above10=int(np.count_nonzero(im>10))) for (s,c,i),im in selected.items()],
        zero_iteration='SHA-bound all-ones initialization; sums two; not a saved frame',cnr_peaks=peaks,
        limits=['Existing response discrepancy is preserved; user continued original method; no new physical calibration claimed',
            '5e10 versus 5e9 includes a tenfold dose difference; not an isolated algorithm comparison',
            'Model-Poisson data are a distinct generated realization and not Geant4 observations',
            'EHE218 background budget is own440200; JSCC218 background budget is prior44010000',
            'Materials, detector coverage, truncation and response approximations differ',
            'Each system retains its full iteration range; same iteration or column is not equal convergence',
            'CNR peaks are descriptive for one acquisition; not a fitted stopping policy or device-performance estimate'])
    write(OUT/'comparison_report.json',report);shutil.copy2(__file__,OUT/Path(__file__).name)
    write(OUT/'artifact_manifest.json',dict(files=hashes(OUT)))
    print('EHE_5E10_ATLAS_READY',OUT,flush=True)


if __name__=='__main__':main()
