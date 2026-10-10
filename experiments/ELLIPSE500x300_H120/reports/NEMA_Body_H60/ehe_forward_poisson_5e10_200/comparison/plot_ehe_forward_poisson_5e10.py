"""Actual accepted synthetic EHE200, transport EHE200 and JSCC10000, fixed source scales."""
import shutil, math
import plot_ehe_5e10 as transport_reference
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ehe_common import CHANNELS, digest, read, write, hashes, verify_files
from ehe_forward_poisson_5e10_workflow import DATA, REPORT
from compare_ehe_5e9 import table
from analyze_nema_result import xy_interpolator, weighted_mean
import compare_ehe_jscc_full_trajectories as prior

OUT=REPORT/'comparison'
NODES=(0,10,20,50,100,150,200)
SYSTEM='EHE matrix+Poisson 5e10'
SYSTEMS=(SYSTEM,'EHE Geant4 5e10','EHE matrix+Poisson','EHE','JSCC')


def scientific_data():
    proof=read(REPORT/'formal_summary.json')
    if not proof['passed'] or proof['iterations']!=200 or proof['frames_per_channel']!=20:raise ValueError('Strict actual formal200 acceptance required')
    result=DATA/'results/formal';verify_files(result,proof['files'])
    f=read(REPORT/'freeze.json');payload=DATA/f['payload_dir'];verify_files(payload,f['sha256'])
    collection=read(REPORT/'generation_summary.json')
    truth,meta,coords,active,vol,scales,histories,native,spheres,provenance,init,source_hashes,old_report=prior.inputs()
    nr,sr=prior.extended_metrics(native,spheres,coords,active,vol,old_report)
    scales[SYSTEM]={**collection['budget']['density_gamma_per_mm3']}
    scales[SYSTEM]['sum']=scales[SYSTEM]['218']+scales[SYSTEM]['440']
    x,y,z=[truth[k+'_mm'] for k in 'xyz'];interp=xy_interpolator(coords[:3301,:2],x,y)
    def image(v):
        full=np.zeros(132040,'<f4');full[active]=v
        return interp(full.reshape(40,3301))
    bgfraction=np.maximum(truth['body_fraction_zyx']-truth['lung_fraction_zyx']-sum(truth[f'sphere_{d}_fraction_zyx'] for d in prior.DIAMETERS),0)
    bg=(bgfraction>=.99)&(abs(z[:,None,None])<=25.5);xx,yy=np.meshgrid(x,y);masks={}
    for s in meta['spheres']:
        d=int(s['diameter_mm']);cx,cy,cz=s['center_mm']
        masks[d]=(((xx-cx)**2+(yy-cy)**2<=(d/2+25)**2)[None]&(abs(z[:,None,None]-cz)<=d/2+3)&(bgfraction>=.99))
    leak=abs(coords[active,2])>30;new_native=[];new_sphere=[]
    for c in CHANNELS:
        path=result/f'Image_{c}_history.float32';h=np.fromfile(path,'<f4').reshape(20,78920)
        histories[SYSTEM,c]=(h,10);e=prior.energy(c)
        for index in range(21):
            iteration=index*10;v=np.full(78920,prior.initialization(c),'<f4') if index==0 else h[index-1]
            im=image(v);b=im[bg];mean=float(b.mean());sd=float(b.std(ddof=1));integral=float(v.astype(float)@vol);peak=int(v.argmax());order=np.argsort(v)
            expected=5e10 if e=='sum' else collection['budget']['expected_primary_photons'][e]
            row=dict(system=SYSTEM,channel=c,iteration=iteration,max_density=float(v[peak]),peak_background_ratio=float(v[peak]/mean),
                peak_x_mm='' if not iteration else float(coords[active[peak],0]),peak_y_mm='' if not iteration else float(coords[active[peak],1]),peak_z_mm='' if not iteration else float(coords[active[peak],2]),
                p99_density=float(np.quantile(v.astype(np.float64),.99)),p999_density=float(np.quantile(v.astype(np.float64),.999)),volume_weighted_p999_density=float(np.interp(.999,np.cumsum(vol[order])/vol.sum(),v[order])),
                background_mean=mean,background_cv=sd/mean,total_integral=integral,integral_recovery=integral/expected,
                source_z_leakage=float(v[leak].astype(float)@vol[leak]/integral))
            new_native.append(row)
            for s in meta['spheres']:
                own=str(s['hot_energy_keV']);d=int(s['diameter_mm'])
                if e!='sum' and e!=own:continue
                local=im[masks[d]];lm=float(local.mean());ls=float(local.std(ddof=1));hot=weighted_mean(im,truth[f'sphere_{d}_fraction_zyx'])
                nominal=9 if e!='sum' else 10*scales[SYSTEM][own]/scales[SYSTEM]['sum']-1
                new_sphere.append(dict(system=SYSTEM,channel=c,iteration=iteration,energy_keV=int(own),diameter_mm=d,hot_mean=hot,local_background_mean=lm,crc=(hot/lm-1)/nominal,cnr='' if iteration==0 else (hot-lm)/ls))
    nr.extend(new_native);sr.extend(new_sphere)
    labels={CHANNELS[1]:'218 corrected single',CHANNELS[0]:'440 single',CHANNELS[2]:'dual single sum'}
    routes=tuple((SYSTEM,c,labels[c],labels[c]) for c in (CHANNELS[1],CHANNELS[0],CHANNELS[2]))+prior.ROUTES
    selected={};source={}
    for system in (SYSTEM,'EHE','JSCC'):
        for e in ('218','440'):source[system,e]=truth[f'activity_{e}_zyx']
        source[system,'sum']=(scales[system]['218']*source[system,'218']+scales[system]['440']*source[system,'440'])/scales[system]['sum']
    for system,c,_,_ in routes:
        h,step=histories[system,c]
        for i in NODES if system!= 'JSCC' else prior.NODES['JSCC']:
            v=np.full(78920,prior.initialization(c),'<f4') if i==0 else h[i//step-1]
            selected[system,c,i]=image(v)/scales[system][prior.energy(c)]
    ix,iy=int(abs(x).argmin()),int(abs(y).argmin())
    extent=(x[0]-1.5,x[-1]+1.5,y[0]-1.5,y[-1]+1.5)
    views={'mip72':(lambda im:prior.axial_mip(im,z,8),extent,'X/Y mm'),
           'axial':(lambda im:im[20],extent,f'X/Y mm, z={z[20]:g}'),
           'coronal':(lambda im:im[:,iy,:],(x[0]-1.5,x[-1]+1.5,-60,60),f'X/Z mm, y={y[iy]:g}'),
           'sagittal':(lambda im:im[:,:,ix],(y[0]-1.5,y[-1]+1.5,-60,60),f'Y/Z mm, x={x[ix]:g}')}
    source_hashes.update(synthetic_formal_summary=digest(REPORT/'formal_summary.json'),synthetic_collection=digest(DATA/'counts/collection.json'),
        plotting_source=digest(__file__),prior_plotting_source=digest(prior.__file__),synthetic_histories={c:digest(result/f'Image_{c}_history.float32') for c in CHANNELS})
    rn,rs,rr,rselected,rsource,_,rscales,_,rinputs,_,_=transport_reference.scientific_data()
    nr=rn+new_native;sr=rs+new_sphere;routes=routes[:3]+rr
    selected.update(rselected);source.update(rsource);scales.update(rscales)
    source_hashes['accepted_Geant4_5e10_and_prior_reference_inputs']=rinputs
    return nr,sr,routes,selected,source,views,scales,collection,source_hashes,new_native,new_sphere


def gallery(kind,routes,selected,source,views):
    selector,extent,units=views[kind]
    height=7.4 if len(routes)==3 and kind in ('coronal','sagittal') else 2.4*len(routes)+1.4
    fig,axes=plt.subplots(len(routes),8,figsize=(24,height),squeeze=False)
    fig.subplots_adjust(left=.13,right=.945,bottom=.04,top=1-1.25/height,wspace=.08,hspace=.36)
    for row,(system,c,label,_) in enumerate(routes):
        nodes=prior.NODES['JSCC'] if system=='JSCC' else NODES
        for col,im in enumerate([source[system,prior.energy(c)]]+[selected[system,c,i] for i in nodes]):
            ax=axes[row,col];color=ax.imshow(selector(im),origin='lower',extent=extent,cmap='gray_r',vmin=0,vmax=10,interpolation='nearest',aspect='equal')
            ax.set_title('H60 3D truth' if col==0 else '0 (uniform init)' if nodes[col-1]==0 else str(nodes[col-1])+' iterations')
            ax.set_xticks([]);ax.set_yticks([])
        bounds=axes[row,0].get_position()
        fig.text(.012,(bounds.y0+bounds.y1)/2,system+'\n'+label+'\n'+units,ha='left',va='center',fontsize=9)
    color_axis=fig.add_axes([.961,.3,.009,.36])
    fig.colorbar(color,cax=color_axis,label='gamma density / emitted-source background; fixed 0-10')
    heading=('EHE matrix forward + independent Poisson, expected emitted dose5e10; ' if len(routes)==3 else 'Matrix-Poisson5e10 / Geant4 5e10 / Matrix-Poisson5e9 / Geant4 5e9 / JSCC; ')+kind
    fig.suptitle(heading+'\nEHE0-200 / JSCC0-10000: separate iteration ranges; crop0, no smoothing, no fitted gain\nModel-generated observations; fixed218 background from own final440; same column is not equal convergence',fontsize=12,y=1-.15/height)
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
    fig.suptitle('Actual 3D sphere ROI '+key.upper()+'; all saved frames; independent iteration ranges; iteration0 CNR undefined')
    return fig


def native_dashboard(rows,metrics,routes):
    fig,axes=plt.subplots(len(metrics),len(SYSTEMS),figsize=(30,len(metrics)*2.7+1),layout='constrained',squeeze=False)
    for row,(key,label,units) in enumerate(metrics):
        values=[float(v[key]) for v in rows if v[key]!=''];low,high=min(values),max(values);pad=max((high-low)*.06,.01)
        for col,system in enumerate(SYSTEMS):
            ax=axes[row,col]
            for i,(_,c,caption,_) in enumerate([r for r in routes if r[0]==system]):
                rr=sorted([v for v in rows if v['system']==system and v['channel']==c],key=lambda v:int(v['iteration']))
                ax.plot([int(v['iteration']) for v in rr],[float(v[key]) if v[key]!='' else np.nan for v in rr],label=caption,color=prior.ROUTE_COLORS[i])
            ax.set(title=system+' / '+key,xlabel='actual iterations',ylabel='fraction' if units=='%' else units,xlim=(0,10000 if system=='JSCC' else 200),ylim=(low-pad,high+pad));ax.grid(alpha=.2)
            if row==0:ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Complete native 120mm metrics; shared vertical scales, independent iteration ranges')
    return fig


def main():
    nr,sr,routes,selected,source,views,scales,collection,inputs,new_native,new_sphere=scientific_data()
    OUT.mkdir(parents=True,exist_ok=False);plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':9,'axes.unicode_minus':False})
    table(OUT/'synthetic_native_metrics.csv',new_native);table(OUT/'synthetic_sphere_metrics.csv',new_sphere)
    # Preserve prior rows verbatim in separate provenance tables; schemas differ at init.
    table(OUT/'reference_native_metrics.csv',[v for v in nr if v['system']!=SYSTEM]);table(OUT/'reference_sphere_metrics.csv',[v for v in sr if v['system']!=SYSTEM])
    def save(fig,name):fig.savefig(OUT/name,dpi=120,bbox_inches="tight",pad_inches=.12);plt.close(fig)
    synthetic=routes[:3]
    for kind in views:save(gallery(kind,synthetic,selected,source,views),'synthetic_'+kind+'.png')
    save(gallery('mip72',routes,selected,source,views),'overall_mip72_18_routes.png')
    for key in ('crc','cnr'):save(sphere_dashboard(sr,routes,key),key+'_full_trajectories.png')
    save(native_dashboard(nr,prior.METRICS[:7],routes),'density_noise_curves.png')
    save(native_dashboard(nr,prior.METRICS[7:],routes),'integral_position_curves.png')
    fixed=np.load(DATA/'results/formal/fixed_cross_background.npy');table(OUT/'counts_by_view.csv',[
        dict(view=i+1,expected218direct=collection['components']['A218']['expected_by_view'][i],sampled218direct=collection['components']['A218']['sampled_by_view'][i],
             expected440=collection['components']['A440']['expected_by_view'][i],sampled440=collection['components']['A440']['sampled_by_view'][i],
             expected_cross=collection['components']['C440to218']['expected_by_view'][i],sampled_cross=collection['components']['C440to218']['sampled_by_view'][i],
             estimated_fixed_cross_from_new440200=float(fixed[:,i].astype(float).sum())) for i in range(20)])
    peaks=[]
    for c in CHANNELS:
        for d in sorted({v['diameter_mm'] for v in new_sphere if v['channel']==c}):
            r=[v for v in new_sphere if v['channel']==c and v['diameter_mm']==d and int(v['iteration'])>0];peak=max(r,key=lambda v:v['cnr'])
            last=next(v for v in r if v['iteration']==200);peaks.append(dict(channel=c,diameter_mm=d,peak_cnr_iteration=peak['iteration'],peak_cnr=peak['cnr'],final200_cnr=last['cnr'],final200_crc=last['crc']))
    table(OUT/'synthetic_cnr_peaks.csv',peaks)
    report=dict(passed=True,inputs_sha256=inputs,density_scales=scales,expected_source_dose=50_000_000_000,
        dose_kind='Expected emitted photons, not transported primaries or detected counts',source_basis=collection['source_basis'],reconstruction_basis=collection['reconstruction_basis'],
        windows=collection['window_counts'],components=collection['components'],generated_cross_fraction=collection['generated_cross_fraction'],
        fixed_cross_background_total=float(fixed.astype(float).sum()),fixed_cross_background_source='This new 440 single final200; no oracle component background',
        noise_seeds=collection['noise_seeds'],model_data_physical_calibration_claim=False,
        nodes={s:prior.NODES['JSCC'] if s=='JSCC' else NODES for s in SYSTEMS},routes=[dict(system=s,channel=c,label=l) for s,c,l,_ in routes],synthetic_native_rows=len(new_native),synthetic_sphere_rows=len(new_sphere),
        source_roi='Actual H60 3D truth and existing 3D sphere fractional masks',crop=0,smoothing_sigma=0,fitted_gain=False,
        display_range=[0,10],colormap='gray_r',mip_z_mm=[-36,36],metrics_z_mm=[-60,60],
        display_clipping=[dict(system=system,channel=channel,iteration=iteration,display_grid_pixels_above10=int(np.count_nonzero(image>10)))
                          for (system,channel,iteration),image in selected.items()],
        zero_iteration='SHA-bound all-ones solver initialization; sums are two; not a saved frame',cnr_peaks=peaks,
        limits=['Matrix model data cannot independently test physical response; original transport HOLD is preserved',
                'Source uses original 3mm Cartesian stencil; reconstruction uses Polar basis; discretizations differ',
                'Single noise realization; peak CNR selection is descriptive, not a tuned stopping policy',
                'Matrix-Poisson versus transport image differences combine model differences and different noise realizations; not a causal mechanism estimate',
                'Same column/iteration is not same convergence; each system retains its own full iteration range',
                'No real-device claim; geometry/material/count and background-budget differences remain'])
    write(OUT/'comparison_report.json',report);shutil.copy2(__file__,OUT/Path(__file__).name)
    write(OUT/'artifact_manifest.json',dict(files=hashes(OUT)))
    print('SYNTHETIC_GALLERY_READY',OUT)


if __name__=='__main__':
    main()
