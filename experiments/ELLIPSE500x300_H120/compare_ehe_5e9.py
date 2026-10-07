"""Actual accepted EHE/JSCC H60 3D histories at fixed emitted-source density scales."""
import csv,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ehe_common import *
from analyze_nema_result import xy_interpolator,weighted_mean
import sys
sys.path.insert(0,str(ROOT))
from mip_projection import axial_mip

JSCC_CHANNELS=CHANNELS+('440_ComptonOnly','440_SinglePlusCompton','440SingleComptonPlus218Single')
ITERATIONS=(50,100,150,200)

def reference_evidence():
    """Obtain bounded read-only summaries; all new files stay in this experiment."""
    from ehe_5e9_workflow import connection,command,put_tree,q
    import hashlib,json
    local=DATA/'reference_evidence';source=HERE/'ehe_reference_evidence.py'
    expected={n:digest(HERE/n) for n in ('ehe_reference_evidence.py','ehe_common.py')}
    key=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()[:16]
    if (local/'measurement_summary.json').exists():
        proof=read(local/'measurement_summary.json');verify_files(local,proof['files'])
        if proof['source_sha256']!=digest(source):raise ValueError('Reference measurement implementation differs')
        return proof
    bundle=DATA/'reference_audit_releases'/key;bundle.mkdir(parents=True,exist_ok=True)
    for name in expected:
        dest=bundle/name
        if not dest.exists():shutil.copy2(HERE/name,dest)
        if digest(dest)!=expected[name]:raise ValueError('Immutable reference audit changed')
    remote=GPU_BASE+'/reference_evidence_'+key;code_remote=GPU_BASE+'/reference_audit_releases/'+key
    project=GPU_BASE.rsplit('/generated/',1)[0]
    registration=read(HERE/'reports/NEMA_Body_H60/compton_energy_probability_v5_5e9_full10000/formal_job.json')
    with connection('gpu') as c:
        command(c,'mkdir -p '+q(GPU_BASE+'/reference_audit_releases'))
        with c.open_sftp() as s:put_tree(s,bundle,code_remote)
        code='from pathlib import Path;from ehe_common import *;verify_files(Path('+repr(code_remote)+'),'+repr(expected)+')'
        command(c,'cd '+q(code_remote)+' && '+q(GPU_PYTHON)+' -c '+q(code))
        text=q(GPU_PYTHON)+' ehe_reference_evidence.py --release '+q(registration['release'])+' --factors '+q(project+'/generated/FactorsCalibrated')+' --inputs '+q(project+'/generated')+' --output '+q(remote)
        # A receipt permits a repeated read-only fetch; a partial audit is kept
        # visible for diagnosis rather than being overwritten.
        command(c,'cd '+q(code_remote)+' && if [ ! -f '+q(remote+'/measurement_summary.json')+' ]; then timeout --signal=TERM --kill-after=10s 1800s '+text+'; fi',1820)
        local.mkdir(parents=True,exist_ok=True)
        with c.open_sftp() as s:
            s.get(remote+'/measurement_summary.json',str(local/'measurement_summary.json'))
            proof=read(local/'measurement_summary.json')
            for name,sha in proof['files'].items():
                s.get(remote+'/'+name,str(local/name))
                if digest(local/name)!=sha:raise ValueError('Reference sensitivity evidence SHA differs')
    if not proof['passed'] or proof['reference_job']!=1669255 or proof['source_sha256']!=digest(source):raise ValueError('Actual reference measurement proof differs')
    return proof

def table(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def compare(job=None):
    if not (REPORT/'formal_summary.json').exists():raise ValueError('Actual accepted EHE200 results are not available yet')
    registered=read(REPORT/'formal_job.json')['job']
    if job is not None and int(job)!=registered:raise ValueError('Requested job differs from latest formal registration')
    proof=read(REPORT/'formal_summary.json')
    if not proof['passed'] or proof['mode']!='formal' or proof['iterations']!=200 or proof['frames_per_channel']!=20:raise ValueError('Actual EHE200 formal acceptance required')
    result=DATA/'results/formal';verify_files(result,proof['files'])
    reference=reference_evidence()
    jscc=HERE/'generated/compton_energy_probability_v5_5e9_full10000/formal_results/1669255/continuous_energy'
    previous=HERE/'reports/NEMA_Body_H60/compton_energy_probability_v5_5e9_full10000'
    jproof=read(previous/'formal_summary.json')
    if not jproof['passed'] or not jproof['full_six_imaging_completed'] or jproof['job']!=1669255:raise ValueError('Actual completed JSCC reference required')
    jverify=read(jscc/'verification.json')
    g=np.load(DATA/'payload/whole_geometry.npz');coords=g['coordinates_mm'];active=g['active_indices'];vol=g['cell_volume_mm3'][active];leak=np.abs(coords[active,2])>30
    truth=np.load(TRUTH);meta=read(TRUTH_META)
    if digest(TRUTH)!=meta['truth_sha256'] or digest(TRUTH)!=read(DATA/'payload/config.json')['truth_sha256']:raise ValueError('H60 3D truth changed')
    x,y,z=[truth[k+'_mm'] for k in 'xyz'];interp=xy_interpolator(coords[:3301,:2],x,y)
    def image(a):
        full=np.zeros(132040,'f4');full[active]=a
        return interp(full.reshape(40,3301))
    counts={'EHE':read(DATA/'transport/collection.json')['primary_counts'],
            'JSCC':read(HERE/'generated/compton_energy_probability_v5_5e9_full10000/formal_payload/transport_collection.json')['primary_counts']}
    scale={system:{e:primary[i]/meta['relative_activity_integral_mm3'][str(e)] for i,e in enumerate((218,440))} for system,primary in counts.items()}
    for system in scale:scale[system]['sum']=scale[system][218]+scale[system][440]
    from ehe_reference_evidence import sensitivity_statistics
    workers=np.load(DATA/'transport/worker_counts.npz');window_report={};sensitivity_report={}
    for e in (218,440):
        total=workers[f'CntStat_{e}'].reshape(20,10,2312).sum(axis=(1,2))
        cross=workers[f'CntStat_{e}_from{440 if e==218 else 218}'].reshape(20,10,2312).sum(axis=(1,2))
        window_report[str(e)]=dict(total=int(total.sum()),by_view=total.tolist(),other_primary_energy_counts=int(cross.sum()),
            actual_cross_fraction=float(cross.sum()/max(int(total.sum()),1)),actual_cross_fraction_by_view=(cross/np.maximum(total,1)).tolist())
    for name in RESPONSES:
        folder=DATA/'factor_evidence'/name;manifest=read(folder/'factor_manifest.json');s=folder/'S_active.float64'
        if digest(s)!=manifest['files']['S_active.float64']:raise ValueError('EHE sensitivity statistics source changed')
        sensitivity_report[name]=dict(EHE=sensitivity_statistics(np.fromfile(s,'<f8'),vol),JSCC=reference['sensitivity'][name])
    predicted_path=jscc/'PredictedCntStat_218_From440.float32'
    if digest(predicted_path)!=jverify['prediction_sha256']:raise ValueError('Frozen JSCC final440 cross prediction changed')
    jscc_predicted=np.fromfile(predicted_path,'<f4').reshape(10496,20)
    source={};histories={};history_sha={}
    for system,channels,folder,step,frames in (('EHE',CHANNELS,result,10,20),('JSCC',JSCC_CHANNELS,jscc,50,200)):
        source[system]={e:truth[f'activity_{e}_zyx'] for e in (218,440)}
        source[system]['sum']=(scale[system][218]*source[system][218]+scale[system][440]*source[system][440])/scale[system]['sum']
        for c in channels:
            path=folder/f'Image_{c}_history.float32'
            if path.stat().st_size!=frames*78920*4:raise ValueError('History byte shape differs')
            sha=digest(path)
            if system=='JSCC':
                v=next(v for v in jverify['outputs'] if v['channel']==c)
                if sha!=v['sha256']['history']:raise ValueError('Frozen JSCC history differs')
            histories[system,c]=(np.memmap(path,'<f4',mode='r',shape=(frames,78920)),step);history_sha[system+'/'+c]=sha
    bgfraction=np.maximum(truth['body_fraction_zyx']-truth['lung_fraction_zyx']-sum(truth[f'sphere_{d}_fraction_zyx'] for d in (10,13,17,22,28,37)),0)
    bg=(bgfraction>=.99)&(abs(z[:,None,None])<=25.5);xx,yy=np.meshgrid(x,y);masks={}
    for s in meta['spheres']:
        d=int(s['diameter_mm']);cx,cy,cz=s['center_mm']
        masks[d]=(((xx-cx)**2+(yy-cy)**2<=(d/2+25)**2)[None]&(abs(z[:,None,None]-cz)<=d/2+3)&(bgfraction>=.99))
    energy=lambda c:218 if c==CHANNELS[1] else 'sum' if c in (CHANNELS[2],'440SingleComptonPlus218Single') else 440
    rows=[];spheres=[];selected={}
    # Native metrics for all 20 EHE frames; preserved JSCC curves are linked below.
    for c in CHANNELS:
        h,step=histories['EHE',c];e=energy(c)
        for index,v in enumerate(h):
            im=image(v);b=im[bg];mean=float(b.mean());sd=float(b.std(ddof=1));integral=float(v.astype(float)@vol);peak=int(v.argmax());order=np.argsort(v)
            if mean<=0 or integral<=0:raise ValueError('Invalid EHE image statistics')
            rows.append(dict(system='EHE',channel=c,iteration=(index+1)*step,max_density=float(v[peak]),peak_background_ratio=float(v[peak]/mean),
                peak_x_mm=float(coords[active[peak],0]),peak_y_mm=float(coords[active[peak],1]),peak_z_mm=float(coords[active[peak],2]),
                p99_density=float(np.quantile(v,.99)),p999_density=float(np.quantile(v,.999)),volume_weighted_p999_density=float(np.interp(.999,np.cumsum(vol[order])/vol.sum(),v[order])),
                background_mean=mean,background_cv=sd/mean,total_integral=integral,integral_recovery=integral/(sum(counts['EHE'][:2]) if e=='sum' else counts['EHE'][0 if e==218 else 1]),
                source_z_leakage=float(v[leak].astype(float)@vol[leak]/integral)))
            for s in meta['spheres']:
                own=s['hot_energy_keV'];d=int(s['diameter_mm'])
                if e!='sum' and e!=own:continue
                local=im[masks[d]];lm=float(local.mean());ls=float(local.std(ddof=1))
                if local.size<50 or lm<=0 or ls<=0:raise ValueError('Existing sphere ROI inadequate')
                hot=weighted_mean(im,truth[f'sphere_{d}_fraction_zyx']);nominal=9 if e!='sum' else 10*scale['EHE'][own]/scale['EHE']['sum']-1
                spheres.append(dict(system='EHE',channel=c,iteration=(index+1)*step,energy_keV=own,diameter_mm=d,hot_mean=hot,local_background_mean=lm,crc=(hot/lm-1)/nominal,cnr=(hot-lm)/ls))
    for (system,c),(h,step) in histories.items():
        for iteration in ITERATIONS+((2000,10000) if system=='JSCC' else ()):
            im=image(h[iteration//step-1]);selected[system,c,iteration]=im/scale[system][energy(c)]
    output=REPORT/'comparison_200';output.mkdir(parents=True,exist_ok=False)
    table(output/'ehe_native_iteration_metrics.csv',rows);table(output/'ehe_sphere_iteration_metrics.csv',spheres)
    prior=previous/'comparison_1669255'
    for name in ('native_iteration_metrics.csv','sphere_iteration_metrics.csv'):
        shutil.copy2(prior/name,output/('jscc_'+name))
    with (output/'jscc_native_iteration_metrics.csv').open() as f:jr=[dict(r,system='JSCC') for r in csv.DictReader(f)]
    with (output/'jscc_sphere_iteration_metrics.csv').open() as f:js=[dict(r,system='JSCC') for r in csv.DictReader(f)]
    groups={'218':(('EHE',CHANNELS[1]),('JSCC',CHANNELS[1])),
        '440':(('EHE',CHANNELS[0]),('JSCC',CHANNELS[0]),('JSCC','440_ComptonOnly'),('JSCC','440_SinglePlusCompton')),
        'dual':(('EHE',CHANNELS[2]),('JSCC',CHANNELS[2]),('JSCC','440SingleComptonPlus218Single'))}
    extent=(x[0]-1.5,x[-1]+1.5,y[0]-1.5,y[-1]+1.5);ix=int(abs(x).argmin());iy=int(abs(y).argmin())
    views={'axial':(lambda im:im[20],extent),'coronal':(lambda im:im[:,iy,:],(x[0]-1.5,x[-1]+1.5,-60,60)),
           'sagittal':(lambda im:im[:,:,ix],(y[0]-1.5,y[-1]+1.5,-60,60)),'mip72':(lambda im:axial_mip(im,z,8),extent)}
    for category,pairs in groups.items():
        for kind,(select,ex) in views.items():
            fig,axes=plt.subplots(len(pairs),5,figsize=(18,3*len(pairs)),layout='constrained',squeeze=False)
            for row,(system,c) in enumerate(pairs):
                images=[source[system][energy(c)]]+[selected[system,c,i] for i in ITERATIONS]
                for col,im in enumerate(images):
                    color=axes[row,col].imshow(select(im),origin='lower',extent=ex,cmap='gray_r',vmin=0,vmax=10,interpolation='nearest')
                    axes[row,col].set(aspect='equal',title='H60 truth' if col==0 else f'iteration {ITERATIONS[col-1]}')
                    if col==0:axes[row,col].set_ylabel(system+'\n'+c,fontsize=8)
            fig.colorbar(color,ax=axes,shrink=.6,label='gamma density / actual emitted-source background density; no fitted gain')
            fig.suptitle(f'EHE / JSCC {category} {kind}; crop0, no smoothing; 218 background budgets differ')
            fig.savefig(output/f'{category}_{kind}.png',dpi=140);plt.close(fig)
    for kind,(select,ex) in views.items():
        pairs=[(s,c) for group in groups.values() for s,c in group if s=='JSCC']
        fig,axes=plt.subplots(len(pairs),3,figsize=(12,len(pairs)*2.8),layout='constrained')
        for row,(system,c) in enumerate(pairs):
            for col,im in enumerate([source[system][energy(c)],selected[system,c,2000],selected[system,c,10000]]):
                color=axes[row,col].imshow(select(im),origin='lower',extent=ex,cmap='gray_r',vmin=0,vmax=10,interpolation='nearest');axes[row,col].set(aspect='equal',title=('truth','JSCC2000','JSCC10000')[col])
                if col==0:axes[row,col].set_ylabel(c,fontsize=8)
        fig.colorbar(color,ax=axes,shrink=.6,label='source density scale');fig.savefig(output/f'jscc_long_reference_{kind}.png',dpi=140);plt.close(fig)
    for category,pairs in groups.items():
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for system,c in pairs:
            r=[v for v in (rows if system=='EHE' else jr) if v['channel']==c and int(v['iteration'])<=200];label=system+' '+c
            for ax,key in zip(axes.flat,('background_cv','max_density','peak_background_ratio','source_z_leakage','p999_density','integral_recovery')):
                ax.plot([int(v['iteration']) for v in r],[float(v[key]) for v in r],label=label);ax.set(title=key,xlabel='iteration');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=6);fig.savefig(output/f'{category}_native_curves.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
        for system,c in pairs:
            r=[v for v in (spheres if system=='EHE' else js) if v['channel']==c and int(v['iteration'])<=200]
            for d in sorted({int(v['diameter_mm']) for v in r}):
                rr=[v for v in r if int(v['diameter_mm'])==d]
                for ax,key in zip(axes,('crc','cnr')):ax.plot([int(v['iteration']) for v in rr],[float(v[key]) for v in rr],label=f'{system} {c} {d}mm');ax.set(title=key,xlabel='iteration');ax.grid(alpha=.2)
        axes[0].legend(fontsize=5);fig.savefig(output/f'{category}_crc_cnr.png',dpi=140);plt.close(fig)
    disclosure=dict(physical_units='gamma/mm3; sums are not Ac225 activity',background_budgets={'EHE':'440 single final200','JSCC':'440 single final10000'},
        density_scales=scale,primary_counts=counts,reference_jscc_job=1669255,selected_iterations=ITERATIONS,jscc_supplement=[2000,10000],
        actual_window_counts={'EHE':window_report,'JSCC':reference['actual_window_counts']},sensitivity_comparison=sensitivity_report,
        jscc_cross_diagnostic=dict(primary_tagged_fraction='unavailable in legacy observations',
            modeled_218_background_sum=float(jscc_predicted.astype(float).sum()),
            modeled_background_fraction=float(jscc_predicted.astype(float).sum()/reference['actual_window_counts']['218']['total']),
            prediction_source='accepted 440 single final10000; model prediction, not measured primary-tagged cross counts'),
        truth_sha256=digest(TRUTH),history_sha256=history_sha,formal_proof_sha256=digest(REPORT/'formal_summary.json'),crop=0,smoothing_sigma=0,
        mip_z_mm=[-36,36],whole_cell_tiny_mass_metric='not applicable',
        unknown_joint_categories=read(previous/'comparison_1669255/comparison_report.json')['unresolved_independent_joint_categories'],
        limits=['Same iteration does not imply same convergence','Different detector coverage/materials; truncation is not an algorithm effect','Two research vacuum-source models; no real-device claim'])
    write(output/'comparison_report.json',disclosure);shutil.copy2(REPORT/'projection_truncation.json',output/'projection_truncation.json');shutil.copy2(DATA/'physical/physical_gate.json',output/'physical_gate.json');shutil.copy2(DATA/'physical/physical_audit.csv',output/'physical_audit.csv')
    shutil.copy2(__file__,output/'compare_ehe_5e9.py')
    write(output/'artifact_manifest.json',dict(files=hashes(output)))
    print('EHE_COMPARISON_CREATED',output,'scientific and visual QA still required')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--job',type=int);a=p.parse_args();compare(a.job)
