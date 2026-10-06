"""Verified six-channel 10000 histories, native metrics and explicit H60 3D truth."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze_nema_result import digest,xy_interpolator,weighted_mean
from mip_projection import axial_mip
from energy_full10000_v5_contract import STUDY,CHANNELS,load_contract,verify_checkpoints

HERE=Path(__file__).resolve().parent
ITERATIONS=(100,500,1000,2000,5000,10000)
ENERGIES={c:218 if c=='218_SinglePhoton_CrossTalkCorrected' else
          'sum' if c in ('440SinglePlus218Single','440SingleComptonPlus218Single') else 440 for c in CHANNELS}


def table(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def compare(result,output,payload,summary_path):
    summary=json.loads(summary_path.read_text())
    if not summary.get('passed') or not summary.get('full_six_imaging_completed') or summary.get('mode')!='formal':
        raise ValueError('Actual completed six-channel 10000 formal acceptance is required')
    contract=payload/'contract.json';cfg=load_contract(contract,'formal',10000,50)
    if summary['contract_sha256']!=digest(contract):raise ValueError('Formal summary contract differs')
    receipt=summary['models']['continuous_energy'];v=json.loads((result/'verification.json').read_text())
    for name,sha in receipt['sha256'].items():
        p=result.parent/'allocation.txt' if name=='allocation.txt' else result/name
        if digest(p)!=sha:raise ValueError('Fetched actual acceptance bytes changed: '+name)
    if (not v['passed'] or v['mode']!='formal' or v['iterations']!=10000 or v['save_step']!=50
        or v['contract_sha256']!=digest(contract) or v['accepted_events']!=483743
        or v['model']!='continuous_energy' or len(v['outputs'])!=6):
        raise ValueError('Six complete verified 10000 histories required')
    run=json.loads((result/'run_manifest.json').read_text())
    if (run['helper_sha256']!=cfg['files'] or run['channels']!=list(CHANNELS)
        or run['input_sha256']!=cfg['input_sha256'] or run['factor_payload_sha256']!=cfg['factor_payload_sha256']
        or run['sensitivity_sha256']!=cfg['files']['continuous_energy_Sensi_full']
        or run['geometry_sha256']!=cfg['whole_geometry_sha256']):
        raise ValueError('Frozen image/source/S/geometry identity differs')
    g=np.load(payload/'whole_geometry.npz');coords=g['coordinates_mm'];active=g['active_indices']
    if len(active)!=78920 or len(coords)!=132040 or not np.isin(g['ellipse_fraction'],[0,1]).all():
        raise ValueError('Whole-cell basis required')
    histories={}
    for item in v['outputs']:
        channel=item['channel'];p=result/f'Image_{channel}_history.float32'
        if p.stat().st_size!=200*78920*4 or digest(p)!=item['sha256']['history']:raise ValueError('Actual history differs')
        histories[channel]=np.memmap(p,'<f4',mode='r',shape=(200,78920))
        if not np.isfinite(histories[channel]).all() or np.any(histories[channel]<0):raise ValueError('Invalid history')
    checkpoints=verify_checkpoints(result,histories,active,digest(contract),'formal')
    if checkpoints!=v['checkpoints'] or len(checkpoints)!=600:raise ValueError('Actual fetched checkpoints differ')
    source_path=HERE/'reports/NEMA_Body_H60/manifest.json';source=json.loads(source_path.read_text())
    truth_path=HERE/'generated/NEMA_Body_H60/truth_3mm.npz'
    if digest(truth_path)!=source['truth_sha256']:raise ValueError('Authoritative H60 3D truth changed')
    t=np.load(truth_path);x,y,z=(t[k+'_mm'] for k in 'xyz')
    if t['activity_440_zyx'].shape!=(40,100,168) or not np.allclose(coords[::3301,2],z):
        raise ValueError('3D truth and polar z geometry differ')
    sphere_sum=sum(t[f'sphere_{d}_fraction_zyx'] for d in (10,13,17,22,28,37))
    bgfraction=np.maximum(t['body_fraction_zyx']-t['lung_fraction_zyx']-sphere_sum,0)
    background=(bgfraction>=.99)&(np.abs(z[:,None,None])<=25.5)
    interpolate=xy_interpolator(coords[:3301,:2],x,y)
    def image(values):
        full=np.zeros(132040,np.float32);full[active]=values
        return interpolate(full.reshape(40,3301))
    collection=json.loads((payload/'transport_collection.json').read_text())
    photons={218:collection['primary_counts'][0],440:collection['primary_counts'][1]}
    photons['sum']=sum(photons.values())
    bg_density={e:photons[e]/source['relative_activity_integral_mm3'][str(e)] for e in (218,440)}
    bg_density['sum']=bg_density[218]+bg_density[440]
    truth={e:t[f'activity_{e}_zyx'] for e in (218,440)}
    truth['sum']=(bg_density[218]*truth[218]+bg_density[440]*truth[440])/bg_density['sum']
    # One fixed 440 JSCC iteration-2000 scale. Transfer across energies using
    # actual emitted photons / truth integral; no independent per-image fits.
    reference=float(image(histories['440_SinglePlusCompton'][39])[background].mean())
    if not np.isfinite(reference) or reference<=0:raise ValueError('Fixed background scale is invalid')
    scales={c:reference*bg_density[ENERGIES[c]]/bg_density[440] for c in CHANNELS}
    xx,yy=np.meshgrid(x,y);masks={}
    for s in source['spheres']:
        d=int(s['diameter_mm']);cx,cy,cz=s['center_mm']
        masks[d]=(((xx-cx)**2+(yy-cy)**2<=(d/2+25)**2)[None]
                  &(np.abs(z[:,None,None]-cz)<=d/2+3)&(bgfraction>=.99))
    volumes=(g['cell_volume_mm3']*g['ellipse_fraction'])[active]
    outside=np.abs(coords[active,2])>30
    rows=[];spheres=[];selected={};history_sha={}
    for c,h in histories.items():
        energy=ENERGIES[c];history_sha[c]=digest(result/f'Image_{c}_history.float32')
        for index,values in enumerate(h):
            iteration=(index+1)*50;im=image(values);bg=im[background]
            mean=float(bg.mean());std=float(bg.std(ddof=1));integral=float(values.astype(float)@volumes)
            if mean<=0 or integral<=0:raise ValueError('Empty image/background')
            peak=int(values.argmax());order=np.argsort(values)
            rows.append(dict(channel=c,iteration=iteration,max_density=float(values[peak]),
                peak_background_ratio=float(values[peak]/mean),peak_x_mm=float(coords[active[peak],0]),
                peak_y_mm=float(coords[active[peak],1]),peak_z_mm=float(coords[active[peak],2]),
                p99_density=float(np.quantile(values,.99)),p999_density=float(np.quantile(values,.999)),
                volume_weighted_p999_density=float(np.interp(.999,np.cumsum(volumes[order])/volumes.sum(),values[order])),
                background_mean=mean,background_cv=std/mean,total_integral=integral,
                integral_recovery=integral/photons[energy],source_z_leakage=float(values[outside].astype(float)@volumes[outside]/integral),
                tiny_mass_fraction='not_applicable_whole_cells'))
            for s in source['spheres']:
                own=s['hot_energy_keV'];d=int(s['diameter_mm'])
                if energy!='sum' and energy!=own:continue
                local=im[masks[d]];local_mean=float(local.mean());local_std=float(local.std(ddof=1))
                if local.size<50 or local_mean<=0 or local_std<=0:raise ValueError('Insufficient existing ROI')
                hot=weighted_mean(im,t[f'sphere_{d}_fraction_zyx'])
                nominal=9. if energy!='sum' else 10*bg_density[own]/bg_density['sum']-1
                if nominal<=0:raise ValueError('Combined nominal contrast not positive')
                spheres.append(dict(channel=c,iteration=iteration,energy_keV=own,diameter_mm=d,
                    hot_mean=hot,local_background_mean=local_mean,nominal_truth_contrast=nominal,
                    crc=(hot/local_mean-1)/nominal,cnr=(hot-local_mean)/local_std))
            if iteration in ITERATIONS:selected[c,iteration]=im/scales[c]
    output.mkdir(parents=True,exist_ok=False)
    table(output/'native_iteration_metrics.csv',rows);table(output/'sphere_iteration_metrics.csv',spheres)
    extent=(x[0]-1.5,x[-1]+1.5,y[0]-1.5,y[-1]+1.5);ix=int(np.argmin(abs(x)));iy=int(np.argmin(abs(y)))
    views={'axial':(lambda im:im[20],extent,'z=+1.5 mm'),
        'coronal':(lambda im:im[:,iy,:],(x[0]-1.5,x[-1]+1.5,-60,60),f'y={y[iy]:+.1f} mm'),
        'sagittal':(lambda im:im[:,:,ix],(y[0]-1.5,y[-1]+1.5,-60,60),f'x={x[ix]:+.1f} mm'),
        'mip72':(lambda im:axial_mip(im,z,8),extent,'central 72 mm MIP')}
    for kind,(select,ex,label) in views.items():
        fig,axes=plt.subplots(6,7,figsize=(23,17),layout='constrained')
        for row,c in enumerate(CHANNELS):
            for col,im in enumerate([truth[ENERGIES[c]]]+[selected[c,i] for i in ITERATIONS]):
                color=axes[row,col].imshow(select(im),origin='lower',extent=ex,cmap='gray_r',vmin=0,vmax=10,interpolation='nearest')
                axes[row,col].set(aspect='equal',title='truth' if col==0 else f'iter {ITERATIONS[col-1]}')
                axes[row,col].tick_params(labelsize=7)
                if col==0:axes[row,col].set_ylabel(c,fontsize=8)
        fig.colorbar(color,ax=axes,shrink=.6,label='Density / fixed iteration-2000 440 JSCC scale, energy weights from actual photons')
        fig.suptitle('NEMA H60 legacy 5e9 continuous-energy full six outputs: '+label+'; no smoothing, crop0')
        fig.savefig(output/f'iterations_{kind}.png',dpi=120);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,(channel,label) in enumerate((('440_ComptonOnly','440 SC Compton'),
        ('440SingleComptonPlus218Single','JSCC 218+440 gamma-density sum'))):
        for col,im in enumerate((truth[ENERGIES[channel]],selected[channel,2000],selected[channel,10000])):
            color=axes[row,col].imshow(axial_mip(im,z,8),origin='lower',extent=extent,
                cmap='gray_r',vmin=0,vmax=10,interpolation='nearest')
            axes[row,col].set(aspect='equal',title=('truth','iter 2000','iter 10000')[col])
            if col==0:axes[row,col].set_ylabel(label)
    fig.colorbar(color,ax=axes,shrink=.7,label='Density / fixed background scale, actual energy weights')
    fig.suptitle('Legacy 5e9: central 72 mm MIP; no smoothing, crop0')
    fig.savefig(output/'requested_mip72.png',dpi=150);plt.close(fig)
    for name,keys in [('spike_noise_leakage',('max_density','peak_background_ratio','background_cv','source_z_leakage')),
                      ('tail_integral',('p999_density','background_mean','total_integral','integral_recovery'))]:
        fig,axes=plt.subplots(6,4,figsize=(17,17),layout='constrained')
        for row,c in enumerate(CHANNELS):
            rr=[r for r in rows if r['channel']==c]
            for col,key in enumerate(keys):
                axes[row,col].plot([r['iteration'] for r in rr],[r[key] for r in rr]);axes[row,col].axvline(2000,color='gray',ls='--')
                axes[row,col].set(title=c+'\n'+key,xlabel='iteration');axes[row,col].grid(alpha=.2)
        fig.savefig(output/f'{name}_curves.png',dpi=120);plt.close(fig)
    fig,axes=plt.subplots(6,2,figsize=(13,17),layout='constrained')
    for row,c in enumerate(CHANNELS):
        diameters=sorted({r['diameter_mm'] for r in spheres if r['channel']==c})
        for d in diameters:
            rr=[r for r in spheres if r['channel']==c and r['diameter_mm']==d]
            for col,key in enumerate(('crc','cnr')):
                axes[row,col].plot([r['iteration'] for r in rr],[r[key] for r in rr],label=f'{d}mm')
                axes[row,col].set(title=c+' '+key,xlabel='iteration');axes[row,col].grid(alpha=.2)
        axes[row,0].legend(fontsize=7)
    fig.savefig(output/'crc_cnr_curves.png',dpi=120);plt.close(fig)
    evolution={}
    for c in CHANNELS:
        baseline=next(r for r in rows if r['channel']==c and r['iteration']==2000)
        final=next(r for r in rows if r['channel']==c and r['iteration']==10000)
        costs=[]
        for s in (r for r in spheres if r['channel']==c and r['iteration']==10000):
            old=next(r for r in spheres if r['channel']==c and r['iteration']==2000 and r['diameter_mm']==s['diameter_mm'])
            costs.append(dict(diameter_mm=s['diameter_mm'],crc_change_percentage_points=100*(s['crc']-old['crc']),
                crc_cost_over_5_percentage_points=old['crc']-s['crc']>.05))
        evolution[c]=dict(iteration_2000=baseline,iteration_10000=final,
            max_density_ratio_10000_to_2000=final['max_density']/baseline['max_density'],
            peak_background_ratio_10000_to_2000=final['peak_background_ratio']/baseline['peak_background_ratio'],crc_costs=costs)
    joint=list(csv.DictReader((payload/'calibration/independent_joint_categories.csv').open()))
    report=dict(study=STUDY,job=summary['job'],through_iteration=10000,selected_iterations=ITERATIONS,frames_per_channel=200,
        truth_sha256=digest(truth_path),truth_manifest_sha256=digest(source_path),contract_sha256=digest(contract),
        formal_summary_sha256=digest(summary_path),history_sha256=history_sha,geometry_sha256=digest(payload/'whole_geometry.npz'),
        fixed_display_scales=scales,expected_background_gamma_density=bg_density,crop=0,smoothing_sigma=0,mip_z_mm=[-36,36],
        interpolation='3mm XY barycentric display/ROI only; peak/integral/high quantiles from native polar cells',
        unresolved_independent_joint_categories=sum(r['adequate']=='False' for r in joint if r['model']=='continuous_energy'),
        evolution=evolution,angular_10000_available=False,combined_units='sum of 218 and 440 gamma densities, not Ac225 activity')
    (output/'comparison_report.json').write_bytes((json.dumps(report,indent=2,allow_nan=False)+'\n').encode())
    shutil.copy2(__file__,output/Path(__file__).name)
    text=['# 完整六路10000次图集与曲线','',f"作业{summary['job']}：六路各200帧/600阶段检查点严格验收。",'',
          '使用本实验真实H60双能3D球体；gray_r、无平滑、crop0，轴/冠/矢及中央72mm MIP。',
          '固定440 JSCC第2000次背景尺度，各能量按实际发射数/真值积分转移尺度；组合真值按对应背景gamma密度加权。',
          'native指标覆盖120mm全活动域，f<0.1不适用。组合CRC按自身混合真值对比度归一，组合不代表Ac225活度。',
          '本轮没有角度核10000次配对，报告只比较连续核2000→10000演化；独立联合类别统计不足仍未判定。','']
    for p in sorted(output.glob('*.png')):text.append(f'![{p.stem}]({p.name})')
    text+=['','[原生200帧指标](native_iteration_metrics.csv) · [球体CRC/CNR](sphere_iteration_metrics.csv) · [科学摘要](comparison_report.json)']
    (output/'README.md').write_bytes(('\n'.join(text)+'\n').encode('utf-8'))
    files={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in output.iterdir() if p.is_file()}
    (output/'artifact_manifest.json').write_bytes((json.dumps(dict(study=STUDY,job=summary['job'],files=files),indent=2)+'\n').encode())
    print('ENERGY_FULL10000_COMPARISON_GENERATED',output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=int,required=True);a=p.parse_args()
    report=HERE/'reports/NEMA_Body_H60'/STUDY;data=HERE/'generated'/STUDY
    compare(data/'formal_results'/str(a.job)/'continuous_energy',report/f'comparison_{a.job}',
            data/'formal_payload',report/'formal_summary.json')
