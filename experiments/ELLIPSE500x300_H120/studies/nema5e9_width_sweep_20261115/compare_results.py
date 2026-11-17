"""Strictly delivered three-width comparison. No reconstruction or image smoothing."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,csv,hashlib,html,json,shutil,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from prepare import ROOT,EXP,DATA,REPORT,sha,read,write
sys.path.insert(0,str(EXP))
from analyze_nema_result import xy_interpolator
from nema_roi_policy import build_masks,measure,POLICY_ID

OUT=ROOT/'Analysis_Plots/research/nema5e9_width_sweep_20261115'
SELECTED=(100,500,1000,2000,5000,10000)
COLORS={100:'#176ca3',80:'#13916c',60:'#c25b32'}

def sources():
    f=read(REPORT/'freeze_r3.json');payload=DATA/f['payload_dir']
    paths=[EXP/'generated/NEMA_Body_H60/truth_3mm.npz',EXP/'reports/NEMA_Body_H60/manifest.json',
           EXP/'nema_body_h60_config.json',EXP/'nema_roi_policy.py',EXP/'analyze_nema_result.py',
           payload/'whole_geometry.npz',payload/'original_iteration_10000.float32',
           ROOT/'Analysis_Plots/research/nema_sc_jscc_he_5e9_5e10_20261114/data/iteration_metrics.csv',
           ROOT/'Analysis_Plots/research/nema_sc_jscc_he_5e9_5e10_20261114/data/sphere_metrics.csv',
           ROOT/'Analysis_Plots/results/NEMA/5e9/440keV/data/metadata.json']
    return {str(p.resolve()):sha(p) for p in paths}

def context():
    manifest=read(EXP/'reports/NEMA_Body_H60/manifest.json');cfg=read(EXP/'nema_body_h60_config.json')
    truthfile=EXP/'generated/NEMA_Body_H60/truth_3mm.npz'
    assert sha(truthfile)==manifest['truth_sha256']=='2612f0ed6839f9460722711e1017a10102e83adf77cf715d5c2553cfaec948af'
    with np.load(truthfile) as z:truth={n:z[n].copy() for n in z.files}
    masks=build_masks(truth,manifest,cfg);assert masks['background'].sum()==97604
    frozen=read(REPORT/'freeze_r3.json');p=DATA/frozen['payload_dir']/'whole_geometry.npz'
    assert sha(p)==frozen['files']['whole_geometry.npz']
    with np.load(p) as z:geo={n:z[n].copy() for n in z.files}
    assert np.allclose(geo['coordinates_mm'][::3301,2],truth['z_mm'],atol=1e-6)
    x,y,z=[truth[a+'_mm'] for a in 'xyz'];slab=np.abs(z)<36;assert slab.sum()==24
    meta=read(ROOT/'Analysis_Plots/results/NEMA/5e9/440keV/data/metadata.json')
    r=next(r for r in meta['routes'] if r['method']=='JSCC')
    assert r['source_job']==1669255 and sum(r['primary_counts'])==5000000000 and r['primary_counts'][1]==3530946267
    norm=r['primary_counts'][1]/manifest['relative_activity_integral_mm3']['440']
    assert np.isclose(norm,r['normalization_density'],rtol=1e-12)
    return dict(truth=truth,manifest=manifest,masks=masks,geo=geo,slab=slab,norm=norm,
        query=xy_interpolator(geo['coordinates_mm'][:3301,:2],x,y),extent=[x[0]-1.5,x[-1]+1.5,y[0]-1.5,y[-1]+1.5])

def analyze(v,c,percent,iteration):
    assert v.shape==(78920,) and np.isfinite(v).all() and (v>=0).all()
    full=np.zeros(132040,np.float32);full[c['geo']['active_indices']]=v
    q=c['query'](full.reshape(40,3301)).astype(float)/c['norm']
    negative=np.minimum(q,0);relative=float(np.linalg.norm(negative)/max(1e-300,np.linalg.norm(q)))
    limit=64*np.finfo(np.float64).eps*max(1.,float(v.max())/c['norm'])
    assert float(-negative.min())<=limit and relative<=1e-12
    q=np.maximum(q,0);bg,spheres=measure(q,c['masks'],c['manifest'],440)
    row=dict(width_percent=percent,iteration=iteration,**bg,BV_percent=100*bg['background_cv'],
        absolute_NRMSE_3D=float(np.linalg.norm(q-c['truth']['activity_440_zyx'])/np.linalg.norm(c['truth']['activity_440_zyx'])),
        interpolation_negative_roundoff_L2=relative,maximum_density=float(q.max()),
        MIP_fraction_above10=float((q[c['slab']].max(0)>10).mean()))
    for s in spheres:
        s.update(width_percent=percent,iteration=iteration,BV_percent=row['BV_percent'],CRC_percent=100*s['crc'],CNR=s['cnr'])
        if s['region_kind']=='hot':assert np.isclose(s['cnr'],900*s['crc']/row['BV_percent'],rtol=1e-12)
        else:assert np.isclose(s['cnr'],-100*s['crc']/row['BV_percent'],rtol=1e-12)
    return row,spheres,q[c['slab']].max(0)

def csvwrite(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def self_test():
    c=context();f=read(REPORT/'freeze_r3.json');p=DATA/f['payload_dir']/'original_iteration_10000.float32'
    assert sha(p)==f['files'][p.name]
    row,spheres,mip=analyze(np.fromfile(p,'<f4'),c,100,10000)
    prior=ROOT/'Analysis_Plots/research/nema_sc_jscc_he_5e9_5e10_20261114/data'
    with (prior/'iteration_metrics.csv').open(encoding='utf-8-sig',newline='') as h:
        old=next(r for r in csv.DictReader(h) if r['dose']=='5e9' and r['method']=='JSCC' and r['energy_keV']=='440' and r['iteration']=='10000')
    errors={k:abs(row[k]-float(old[k]))/max(1.,abs(float(old[k]))) for k in ('background_mean','background_std','BV_percent')}
    with (prior/'sphere_metrics.csv').open(encoding='utf-8-sig',newline='') as h:
        oldrows={int(r['diameter_mm']):r for r in csv.DictReader(h) if r['dose']=='5e9' and r['method']=='JSCC' and r['energy_keV']=='440' and r['iteration']=='10000'}
    for s in spheres:
        for k in ('sphere_mean','crc','cnr'):
            oldvalue=float(oldrows[s['diameter_mm']][k]);errors[str(s['diameter_mm'])+'_'+k]=abs(s[k]-oldvalue)/max(1.,abs(oldvalue))
    assert max(errors.values())<=2e-6,errors
    result=dict(passed=True,reference_job=1669255,actual_original_final_frame=True,ROI_policy=POLICY_ID,
        maximum_scaled_error=max(errors.values()),errors=errors,new_results_fabricated=False)
    write(REPORT/'comparison_pipeline_test.json',result);print(json.dumps(result))

def draw(ax,mip,c,title,vmax):
    h=ax.imshow(mip,origin='lower',extent=c['extent'],cmap='gray_r',vmin=0,vmax=vmax,interpolation='nearest')
    ax.set_title(title,fontsize=11);ax.set_aspect('equal');ax.set_xticks([-200,0,200]);ax.set_yticks([-100,0,100]);ax.tick_params(labelsize=8)
    return h

def render(c,bg,spheres,mips):
    figures=[]
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=190,facecolor='white')
        plt.close(fig);figures.append(name)
    truth=c['truth']['activity_440_zyx'][c['slab']].max(0)
    fullscale=max(10.,max(float(x.max()) for x in mips.values()))
    for suffix,vmax in [('',10.),('_unclipped',fullscale)]:
        fig,axes=plt.subplots(1,4,figsize=(15,3.6),layout='constrained')
        draw(axes[0],truth,c,'Truth',vmax)
        for ax,p in zip(axes[1:],(100,80,60)):h=draw(ax,mips[(p,10000)],c,f'{p}% width · 10000',vmax)
        fig.colorbar(h,ax=axes.tolist(),label='Relative activity density',shrink=.7)
        fig.suptitle('NEMA 5e9 · 440 keV JSCC · central 72 mm MIP · no smoothing')
        save(fig,'final_comparison'+suffix)
        fig,axes=plt.subplots(3,7,figsize=(20,8),layout='constrained')
        for row,p in enumerate((100,80,60)):
            draw(axes[row,0],truth,c,f'{p}% · truth',vmax)
            for ax,i in zip(axes[row,1:],SELECTED):h=draw(ax,mips[(p,i)],c,str(i),vmax)
        fig.colorbar(h,ax=axes.ravel().tolist(),label='Relative activity density',shrink=.65)
        fig.suptitle('NEMA 5e9 · JSCC width comparison · actual saved iterations')
        save(fig,'iteration_comparison'+suffix)
    for kind,diameters in [('hot',(13,22,37)),('cold',(10,17,28))]:
        fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
        for column,d in enumerate(diameters):
            for p in (100,80,60):
                rows=[r for r in spheres if r['width_percent']==p and r['diameter_mm']==d]
                assert len(rows)==200
                for row,key in enumerate(('CRC_percent','CNR')):
                    ax=axes[row,column];ax.plot([r['BV_percent'] for r in rows],[r[key] for r in rows],color=COLORS[p],label=f'{p}%',lw=1.6)
                    ax.scatter(rows[-1]['BV_percent'],rows[-1][key],color=COLORS[p],s=26)
                    ax.set(xlabel='BV (%)',ylabel='CRC (%)' if row==0 else 'CNR',title=f'{d} mm {kind} sphere');ax.grid(alpha=.2)
            axes[0,column].legend()
        fig.suptitle('JSCC width · every saved frame 50–10000 · dots = 10000')
        save(fig,'crc_cnr_bv_'+kind)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for ax,key,title in zip(axes,('BV_percent','background_mean','absolute_NRMSE_3D'),('BV (%)','Background mean (truth = 1)','3D NRMSE (no fitted gain)')):
        for p in (100,80,60):
            rows=[r for r in bg if r['width_percent']==p];ax.plot([r['iteration'] for r in rows],[r[key] for r in rows],label=f'{p}%',color=COLORS[p])
        ax.set(xlabel='Iteration',ylabel=title);ax.grid(alpha=.2);ax.legend()
    save(fig,'iteration_metrics')
    return figures

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test();return
    reg=read(REPORT/'formal_registration_resume_r5.json');job=reg['job'];receipt=REPORT/f'strict_delivery_{job}.json'
    accepted=read(receipt);assert accepted['passed'] and accepted['frames']==600 and accepted['inherited_checkpoint_count']==199
    assert not (OUT/'data/completed.json').exists(),'Comparison already generated; use existing accepted artifacts'
    for name in ('data','figures','scripts'):(OUT/name).mkdir(parents=True,exist_ok=True)
    c=context();background=[];sphere_rows=[];mips={};inputs=sources();inputs[str(receipt)]=sha(receipt)
    base=DATA/f'delivered_{job}/result'
    for percent in (100,80,60):
        branch=base/f'width{percent}';done=read(branch/'completed.json');history=branch/'Image_JSCC_history.float32'
        assert sha(history)==done['files'][history.name] and history.stat().st_size==200*78920*4
        inputs[str(history)]=sha(history);hist=np.memmap(history,'<f4',mode='r',shape=(200,78920))
        for index in range(200):
            i=(index+1)*50;bg,ss,mip=analyze(np.asarray(hist[index]),c,percent,i)
            background.append(bg);sphere_rows.extend(ss)
            if i in SELECTED:mips[(percent,i)]=mip
        print('COMPARISON_WIDTH_COMPLETE',percent,flush=True)
    csvwrite(OUT/'data/background_metrics.csv',background);csvwrite(OUT/'data/sphere_metrics.csv',sphere_rows)
    endpoints=[r for r in sphere_rows if r['iteration']==10000];csvwrite(OUT/'data/final_sphere_metrics.csv',endpoints)
    choice=read(base/'width100/benchmark_choice.json');performance=[]
    baseline=choice['times']['baseline']
    for mode,seconds in choice['times'].items():performance.append(dict(kind='paired_50_step_GPU_measurement',mode=mode,width_percent=100,iterations=50,median_seconds=seconds,seconds_per_iteration=seconds/50,speedup=baseline/seconds,source_job='1813383'))
    for p in (100,80,60):
        rank=read(base/f'width{p}/rank0.json');performance.append(dict(kind='formal_solve_only',mode=rank['chosen'],width_percent=p,iterations=rank['new_iterations'],median_seconds=None,seconds_per_iteration=rank['seconds_per_iteration'],speedup=None,source_job=job))
    csvwrite(OUT/'data/performance.csv',performance)
    figures=render(c,background,sphere_rows,mips)
    np.savez_compressed(OUT/'data/display_arrays.npz',truth=c['truth']['activity_440_zyx'][c['slab']].max(0),**{f'width{p}_iter{i}':v for (p,i),v in mips.items()})
    shutil.copyfile(Path(__file__),OUT/'scripts'/Path(__file__).name)
    write(OUT/'data/input_sha256.json',inputs)
    metadata=dict(job=job,scales=[100,80,60],frames=600,energy_keV=440,total_dual_energy_gamma=5000000000,
        normalization_density=c['norm'],normalization='Actual 440 primary photons / truth activity integral; no fitted brightness',
        truth_sha256=c['manifest']['truth_sha256'],ROI_policy=POLICY_ID,background_voxels=97604,
        smoothing=False,crop_XY=0,central_MIP_mm=72,historical_comparison=accepted['historical_comparison'],
        inherited_iterations_100=9950,original_benchmark_job='1813383',benchmark_choice=choice,
        figures=figures,numerical_QA_passed=True,visual_QA_passed=False,physical_calibration_passed=False)
    write(OUT/'data/metadata.json',metadata)
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>NEMA 5e9 JSCC宽度对比</title><style>body{font:16px system-ui;max-width:1500px;margin:24px auto;padding:0 20px;color:#223}img{width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}</style><h1>NEMA 5e9 · 440 keV JSCC · 100% / 80% / 60%</h1>'
    page+='<p>固定原真实事件、选择和随机实现；三档统一缩放电子学、位置、材料残差宽度，保持材料均值。80%/60%各自重算独立匹配S。各10000次、200帧。5e9指双能gamma总发射量，不是225Ac衰变数。</p>'
    page+='<p>原3D真值与球ROI、97604体素共同背景；无平滑、无亮度拟合、无跨能量求和。主图范围0–10，另附共同全幅色标图。72 mm仅限定MIP轴向范围，完整XY和120 mm重建支持保持。</p>'
    page+='<p>100%继承1813383的50–9950检查点，从9950续算最后50次；历史1669255差异按用户决定保留为诊断，不改判旧门槛通过。数值验收、图形人工检查、物理认证分别记录。</p>'
    page+='<table><tr><th>宽度</th><th>球径</th><th>CRC (%)</th><th>CNR</th><th>BV (%)</th></tr>'
    for r in endpoints:
        if r['region_kind']=='hot':page+=f"<tr><td>{r['width_percent']}%</td><td>{r['diameter_mm']} mm</td><td>{r['CRC_percent']:.2f}</td><td>{r['CNR']:.3f}</td><td>{r['BV_percent']:.2f}</td></tr>"
    page+='</table>'
    for name in figures:page+=f'<h2>{html.escape(name)}</h2><a href="figures/{name}.png"><img src="figures/{name}.png"></a>'
    page+='<h2>工程测速</h2><p>原任务中完整真实输入、同一8×5090、三次50步配对：合并同步传输约5.90倍；传输字节和原32事件算术块不变。预取比同步只快约1.88%，因此正式使用同步合并。此倍数属于受控迭代耗时，不把准备/校准/排队时间计入，也不把不同宽度模型的耗时混为工程加速。</p><p><a href="data/performance.csv">实际性能表</a> · <a href="data/sphere_metrics.csv">所有球指标</a> · <a href="data/background_metrics.csv">背景与3D指标</a> · <a href="data/metadata.json">身份与限制</a></p><p>图形已生成，人工视觉QA仍待检查；不预设缩窄响应改善图像，不据此宣称物理标定通过。</p></html>'
    (OUT/'index.html').write_text(page,encoding='utf8')
    write(OUT/'data/artifact_sha256.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('artifact_sha256.json','completed.json')})
    write(OUT/'data/completed.json',dict(job=job,frames=600,figures_generated=True,numerical_QA_passed=True,visual_QA_passed=False))
    print('WIDTH_COMPARISON_GENERATED',OUT/'index.html',flush=True)

if __name__=='__main__':main()
