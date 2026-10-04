"""Matched, unfiltered NEMA 5e9 baseline/cut3 figures and all-frame metrics."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_nema_result import digest,xy_interpolator,weighted_mean
from mip_projection import axial_mip

HERE=Path(__file__).resolve().parent
CHANNELS=('440_ComptonOnly','440_SinglePlusCompton')
SELECTED=(100,500,1000,2000,5000,10000)


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--through-iteration',type=int,choices=(2000,10000),default=10000)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    baseline=HERE/'generated/RemoteResults/NEMA_Body_H60_5e9_1644876'
    verification=json.loads((args.result/'verification.json').read_text())
    interim=args.through_iteration==2000
    expected_mode='interim' if interim else 'formal'
    if not verification['passed'] or verification['mode']!=expected_mode: raise ValueError('Verified result/snapshot required')
    manifest_name='checkpoint_manifest.json' if interim else 'run_manifest.json'
    manifest_hash='snapshot_manifest_sha256' if interim else 'run_manifest_sha256'
    if digest(args.result/manifest_name)!=verification[manifest_hash]:
        raise ValueError('Verified run manifest changed')
    run=json.loads((args.result/manifest_name).read_text())
    selected_iterations=tuple(i for i in SELECTED if i<=args.through_iteration)
    frames=args.through_iteration//50
    baseline_run=json.loads((baseline/'run_manifest.json').read_text())
    cfg_path=HERE/'response_mismatch_cut3_v1.json'; cfg=json.loads(cfg_path.read_text())
    if digest(cfg_path)!=verification['config_sha256']: raise ValueError('Frozen study config changed')
    if run['input_sha256']!=baseline_run['input_sha256']: raise ValueError('Inputs differ')
    truth_path=HERE/'generated/NEMA_Body_H60/truth_3mm.npz'
    source_manifest=json.loads((HERE/'reports/NEMA_Body_H60/manifest.json').read_text())
    if digest(truth_path)!=source_manifest['truth_sha256']: raise ValueError('Actual source truth changed')
    t=np.load(truth_path)
    x,y,z=(t[f'{a}_mm'] for a in 'xyz')
    truth=t['activity_440_zyx']
    all_spheres=sum(t[f'sphere_{d}_fraction_zyx'] for d in (10,13,17,22,28,37))
    bg_fraction=np.maximum(t['body_fraction_zyx']-t['lung_fraction_zyx']-all_spheres,0)
    background=(bg_fraction>=.99)&(np.abs(z[:,None,None])<=25.5)
    g=np.load(HERE/'generated/Geometry/geometry.npz')
    coords=g['coordinates_mm']; active=g['active_indices']
    volumes=(g['cell_volume_mm3']*g['ellipse_fraction'])[active]
    tiny=g['ellipse_fraction'][active]<.1; outside_z=np.abs(coords[active,2])>30
    sampler=xy_interpolator(coords[:3301,:2],x,y)
    xx,yy=np.meshgrid(x,y); ellipse=(xx/250)**2+(yy/150)**2<=1
    masks={}; spheres={}
    for diameter in (13,22,37):
        cx,cy,cz=next(s['center_mm'] for s in source_manifest['spheres'] if s['diameter_mm']==diameter)
        masks[diameter]=(((xx-cx)**2+(yy-cy)**2<=(diameter/2+25)**2)[None]
            &(np.abs(z[:,None,None]-cz)<=diameter/2+3)&(bg_fraction>=.99))
        spheres[diameter]=t[f'sphere_{diameter}_fraction_zyx']
    def volume_image(values):
        full=np.zeros(132040,np.float32); full[active]=values
        image=sampler(full.reshape(40,3301)); image[:,~ellipse]=0
        return image
    rows=[]; sphere_rows=[]; selected={}; normalizers={}; input_hashes={}
    for channel in CHANNELS:
        baseline_history=baseline/f'Image_{channel}_history.float32'
        if baseline_history.stat().st_size!=200*82040*4: raise ValueError('Baseline history size differs')
        integrity=json.loads((HERE/'reports/NEMA_Body_H60_5e9_1644876_integrity.json').read_text())
        old_expected=next(r['sha256']['history'] for r in integrity['outputs'] if r['channel']==channel)
        if digest(baseline_history)!=old_expected: raise ValueError('Baseline history hash differs')
        old=np.memmap(baseline_history,'<f4',mode='r',shape=(200,82040))
        normalizers[channel]=float(volume_image(old[-1])[background].mean())
        for group,root in [('baseline',baseline),('cut3',args.result)]:
            path=root/f'Image_{channel}_history.float32'
            group_frames=200 if group=='baseline' else frames
            if path.stat().st_size!=group_frames*82040*4: raise ValueError('History size differs')
            sha=digest(path); input_hashes[group+'/'+channel]=sha
            if group=='cut3' and sha!=next(r['sha256']['history'] for r in verification['outputs'] if r['channel']==channel):
                raise ValueError('Verified new history changed')
            history=np.memmap(path,'<f4',mode='r',shape=(group_frames,82040))
            for frame,values in enumerate(history[:frames]):
                if not np.isfinite(values).all() or np.any(values<0): raise ValueError('Invalid history')
                iteration=(frame+1)*50
                image=volume_image(values); bg=image[background]
                mean=float(bg.mean()); std=float(bg.std(ddof=1))
                integral=float(values.astype(np.float64)@volumes)
                peak=int(values.argmax()); order=np.argsort(values)
                p999=float(np.interp(.999,np.cumsum(volumes[order])/volumes.sum(),values[order]))
                row={'group':group,'channel':channel,'iteration':iteration,'max_density':float(values[peak]),
                    'peak_background_ratio':float(values[peak]/mean),'peak_x_mm':float(coords[active[peak],0]),
                    'peak_y_mm':float(coords[active[peak],1]),'peak_z_mm':float(coords[active[peak],2]),
                    'p99_density':float(np.quantile(values,.99)),'p999_density':float(np.quantile(values,.999)),
                    'volume_weighted_p999_density':p999,'background_mean':mean,'background_cv':std/mean,
                    'tiny_mass_fraction':float(values[tiny].astype(np.float64)@volumes[tiny]/integral),
                    'source_z_leakage':float(values[outside_z].astype(np.float64)@volumes[outside_z]/integral),
                    'total_integral':integral,'integral_recovery':integral/3530946267}
                rows.append(row)
                for diameter in (13,22,37):
                    local=image[masks[diameter]]; local_mean=float(local.mean()); local_std=float(local.std(ddof=1))
                    hot=weighted_mean(image,spheres[diameter])
                    sphere_rows.append({'group':group,'channel':channel,'iteration':iteration,'diameter_mm':diameter,
                        'hot_mean':hot,'local_background_mean':local_mean,'crc':(hot/local_mean-1)/9,
                        'cnr':(hot-local_mean)/local_std})
                if iteration in selected_iterations: selected[group,channel,iteration]=image/normalizers[channel]
    write_csv(args.output/'native_iteration_metrics.csv',rows)
    write_csv(args.output/'sphere_iteration_metrics.csv',sphere_rows)
    extent=(x[0]-1.5,x[-1]+1.5,y[0]-1.5,y[-1]+1.5)
    for kind in ('center','mip72'):
        fig,axes=plt.subplots(4,len(selected_iterations)+1,figsize=(22,11),layout='constrained')
        for row,(channel,group) in enumerate((c,gp) for c in CHANNELS for gp in ('baseline','cut3')):
            images=[truth]+[selected[group,channel,i] for i in selected_iterations]
            for col,image in enumerate(images):
                panel=image[20] if kind=='center' else axial_mip(image,z,8)
                im=axes[row,col].imshow(panel,origin='lower',extent=extent,cmap='gray_r',vmin=0,vmax=10,interpolation='nearest')
                axes[row,col].set(xlim=(-252,252),ylim=(-150,150),aspect='equal',title='truth' if col==0 else f'iter {selected_iterations[col-1]}')
                if col==0: axes[row,col].set_ylabel(channel+'\n'+group,fontsize=9)
                axes[row,col].tick_params(labelsize=7)
        fig.colorbar(im,ax=axes,shrink=.7,label='Density / common baseline final-background mean')
        fig.suptitle(f'NEMA 5e9: baseline vs cut3; {"z=+1.5 mm" if kind=="center" else "central 72 mm axial MIP"}; no smoothing'+('; INTERIM 2000, not final' if interim else ''))
        fig.savefig(args.output/f'iterations_{kind}.png',dpi=120); plt.close(fig)
    ix=int(np.argmin(abs(x))); iy=int(np.argmin(abs(y)))
    fig,axes=plt.subplots(2,12,figsize=(26,7),layout='constrained')
    for row,channel in enumerate(CHANNELS):
        for group_index,(label,image) in enumerate([('truth',truth),('baseline',selected['baseline',channel,args.through_iteration]),('cut3',selected['cut3',channel,args.through_iteration])]):
            panels=[(image[20],extent),(image[:,iy,:],(-252,252,-60,60)),
                    (image[:,:,ix],(-150,150,-60,60)),(axial_mip(image,z,8),extent)]
            for kind,(panel,ex) in enumerate(panels):
                ax=axes[row,group_index*4+kind]
                im=ax.imshow(panel,origin='lower',extent=ex,cmap='gray_r',vmin=0,vmax=10,interpolation='nearest')
                ax.set_title(label+' '+('axial','coronal','sagittal','MIP72')[kind],fontsize=8); ax.tick_params(labelsize=6)
        axes[row,0].set_ylabel(channel)
    fig.colorbar(im,ax=axes,shrink=.7); fig.savefig(args.output/'final_multiplanar.png',dpi=125); plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    for row,channel in enumerate(CHANNELS):
        for group,style in [('baseline','--'),('cut3','-')]:
            rr=[r for r in rows if r['channel']==channel and r['group']==group]
            for col,key in enumerate(('peak_background_ratio','background_cv','tiny_mass_fraction','source_z_leakage')):
                axes[row,col].plot([r['iteration'] for r in rr],[r[key] for r in rr],style,label=group)
                axes[row,col].set(title=channel+'\n'+key,xlabel='iteration'); axes[row,col].grid(alpha=.2)
            axes[row,0].set_yscale('log')
        axes[row,0].legend()
    fig.savefig(args.output/'spike_noise_leakage_curves.png',dpi=145); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for row,channel in enumerate(CHANNELS):
        for group,style in [('baseline','--'),('cut3','-')]:
            for diameter,color in zip((13,22,37),('tab:blue','tab:orange','tab:green')):
                rr=[r for r in sphere_rows if r['channel']==channel and r['group']==group and r['diameter_mm']==diameter]
                for col,key in enumerate(('crc','cnr')):
                    axes[row,col].plot([r['iteration'] for r in rr],[r[key] for r in rr],style,color=color,label=f'{group} {diameter}mm')
                    axes[row,col].set(title=channel+' '+key,xlabel='iteration'); axes[row,col].grid(alpha=.2)
        axes[row,0].legend(fontsize=8)
    fig.savefig(args.output/'crc_cnr_curves.png',dpi=145); plt.close(fig)
    verdict={}
    for channel in CHANNELS:
        old=next(r for r in rows if r['channel']==channel and r['group']=='baseline' and r['iteration']==args.through_iteration)
        new=next(r for r in rows if r['channel']==channel and r['group']=='cut3' and r['iteration']==args.through_iteration)
        reductions={key:1-new[key]/old[key] for key in ('max_density','peak_background_ratio')}
        costs=[]
        for diameter in (13,22,37):
            a=next(r['crc'] for r in sphere_rows if r['channel']==channel and r['group']=='baseline' and r['iteration']==args.through_iteration and r['diameter_mm']==diameter)
            b=next(r['crc'] for r in sphere_rows if r['channel']==channel and r['group']=='cut3' and r['iteration']==args.through_iteration and r['diameter_mm']==diameter)
            costs.append({'diameter_mm':diameter,'crc_change':b-a,'cost_over_5_percentage_points':a-b>.05})
        verdict[channel]={'extreme_spike_improved':None if interim else all(v>=.5 for v in reductions.values()),'reductions':reductions,'crc_costs':costs,
                          'baseline_final':old,'cut3_final':new}
    observations=[]
    for channel,item in verdict.items():
        old=item['baseline_final']; new=item['cut3_final']; change=item['reductions']
        observations.append(f"{channel}，第{args.through_iteration}次：最大密度下降{100*change['max_density']:.2f}%，"
            f"峰/背景比下降{100*change['peak_background_ratio']:.2f}%；背景CV {old['background_cv']:.4f}→{new['background_cv']:.4f}，"
            f"源外轴向质量{100*old['source_z_leakage']:.2f}%→{100*new['source_z_leakage']:.2f}%。")
        observations.append('13/22/37mm球CRC变化（百分点）：'+', '.join(
            f"{c['diameter_mm']}mm {100*c['crc_change']:+.2f}"+('，恢复损失超过5个百分点' if c['cost_over_5_percentage_points'] else '')
            for c in item['crc_costs'])+'。')
        if not interim:
            observations.append('达到预先固定的极端尖峰工作判据；结论首先限定为这些事件显著影响局部尖峰。'
                if item['extreme_spike_improved'] else '未达到预先固定的工作判据；本次3σ失配子集不足以解释主要尖峰，不排除其他响应模型问题。')
    report={'study':cfg['study'],'truth_sha256':digest(truth_path),'input_history_sha256':input_hashes,
        'verification_sha256':digest(args.result/'verification.json'),'common_display_normalizers':normalizers,
        'selected_iterations':selected_iterations,'interim_only':interim,'through_iteration':args.through_iteration,
        'mip':'central 72 mm; 8 layers excluded per end; other metrics all 40 layers',
        'roi':'same NEMA 3D fractional spheres/local backgrounds and global |z|<=25.5mm background as baseline',
        'display':'full ellipse; gray_r; 0..10; no smoothing; one baseline final-background scale per channel for both groups/all frames',
        'verdict':verdict,'observations':observations}
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    scripts=args.output/'scripts'; scripts.mkdir(exist_ok=True)
    for name in ('compare_response_mismatch.py','analyze_nema_result.py','mip_projection.py'):
        shutil.copy2(HERE/name,scripts/name)
    (args.output/'README.md').write_text('# NEMA 5e9：基线与3σ失配事件删除对照\n\n'+
        ('**第2000次中期观察，不作正式结论，不改变阈值或提前停止。**\n\n' if interim else '')+
        '两组使用同一真实球体体模、输入及初值，删除组匹配独立灵敏度。完整指标使用120mm，MIP使用中央72mm。'
        '所有图像未平滑，共同色标使用各通道基线最终背景均值，两个组及全部迭代固定同一尺度。\n\n'+
        '\n\n'.join(observations)+'\n\n'+
        '完整原始数值见comparison.json及两份CSV；工作判据是最大密度与峰/背景比均下降至少50%，CRC损失超过5个百分点单独标记。\n\n'
        '![中心层迭代](iterations_center.png)\n\n![中央72mm MIP](iterations_mip72.png)\n\n'
        '![多平面](final_multiplanar.png)\n\n![尖峰噪声泄漏](spike_noise_leakage_curves.png)\n\n![CRC/CNR](crc_cnr_curves.png)\n')
    print(json.dumps(verdict,indent=2))


if __name__=='__main__': main()
