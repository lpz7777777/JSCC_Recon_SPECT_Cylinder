"""Full-height, unscaled FOV120 Contrast closed-loop figures and HTML report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--rods', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root, out = args.inputs, args.output
    out.mkdir(parents=True, exist_ok=False)
    (out/'data').mkdir(); (out/'scripts').mkdir()
    shutil.copy2(__file__, out/'scripts'/Path(__file__).name)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    factor = root/'Factors/440keV_RotateNum20'
    coords = np.loadtxt(factor/'coor_polar_full.csv', delimiter=',')
    volume = np.fromfile(factor/'polar_cell_volume_mm3.float64', dtype='<f8')
    z = np.unique(coords[:, 2]); nz = len(z); nxy = len(coords)//nz
    assert len(coords) == 51240 and nz == 40 and nxy == 1281
    truth_path = root/'Truth_Contrast_1e9.npz'
    truth = np.load(truth_path)
    shutil.copy2(truth_path, out/'data'/truth_path.name)
    shutil.copy2(args.rods, out/'data/Contrast_truth.json')
    rods = json.loads(args.rods.read_text())['rods']
    histories, final = {}, {}
    source_paths = [truth_path, args.rods, factor/'coor_polar_full.csv', factor/'polar_cell_volume_mm3.float64']
    for kind in ('Noiseless', 'Poisson'):
        manifests = list((root/'ClosedLoop'/kind).rglob('run_manifest.json'))
        if len(manifests) != 1: raise ValueError('Ambiguous result')
        manifest = json.loads(manifests[0].read_text()); source_paths.append(manifests[0])
        for task in manifest['tasks']:
            energy = {'Direct440': 440, 'CrossTalkCorrected218': 218}.get(task['type'])
            if energy is None: continue
            fp = manifests[0].parent/task['output_file']; hp = manifests[0].parent/task['history_file']
            source_paths += [fp, hp]
            x = np.fromfile(fp, dtype='<f4'); h = np.fromfile(hp, dtype='<f4')
            assert x.size == len(coords) and h.size == 20*len(coords)
            h = h.reshape(20, -1)
            assert np.isfinite(h).all() and (h >= 0).all() and np.array_equal(x, h[-1])
            assert task['iterations'] == 1000 and task['save_iter_step'] == 50
            assert truth[f'rho{energy}'].shape == x.shape
            histories[kind, energy] = h; final[kind, energy] = x
    tri = Delaunay(coords[:nxy, :2]); axis = np.arange(-148.5, 150, 3)
    xx, yy = np.meshgrid(axis, axis)
    def cart(x):
        return np.stack([LinearNDInterpolator(tri, plane, fill_value=0)(xx, yy)
                         for plane in x.reshape(nz, nxy)])
    def image(ax, data, extent, vmax, title, cmap='gray_r', vmin=0):
        im = ax.imshow(data, origin='lower', extent=extent, aspect='equal',
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(title); ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)' if extent[-1] == 150 else 'z (mm)')
        return im
    figures = []
    def save(fig, name, title):
        fig.savefig(out/f'{name}.png', dpi=160, facecolor='white'); plt.close(fig)
        figures.append((name, title))
    zi = [int(np.argmin(abs(z-target))) for target in (-45, 0, 45)]
    for energy in (440, 218):
        t = truth[f'rho{energy}']; tc = cart(t); vmax = float(t.max())
        cubes = [tc, cart(final['Noiseless', energy]), cart(final['Poisson', energy])]
        labels = ['Truth', 'Noiseless, iter 1000', 'Poisson, iter 1000']
        fig, axs = plt.subplots(3, 4, figsize=(15, 11), layout='constrained')
        for row, (cube, label) in enumerate(zip(cubes, labels)):
            for col, index in enumerate(zi):
                im = image(axs[row, col], cube[index], (-150,150,-150,150), vmax,
                           f'{label}\nz={z[index]:g} mm')
            image(axs[row,3], cube.max(axis=1), (-150,150,-60,60), vmax, label+'\nCoronal MIP (over y)')
        fig.colorbar(im, ax=axs, shrink=.5, label='Emitted photon density (photons/mm³)')
        fig.suptitle(f'{energy} keV'+(' (218 reconstructed with fixed cross-talk background)' if energy==218 else '')+
                     '\nFull FOV; no smoothing, crop, or intensity fit; common truth color scale')
        save(fig, f'gallery_{energy}', f'{energy} keV：真值、无噪声与 Poisson 的三个轴位及全高 MIP')
        fig, axs = plt.subplots(3, 3, figsize=(13, 9), layout='constrained')
        for row, (cube, label) in enumerate(zip(cubes, labels)):
            for col, (panel, title) in enumerate(((cube[:,50,:], 'Coronal y=1.5 mm'),
                                                  (cube[:,:,50], 'Sagittal x=1.5 mm'),
                                                  (cube.max(axis=1), 'Coronal MIP'))):
                im = image(axs[row,col], panel, (-150,150,-60,60), vmax, label+'\n'+title)
                if col == 1: axs[row,col].set_xlabel('y (mm)')
        fig.colorbar(im, ax=axs, shrink=.6, label='photons/mm³')
        fig.suptitle(f'{energy} keV: full 120-mm longitudinal views; fixed planes can miss off-plane rods')
        save(fig, f'longitudinal_{energy}', f'{energy} keV：全高冠状位、矢状位与 MIP')
        fig, axs = plt.subplots(2, 5, figsize=(18, 6), layout='constrained')
        for row, kind in enumerate(('Noiseless','Poisson')):
            image(axs[row,0], tc.max(axis=1), (-150,150,-60,60), vmax, kind+'\nTruth')
            for col, iteration in enumerate((50,200,500,1000), 1):
                image(axs[row,col], cart(histories[kind,energy][iteration//50-1]).max(axis=1),
                      (-150,150,-60,60), vmax, f'Iteration {iteration}')
        fig.suptitle(f'{energy} keV: actual saved iterations, coronal MIP, common truth scale')
        save(fig, f'iterations_{energy}', f'{energy} keV：50、200、500、1000 次迭代演化')
        fig, axs = plt.subplots(2,3,figsize=(12,8),layout='constrained')
        for row, kind in enumerate(('Noiseless','Poisson')):
            error = cubes[row+1]-tc
            for col, index in enumerate(zi):
                im = image(axs[row,col],error[index],(-150,150,-150,150),vmax,
                           f'{kind}: reconstruction - truth\nz={z[index]:g} mm','RdBu_r',-vmax)
        fig.colorbar(im,ax=axs,shrink=.6,label='Density error (photons/mm³); blue = under-recovery')
        save(fig,f'error_{energy}',f'{energy} keV：同一绝对尺度的重建减真值误差')
    all_mask = np.maximum.reduce([truth[f'mask_rod_{i+1}'] for i in range(len(rods))])
    rows, summary = [], []
    def mean(x,w): return float(np.dot(x,w)/w.sum())
    for (kind, energy), frames in histories.items():
        t = truth[f'rho{energy}']
        for k,x in enumerate(frames):
            summary.append(dict(kind=kind,energy=energy,iteration=50*(k+1),
                                error=float(np.sqrt(np.dot((x-t)**2,volume)/np.dot(t*t,volume))),
                                recovery=float(np.dot(x,volume)/np.dot(t,volume))))
            for i,rod in enumerate(rods):
                if rod['energy'] != energy: continue
                hot = volume*truth[f'mask_rod_{i+1}']
                bg = volume*((np.hypot(coords[:,0],coords[:,1])<=120)&(all_mask==0)&
                             (abs(coords[:,2]-rod['center_mm'][2])<=rod['height_mm']/2))
                hm,bm = mean(x,hot),mean(x,bg); sd=np.sqrt(mean((x-bm)**2,bg))
                crc=(hm/bm-1)/(mean(t,hot)/mean(t,bg)-1)
                rows.append(dict(kind=kind,energy=energy,iteration=50*(k+1),rod=i+1,
                                 diameter=2*rod['radius_mm'],z=rod['center_mm'][2],
                                 crc=crc,cnr=(hm-bm)/sd if sd>0 else None))
    for name,data in [('rod_metrics',rows),('global_metrics',summary)]:
        with (out/'data'/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for energy,color in ((218,'tab:blue'),(440,'tab:orange')):
        for kind,style in (('Noiseless','-'),('Poisson','--')):
            data=[r for r in summary if r['energy']==energy and r['kind']==kind]
            for ax,key in zip(axs,('error','recovery')):
                ax.plot([r['iteration'] for r in data],[100*r[key] for r in data],style,color=color,label=f'{energy} {kind}')
    axs[0].set_ylabel('Volume-weighted relative L2 error (%)');axs[1].set_ylabel('Volume integral recovery (%)')
    axs[1].axhline(100,color='gray',lw=1)
    for ax in axs: ax.set_xlabel('Iteration');ax.grid(alpha=.2);ax.legend(fontsize=8)
    save(fig,'convergence','全局图像误差与总量恢复：投影闭合不等于空间恢复')
    for metric in ('crc','cnr'):
        fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for row,energy in enumerate((218,440)):
            diameters=sorted(set(r['diameter'] for r in rows if r['energy']==energy))
            for col,zc in enumerate((-45,0,45)):
                ax=axs[row,col]
                for diameter,color in zip(diameters,('tab:blue','tab:orange','tab:green')):
                    for kind,style in (('Noiseless','-'),('Poisson','--')):
                        data=[r for r in rows if r['energy']==energy and r['z']==zc and r['diameter']==diameter and r['kind']==kind]
                        ax.plot([r['iteration'] for r in data],[(100 if metric=='crc' else 1)*r[metric] for r in data],style,color=color,label=f'D{diameter:g} {kind}')
                ax.set_title(f'{energy} keV, rod z={zc} mm');ax.set_xlabel('Iteration');ax.set_ylabel('CRC (%)' if metric=='crc' else 'CNR')
                ax.axhline(0,color='gray',lw=.7);ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('Fractional-volume rod ROIs; local same-z background r<=120 mm excluding all rods')
        save(fig,metric+'_curves',('热柱 CRC' if metric=='crc' else '热柱 CNR')+'：直径、轴向位置、噪声和迭代的影响')
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    metadata=dict(crop_mm=0,filter_sigma=0,intensity_fit=False,colormap='gray_r',
                  interpolation='Linear in-plane only for display, no z interpolation; metrics on original polar cells',
                  axial_slice_centers_mm=[float(z[i]) for i in zi],source_sha256=hashes,
                  metric_note='CNR uses volume-weighted spatial population SD; CRC reference uses the same fractional ROI on truth')
    (out/'data/manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    sections=''.join(f'<section><h2>{title}</h2><a href="{name}.png"><img src="{name}.png"></a></section>' for name,title in figures)
    html='''<!doctype html><meta charset="utf-8"><title>FOV120 闭环结果</title>
<style>body{font:17px/1.65 system-ui;max-width:1450px;margin:35px auto;padding:0 25px;background:#fafafa;color:#20242a}img{width:100%;background:white}section{margin:45px 0}h1,h2{line-height:1.3}.note{padding:18px;background:#edf3fa;border-left:5px solid #376ea6}</style>
<h1>FOV120 双能单光子闭环：详细可视化</h1>
<div class="note">本报告为同一系统矩阵生成的无噪声 / Poisson 投影闭环，不是 Geant4 成像，也不含 Compton 六路结果。总发射预算 10⁹，1000 次 MLEM，每 50 次保存。218 使用重建 440 的固定串窗背景校正。所有图像不裁剪、不平滑、不做强度拟合；灰度图白色低、黑色高，使用各通道真值的共同绝对色标。</div>
<p>判断要点：总量恢复接近 100% 不能说明热柱已恢复。请同时查看误差图、CRC 和 CNR 随迭代的变化。无噪声结果仍有明显恢复不足，Poisson 结果后期噪声增大。此处不作有效 FOV 达标结论。</p>
<p>显示为极坐标层内线性插值，统计量直接在原始极坐标上按体积权重计算。轴位选择最近的真实保存层；冠状/矢状固定切片可能错过离平面的热柱，需配合 MIP 判断。</p>
<p>下载原始表格：<a href="data/rod_metrics.csv">热柱 CRC/CNR</a> · <a href="data/global_metrics.csv">全局误差/积分</a> · <a href="data/manifest.json">参数与来源哈希</a></p>'''+sections
    (out/'index.html').write_text(html,encoding='utf-8')
    print(out/'index.html')


if __name__=='__main__': main()
