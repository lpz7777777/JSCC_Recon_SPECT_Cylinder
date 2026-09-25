"""Plot point-source MC/model efficiencies and layer ratios, with missing cells explicit."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=json.loads(a.comparison.read_text());out=a.output;out.mkdir(exist_ok=True,parents=True)
    channels=('218','440','cross');zs=sorted({r['z_mm'] for r in m['rows']})
    positions=[(0,0)]+[(r,az) for r in (75,135) for az in (0,90,180,270)]
    fig,axs=plt.subplots(1,3,figsize=(17,6),layout='constrained')
    for ax,c in zip(axs,channels):
        values=np.full((len(positions),len(zs)),np.nan)
        for r in m['rows']:
            if r['channel']==c:values[positions.index((r['r_mm'],r['azimuth_deg'])),zs.index(r['z_mm'])]=r['ratio']
        cm=plt.get_cmap('RdBu_r').copy();cm.set_bad('#d5d5d5')
        im=ax.imshow(values,cmap=cm,vmin=.95,vmax=1.05,aspect='auto')
        ax.set_xticks(range(len(zs)),zs);ax.set_yticks(range(len(positions)),[f'r={r}, a={az}' for r,az in positions]);ax.set_xlabel('Source z (mm)');ax.set_title(c+' keV' if c!='cross' else '440 -> 218 window')
        for i in range(len(positions)):
            for j in range(len(zs)):ax.text(j,i,'--' if np.isnan(values[i,j]) else f'{values[i,j]:.3f}',ha='center',va='center',fontsize=7)
    fig.colorbar(im,ax=axs,label='Observed / predicted total counts; 1 = agreement',shrink=.65)
    fig.suptitle(f'Independent Geant4 point responses: {m["complete_workers"]}/{m["expected_workers"]} workers; gray = missing\nCalibrated B / cell volume, linear xy/z interpolation; one view; no 1/20 scaling')
    fig.savefig(out/'point_count_ratios.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    for row,c in enumerate(channels):
        for col,radius in enumerate((0,75,135)):
            ax=axs[row,col]
            for az,color in zip((0,90,180,270),('tab:blue','tab:orange','tab:green','tab:red')):
                rows=sorted([r for r in m['rows'] if r['channel']==c and r['r_mm']==radius and r['azimuth_deg']==az],key=lambda r:r['z_mm'])
                if not rows:continue
                x=[r['z_mm'] for r in rows]
                ax.errorbar(x,[r['observed_efficiency'] for r in rows],yerr=[np.sqrt(r['observed'])/r['photons'] for r in rows],color=color,fmt='o-',ms=3,lw=1,label=f'MC a={az}')
                ax.plot(x,[r['predicted_efficiency'] for r in rows],'--',color=color,label=f'Model a={az}')
            ax.set_title(f'{c}, r={radius} mm');ax.set_xlabel('z (mm)');ax.set_ylabel('Counts / emitted photon');ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2)
    fig.suptitle('Absolute point response vs z; error bars = Poisson counting SE')
    fig.savefig(out/'point_efficiency.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,c in zip(axs,channels):
        for i,normal in enumerate((200,230,260,290)):
            rs=[r for r in m['rows'] if r['channel']==c]
            ax.scatter([r['z_mm']+(i-1.5)*.5 for r in rs],[r['layers'][i]['ratio'] for r in rs],s=12,alpha=.55,label=f'Layer {normal} mm')
        ax.axhline(1,color='black',lw=1);ax.set_title(c);ax.set_xlabel('z (mm), small offsets by layer');ax.set_ylabel('MC / model layer counts');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Layer totals across all radii and azimuths; inspect counting uncertainty in source JSON')
    fig.savefig(out/'point_layer_ratios.png',dpi=160);plt.close(fig)
    if (out/'index.html').exists():
        (out/'data/point_comparison.json').write_text(json.dumps(m,indent=2)+'\n')
        html=(out/'index.html').read_text(encoding='utf-8').split('<!-- point-response -->')[0]
        html+='<!-- point-response --><section><h2>独立 Geant4 点源响应核查</h2>'
        html+=f'<p>已收集 {m["complete_workers"]}/{m["expected_workers"]} 个单视角点源，实际 {m["actual_photons"]:,} 初级光子。缺失任务：{m["missing_indices"]}。灰色单元表示缺失，不能解释为零响应。此项不是点源定位或分辨率重建。</p>'
        html+='<p>模型在除去体积权重后作层内重心插值和轴向线性插值。总计数吻合不代表逐晶体响应完全吻合；分层离散也包含计数统计误差。<a href="data/point_comparison.json">完整数值、分层统计误差和去泊松噪声的逐晶体差异估计</a></p>'
        for name,title in [('point_count_ratios','Geant4 / 模型总计数比'),('point_efficiency','轴向绝对探测响应'),('point_layer_ratios','分层总响应比')]:
            html+=f'<h3>{title}</h3><a href="{name}.png"><img src="{name}.png"></a>'
        (out/'index.html').write_text(html+'</section>',encoding='utf-8')
        import shutil
        shutil.copy2(__file__,out/'scripts'/Path(__file__).name)
    print(out)


if __name__=='__main__':main()
