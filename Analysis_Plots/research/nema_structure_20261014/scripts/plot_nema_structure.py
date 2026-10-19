"""One source-bound NEMA H60 structure / dual-energy filling illustration.

Analytic z=0 diagrams are drawn from the registered geometry, not reconstructed
images or MIPs. The saved 3-mm 3D truth is read-only and independently checked.
"""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'experiments/ELLIPSE500x300_H120').is_dir())
H=ROOT/'experiments/ELLIPSE500x300_H120'
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(H))
from make_nema_body_h60 import body_boundary

NAVY='#18283B'; BLUE='#426CC1'; TEAL='#128C91'; GRAY='#8796A8'
MUTED='#647488'; BG='#E9EDF1'; RULE='#DFE6ED'
COLORS={440:BLUE,218:TEAL}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def text(ax,x,y,s,size=14,color=NAVY,**kw):
    return ax.text(x,y,s,fontsize=size,color=color,ha=kw.pop('ha','center'),
                   va=kw.pop('va','center'),**kw)

def main():
    for folder in ('figures','data'):(OUT/folder).mkdir(parents=True,exist_ok=True)
    cp=H/'nema_body_h60_config.json';mp=H/'reports/NEMA_Body_H60/manifest.json'
    tp=H/'generated/NEMA_Body_H60/truth_3mm.npz'
    cfg=json.loads(cp.read_text()); manifest=json.loads(mp.read_text())
    bindings={str(p.relative_to(ROOT)):sha(p) for p in (cp,mp,tp,H/'make_nema_body_h60.py')}
    assert sha(tp)==manifest['truth_sha256'] and sha(cp)==manifest['configuration_sha256']
    spheres=manifest['spheres']
    assert [s['diameter_mm'] for s in spheres]==[10,13,17,22,28,37]
    assert {e:[s['diameter_mm'] for s in spheres if s['hot_energy_keV']==e] for e in (218,440)}=={218:[10,17,28],440:[13,22,37]}
    for source,s in zip(cfg['spheres'],spheres):
        theta=np.deg2rad(source['angle_deg'])
        np.testing.assert_allclose(s['center_mm'],[57*np.cos(theta),57*np.sin(theta),0],atol=1e-12)
        assert s['hot_energy_keV']==source['hot_energy_keV']
    with np.load(tp) as truth:
        all_spheres=sum(truth[f'sphere_{d}_fraction_zyx'] for d in (10,13,17,22,28,37))
        background=np.maximum(truth['body_fraction_zyx']-truth['lung_fraction_zyx']-all_spheres,0)
        for e in (218,440):
            predicted=background.copy()
            for s in spheres:
                if s['hot_energy_keV']==e:predicted+=10*truth[f'sphere_{int(s["diameter_mm"])}_fraction_zyx']
            np.testing.assert_array_equal(predicted,truth[f'activity_{e}_zyx'])
            assert np.isfinite(predicted).all() and predicted.min()==0 and predicted.max()==10
    _,inner,_,outer=body_boundary(cfg)
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':14,'axes.unicode_minus':False,
                         'svg.fonttype':'path','pdf.fonttype':42,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(18,9.5),facecolor='white')
    fig.text(.045,.94,'NEMA Body｜结构与双能分布',fontsize=27,color=NAVY,weight='bold')
    fig.text(.047,.895,'本研究 H60 体模   ·   真实三维球体   ·   两路活度分布分别定义',fontsize=14,color=MUTED)
    fig.add_artist(Line2D([.045,.955],[.862,.862],color=RULE,lw=1.2))
    fig.text(.053,.812,'三维结构',fontsize=21,color=NAVY,weight='bold')
    fig.text(.393,.812,'440 keV',fontsize=23,color=BLUE,weight='bold')
    fig.text(.703,.812,'218 keV',fontsize=23,color=TEAL,weight='bold')
    fig.text(.393,.774,'横断面  z = 0',fontsize=12,color=MUTED)
    fig.text(.703,.774,'横断面  z = 0',fontsize=12,color=MUTED)

    # Geometry-preserving 3D view: open top, six actual spheres, lung column.
    ax=fig.add_axes([.025,.29,.35,.48],projection='3d')
    ax.set_proj_type('ortho');ax.view_init(elev=37,azim=-68)
    floor=np.column_stack((outer,np.full(len(outer),-30.0)))
    ax.add_collection3d(Poly3DCollection([floor],facecolors='#F1F4F7',edgecolors='none',alpha=.10,zsort='min'))
    reduced=outer[::8]
    walls=[]
    for i in range(len(reduced)):
        p=reduced[i];q=reduced[(i+1)%len(reduced)]
        walls.append([(p[0],p[1],-30),(q[0],q[1],-30),(q[0],q[1],30),(p[0],p[1],30)])
    ax.add_collection3d(Poly3DCollection(walls,facecolors='#B9C8D5',edgecolors='none',alpha=.055))
    for z in (-30,30):ax.plot(outer[:,0],outer[:,1],z,color='#97A9BA',lw=1.1,alpha=.8)
    # Full-height cold central cylinder, translucent to expose the spheres.
    t=np.linspace(0,2*np.pi,70);z=np.array([-30.,30.]);tt,zz=np.meshgrid(t,z)
    ax.plot_surface(25.5*np.cos(tt),25.5*np.sin(tt),zz,color='#AEBAC6',alpha=.18,
                    linewidth=0,shade=False,antialiased=True)
    for z in (-30,30):ax.plot(25.5*np.cos(t),25.5*np.sin(t),z,color=GRAY,lw=.8)
    u=np.linspace(0,2*np.pi,48);v=np.linspace(0,np.pi,30)
    for s in spheres:
        cx,cy,cz=s['center_mm'];r=s['diameter_mm']/2
        xx=cx+r*np.outer(np.cos(u),np.sin(v))
        yy=cy+r*np.outer(np.sin(u),np.sin(v))
        zz=cz+r*np.outer(np.ones_like(u),np.cos(v))
        ax.plot_surface(xx,yy,zz,color=COLORS[s['hot_energy_keV']],linewidth=0,
                        shade=True,antialiased=True,rcount=48,ccount=30)
    # Actual ring and equator are explicit, not a cylindrical insert model.
    ax.plot(57*np.cos(t),57*np.sin(t),np.zeros(len(t)),color=GRAY,lw=.8,ls=(0,(3,4)),alpha=.8)
    ax.set_xlim(-170,170);ax.set_ylim(-140,140);ax.set_zlim(-50,70)
    ax.set_box_aspect((340,280,120),zoom=1.20);ax.axis('off')
    ax.text(0,0,57,'肺柱 Φ51',color=MUTED,fontsize=12,ha='center')
    ax.plot([0,0],[0,0],[31,48],color=GRAY,lw=.9)
    fig.text(.055,.285,'300 × 230 × 60 mm',fontsize=19,color=NAVY,weight='bold')
    fig.text(.055,.237,'球心环 R57 mm   ·   所有球心 z = 0',fontsize=13,color=MUTED)
    # Color key represents energy ownership, not an energy-summed image.
    for x,c,label in [(.055,BLUE,'440 keV球'),(.205,TEAL,'218 keV球')]:
        fig.add_artist(Line2D([x,x+.012],[.19,.19],color=c,lw=8,solid_capstyle='round'))
        fig.text(x+.022,.185,label,fontsize=13,color=NAVY)

    labelpos={10:(105,5),13:(75,83),17:(-75,83),22:(-110,5),28:(-75,-84),37:(82,-84)}
    axes=[]
    for e,left in ((440,.385),(218,.695)):
        a=fig.add_axes([left,.335,.275,.40])
        a.set_aspect('equal');a.set_xlim(-170,170);a.set_ylim(-135,135);a.axis('off')
        a.add_patch(Polygon(outer,fc='#AAB7C4',ec=GRAY,lw=.8))
        a.add_patch(Polygon(inner,fc=BG,ec='none'))
        a.add_patch(Circle((0,0),25.5,fc='white',ec='#93A0AE',lw=1.1))
        text(a,0,0,'肺\n0',11,MUTED)
        for s in spheres:
            x,y,_=s['center_mm'];d=s['diameter_mm'];hot=s['hot_energy_keV']==e
            a.add_patch(Circle((x,y),d/2,fc=COLORS[e] if hot else 'white',
                              ec=COLORS[e] if hot else '#93A0AE',lw=1.25 if hot else 1,
                              ls='-' if hot else (0,(3,2))))
            if hot:
                px,py=labelpos[d]
                a.annotate(f'Φ{d:g}',(x,y),xytext=(px,py),ha='center',va='center',
                           fontsize=14,color=COLORS[e],weight='bold',
                           arrowprops=dict(arrowstyle='-',lw=.85,color=COLORS[e],shrinkA=4,shrinkB=4))
        # Tiny consistent x/y axes, leaving the body unmirrored.
        a.annotate('',(-116,-99),(-139,-99),arrowprops=dict(arrowstyle='->',lw=.8,color=GRAY))
        a.annotate('',(-139,-76),(-139,-99),arrowprops=dict(arrowstyle='->',lw=.8,color=GRAY))
        text(a,-110,-99,'x',10,MUTED);text(a,-139,-70,'y',10,MUTED)
        fig.text(left+.012,.29,'热球：'+('13 / 22 / 37 mm' if e==440 else '10 / 17 / 28 mm'),
                 fontsize=16,color=COLORS[e],weight='bold')
        fig.text(left+.012,.245,'本能量热球 10   ·   背景 1',fontsize=14,color=NAVY)
        fig.text(left+.012,.202,'另一组球及肺柱：0',fontsize=13,color=MUTED)
        axes.append(a)
    fig.add_artist(Line2D([.365,.365],[.17,.815],color=RULE,lw=1))
    fig.text(.047,.106,'相对活度浓度：各能量以自身背景为 1；热球 / 背景 = 10 : 1',fontsize=14,color=NAVY)
    fig.text(.047,.065,'球径单位 mm；右侧为解析几何示意，实际输运源使用3 mm体素真值。',fontsize=12,color=MUTED)
    fig.text(.955,.065,'H60研究改型 · 未建模体内衰减',fontsize=12,color=MUTED,ha='right')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    bad=[]
    for item in fig.texts:
        box=item.get_window_extent(renderer)
        if box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height:bad.append(item.get_text())
    assert not bad,bad
    basename='nema_structure_218_440'
    for ext in ('png','pdf','svg'):fig.savefig(OUT/'figures'/f'{basename}.{ext}',dpi=300)
    plt.close(fig)
    assert all(sha(ROOT/p)==value for p,value in bindings.items())
    write(OUT/'data/source_and_scientific_qa.json',dict(passed=True,source_sha256=bindings,
          original_sources_unchanged=True,source_arrays_reproduced_exactly=True,
          geometry=dict(body_dimensions_mm=[300,230,60],lung_diameter_mm=51,sphere_center_ring_radius_mm=57,
                        spheres=spheres),activity=dict(background=1,own_energy_hot_sphere=10,opposite_energy_sphere=0,lung=0),
          cross_section='Analytic geometry at z=0, not MIP and not a reconstructed image',
          source_truth_spacing_mm=3,energy_maps_are_separate=True,no_cross_energy_sum=True,
          three_dimensional_spheres=True,body_attenuation_model=False,off_canvas_text=bad,
          script_sha256=sha(__file__),outputs_sha256={ext:sha(OUT/'figures'/f'{basename}.{ext}') for ext in ('png','pdf','svg')}))
    print(str(OUT/'figures'/f'{basename}.png'))

if __name__=='__main__':main()
