"""Render source-bound advisor diagrams; no simulation or source-array mutation."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Polygon, Circle, Arc
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
H = ROOT / 'experiments/ELLIPSE500x300_H120'
sys.path.insert(0, str(H))
from geometry import grid

NAVY='#18283B'; TEAL='#128C91'; BLUE='#426CC1'; ORANGE='#DA9952'
GRAY='#8995A2'; LIGHT='#E9EEF3'; PALE='#EAF6F5'; MUTED='#647488'
plt.rcParams.update({'font.family':'Microsoft YaHei', 'font.size':14,
                     'axes.unicode_minus':False, 'svg.fonttype':'path',
                     'pdf.fonttype':42, 'savefig.facecolor':'white'})
for sub in ('figures','data'):
    (OUT/sub).mkdir(parents=True, exist_ok=True)
config=json.loads((H/'config.json').read_text())
geom_path=H/'generated/ehe_spect_5e9_200/payload/whole_geometry.npz'
with np.load(geom_path) as f:
    xyz=f['coordinates_mm']; active=f['active_indices']; support=f['ellipse_fraction']
coords,cells,_=grid(config)
np.testing.assert_allclose(coords,xyz,atol=1e-12)
assert len(xyz)==132040 and len(active)==78920
assert set(np.unique(support))=={0,1}
np.testing.assert_array_equal(active,np.flatnonzero((xyz[:,0]/250)**2+(xyz[:,1]/150)**2<=1+1e-12))
cm_path=ROOT/'Geant4Sim/Geant4Code/CrystalMatrix.txt'
cm=np.loadtxt(cm_path,dtype=int).ravel().reshape(31,64,32)
assert [int((cm[k]==1).sum()) for k in (0,10,20)]==[512,768,1024]
assert np.count_nonzero(cm[:30]==1)+128*64==10496
ev=H/'reports/NEMA_Body_H60/ehe_spect_5e9_200/geometry_evidence/raw'
holes=np.genfromtxt(ev/'EHE_CollimatorHoles.csv',delimiter=',',names=True)
det=np.genfromtxt(ev/'EHE_DetectorGeometry.csv',delimiter=',',names=True)
assert len(holes)==1250 and len(det)==2312
assert np.allclose(holes['y1_mm']+345,298.5) and np.allclose(holes['y2_mm']-holes['y1_mm'],50.5)
assert np.allclose(holes['radius_mm'],1.25)

def txt(ax,x,y,s,size=14,color=NAVY,ha='left',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,ha=ha,va=va,**kw)

def dim(ax,p,q,s,offset=(0,0),size=13,rotation=0,color=MUTED):
    ax.annotate('',q,p,arrowprops=dict(arrowstyle='<->',lw=1.1,color=color,shrinkA=0,shrinkB=0))
    m=(np.array(p)+np.array(q))/2+offset
    txt(ax,*m,s,size,color,ha='center',rotation=rotation,
        bbox=dict(fc='white',ec='none',pad=1.5))

def line(ax,p,q,color=LIGHT,lw=1,**kw):
    ax.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=lw,**kw)

def canvas(title,subtitle):
    fig=plt.figure(figsize=(18,10),facecolor='white')
    fig.text(.045,.94,title,fontsize=27,weight='bold',color=NAVY)
    fig.text(.047,.895,subtitle,fontsize=13.5,color=MUTED)
    return fig

def clean(fig,rect,lim=None):
    ax=fig.add_axes(rect)
    ax.axis('off')
    if lim:
        ax.set_xlim(*lim[0]);ax.set_ylim(*lim[1]);ax.set_aspect('equal')
    return ax

def save(fig,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=240)
    plt.close(fig)

def scene():
    fig=canvas('01｜成像场景与 FOV','218 / 440 keV 双能成像   ·   同一几何，20个相对视角')
    fig.text(.048,.832,'横断面与探头位置',fontsize=18,weight='bold',color=NAVY)
    fig.text(.49,.832,'轴向范围',fontsize=18,weight='bold',color=NAVY)
    fig.text(.695,.832,'柱坐标离散',fontsize=18,weight='bold',color=NAVY)
    ax=clean(fig,[.035,.19,.425,.60],((-330,350),(-225,435)))
    ax.add_patch(Ellipse((0,0),500,300,facecolor=PALE,edgecolor=TEAL,lw=2))
    # A phantom footprint is explanatory, not a substituted truth contour.
    txt(ax,0,38,'椭圆柱 FOV',18,TEAL,ha='center',weight='bold')
    ax.plot(0,0,'o',color=TEAL,ms=4)
    line(ax,(0,-155),(0,414),LIGHT,1,ls=(0,(4,4)))
    line(ax,(-252,0),(256,0),LIGHT,1,ls=(0,(4,4)))
    for j,y in enumerate((300,330,360,390)):
        ax.add_patch(Rectangle((-134.4,y-(1.5 if j<3 else 3)),268.8,3 if j<3 else 6,
                               fc=TEAL if j<3 else BLUE,ec='none'))
    txt(ax,0,426,'JSCC：四层探头',15,ha='center',weight='bold')
    txt(ax,-165,345,'层中心\n300 / 330 / 360 / 390',11,MUTED,ha='right')
    line(ax,(-158,345),(-137,345),GRAY)
    line(ax,(135,298.5),(291,298.5),GRAY)
    line(ax,(0,0),(291,0),GRAY)
    dim(ax,(284,0),(284,298.5),'298.5 mm',offset=(21,0),size=14,rotation=90,color=NAVY)
    line(ax,(-250,-8),(-250,-189),LIGHT)
    line(ax,(250,-8),(250,-189),LIGHT)
    dim(ax,(-250,-181),(250,-181),'500 mm',offset=(0,-17),size=14)
    line(ax,(-250,-150),(-290,-150),LIGHT)
    line(ax,(-250,150),(-290,150),LIGHT)
    dim(ax,(-284,-150),(-284,150),'300 mm',offset=(-20,0),size=14,rotation=90)
    ax.annotate('',(-190,-80),(-230,-80),arrowprops=dict(arrowstyle='->',lw=1.2,color=MUTED))
    ax.annotate('',(-230,-40),(-230,-80),arrowprops=dict(arrowstyle='->',lw=1.2,color=MUTED))
    txt(ax,-184,-80,'x',12,MUTED);txt(ax,-231,-29,'y',12,MUTED,ha='center')
    fig.text(.065,.132,'前表面对齐：JSCC首层晶体 / EHE准直器',fontsize=13,color=NAVY)
    fig.text(.065,.096,'原点为FOV中心；长度单位 mm',fontsize=12,color=MUTED)

    # 40 true axial layers, with the H60 source distinguished from H120 FOV.
    az=clean(fig,[.48,.47,.18,.29],((-110,130),(-85,85)))
    az.add_patch(Rectangle((-75,-60),150,120,fc=PALE,ec=TEAL,lw=1.5))
    for z in np.arange(-60,61,3):
        line(az,(-75,z),(75,z),'#ABD2D0',.55)
    az.add_patch(Rectangle((-42,-30),84,60,fc='#F4D9B8',ec=ORANGE,lw=1.7))
    txt(az,0,5,'NEMA',14,ha='center',weight='bold')
    txt(az,0,-15,'H = 60',12,ha='center')
    dim(az,(96,-60),(96,60),'120 mm',(20,0),13,90)
    txt(az,-90,72,'z',13,MUTED)
    fig.text(.483,.448,'40层 × 3 mm',fontsize=15,color=TEAL,weight='bold')
    av=clean(fig,[.488,.225,.17,.185],((-1.35,1.35),(-1.35,1.35)))
    for a in np.arange(20)*2*np.pi/20:
        av.plot(np.cos(a),np.sin(a),'o',color=TEAL,ms=5)
    av.add_patch(Circle((0,0),1,fc='none',ec=LIGHT,lw=1))
    av.add_patch(Arc((0,0),1.65,1.65,theta1=18,theta2=342,ec=TEAL,lw=1.7))
    a=np.deg2rad(342)
    av.annotate('',(.825*np.cos(a),.825*np.sin(a)),(.825*np.cos(a-.13),.825*np.sin(a-.13)),
                arrowprops=dict(arrowstyle='->',lw=1.6,color=TEAL))
    txt(av,0,.10,'20 × 18°',17,TEAL,ha='center',weight='bold')
    txt(av,0,-.3,'覆盖360°',12,MUTED,ha='center')
    fig.text(.483,.17,'围绕 z 轴相对旋转',fontsize=13,color=NAVY)

    # Draw every stored cell; membership is selected by cell centre, then
    # the entire polar cell is retained. No fractional intersection claim.
    ag=clean(fig,[.685,.405,.28,.35],((-275,275),(-275,275)))
    polys=[]; colors=[]
    for idx,(r0,r1,t0,t1) in enumerate(cells):
        n=30 if r0==0 else 5
        t=np.linspace(t0,t1,n)
        poly=np.concatenate([np.column_stack((r1*np.cos(t),r1*np.sin(t))),
                             np.column_stack((r0*np.cos(t[::-1]),r0*np.sin(t[::-1])))])
        polys.append(poly);colors.append('#C5E7E3' if support[idx] else '#F4F6F8')
    ag.add_collection(PolyCollection(polys,facecolors=colors,edgecolors='#9DB7BA',linewidths=.24))
    ag.add_patch(Ellipse((0,0),500,300,fc='none',ec=TEAL,lw=1.7))
    ag.add_patch(Circle((0,0),255,fc='none',ec=GRAY,lw=.9,ls=(0,(3,4))))
    fig.text(.695,.365,'Δr = 6 mm   ·   Δz = 3 mm',fontsize=15,color=TEAL,weight='bold')
    fig.text(.695,.315,'每环20–140个角向单元',fontsize=13,color=NAVY)
    fig.text(.695,.265,'78,920 个活动单元',fontsize=17,color=NAVY,weight='bold')
    fig.text(.695,.223,'中心位于椭圆内 → 保留完整单元',fontsize=12,color=MUTED)
    fig.text(.695,.17,'完整圆网格：R 255 mm / 132,040点',fontsize=12,color=MUTED)
    fig.text(.695,.115,'网格按实际数据绘制',fontsize=11,color=MUTED)
    save(fig,'01_imaging_scene')

def face_points(x,z,base):
    return np.column_stack((base[0]+.68*np.asarray(x),base[1]+.82*np.asarray(z)-.22*np.asarray(x)))

def face(ax,base,w,h,color,edge=NAVY,lw=.8,zorder=1):
    points=face_points([-w/2,w/2,w/2,-w/2],[-h/2,-h/2,h/2,h/2],base)
    ax.add_patch(Polygon(points,fc=color,ec=edge,lw=lw,zorder=zorder))
    return points

def fine_grid(ax,base,nx,nz,pitch,color,spacing=None,zorder=1.1):
    w=nx*pitch;h=nz*pitch
    seg=[]
    for x in np.linspace(-w/2,w/2,nx+1): seg.append(face_points([x,x],[-h/2,h/2],base))
    for z in np.linspace(-h/2,h/2,nz+1): seg.append(face_points([-w/2,w/2],[z,z],base))
    ax.add_collection(LineCollection(seg,colors=color,linewidths=.19,zorder=zorder))

def systems():
    fig=canvas('02｜两种探头设计','同一FOV与前表面距离   ·   保留各自材料与结构')
    fig.text(.05,.825,'JSCC',fontsize=23,weight='bold',color=TEAL)
    fig.text(.05,.786,'多层编码结构 + Compton事件',fontsize=15,color=NAVY)
    fig.text(.565,.825,'EHE SPECT',fontsize=23,weight='bold',color=BLUE)
    fig.text(.565,.786,'高能平行孔准直器 + NaI探测面',fontsize=15,color=NAVY)
    sep=clean(fig,[.522,.13,.005,.69]);sep.plot([.5,.5],[0,1],color=LIGHT,lw=1)
    ax=clean(fig,[.025,.315,.51,.435],((-30,660),(-70,265)))
    bases=[(100,80),(235,103),(370,126),(505,149)]
    # Draw rear-to-front so foreground faces occlude background correctly.
    for layer in (3,2,1,0):
        base=bases[layer]
        if layer==3:
            face(ax,base,268.8,134.4,'#ACC2EA',BLUE)
            fine_grid(ax,base,128,64,2.1,'#F5F8FF')
        else:
            face(ax,base,268.8,134.4,'#F7FAFB','#B1BEC9')
            polys=[];colors=[]
            for j,i in np.argwhere(cm[layer*10]>0):
                x=4.2*(-32+j+.5);z=4.2*(-16+i+.5)
                polys.append(face_points([x-1.5,x+1.5,x+1.5,x-1.5],
                                          [z-1.5,z-1.5,z+1.5,z+1.5],base))
                colors.append(TEAL if cm[layer*10,j,i]==1 else '#7A8795')
            ax.add_collection(PolyCollection(polys,facecolors=colors,edgecolors='none'))
        n=[512,768,1024,8192][layer]
        bottom=face_points([-134.4],[-67.2],base)[0]
        line(ax,tuple(bottom),(bottom[0],-24),LIGHT)
        txt(ax,base[0]-60,-40,f'L{layer+1}  ·  {n:,}',12,TEAL if layer<3 else BLUE,ha='center',weight='bold')
    txt(ax,310,245,'10,496 个 GAGG晶体',18,NAVY,ha='center',weight='bold')
    ax.annotate('',(40,72),(-17,62),arrowprops=dict(arrowstyle='->',lw=2,color=ORANGE))
    txt(ax,-17,42,'入射γ',12,ORANGE)
    fig.text(.06,.305,'前三层',fontsize=13,color=TEAL,weight='bold')
    fig.text(.06,.266,'GAGG / W：3 × 3 × 3 mm',fontsize=14,color=NAVY)
    fig.text(.06,.226,'32 × 64格位；节距4.2 mm',fontsize=13,color=MUTED)
    fig.text(.29,.305,'末层',fontsize=13,color=BLUE,weight='bold')
    fig.text(.29,.266,'GAGG：2 × 6 × 2 mm',fontsize=14,color=NAVY)
    fig.text(.29,.226,'128 × 64；节距2.1 mm',fontsize=13,color=MUTED)
    fig.text(.06,.17,'层中心间距 30 mm',fontsize=15,weight='bold',color=NAVY)
    legend=clean(fig,[.055,.09,.42,.045]);legend.set_xlim(0,1);legend.set_ylim(0,1)
    for x,col,label in [(0,TEAL,'GAGG'),(.25,'#7A8795','W'),(.43,'#F7FAFB','空位')]:
        legend.add_patch(Rectangle((x,.25),.032,.4,fc=col,ec='#B1BEC9',lw=.6))
        txt(legend,x+.045,.46,label,12,NAVY)
    txt(legend,.65,.46,'实际编码图案',11,MUTED)

    ae=clean(fig,[.55,.40,.415,.35],((-165,335),(-95,185)))
    base=(5,20);back=(66,48);nai=(176,80)
    # Small exploded separation is for readability, stated in footer.
    face(ae,nai,272,136,'#B7C9EE',BLUE)
    fine_grid(ae,nai,68,34,4,'#F3F6FB')
    p0=face_points([-165,165,165,-165],[-82.5,-82.5,82.5,82.5],base)
    p1=face_points([-165,165,165,-165],[-82.5,-82.5,82.5,82.5],back)
    ae.add_patch(Polygon(np.array([p0[1],p0[2],p1[2],p1[1]]),fc='#65778B',ec=NAVY,lw=.8,zorder=3))
    ae.add_patch(Polygon(np.array([p0[2],p0[3],p1[3],p1[2]]),fc='#9AA8B7',ec=NAVY,lw=.8,zorder=3))
    face(ae,base,330,165,'#C5CED8',GRAY,zorder=4)
    rings=[]
    th=np.linspace(0,2*np.pi,17)
    for row in holes:
        rings.append(face_points(row['x_mm']+1.25*np.cos(th),row['z_mm']+1.25*np.sin(th),base))
    ae.add_collection(PolyCollection(rings,fc='white',ec='#77899B',lw=.24,zorder=5))
    txt(ae,-97,143,'Pb',14,MUTED,weight='bold')
    txt(ae,293,135,'NaI',14,BLUE,weight='bold')
    line(ae,(286,125),(254,105),BLUE)
    txt(ae,-103,-88,'1,250孔 · 三角晶格',15,NAVY,weight='bold')
    txt(ae,222,-50,'2312像素',14,BLUE,ha='center',weight='bold')
    fig.text(.565,.38,'孔径 2.5   /   隔厚 3.4   /   孔长 50.5 mm',fontsize=14,color=NAVY)

    # True transverse section through one hole row: 2.5-mm channels,
    # 5.9-mm pitch, 50.5-mm lead thickness, contiguous 10-mm NaI slab.
    ac=clean(fig,[.565,.17,.20,.17],((-8,77),(-12,24)))
    ac.add_patch(Rectangle((0,-8.85),50.5,23.6,fc='#AAB6C4',ec=GRAY,lw=.8))
    for zz in (-5.9,0,5.9,11.8):
        ac.add_patch(Rectangle((0,zz-1.25),50.5,2.5,fc='white',ec='none'))
    ac.add_patch(Rectangle((50.5,-8.85),10,23.6,fc='#B7C9EE',ec=BLUE,lw=.8))
    for z in np.arange(-8.85,14.8,4):line(ac,(50.5,z),(60.5,z),'#F8FAFD',.7)
    ac.annotate('',(56,0),(-7,0),arrowprops=dict(arrowstyle='->',color=ORANGE,lw=1.5))
    dim(ac,(0,21),(50.5,21),'50.5',(0,0),11)
    txt(ac,66,5,'10',11,BLUE,ha='center')
    fig.text(.79,.29,'探测面：272 × 136 mm',fontsize=13.5,color=NAVY)
    fig.text(.79,.251,'NaI厚 10 mm',fontsize=13,color=NAVY)
    fig.text(.79,.212,'68 × 34像素 · 4 mm节距',fontsize=13,color=MUTED)
    fig.text(.565,.118,'结构展开示意；局部剖面显示平行孔',fontsize=11,color=MUTED)
    save(fig,'02_detector_design')

if __name__=='__main__':
    scene();systems()
    paths=[H/'config.json',H/'geometry.py',H/'build_whole_cell_geometry_v4.py',geom_path,
           cm_path,ROOT/'Geant4Sim/Geant4Code/src/DetectorConstruction.cc',
           ev/'EHE_CollimatorHoles.csv',ev/'EHE_DetectorGeometry.csv',ev/'EHE_GeometrySummary.txt']
    evidence={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    metadata=dict(source_sha256=evidence,full_grid_points=132040,active_cells=78920,
                  active_membership='cell centre lies in ellipse; retain entire cell',
                  actual_jscc_crystals_per_plane=[512,768,1024,8192],
                  common_front_face_distance_mm=298.5,
                  front_distance_definition='FOV centre to first JSCC crystal front / EHE lead front',
                  figure2='exploded spacing illustrative; material cell patterns and counts are actual',
                  dimensions_order='JSCC crystal: x,y,z; EHE detector face: x,z and thickness y',
                  no_simulation_or_reconstruction=True)
    (OUT/'data/geometry_sources.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(OUT/'figures'),'checks':'geometry / active support / crystal counts / EHE holes passed'},ensure_ascii=False))
