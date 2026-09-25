"""Render unfiltered full-FOV Geant4 results against explicit polar truth."""
import json,csv,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
base=Path(__file__).resolve().parent/'generated'
r=base/'Results/Contrast_1e9_1626525';out=r/'VisualReport';out.mkdir(exist_ok=True)
v=base/'VisualizationInputs';fp=v/'Factors/440keV_RotateNum20'
c=np.loadtxt(fp/'coor_polar_full.csv',delimiter=',');w=np.fromfile(fp/'polar_cell_volume_mm3.float64',dtype='<f8');t=np.load(v/'Truth_Contrast_1e9.npz')
m=json.loads((r/'run_manifest.json').read_text());assert m['iterations']==10000 and m['pixel_count']==len(c)==51240
z=np.unique(c[:,2]);tri=Delaunay(c[:1281,:2]);a=np.arange(-148.5,150,3);xx,yy=np.meshgrid(a,a)
def cart(x):return np.stack([LinearNDInterpolator(tri,p,fill_value=0)(xx,yy) for p in x.reshape(40,1281)])
def show(ax,x,title,vmax,extent):
 im=ax.imshow(x,origin='lower',extent=extent,cmap='gray_r',vmin=0,vmax=vmax,interpolation='nearest');ax.set_title(title,fontsize=9);return im
names=m['outputs'];its=[100,500,1000,2000,5000,10000];hist={};truths={};metrics=[]
for name in names:
 f=np.fromfile(r/name,dtype='<f4');h=np.fromfile(r/(name+'_Iter_10000_200.selected'),dtype='<f4').reshape(6,51240)
 assert np.isfinite(h).all() and (h>=0).all() and np.array_equal(f,h[-1])
 tr=t['rho218'] if name=='Image_218_SinglePhoton_CrossTalkCorrected' else (t['rho440']+t['rho218'] if 'Plus218' in name else t['rho440'])
 hist[name]=h;truths[name]=tr
 for it,x in zip(its,h):metrics.append({'channel':name,'iteration':it,'relative_L2':float(np.sqrt(np.dot((x-tr)**2,w)/np.dot(tr*tr,w))),'integral_ratio':float(np.dot(x,w)/np.dot(tr,w))})
for title,keys in [('440_channels',names[:3]),('218_and_composites',names[3:])]:
 fig,axs=plt.subplots(6,4,figsize=(14,19),layout='constrained')
 rows=[item for key in keys for item in [('Truth: '+key.replace('Image_',''),truths[key]),(key.replace('Image_',''),hist[key][-1])]]
 for row,(label,x) in enumerate(rows):
  cube=cart(x);scale=float(truths[keys[row//2]].max())
  for col,target in enumerate([-45,0,45]):
   zi=np.argmin(abs(z-target));im=show(axs[row,col],cube[zi],label+'\nz='+str(z[zi])+' mm',scale,[-150,150,-150,150])
  show(axs[row,3],cube.max(axis=1),label+'\nFull-height coronal MIP',scale,[-150,150,-60,60])
 fig.suptitle('Geant4 1e9; iteration 10000; no smoothing / cropping / fitted scale\nEach channel uses its matching truth maximum; composites are gamma-channel sums')
 fig.savefig(out/(title+'.png'),dpi=140);plt.close(fig)
for group,keys in [('440_iterations',names[:3]),('218_iterations',[names[3]])]:
 fig,axs=plt.subplots(len(keys),7,figsize=(21,3.4*len(keys)),squeeze=False,layout='constrained')
 for row,key in enumerate(keys):
  scale=float(truths[key].max());zi=np.argmin(abs(z))
  for col,(label,x) in enumerate([('Truth',truths[key])]+[(str(it),h) for it,h in zip(its,hist[key])]):show(axs[row,col],cart(x)[zi],key.replace('Image_','')+'\n'+label,scale,[-150,150,-150,150])
 fig.suptitle('Central axial slice z=-1.5 mm; actual saved iterations; no smoothing or intensity fit')
 fig.savefig(out/(group+'.png'),dpi=140);plt.close(fig)
with (out/'metrics.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(metrics[0]));wr.writeheader();wr.writerows(metrics)
(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
files=[r/'run_manifest.json',v/'Truth_Contrast_1e9.npz',fp/'coor_polar_full.csv',fp/'polar_cell_volume_mm3.float64']+list(r.glob('*.selected'))
(out/'provenance.json').write_text(json.dumps({'selected_iterations':its,'display':'In-plane linear interpolation only; no z interpolation, smoothing, cropping, fitted intensity scaling; gray_r; values above truth maximum saturate','hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))
shutil.copy2(__file__,out/Path(__file__).name)
(out/'index.html').write_text('<meta charset="utf-8"><h1>Geant4 Contrast 1e9 / 10000 iterations</h1><p>全视野，无平滑、裁剪或强度拟合；灰度上限为对应真值最大值，超出部分饱和。组合图是 γ 通道和。</p>'+''.join('<h2>'+p.stem+'</h2><img style="width:100%" src="'+p.name+'">' for p in out.glob('*.png')),encoding='utf-8')
print(json.dumps([x for x in metrics if x['iteration'] in [1000,10000]],indent=2))
