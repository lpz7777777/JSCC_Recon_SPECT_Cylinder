"""Measure axial-edge stability in the independent Uniform FOV120 Geant4 reconstruction."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--result',type=Path,required=True)
p.add_argument('--truth',type=Path,required=True)
p.add_argument('--coordinates',type=Path,required=True)
p.add_argument('--volume',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
m=json.loads((a.result/'run_manifest.json').read_text())
assert (m['dataset'],m['count_level'],m['pixel_count'],m['z_layers'])==('Uniform','1e9',51240,40)
c=np.loadtxt(a.coordinates,delimiter=',');v=np.fromfile(a.volume,dtype='<f8')
assert c.shape==(51240,3) and v.shape==(51240,) and np.all(v>0)
t=np.load(a.truth)
z=np.unique(c[:,2]);r=np.hypot(c[:,0],c[:,1]);roi=r<=135
iters=[100,500,1000,2000,5000,10000]
channels={
 '440 single':('Image_440_SinglePhoton','rho440'),
 '440 Compton':('Image_440_ComptonOnly','rho440'),
 '440 JSCC':('Image_440_SinglePlusCompton','rho440'),
 '218 corrected':('Image_218_SinglePhoton_CrossTalkCorrected','rho218'),
}
rows=[];summaries=[]
for label,(file,key) in channels.items():
 tr=t[key];data=np.fromfile(a.result/(file+'_Iter_10000_200.selected'),dtype='<f4').reshape(len(iters),-1)
 assert np.isfinite(data).all() and data.shape==(6,51240)
 for it,x in zip(iters,data):
  for zi,zz in enumerate(z):
   sel=roi & (c[:,2]==zz);w=v[sel];xx=x[sel];tt=tr[sel]
   mean=np.average(xx,weights=w);ref=np.average(tt,weights=w)
   cv=np.sqrt(np.average((xx-mean)**2,weights=w))/mean if mean>0 else np.nan
   nrmse=np.sqrt(np.dot(w,(xx-tt)**2)/np.dot(w,tt**2))
   rows.append(dict(channel=label,iteration=it,z_mm=zz,mean_recovery=mean/ref,cv=cv,nrmse=nrmse,volume_mm3=w.sum()))
  for region,sel in [('center',roi & (abs(c[:,2])<=30)),('middle',roi & (abs(c[:,2])>30)&(abs(c[:,2])<=45)),('edge',roi & (abs(c[:,2])>45))]:
   w=v[sel];xx=x[sel];tt=tr[sel];mean=np.average(xx,weights=w);ref=np.average(tt,weights=w)
   cv=np.sqrt(np.average((xx-mean)**2,weights=w))/mean
   nrmse=np.sqrt(np.dot(w,(xx-tt)**2)/np.dot(w,tt**2))
   summaries.append(dict(channel=label,iteration=it,region=region,mean_recovery=mean/ref,cv=cv,nrmse=nrmse,volume_mm3=w.sum()))
with (a.output/'layer_metrics.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)
with (a.output/'region_metrics.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=summaries[0]);wr.writeheader();wr.writerows(summaries)
fig,axs=plt.subplots(4,3,figsize=(14,12),sharex=True,layout='constrained')
for i,label in enumerate(channels):
 for it in [500,1000,5000,10000]:
  sub=[q for q in rows if q['channel']==label and q['iteration']==it]
  for j,key in enumerate(['mean_recovery','cv','nrmse']):axs[i,j].plot(z,[q[key] for q in sub],label=str(it))
 for j in range(3):
  axs[i,j].axvline(-45,color='.7',ls=':');axs[i,j].axvline(45,color='.7',ls=':')
  axs[i,j].set_xlabel('z (mm)');axs[i,j].set_ylabel(['mean / truth','CV','relative L2'][j]);axs[i,j].grid(alpha=.2)
 axs[i,0].set_title(label);axs[i,2].legend(title='iteration',fontsize=8)
fig.suptitle('Uniform Geant4 1e9: axial stability within r<=135 mm; raw polar density, no smoothing')
fig.savefig(a.output/'axial_profiles.png',dpi=170);plt.close(fig)
(a.output/'provenance.json').write_text(json.dumps({'result':str(a.result),'truth':str(a.truth),'radius_limit_mm':135,'regions_mm':{'center':'|z|<=30','middle':'30<|z|<=45','edge':'45<|z|<=60'},'source_sha256':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [a.result/'run_manifest.json',a.truth,a.coordinates,a.volume]}},indent=2))
for label in channels:
 for it in [1000,10000]:
  print(label,it,[(q['region'],round(q['mean_recovery'],3),round(q['cv'],3),round(q['nrmse'],3)) for q in summaries if q['channel']==label and q['iteration']==it])
