"""Readable verified A/B multiplanar figures and all remaining native metrics."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze_nema_result import digest,xy_interpolator
from mip_projection import axial_mip,axial_selection

HERE=Path(__file__).resolve().parent
CHANNELS=('440_ComptonOnly','440_SinglePlusCompton')

def plot(comparison,results):
    report=json.loads((comparison/'comparison.json').read_text())
    manifest=json.loads((comparison/'artifact_manifest.json').read_text())
    for rel,sha in manifest.items():
        if digest(comparison/rel)!=sha:raise ValueError('Existing comparison artifact differs: '+rel)
    truthpath=HERE/'generated/NEMA_Body_H60/truth_3mm.npz'
    if digest(truthpath)!=report['truth_sha256']:raise ValueError('Truth differs')
    truth=np.load(truthpath);x,y,z=(truth[k+'_mm'] for k in 'xyz')
    geompath=HERE/'generated/Geometry/geometry.npz';g=np.load(geompath)
    coords=g['coordinates_mm'];active=g['active_indices']
    interpolate=xy_interpolator(coords[:3301,:2],x,y)
    xx,yy=np.meshgrid(x,y);ellipse=(xx/250)**2+(yy/150)**2<=1
    def image(frame):
        full=np.zeros(132040,np.float32);full[active]=frame
        im=interpolate(full.reshape(40,3301));im[:,~ellipse]=0;return im
    extent=(float(x[0]-1.5),float(x[-1]+1.5),float(y[0]-1.5),float(y[-1]+1.5))
    ix=int(np.argmin(abs(x)));iy=int(np.argmin(abs(y)))
    for channel in CHANNELS:
        images=[('truth',truth['activity_440_zyx'])]
        for group in ('legacy','ideal'):
            root=results/group;run=json.loads((root/'run_manifest.json').read_text())
            if digest(geompath)!=run['geometry_sha256']:raise ValueError('Geometry differs')
            path=root/f'Image_{channel}_history.float32'
            if digest(path)!=report['history_sha256'][group+'/'+channel]:raise ValueError('History differs')
            frame=np.memmap(path,'<f4',mode='r',shape=(40,82040))[-1]
            images.append(('A legacy' if group=='legacy' else 'B first scatter',image(frame)/report['common_normalizers'][channel]))
        fig,axes=plt.subplots(3,4,figsize=(15,10),layout='constrained')
        for row,(label,im) in enumerate(images):
            panels=((im[20],extent),(im[:,iy,:],(extent[0],extent[1],-60,60)),
                    (im[:,:,ix],(extent[2],extent[3],-60,60)),(axial_mip(im,z,8),extent))
            for col,((panel,ex),kind) in enumerate(zip(panels,('axial z=+1.5 mm','coronal','sagittal','central 72 mm MIP'))):
                color=axes[row,col].imshow(panel,origin='lower',extent=ex,cmap='gray_r',vmin=0,vmax=10,interpolation='nearest')
                axes[row,col].set_title(label+'\n'+kind);axes[row,col].tick_params(labelsize=8)
                axes[row,col].set_xlabel('mm')
            axes[row,0].set_ylabel(label+'; mm')
        fig.colorbar(color,ax=axes,shrink=.75,label='Density / same channel A2000 background; truth background=1')
        fig.suptitle(channel+': NEMA 1e9, iteration 2000; no smoothing; full ellipse FOV')
        fig.savefig(comparison/f'final_planes_{channel}.png',dpi=150);plt.close(fig)
    rows=list(csv.DictReader((comparison/'native_iteration_metrics.csv').open()))
    for name,keys,layout in (
        ('density_integral_curves',('max_density','background_mean','integral_recovery','p99_density','p999_density','volume_weighted_p999_density'),(4,3)),
        ('peak_location_curves',('peak_x_mm','peak_y_mm','peak_z_mm'),(2,3))):
        fig,axes=plt.subplots(*layout,figsize=(16,4*layout[0]),layout='constrained')
        for ci,channel in enumerate(CHANNELS):
            for ki,key in enumerate(keys):
                ax=axes[ci*(len(keys)//3)+ki//3,ki%3]
                for group,style in (('legacy','--'),('ideal','-')):
                    rr=[r for r in rows if r['channel']==channel and r['group']==group]
                    ax.plot([int(r['iteration']) for r in rr],[float(r[key]) for r in rr],style,label=group)
                ax.set(title=channel+'\n'+key,xlabel='iteration');ax.grid(alpha=.2);ax.legend()
                if key=='max_density':ax.set_yscale('log')
        fig.suptitle('All 40 saved frames; full 120 mm native active-domain metrics')
        fig.savefig(comparison/(name+'.png'),dpi=140);plt.close(fig)
    _,mip= axial_selection(z,8)
    (comparison/'display_contract.json').write_text(json.dumps(dict(mip=mip,common_normalizers=report['common_normalizers'],
        smoothing=False,per_panel_rescaling=False,truth_sha256=report['truth_sha256']),indent=2)+'\n')
    shutil.copy2(Path(__file__),comparison/'scripts'/Path(__file__).name)
    allfiles={p.relative_to(comparison).as_posix():digest(p) for p in comparison.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    (comparison/'artifact_manifest.json').write_text(json.dumps(allfiles,indent=2)+'\n')
    print('SUPPLEMENTAL_PAIRED_FIGURES_VERIFIED')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    a=p.parse_args();plot(a.comparison,a.results)
