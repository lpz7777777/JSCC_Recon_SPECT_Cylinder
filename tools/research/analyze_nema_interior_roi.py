"""Re-measure completed H60 histories under the current ROI/output policy.

No transport, response generation, reconstruction or cross-energy image sums.
The historical report and its accepted artifacts are never overwritten.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / 'experiments/ELLIPSE500x300_H120'
sys.path.insert(0, str(H))
from nema_roi_policy import build_masks, measure, POLICY_ID
from reconstruction_output_policy import EHE_CHANNELS, JSCC_CHANNELS, RETIRED_SUM_CHANNELS

OUT = H / 'reports/nema_interior_roi_20261010'
R = H / 'reports/NEMA_Body_H60'
OLD = H / 'reports/dual_energy_review_20261010/scientific_sources.json'
LATEST = R / 'ehe_forward_poisson_5e10_200/comparison/comparison_report.json'
STUDIES = {
    'EHE': 'ehe_spect_5e9_200',
    'EHE matrix+Poisson': 'ehe_forward_poisson_5e9_200',
    'EHE Geant4 5e10': 'ehe_spect_5e10_200',
    'EHE matrix+Poisson 5e10': 'ehe_forward_poisson_5e10_200',
}
LABELS = {'EHE': 'G4 5e9', 'EHE matrix+Poisson': 'Poisson 5e9',
    'EHE Geant4 5e10': 'G4 5e10', 'EHE matrix+Poisson 5e10': 'Poisson 5e10', 'JSCC': 'JSCC 5e9'}
CHANNEL_LABELS = {'440_SinglePhoton': '440 single',
    '218_SinglePhoton_CrossTalkCorrected': '218 corrected single',
    '440_ComptonOnly': '440 Compton', '440_SinglePlusCompton': '440 JSCC'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csv_write(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def routes():
    jscc = H / 'generated/compton_energy_probability_v5_5e9_full10000/formal_results/1669255/continuous_energy'
    for system, study in STUDIES.items():
        folder = H / 'generated' / study / 'results/formal'
        authority_path = R / study / 'formal_summary.json'
        authority = read(authority_path)
        if not authority['passed'] or authority['iterations'] != 200:
            raise ValueError('Completed formal200 authority is missing')
        for channel in EHE_CHANNELS:
            name = f'Image_{channel}_history.float32'
            yield system, channel, folder/name, authority['files'][name], 20, 10, authority_path
    authority_path = jscc / 'verification.json'; authority = read(authority_path)
    if not authority['passed'] or authority['iterations'] != 10000:
        raise ValueError('Completed JSCC formal10000 authority is missing')
    for channel in JSCC_CHANNELS:
        item = next(v for v in authority['outputs'] if v['channel'] == channel)
        yield 'JSCC', channel, jscc/f'Image_{channel}_history.float32', item['sha256']['history'], 200, 50, authority_path


def plot(output, rows, spheres, masks, truth, meta, galleries):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from mip_projection import axial_mip
    output.mkdir(exist_ok=True)
    hot = [r for r in spheres if r['region_kind'] == 'hot']
    for system_type in ('ehe', 'jscc'):
        for metric in ('cnr', 'crc'):
            fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
            for ax, d in zip(axes.flat, (10, 17, 28, 13, 22, 37)):
                channels = (EHE_CHANNELS[1],) if d in (10,17,28) else JSCC_CHANNELS[::2] + ('440_SinglePlusCompton',)
                # JSCC_CHANNELS[::2] contains exactly 440 single and Compton.
                pairs = [(s, EHE_CHANNELS[1] if d in (10,17,28) else EHE_CHANNELS[0]) for s in STUDIES] if system_type == 'ehe' else [('JSCC', c) for c in channels]
                for system, channel in pairs:
                    rr = [r for r in hot if (r['system'],r['channel'],r['diameter_mm']) == (system,channel,d)]
                    label = LABELS[system] if system_type == 'ehe' else CHANNEL_LABELS[channel]
                    ax.plot([r['iteration'] for r in rr], [np.nan if r[metric] is None else r[metric] for r in rr], label=label)
                ax.set(title=f'{d} mm / {218 if d in (10,17,28) else 440} keV', xlabel='Actual iterations', ylabel=metric.upper(), xlim=(0,200 if system_type == 'ehe' else 10000))
                if metric == 'crc':
                    ax.axhline(1,color='gray',lw=.7,ls=':')
                ax.grid(alpha=.2)
            axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
            fig.suptitle(f'{system_type.upper()} | sphere centers at least 1.5 mm inside; one common background',fontsize=12)
            fig.savefig(output/f'{system_type}_hot_{metric}.png', dpi=180);plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(11,6.5),layout='constrained')
    for i, e in enumerate((218,440)):
        for j, st in enumerate(('EHE','JSCC')):
            channels = (EHE_CHANNELS[1],) if e==218 else (JSCC_CHANNELS[0],JSCC_CHANNELS[2],JSCC_CHANNELS[3])
            pairs = [(s,EHE_CHANNELS[1] if e==218 else EHE_CHANNELS[0]) for s in STUDIES] if st=='EHE' else [('JSCC',c) for c in channels]
            for system, channel in pairs:
                rr = [r for r in rows if (r['system'],r['channel'])==(system,channel)]
                axes[i,j].plot([r['iteration'] for r in rr],[r['background_cv'] for r in rr],label=LABELS[system] if st=='EHE' else CHANNEL_LABELS[channel])
            axes[i,j].set(title=f'{st} / {e} keV',xlabel='Actual iterations',ylabel='Common background CV',xlim=(0,200 if st=='EHE' else 10000))
            axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=8)
    fig.suptitle(f'Same {int(masks["background"].sum())} background voxels for every sphere/channel/system')
    fig.savefig(output/'common_background_cv.png',dpi=180);plt.close(fig)
    z = truth['z_mm'];k=int(np.flatnonzero(z==1.5)[0]);x=truth['x_mm'];y=truth['y_mm']
    extent=(x[0]-1.5,x[-1]+1.5,y[0]-1.5,y[-1]+1.5)
    fig, axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    bg=masks['background'][k]
    axes[0].imshow(bg,origin='lower',extent=extent,cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
    selected=np.zeros_like(bg,dtype=np.int8)
    for mask in masks['spheres'].values():selected+=mask[k]
    axes[1].imshow(selected,origin='lower',extent=extent,cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
    for ax in axes:
        for s in meta['spheres']:
            cx,cy,_=s['center_mm'];ax.add_patch(Circle((cx,cy),s['diameter_mm']/2,fill=False,ec='#c23b36',lw=.9))
            ax.text(cx,cy,int(s['diameter_mm']),ha='center',va='center',fontsize=8,color='#1672a2')
        ax.add_patch(Circle((0,0),25.5,fill=False,ec='#1672a2',lw=.9))
        ax.set(xlabel='Object x (mm)',ylabel='Object y (mm)',xlim=(-160,160),ylim=(-120,120))
    axes[0].set_title('Common background / z=+1.5 mm');axes[1].set_title('Sphere center ROIs / z=+1.5 mm')
    fig.savefig(output/'roi_masks.png',dpi=180);plt.close(fig)
    for system in (*STUDIES,'JSCC'):
        channels = EHE_CHANNELS if system!='JSCC' else JSCC_CHANNELS
        nodes = (0,50,100,150,200) if system!='JSCC' else (0,2000,5000,7500,10000)
        height=1.15*len(channels)+.7
        fig,axes=plt.subplots(len(channels),6,figsize=(12,height),squeeze=False)
        fig.subplots_adjust(left=.1,right=.92,top=1-.65/height,bottom=.1/height,wspace=.08,hspace=.16)
        for row,channel in enumerate(channels):
            e=218 if channel==EHE_CHANNELS[1] else 440
            images=[truth[f'activity_{e}_zyx']]+[galleries[(system,channel)][n] for n in nodes]
            for col,image in enumerate(images):
                ax=axes[row,col];handle=ax.imshow(axial_mip(image,z,8),origin='lower',extent=extent,cmap='gray_r',vmin=0,vmax=10,interpolation='nearest',aspect='equal')
                if row==0:ax.set_title('H60 truth' if col==0 else f'Iteration {nodes[col-1]}',fontsize=9)
                ax.set_xticks([]);ax.set_yticks([])
            pos=axes[row,0].get_position();fig.text(.006,(pos.y0+pos.y1)/2,CHANNEL_LABELS[channel].replace(' ','\n'),fontsize=8,va='center')
        cb=fig.add_axes([.945,.2,.012,.56]);fig.colorbar(handle,cax=cb,label='Density / emitted background (0-10)')
        fig.suptitle(LABELS[system]+' | central 72 mm MIP, full XY field\nSeparate energy channels; no smoothing or fitted gain',fontsize=11)
        fig.savefig(output/f'gallery_{list((*STUDIES,"JSCC")).index(system)}.png',dpi=180);plt.close(fig)


def main():
    import numpy as np
    from analyze_nema_result import xy_interpolator
    started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--refresh-derived',action='store_true',help='Refresh only this dated, derived analysis; never original results')
    a=parser.parse_args()
    if a.output.exists():
        if not a.refresh_derived or a.output.resolve()!=OUT.resolve() or read(a.output/'scientific_acceptance.json')['policy_id']!=POLICY_ID:
            raise ValueError('Refusing to overwrite an unrelated or historical result directory')
    a.output.mkdir(parents=True,exist_ok=True)
    old=read(OLD);config=read(H/'nema_body_h60_config.json');meta=read(R/'manifest.json')
    truth_path=H/'generated/NEMA_Body_H60/truth_3mm.npz';geometry_path=H/'generated/ehe_spect_5e9_200/payload/whole_geometry.npz'
    if sha(truth_path)!=old['truth_sha256'] or sha(truth_path)!=meta['truth_sha256'] or sha(geometry_path)!=old['geometry_sha256']:
        raise ValueError('Actual truth or registered geometry changed')
    if sha(LATEST)!=old['csv_sha256'][str(LATEST.relative_to(ROOT))]:
        raise ValueError('Original emitted-density scale record changed')
    truth=np.load(truth_path);g=np.load(geometry_path);masks=build_masks(truth,meta,config)
    active=g['active_indices'];coords=g['coordinates_mm'];scales=read(LATEST)['density_scales']
    if len(active)!=78920 or len(coords)!=132040 or not np.array_equal(np.unique(coords[:,2]),truth['z_mm']):
        raise ValueError('Complete grid or matching z centers differ')
    interpolate=xy_interpolator(coords[:3301,:2],truth['x_mm'],truth['y_mm'])
    roi_record=dict(policy_id=POLICY_ID,pitch_mm=3,sphere_rule='center distance <= radius - 1.5 mm',
        sphere_center_margin_mm=1.5,extra_erosion_mm=0,sphere_cube_corners_may_cross_surface=True,
        background_rule='whole voxel strictly inside inner body; no contact/intersection with any sphere or lung cylinder',
        background_shared_by_all_spheres=True,background_voxels=int(masks['background'].sum()),
        background_std_ddof=1,lung_excluded=True,boundary_contact_excluded_from_background=True,
        background_z_centers_mm=truth['z_mm'][masks['background'].any(axis=(1,2))].tolist(),
        background_mask_sha256=hashlib.sha256(masks['background'].tobytes()).hexdigest(),spheres=[])
    for s in meta['spheres']:
        d=int(s['diameter_mm']);mask=masks['spheres'][d];fraction=truth[f'sphere_{d}_fraction_zyx'][mask]
        own=int(s['hot_energy_keV']);sampled=truth[f'activity_{own}_zyx'][mask].astype(float)
        roi_record['spheres'].append(dict(diameter_mm=d,energy_keV=own,center_mm=s['center_mm'],voxels=int(mask.sum()),
            mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),truth_fraction_min=float(fraction.min()) if len(fraction) else None,
            sampled_truth_hot_mean=float(sampled.mean()) if len(sampled) else None,
            sampled_truth_crc=(float(sampled.mean())-1)/9 if len(sampled) else None))
    write(a.output/'roi_definition.json',roi_record)
    rows=[];sphere_rows=[];bindings={};proofs={};galleries={};coverage=[]
    for system,channel,path,expected,frames,step,authority_path in routes():
        if channel in RETIRED_SUM_CHANNELS or 'Plus218' in channel:
            raise ValueError('Cross-energy sum route must never be opened')
        if expected!=old['histories_sha256'][str(path.relative_to(ROOT))] or path.stat().st_size!=frames*78920*4 or sha(path)!=expected:
            raise ValueError('Complete original history SHA/length differs: '+str(path))
        bindings[str(path.relative_to(ROOT))]=expected;proofs[str(authority_path.relative_to(ROOT))]=sha(authority_path)
        hist=np.memmap(path,dtype='<f4',mode='r',shape=(frames,78920));e=218 if channel==EHE_CHANNELS[1] else 440
        gallery_nodes=(0,50,100,150,200) if system!='JSCC' else (0,2000,5000,7500,10000)
        galleries[(system,channel)]={}
        for k in range(frames+1):
            n=k*step;v=np.ones(78920,dtype='<f4') if k==0 else hist[k-1]
            if not np.isfinite(v).all() or np.any(v<0):raise ValueError('Original history has invalid values')
            full=np.zeros(132040,dtype='<f4');full[active]=v;image=interpolate(full.reshape(40,3301))
            base,rr=measure(image,masks,meta,e)
            # Confirm every sphere uses exactly the same mean and sample std.
            if any(r['background_mean']!=base['background_mean'] or r['background_std']!=base['background_std'] for r in rr):
                raise ValueError('Background changed between sphere ROIs')
            rows.append(dict(system=system,channel=channel,iteration=n,energy_keV=e,**base,
                emitted_background_density=scales[system][str(e)],background_mean_over_emitted=base['background_mean']/scales[system][str(e)]))
            sphere_rows.extend(dict(system=system,channel=channel,iteration=n,**r) for r in rr)
            if n in gallery_nodes:galleries[(system,channel)][n]=image/scales[system][str(e)]
        coverage.append(dict(system=system,channel=channel,frames=frames+1,iterations_start=0,iterations_end=frames*step,save_step=step,
            iteration_zero='all-one active density; no iteration rerun'))
        print('ROI_HISTORY_COMPLETE',system,channel,frames+1,flush=True)
    csv_write(a.output/'common_background_iteration_metrics.csv',rows)
    csv_write(a.output/'sphere_iteration_metrics.csv',sphere_rows)
    endpoints=[];peaks=[]
    for system,channel,*_ in routes():
        for s in meta['spheres']:
            d=int(s['diameter_mm']);ss=[r for r in sphere_rows if (r['system'],r['channel'],r['diameter_mm'])==(system,channel,d)]
            endpoints.append(ss[-1])
            valid=[r for r in ss if r['cnr'] is not None]
            if ss[-1]['region_kind']=='hot' and valid:
                peak=max(valid,key=lambda r:r['cnr'])
                peaks.append(dict(system=system,channel=channel,diameter_mm=d,peak_iteration=peak['iteration'],peak_cnr=peak['cnr'],
                    endpoint_iteration=ss[-1]['iteration'],endpoint_cnr=ss[-1]['cnr'],endpoint_crc=ss[-1]['crc']))
    csv_write(a.output/'sphere_endpoint_metrics.csv',endpoints);csv_write(a.output/'hot_sphere_cnr_peak_and_final.csv',peaks)
    plot(a.output/'figures',rows,sphere_rows,masks,truth,meta,galleries)
    code={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),H/'nema_roi_policy.py',H/'reconstruction_output_policy.py',H/'mip_projection.py')}
    analysis_names={'roi_definition.json','common_background_iteration_metrics.csv','sphere_iteration_metrics.csv',
        'sphere_endpoint_metrics.csv','hot_sphere_cnr_peak_and_final.csv'}
    outputs={str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file() and
        (p.parent==a.output/'figures' or p.name in analysis_names)}
    write(a.output/'scientific_acceptance.json',dict(passed=True,policy_id=POLICY_ID,code_sha256=code,
        histories_sha256=bindings,authorities_sha256=proofs,truth_sha256=sha(truth_path),geometry_sha256=sha(geometry_path),
        source_manifest_sha256=sha(OLD),phantom_manifest_sha256=sha(R/'manifest.json'),source_scale_record_sha256=sha(LATEST),source_config_sha256=sha(H/'nema_body_h60_config.json'),
        original_interpolator_sha256=sha(H/'analyze_nema_result.py'),route_coverage=coverage,total_routes=len(coverage),
        iteration_rows=len(rows),sphere_rows=len(sphere_rows),endpoint_rows=len(endpoints),background_identical_within_each_frame=True,
        sum_histories_read=0,cross_energy_sum_images_computed=0,reconstruction_run=False,transport_run=False,response_run=False,
        EHE_iterations=[0,200],JSCC_iterations=[0,10000],matched_iteration_convergence_claim=False,
        no_smoothing=True,crop=0,fitted_gain=False,display_scale='original emitted source density',display_range=[0,10],
        hot_crc_denominator=9,cold_crc_definition='1 - sphere_mean / background_mean',cnr_definition='signed (sphere_mean - background_mean) / spatial sample std',
        outputs_sha256=outputs,elapsed_seconds=time.monotonic()-started))
    print('CURRENT_ROI_ANALYSIS_COMPLETE',len(coverage),len(rows),len(sphere_rows),flush=True)


if __name__=='__main__':main()
