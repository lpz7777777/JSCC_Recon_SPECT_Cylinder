"""Independent raw-array metric identities and file/layout inventory for the new atlas."""
import csv
import numpy as np
from PIL import Image
from ehe_common import HERE, CHANNELS, read, write, digest, verify_files
from ehe_5e10_workflow import DATA, REPORT
from plot_ehe_5e10 import acquisition_budget


def verify():
    out=REPORT/'comparison';manifest=read(out/'artifact_manifest.json');verify_files(out,manifest['files'])
    report=read(out/'comparison_report.json');proof=read(REPORT/'formal_summary.json')
    verify_files(DATA/'results/formal',proof['files'])
    if report['physical_calibration_claim'] or report['actual_source_dose']!=5e10 or report['fitted_gain'] or report['crop']!=0 or report['smoothing_sigma']!=0:
        raise ValueError('Scientific display/dose contract differs')
    g=np.load(DATA/'payload/whole_geometry.npz');active=g['active_indices'];vol=g['cell_volume_mm3'][active];coords=g['coordinates_mm'][active]
    with (out/'transport_native_metrics.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    checked=0
    for c in CHANNELS:
        rr=[v for v in rows if v['channel']==c]
        if [int(v['iteration']) for v in rr]!=list(range(0,201,10)):raise ValueError('Complete 0-200 trajectory missing')
        hist=np.fromfile(DATA/'results/formal'/f'Image_{c}_history.float32','<f4').reshape(20,78920)
        e='218' if c==CHANNELS[1] else 'sum' if c==CHANNELS[2] else '440'
        budget=5e10 if e=='sum' else acquisition_budget()[1][e]
        for row in rr:
            iteration=int(row['iteration']);v=np.ones(78920,np.float64)*(2 if e=='sum' else 1) if iteration==0 else hist[iteration//10-1].astype(np.float64)
            integral=np.sum(v*vol);order=np.argsort(v);leak=np.sum(v[abs(coords[:,2])>30]*vol[abs(coords[:,2])>30])/integral
            expected=dict(total_integral=integral,integral_recovery=integral/budget,max_density=float(v.max()),p99_density=np.quantile(v,.99),p999_density=np.quantile(v,.999),source_z_leakage=leak,
                volume_weighted_p999_density=np.interp(.999,np.cumsum(vol[order])/vol.sum(),v[order]))
            for key,value in expected.items():
                if not np.isclose(float(row[key]),value,rtol=1e-12,atol=1e-9):raise ValueError('Native metric/raw identity differs: '+key)
                checked+=1
            if iteration:
                peak=coords[int(v.argmax())]
                if any(float(row['peak_'+axis+'_mm'])!=peak[i] for i,axis in enumerate('xyz')):raise ValueError('Peak coordinate differs')
            elif any(row['peak_'+axis+'_mm']!='' for axis in 'xyz'):raise ValueError('Uniform initialization peak is nonunique')
    with (out/'transport_sphere_metrics.csv').open(encoding='utf-8') as f:spheres=list(csv.DictReader(f))
    if len(spheres)!=252 or any(v['cnr']!='' for v in spheres if v['iteration']=='0'):raise ValueError('Sphere frames/undefined init CNR differ')
    # Re-evaluate the actual 3D fractional sphere and local background ROIs
    # in float64, independently of the plotting module's metric loop.
    from analyze_nema_result import xy_interpolator
    truth=np.load(DATA/'payload/truth_3mm.npz');meta=read(HERE/'reports/NEMA_Body_H60/manifest.json')
    if digest(DATA/'payload/truth_3mm.npz')!=meta['truth_sha256']:raise ValueError('Authoritative 3D truth differs')
    x,y,z=[truth[k+'_mm'] for k in 'xyz'];interp=xy_interpolator(g['coordinates_mm'][:3301,:2],x,y)
    fractional_bg=np.maximum(truth['body_fraction_zyx']-truth['lung_fraction_zyx']-
        sum(truth[f'sphere_{d}_fraction_zyx'] for d in (10,13,17,22,28,37)),0)
    xx,yy=np.meshgrid(x,y);roi={}
    for sphere in meta['spheres']:
        d=int(sphere['diameter_mm']);cx,cy,cz=sphere['center_mm']
        roi[d]=(((xx-cx)**2+(yy-cy)**2<=(d/2+25)**2)[None]&(abs(z[:,None,None]-cz)<=d/2+3)&(fractional_bg>=.99))
    global_bg=(fractional_bg>=.99)&(abs(z[:,None,None])<=25.5)
    density=acquisition_budget()[2];total_density=sum(density.values());roi_checks=0;background_checks=0
    for c in CHANNELS:
        hist=np.fromfile(DATA/'results/formal'/f'Image_{c}_history.float32','<f4').reshape(20,78920)
        for iteration in range(0,201,10):
            values=np.full(78920,2 if c==CHANNELS[2] else 1,'<f4') if iteration==0 else hist[iteration//10-1]
            full=np.zeros(132040,'<f4');full[active]=values;im=interp(full.reshape(40,3301)).astype(np.float64)
            b=im[global_bg];mean=b.mean();row=next(v for v in rows if v['channel']==c and int(v['iteration'])==iteration)
            for key,value in dict(background_mean=mean,background_cv=b.std(ddof=1)/mean,peak_background_ratio=values.max()/mean).items():
                if not np.isclose(float(row[key]),value,rtol=1e-6,atol=1e-6):raise ValueError('Independent float64 background metric differs: '+key)
                background_checks+=1
            for row in [v for v in spheres if v['channel']==c and int(v['iteration'])==iteration]:
                d=int(row['diameter_mm']);local=im[roi[d]];mean=local.mean();sd=local.std(ddof=1)
                weights=truth[f'sphere_{d}_fraction_zyx'].astype(np.float64);hot=np.sum(im*weights)/weights.sum()
                nominal=9 if c!=CHANNELS[2] else 10*density[row['energy_keV']]/total_density-1
                expected=dict(hot_mean=hot,local_background_mean=mean,crc=(hot/mean-1)/nominal)
                if iteration:expected['cnr']=(hot-mean)/sd
                for key,value in expected.items():
                    if not np.isclose(float(row[key]),value,rtol=1e-6,atol=1e-6):raise ValueError('Independent float64 3D ROI metric differs: '+key)
                    roi_checks+=1
    inventory={}
    for name in sorted(manifest['files']):
        if not name.endswith('.png'):continue
        with Image.open(out/name) as im:
            im.verify()
        with Image.open(out/name) as im:
            array=np.asarray(im.convert('RGB'));height,width=array.shape[:2]
            if min(height,width)<500 or not np.any(array<200):raise ValueError('Empty or undersized scientific figure')
            inventory[name]=dict(width=width,height=height,sha256=digest(out/name))
    if len(inventory)!=9:raise ValueError('Expected nine result figures')
    proof=dict(passed=True,native_scalar_identities_checked=checked,independent_float64_background_scalar_checks=background_checks,independent_float64_sphere_roi_scalar_checks=roi_checks,roi_metric_tolerance=1e-6,native_rows=63,sphere_rows=252,
               zero_peak_undefined=True,zero_cnr_undefined=True,figures=inventory,
               qa_source_sha256=digest(__file__),artifact_manifest_sha256=digest(out/'artifact_manifest.json'),
               visual_review='Pending direct viewing of every scientific figure')
    write(REPORT/'scientific_figure_acceptance.json',proof);print('RAW_METRIC_AND_IMAGE_FILE_QA_PASS')


if __name__=='__main__':verify()
