"""Read-only scientific QA of accepted histories and their displayed metrics."""
import csv
import math
from pathlib import Path
import numpy as np
from ehe_common import *
from analyze_nema_result import xy_interpolator
from mip_projection import axial_selection


def close(actual, expected, label):
    if not math.isfinite(float(actual)) or not np.isclose(actual, expected, rtol=1e-6, atol=1e-7):
        raise ValueError('Scientific output differs: ' + label)


def verify():
    folder = REPORT / 'comparison_200'
    manifest = read(folder / 'artifact_manifest.json')
    verify_files(folder, manifest['files'])
    formal = read(REPORT / 'formal_summary.json')
    disclosure = read(folder / 'comparison_report.json')
    result = DATA / 'results/formal'
    if not formal['passed'] or formal['iterations'] != 200 or formal['frames_per_channel'] != 20:
        raise ValueError('Actual full formal numerical acceptance required')
    if disclosure['formal_proof_sha256'] != digest(REPORT / 'formal_summary.json'):
        raise ValueError('Figure result authority differs')
    if formal['physical_calibration_passed'] is not False or disclosure['physical_calibration_passed'] is not False:
        raise ValueError('Original physical limitation must remain disclosed')
    if (disclosure['crop'], disclosure['smoothing_sigma']) != (0, 0):
        raise ValueError('Fixed unfiltered complete display contract differs')
    meta = read(TRUTH_META)
    if digest(TRUTH) != meta['truth_sha256'] or disclosure['truth_sha256'] != meta['truth_sha256']:
        raise ValueError('Existing H60 3D truth differs')
    truth = np.load(TRUTH)
    g = np.load(DATA / 'payload/whole_geometry.npz')
    coords, active = g['coordinates_mm'], g['active_indices']
    volume = g['cell_volume_mm3'][active]
    if len(active) != 78920 or coords.shape != (132040, 3) or np.any(volume <= 0):
        raise ValueError('Native whole geometry differs')
    x, y, z = [truth[k + '_mm'] for k in 'xyz']
    if tuple(meta['shape_zyx']) != (40, len(y), len(x)) or len(z) != 40:
        raise ValueError('Authoritative 3D shape differs')
    selection, mip = axial_selection(z, 8)
    if mip['retained_slab_bounds_mm'] != [-36., 36.] or mip['retained_layer_count'] != 24:
        raise ValueError('72 mm display-only MIP differs')
    interpolate = xy_interpolator(coords[:3301, :2], x, y)
    bg_fraction = np.maximum(truth['body_fraction_zyx'] - truth['lung_fraction_zyx'] -
        sum(truth[f'sphere_{d}_fraction_zyx'] for d in (10, 13, 17, 22, 28, 37)), 0)
    bg = (bg_fraction >= .99) & (abs(z[:, None, None]) <= 25.5)
    xx, yy = np.meshgrid(x, y)
    local_roi = {}
    for sphere in meta['spheres']:
        d = int(sphere['diameter_mm']); cx, cy, cz = sphere['center_mm']
        local_roi[d] = (((xx-cx)**2+(yy-cy)**2 <= (d/2+25)**2)[None] &
            (abs(z[:, None, None]-cz) <= d/2+3) & (bg_fraction >= .99))
    with (folder / 'ehe_native_iteration_metrics.csv').open() as f:
        native = {(r['channel'], int(r['iteration'])): r for r in csv.DictReader(f)}
    with (folder / 'ehe_sphere_iteration_metrics.csv').open() as f:
        sphere_rows = {(r['channel'], int(r['iteration']), int(r['diameter_mm'])): r for r in csv.DictReader(f)}
    if len(native) != 60 or len(sphere_rows) != 240:
        raise ValueError('All 20-frame native/spherical metrics required')
    collection = read(DATA / 'transport/collection.json')
    primary = collection['primary_counts']
    for i, energy in enumerate((218, 440)):
        close(disclosure['density_scales']['EHE'][str(energy)],
              primary[i] / meta['relative_activity_integral_mm3'][str(energy)], 'emitted-source scale')
    scales = disclosure['density_scales']['EHE']
    close(scales['sum'], scales['218'] + scales['440'], 'dual density scale')
    finals = {}
    for channel in CHANNELS:
        path = result / f'Image_{channel}_history.float32'
        if path.stat().st_size != 20*78920*4 or digest(path) != disclosure['history_sha256']['EHE/'+channel]:
            raise ValueError('Actual 20 saved frame identity differs')
        h = np.fromfile(path, '<f4').reshape(20, 78920)
        if np.any(~np.isfinite(h)) or np.any(h < 0):
            raise ValueError('Finite nonnegative native history required')
        energy = 218 if channel == CHANNELS[1] else 'sum' if channel == CHANNELS[2] else 440
        for index, image in enumerate(h):
            iteration = (index+1)*10
            row = native[channel, iteration]
            full = np.zeros(132040, 'f4'); full[active] = image
            cart = interpolate(full.reshape(40, 3301))
            b = cart[bg]; mean = float(b.mean()); std = float(b.std(ddof=1))
            integral = float(np.sum(image.astype(float)*volume, dtype=np.float64))
            peak = int(np.argmax(image))
            order = np.argsort(image)
            expected = dict(max_density=float(image[peak]), background_mean=mean,
                background_cv=std/mean, peak_background_ratio=float(image[peak]/mean),
                total_integral=integral,
                integral_recovery=integral/(sum(primary[:2]) if energy=='sum' else primary[0 if energy==218 else 1]),
                source_z_leakage=float(np.sum(image[np.abs(coords[active, 2])>30].astype(float)*
                    volume[np.abs(coords[active, 2])>30], dtype=np.float64)/integral),
                p99_density=float(np.quantile(image, .99)), p999_density=float(np.quantile(image, .999)),
                volume_weighted_p999_density=float(np.interp(.999, np.cumsum(volume[order])/volume.sum(), image[order])))
            for axis, value in zip('xyz', coords[active[peak]]): expected[f'peak_{axis}_mm'] = float(value)
            for key, value in expected.items(): close(float(row[key]), value, channel+'/'+str(iteration)+'/'+key)
            for sphere in meta['spheres']:
                own, d = sphere['hot_energy_keV'], int(sphere['diameter_mm'])
                if energy != 'sum' and energy != own: continue
                sr = sphere_rows[channel, iteration, d]
                weight = truth[f'sphere_{d}_fraction_zyx']
                hot = float(np.sum(cart*weight, dtype=np.float64)/np.sum(weight, dtype=np.float64))
                local = cart[local_roi[d]]; lm = float(local.mean()); ls = float(local.std(ddof=1))
                contrast = 9 if energy != 'sum' else 10*scales[str(own)]/scales['sum']-1
                for key, value in dict(hot_mean=hot, local_background_mean=lm,
                        crc=(hot/lm-1)/contrast, cnr=(hot-lm)/ls).items():
                    close(float(sr[key]), value, channel+'/'+str(iteration)+'/'+str(d)+'/'+key)
        finals[channel] = dict(native[channel, 200])
    fixed = np.load(result / 'fixed_cross_background.npy')
    close(disclosure['ehe_cross_diagnostic']['modeled_218_background_sum'], float(fixed.astype(float).sum()), 'fixed cross background sum')
    if disclosure['ehe_cross_diagnostic']['fixed_background_sha256'] != digest(result/'fixed_cross_background.npy'):
        raise ValueError('Final440-derived fixed background identity differs')
    for name, expected in [('ehe_window_and_background_by_view.csv', 20),
                           ('jscc_native_iteration_metrics.csv', 1200),
                           ('jscc_sphere_iteration_metrics.csv', 4800)]:
        with (folder/name).open() as f: rows=list(csv.DictReader(f))
        if len(rows)!=expected: raise ValueError('Complete comparison table differs: '+name)
    images = sorted(p.name for p in folder.glob('*.png'))
    if len(images) != 28: raise ValueError('Complete gallery/curve set required')
    proof = dict(scientific_output_qa_passed=True, visual_qa_pending=True,
        physical_calibration_passed=False, formal_job=read(REPORT/'formal_job.json')['job'],
        native_rows=60, spherical_rows=240, jscc_frames_per_channel=200, figures=images,
        truth_sha256=digest(TRUTH), truth_dimensions_zyx=meta['shape_zyx'],
        mip_display=mip, crop=0, smoothing_sigma=0, fitted_image_gains=False,
        final_native_metrics=finals, verifier_sha256=digest(__file__),
        formal_authority_sha256=digest(REPORT/'formal_summary.json'),
        comparison_manifest_sha256=digest(folder/'artifact_manifest.json'))
    write(REPORT/'comparison_scientific_qa.json', proof)
    print('EHE_COMPARISON_SCIENTIFIC_QA_PASS; visual inspection still required')


if __name__ == '__main__':
    verify()
