"""Independent read-only data QA and saved-frame CNR maxima for the atlas."""
import csv
from collections import Counter
from pathlib import Path
import shutil

from ehe_common import HERE, REPORT, digest, read, write

ATLAS = REPORT / 'comparison_full_trajectories'
OLD = REPORT / 'comparison_200'


def rows(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def verify():
    metadata = read(ATLAS / 'comparison_metadata.json')
    manifest = read(ATLAS / 'artifact_manifest.json')['files']
    for name, sha in manifest.items():
        assert digest(ATLAS / name) == sha, ('Output SHA', name)
    assert metadata['producer_sha256'] == digest(HERE / 'compare_ehe_jscc_full_trajectories.py')
    native = rows(ATLAS / 'all_native_metrics.csv')
    spheres = rows(ATLAS / 'all_sphere_metrics.csv')
    assert len(native) == 1269 and len(spheres) == 5076
    assert len([v for v in native if int(v['iteration']) == 0]) == 9
    assert len([v for v in spheres if int(v['iteration']) == 0]) == 36
    verified_fields = 0
    for system in ('EHE', 'JSCC'):
        for kind, actual in (('native', native), ('sphere', spheres)):
            path = OLD / f'{system.lower()}_{kind}_iteration_metrics.csv'
            original = rows(path)
            def key(row):
                return (system, row['channel'], row['iteration'], row.get('diameter_mm'))
            now = {key(v): v for v in actual if v['system'] == system and int(v['iteration']) > 0}
            assert len(now) == len(original)
            for old in original:
                new = now[key(old)]
                for field, value in old.items():
                    if field not in ('system', 'tiny_mass_fraction', 'nominal_truth_contrast'):
                        assert new[field] == value, ('Metric changed', key(old), field)
                        verified_fields += 1
    for row in native:
        if int(row['iteration']) == 0:
            assert row['peak_x_mm'] == row['peak_y_mm'] == row['peak_z_mm'] == ''
            assert float(row['background_cv']) == 0
            assert row['frame_provenance'] == 'known_uniform_initialization'
    for row in spheres:
        if int(row['iteration']) == 0:
            assert row['cnr'] == '' and float(row['crc']) == 0
    for route in metadata['routes']:
        system, channel = route['system'], route['channel']
        iterations = [int(v['iteration']) for v in native if v['system'] == system and v['channel'] == channel]
        step, budget = (10, 200) if system == 'EHE' else (50, 10000)
        assert iterations == list(range(0, budget + 1, step))
        for node in metadata['display_nodes']:
            if node['system'] == system and node['channel'] == channel:
                assert node['iteration'] in iterations
                assert node['frame_index'] == (None if node['iteration'] == 0 else node['iteration'] // step - 1)
    peaks = []
    for route in metadata['routes']:
        system, channel = route['system'], route['channel']
        route_spheres = [v for v in spheres if v['system'] == system and v['channel'] == channel and v['cnr'] != '']
        for diameter in sorted({int(v['diameter_mm']) for v in route_spheres}):
            rr = [v for v in route_spheres if int(v['diameter_mm']) == diameter]
            best = max(rr, key=lambda v: float(v['cnr']))
            end = max(rr, key=lambda v: int(v['iteration']))
            n = next(v for v in native if v['system'] == system and v['channel'] == channel and v['iteration'] == best['iteration'])
            peaks.append(dict(system=system, channel=channel, sphere_diameter_mm=diameter,
                              saved_frame_CNR_peak_iteration=best['iteration'], saved_frame_CNR_peak=best['cnr'],
                              CRC_at_CNR_peak=best['crc'], background_CV_at_CNR_peak=n['background_cv'],
                              budget_endpoint_iteration=end['iteration'], budget_endpoint_CNR=end['cnr'],
                              provenance='maximum among actual saved frames, not an unsaved/global optimum or stopping prescription'))
    with (ATLAS / 'cnr_peaks_saved_frames.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, list(peaks[0])); writer.writeheader(); writer.writerows(peaks)
    proof = dict(passed=True, metric_fields_compared_unchanged=verified_fields,
                 metric_rows={'native': len(native), 'sphere': len(spheres)},
                 complete_independent_iterations=True, saved_nodes_mapping=True,
                 initialization_not_saved_snapshot=True, undefined_CNR0_and_peak_xyz_omitted=True,
                 original_input_sha_checks=metadata['scientific_checks'],
                 artifact_files_verified=len(manifest),
                 analysis='CNR maxima only among available saved frames; no equality of convergence is asserted',
                 CNR_peak_iteration_range={s: [min(int(v['saved_frame_CNR_peak_iteration']) for v in peaks if v['system'] == s),
                                             max(int(v['saved_frame_CNR_peak_iteration']) for v in peaks if v['system'] == s)]
                                           for s in ('EHE', 'JSCC')},
                 verifier_sha256=digest(Path(__file__)), producer_sha256=metadata['producer_sha256'])
    write(ATLAS / 'scientific_qa.json', proof)
    shutil.copy2(Path(__file__), ATLAS / Path(__file__).name)
    print(proof)


if __name__ == '__main__':
    verify()
