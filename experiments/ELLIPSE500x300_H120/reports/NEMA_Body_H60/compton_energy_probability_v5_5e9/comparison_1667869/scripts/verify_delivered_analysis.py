"""Read-only scientific QA of the actual v5 delivery, run from repository root."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

E = Path('experiments/ELLIPSE500x300_H120')
R = E / 'reports/NEMA_Body_H60/compton_energy_probability_v5_5e9'
D = E / 'generated/compton_energy_probability_v5_5e9'
C = R / 'comparison_1667869'
GROUPS = ('angular', 'continuous_energy')
CHANNELS = ('440_ComptonOnly', '440_SinglePlusCompton')
ITERATIONS = tuple(range(50, 2001, 50))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def table(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    comparison = read_json(C / 'comparison.json')
    summary = read_json(R / 'formal_summary.json')
    remote = read_json(R / 'native_crosscheck_1667869.json')
    artifacts = read_json(C / 'artifact_manifest.json')
    assert summary['passed'] and summary['paired_imaging_completed']
    assert summary['job'] == comparison['job'] == remote['job'] == 1667869
    assert sha(R / 'formal_summary.json') == comparison['formal_summary_sha256']
    assert remote['geometry_sha256'] == comparison['geometry_sha256']
    assert remote['history_sha256'] == comparison['history_sha256']
    for name, digest in artifacts.items():
        assert sha(C / name) == digest, name
    source = read_json(E / 'reports/NEMA_Body_H60/manifest.json')
    truth = np.load(C / 'truth_3mm.npz')
    assert sha(C / 'truth_3mm.npz') == source['truth_sha256'] == comparison['truth_sha256']
    geometry_path = D / 'formal_payload/whole_geometry.npz'
    assert sha(geometry_path) == comparison['geometry_sha256']
    geometry = np.load(geometry_path)
    active = geometry['active_indices']
    coordinates = geometry['coordinates_mm'][active]
    volumes = geometry['cell_volume_mm3'][active].astype(np.float64)
    assert len(active) == 78920 and len(geometry['coordinates_mm']) == 132040
    assert np.all(geometry['ellipse_fraction'][active] == 1)
    native = table(C / 'native_iteration_metrics.csv')
    spheres = table(C / 'sphere_iteration_metrics.csv')
    key = lambda row: (row['group'], row['channel'], int(row['iteration']))
    expected = {(g, c, i) for g in GROUPS for c in CHANNELS for i in ITERATIONS}
    assert len(native) == 160 and {key(row) for row in native} == expected
    assert len(spheres) == 480 and {(key(row), int(row['diameter_mm'])) for row in spheres} == {
        (item, d) for item in expected for d in (13, 22, 37)}
    for rows in (native, spheres):
        for row in rows:
            for field, value in row.items():
                if field not in ('group', 'channel', 'tiny_mass_fraction'):
                    assert math.isfinite(float(value)), (key(row), field)
    remote_rows = {key(row): row for row in remote['rows']}
    assert len(remote['rows']) == 160 and set(remote_rows) == expected
    independent_errors = {field: 0.0 for field in (
        'max_density', 'peak_x_mm', 'peak_y_mm', 'peak_z_mm', 'total_integral', 'source_z_leakage')}
    roots = {group: D / f'formal_results/1667869/{group}' for group in GROUPS}
    histories = {}
    for group, root in roots.items():
        verification = read_json(root / 'verification.json')
        for name, digest in summary['models'][group]['sha256'].items():
            path = root.parent / name if name == 'allocation.txt' else root / name
            assert sha(path) == digest, (group, name)
        checkpoint_dirs = sorted(root.glob('checkpoint_*'))
        assert [p.name for p in checkpoint_dirs] == [f'checkpoint_{i:06d}' for i in ITERATIONS]
        assert len(verification['checkpoints']) == 40
        for channel in CHANNELS:
            path = root / f'Image_{channel}_history.float32'
            assert path.stat().st_size == 40 * 78920 * 4
            assert sha(path) == comparison['history_sha256'][group + '/' + channel]
            hist = np.memmap(path, '<f4', mode='r', shape=(40, 78920))
            assert np.isfinite(hist).all() and np.all(hist >= 0)
            assert np.array_equal(hist[-1], np.fromfile(root / f'Image_{channel}_active.float32', '<f4'))
            histories[group, channel] = hist
    for row in native:
        values = histories[row['group'], row['channel']][int(row['iteration']) // 50 - 1]
        index = int(values.argmax())
        mass = values.astype(np.float64) * volumes
        measured = dict(max_density=float(values[index]), total_integral=float(mass.sum(dtype=np.float64)),
                        source_z_leakage=float(mass[np.abs(coordinates[:, 2]) > 30].sum(dtype=np.float64) / mass.sum(dtype=np.float64)))
        measured.update(zip(('peak_x_mm', 'peak_y_mm', 'peak_z_mm'), coordinates[index]))
        for field, actual in measured.items():
            for reference in (float(row[field]), remote_rows[key(row)][field]):
                error = abs(actual - reference) / max(abs(reference), 1e-300)
                independent_errors[field] = max(independent_errors[field], error)
                assert error <= 1e-12, (key(row), field, error)
        assert abs(float(row['peak_background_ratio']) / (float(row['max_density']) / float(row['background_mean'])) - 1) < 1e-6
        assert row['tiny_mass_fraction'] == 'not_applicable_whole_cells'
    sphere_values = {(key(row), int(row['diameter_mm'])): float(row['crc']) for row in spheres}
    costs = []
    minimum_changes = {}
    for channel in CHANNELS:
        for diameter in (13, 22, 37):
            changes = [(i, 100 * (sphere_values[(('continuous_energy', channel, i), diameter)] -
                                  sphere_values[(('angular', channel, i), diameter)])) for i in ITERATIONS]
            minimum_changes[f'{channel}/{diameter}'] = dict(iteration=min(changes, key=lambda p: p[1])[0],
                                                            minimum_change_pp=min(v for _, v in changes))
            costs.extend(dict(channel=channel, diameter_mm=diameter, iteration=i, change_pp=v)
                         for i, v in changes if v < -5)
    peak_regions = {}
    for channel in CHANNELS:
        row = comparison['verdict'][channel]['continuous_energy_final']
        point = np.array([row[f'peak_{axis}_mm'] for axis in 'xyz'])
        containing = [s['diameter_mm'] for s in source['spheres']
                      if np.linalg.norm(point - s['center_mm']) <= s['diameter_mm'] / 2]
        z, y, x = [int(np.abs(truth[f'{axis}_mm'] - point[n]).argmin()) for axis, n in [('z', 2), ('y', 1), ('x', 0)]]
        peak_regions[channel] = dict(point_mm=point.tolist(), containing_spheres_mm=containing,
                                     nearest_truth_440=float(truth['activity_440_zyx'][z, y, x]))
    receipt = dict(passed=True, job=1667869, formal_summary_sha256=sha(R / 'formal_summary.json'),
                   comparison_sha256=sha(C / 'comparison.json'), geometry_sha256=sha(geometry_path),
                   source_manifest_sha256=sha(E / 'reports/NEMA_Body_H60/manifest.json'),
                   truth_sha256=comparison['truth_sha256'], native_rows=160, sphere_rows=480,
                   history_frames=[40] * 4, checkpoint_directories=80,
                   independent_native_max_relative_errors=independent_errors,
                   independent_method='Local fetched float64 elementwise density*whole_cell_volume then sum; compare all 160 rows with analysis dot products and independent remote receipt',
                   all_40_frame_crc_costs_over_5pp=costs, minimum_crc_changes_pp=minimum_changes,
                   candidate_peak_regions=peak_regions, common_normalizers=comparison['common_normalizers'],
                   image_review=dict(files=['iterations_center.png', 'iterations_coronal.png', 'iterations_sagittal.png',
                                            'iterations_mip72.png', 'final_multiplanar.png', 'spike_noise_leakage_curves.png',
                                            'crc_cnr_curves.png', 'tail_integral_curves.png'],
                                     reviewed_by='Codex visual inspection of actual PNGs',
                                     result='Legible axes, matching truth/group/channel labels, shared display scale, no plot clipping; wide final overview supplemented by separate galleries'),
                   filtering='No smoothing; MIP z edges [-36,36]mm only; all native quantitative metrics cover full 120mm',
                   limitations=comparison['limitation'])
    destination = R / 'final_analysis_qa_1667869.json'
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
