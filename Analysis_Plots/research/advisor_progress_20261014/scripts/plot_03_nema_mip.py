"""Plot accepted H60 reconstruction histories. No transport or reconstruction.

Explicit figure contract: actual Geant4 projections only, both energies,
no smoothing, full XY field, central 72-mm axial MIP, no truth-fitted gains.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'experiments/ELLIPSE500x300_H120').is_dir())
H = ROOT / 'experiments/ELLIPSE500x300_H120'
OUT = Path(__file__).resolve().parents[1]
DATA = OUT / 'data'
FIGURES = OUT / 'figures'
R = H / 'reports/NEMA_Body_H60'
JSCC = H / 'generated/compton_energy_probability_v5_5e9_full10000/formal_results/1669255/continuous_energy'
sys.path.insert(0, str(H))
from analyze_nema_result import xy_interpolator
from mip_projection import axial_mip, axial_selection

NAVY = '#18283B'
TEAL = '#128C91'
BLUE = '#426CC1'
ORANGE = '#DA9952'
MUTED = '#687686'
RULE = '#E0E6EB'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + '\n').encode('utf-8'))


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def route(kind, energy, label, dose, color):
    # User definition (2026-10-14): SC means single-photon imaging.
    # It explicitly supersedes this repository's historical "SC Compton" label.
    channel = '218_SinglePhoton_CrossTalkCorrected' if energy == 218 else (
        '440_SinglePlusCompton' if kind == 'JSCC' else '440_SinglePhoton')
    if kind in ('SC', 'JSCC'):
        folder = JSCC
        authority_path = folder / 'verification.json'
        authority = read(authority_path)
        expected = next(v for v in authority['outputs'] if v['channel'] == channel)['sha256']['history']
        collection = H / 'generated/compton_energy_probability_v5_5e9_full10000/formal_payload/transport_collection.json'
        indices, frames, step = [100, 1000, 5000, 10000], 200, 50
        checkpoint_folder = 'checkpoints_218_corrected' if energy == 218 else (
            'checkpoints_compton_jscc' if kind == 'JSCC' else 'checkpoints_440_single')
        checkpoint = lambda iteration: folder / checkpoint_folder / f'checkpoint_{iteration:06d}' / f'Image_{channel}_active.float32'
    else:
        study = 'ehe_spect_5e9_200' if dose == 5_000_000_000 else 'ehe_spect_5e10_200'
        folder = H / 'generated' / study / 'results/formal'
        authority_path = R / study / 'formal_summary.json'
        authority = read(authority_path)
        expected = authority['files'][f'Image_{channel}_history.float32']
        collection = H / 'generated' / study / (
            'transport/collection.json' if dose == 5_000_000_000 else 'counts/collection.json')
        indices, frames, step = [10, 50, 100, 200], 20, 10
        checkpoint = lambda iteration: folder / 'checkpoints' / channel / f'{iteration:04d}' / 'active.float32'
    assert authority['passed'] and authority['iterations'] == frames * step
    return dict(kind=kind, energy=energy, label=label, dose=dose, color=color,
                channel=channel, folder=folder, authority_path=authority_path,
                authority=authority, expected=expected, collection=collection,
                indices=indices, frames=frames, step=step, checkpoint=checkpoint)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 14,
                         'text.color': NAVY, 'axes.labelcolor': NAVY,
                         'axes.unicode_minus': False, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'path',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    truth_path = H / 'generated/NEMA_Body_H60/truth_3mm.npz'
    manifest_path = R / 'manifest.json'
    geometry_path = H / 'generated/ehe_spect_5e9_200/payload/whole_geometry.npz'
    manifest = read(manifest_path)
    assert sha(truth_path) == manifest['truth_sha256']
    truth = np.load(truth_path)
    geometry = np.load(geometry_path)
    active, coords = geometry['active_indices'], geometry['coordinates_mm']
    assert len(active) == 78920 and coords.shape == (132040, 3)
    assert np.array_equal(np.unique(coords[:, 2]), truth['z_mm'])
    assert truth['activity_218_zyx'].shape == truth['activity_440_zyx'].shape == (40, 100, 168)
    interpolate = xy_interpolator(coords[:3301, :2], truth['x_mm'], truth['y_mm'])
    extent = [float(truth['x_mm'][0] - 1.5), float(truth['x_mm'][-1] + 1.5),
              float(truth['y_mm'][0] - 1.5), float(truth['y_mm'][-1] + 1.5)]
    _, axial_meta = axial_selection(truth['z_mm'], 8)
    bindings = {rel(p): sha(p) for p in (truth_path, manifest_path, geometry_path,
                H / 'analyze_nema_result.py', H / 'mip_projection.py')}
    routes = [route('SC', 440, 'SC', 5_000_000_000, TEAL),
              route('JSCC', 440, 'JSCC', 5_000_000_000, BLUE),
              route('EHE', 440, 'EHE', 5_000_000_000, ORANGE),
              route('EHE', 440, 'EHE', 50_000_000_000, ORANGE),
              route('SC', 218, 'SC', 5_000_000_000, TEAL),
              route('EHE', 218, 'EHE', 5_000_000_000, ORANGE),
              route('EHE', 218, 'EHE', 50_000_000_000, ORANGE)]
    panels, rows, route_meta = {}, [], []
    truth_mips = {e: axial_mip(truth[f'activity_{e}_zyx'], truth['z_mm'], 8) for e in (218, 440)}
    np.savez_compressed(DATA / '03_truth_arrays.npz',
                        x_mm=truth['x_mm'], y_mm=truth['y_mm'], z_mm=truth['z_mm'],
                        activity_218_zyx=truth['activity_218_zyx'], activity_440_zyx=truth['activity_440_zyx'])
    panels.update({f'truth_{e}': m for e, m in truth_mips.items()})
    for i, item in enumerate(routes):
        path = item['folder'] / f'Image_{item["channel"]}_history.float32'
        # This whitelist excludes all synthetic forward-plus-Poisson histories.
        assert 'forward_poisson' not in str(path) and 'Plus218' not in path.name
        assert path.stat().st_size == item['frames'] * 78920 * 4
        assert sha(path) == item['expected'], f'History identity failed: {path}'
        collection = read(item['collection'])
        primary_counts = collection['primary_counts']
        assert sum(primary_counts) == item['dose']
        e = item['energy']
        emitted_background = primary_counts[0 if e == 218 else 1] / manifest['relative_activity_integral_mm3'][str(e)]
        run_path = item['folder'] / 'run_manifest.json'
        run = read(run_path)
        assert run['iterations'] == item['frames'] * item['step'] and run['save_step'] == item['step']
        if item['kind'] == 'EHE':
            assert sha(item['collection']) == run['counts_sha256'] == item['authority']['counts_sha256']
            if item['dose'] == 50_000_000_000:
                assert item['authority']['data_kind'] == 'Geant4_transport'
                assert item['authority']['transport_closure']['actual_primary_photons'] == item['dose']
            else:
                transport_path = R / 'ehe_spect_5e9_200/transport_acceptance.json'
                transport = read(transport_path)
                assert transport['passed'] and transport['primary_counts'] == primary_counts
                assert transport['collection_sha256'] == sha(item['collection'])
                bindings[rel(transport_path)] = sha(transport_path)
        for p in (path, item['authority_path'], item['collection'], run_path):
            bindings[rel(p)] = sha(p)
        hist = np.memmap(path, dtype='<f4', mode='r', shape=(item['frames'], 78920))
        assert np.isfinite(hist).all() and not np.any(hist < 0)
        item['mips'] = [truth_mips[e]]
        for iteration in item['indices']:
            frame_index = iteration // item['step'] - 1
            assert iteration % item['step'] == 0
            values = np.asarray(hist[frame_index])
            cp = item['checkpoint'](iteration)
            assert cp.stat().st_size == values.nbytes and sha(cp) == array_sha(values)
            bindings[rel(cp)] = sha(cp)
            full = np.zeros(132040, dtype='<f4')
            full[active] = values
            assert np.array_equal(full[active], values)
            assert np.count_nonzero(full) <= len(active)
            image = interpolate(full.reshape(40, 3301)) / emitted_background
            mip = axial_mip(image, truth['z_mm'], 8)
            assert np.array_equal(mip, image[8:32].max(axis=0))
            assert np.isfinite(mip).all() and mip.min() >= -1e-6
            key = f'row{i}_{e}_{iteration}'
            panels[key] = mip
            item['mips'].append(mip)
            rows.append(dict(row=i, system=item['label'], energy_keV=e, dose=item['dose'],
                channel=item['channel'], iteration=iteration, frame_index_zero_based=frame_index,
                input_history=rel(path), history_sha256=item['expected'],
                active_frame_sha256=array_sha(values), checkpoint_identical=True,
                emitted_background_density=emitted_background, fitted_gain=False,
                display_min=float(mip.min()), display_max=float(mip.max()),
                pixels_above_display_max=int(np.count_nonzero(mip > 10)),
                displayed_mip_sha256=array_sha(mip)))
        route_meta.append(dict(row=i, system=item['kind'], energy_keV=e, dose=item['dose'],
            channel=item['channel'], source_directory=rel(item['folder']),
            iterations=item['indices'], frames_in_history=item['frames'], save_step=item['step'],
            primary_counts=primary_counts, emitted_background_density=emitted_background,
            data_kind='Geant4_transport', reconstruction_complete=True,
            physical_calibration_passed=False if item['kind'] == 'EHE' else None))
        print(f'ACCEPTED_ROW {i} {item["kind"]} {e} {item["dose"]}', flush=True)

    fig = plt.figure(figsize=(20, 13))
    fig.text(.035, .952, '03  |  NEMA Body：迭代成像', fontsize=30, weight='bold', va='top')
    fig.text(.035, .898, '实际 Geant4 投影 · 标注数量为双能初级γ总数', fontsize=17, color=MUTED)
    fig.text(.965, .898, '各图标注实际迭代次数', ha='right', fontsize=17, color=MUTED)
    fig.add_artist(Line2D([.035, .965], [.875, .875], color=RULE, lw=1.5))
    fig.add_artist(Line2D([.502, .502], [.13, .855], color=RULE, lw=1.3))
    block_x = [.035, .535]
    block_w = .430
    fig.text(block_x[0], .831, '440 keV', fontsize=23, color=BLUE, weight='bold')
    fig.text(block_x[1], .831, '218 keV', fontsize=23, color=TEAL, weight='bold')
    fig.text(block_x[1]+.155, .832, '单光子 · 串扰校正', fontsize=15, color=MUTED)
    # Equal-size tiles retain every XY pixel; no field crop or stretch.
    label_w = .072
    tile_w = (block_w - label_w) / 5
    gap = .0045
    image_h = (tile_w - gap) * 20 / 13 * (300 / 504)
    row_top = [.765, .603, .441, .279]
    rendered = []
    for i, item in enumerate(routes):
        block, row = (0, i) if i < 4 else (1, i-4)
        left = block_x[block]
        top = row_top[row]
        middle = top - image_h / 2
        fig.text(left, middle+.020, item['label'], fontsize=18, weight='bold', color=item['color'], va='center')
        fig.text(left, middle-.015, r'$5\times10^{9}$' if item['dose'] == 5_000_000_000 else r'$5\times10^{10}$',
                 fontsize=15, color=NAVY, va='center')
        if item['kind'] == 'JSCC':
            fig.text(left, middle-.048, '联合重建', fontsize=11, color=MUTED, va='center')
        elif item['kind'] == 'SC':
            fig.text(left, middle-.048, '单光子校正' if item['energy']==218 else '单光子', fontsize=11, color=MUTED, va='center')
        for col, mip in enumerate(item['mips']):
            x0 = left + label_w + col*tile_w
            ax = fig.add_axes([x0, top-image_h, tile_w-gap, image_h])
            ax.imshow(mip, origin='lower', extent=extent, cmap='gray_r', vmin=0, vmax=10,
                      interpolation='nearest', aspect='equal')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(RULE); spine.set_linewidth(.65)
            label = '真值' if col == 0 else f'{item["indices"][col-1]:,}'
            title = fig.text(x0+(tile_w-gap)/2, top+.010, label, ha='center', va='bottom',
                            fontsize=13, color=MUTED if col == 0 else NAVY)
            rendered.append(dict(row=i, column=col, actual_label=label,
                                 image_bounds=list(ax.get_position().bounds)))
        fig.add_artist(Line2D([left, left+block_w], [top-image_h-.043, top-image_h-.043], color=RULE, lw=.8))

    # Only compact geometry and display notes occupy the unused fourth 218 row.
    notes_x, notes_y = .535, .279
    fig.text(notes_x, notes_y+.010, 'NEMA Body', fontsize=18, color=NAVY, weight='bold')
    fig.text(notes_x, notes_y-.028, '300 × 230 × 60 mm     热球 / 背景  10:1', fontsize=15, color=MUTED)
    fig.text(notes_x, notes_y-.065, '中央 72 mm MIP     无平滑     全视野', fontsize=15, color=MUTED)
    fig.text(notes_x, notes_y-.102, '统一发射背景密度标尺', fontsize=14, color=MUTED)
    cax = fig.add_axes([.752, .174, .193, .013])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 10), cmap='gray_r'), cax=cax, orientation='horizontal')
    cb.set_ticks([0, 1, 5, 10]); cb.ax.tick_params(labelsize=11, length=3, color=MUTED)
    cb.outline.set_edgecolor(RULE)
    fig.text(.035, .081, 'SC：单光子成像   ·   JSCC：440 keV 单光子 + Compton 联合重建', fontsize=14, color=MUTED)
    fig.text(.035, .047, 'EHE 物理标定待完善；迭代次数不代表跨系统相同收敛程度。', fontsize=13, color=MUTED)
    fig.text(.965, .047, '218 keV 无独立联合重建通道', ha='right', fontsize=13, color=MUTED)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bad = []
    for text_obj in fig.texts:
        box = text_obj.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
            bad.append(text_obj.get_text())
    assert not bad, f'Off-canvas text: {bad}'
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(FIGURES / f'03_nema_mip_iterations.{ext}', dpi=300,
                    metadata={'Creator': 'Reproducible project history visualization'} if ext != 'svg' else None)
    plt.close(fig)
    np.savez_compressed(DATA / '03_displayed_mips.npz', **panels)
    with (DATA / '03_panel_metadata.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # Verify source bytes again after all rendering, including raw histories.
    assert all(sha(ROOT / path) == value for path, value in bindings.items())
    files = [FIGURES/f'03_nema_mip_iterations.{ext}' for ext in ('png', 'pdf', 'svg')]
    files += [DATA/'03_truth_arrays.npz', DATA/'03_displayed_mips.npz', DATA/'03_panel_metadata.csv']
    write(DATA / '03_scientific_qa.json', dict(passed=True, original_sources_unchanged=True,
        provenance_sha256=bindings, script_sha256=sha(__file__), route_metadata=route_meta,
        displayed_image_count=35, reconstruction_panels=28, truth_panels=7,
        all_selected_history_frames_equal_separate_saved_checkpoints=True,
        all_source_histories_finite_nonnegative=True, interpolation='existing linear Delaunay polar-to-Cartesian display mapping',
        truth_catalog_override='Authoritative project NEMA_Body_H60 3D truth, not packaged 2D NEMA',
        full_display_extent_mm=extent, truth_shape_zyx=[40,100,168], axial_mip=axial_meta,
        no_smoothing=True, crop_xy_pixels=0, fitted_gain=False,
        display_scale='actual energy-tagged primary photons / authoritative activity integral',
        display_range=[0,10], colormap='gray_r', rows_are_distinct_energy_channels=True,
        nomenclature_authority='User explicit definition on 2026-10-14: SC means single-photon imaging, overriding historical SC Compton nomenclature.',
        SC_channels={'440':'440_SinglePhoton','218':'218_SinglePhoton_CrossTalkCorrected'},
        labeled_dose_definition='Total initial 218 + 440 keV gamma photons summed across all 20 views; not Ac-225 decays and not the dose of each energy window separately.',
        fake_218_joint_image=False, synthetic_projection_histories_loaded=0,
        cross_energy_sums_loaded=0, reconstruction_or_transport_run=False,
        EHE_physical_calibration_claim=False, matched_iteration_convergence_claim=False,
        frame_labels_and_bounds=rendered, off_canvas_text=[],
        outputs_sha256={rel(p):sha(p) for p in files}))
    print('FIGURE_03_COMPLETE', FIGURES/'03_nema_mip_iterations.png', flush=True)


if __name__ == '__main__':
    main()
