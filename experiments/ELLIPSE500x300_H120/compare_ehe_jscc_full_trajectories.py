"""Read-only derivative atlas: separate EHE0-200 and JSCC0-10000 histories.

Iteration zero is the SHA-bound uniform solver initialization, never a claimed
saved snapshot. Positive iterations and metrics are taken from accepted files.
This module neither invokes a solver nor accesses remote systems.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

from ehe_common import HERE, ROOT, DATA, REPORT, TRUTH, TRUTH_META, digest, read, write
from analyze_nema_result import xy_interpolator
sys.path.insert(0, str(ROOT))
from mip_projection import axial_mip

OLD = REPORT / 'comparison_200'
OUT = REPORT / 'comparison_full_trajectories'
JREPORT = HERE / 'reports/NEMA_Body_H60/compton_energy_probability_v5_5e9_full10000'
JDATA = HERE / 'generated/compton_energy_probability_v5_5e9_full10000'
JFOLDER = JDATA / 'formal_results/1669255/continuous_energy'
NODES = {'EHE': (0, 10, 20, 50, 100, 150, 200),
         'JSCC': (0, 500, 1000, 2500, 5000, 7500, 10000)}
ROUTES = (
    ('EHE', '218_SinglePhoton_CrossTalkCorrected', '218 corrected single', '218校正单光子'),
    ('JSCC', '218_SinglePhoton_CrossTalkCorrected', '218 corrected single', '218校正单光子'),
    ('EHE', '440_SinglePhoton', '440 single', '440单光子'),
    ('JSCC', '440_SinglePhoton', '440 single', '440单光子'),
    ('JSCC', '440_ComptonOnly', '440 SC Compton', '440 SC Compton'),
    ('JSCC', '440_SinglePlusCompton', '440 JSCC', '440 JSCC'),
    ('EHE', '440SinglePlus218Single', 'dual single sum', '双能单光子和'),
    ('JSCC', '440SinglePlus218Single', 'dual single sum', '双能单光子和'),
    ('JSCC', '440SingleComptonPlus218Single', '440 JSCC + 218 single', '440 JSCC + 218单光子和'),
)
DIAMETERS = (10, 13, 17, 22, 28, 37)
COLORS = dict(zip(DIAMETERS, ('#3366cc', '#e67e22', '#229954', '#bf4375', '#7858a6', '#008c99')))
ROUTE_COLORS = ('#3366cc', '#229954', '#bf4375', '#e67e22', '#7858a6', '#008c99')
METRICS = (
    ('background_mean', '背景均值 / Background mean', 'gamma/mm³'),
    ('background_cv', '背景CV / Background CV', '%'),
    ('max_density', '最大密度 / Maximum density', 'gamma/mm³'),
    ('peak_background_ratio', '峰值/背景 / Peak-background ratio', 'ratio'),
    ('p99_density', 'P99 density', 'gamma/mm³'),
    ('p999_density', 'P99.9 density', 'gamma/mm³'),
    ('volume_weighted_p999_density', '体积加权P99.9 / Volume-weighted P99.9', 'gamma/mm³'),
    ('total_integral', '总积分 / Total emitted gamma integral', 'gamma'),
    ('integral_recovery', '积分恢复 / Integral recovery', '%'),
    ('source_z_leakage', '|z|>30 mm 泄漏 / Axial leakage', '%'),
    ('peak_x_mm', '峰位置X / Peak X', 'mm'),
    ('peak_y_mm', '峰位置Y / Peak Y', 'mm'),
    ('peak_z_mm', '峰位置Z / Peak Z', 'mm'),
)


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def csv_read(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def energy(channel):
    if channel == '218_SinglePhoton_CrossTalkCorrected':
        return '218'
    if channel in ('440SinglePlus218Single', '440SingleComptonPlus218Single'):
        return 'sum'
    return '440'


def initialization(channel):
    return 2.0 if energy(channel) == 'sum' else 1.0


def initialize_evidence():
    ef = read(REPORT / 'reconstruction_execution_freeze.json')
    jf = read(JREPORT / 'formal_freeze.json')
    evidence = {}
    for name in ('single_checkpoint_mlem.py', 'torch_active_operator.py', 'run_energy_full10000_v5.py'):
        sha = digest(HERE / name)
        ensure(jf['sha256'][name] == sha, 'JSCC initialization source differs: ' + name)
        if name != 'run_energy_full10000_v5.py':
            ensure(ef['sha256'][name] == sha, 'EHE initialization source differs: ' + name)
        evidence[name] = sha
    single = (HERE / 'single_checkpoint_mlem.py').read_text(encoding='utf-8')
    active = (HERE / 'torch_active_operator.py').read_text(encoding='utf-8')
    ensure('image = torch.ones((n, 1)' in single, 'Single initialization definition absent')
    compact = ''.join(active.split())
    ensure('image_d=torch.ones((n,1)' in compact and 'image_j=image_d.clone()' in compact,
           'SC/joint initialization definition absent')
    return dict(source_sha256=evidence, ehe_freeze_sha256=digest(REPORT / 'reconstruction_execution_freeze.json'),
                jscc_freeze_sha256=digest(JREPORT / 'formal_freeze.json'),
                provenance='SHA-bound executed all-ones initialization; not a saved reconstruction snapshot',
                single_density_gamma_per_mm3=1, sum_density_gamma_per_mm3=2,
                zero_CNR='undefined: uniform background standard deviation is zero',
                zero_peak_location='undefined: uniform initialization has no unique maximum')


def inputs():
    ep = read(REPORT / 'formal_summary.json')
    jp = read(JREPORT / 'formal_summary.json')
    jv = read(JFOLDER / 'verification.json')
    ensure(ep['passed'] and ep['iterations'] == 200 and ep['frames_per_channel'] == 20,
           'Accepted EHE200 required')
    ensure(jp['passed'] and jp['job'] == 1669255 and jp['full_six_imaging_completed'] and jv['passed'],
           'Accepted JSCC10000 required')
    old_manifest = read(OLD / 'artifact_manifest.json')['files']
    old_report = read(OLD / 'comparison_report.json')
    ensure(digest(OLD / 'comparison_report.json') == old_manifest['comparison_report.json'],
           'Prior density/count provenance changed')
    meta = read(TRUTH_META)
    ensure(digest(TRUTH) == meta['truth_sha256'] == old_report['truth_sha256'], '3D truth differs')
    truth = np.load(TRUTH)
    ensure(truth['activity_218_zyx'].shape == (40, 100, 168), 'Truth shape differs')
    geometry = DATA / 'payload/whole_geometry.npz'
    ensure(digest(geometry) == '8e7a5f54e59b7bd7ce0db723a42d71b131e8c840f22982d541deb5e77df72aa4',
           'Full EHE geometry differs')
    ensure(digest(geometry) == digest(JDATA / 'formal_payload/whole_geometry.npz'), 'Systems use different active geometry')
    g = np.load(geometry)
    coords, active = g['coordinates_mm'], g['active_indices']
    vol = g['cell_volume_mm3'][active]
    ensure(coords.shape == (132040, 3) and active.size == 78920, 'Whole geometry domain differs')
    scales = old_report['density_scales']
    for system in ('EHE', 'JSCC'):
        for i, e in enumerate(('218', '440')):
            actual = old_report['primary_counts'][system][i] / meta['relative_activity_integral_mm3'][e]
            ensure(abs(scales[system][e] / actual - 1) < 1e-14, 'Actual source density scale differs')
    native, sphere, provenance, histories, source_hashes = {}, {}, {}, {}, {}
    for system in ('EHE', 'JSCC'):
        for kind, store in (('native', native), ('sphere', sphere)):
            filename = f'{system.lower()}_{kind}_iteration_metrics.csv'
            path = OLD / filename
            ensure(digest(path) == old_manifest[filename], 'Accepted metric CSV differs: ' + filename)
            rows = csv_read(path)
            for row in rows:
                key = (system, row['channel'])
                store.setdefault(key, []).append(row)
            source_hashes[filename] = digest(path)
    for system, channel, _, _ in ROUTES:
        key = (system, channel)
        frames, step = (20, 10) if system == 'EHE' else (200, 50)
        path = (DATA / 'results/formal' if system == 'EHE' else JFOLDER) / f'Image_{channel}_history.float32'
        ensure(path.stat().st_size == frames * 78920 * 4, 'History byte shape differs')
        sha = digest(path)
        authority = ep['files'][path.name] if system == 'EHE' else next(
            item['sha256']['history'] for item in jv['outputs'] if item['channel'] == channel)
        ensure(sha == authority == old_report['history_sha256'][system + '/' + channel], 'History SHA differs')
        history = np.memmap(path, '<f4', 'r', shape=(frames, 78920))
        ensure(np.isfinite(history).all() and (history >= 0).all(), 'Nonfinite/negative history')
        ensure([int(row['iteration']) for row in native[key]] == list(range(step, step * frames + 1, step)),
               'Native curve is incomplete')
        dlist = DIAMETERS if energy(channel) == 'sum' else meta['hot_sphere_diameters_by_energy_keV'][energy(channel)]
        for d in dlist:
            ensure([int(row['iteration']) for row in sphere[key] if int(row['diameter_mm']) == d]
                   == list(range(step, step * frames + 1, step)), 'Sphere curve is incomplete')
        for iteration in NODES[system][1:]:
            ensure(iteration % step == 0 and 0 <= iteration // step - 1 < frames, 'Unsaved image node')
        histories[key] = (history, step)
        provenance[system + '/' + channel] = dict(history_sha256=sha, saved_frames=frames, save_step=step,
                                                 final_iteration=step * frames, selected_iterations=NODES[system])
    init = initialize_evidence()
    source_hashes.update(truth_npz=digest(TRUTH), truth_manifest=digest(TRUTH_META), whole_geometry=digest(geometry),
                         EHE_formal_summary=digest(REPORT / 'formal_summary.json'),
                         JSCC_verification=digest(JFOLDER / 'verification.json'),
                         accepted_comparison_report=digest(OLD / 'comparison_report.json'))
    return truth, meta, coords, active, vol, scales, histories, native, sphere, provenance, init, source_hashes, old_report


def extended_metrics(native, sphere, coords, active, vol, old_report):
    """Preserve every positive-iteration string; prepend explicit analytical init."""
    nr, sr = [], []
    for system, channel, _, _ in ROUTES:
        value = initialization(channel)
        integral = float(value * vol.sum())
        primary = old_report['primary_counts'][system]
        e = energy(channel)
        denominator = sum(primary[:2]) if e == 'sum' else primary[0 if e == '218' else 1]
        zero = {k: '' for k in native[system, channel][0] if k not in ('system', 'tiny_mass_fraction')}
        zero.update(channel=channel, iteration='0', max_density=value, peak_background_ratio=1,
                    p99_density=value, p999_density=value, volume_weighted_p999_density=value,
                    background_mean=value, background_cv=0, total_integral=integral,
                    integral_recovery=integral / denominator,
                    source_z_leakage=float(vol[np.abs(coords[active, 2]) > 30].sum() / vol.sum()))
        for row in [zero] + native[system, channel]:
            nr.append(dict(system=system, **{k: v for k, v in row.items() if k not in ('system', 'tiny_mass_fraction')},
                           frame_provenance='known_uniform_initialization' if int(row['iteration']) == 0 else 'accepted_saved_frame'))
        ds = sorted({int(row['diameter_mm']) for row in sphere[system, channel]})
        for d in ds:
            own = next(row['energy_keV'] for row in sphere[system, channel] if int(row['diameter_mm']) == d)
            sr.append(dict(system=system, channel=channel, iteration=0, energy_keV=own, diameter_mm=d,
                           hot_mean=value, local_background_mean=value, crc=0, cnr='',
                           frame_provenance='known_uniform_initialization'))
        for row in sphere[system, channel]:
            sr.append(dict(system=system, **{k: v for k, v in row.items() if k not in ('system', 'nominal_truth_contrast')},
                           frame_provenance='accepted_saved_frame'))
    # Every accepted positive iteration must retain its original metric values.
    for store, originals in ((nr, native), (sr, sphere)):
        for key, rows in originals.items():
            for old in rows:
                candidates = [row for row in store if row['system'] == key[0] and row['channel'] == key[1]
                              and int(row['iteration']) == int(old['iteration'])
                              and ('diameter_mm' not in old or int(row['diameter_mm']) == int(old['diameter_mm']))]
                ensure(len(candidates) == 1, 'Metric mapping is ambiguous')
                for field, value in old.items():
                    if field not in ('system', 'tiny_mass_fraction', 'nominal_truth_contrast'):
                        ensure(candidates[0][field] == value, 'Positive-iteration metric altered')
    return nr, sr


def decorate(fig, title, subtitle, footer):
    fig.text(.02, .984, title, fontsize=23, weight='bold', va='top', color='#172e43')
    fig.text(.02, .953, subtitle, fontsize=12, va='top', color='#44576b')
    fig.text(.02, .018, footer, fontsize=10, va='bottom', color='#44576b')


def atlas(view, selected, source, views, sphere, scales):
    selector, extent, axis_names, caption = views[view]
    fig = plt.figure(figsize=(32, 22))
    grid = fig.add_gridspec(9, 12, left=.015, right=.985, bottom=.09, top=.914,
                            width_ratios=[1.45] + [1.5] * 8 + [.23, 2.5, 2.5], hspace=.65, wspace=.21)
    for row, (system, channel, label, cn_label) in enumerate(ROUTES):
        budget = 200 if system == 'EHE' else 10000
        ax = fig.add_subplot(grid[row, 0]); ax.axis('off')
        ax.text(0, .55, f'{system}\n{cn_label}\n{label}\n0-{budget}', fontsize=10.5, weight='bold', va='center',
                color='#215e86' if system == 'EHE' else '#6b4385', linespacing=1.5)
        arrays = [source[system, energy(channel)]] + [selected[system, channel, i] for i in NODES[system]]
        titles = ['3D真值 / Truth'] + [('0 初值 / init' if i == 0 else f'{i} iterations') for i in NODES[system]]
        for col, (image, title) in enumerate(zip(arrays, titles), 1):
            ax = fig.add_subplot(grid[row, col])
            color = ax.imshow(selector(image), origin='lower', extent=extent, cmap='gray_r', vmin=0, vmax=10,
                              interpolation='nearest', aspect='equal')
            ax.set_title(title, fontsize=9.5, pad=4)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#c2ccd5'); spine.set_linewidth(.5)
            if col == 1:
                ax.text(.015, .03, axis_names, transform=ax.transAxes, fontsize=7, color='#55616f')
        rr = [v for v in sphere if v['system'] == system and v['channel'] == channel]
        for col, metric in ((10, 'crc'), (11, 'cnr')):
            ax = fig.add_subplot(grid[row, col])
            for d in sorted({int(v['diameter_mm']) for v in rr}):
                points = sorted((v for v in rr if int(v['diameter_mm']) == d), key=lambda v: int(v['iteration']))
                ax.plot([int(v['iteration']) for v in points],
                        [float(v[metric]) if v[metric] != '' else np.nan for v in points], color=COLORS[d], lw=1.55)
            ax.set_title(metric.upper() + (' (%)' if metric == 'crc' else ''), fontsize=10)
            if metric == 'crc':
                # Tick labels in percent while stored metric and data remain fractions.
                from matplotlib.ticker import PercentFormatter
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                ax.set_ylim(-.2, 1.1)
            else:
                ax.set_ylim(-1, 7.2)
            ax.set_xlim(0, budget)
            ax.set_xticks((0, 50, 100, 150, 200) if system == 'EHE' else (0, 2500, 5000, 7500, 10000))
            ax.tick_params(labelsize=7.5)
            ax.set_xlabel('actual iterations', fontsize=8)
            ax.axhline(0, color='#9da7b1', lw=.6)
            ax.grid(alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
    legend = [Line2D([0], [0], color=COLORS[d], lw=2, label=f'{d} mm') for d in DIAMETERS]
    fig.legend(handles=legend, loc='lower right', bbox_to_anchor=(.987, .046), ncol=6, fontsize=11,
               title='真实3D球ROI / Sphere diameter', title_fontsize=10, frameon=False)
    ca = fig.add_axes([.08, .068, .30, .009])
    cb = fig.colorbar(color, cax=ca, orientation='horizontal', ticks=(0, 1, 5, 10))
    cb.set_label('密度 / 各系统实际源背景密度 (fixed emitted-source density scale)', fontsize=10)
    cb.ax.tick_params(labelsize=8)
    decorate(fig, f'整体迭代对比 / {caption}',
             'EHE 0-200 | JSCC 0-10000 | 9 result routes | independent iteration ranges; columns do not imply equal convergence',
             '0 = known all-ones initialization (single 1, sum 2 gamma/mm³), not a saved frame; CNR at 0 undefined.\n'
             'Crop 0; no smoothing or fitted gain; fixed gray_r 0-10. Metrics: native 120 mm / real 3D ROI. EHE physical audit remains HOLD.\n'
             '218 background budget: EHE440 final200; JSCC440 final10000. Detector/material/count differences are not solely algorithm effects.')
    return fig


def metric_dashboard(rows, metrics, title):
    fig = plt.figure(figsize=(21, 3.05 * len(metrics) + 2))
    grid = fig.add_gridspec(len(metrics), 2, left=.08, right=.97, bottom=.075, top=.90, hspace=.47, wspace=.18)
    for index, (key, label, units) in enumerate(metrics):
        factor = 100 if units == '%' else 1
        all_values = [float(v[key]) * factor for v in rows if v[key] != '']
        minimum, maximum = min(all_values), max(all_values)
        padding = max((maximum - minimum) * .06, abs(maximum) * .01, .01)
        for col, system in enumerate(('EHE', 'JSCC')):
            ax = fig.add_subplot(grid[index, col])
            routes = [route for route in ROUTES if route[0] == system]
            for ri, (_, channel, route_label, _) in enumerate(routes):
                values = sorted([v for v in rows if v['system'] == system and v['channel'] == channel],
                                key=lambda v: int(v['iteration']))
                ax.plot([int(v['iteration']) for v in values],
                        [float(v[key]) * factor if v[key] != '' else np.nan for v in values],
                        label=route_label, color=ROUTE_COLORS[ri], lw=1.6)
            budget = 200 if system == 'EHE' else 10000
            ax.set_xlim(0, budget); ax.set_ylim(minimum - padding, maximum + padding)
            ax.set_title(f'{system}: {label}', fontsize=12, loc='left')
            ax.set_ylabel(units, fontsize=10); ax.set_xlabel(f'actual iterations (0-{budget})', fontsize=10)
            ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False)
            if index == 0:
                ax.legend(ncol=2, fontsize=8.5, loc='best')
    decorate(fig, title,
             'Left: EHE 0-200 (20 saved frames) | Right: JSCC 0-10000 (200 saved frames) | same vertical scale for each metric',
             'All positive-iteration metrics copied unchanged from accepted CSVs; native complete 120 mm active cells.\n'
             '0 = known uniform initialization; nonunique peak X/Y/Z omitted. No claim of equal convergence or convergence by the budget endpoint.')
    return fig


def endpoint_table(nr, sr):
    rows = []
    for system, channel, label, _ in ROUTES:
        n = next(v for v in nr if v['system'] == system and v['channel'] == channel and
                 int(v['iteration']) == (200 if system == 'EHE' else 10000))
        for s in sr:
            if s['system'] == system and s['channel'] == channel and int(s['iteration']) == int(n['iteration']):
                rows.append(dict(system=system, result_type=label, channel=channel, actual_iteration=n['iteration'],
                                 sphere_diameter_mm=s['diameter_mm'], crc=s['crc'], cnr=s['cnr'],
                                 background_cv=n['background_cv'], integral_recovery=n['integral_recovery'],
                                 source_z_leakage=n['source_z_leakage']))
    return rows


def data_context_page(report, endpoints):
    fig = plt.figure(figsize=(22, 17))
    decorate(fig, '数据与终点结果 / Data context and budget endpoints',
             'EHE200 and JSCC10000 are separate requested budget endpoints; no equality of convergence is assumed.',
             'Actual counts are not matched. JSCC legacy primary-tagged cross-window counts are unavailable.\n'
             'Fixed 218 additive background: EHE440200 / JSCC44010000. Original EHE physical audit remains HOLD; original method was user-authorized.')
    ax = fig.add_axes([.035, .72, .93, .18]); ax.axis('off')
    count_rows = []
    for system in ('EHE', 'JSCC'):
        counts = report['actual_window_counts'][system]
        primary = report['primary_counts'][system]
        count_rows.append([system, f'{sum(primary[:2]):,}', f"{counts['218']['total']:,}", f"{counts['440']['total']:,}",
                           '9,869 (measured primary label)' if system == 'EHE' else 'unavailable (legacy evidence)'])
    tab = ax.table(cellText=count_rows, colLabels=('System', 'Emitted gamma', '218 window', '440 window', '440 to 218 measured counts'),
                   cellLoc='center', loc='center', colWidths=(.12, .18, .18, .18, .34))
    tab.auto_set_font_size(False); tab.set_fontsize(12); tab.scale(1, 2.3)
    ax.set_title('同一H60双能3D真值 / Same H60 dual-energy 3D source; both 5e9 emitted gamma', fontsize=16, loc='left')
    # Endpoint CRC/CNR are paired only within the same sphere, with each actual
    # requested iteration explicitly written on every table row.
    ax = fig.add_axes([.035, .055, .93, .62]); ax.axis('off')
    body = []
    for system, channel, label, _ in ROUTES:
        rr = [v for v in endpoints if v['system'] == system and v['channel'] == channel]
        body.append([f'{system} {rr[0]["actual_iteration"]}', label,
                     ' / '.join(str(v['sphere_diameter_mm']) for v in rr),
                     ' / '.join(f'{100 * float(v["crc"]):.1f}' for v in rr),
                     ' / '.join(f'{float(v["cnr"]):.2f}' for v in rr),
                     f'{100 * float(rr[0]["background_cv"]):.1f}',
                     f'{100 * float(rr[0]["integral_recovery"]):.1f}',
                     f'{100 * float(rr[0]["source_z_leakage"]):.1f}'])
    tab = ax.table(cellText=body, colLabels=('System / iter', 'Result type', 'Sphere mm', 'CRC %', 'CNR', 'BG CV %', 'Integral %', '|z|>30 %'),
                   cellLoc='center', loc='center', colWidths=(.115, .205, .13, .18, .16, .07, .075, .075))
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 3.0)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('#cbd4dd')
        if r == 0:
            cell.set_facecolor('#e8eef4'); cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('#f5f9fc' if body[r - 1][0].startswith('EHE') else '#ffffff')
    ax.set_title('各自迭代预算末帧 / Final saved frame of each budget (same order of sphere, CRC and CNR)', fontsize=15, loc='left')
    return fig


def main():
    (truth, meta, coords, active, vol, scales, histories, native, spheres, provenance, init,
     source_hashes, old_report) = inputs()
    ensure(not OUT.exists(), 'Existing derivative atlas is preserved; use a fresh output directory')
    OUT.mkdir()
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'pdf.fonttype': 42,
                         'font.size': 10, 'savefig.facecolor': 'white'})
    x, y, z = [truth[k + '_mm'] for k in 'xyz']
    interp = xy_interpolator(coords[:3301, :2], x, y)
    def image(values):
        full = np.zeros(132040, '<f4'); full[active] = values
        return interp(full.reshape(40, 3301))
    source, selected, display_nodes = {}, {}, []
    for system in ('EHE', 'JSCC'):
        for e in ('218', '440'):
            source[system, e] = truth[f'activity_{e}_zyx']
        source[system, 'sum'] = (scales[system]['218'] * source[system, '218'] +
                                 scales[system]['440'] * source[system, '440']) / scales[system]['sum']
    for system, channel, _, _ in ROUTES:
        h, step = histories[system, channel]
        for iteration in NODES[system]:
            values = np.full(78920, initialization(channel), '<f4') if iteration == 0 else h[iteration // step - 1]
            selected[system, channel, iteration] = image(values) / scales[system][energy(channel)]
            ensure(np.isfinite(selected[system, channel, iteration]).all(), 'Nonfinite display grid')
            display_nodes.append(dict(system=system, channel=channel, iteration=iteration,
                                      frame_index=None if iteration == 0 else iteration // step - 1,
                                      density_scale=scales[system][energy(channel)],
                                      provenance=init['provenance'] if iteration == 0 else 'accepted saved history frame',
                                      clipped_high_pixels=int((selected[system, channel, iteration] > 10).sum())))
    nr, sr = extended_metrics(native, spheres, coords, active, vol, old_report)
    csv_write(OUT / 'all_native_metrics.csv', nr)
    csv_write(OUT / 'all_sphere_metrics.csv', sr)
    endpoints = endpoint_table(nr, sr); csv_write(OUT / 'budget_endpoints.csv', endpoints)
    ix, iy = int(abs(x).argmin()), int(abs(y).argmin())
    extent = (x[0] - 1.5, x[-1] + 1.5, y[0] - 1.5, y[-1] + 1.5)
    views = {'mip72': (lambda im: axial_mip(im, z, 8), extent, 'X / Y mm', '72 mm axial MIP'),
             'axial': (lambda im: im[20], extent, 'X / Y mm', f'Axial z={z[20]:g} mm'),
             'coronal': (lambda im: im[:, iy, :], (x[0] - 1.5, x[-1] + 1.5, -60, 60), 'X / Z mm', f'Coronal y={y[iy]:g} mm'),
             'sagittal': (lambda im: im[:, :, ix], (y[0] - 1.5, y[-1] + 1.5, -60, 60), 'Y / Z mm', f'Sagittal x={x[ix]:g} mm')}
    pages = []
    with PdfPages(OUT / 'overall_comparison.pdf', metadata={'Title': 'EHE0-200 / JSCC0-10000 full trajectories',
                                                         'Author': 'JSCC reconstruction research'}) as pdf:
        for view in views:
            fig = atlas(view, selected, source, views, sr, scales)
            name = f'overall_{view}.png'; fig.savefig(OUT / name, dpi=170); pdf.savefig(fig, dpi=150)
            pages.append(name); plt.close(fig)
            print('Saved', name, flush=True)
        for name, metrics, title in (
                ('native_density_curves.png', METRICS[:7], '完整原生密度与噪声轨迹 / Native density and noise trajectories'),
                ('native_integral_position_curves.png', METRICS[7:], '完整积分、泄漏与峰位置 / Integral, leakage and peak-position trajectories')):
            fig = metric_dashboard(nr, metrics, title)
            fig.savefig(OUT / name, dpi=140); pdf.savefig(fig, dpi=140); pages.append(name); plt.close(fig)
            print('Saved', name, flush=True)
        fig = data_context_page(old_report, endpoints)
        name = 'data_and_endpoints.png'; fig.savefig(OUT / name, dpi=150); pdf.savefig(fig, dpi=150)
        pages.append(name); plt.close(fig)
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)
    metadata = dict(task='Separate complete iteration trajectories, never equal-iteration/equal-convergence pairing',
                    reference_jobs={'EHE': read(REPORT / 'formal_job.json')['job'], 'JSCC': 1669255},
                    routes=[dict(system=s, channel=c, label=l, label_zh=cn) for s, c, l, cn in ROUTES],
                    source_sha256=source_hashes, histories=provenance, initialization=init, display_nodes=display_nodes,
                    selected_iterations=NODES, physical_calibration_passed=False,
                    density_scales=scales, physical_units='gamma/mm3; dual sum is not Ac225 activity',
                    display=dict(crop=0, smoothing_sigma=0, colormap='gray_r', range=[0, 10], fitted_gain=False,
                                 truth='Actual H60 3D dual-energy source', mip_z_edges_mm=[-36, 36],
                                 axial_z_mm=float(z[20]), coronal_y_mm=float(y[iy]), sagittal_x_mm=float(x[ix])),
                    native_metrics_extent_mm=120, background_budgets=old_report['background_budgets'],
                    metric_rows=dict(native=len(nr), sphere=len(sr)), pages=pages,
                    scientific_checks=dict(history_sha=True, byte_shape=True, finite_nonnegative=True,
                                           actual_saved_node_mapping=True, all_positive_metrics_unchanged=True,
                                           full_frame_curves=True, frozen_initialization_sha=True,
                                           no_equal_convergence_claim=True, no_simulation_or_reconstruction=True),
                    producer_sha256=digest(Path(__file__)))
    write(OUT / 'comparison_metadata.json', metadata)
    text = '''# EHE 0-200 / JSCC 0-10000 整体对比

主总图：[72mm MIP](overall_mip72.png)。PDF为7页完整图集：[整体PDF](overall_comparison.pdf)。
图像总览另有[轴位](overall_axial.png)、[冠状位](overall_coronal.png)、[矢状位](overall_sagittal.png)。
覆盖EHE三路与JSCC六路，按218、440、双能和排列。每行左为实际三维真值，中为自身迭代轨迹，右为完整3D球ROI CRC/CNR。

EHE图像节点为0/10/20/50/100/150/200；JSCC为0/500/1000/2500/5000/7500/10000。
曲线分别使用20/200个实际已保存帧，横轴分别0-200/0-10000。同列只用于排版，不表示同等迭代、进度或收敛。
末帧仅是各自请求的预算终点，不称两者已同等收敛。

**0次来源**：执行冻结源码SHA核对后的全1 gamma/mm³初值；双能相加为2，域外为0。
没有保存0次快照，图中明确标注init，不冒充已保存重建帧。均匀初值CNR的背景标准差为0，因此不定义/不绘制CNR0；峰位置不唯一也留空。CRC0为0。
固定真值背景尺度下初值很淡是实际尺度结果，不调亮。

所有图使用各系统本次实际初级gamma/真实源积分确定的固定尺度，gray_r 0-10，crop0，无平滑/逐图拟合亮度。
轴位z=+1.5mm，冠状y=-1.5mm，矢状x=-1.5mm；MIP中心72mm只用于显示。
原生指标覆盖完整120mm；球指标用manifest现有3D球ROI。三维真值SHA及源尺度见comparison_metadata.json。

原生全轨迹：[密度与噪声](native_density_curves.png)、[积分、泄漏与峰位置](native_integral_position_curves.png)。
左右横轴独立，每项指标纵轴在两系统间保持一致。峰位置变化可能来自不同最大像素，不等同运动。
数据：[原生CSV](all_native_metrics.csv)、[球ROI CSV](all_sphere_metrics.csv)、[各自终点CSV](budget_endpoints.csv)。
所有正迭代指标逐字段保持原验收CSV数值，不重新估计/筛选曲线。
[计数与终点表](data_and_endpoints.png)给出实际计数及各自终点。

EHE218背景来自EHE440末图200；JSCC218背景来自JSCC440末图10000。两边实际计数未匹配，探测结构/材料/覆盖差异不能全归因算法。
原EHE物理审计HOLD保留；用户已授权按原方法继续，该图集不改变科学结论或生产模型。旧JSCC没有初级能量标记的实测串窗计数。
本次只从本地已验收数据生成图表，不运行模拟、系统矩阵、重建或远端作业。compton-v5保持PAUSED。
执行代码、输入SHA、图像节点映射及科学检查随图归档；visual_qa.json为实际图像/PDF渲染复核，artifact_manifest.json封存输出SHA。
'''
    (OUT / 'README.md').write_text(text, encoding='utf-8', newline='')
    write(OUT / 'artifact_manifest.json', {'files': {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file()}})
    print(json.dumps(dict(output=str(OUT), native_rows=len(nr), sphere_rows=len(sr), pages=len(pages)), ensure_ascii=False))


if __name__ == '__main__':
    main()
