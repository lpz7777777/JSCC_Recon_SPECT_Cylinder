"""Four requested channels from accepted full10000 histories; no reconstruction."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

STUDY = 'compton_energy_probability_v5_5e9_full10000'
ITERATIONS = (100, 500, 1000, 2000, 5000, 10000)
ROWS = (
    ('218_SinglePhoton_CrossTalkCorrected', 218, '218 keV SC single\n(crosstalk corrected)'),
    ('440_SinglePhoton', 440, '440 keV SC single'),
    ('440_ComptonOnly', 440, '440 keV SC Compton'),
    ('440_SinglePlusCompton', 440, '440 keV JSCC\n(single + Compton)'),
)


def plot(job, experiment, output):
    sys.path.insert(0, str(experiment))
    from analyze_nema_result import digest, xy_interpolator
    from mip_projection import axial_mip

    report_root = experiment / 'reports/NEMA_Body_H60' / STUDY
    summary_path = report_root / 'formal_summary.json'
    summary = json.loads(summary_path.read_text())
    if (not summary.get('passed') or not summary.get('full_six_imaging_completed')
            or summary.get('mode') != 'formal' or summary['job'] != job):
        raise ValueError('Accepted completed formal10000 job required')
    data = experiment / 'generated' / STUDY
    result = data / 'formal_results' / str(job) / 'continuous_energy'
    payload = data / 'formal_payload'
    receipt = summary['models']['continuous_energy']['sha256']
    for name in ('run_manifest.json', 'verification.json'):
        if digest(result / name) != receipt[name]:
            raise ValueError('Accepted proof bytes changed: ' + name)
    verification = json.loads((result / 'verification.json').read_text())
    if (not verification['passed'] or verification['iterations'] != 10000
            or verification['save_step'] != 50 or verification['accepted_events'] != 483743):
        raise ValueError('Actual full10000 validation differs')

    original = report_root / f'comparison_{job}'
    old_manifest = json.loads((original / 'artifact_manifest.json').read_text())
    old_report_path = original / 'comparison_report.json'
    if digest(old_report_path) != old_manifest['files']['comparison_report.json']['sha256']:
        raise ValueError('Accepted comparison metadata changed')
    old_report = json.loads(old_report_path.read_text())
    if old_report['formal_summary_sha256'] != digest(summary_path):
        raise ValueError('Comparison formal acceptance differs')

    geometry_path = payload / 'whole_geometry.npz'
    if digest(geometry_path) != old_report['geometry_sha256']:
        raise ValueError('Accepted whole-cell geometry changed')
    geometry = np.load(geometry_path)
    coords = geometry['coordinates_mm']
    active = geometry['active_indices']
    if coords.shape != (132040, 3) or active.shape != (78920,):
        raise ValueError('Whole-cell dimensions differ')
    truth_path = experiment / 'generated/NEMA_Body_H60/truth_3mm.npz'
    source_path = experiment / 'reports/NEMA_Body_H60/manifest.json'
    source = json.loads(source_path.read_text())
    if (digest(truth_path) != source['truth_sha256']
            or digest(truth_path) != old_report['truth_sha256']
            or digest(source_path) != old_report['truth_manifest_sha256']):
        raise ValueError('Authoritative H60 three-dimensional truth differs')
    truth = np.load(truth_path)
    x, y, z = (truth[k + '_mm'] for k in 'xyz')
    for energy in (218, 440):
        t = truth[f'activity_{energy}_zyx']
        if t.shape != (40, 100, 168) or not np.isfinite(t).all() or np.any(t < 0):
            raise ValueError('Invalid authoritative 3D truth')
    if not np.allclose(coords[::3301, 2], z):
        raise ValueError('Truth and polar axial coordinates differ')
    interpolate = xy_interpolator(coords[:3301, :2], x, y)
    accepted_outputs = {r['channel']: r for r in verification['outputs']}
    selected = {}
    snapshot_rows = []
    history_shas = {}
    scales = old_report['fixed_display_scales']
    for channel, energy, label in ROWS:
        path = result / f'Image_{channel}_history.float32'
        sha = digest(path)
        if (path.stat().st_size != 200 * 78920 * 4 or accepted_outputs[channel]['frames'] != 200
                or sha != accepted_outputs[channel]['sha256']['history']
                or sha != old_report['history_sha256'][channel]):
            raise ValueError('Accepted history size/SHA differs: ' + channel)
        history_shas[channel] = sha
        history = np.memmap(path, dtype='<f4', mode='r', shape=(200, 78920))
        if not np.isfinite(history).all() or np.any(history < 0):
            raise ValueError('Invalid original history: ' + channel)
        scale = scales[channel]
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid fixed display scale')
        for iteration in ITERATIONS:
            index = iteration // 50 - 1
            values = history[index]
            full = np.zeros(132040, dtype=np.float32)
            full[active] = values
            image = interpolate(full.reshape(40, 3301)) / scale
            if image.shape != (40, 100, 168) or not np.isfinite(image).all():
                raise ValueError('Invalid selected display array')
            selected[channel, iteration] = image
            snapshot_rows.append(dict(channel=channel, energy_keV=energy, iteration=iteration,
                saved_frame_index=index, fixed_scale=scale, native_max_density=float(values.max()),
                display_max=float(image.max()), display_pixels_above_10=int(np.count_nonzero(image > 10))))

    extent = (x[0] - 1.5, x[-1] + 1.5, y[0] - 1.5, y[-1] + 1.5)
    ix, iy = int(np.argmin(abs(x))), int(np.argmin(abs(y)))
    views = {
        'axial': (lambda im: im[20], extent, 'Axial z=+1.5 mm', (24, 11.5)),
        'coronal': (lambda im: im[:, iy, :], (x[0] - 1.5, x[-1] + 1.5, -60, 60),
                    f'Coronal y={y[iy]:+.1f} mm', (24, 7)),
        'sagittal': (lambda im: im[:, :, ix], (y[0] - 1.5, y[-1] + 1.5, -60, 60),
                     f'Sagittal x={x[ix]:+.1f} mm', (24, 7)),
        'mip72': (lambda im: axial_mip(im, z, 8), extent, 'Central 72 mm MIP', (24, 11.5)),
    }
    output.mkdir(parents=True, exist_ok=False)
    for kind, (select, ex, name, size) in views.items():
        fig, axes = plt.subplots(4, 7, figsize=size, layout='constrained')
        for row, (channel, energy, label) in enumerate(ROWS):
            panels = [truth[f'activity_{energy}_zyx']] + [selected[channel, i] for i in ITERATIONS]
            for col, image in enumerate(panels):
                color = axes[row, col].imshow(select(image), origin='lower', extent=ex,
                    cmap='gray_r', vmin=0, vmax=10, interpolation='nearest')
                axes[row, col].set(aspect='equal')
                axes[row, col].tick_params(labelsize=8)
                if row == 0:
                    axes[row, col].set_title('Truth' if col == 0 else f'Iteration {ITERATIONS[col - 1]}', fontsize=12)
                if col == 0:
                    axes[row, col].set_ylabel(label, fontsize=11)
        fig.colorbar(color, ax=axes, shrink=.8, label='Gamma density / fixed energy background scale')
        fig.suptitle(f'Legacy 5e9, job {job}: {name}; no smoothing, crop0', fontsize=15)
        fig.savefig(output / f'four_channels_{kind}.png', dpi=180)
        plt.close(fig)

    with (output / 'selected_frames.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(snapshot_rows[0]))
        writer.writeheader()
        writer.writerows(snapshot_rows)
    metadata = dict(job=job, study=STUDY, channels=[r[0] for r in ROWS],
        iterations=list(ITERATIONS), frames_per_history=200, history_sha256=history_shas,
        formal_summary_sha256=digest(summary_path), verification_sha256=digest(result / 'verification.json'),
        original_comparison_sha256=digest(old_report_path), geometry_sha256=digest(geometry_path),
        truth_source=str(truth_path.resolve()), truth_sha256=digest(truth_path),
        truth_manifest_sha256=digest(source_path), truth_shape_zyx=[40, 100, 168],
        fixed_display_scales={c: scales[c] for c, _, _ in ROWS},
        scale_reference='This run 440 JSCC iteration2000 background; actual photon/truth-integral transfer to218',
        smoothing_sigma_pixels=0, crop_pixels=0, colormap='gray_r', display_range=[0, 10],
        mip_z_edges_mm=[-36, 36], mip_z_centers_mm=[-34.5, 34.5],
        interpolation='3mm XY barycentric display only; no z or iteration interpolation',
        single_218_method='Single-photon MLEM with fixed additive 440-to218 crosstalk background',
        jscc_440_method='440 single-photon plus continuous-energy Compton; not an218+440 density sum',
        raw_reconstruction_arrays_modified=False, new_reconstruction_run=False)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    shutil.copy2(__file__, output / Path(__file__).name)
    text = ['# 218/440单光子、440康普顿与440 JSCC随迭代图像', '',
        f'复用严格验收的作业{job}，四路200帧原始历史SHA全部复核；本次只生成图像。', '',
        '行顺序：218单光子（固定440→218串窗背景校正）、440单光子、440 SC Compton、440 JSCC（单光子+康普顿）。',
        '列顺序：真实H60三维球体真值、100、500、1000、2000、5000、10000次。未插值迭代帧。', '',
        'gray_r白低黑高；无平滑、crop0，色标固定0–10，超过10显示黑色饱和而不改原数组。',
        '440三路共同固定为本次440 JSCC第2000次背景尺度943.35180664；218按实际发射数/真值积分转移为415.22638575。',
        '显示使用既有3mm XY插值，无轴向插值；中央72mm MIP为24层中心−34.5～34.5mm，冠/矢图展示完整120mm。',
        '真值为工程generated/NEMA_Body_H60/truth_3mm.npz，SHA和位置记录在metadata.json；没有使用二维NEMA替代。', '',
        '[原始六路科学结果与200帧曲线](../comparison_' + str(job) + '/README.md)', '']
    for kind in views:
        text.append(f'![{kind}](four_channels_{kind}.png)')
    (output / 'README.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    manifest = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
                for p in output.iterdir() if p.is_file()}
    (output / 'artifact_manifest.json').write_text(json.dumps(dict(job=job, files=manifest), indent=2) + '\n', encoding='utf-8')
    print('FOUR_CHANNEL_ITERATION_GALLERIES_CREATED', output)


if __name__ == '__main__':
    experiment = next(p for p in Path(__file__).resolve().parents if p.name == 'ELLIPSE500x300_H120')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=int, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir or experiment / 'reports/NEMA_Body_H60' / STUDY / f'four_channel_iterations_{args.job}'
    plot(args.job, experiment, output)
