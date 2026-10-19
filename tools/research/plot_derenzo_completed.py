"""Plot only strictly delivered monoenergy Derenzo reconstructions.

No reconstruction, filtering, fitted intensity gain, cross-energy addition or
result mutation. Authoritative rod geometry, actual emitted photon counts and
the existing native polar-to-Cartesian display operator are used explicitly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse
import numpy as np

STUDY = 'derenzo_equal_regions_mono_5e9_20261016'
PROTOCOL = 'derenzo-source-density-no-fit-full120-v1'
CASES = ['JSCC_218', 'EHE_218', 'EHE_440']
TITLES = ['JSCC SC · 218 keV', 'EHE SC · 218 keV', 'EHE SC · 440 keV']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def csv_write(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def truth_from_rods(design, x, y, z, samples=24):
    """Area-average exact analytic circles; exact axial interval overlap.

    The only approximation is documented 24x24 subpixel area integration of
    boundary pixels, consistent with the catalog's analytic-source convention.
    """
    area = np.zeros((len(y), len(x)), dtype=np.float64)
    sub = (np.arange(samples)+.5)/samples*3-1.5
    for rod in design['rods']:
        cx, cy, r = rod['x_mm'], rod['y_mm'], rod['diameter_mm']/2
        ix = np.flatnonzero(np.abs(x-cx) <= r+1.5)
        iy = np.flatnonzero(np.abs(y-cy) <= r+1.5)
        dx = x[ix][None, :, None, None] + sub[None, None, None, :] - cx
        dy = y[iy][:, None, None, None] + sub[None, None, :, None] - cy
        fraction = (dx*dx+dy*dy <= r*r).mean(axis=(2, 3))
        area[np.ix_(iy, ix)] += fraction*rod['relative_emission_density']
        if (rod['z_min_mm'], rod['z_max_mm']) != (-30, 30):
            raise ValueError('This plot requires the registered H60 rods')
    if area.max() > 1+1e-12 or not np.isfinite(area).all():
        raise ValueError('Truth rods overlap or have invalid activity')
    axial = np.maximum(0, np.minimum(z+1.5, 30)-np.maximum(z-1.5, -30))/3
    truth = (axial[:, None, None]*area[None]).astype('f4')
    measured = float(truth.sum(dtype='f8')*27)
    relative = measured/design['total_hot_volume_mm3']-1
    if abs(relative) > .002:
        raise ValueError('Analytic truth volume closure fails 0.2 percent tolerance')
    return truth, dict(subsamples_per_axis=samples, analytic_hot_volume_mm3=design['total_hot_volume_mm3'],
                       sampled_hot_volume_mm3=measured, relative_volume_error=relative)


def configure_plotting():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})


def draw(ax, image, extent, vmax, title=None, axes=False):
    item = ax.imshow(image, origin='lower', extent=extent, cmap='gray_r',
                     vmin=0, vmax=vmax, interpolation='nearest', aspect='equal')
    ax.add_patch(Ellipse((0, 0), 500, 300, fill=False, edgecolor='#c4cbd1', linewidth=.55))
    if title:
        ax.set_title(title, fontsize=11, pad=9)
    if axes:
        ax.set(xticks=[-250, 0, 250], yticks=[-150, 0, 150], xlabel='x (mm)')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#d8dee4')
    return item


def save(fig, folder, name):
    fig.savefig(folder/(name+'.png'), dpi=220)
    fig.savefig(folder/(name+'.pdf'))
    plt.close(fig)


def final_figure(cases, extent, folder, vmax, slice_mode=False, name='final_mip'):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.25))
    fig.subplots_adjust(left=.085, right=.89, top=.84, bottom=.12, wspace=.10, hspace=.18)
    for col, case in enumerate(cases):
        truth = case['truth_slice'] if slice_mode else case['truth_mip']
        image = case['final_slice'] if slice_mode else case['final_mip']
        title = f"{case['title']}\n{case['iterations']:,} 次"
        draw(axes[0, col], truth, extent, vmax, title)
        item = draw(axes[1, col], image, extent, vmax, axes=True)
        axes[0, col].text(.5, -.07, '棒径 '+ '/'.join(map(str, case['diameters']))+' mm',
                          transform=axes[0, col].transAxes, ha='center', va='top', fontsize=8.5)
    axes[0, 0].set_ylabel('几何真值', fontsize=12, labelpad=18)
    axes[1, 0].set_ylabel('中心切片' if slice_mode else '重建 MIP', fontsize=12, labelpad=18)
    cb = fig.add_axes([.915, .19, .014, .55])
    fig.colorbar(item, cax=cb).set_label('重建密度 / 理论棒内发射密度', fontsize=10)
    view = '中心轴向切片 z = +1.5 mm' if slice_mode else '完整 120 mm 轴向 MIP'
    fig.suptitle('已完成 Derenzo 重建 · 每组 5e9 实际单能光子', y=.972, fontsize=16)
    fig.text(.5, .907, view+'  |  固定源密度尺度 · 无平滑 · 无亮度拟合', ha='center', fontsize=10.5)
    cap = f'统一色标 0–{vmax:g}；超出上限显示黑色，原值保留。' if vmax == 2 else f'统一色标 0–{vmax:g}；当前显示图像无截顶。'
    fig.text(.5, .040, cap+'  JSCC 10000 次与 EHE 200 次不代表相同收敛程度。', ha='center', fontsize=9)
    fig.text(.5, .012, 'EHE 复用现有响应；未作新的独立绝对活度标定。', ha='center', fontsize=8.5, color='#59636d')
    save(fig, folder, name)


def iteration_figure(cases, extent, folder, name='iteration_mip_gallery'):
    fig, axes = plt.subplots(len(cases), 7, figsize=(19, 2.6*len(cases)+1.1), squeeze=False)
    fig.subplots_adjust(left=.092, right=.94, top=.86, bottom=.10, wspace=.075, hspace=.28)
    for row, case in enumerate(cases):
        draw(axes[row, 0], case['truth_mip'], extent, 2, '真值')
        for col, iteration in enumerate(case['selected_iterations'], 1):
            item = draw(axes[row, col], case['gallery'][iteration], extent, 2, f'{iteration:,} 次')
        pos = axes[row, 0].get_position()
        fig.text(.011, (pos.y0+pos.y1)/2, case['title'].replace(' · ', '\n'), fontsize=11, va='center')
    cb = fig.add_axes([.954, .20, .010, .50])
    fig.colorbar(item, cax=cb).set_label('密度 / 理论棒内发射密度', fontsize=10)
    fig.suptitle('迭代变化 · 完整 120 mm 轴向 MIP', fontsize=17, y=.98)
    fig.text(.5, .925, '5e9 实际光子/组  |  统一色标 0–2；高值显示截顶  |  无平滑、无亮度拟合', ha='center', fontsize=10)
    fig.text(.5, .035, '各行使用实际保存迭代；同一列不代表相同迭代或相同收敛。显示采样 3 mm，原生径向网格仍为 6 mm。', ha='center', fontsize=9.5)
    fig.text(.5, .011, 'EHE 复用现有响应；未作新的独立绝对活度标定。', ha='center', fontsize=8.5, color='#59636d')
    save(fig, folder, name)


def curve_figure(cases, folder):
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5))
    fig.subplots_adjust(left=.07, right=.98, top=.85, bottom=.13, wspace=.27, hspace=.28)
    for col, case in enumerate(cases):
        rows = case['metrics']
        x = [r['iteration'] for r in rows]
        axes[0, col].plot(x, [r['native_emission_integral_ratio'] for r in rows], color='#196a80', lw=1.6)
        axes[0, col].axhline(1, color='#c4663b', lw=.9, ls='--')
        axes[0, col].set_title(case['title'])
        axes[1, col].plot(x, [r['native_peak_over_nominal_rod_density'] for r in rows], color='#724b91', lw=1.6)
        axes[1, col].set_xlabel('实际迭代次数')
        for ax in axes[:, col]:
            ax.set_xscale('log')
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel('全域发射积分 / 实际发射数')
    axes[1, 0].set_ylabel('原生峰值 / 理论棒内密度')
    fig.suptitle('原生完整三维数组 · 200 个保存帧', fontsize=16, y=.98)
    fig.text(.5, .92, '独立迭代范围，无空间平滑或拟合；曲线不作为绝对活度标定或空间分辨率证书。', ha='center', fontsize=10)
    save(fig, folder, 'native_convergence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    h = root/'experiments/ELLIPSE500x300_H120'
    data = h/'generated'/STUDY
    report = h/'reports/Derenzo'/STUDY
    output = (args.output or root/'Analysis_Plots/research/derenzo_completed_5e9_20261019').resolve()
    if (output/'data/metadata.json').exists() and read(output/'data/metadata.json')['protocol'] != PROTOCOL:
        raise ValueError('Preserve unrelated visualization output')
    for name in ['figures', 'data', 'scripts']:
        (output/name).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(h))
    from analyze_nema_result import xy_interpolator
    geometry_path = data/'gpu_payload/whole_geometry.npz'
    geometry_sha = '8e7a5f54e59b7bd7ce0db723a42d71b131e8c840f22982d541deb5e77df72aa4'
    if sha(geometry_path) != geometry_sha:
        raise ValueError('Native geometry identity differs')
    with np.load(geometry_path) as g:
        coords = g['coordinates_mm'].copy()
        active = g['active_indices'].copy()
        volume = (g['cell_volume_mm3']*g['ellipse_fraction'])[active]
    if coords.shape != (132040, 3) or active.shape != (78920,):
        raise ValueError('Frozen native grid shape differs')
    z = np.unique(coords[:, 2])
    if not np.array_equal(z, np.arange(-58.5, 60, 3)) or not np.allclose(coords[:, 2].reshape(40, 3301), z[:, None]):
        raise ValueError('Native z ordering differs')
    # Same 3-mm display grid as established full-FOV NEMA analyses, extending
    # 2 mm beyond the actual x=+/-250-mm field. Preserve boundary interpolation
    # rather than masking or falsely describing these samples as zero padding.
    x = np.arange(-250.5, 251, 3)
    y = np.arange(-148.5, 150, 3)
    extent = (float(x[0]-1.5), float(x[-1]+1.5), float(y[0]-1.5), float(y[-1]+1.5))
    interpolate = xy_interpolator(coords[:3301, :2], x, y)
    roundoff = []

    def sample_native(values, nominal, label):
        sampled = interpolate(values.reshape(40, 3301))/nominal
        minimum = float(sampled.min())
        if not np.isfinite(sampled).all() or minimum < -1e-8:
            raise ValueError('Display barycentric operator exceeds roundoff tolerance')
        negative = int(np.count_nonzero(sampled < 0))
        roundoff.append(dict(label=label, minimum_before_cleanup=minimum,
                             negative_samples=negative, absolute_zero_tolerance=1e-8))
        # Native arrays were verified nonnegative. Tiny negative barycentric
        # roundoff is zeroed only in derived display samples and fully recorded.
        return np.maximum(sampled, 0)
    zi = int(np.flatnonzero(z == 1.5)[0])
    truth = {}
    truth_meta = {}
    inputs = {str(geometry_path.relative_to(root)): geometry_sha}
    source_freeze = read(report/'transport_freeze.json')
    for energy in [218, 440]:
        p = root/f'Analysis_Plots/research/derenzo_218_440_design_20261015_equal_regions/data/derenzo_{energy}keV_geometry.json'
        expected = source_freeze['geometry_checks'][str(energy)]['geometry_sha256']
        if sha(p) != expected:
            raise ValueError('Authoritative source geometry changed')
        design = read(p)
        truth[energy], truth_meta[energy] = truth_from_rods(design, x, y, z)
        truth_meta[energy].update(source_geometry_sha256=expected,
                                  rod_count=design['rod_count'],
                                  nominal_rod_emitted_density_gamma_mm3=5_000_000_000/design['total_hot_volume_mm3'],
                                  diameters_mm=design['diameters_mm'])
        inputs[str(p.relative_to(root))] = expected
        np.savez_compressed(output/'data'/f'truth_{energy}keV_3mm.npz',
                            activity_relative_zyx=truth[energy], x_mm=x, y_mm=y, z_mm=z)
    cases = []
    metrics_all = []
    selected_rows = []
    projections = dict(x_mm=x, y_mm=y, z_mm=z,
                       truth_218_mip=truth[218].max(axis=0), truth_440_mip=truth[440].max(axis=0))
    for dataset, title in zip(CASES, TITLES):
        acceptance_path = report/dataset/'formal_acceptance.json'
        accepted = read(acceptance_path)
        if not accepted['passed'] or not accepted['strict_fetch_passed'] or accepted['stage'] != 'formal':
            raise ValueError('Only fully delivered formal results may be visualized')
        p = Path(accepted['local_result'])
        for name, expected in accepted['all_files_sha256'].items():
            if sha(p/name) != expected:
                raise ValueError('Strict accepted result changed: '+dataset+'/'+name)
        manifest = read(p/'run_manifest.json')
        verification = read(p/'verification.json')
        energy = int(dataset.split('_')[1])
        channel = f'{energy}_SinglePhoton'
        total = 10000 if dataset.startswith('JSCC') else 200
        step = 50 if dataset.startswith('JSCC') else 1
        if (manifest['mode'], manifest['iterations'], manifest['save_step'], manifest['channels']) != ('formal', total, step, [channel]):
            raise ValueError('Route or actual iteration budget differs')
        if not verification['passed'] or verification['mode'] != 'formal':
            raise ValueError('Independent numerical acceptance is missing')
        collection = read(data/dataset/'input/collection.json')
        source_geometry = truth_meta[energy]['source_geometry_sha256']
        if (collection['total_primary_photons'] != 5_000_000_000
                or collection['data_kind'] != 'actual_Geant4_transport'
                or collection['source_geometry_sha256'] != source_geometry
                or sha(data/dataset/'input/collection.json') != manifest['collection_sha256']):
            raise ValueError('Dose/actual transport/source geometry identity differs')
        inputs[str(acceptance_path.relative_to(root))] = sha(acceptance_path)
        inputs[str((data/dataset/'input/collection.json').relative_to(root))] = manifest['collection_sha256']
        image_path = p/f'Image_{channel}_active.float32'
        full_path = p/f'Image_{channel}_full.float32'
        history_path = p/f'Image_{channel}_history.float32'
        if image_path.stat().st_size != 78920*4 or full_path.stat().st_size != 132040*4 or history_path.stat().st_size != 200*78920*4:
            raise ValueError('Snapshot byte shape differs')
        image = np.fromfile(image_path, '<f4')
        full = np.fromfile(full_path, '<f4')
        history = np.memmap(history_path, mode='r', dtype='<f4', shape=(200, 78920))
        if not np.array_equal(image, history[-1]) or not np.array_equal(full[active], image):
            raise ValueError('Final/history/native-grid identity differs')
        outside = np.ones(132040, bool)
        outside[active] = False
        if np.any(full[outside] != 0) or not np.isfinite(history).all() or np.any(history < 0):
            raise ValueError('Nonfinite/negative/outside-domain values')
        for source in [image_path, full_path, history_path, p/'run_manifest.json', p/'verification.json']:
            inputs[str(source.relative_to(root))] = sha(source)
        nominal = truth_meta[energy]['nominal_rod_emitted_density_gamma_mm3']
        native_volume = full.reshape(40, 3301)
        scaled = sample_native(native_volume, nominal, dataset+'_final')
        selected = [100, 500, 1000, 2000, 5000, 10000] if total == 10000 else [2, 10, 20, 40, 100, 200]
        gallery = {}
        metrics = []
        leak = np.abs(coords[active, 2]) > 30
        for index, frame in enumerate(history):
            iteration = (index+1)*step
            mass = float(np.dot(frame.astype('f8'), volume))
            row = dict(dataset=dataset, iteration=iteration,
                       native_emission_integral_ratio=mass/5_000_000_000,
                       native_peak_over_nominal_rod_density=float(frame.max()/nominal),
                       native_fraction_above_display2=float(np.mean(frame/nominal > 2)),
                       native_source_z_leak_fraction=float(np.dot(frame[leak].astype('f8'), volume[leak])/mass))
            metrics.append(row)
            metrics_all.append(row)
            if iteration in selected:
                frame_full = np.zeros(132040, dtype='f4')
                frame_full[active] = frame
                cart = sample_native(frame_full, nominal, f'{dataset}_iter{iteration}')
                gallery[iteration] = cart.max(axis=0)
                selected_rows.append(dict(dataset=dataset, target_iteration=iteration, actual_iteration=iteration,
                    saved_frame_index=index, mip_max=float(gallery[iteration].max()),
                    mip_full_raster_fraction_above_display2=float(np.mean(gallery[iteration] > 2))))
                projections[f'{dataset}_mip_iter{iteration}'] = gallery[iteration]
        if not np.array_equal(gallery[total], scaled.max(axis=0)):
            raise ValueError('Final MIP and saved iteration mismatch')
        cases.append(dict(dataset=dataset, title=title, energy=energy, iterations=total,
            job=accepted['source_job'], channel=channel, selected_iterations=selected,
            diameters=truth_meta[energy]['diameters_mm'], truth_mip=truth[energy].max(axis=0),
            truth_slice=truth[energy][zi], final_mip=scaled.max(axis=0), final_slice=scaled[zi],
            gallery=gallery, metrics=metrics, source=p, source_nominal_density=nominal))
        projections[f'{dataset}_final_slice_z1p5'] = scaled[zi]
    configure_plotting()
    figures = output/'figures'
    final_figure(cases, extent, figures, 2)
    maximum = max(max(c['final_mip'].max(), c['final_slice'].max()) for c in cases)
    unclipped = math.ceil(float(maximum))
    final_figure(cases, extent, figures, unclipped, name='final_mip_unclipped')
    final_figure(cases, extent, figures, 2, slice_mode=True, name='final_central_slices')
    iteration_figure(cases, extent, figures)
    for case in cases:
        iteration_figure([case], extent, figures, name='iterations_'+case['dataset'])
    curve_figure(cases, figures)
    np.savez_compressed(output/'data/display_projections.npz', **projections)
    csv_write(output/'data/native_iteration_metrics.csv', metrics_all)
    csv_write(output/'data/displayed_iterations.csv', selected_rows)
    summary = []
    for case in cases:
        summary.append(dict(dataset=case['dataset'], source_job=case['job'], iterations=case['iterations'],
                            nominal_rod_emitted_density_gamma_mm3=case['source_nominal_density'],
                            mip_max=float(case['final_mip'].max()),
                            mip_fraction_above_display2=float(np.mean(case['final_mip'] > 2)),
                            **{k:v for k,v in case['metrics'][-1].items() if k not in ['dataset', 'iteration']}))
    csv_write(output/'data/final_summary.csv', summary)
    script = Path(__file__).resolve()
    archive = output/'scripts'/script.name
    if script != archive:
        shutil.copy2(script, archive)
    meta = dict(protocol=PROTOCOL, study=STUDY,
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), datasets=CASES,
        all_input_hashes_verified=True, accepted_input_files_checked=3*608,
        inputs_sha256=inputs, authoritative_truth=truth_meta,
        native_grid=dict(full_points=132040, active_points=78920, radial_pitch_mm=6, z_pitch_mm=3),
        display_grid=dict(shape_zyx=[40, len(y), len(x)], xy_pitch_mm=3, extent_xy_mm=list(extent),
                          field_ellipse_diameters_mm=[500,300], raster_extension_x_mm=2,
                          ellipse_display_mask_applied=False, unmasked_boundary_interpolation_visible=True,
                          z_centers_mm=z.tolist(), central_slice_z_mm=1.5,
                          axial_mip_all_40_slices=True, origin='lower', x_right_y_up=True,
                          negative_barycentric_roundoff_cleanup=roundoff,
                          xy_operator='Existing analyze_nema_result.xy_interpolator: barycentric linear; no z interpolation'),
        intensity=dict(scale='native image / (actual monoenergy primaries / exact analytic hot volume)',
                       fitted_gain=False, per_frame_normalization=False, actual_primaries_each=5_000_000_000,
                       truth_rod_density=1, primary_display_range=[0,2], unclipped_display_range=[0,unclipped],
                       raw_values_unchanged=True, smoothing_sigma_pixels=0, crop_pixels=0, colormap='gray_r'),
        final_summary=summary,
        limitations=['6-mm radial basis limits interpretation of the minimum 6-mm rods.',
                     'EHE existing response has physical calibration limitations; these data are not a new absolute activity calibration.',
                     'Different iteration budgets and equal gallery columns do not imply matched convergence.',
                     'Only completed strictly delivered Derenzo formal results included; no NEMA5e10 validation preview is represented as formal.'],
        skill_overrides=['New Derenzo geometry absent from packaged truth catalog: use SHA-bound authoritative analytic rod geometry.',
                         'Formal run_manifest and strict acceptance replace legacy reconstruction_metadata schema.',
                         'Prior scientific preferences override generic smoothing, crop3 and truth-fit gain defaults.',
                         'Independent Analysis_Plots output preserves immutable accepted result directories.'],
        script_sha256=sha(archive), projection_archive_sha256=sha(output/'data/display_projections.npz'))
    write(output/'data/metadata.json', meta)
    write(output/'data/scientific_qa.json', dict(passed=True, protocol=PROTOCOL,
        all_accepted_input_sha256_pass=True, history_bytes_and_indices_pass=True,
        native_full_active_history_final_exact_match=True, finite_nonnegative_outside_zero_pass=True,
        source_dose_and_geometry_identity_pass=True, truth_volume_closure_pass=True,
        final_snapshot_display_exact_match=True, all_unclipped_panels_below_vmax=bool(maximum <= unclipped),
        metadata_sha256=sha(output/'data/metadata.json')))
    text = """Derenzo已完成重建可视化（2026-10-19）

仅展示已正式完成、独立验收并严格取回的JSCC218（10000次）、EHE218/440（200次）；每组实际5e9单能gamma，不是Ac225衰变次数。

figures/final_mip.png：真值与完整120mm轴向MIP，统一0–2色标；超过2显示黑色，原值未截断。
figures/final_mip_unclipped.png：同一数据的统一、不截顶色标版本。
figures/final_central_slices.png：z=+1.5mm中心切片。
figures/iteration_mip_gallery.png：实际保存帧的迭代变化；JSCC100/500/1000/2000/5000/10000，EHE2/10/20/40/100/200。
figures/native_convergence.png：完整原生三维数组的发射积分与峰值变化。
每张图同时保存PNG和PDF。data/包含精确源几何生成的3mm真值、显示投影NPZ、全部200帧的原生指标、SHA与科学QA。

显示以实际发射数/解析棒体积确定固定尺度，真值棒内=1；不进行单图亮度拟合、平滑或跨能量叠加。白色低、黑色高。XY采用工程既有重心线性显示采样，轴向不插值；显示3mm不改变原生6mm径向分辨单元。完整FOV500×300mm，显示栅格在x方向两边各外延2mm，保留边界插值外延，不施加椭圆显示掩膜，crop0。插值浮点舍入产生的低于1e-8尺度的负数仅在派生显示数组中置零，数量及最小值逐一记录，原生数组未修改。
EHE原响应仍有物理标定限制，峰值和积分不能替代独立绝对活度校准；不同迭代数不等于相同收敛程度。最小6mm棒受本轮粗网格限制，不据此直接宣布6mm分辨率。

复现：python scripts/plot_derenzo_completed.py --project-root <工程绝对路径> --output <本目录绝对路径>
"""
    (output/'README.txt').write_text(text, encoding='utf-8')
    write(output/'data/artifact_sha256.json', {str(p.relative_to(output)):sha(p)
          for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'artifact_sha256.json'})
    print(json.dumps(dict(output=str(output), summary=summary, scientific_qa=True), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
