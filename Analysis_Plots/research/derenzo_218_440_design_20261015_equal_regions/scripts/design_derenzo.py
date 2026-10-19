"""Design two independent monoenergetic Derenzo-like emission phantoms.

Analytic source geometry only: no transport, reconstruction, material changes,
cluster access, or modification of existing experimental data.
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
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.lines import Line2D

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'experiments/ELLIPSE500x300_H120').is_dir())
H = ROOT / 'experiments/ELLIPSE500x300_H120'
OUT = Path(__file__).resolve().parents[1]
NAVY = '#18283B'
MUTED = '#647488'
RULE = '#DFE6ED'
BG = '#F0F3F6'
COLORS = {440: '#426CC1', 218: '#128C91'}
SECTOR_DIAMETERS = {218: [6, 8, 10, 12, 14, 16],
                   440: [18, 20, 10, 12, 14, 16]}
SHARED_DIAMETERS = sorted(set(SECTOR_DIAMETERS[218]) & set(SECTOR_DIAMETERS[440]))
REGION_SIDE_MM = 115.0
REGION_CENTER_RADIUS_MM = 90.0
CENTRAL_ROD_FREE_RADIUS_MM = 30.0
PREVIOUS = OUT.parent / 'derenzo_218_440_design_20261015'
R_ENVELOPE = 140.0
R_PACKING = 135.0
HEIGHT = 60.0
REFERENCE = ('https://pure.tudelft.nl/ws/portalfiles/portal/46265516/'
             'Borghi_2018_Phys._Med._Biol._63_155006.pdf')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def build(energy):
    rods, sectors = [], []
    for si, d in enumerate(SECTOR_DIAMETERS[energy]):
        angle = 30 + 60 * si
        theta = np.deg2rad(angle)
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        # A complete n-row triangle of disk centers has side 2*d*(n-1).
        # Its enclosing equilateral triangle, including radii, adds sqrt(3)*d.
        rows = int(np.floor((REGION_SIDE_MM - np.sqrt(3)*d)/(2*d))) + 1
        occupied_side = 2*d*(rows-1) + np.sqrt(3)*d
        inward_tip = REGION_CENTER_RADIUS_MM - REGION_SIDE_MM/np.sqrt(3)
        outer_base = REGION_CENTER_RADIUS_MM + REGION_SIDE_MM/(2*np.sqrt(3))
        region_local = np.array([[inward_tip, 0],
                                 [outer_base, -REGION_SIDE_MM/2],
                                 [outer_base, REGION_SIDE_MM/2]])
        count = 0
        for k in range(rows):
            # Both the rod cluster and its common-size region have centroid R90.
            local = np.array([(REGION_CENTER_RADIUS_MM + np.sqrt(3)*d*(k-2*(rows-1)/3),
                               d*(2*j-k))
                              for j in range(k+1)])
            assert np.max(np.linalg.norm(local, axis=1) + d/2) <= R_PACKING
            xy = local @ rotation.T
            for j, (x, y) in enumerate(xy):
                rods.append({'rod_id': len(rods)+1, 'sector_id': si+1,
                             'row_index': k, 'column_index': j,
                             'diameter_mm': float(d), 'pitch_mm': float(2*d),
                             'x_mm': float(x), 'y_mm': float(y), 'z_mm': 0.0,
                             'z_min_mm': -HEIGHT/2, 'z_max_mm': HEIGHT/2,
                             'relative_emission_density': 1.0,
                             'volume_mm3': float(np.pi*(d/2)**2*HEIGHT)})
                count += 1
        sectors.append({'sector_id': si+1, 'axis_angle_deg': angle,
                        'diameter_mm': d, 'pitch_mm': 2*d, 'edge_gap_mm': d,
                        'rows': rows, 'rod_count': count,
                        'region_side_mm': REGION_SIDE_MM,
                        'region_area_mm2': np.sqrt(3)*REGION_SIDE_MM**2/4,
                        'region_centroid_radius_mm': REGION_CENTER_RADIUS_MM,
                        'region_centroid_xy_mm': (np.array([REGION_CENTER_RADIUS_MM, 0]) @ rotation.T).tolist(),
                        'region_vertices_xy_mm': (region_local @ rotation.T).tolist(),
                        'region_inward_tip_radius_mm': inward_tip,
                        'occupied_circumscribed_triangle_side_mm': occupied_side,
                        'occupied_circumscribed_triangle_area_mm2': np.sqrt(3)*occupied_side**2/4,
                        'rod_cluster_apex_radial_coordinate_mm': REGION_CENTER_RADIUS_MM - 2*np.sqrt(3)*d*(rows-1)/3})
    total_volume = sum(r['volume_mm3'] for r in rods)
    for r in rods:
        # Uniform activity density: cylinder selection MUST be volume weighted.
        r['source_selection_probability'] = r['volume_mm3']/total_volume
    return {'design_id': f'derenzo_{energy}keV_H60_v4_equal_regions', 'revision_date': '2026-10-15',
            'status': 'design_only',
            'energy_keV': energy, 'particle': 'gamma', 'direction': 'isotropic_4pi',
            'one_primary_gamma_per_event': True, 'events_total': None,
            'ac225_decay_chain_simulated': False,
            'diameters_mm': sorted(SECTOR_DIAMETERS[energy]),
            'coordinate_frame': 'FOV centered, mm; rods parallel to z',
            'center_fov_mm': [0, 0, 0], 'height_mm': HEIGHT,
            'reference_envelope_diameter_mm': 2*R_ENVELOPE,
            'rod_outer_radius_limit_mm': R_PACKING,
            'reference_envelope_is_material_boundary': False,
            'material_model': 'not specified by this emission-geometry design',
            'rod_relative_emission_density': 1, 'background_emission_density': 0,
            'other_energy_primary_emission_density': 0,
            'rod_count': len(rods), 'total_hot_volume_mm3': total_volume,
            'region_side_mm': REGION_SIDE_MM,
            'region_centroid_radius_mm': REGION_CENTER_RADIUS_MM,
            'guaranteed_central_rod_free_radius_mm': CENTRAL_ROD_FREE_RADIUS_MM,
            'all_regions_equal_area': True,
            'sectors': sectors, 'rods': rods}


def verify(designs, cfg):
    sys.path.insert(0, str(H))
    from geometry import grid, ring_counts
    xy = grid(cfg)[0][:cfg['points_per_layer'], :2]
    qa = {'all_geometry_checks_passed': True, 'energies': {},
          'existing_reconstruction_grid_is_not_validated_for_fine_rods': True,
          'existing_grid_radial_step_mm': cfg['radial_centers_mm'][1],
          'max_ring_center_arc_step_inside_envelope_mm': max(
              2*np.pi*r/n for r, n in ring_counts(cfg) if r <= R_ENVELOPE),
          'recommended_transverse_grid_pitch_mm': 1.0,
          'recommendation_is_design_choice_not_measured_resolution': True}
    for energy, design in designs.items():
        rods = design['rods']
        previous_file = PREVIOUS/'data'/f'derenzo_{energy}keV_geometry.json'
        previous = json.loads(previous_file.read_text(encoding='utf-8'))
        assert design['diameters_mm'] == previous['diameters_mm']
        assert len(rods) < previous['rod_count']
        centers = np.array([[r['x_mm'], r['y_mm']] for r in rods])
        radii = np.array([r['diameter_mm']/2 for r in rods])
        distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        gap = distances - radii[:, None] - radii[None, :]
        np.fill_diagonal(gap, np.inf)
        assert gap.min() >= 4-1e-10
        outer = np.linalg.norm(centers, axis=1) + radii
        inner = np.linalg.norm(centers, axis=1) - radii
        assert outer.max() <= R_PACKING + 1e-10
        assert inner.min() >= CENTRAL_ROD_FREE_RADIUS_MM
        assert R_ENVELOPE < min(cfg['semi_axes_mm'])
        assert HEIGHT <= cfg['height_mm']
        assert np.isclose(sum(r['source_selection_probability'] for r in rods), 1)
        densities = [r['source_selection_probability']/r['volume_mm3'] for r in rods]
        np.testing.assert_allclose(densities, densities[0], rtol=1e-13)
        groups = []
        for sector in design['sectors']:
            ix = [i for i, r in enumerate(rods) if r['sector_id'] == sector['sector_id']]
            assert len(ix) == sector['rows']*(sector['rows']+1)//2
            assert len(ix) >= 6
            dd = distances[np.ix_(ix, ix)].copy()
            np.fill_diagonal(dd, np.inf)
            np.testing.assert_allclose(dd.min(axis=1), sector['pitch_mm'], atol=1e-10)
            angle = np.deg2rad(sector['axis_angle_deg'])
            u = centers[ix] @ np.array([np.cos(angle), np.sin(angle)])
            v = centers[ix] @ np.array([-np.sin(angle), np.cos(angle)])
            # Independent half-plane and shoelace checks include full rod disks.
            tip = REGION_CENTER_RADIUS_MM - REGION_SIDE_MM/np.sqrt(3)
            base = REGION_CENTER_RADIUS_MM + REGION_SIDE_MM/(2*np.sqrt(3))
            side_clearance = (u-tip)/2 - np.abs(v)*np.sqrt(3)/2 - radii[ix]
            base_clearance = base-u-radii[ix]
            assert min(side_clearance.min(), base_clearance.min()) >= -1e-10
            np.testing.assert_allclose([u.mean(), v.mean()], [REGION_CENTER_RADIUS_MM, 0], atol=1e-10)
            vertices = np.array(sector['region_vertices_xy_mm'])
            sides = np.linalg.norm(vertices-np.roll(vertices, 1, axis=0), axis=1)
            np.testing.assert_allclose(sides, REGION_SIDE_MM, atol=1e-10)
            area = abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1))-
                       np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))/2
            np.testing.assert_allclose(area, np.sqrt(3)*REGION_SIDE_MM**2/4, atol=1e-9)
            assert np.linalg.norm(vertices, axis=1).max() <= R_ENVELOPE
            # Every region is strictly inside its own 60-degree wedge.
            region_u = vertices @ np.array([np.cos(angle), np.sin(angle)])
            region_v = vertices @ np.array([-np.sin(angle), np.cos(angle)])
            assert (region_u/2-np.abs(region_v)*np.sqrt(3)/2).min() > 0
            wedge_edge_clearance = u/2 - np.abs(v)*np.sqrt(3)/2 - radii[ix]
            assert wedge_edge_clearance.min() >= 2-1e-10
            center_hits = [int(np.sum(np.linalg.norm(xy-centers[i], axis=1) <= radii[i])) for i in ix]
            groups.append({'diameter_mm': sector['diameter_mm'], 'rod_count': len(ix),
                           'region_area_mm2_checked_by_shoelace': float(area),
                           'minimum_rod_to_region_edge_clearance_mm': float(min(side_clearance.min(), base_clearance.min())),
                           'region_centroid_radius_mm': float(np.linalg.norm(centers[ix].mean(axis=0))),
                           'nearest_neighbor_pitch_mm': float(dd.min()),
                           'legacy_grid_rods_with_no_cell_center': sum(n == 0 for n in center_hits),
                           'legacy_grid_centers_per_rod_min_max': [min(center_hits), max(center_hits)]})
        qa['energies'][str(energy)] = {
            'rod_count': len(rods), 'total_hot_volume_mm3': design['total_hot_volume_mm3'],
            'count_reduced_from_original_v2': previous['rod_count'],
            'all_regions_equal_area_mm2': np.sqrt(3)*REGION_SIDE_MM**2/4,
            'region_centroids_radius_mm': REGION_CENTER_RADIUS_MM,
            'minimum_rod_surface_radius_mm': float(inner.min()),
            'guaranteed_central_rod_free_radius_mm': CENTRAL_ROD_FREE_RADIUS_MM,
            'previous_geometry_sha256': sha(previous_file),
            'minimum_rod_surface_gap_mm': float(gap.min()),
            'outermost_rod_surface_radius_mm': float(outer.max()),
            'minimum_envelope_margin_mm': float(R_ENVELOPE-outer.max()),
            'fov_containment': 'analytic: radius 140 < ellipse minimum semiaxis 150; height 60 < 120',
            'groups': groups}
    for d in SHARED_DIAMETERS:
        a = [r for r in designs[218]['rods'] if r['diameter_mm'] == d]
        b = [r for r in designs[440]['rods'] if r['diameter_mm'] == d]
        np.testing.assert_array_equal([[r[k] for k in ('x_mm', 'y_mm', 'z_mm', 'diameter_mm')] for r in a],
                                      [[r[k] for k in ('x_mm', 'y_mm', 'z_mm', 'diameter_mm')] for r in b])
    qa['shared_diameters_mm'] = SHARED_DIAMETERS
    qa['shared_groups_geometry_identical'] = True
    qa['equal_emission_density_total_event_ratio_N440_over_N218'] = (
        designs[440]['total_hot_volume_mm3']/designs[218]['total_hot_volume_mm3'])
    return qa


def label(ax, x, y, text, size=14, color=NAVY, **kw):
    return ax.text(x, y, text, fontsize=size, color=color,
                   ha=kw.pop('ha', 'center'), va=kw.pop('va', 'center'), **kw)


def phantom(ax, design, size=16):
    energy = design['energy_keV']
    ax.set_aspect('equal')
    ax.set_xlim(-176, 176)
    ax.set_ylim(-176, 176)
    ax.axis('off')
    ax.add_patch(Circle((0, 0), R_ENVELOPE, fc=BG, ec='#A8B6C5', lw=1.3))
    for sector in design['sectors']:
        ax.add_patch(Polygon(sector['region_vertices_xy_mm'], closed=True,
                             fc='white', ec='#AEBCCB', lw=1, ls=(0, (3, 3))))
    ax.plot(0, 0, '+', color=MUTED, ms=6, mew=1)
    label(ax, 0, -10, 'FOV中心', size=9, color=MUTED)
    for r in design['rods']:
        ax.add_patch(Circle((r['x_mm'], r['y_mm']), r['diameter_mm']/2,
                            fc=COLORS[energy], ec='none'))
    for sector in design['sectors']:
        theta = np.deg2rad(sector['axis_angle_deg'])
        label(ax, 159*np.cos(theta), 159*np.sin(theta), f"Φ{sector['diameter_mm']}",
              size=size, weight='bold', color=COLORS[energy])
    # Exact XY projection of the analytic cylinders; no interpolation/smoothing.


def pitch_sketch(ax):
    ax.set_aspect('equal'); ax.set_xlim(-8, 46); ax.set_ylim(-19, 17); ax.axis('off')
    for x in (0, 20):
        ax.add_patch(Circle((x, 0), 5, fc='#8297AF', ec='none'))
        ax.plot(x, 0, '.', color='white', ms=3)
    ax.annotate('', (20, 11), (0, 11), arrowprops={'arrowstyle': '<->', 'color': NAVY, 'lw': 1})
    label(ax, 10, 15, '2d', 13)
    for x in (0, 20): ax.plot([x, x], [6, 12], color=RULE, lw=1)
    ax.annotate('', (5, -10), (-5, -10), arrowprops={'arrowstyle': '<->', 'color': NAVY, 'lw': 1})
    label(ax, 0, -15, 'd', 13)
    label(ax, 34, 0, '边缘间距\n= d', 11, MUTED)


def side_sketch(ax):
    ax.set_aspect('equal'); ax.set_xlim(-5, 87); ax.set_ylim(-13, 76); ax.axis('off')
    ax.add_patch(Rectangle((5, 0), 12, 60, fc='#8297AF', ec='none'))
    ax.annotate('', (30, 60), (30, 0), arrowprops={'arrowstyle': '<->', 'color': NAVY, 'lw': 1})
    label(ax, 40, 30, '60 mm', 13, ha='left')
    label(ax, 11, -9, '沿 z 方向', 11, MUTED)


def save_figure(fig, stem):
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(OUT/'figures'/f'{stem}.{extension}', dpi=300, facecolor='white')
    plt.close(fig)


def figures(designs):
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 14,
                         'axes.unicode_minus': False, 'svg.fonttype': 'path', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(17.6, 10.2), facecolor='white')
    fig.text(.05, .945, 'Derenzo式分辨率体模', color=NAVY, fontsize=27, weight='bold')
    fig.text(.051, .900, '等面积三角形分区   ·   各组向外移   ·   棒径单位：mm', color=MUTED, fontsize=14)
    fig.add_artist(Line2D([.05, .95], [.873, .873], color=RULE, lw=1.1))
    for energy, left in ((440, .05), (218, .53)):
        design = designs[energy]
        fig.text(left, .823, f'{energy} keV', color=COLORS[energy], fontsize=25, weight='bold')
        fig.text(left+.185, .827, f"{min(design['diameters_mm'])}–{max(design['diameters_mm'])} mm  ·  {design['rod_count']}根", color=MUTED, fontsize=14)
        phantom(fig.add_axes([left-.012, .245, .43, .57]), design, size=17)
    fig.text(.5, .237, '分区边长115 mm   ·   形心R90 mm   ·   中心R30 mm内不布棒', ha='center', color=MUTED, fontsize=13)
    fig.add_artist(Line2D([.05, .95], [.205, .205], color=RULE, lw=1.1))
    fig.text(.052, .157, '外包络 Φ280 × 60 mm', fontsize=19, color=NAVY, weight='bold')
    fig.text(.053, .112, '彩色热棒 = 1   ·   背景 = 0', fontsize=14, color=MUTED)
    fig.text(.053, .064, '设计示意，非重建结果', fontsize=11, color=MUTED)
    pitch_sketch(fig.add_axes([.49, .045, .23, .145]))
    side_sketch(fig.add_axes([.77, .045, .18, .145]))
    save_figure(fig, 'derenzo_440_218_design')
    for energy, design in designs.items():
        fig = plt.figure(figsize=(11.5, 10.5), facecolor='white')
        fig.text(.06, .94, f'{energy} keV｜Derenzo式分辨率体模', fontsize=25, color=COLORS[energy], weight='bold')
        fig.text(.061, .897, '等面积三角形分区   ·   形心R90 mm   ·   中心R30 mm内不布棒', fontsize=14, color=MUTED)
        phantom(fig.add_axes([.04, .225, .72, .66]), design, size=18)
        fig.text(.765, .72, '棒径 / mm', fontsize=14, color=MUTED)
        fig.text(.765, .66, '  '.join(map(str, design['diameters_mm'][:3])), fontsize=21, color=COLORS[energy], weight='bold')
        fig.text(.765, .61, '  '.join(map(str, design['diameters_mm'][3:])), fontsize=21, color=COLORS[energy], weight='bold')
        fig.text(.765, .52, f"共{design['rod_count']}根热棒", fontsize=15, color=NAVY)
        fig.text(.765, .46, '外包络 Φ280 mm', fontsize=13, color=MUTED)
        fig.text(.765, .415, '棒长 60 mm', fontsize=13, color=MUTED)
        fig.text(.765, .365, '分区边长115 mm', fontsize=13, color=MUTED)
        fig.add_artist(Line2D([.06, .94], [.215, .215], color=RULE, lw=1.1))
        pitch_sketch(fig.add_axes([.055, .035, .40, .16]))
        side_sketch(fig.add_axes([.51, .035, .28, .16]))
        fig.text(.94, .035, '设计示意，非重建结果', fontsize=10, color=MUTED, ha='right')
        save_figure(fig, f'derenzo_{energy}keV_design')


def design_notes(designs, qa):
    ratio = qa['equal_emission_density_total_event_ratio_N440_over_N218']
    summaries = []
    for energy, design in designs.items():
        groups = sorted(design['sectors'], key=lambda s: s['diameter_mm'])
        ds = '、'.join(str(s['diameter_mm']) for s in groups)
        ns = '、'.join(str(s['rod_count']) for s in groups)
        summaries.append(f"{energy} keV：棒径{ds} mm；分别{ns}根，共{design['rod_count']}根。")
    common = '/'.join(map(str, SHARED_DIAMETERS))
    notes = f'''Derenzo式双方案：独立440 / 218 keV单能模拟源设计
版本：2026-10-15 v4 equal_regions；按用户要求统一分区面积并增加中心间距，仅设计，尚未输运或重建。

1. 方案
{chr(10).join(summaries)}
尺寸范围按用户要求设置，不能代表系统已达到的分辨率。
两套{common} mm组的每根棒位置、长径、数量和排列完全相同。
每套均为六个同样大小的等边三角形分区，边长115 mm、面积5726.592983 mm²，形心距FOV中心90 mm。
分区轴角每隔60度，尖端朝向FOV中心；采用等边三角晶格，相邻棒中心距2d，棒表面间距d。
棒沿z轴，z∈[-30,30] mm；参考外包络直径280 mm。所有棒表面限制在半径135 mm内。
每区面积相同，棒数按棒径确定：小棒组更多，大棒组更少；各组保留完整的三角形行。
整数行约束使实际热棒阵列的外接三角形和边缘留白略有差异，文件分别记录分区面积与阵列外接面积。
各三角形分区的最近顶点距FOV中心23.604719 mm；全部热棒均在中心半径30 mm之外。
三角形与棒阵列均围绕各自形心排布，各组之间存在间隙；共同棒径组在两套设计中的几何完全一致。
源抽样概率按本版总体积重新归一化。440/218总棒数分别53/114，小于最初v2的73/163。

2. 坐标与FOV
CSV/JSON坐标均相对FOV中心，单位mm；世界中心沿用当前配置[0,-345,0] mm。
当前FOV为500×300×120 mm椭圆柱，完整容纳直径280×60 mm设计包络。
JSON给出分区、全部棒中心、尺寸、体积和源抽样概率；CSV便于后续仿真实现。
分区边长L=115，形心半径C=90；行数n=floor((L-sqrt(3)*d)/(2*d))+1。
三角阵列局部坐标：u=C+sqrt(3)*d*(k-2*(n-1)/3)；v=d*(2*j-k)，k=0..n-1，j=0..k。
将(u,v)按分区轴角30+60*s度旋转。阵列外接等边三角形边长为2*d*(n-1)+sqrt(3)*d。
分区局部顶点为(C-L/sqrt(3),0)、(C+L/(2*sqrt(3)),±L/2)。绘图虚线与JSON顶点一致。
独立检查面积、三边长度、形心半径、完整圆盘不越三角形、中心无棒区、FOV边界和棒间距。

3. 发射定义
两套分别仅发射指定218或440 keV单色gamma，方向均匀4π，每event一个初级gamma。
所有热棒具有相同单位体积发射密度；背景和另一能量初级源均为0。
抽棒概率必须正比于π*(d/2)^2*60，再在圆柱内部均匀抽样；不可每根棒等概率。
这是纯单能gamma源，不是Ac225衰变链源，不应把事件数称为Ac225衰变数。
218纯单能实验没有440源贡献，后续重建不能机械套用旧双能440→218固定串扰项。
参考外包络只约束源位置，不自动创建PMMA外壳、水体或其他材料；实际材料模型需在输运实施时明确。
源真值应使用解析圆柱定义；绘图不是体素化后的模拟源，也不是重建/MIP结果。

4. 采样与公平比较
当前径向网格为6 mm，包络内环中心最大切向弧长约{qa['max_ring_center_arc_step_inside_envelope_mm']:.3f} mm。
该网格不足以严谨评价6 mm等细棒组；插值或缩小输出像素不能补足原求解网格信息。
建议正式横向分辨率研究用1 mm级重建网格及匹配的系统矩阵，并进行网格收敛检查。
1 mm是本设计用于控制数值采样的建议，不是已测得的系统分辨率；本次未更改网格或生成矩阵。
轴向棒长60 mm，当前3 mm层厚可保留；横向分辨率定量建议先用中间轴向区域的同厚度切片平均。
目前20视角、18度间隔可作为现有采样条件；最终结果需与视角数一并报告，不能只归因于能量。
不加平滑，不用亮度拟合；显示与定量固定规则。可在三种晶格方向提取中心连线峰谷曲线。
报告最小可分辨棒径、对应中心距与曲线；不能直接把可分辨棒径称为点源FWHM。
共同尺寸在同一位置，可用于比较；不同尺寸和其他分区不能混作纯能量变化的单一因果对照。

5. 计数
本次未指定/提交总事件数。记录每能量实际发射N_gamma和最终有效事件数。
若以相同热棒单位体积发射密度×采集时间比较，共同棒的预期发射光子数相同，
则N440/N218 = V440/V218 = {ratio:.12f}，两套总光子数不会完全相同。
若选择两套相同总N_gamma，应同时报告发射密度差异，不能声称共同棒也有相同发射光子数。
体积：218 = {designs[218]['total_hot_volume_mm3']:.6f} mm³；440 = {designs[440]['total_hot_volume_mm3']:.6f} mm³。

6. 来源
六分区与中心距2d参考Derenzo-like结构，棒径及其位置为本工程新设计：
Borghi et al., Phys. Med. Biol. 63 (2018) 155006，Derenzo phantom方法部分。
{REFERENCE}
工程约束来自experiments/ELLIPSE500x300_H120/config.json与geometry.py。
旧HotRod MATLAB生成器仅参考其三角阵列和体积加权思想，未复用其旧坐标/剂量/旋转设置。
'''
    (OUT/'design_notes_zh.txt').write_text(notes, encoding='utf-8')


def main():
    for name in ('data', 'figures'):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    cfg = json.loads((H/'config.json').read_text(encoding='utf-8'))
    source_paths = [H/'config.json', H/'geometry.py', ROOT/'Geant4Sim/GenPhan_HotRodPhantom_Rotate_3D.m']
    before = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    designs = {e: build(e) for e in (218, 440)}
    qa = verify(designs, cfg)
    for energy, design in designs.items():
        write_json(OUT/'data'/f'derenzo_{energy}keV_geometry.json', design)
        path = OUT/'data'/f'derenzo_{energy}keV_rods.csv'
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(design['rods'][0]))
            writer.writeheader(); writer.writerows(design['rods'])
    figures(designs)
    design_notes(designs, qa)
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    qa['source_files_sha256'] = before
    qa['source_files_unchanged_after_generation'] = True
    qa['generator_sha256'] = sha(__file__)
    qa['structural_reference'] = REFERENCE
    write_json(OUT/'data'/'geometry_qa.json', qa)
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'artifact_sha256.json' and '__pycache__' not in p.parts)
    write_json(OUT/'data'/'artifact_sha256.json', {str(p.relative_to(OUT)): sha(p) for p in files})
    print(json.dumps({'output': str(OUT), 'checks_passed': True,
                      'rod_counts': {e: d['rod_count'] for e, d in designs.items()},
                      'shared_groups_identical': True,
                      'same_emission_density_N440_over_N218': qa['equal_emission_density_total_event_ratio_N440_over_N218']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
