#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""可视化 Geant4 GPS macro 里的 Contrast Phantom 源分布。

解析 .mac 文件里的 /gps/source 定义，用 Plotly 画 3D 交互式 HTML。
用法：
    python visualize_phantom.py <macro_file.mac> <output.html>
    python visualize_phantom.py Macro/.../1.mac phantom_3d.html
"""
import sys
import re
import numpy as np
import plotly.graph_objects as go


def parse_macro(mac_path):
    """解析 GPS macro，提取所有 source 定义。"""
    with open(mac_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    sources = []
    current = {}
    source_idx = 0

    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue

        if '/gps/source/add' in line:
            # 保存前一个 source
            if current:
                sources.append(current)
            current = {'weight': float(line.split()[-1])}
            source_idx += 1
        elif '/gps/particle' in line and not current:
            # 第一个 source（无 /gps/source/add，隐含权重 1.0）
            current = {'weight': 1.0}
        elif '/gps/energy' in line:
            parts = line.split()
            current['energy'] = float(parts[-2])
        elif '/gps/pos/centre' in line:
            parts = line.split()
            current['cx'] = float(parts[-4])
            current['cy'] = float(parts[-3])
            current['cz'] = float(parts[-2])
        elif '/gps/pos/radius' in line:
            current['radius'] = float(line.split()[-2])
        elif '/gps/pos/halfz' in line:
            current['halfz'] = float(line.split()[-2])
        elif '/run/beamOn' in line:
            if current:
                sources.append(current)
                current = {}

    if current:
        sources.append(current)

    # 标注能量类型
    for s in sources:
        e = s.get('energy', 0)
        if abs(e - 218) < 5:
            s['type'] = 'Fr-218'
            s['color'] = 'rgb(250,200,40)'   # 黄色
        elif abs(e - 440) < 5:
            s['type'] = 'Bi-440'
            s['color'] = 'rgb(30,90,200)'    # 蓝色
        else:
            s['type'] = f'{e:.0f}keV'
            s['color'] = 'rgb(150,150,150)'

    return sources


def cylinder_mesh(cx, cy, cz, radius, halfz, color, opacity, name, n_seg=32):
    """构造一个圆柱体的 Mesh3d（沿 Z 轴）。"""
    theta = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # 上下环
    xu = (cx + radius * cos_t).tolist() * 1
    yu = (cy + radius * sin_t).tolist() * 1
    zu = [cz + halfz] * n_seg
    xd = (cx + radius * cos_t).tolist() * 1
    yd = (cy + radius * sin_t).tolist() * 1
    zd = [cz - halfz] * n_seg

    xs = xu + xd
    ys = yu + yd
    zs = zu + zd

    ii, jj, kk = [], [], []
    for t in range(n_seg):
        t2 = (t + 1) % n_seg
        # 侧面三角面
        ii += [t, t2]
        jj += [t2, t]
        kk += [t + n_seg, t + n_seg]
        # 底面
        ii += [0, t2]
        jj += [t, 0]
        kk += [t2, t]
        # 顶面
        ii += [n_seg, n_seg + t2]
        jj += [n_seg + t, n_seg]
        kk += [n_seg + t2, n_seg + t]

    return go.Mesh3d(x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
                     color=color, opacity=opacity, name=name,
                     flatshading=True, showscale=False, hoverinfo='name')


def main():
    if len(sys.argv) < 2:
        print("用法: python visualize_phantom.py <macro.mac> [output.html]")
        sys.exit(1)

    mac_path = sys.argv[1]
    html_path = sys.argv[2] if len(sys.argv) > 2 else 'phantom_3d.html'

    sources = parse_macro(mac_path)
    print(f"解析到 {len(sources)} 个源:")
    for i, s in enumerate(sources):
        r = s.get('radius', 0)
        hz = s.get('halfz', 0)
        cx = s.get('cx', 0)
        cy = s.get('cy', 0)
        print(f"  [{i}] {s['type']:8s} E={s.get('energy',0):.0f}keV  "
              f"r={r:.1f}mm hz={hz:.1f}mm  center=({cx:.1f},{cy:.1f},{s.get('cz',0):.1f})  "
              f"w={s['weight']:.4f}")

    traces = []
    annotations = []

    for i, s in enumerate(sources):
        r = s.get('radius', 1)
        hz = s.get('halfz', 1)
        cx = s.get('cx', 0)
        cy = s.get('cy', 0)
        cz = s.get('cz', 0)

        # 判断是否背景（大圆柱）
        is_bg = r > 50
        opacity = 0.15 if is_bg else 0.85
        name = f"{s['type']} {'bg' if is_bg else f'rod'}"

        traces.append(cylinder_mesh(cx, cy, cz, r, hz,
                                     s['color'], opacity, name))

        # 标注（非背景）
        if not is_bg:
            annotations.append(
                dict(x=cx, y=cy, z=cz + hz + 2,
                     text=f"{s['type']}\n{r*2:.0f}mm\nw={s['weight']:.3f}",
                     font=dict(size=9, color=s['color']),
                     showarrow=False))

    layout = go.Layout(
        title=f'Contrast Phantom Sources ({mac_path.split("/")[-1]})',
        scene=dict(
            xaxis=dict(title='X (mm)', backgroundcolor='rgb(240,240,250)'),
            yaxis=dict(title='Y (mm) 深度', backgroundcolor='rgb(240,250,240)'),
            zaxis=dict(title='Z (mm) 轴向', backgroundcolor='rgb(250,240,240)'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            annotations=annotations,
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode='closest',
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.write_html(html_path, include_plotlyjs='cdn', full_html=True)
    print(f"\n可视化已保存: {html_path}")


if __name__ == '__main__':
    main()
