#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plotly 3D 交互式可视化：SPECT 系统几何（探测器晶体 + 准直器孔 + FOV）。

晶体/屏蔽体画为立方体（按真实尺寸），准直器孔画为圆柱孔，FOV/准直器板为半透明立方体。
由 MATLAB 端 plot_geometry_3d.m 调用。直接读取 MATLAB 保存的 .mat 几何数据，输出交互式 HTML。

用法：
    python plot_geometry_3d.py <input.mat> <output.html> "<title>"

.mat 文件字段（由 plot_geometry_3d.m 写入）：
    det_x, det_y, det_z          : (N,) 晶体中心坐标 mm（已平移到 FOV 原点坐标系）
    det_dx, det_dy, det_dz       : (N,) 晶体尺寸 mm（X, Y_朝FOV, Z）
    det_flag                     : (N,) CrystalMatrix 标签（1=闪烁体, >1=屏蔽体）
    hole_cx, hole_cy1, hole_cy2, hole_cz, hole_r : (M,) 准直器孔参数
    col_w, col_h                 : 标量，准直器板宽/高 mm
    fov_xr, fov_yr, fov_zr       : (2,) FOV 的 X/Y/Z 范围
"""
import sys
import numpy as np
import scipy.io as sio
import plotly.graph_objects as go


def box_mesh_single(xr, yr, zr, color, opacity, name):
    """单个半透明立方体 Mesh3d（8 顶点 12 三角面）。"""
    xm, xM = xr; ym, yM = yr; zm, zM = zr
    x = [xm, xM, xM, xm, xm, xM, xM, xm]
    y = [ym, ym, yM, yM, ym, ym, yM, yM]
    z = [zm, zm, zm, zm, zM, zM, zM, zM]
    i = [0, 1, 0, 4, 0, 1, 2, 3, 4, 5, 1, 2]
    j = [1, 2, 1, 5, 2, 3, 3, 7, 5, 6, 5, 6]
    k = [2, 3, 2, 7, 4, 7, 6, 6, 7, 7, 6, 3]
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                     color=color, opacity=opacity, name=name,
                     flatshading=True, showscale=False, hoverinfo='name')


def boxes_mesh_array(cx, cy, cz, sx, sy, sz, color, opacity, name):
    """多个立方体合并到一个 Mesh3d（数组化，性能优）。

    每个立方体中心 (cx,cy,cz)，尺寸 (sx,sy,sz)。返回单个 Mesh3d trace。
    晶体较多时这样比每晶体一个 trace 快得多。
    """
    n = len(cx)
    # 每个立方体 8 顶点：off 是归一化方向（±1），half 是半边长（size/2）
    # 顶点 = center + off * half，这样边长 = size（正确）
    off = np.array([[ -1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
                    [ -1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]], dtype=float)
    # 12 三角面（标准立方体面定义，每个矩形面拆 2 个三角形）
    faces = np.array([
        [0, 2, 1], [0, 3, 2],   # 底面 z-
        [4, 5, 6], [4, 6, 7],   # 顶面 z+
        [0, 1, 5], [0, 5, 4],   # 前 y-
        [2, 3, 7], [2, 7, 6],   # 后 y+
        [0, 4, 7], [0, 7, 3],   # 左 x-
        [1, 2, 6], [1, 6, 5],   # 右 x+
    ])

    # 构造所有顶点：(n*8, 3)
    half = np.stack([sx, sy, sz], axis=1) * 0.5   # (n,3)
    centers = np.stack([cx, cy, cz], axis=1)      # (n,3)
    verts = (centers[:, None, :] + off[None, :, :] * half[:, None, :]).reshape(-1, 3)
    # 构造所有面：(n*12, 3)，每个面的顶点全局索引 = base + face_local
    bases = (np.arange(n) * 8)[:, None, None]
    all_faces = (bases + faces[None, :, :]).reshape(-1, 3)

    return go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                     i=all_faces[:, 0], j=all_faces[:, 1], k=all_faces[:, 2],
                     color=color, opacity=opacity, name=name,
                     flatshading=True, showscale=False, hoverinfo='name')


def cylinders_mesh_array(cx, cy, cz, radius, height, color, opacity, name):
    """多个圆柱体（沿 Y 轴）合并到一个 Mesh3d。用于批量准直器孔。

    每个圆柱中心 (cx,cy,cz)，半径 radius，高度 height（沿 Y）。
    用 n_seg 段离散圆周，只画侧面（不画顶/底面，因为孔是穿透的）。
    """
    n = len(cx)
    n_seg = 16
    theta = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # 每个圆柱 2*n_seg 个顶点（上环 + 下环）
    # 顶点偏移：(n_seg 上, n_seg 下)
    # 构造所有顶点：(n * 2*n_seg, 3)
    # 上环
    xu = (cx[:, None] + radius[:, None] * cos_t[None, :]).ravel()
    zu = (cz[:, None] + radius[:, None] * sin_t[None, :]).ravel()
    yu = np.repeat(cy + height / 2, n_seg)
    # 下环
    xd = (cx[:, None] + radius[:, None] * cos_t[None, :]).ravel()
    zd = (cz[:, None] + radius[:, None] * sin_t[None, :]).ravel()
    yd = np.repeat(cy - height / 2, n_seg)
    xs = np.concatenate([xu, xd])
    ys = np.concatenate([yu, yd])
    zs = np.concatenate([zu, zd])
    n_per = n_seg   # 每环点数
    # 每个圆柱的侧面三角面：相邻两点 (t, t+1) 形成两个三角形（上t, 下t, 上t+1）和（上t+1, 下t, 下t+1）
    ii, jj, kk = [], [], []
    for i in range(n):
        base_u = i * n_per
        base_d = n * n_per + i * n_per
        for t in range(n_per):
            t2 = (t + 1) % n_per
            # 三角形 1: 上t, 下t, 上t2
            ii.append(base_u + t);  jj.append(base_d + t);  kk.append(base_u + t2)
            # 三角形 2: 上t2, 下t, 下t2
            ii.append(base_u + t2); jj.append(base_d + t);  kk.append(base_d + t2)
    return go.Mesh3d(x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
                     color=color, opacity=opacity, name=name,
                     flatshading=True, showscale=False, hoverinfo='name')


def main():
    if len(sys.argv) < 3:
        print("usage: python plot_geometry_3d.py <input.mat> <output.html> [title]")
        sys.exit(1)

    mat_path = sys.argv[1]
    html_path = sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "SPECT Geometry"

    d = sio.loadmat(mat_path)
    def get(name):
        return np.asarray(d[name]).ravel()

    det_x = get('det_x'); det_y = get('det_y'); det_z = get('det_z')
    det_dx = get('det_dx'); det_dy = get('det_dy'); det_dz = get('det_dz')
    det_flag = get('det_flag')
    det_is_highz = det_flag > 1    # 标签 >1 = 屏蔽体（W/Pb）；=1 = 闪烁体

    hole_cx = get('hole_cx'); hole_cz = get('hole_cz'); hole_r = get('hole_r')
    hole_cy1 = float(get('hole_cy1').item())
    hole_cy2 = float(get('hole_cy2').item())
    col_w = float(get('col_w').item())
    col_h = float(get('col_h').item())

    fov_xr = get('fov_xr'); fov_yr = get('fov_yr'); fov_zr = get('fov_zr')

    traces = []

    # 颜色方案（贴近普遍认知）
    COLOR_FOV       = 'rgb(50,180,80)'     # 绿色
    COLOR_COL_PLATE = 'rgb(55,55,60)'      # 深灰/黑灰（准直器板）
    COLOR_SCIN      = 'rgb(250,200,40)'    # 黄色（闪烁体晶体）
    COLOR_SHIELD    = 'rgb(30,60,180)'     # 深蓝色（高Z屏蔽体）
    COLOR_HOLE      = 'rgb(245,245,250)'   # 近白（准直器孔）

    # ---- FOV（绿色半透明立方体）----
    traces.append(box_mesh_single(fov_xr, fov_yr, fov_zr,
                                  COLOR_FOV, 0.10, 'FOV'))

    # ---- 准直器板（深灰半透明立方体）----
    traces.append(box_mesh_single([-col_w/2, col_w/2], [hole_cy1, hole_cy2], [-col_h/2, col_h/2],
                                  COLOR_COL_PLATE, 0.55, '准直器板'))

    # ---- 闪烁体晶体（黄色立方体，完全不透明；按 Y 层分组，可单独隐藏某层）----
    scin = ~det_is_highz
    if np.any(scin):
        y_vals = np.unique(det_y[scin])
        for yv in y_vals:
            mask = scin & (det_y == yv)
            if np.any(mask):
                traces.append(boxes_mesh_array(
                    det_x[mask], det_y[mask], det_z[mask],
                    det_dx[mask], det_dy[mask], det_dz[mask],
                    COLOR_SCIN, 1.0, f'闪烁体 Y={yv:.0f}mm'))

    # ---- 高Z屏蔽体（深蓝色立方体，完全不透明；按 Y 层分组）----
    if np.any(det_is_highz):
        y_vals = np.unique(det_y[det_is_highz])
        for yv in y_vals:
            mask = det_is_highz & (det_y == yv)
            if np.any(mask):
                traces.append(boxes_mesh_array(
                    det_x[mask], det_y[mask], det_z[mask],
                    det_dx[mask], det_dy[mask], det_dz[mask],
                    COLOR_SHIELD, 1.0, f'屏蔽体 Y={yv:.0f}mm'))

    # ---- 准直器孔（圆柱孔，近白色；合并为单个 Mesh3d）----
    n_holes = len(hole_cx)
    col_t = abs(hole_cy2 - hole_cy1)
    cy_mid = (hole_cy1 + hole_cy2) / 2
    step = max(1, n_holes // 200)
    idx_arr = np.arange(0, n_holes, step)
    traces.append(cylinders_mesh_array(
        hole_cx[idx_arr], np.full(len(idx_arr), cy_mid), hole_cz[idx_arr],
        hole_r[idx_arr], np.full(len(idx_arr), col_t * 1.05),
        COLOR_HOLE, 0.95, f'准直器孔({n_holes}个,显示{len(idx_arr)})'))

    # ---- 布局 ----
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X (mm) 横向', backgroundcolor='rgb(240,240,250)'),
            yaxis=dict(title='Y (mm) 深度', backgroundcolor='rgb(240,250,240)'),
            zaxis=dict(title='Z (mm) 轴向', backgroundcolor='rgb(250,240,240)'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.85)'),
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode='closest',
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.write_html(html_path, include_plotlyjs='cdn', full_html=True)
    print(f"  3D interactive visualization saved: {html_path}")


if __name__ == '__main__':
    main()
