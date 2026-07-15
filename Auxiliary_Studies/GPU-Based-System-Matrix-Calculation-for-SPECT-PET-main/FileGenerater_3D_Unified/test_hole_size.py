import numpy as np
import plotly.graph_objects as go

# 单个孔：半径3mm，导出 PNG 测量视觉直径
def cylinders_mesh_array(cx, cy, cz, radius, height, color, opacity, name):
    n = len(cx); n_seg = 32
    theta = np.linspace(0, 2*np.pi, n_seg, endpoint=False)
    cos_t = np.cos(theta); sin_t = np.sin(theta)
    xu = (cx[:,None] + radius[:,None]*cos_t[None,:]).ravel()
    zu = (cz[:,None] + radius[:,None]*sin_t[None,:]).ravel()
    yu = np.repeat(cy + height/2, n_seg)
    xd = (cx[:,None] + radius[:,None]*cos_t[None,:]).ravel()
    zd = (cz[:,None] + radius[:,None]*sin_t[None,:]).ravel()
    yd = np.repeat(cy - height/2, n_seg)
    xs = np.concatenate([xu, xd]); ys = np.concatenate([yu, yd]); zs = np.concatenate([zu, zd])
    ii, jj, kk = [], [], []
    for i in range(n):
        bu = i*n_seg; bd = n*n_seg + i*n_seg
        for t in range(n_seg):
            t2 = (t+1) % n_seg
            ii += [bu+t, bu+t2]; jj += [bd+t, bd+t]; kk += [bu+t2, bd+t2]
    return go.Mesh3d(x=xs, y=ys, z=zs, i=ii, j=jj, k=kk, color=color, opacity=opacity, name=name)

# 孔：中心(0,0,0)，半径3mm，高3mm
hole = cylinders_mesh_array(np.array([0.0]), np.array([0.0]), np.array([0.0]),
                            np.array([3.0]), np.array([3.0]), 'rgb(245,245,250)', 0.95, 'hole')
# 参考线：X=±3 (半径), X=±6 (直径) 的标记点
ref_r = go.Scatter3d(x=[-3,3], y=[0,0], z=[0,0], mode='markers+text',
                     text=['X=-3(r)', 'X=+3(r)'], marker=dict(size=3, color='red'))
ref_d = go.Scatter3d(x=[-6,6], y=[0,0], z=[0,0], mode='markers+text',
                     text=['X=-6(D?)', 'X=+6(D?)'], marker=dict(size=3, color='blue'))

fig = go.Figure([hole, ref_r, ref_d])
fig.update_layout(scene=dict(aspectmode='data',
                             xaxis=dict(range=[-8,8], title='X(mm)'),
                             yaxis=dict(title='Y(mm)'),
                             zaxis=dict(title='Z(mm)')),
                  title='Hole radius=3mm test (expect diameter 6mm)')
fig.write_image('test_hole.png', width=800, height=600)
print('saved test_hole.png')

# 同时打印顶点范围确认
cx=np.array([0.0]); cz=np.array([0.0]); radius=np.array([3.0])
theta=np.linspace(0,2*np.pi,32,endpoint=False)
xu=(cx[:,None]+radius[:,None]*np.cos(theta)[None,:]).ravel()
print('vertex X range: %.2f to %.2f (diameter=%.1f, expect 6.0)' % (xu.min(), xu.max(), xu.max()-xu.min()))
