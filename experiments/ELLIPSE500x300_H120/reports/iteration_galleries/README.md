# 1e9 正式重建：六路迭代图集

所有图均依次展示真值、50、500、1000、3000、5000、10000 次迭代。
六行分别为 440 单光子、440 Compton、440 JSCC 联合、218 串窗校正、
440 单光子＋218、440 JSCC＋218。黑色为高值，白色为低值；显示完整
横向 FOV，无图像平滑。重建的极坐标点按每层 XY 线性插值到 3 mm
Cartesian 网格。每个小图独立以本图非零值的 99.5 百分位作显示上限，
**不能凭小图灰度比较不同迭代的绝对强度**；对应的峰值和切片总和记录在
每张图的同名 JSON 中，中心层另有强度变化曲线。

| 数据组 | 中心层 | 下端/下侧 | 上端/上侧 | 强度曲线 |
|---|---|---|---|---|
| Φ300×120，新距离 | [z=+1.5](CircleNewDist_1e9_1640673_center.png) | [z=−58.5](CircleNewDist_1e9_1640673_lower_edge.png) | [z=+58.5](CircleNewDist_1e9_1640673_upper_edge.png) | [曲线](CircleNewDist_1e9_1640673_amplitude.png) |
| 椭圆均匀源 | [z=+1.5](EllipseUniform_1e9_1640929_center.png) | [z=−58.5](EllipseUniform_1e9_1640929_lower_edge.png) | [z=+58.5](EllipseUniform_1e9_1640929_upper_edge.png) | [曲线](EllipseUniform_1e9_1640929_amplitude.png) |
| 椭圆热柱 | [z=+1.5](EllipseContrast_1e9_1641013_center.png) | [z=−46.5](EllipseContrast_1e9_1641013_lower_rods.png) | [z=+46.5](EllipseContrast_1e9_1641013_upper_rods.png) | [曲线](EllipseContrast_1e9_1641013_amplitude.png) |
| XCAT | [z=+1.5](XCAT_1e9_1641014_center.png) | [z=−46.5](XCAT_1e9_1641014_lower.png) | [z=+46.5](XCAT_1e9_1641014_upper.png) | [曲线](XCAT_1e9_1641014_amplitude.png) |

热柱放大图以红线标出对应能量的真实热柱边界，包含 218 校正与 440
JSCC 两路：[中心](EllipseContrast_1e9_1641013_zoom_center.png)、
[长轴正侧](EllipseContrast_1e9_1641013_zoom_long_plus.png)、
[长轴负侧](EllipseContrast_1e9_1641013_zoom_long_minus.png)、
[短轴正侧](EllipseContrast_1e9_1641013_zoom_short_plus.png)、
[短轴负侧](EllipseContrast_1e9_1641013_zoom_short_minus.png)。

真值来自本实验 Geant4 规则体模几何及 XCAT 的 `truth_3mm.npz`。
末两路真值列显示两能量空间分布之和，仅用作结构参考；对应重建也是
γ 通道图之和，**不是 ²²⁵Ac 母体活度图**。所选历史帧来自已核验的
1e9、20 视角、10000 次正式结果；每路第 10000 次历史帧逐字节哈希
与最终图一致。原始历史及选中帧在各集群实验目录，图与 JSON 可由
`fetch_gallery_frames.py`、`plot_iteration_galleries.py`、
`plot_contrast_rod_galleries.py`、`plot_iteration_amplitude.py` 复现。
