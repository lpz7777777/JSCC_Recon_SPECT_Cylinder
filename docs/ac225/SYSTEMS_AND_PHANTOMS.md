# 系统与体模

## 共用FOV与成像网格

当前椭圆FOV为500×300mm、完整轴向120mm；132040全圆点中78920个活动单元。径向间距6mm、轴向3mm，角点数随半径分段增加，单元体积不相同。真实采集默认20个旋转角，18°间隔；60角为独立EHE模型合成对照，不改变原生图像基底。

NEMA和Derenzo源主体H60。图中中心72mm MIP只改变显示范围，不把响应域或真实源高度改成72mm。当前体积和重叠定义见[geometry.py](../../experiments/ELLIPSE500x300_H120/geometry.py)及[科学口径](SCIENCE_CONVENTIONS.md)。

## JSCC与EHE

JSCC探头支持单光子计数及Compton双作用点观测；图中SC与同能量SC+Compton联合使用不同观测链，纯Compton单独标记。探头尺寸、晶体顺序和材料按各正式发布中的参数及Geant4源代码身份核对，不能从历史目录名称推断当前二进制。

EHE为准直单光子系统。NaI物理bin为4×10×4mm（X、深度Y、Z），68×34共2312个，正面272×136mm、深度10mm，本轮未细分DOI。65536个点/可见面是响应面积分节点，不是把物理晶体切成65536个读出像素。当前加密A218/A440保持探头几何，C440→218未同步重算。

当前EHE未增加新侧屏蔽。四侧10mm铅框只保留为设计方案；准直器前端298.5mm、完整FOV最小48.5mm间隙来自独立几何审计。

[探头与场景图](../../Analysis_Plots/research/advisor_progress_20261014/使用说明.txt) · [EHE几何间隙](../../Analysis_Plots/research/ehe_fov_clearance_20261102/REPORT.html) · [侧屏蔽方案](../../Analysis_Plots/research/ehe_side_shield_design_20261029/REPORT.html)

## NEMA Body

采用原3mm三维真值和manifest的原3D球心，六球直径10/13/17/22/28/37mm，并包含中心肺结构。218热球为10/17/28mm，440热球为13/22/37mm；精确背景、球、肺及源分布以原数组为准，不从重建图反推。两能场由合同定义的gamma代理源生成，不是假称完整225Ac衰变链。

[分能量体模结构](../../Analysis_Plots/research/nema_structure_20261014/README.md) · [原源manifest](../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/manifest.json) · [truth_3mm.npz](../../experiments/ELLIPSE500x300_H120/generated/NEMA_Body_H60/truth_3mm.npz)

## 独立单能Derenzo

当前唯一设计为20261015_equal_regions，各棒径对应的三角区域面积相近，棒区与FOV中心之间留空。

| 能量 | 棒径/mm | 棒数 | 数据语义 |
|---|---|---:|---|
| 218keV | 6 / 8 / 10 / 12 / 14 / 16 | 114 | 此体模独立单能输运 |
| 440keV | 10 / 12 / 14 / 16 / 18 / 20 | 53 | 此体模独立单能输运 |

两个能量的几何不同，每个探头/能量各自发射5e9或5e10 gamma；单能218不注入另一440体模的背景。现轮复用原矩阵，不据径向6mm网格图像宣布最小6mm棒分辨率。

[当前解析设计与原图](../../Analysis_Plots/research/derenzo_218_440_design_20261015_equal_regions/design_notes_zh.txt) · [设计研究页](studies/design/derenzo-design-selected.md)
