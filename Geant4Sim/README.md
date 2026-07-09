# Geant4Sim

本目录保存 Geant4 蒙卡模拟相关代码和输入生成脚本。它与 `GenProj/` 的职责不同：

- `Geant4Sim/`：生成 Geant4 macro、体模源、预览文件，并通过 `Geant4Code/` 做蒙卡模拟，输出 `CntStat` 和 `List`。
- `GenProj/`：不运行 Geant4，只用已有系统矩阵 `SysMat_polar` 做前投影，快速生成用于单光子验证的 `CntStat`。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `Geant4Code/` | Geant4 C++ 工程，定义探测器、事件分类、运行输出等 |
| `Macro/` | MATLAB 脚本生成的 Geant4 macro，每个旋转角通常一个 `.mac` |
| `Preview/` | MATLAB 生成的体模预览、体模 raw/mhd/mat 和 voxel list |
| `3D_DRO_Hoffman_v6_20160331_DICOM/` | Hoffman 脑模体原始 DICOM 输入 |
| `3D_DRO_Hoffman_v6_raw/` | Hoffman 原始 raw 数据 |

## MATLAB macro / 体模生成脚本

| 文件 | 用途 |
| --- | --- |
| `ContrastPhantom_Rotate_3D.m` | 单能量 contrast phantom 旋转 macro |
| `ContrastPhantom_DualEnergy_Rotate_3D.m` | 225Ac 双能量 contrast phantom，背景 218/440 keV 按分支比发射，6 个热圆柱交替为 218/440 keV |
| `GenPhan_HotRodPhantom_Rotate_3D.m` | hot-rod phantom 旋转 macro |
| `Cylinder_Phantom_Rotate_3D.m` | 圆柱 phantom macro |
| `point_array_Rotate_3D.m` | 点源阵列 macro |
| `BrainPhantom_HoffmanMontage_3D.m` | Hoffman montage 脑模体生成 |
| `BrainPhantom_HoffmanRawCompressed_3D.m` | Hoffman raw 压缩体模生成 |
| `BrainPhantom_SliceStack_3D.m` | 基于切片堆栈的脑模体生成 |
| `BrainPhantom_Truncated_3D.m` | 截断脑模体生成 |
| `HoffmanCompressed_Rotate_3D.m` | Hoffman 压缩体模旋转 macro |
| `visualize_phantom.py` | 从 macro 解析源几何并生成 HTML 预览 |

## 当前双能量 225Ac contrast phantom

`ContrastPhantom_DualEnergy_Rotate_3D.m` 当前设定：

- 背景圆柱直径：`240 mm`
- 热圆柱直径：`10:4:30 mm`
- 圆柱高度：`30 mm`
- 背景中心：Geant4 世界坐标 `(0, -245, 0) mm`
- 218 keV 分支比：`0.114`
- 440 keV 分支比：`0.261`
- 热圆柱能量：rod 1/3/5 为 218 keV，rod 2/4/6 为 440 keV
- 默认旋转数：`rotate_num = 60`
- 输出目录：`Macro/ContrastPhantom_DualEnergy_10_30_240_30_225Ac/`

运行该脚本会生成每个角度的 `.mac`，并调用 `visualize_phantom.py` 生成 `phantom_3d.html` 预览。预览只用于核对源几何和能量分布，不参与重建。

## Geant4Code 输出

`Geant4Code/` 的 C++ 工程负责探测器响应和事件分类。当前代码中双能量单光子计数按能窗拆分输出：

| 输出文件 | 含义 |
| --- | --- |
| `CntStat_218.csv` | 落入 218 keV 能窗的单光子计数 |
| `CntStat_440.csv` | 落入 440 keV 能窗的单光子计数 |
| `List.csv` | 康普顿候选事件列表，列格式为 `[det1, energy1, det2, energy2, flag]` |
| `EnergySpectrum.csv` | 展宽后能量谱统计 |
| `EventType.csv` | 事件类别统计 |
| `Detector.csv` | Geant4 探测器单元编号与坐标 |

事件分类逻辑在 `Geant4Code/src/EventAction.cc` 中：优先判断单光子能窗，若事件最大晶体能量落入 440 keV 或 218 keV 能窗，则分别累计到 `CntStat_440` 或 `CntStat_218`；否则再进入康普顿事件判断。能量分辨率以 511 keV 处 13% 为参考，并按 `1/sqrt(E)` 标度到 218/440 keV。

## 与重建数据目录的关系

Geant4 运行通常在每个旋转角输出一行 `CntStat_*.csv`，以及对应角度的 `List.csv`。整理到主重建工程时，建议使用以下结构：

```text
CntStat/<energy>keV_RotateNum<N>/CntStat_<phantom>_<count>.csv
List/<energy>keV_RotateNum<N>/List_<phantom>_<count>/<angle>.csv
```

例如双能量 225Ac contrast phantom 可整理为：

```text
CntStat/218keV_RotateNum60/CntStat_ContrastPhantom_DualEnergy_10_30_240_30_225Ac_<count>.csv
CntStat/440keV_RotateNum60/CntStat_ContrastPhantom_DualEnergy_10_30_240_30_225Ac_<count>.csv
```

若只是验证单光子重建链路，可先使用 `GenProj/GenProj_ContrastPhantom_DualEnergy_PolarCoor.m` 通过系统矩阵前投影生成 `RotateNum20` 的基础 `CntStat`，再决定是否运行完整 Geant4。
