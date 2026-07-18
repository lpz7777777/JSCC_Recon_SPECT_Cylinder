# Geant4Sim

## GAGG intrinsic-response diagnostic

`Geant4Code_GAGGIntrinsicResponse` is an isolated single-crystal project used
to measure physical photopeak containment for the JSCC `3x3x3 mm` and
`2x6x2 mm` GAGG crystals at 218 and 440 keV. It separates first-PE histories
from first-Compton-then-PE histories and is intended to supply component-wise
containment factors for PEGen and the detector-local scatter lookup. Build and
run instructions are in
`Geant4Code_GAGGIntrinsicResponse/README.md`.

本目录保存 Geant4 蒙卡模拟相关代码和输入生成脚本。它与 `GenProj/` 的职责不同：

- `Geant4Sim/`：生成 Geant4 macro、体模源、预览文件，并通过对应探测器的 C++ 工程做蒙卡模拟，输出 `CntStat` 和 `List`。
- `GenProj/`：不运行 Geant4，只用已有系统矩阵 `SysMat_polar` 做前投影，快速生成用于单光子验证的 `CntStat`。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `Geant4Code/` | JSCC/GAGG Geant4 工程，10496 detector bins，依赖 `CrystalMatrix.txt` |
| `Geant4Code_CntStatResponseStudy/` | JSCC 临时响应诊断工程；在同一 218+440 run 中按 primary 能量拆分 218/440 CntStat，并统计实际 primary 数 |
| `Geant4Code_EHE/` | EHE 平行孔 Pb/NaI 纯 CntStat Geant4 工程，1250 孔、2312 detector bins；不判定 Compton、不输出 List；附 Params 逐点核对工具 |
| `Macro/` | MATLAB 脚本生成的 Geant4 macro，每个旋转角通常一个 `.mac` |
| `Preview/` | MATLAB 生成的体模预览、体模 raw/mhd/mat 和 voxel list |
| `3D_DRO_Hoffman_v6_20160331_DICOM/` | Hoffman 脑模体原始 DICOM 输入 |
| `3D_DRO_Hoffman_v6_raw/` | Hoffman 原始 raw 数据 |

## MATLAB macro / 体模生成脚本

| 文件 | 用途 |
| --- | --- |
| `ContrastPhantom_Rotate_3D.m` | 单能量 contrast phantom 旋转 macro |
| `ContrastPhantom_DualEnergy_Rotate_3D.m` | 225Ac 双伽马代理源；Fr/Bi 分布错位，整个 run 的 218/440 期望发射数按分支比归一化 |
| `GenPhan_HotRodPhantom_Rotate_3D.m` | hot-rod phantom 旋转 macro |
| `Cylinder_Phantom_Rotate_3D.m` | 圆柱 phantom macro |
| `point_array_Rotate_3D.m` | 点源阵列 macro |
| `GenerateSensitivityPointArrayMacro.m` | 从 `coor_polar_full.csv` 生成等权极坐标点阵 sensitivity macro；默认是 440 keV、25600 点、单个 `beamOn` |
| `BrainPhantom_HoffmanMontage_3D.m` | Hoffman montage 脑模体生成 |
| `BrainPhantom_HoffmanRawCompressed_3D.m` | Hoffman raw 压缩体模生成 |
| `BrainPhantom_SliceStack_3D.m` | 基于切片堆栈的脑模体生成 |
| `BrainPhantom_Truncated_3D.m` | 截断脑模体生成 |
| `HoffmanCompressed_Rotate_3D.m` | Hoffman 压缩体模旋转 macro |
| `visualize_phantom.py` | 从 macro 解析源几何并生成 HTML 预览 |

### 440 keV `Sensi_d` 等权点阵 macro

运行：

```matlab
cd Geant4Sim
GenerateSensitivityPointArrayMacro
```

默认读取：

```text
../Factors/440keV_RotateNum20/coor_polar_full.csv
```

并生成：

```text
Macro/SensitivityPointArray_440keV_RotateNum20/
├── SensitivityPointArray_440keV_RotateNum20_View01.mac
└── generation_manifest.json
```

默认 macro 的定义为：

- 读取全部 25600 个极坐标位置，不做体积/Jacobian 加权；
- 所有 GPS source 的 intensity 都是 `1`；
- 第一个点沿用 GPS 默认 source，其默认 intensity 是 `1`；
- 第 2 至 25600 个点分别使用 `/gps/source/add 1`；
- `/gps/source/multiplevertex false`，每个 event 只从一个位置发射一个 primary；
- 每个 source 都显式设置 `/gps/number 1`，因此被选中时只发射一个 gamma；
- 所有 primary 都是 440 keV gamma，方向为 4-pi 各向同性；
- 每个局部坐标按现有约定平移到 Geant4 FOV 中心 `(0,-245,0) mm`；
- 默认视角是 `RotationIndex=1`，即 `phi=0`；
- 文件末尾只有一次 `/run/beamOn`；
- 默认每点期望 20000 个 primary，因此总数为 `25600*20000=512000000`。

GPS 的等权 source 选择保证每个点的**期望**发射数相同。单次有限统计运行中，每点
实际被选次数服从多项分布，不会被强制成完全相同的整数。若必须让每点实际次数
严格相同，需要修改 C++ primary generator 做循环/分层抽样，单靠一个 GPS mixture
和一个 `beamOn` 无法保证。

可在调用时调整每点期望光子数：

```matlab
GenerateSensitivityPointArrayMacro( ...
    'PrimaryEventsPerPoint', 40000, ...
    'OutputFileName', 'SensitivityPointArray_440keV_1p024e9.mac', ...
    'Overwrite', true)
```

标准 Geant4 `/run/beamOn` 使用有符号 32 位整数；脚本会拒绝超过 `2147483647` 的
总 event 数。25600 个点时，每点最多可设置 83886 个 event。更高统计量必须拆成
多个独立 run，最后合并 List，并把所有 run 的 `beamOn` 之和作为 sensitivity
程序的 `--source-photons`。

这个 macro 只改变 GPS 的发射位置和能量，不创建水、PMMA 或其他实体圆柱，材料
环境仍完全由 `Geant4Code/src/DetectorConstruction.cc` 决定。因此它保持了与当前
无物体衰减 `SysMat_polar` 相同的世界/探测器环境。

## 当前双能量 225Ac contrast phantom

`ContrastPhantom_DualEnergy_Rotate_3D.m` 当前设定：

- 背景圆柱直径：`240 mm`
- 热圆柱直径：`10:4:30 mm`
- 圆柱高度：`30 mm`
- 背景中心：Geant4 世界坐标 `(0, -245, 0) mm`
- 218 keV 分支比：`0.114`
- 440 keV 分支比：`0.261`
- 热圆柱能量：rod 1/3/5 为 218 keV，rod 2/4/6 为 440 keV
- 热柱活度：相对各自能量背景均为 `6` 倍
- 默认旋转数：`rotate_num = 20`
- 总初级光子数：`1e9`，即每视角 `5e7`
- 输出目录：`Macro/ContrastPhantom_DualEnergy_10_30_240_30_225Ac/`

运行该脚本会生成每个角度的 `.mac`，并调用 `visualize_phantom.py` 生成 `phantom_3d.html` 预览。预览只用于核对源几何和能量分布，不参与重建。

当前不直接使用 Geant4 放射性衰变模块生成 225Ac 离子，而是使用两个 gamma GPS
源族。Fr 与 Bi 的空间分布有意不同，用来模拟 alpha 衰变后子体离开螯合药物：

```text
q218(r) = 0.114 * x_Fr(r) / sum(x_Fr)
q440(r) = 0.261 * x_Bi(r) / sum(x_Bi)
```

因此，不论两张空间分布的积分是否相同，整个 run 中 GPS 抽样的期望份额固定为
218 keV 的 `30.4%` 和 440 keV 的 `69.6%`。GPS 每个 event 发射一个初级光子，
所以 `/run/beamOn` 表示选定两条 gamma 线的总光子数，不是 225Ac 母体衰变次数；
实际抽样计数会有有限统计涨落。

## 探测器矩阵与 10496 bins

`Geant4Code/CrystalMatrix.txt` 是
`CrystalMatrix_20250307_JSCCGC_32x64x4.mat` 的 `32×64×31` 完整展开。代码的
前 30 层只采用标签为 `1` 的 2304 个闪烁体，标签 `2` 表示钨块、不计入探测器
bin。文本矩阵的第 31 层不直接放置，而是由 C++ 中写死的 `128×64=8192` 个
`2.1 mm` 细分晶体替换，最终探测器数为：

```text
2304 + 8192 = 10496
```

启动时会检查矩阵必须有 `32×64×31=63488` 个标签、前 30 层必须有 2304 个
闪烁体，并在几何构造后确认实际放置数为 10496。该顺序与
`Factors/{218,440}keV_RotateNum20/Detector.csv` 的 10496 行一致。

## Geant4Code 输出

`Geant4Code/` 的 C++ 工程负责探测器响应和事件分类。当前代码中双能量单光子计数按能窗拆分输出：

| 输出文件 | 含义 |
| --- | --- |
| `CntStat_218.csv` | 落入 218 keV 能窗的单光子计数 |
| `CntStat_440.csv` | 落入 440 keV 能窗的单光子计数 |
| `List.csv` | 康普顿候选事件列表，列格式为 `[det1, energy1, det2, energy2, flag]` |
| `EnergySpectrum.csv` | 诊断输出，当前 `RunAction.cc` 中已注释 |
| `EventType.csv` | 诊断输出，当前 `RunAction.cc` 中已注释 |
| `Detector.csv` | 几何诊断输出，当前 `OutputDet=0`，默认不写 |

事件分类逻辑在 `Geant4Code/src/EventAction.cc` 中：能量展宽后遍历全部晶体，任一晶体落入 440 keV 或 218 keV 能窗都会给对应 detector bin 加 1；同一 event 可以增加多个 CntStat bin。完成能窗遍历后再独立执行康普顿判定，因此同一 event 可以同时贡献 CntStat 和 List。能量分辨率以 511 keV 处 13% 为参考，并按 `1/sqrt(E)` 标度到 218/440 keV。440 keV 光子散射后落入 218 能窗时会自然计入 `CntStat_218.csv`。`List` 按接收事件动态增长，不再预分配固定 1000 万行；输出固定为 5 列，不带尾随空列。

当前 GPS 圆柱只定义源位置，不会自动创建具有材料的实体模体。
`DetectorConstruction.cc` 中的水/PMMA Contrast Phantom 尚未启用，世界材料近似
真空，因此现阶段不包含物体内衰减与物体散射，只包含探测器和钨结构中的相互作用。

所有 CSV 都写入进程当前工作目录并使用追加模式。20 个视角可以在同一目录中
**顺序运行**以形成 20 行 CntStat；若并行运行，每个进程必须使用独立工作目录，
否则同名 CSV 会发生写入竞争。当前随机种子来自秒级时间戳，并行启动还需要显式
确保不同 seed；这是后续批处理脚本需要解决的事项。

## 与重建数据目录的关系

Geant4 运行通常在每个旋转角输出一行 `CntStat_*.csv`，以及对应角度的 `List.csv`。整理到主重建工程时，建议使用以下结构：

```text
CntStat/<energy>keV_RotateNum<N>/CntStat_<phantom>_<count>.csv
List/<energy>keV_RotateNum<N>/List_<phantom>_<count>/<angle>.csv
```

例如双能量 225Ac contrast phantom 可整理为：

```text
CntStat/218keV_RotateNum20/CntStat_ContrastPhantom_DualEnergy_10_30_240_30_225Ac_<count>.csv
CntStat/440keV_RotateNum20/CntStat_ContrastPhantom_DualEnergy_10_30_240_30_225Ac_<count>.csv
```

若只是验证单光子重建链路，可先使用 `GenProj/GenProj_ContrastPhantom_DualEnergy_PolarCoor.m` 通过系统矩阵前投影生成 `RotateNum20` 的基础 `CntStat`，再决定是否运行完整 Geant4。

### 已整理的 Geant4 双能 Contrast Phantom 数据

2026-07-16 已将 `ContrastPhantom_DualEnergy_Rotate_3D.m` 生成 macro 后在
`Geant4Code` 实际运行得到的 `1e9` 和 `1e10` 数据整理到独立命名空间，避免覆盖
GenProj 生成的同名数据：

```text
CntStat/218keV_RotateNum20_Geant4JSCC/
├── CntStat_ContrastPhantom_DualEnergy_10_30_240_30_225Ac_1e9.csv
└── CntStat_ContrastPhantom_DualEnergy_10_30_240_30_225Ac_1e10.csv

CntStat/440keV_RotateNum20_Geant4JSCC/
├── CntStat_ContrastPhantom_DualEnergy_10_30_240_30_225Ac_1e9.csv
└── CntStat_ContrastPhantom_DualEnergy_10_30_240_30_225Ac_1e10.csv

List/218-440keV_RotateNum20_Geant4JSCC/
├── List_ContrastPhantom_DualEnergy_10_30_240_30_225Ac_1e9/1.csv ... 20.csv
└── List_ContrastPhantom_DualEnergy_10_30_240_30_225Ac_1e10/1.csv ... 20.csv
```

四个 CntStat 均已核验为 `20 x 10496` 非负整数矩阵。事件总数为：

| 初级计数水平 | 218 能窗 CntStat | 440 能窗 CntStat | List 行数 |
| ---: | ---: | ---: | ---: |
| `1e9` | 4,649,413 | 1,965,176 | 1,984,776 |
| `1e10` | 46,478,635 | 19,641,521 | 19,856,413 |

**状态说明：这两组文件已由修正后的 Geant4 结果替换。** 当前 `EventAction`
遍历全部晶体，并让 CntStat 能窗判断与 Compton List 分类彼此独立；同一个 event
可以给多个 CntStat bin 计数，也可以同时写入 List。四个 CntStat 和 40 个 List
视角文件已通过尺寸、有限性、非负性、五列 schema、晶体编号和 flag 检查。它们
取代了此前的 legacy 数据；详细计数见
`Geant4Data_ContrastPhantom_DualEnergy_225Ac_manifest.json`。

每个 List 数据集有 20 个视角文件，固定 5 列，探测器编号范围为 `1..10496`，
flag 均为 1。必须注意：macro 的每个 event 从混合 218/440 GPS 源中选取一个
primary，但当前 List 不保存 primary energy。因此这些 List 只能标记为“双能混合
List”，不能复制到 `List/218keV_*` 或 `List/440keV_*` 后当作纯单能数据使用。

本地仅 CntStat 重建示例：

```powershell
C:\ProgramData\anaconda3\envs\pytorch\python.exe `
  main_local_multi_energy_cntstat.py `
  --count-levels 1e9 1e10 `
  --cntstat-dir-suffix _Geant4JSCC `
  --single-sc-iter 5000 --single-sc-save-step 50 `
  --output-root ./Results/Reconstruction/Figure_Local_SC_MultiOutput_Geant4JSCC
```

不要同时设置 `--factor-dir-suffix _Geant4JSCC`；不存在这样的 Factors，这批数据
与标准 `Factors/218keV_RotateNum20`、`Factors/440keV_RotateNum20` 和
`Factors/440keV_to218win_RotateNum20` 配套。机器可读清单位于
`Geant4Data_ContrastPhantom_DualEnergy_225Ac_manifest.json`。
