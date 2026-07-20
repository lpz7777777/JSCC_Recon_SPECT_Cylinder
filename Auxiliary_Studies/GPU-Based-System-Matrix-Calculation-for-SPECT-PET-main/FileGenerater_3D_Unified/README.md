# FileGenerater_3D_Unified

统一参数生成器：为 GPU 系统矩阵计算引擎（PEGen / ScatterGen）生成 `Params_*.dat` 输入文件。
支持两种几何、多能量批量生成、散射开关，产出的文件**新旧两版 CUDA 引擎都能直接消费**。

## 文件清单

| 文件 | 作用 |
|---|---|
| `generate_all.m` | **主入口**。读配置 → 遍历能量 → 调用各构建器 → 写入分目录 → 调用可视化 |
| `config_geometry.m` | **配置定义**。几何类型、能量列表、材料、散射开关等所有参数 |
| `material_db.m` | **材料系数库**。NaI/GAGG/Pb/W 在各能量的 [μ_total, μ_PE, μ_Compton] |
| `build_detector.m` | 探测器几何构建（JSCC 多层 + 传统平面），**修复了原版越界 bug 与第4层偏移 bug** |
| `build_collimator.m` | 准直器几何构建（随机孔板 + 三角晶格平行孔） |
| `write_dat_files.m` | float32 二进制写入 + Image/Physics 字段构造 |
| `plot_geometry_3d.m` | **3D 可视化入口**：导出几何数据 → 调用 Python 渲染交互式 HTML |
| `plot_geometry_3d.py` | **Plotly 交互式 3D 渲染**（由 .m 自动调用，读 .mat 输出 HTML） |

数据文件（已随目录提供）：
- `CrystalMatrix_20250307_JSCCGC_32x64x4.mat` — JSCC 探测器材料标签矩阵（32×64×31，值{0,1,2}）
- `randomPoints.mat` — JSCC 准直器随机孔中心（500 点）

## 用法

### 1. 编辑配置

打开 `config_geometry.m`，修改字段：

```matlab
cfg.geometry_type = 'JSCC';              % 'JSCC' 或 'ConventionalSPECT'
cfg.energy_list_keV = [140, 511];        % 批量生成的能量列表
cfg.detector_material = 'NaI';           % 探测器闪烁体材料
cfg.collimator_material = 'Pb';          % 准直器/屏蔽材料
cfg.enable_compton = false;              % 是否启用散射（新版引擎）
```

### 2. 运行

```matlab
>> cd FileGenerater_3D_Unified
>> generate_all
```

### 3. 输出

```
output/
├── JSCC_140keV/
│   ├── Params_Collimator.dat
│   ├── Params_Detector.dat
│   ├── Params_Image.dat
│   ├── Params_Physics.dat
│   └── geometry_3d_JSCC_140keV.html    ← 交互式 3D（浏览器打开）
├── JSCC_511keV/
│   └── geometry_3d_JSCC_511keV.html
├── ConventionalSPECT_140keV/
└── ...
```

每个子目录含完整 4 个 `.dat`，直接拷到 CUDA 引擎工作目录即可运行。
每个子目录还会自动生成一张 3D 几何示意图 PNG。

### 3D 可视化说明

`generate_all` 会在每个能量目录下生成 `geometry_3d_<几何>_<能量>keV.html` ——
**用浏览器打开即可交互式查看**（鼠标拖动旋转、滚轮缩放、悬停查看每个晶体/孔的坐标）。
渲染由 Python+Plotly 完成（WebGL，万级点流畅），由 MATLAB 自动调用，对用户透明。

#### 交互功能
- **左键拖动**：旋转视角
- **滚轮**：缩放
- **悬停晶体/孔**：显示坐标和尺寸（如"闪烁体 X=-132.3 Y=30.0 Z=-65.1 尺寸=3.0×3.0×3.0"）
- **图例点击**：切换某类元素的显示/隐藏
- **右上角工具栏**：导出 PNG、重置视角、框选放大等

#### 图元说明

| 元素 | 颜色 | 说明 |
|---|---|---|
| FOV（视野） | 🟢 绿色半透明立方体 | 体素网格的物理范围（中心在原点） |
| 准直器板 | ⬛ 深灰半透明立方体 | 平面板/平行孔准直器 |
| 准直器孔 | ⚪ 白色圆柱孔 | 穿透准直器板的圆孔（孔多时抽样显示约 200 个） |
| 闪烁体晶体 | 🟡 黄色**立方体**（不透明） | CrystalMatrix 标签 =1（NaI/GAGG），按真实尺寸 [X, Y_朝FOV, Z] |
| 屏蔽体 | 🔵 深蓝色**立方体**（不透明） | CrystalMatrix 标签 >1（W/Pb 屏蔽层，非晶体） |

晶体/屏蔽体为**真实尺寸的立方体**，完全不透明（前排挡住后排，物理真实）。
闪烁体与屏蔽体按 Y 深度（层）拆成独立图例项（如 `闪烁体 Y=200mm`），**点击图例可单独隐藏某层**以查看后排结构。

屏蔽体判定：凡 CrystalMatrix 标签 >1 的位置均视为屏蔽体（W/Pb），不计为闪烁晶体。
该标签存入 `Detector[id*12+12]`（flag 字段），可视化时直接读取，不依赖能量阈值。

坐标约定：FOV 中心在原点 (0,0,0)；X = 横向，Y = 深度（FOV→准直器→探测器，Y 增大），Z = 轴向，单位 mm。
探测器/准直器的 Y 在 `.dat` 里是局部坐标，可视化时加 `fov2collimator0` 平移到 FOV 原点坐标系（与 CUDA 引擎运行时一致）。

#### 依赖
Python 端需要（已装则自动使用）：
```
pip install plotly scipy numpy
```

若 Python 缺失或库未装，可视化会跳过（打印警告），不影响 `.dat` 文件生成。

若需单独绘制（不重新生成 .dat），可直接调用：
```matlab
>> plot_geometry_3d(cfg, det_params, col_params, energy_keV, '输出路径名')
```

## 支持的几何

### JSCC（多层 depth-of-interaction）
- 探测器：平面 4 层（Y=30/60/90/120mm），前 3 层 64×32 晶体（3mm 立方），
  第 4 层细分 128×64（2×6×2mm），共 11520 晶体
- 准直器：平面 500×500×3mm 板，500 个随机分布圆孔（半径 3mm）
- 材料由 `CrystalMatrix` 标签决定：1=闪烁体（NaI/GAGG），2=高 Z 屏蔽（Pb/W）

### ConventionalSPECT（传统平行孔）
- 探测器：34×68×1 NaI 平面阵列（4×4×10mm），共 2312 晶体
- 准直器：Siemens Symbia EHE，三角晶格 25×50=1250 孔（孔径 2.5mm，隔片 3.4mm，厚 50.5mm）

## 材料系数库

`material_db.m` 从 `physics_data/nist_xcom_materials_1_1000keV.csv` 读取衰减系数
（来源：NIST XCOM × 密度，单位 1/mm），并对非整数能量线性插值：

| 材料 | 密度 | 可用能量 |
|---|---|---|
| NaI | 3.67 g/cm³ | 1–1000 keV |
| GAGG | 6.60 g/cm³ | 1–1000 keV |
| Pb | 11.35 g/cm³ | 1–1000 keV |
| W | 19.35 g/cm³ | 1–1000 keV |
| Vacuum | — | 任意（μ=0，等效无准直器） |

约定：`μ_total（不含瑞利/相干散射）= μ_PE + μ_Compton`

### 数据来源与验证

- 光电吸收与非相干（Compton）质量相互作用系数由
  `physics_data/download_nist_xcom.py` 直接从 NIST XCOM 官方 CGI 下载。
- CSV 同时保存原始 `cm²/g` 数据和按密度换算后的 `1/mm` 数据。
- CUDA 使用同一数据生成的静态头文件，对散射后光子能量插值；不再使用
  `μ_PE∝E^-3`、`μ_Compton∝E^-1` 的近似。
- 数据再生方法和材料化学式见 `physics_data/README.md`。

## 散射配置（新版引擎）

`config_geometry.m` 的散射开关控制 `Params_Physics.dat`：

| 字段 | Physics 索引 | 含义 |
|---|---|---|
| `enable_compton` | [0] | Compton 散射总开关 |
| `save_compton_only` | [2] | 仅保存散射系统矩阵 |
| `save_combined_sysmat` | [3] | 保存 PE+Compton 合并矩阵 |
| `use_same_energy_window` | [4] | 用统一能量窗（否则按探测器分辨率推） |
| `energy_window_lower/upper_keV` | [5][6] | 能量窗阈值 |
| `compute_geo_relationship` | [8][9] | 首次跑散射时让引擎计算几何关系位图 |
| `enable_detector_recoil_escape` | [10] | A 中首次 Compton 后光子逃逸时，按 `E0-E'` 记录 A 的反冲脉冲并做能窗卷积 |
| `enable_self_scatter_photopeak` | [11] | A 中一次 Compton 后在 A 内 PE，按总沉积 `E0` 记录全能峰并做能窗卷积 |

启用散射的典型配置：
```matlab
cfg.enable_compton = true;
cfg.save_combined_sysmat = true;
cfg.use_same_energy_window = true;
cfg.energy_window_lower_keV = 126;   % 140keV ± 10%
cfg.energy_window_upper_keV = 154;
cfg.enable_detector_recoil_escape = true;
cfg.enable_self_scatter_photopeak = true;
```

`Params_Physics.dat` 现在固定写出 12 个 `float32`（48 bytes）。专用
218/440 生成器对 218、440 直接光峰场景设置 `[10]=1,[11]=1`，对
`440->218` 交叉能窗设置 `[10]=1,[11]=0`。后者保留 A 中反冲沉积进入
218 窗的概率，同时跳过与该交叉项无关的 A 内全能峰计算。
两个局部开关都受总开关 `[0]` 控制；`enable_compton=false` 时不会计算
任何局部 Compton 响应。

首次 Compton 后的 A 内分支使用 XCOM 在散射能量 `E'` 的插值系数：

```text
P_escape + P_second_PE + P_second_Compton = 1
P_escape = exp[-(mu_PE(E') + mu_Compton(E')) * L_A]
```

`enable_detector_recoil_escape` 对所有逃逸方向只给 A 累加一次，不随目标
B 的枚举重复。`enable_self_scatter_photopeak` 对一次 Compton 后紧接 PE 的
分支，在同一晶体内将两次沉积合并为 `E0` 后只展宽一次。二次 Compton
分支目前只进入概率守恒检查，不继续输运。首次相互作用位置沿用现有的
晶体中心近似。

JSCC 和 EHE 的每个探测器记录均作为独立读出通道；A->B 事件可以同时给
A、B 两行贡献期望计数。EHE 连续 NaI 晶体的光共享、Anger 质心定位和
事件级能量求和不在当前解析像素模型范围内。

### 能量分辨率（基准值 + 自动标度）

能量分辨率用一个基准值指定，`generate_all` 会按 R ∝ 1/√E 律为每个能量自动标度：

```matlab
cfg.energy_resolution_ref = 0.13;           % 基准 FWHM 分辨率（如 511keV 处 13%）
cfg.energy_resolution_ref_keV = 511;        % 基准能量
% generate_all 内：res(E) = ref × √(ref_keV / E)
%   218keV -> 0.199, 440keV -> 0.140, 511keV -> 0.130
```

标度后的值填入 `Detector[id*12+10]`。散射引擎读取此值作为 E₀ 处的标称相对 FWHM 分辨率，内部按闪烁探测器统计模型 `R(E') = R(E₀)·√(E₀/E')` 进一步外推到每个散射角的退降能量 E'，因此低能散射光子的相对能量分辨率更差。

## JSCC 218/440 同时成像参数

为 218/440 keV 同时成像新增了专用入口：

```matlab
>> cd FileGenerater_3D_Unified
>> generate_jscc_218_440_response_params
```

该脚本会生成并同步三套小参数文件，不复制或覆盖大矩阵：

| 目录 | 含义 | 能窗 | 矩阵用途 |
|---|---|---|---|
| `output/JSCC_218keV` 与 `../runs/JSCC_218keV` | 218 keV 发射 | 自动 218 光峰窗 `[196.305380, 239.694620] keV` | `A_218win<-218` |
| `output/JSCC_440keV` 与 `../runs/JSCC_440keV` | 440 keV 发射 | 自动 440 光峰窗 `[409.178757, 470.821243] keV` | `A_440win<-440` |
| `output/JSCC_440keV_to_218keVwin` 与 `../runs/JSCC_440keV_to_218keVwin` | 440 keV 发射 | 强制 218 能窗 `[196.305380, 239.694620] keV` | `A_218win<-440` 串扰项 |

三套参数已按当前 JSCC 配置生成：

```text
JSCC_218keV:
  Physics = [1, 1, 1, 1, 0, 0, 0, 218, 1, 1, 1, 1]
  Detector[id*12+10] = 0.199033216

JSCC_440keV:
  Physics = [1, 1, 1, 1, 0, 0, 0, 440, 1, 1, 1, 1]
  Detector[id*12+10] = 0.140096560

JSCC_440keV_to_218keVwin:
  Physics = [1, 1, 1, 0, 1, 196.30538, 239.69462, 440, 1, 1, 1, 0]
  Detector[id*12+10] = 0.140096560
```

`JSCC_440keV_to_218keVwin` 是交叉响应参数。运行时应复用/指定未加窗的
440 keV `PE_SysMat_*.sysmat` 作为 ScatterGen 的首次相互作用输入。当前交叉参数
关闭合并输出，因此仍取 `Scatter_SysMat_*.sysmat` 作为 `A_218win<-440`。
PEGen 同时写出 `PE_Windowed_SysMat_*.sysmat`；同能量合并矩阵使用落窗 PE，
不再把全部直接光电事件无条件加入能窗。

### 225Ac 分支比与交叉项权重

是的，把 440 keV 对 218 能窗的串扰贡献加入 218 能窗计数时，应按 225Ac 衰变链中两条 gamma 线的发射概率加权。系统矩阵本身按“每个给定能量的发射光子”归一化，不包含核素分支比；分支比应在生成投影均值、CntStat 或重建 forward model 时乘进去。

对 225Ac 同时成像，建议写成：

```text
Mean_218win = N_decay * (Y218 * A_218win<-218 * x_Fr
                       + Y440 * A_218win<-440 * x_Bi)

Mean_440win = N_decay * (Y440 * A_440win<-440 * x_Bi)
```

其中 `x_Fr` 是 221Fr/218 keV 对应空间分布，`x_Bi` 是 213Bi/440 keV 对应空间分布。常用核数据近似为：

```text
Y218 = 0.114   % 221Fr, 218 keV
Y440 = 0.259~0.261   % 213Bi, 440 keV
```

本项目此前 Geant4 macro 约定使用 `Y218 = 0.114`、`Y440 = 0.261`，对应：

```text
Y440 / Y218 = 0.261 / 0.114 = 2.28947
```

因此，如果你的 218 直接项以 `Y218` 为 1 做相对归一化，那么 440 串扰项应乘以 `Y440/Y218` 后再和 218 直接项相加。也就是说，按 218 光子产额归一化的 225Ac 等效 218-window 系统矩阵应定义为：

```text
A_225Ac_218win_norm218 =
    A_218win<-218 + (Y440/Y218) * A_218win<-440
```

这正对应“用真实 225Ac 分支比在同一个 run 里发出 218 和 440 keV 光子”的点源蒙卡/实测系统矩阵：218 能窗计数天然包含 440 光子落入 218 能窗的串扰；如果最后按发出的 218 keV 光子数或 218 产额归一化，440 串扰项就会表现为 `Y440/Y218` 倍。

等价地，也可以在绝对计数模型里分别乘 `Y218` 和 `Y440`。

注意两点：

1. 如果 Geant4 或真实实验的 `CntStat` 已经来自按分支比混合发射的事件流，则读入实测/蒙卡 `CntStat` 时不要再乘一次 `Y440/Y218`，否则会重复加权。这个提醒针对“已经生成好的计数数据”，不是针对构造 `A_225Ac_218win_norm218`。
2. 如果 218 和 440 的空间分布不同，不应把 `A_218win<-440` 简单加到 `A_218win<-218` 后乘同一个图像；应保留 `A_218win<-218*x_Fr + A_218win<-440*x_Bi` 的双源项。只有在 225Ac 点源系统矩阵、局部平衡共定位源，或假设 `x_Fr = x_Bi` 的单图像近似下，才可以构造 `A_225Ac_218win_norm218 = A_218win<-218 + (Y440/Y218)*A_218win<-440`。

Fr/Bi 空间分布不同时的后续双图耦合重建方案见 [`distributed/FRBI_COUPLED_RECON_DESIGN.md`](../../../distributed/FRBI_COUPLED_RECON_DESIGN.md)。该文档记录了 218 单光子、440 单光子、440 康普顿/List 联合使用时的 forward model、OSEM 更新式、开发接口和验证用例。

由于散射光子能量分辨率、A 出射衰减、两个 A 局部响应、多角度 PE 切片索引以及跨晶体目标表面积分均已修复，已有的散射矩阵都需要用新编译的 ScatterGen 重新生成。旧版跨晶体核会把目标中心的散射能量接受率用于整个外接球宽角区间，在 JSCC `2×6×2 mm` 末层近邻晶体中产生明显高估。只重新生成 `Params_*.dat` 不会改变已有 `.sysmat` 文件。

后续矩阵生成建议顺序：

1. 重新编译 `ScatterGen_RayTracing_CircularHole`，并确保 `runs/JSCC_218keV`、`runs/JSCC_440keV`、`runs/JSCC_440keV_to_218keVwin` 使用的是新编译的 `ScatterGen_CircularHole`。
2. 对 `runs/JSCC_218keV` 重跑 Scatter，得到新的 `A_218win<-218`；若当前 `PE_SysMat` 已通过校验，可以直接复用，不需要因为本次 ScatterGen 修改而重算 PE。
3. 对 `runs/JSCC_440keV` 重跑 Scatter，得到新的 `A_440win<-440`；同样可以复用已有 440 keV PE。
4. 对 `runs/JSCC_440keV_to_218keVwin` 只需要重跑 ScatterGen，并把 440 keV 的 PE 矩阵作为输入，例如：

```bash
cd runs/JSCC_440keV_to_218keVwin
./ScatterGen_CircularHole \
  -PE ../JSCC_440keV/PE_SysMat_shift_0.000000_0.000000_0.000000_v3.sysmat \
  -cuda <GPU_ID>
```

该目录输出的 `Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat` 即为 `A_218win<-440`。后续转 Factors 时建议使用独立目录名，例如 `Factors/440keV_to218win_RotateNum20`，避免和标准 440 光峰窗矩阵混淆。

正式生成默认使用远场 `1×1`、近场 `8×8` 可见目标面积分。第一次验证建议为交叉矩阵额外设置 `SCATTER_WRITE_COMPONENTS=1`，得到 `C_intercrystal`、`C_highZ_to_crystal`、`C_local_recoil` 等分量；确认后正式大规模运行应关闭该开关以节省一张 GPU 矩阵和一张 pinned-host 矩阵的额外内存。近场收敛检查可把 `SCATTER_NEAR_TARGET_FACE_SUBDIV` 从默认 8 提高到 16，对少量 scatter-crystal 范围试跑后比较。

### Vacuum 准直器

准直器材料设为 `Vacuum` 时（`cfg.collimator_material = 'Vacuum'`），μ_total=μ_PE=μ_Compton=0，等效于无准直器（光子自由穿过）。散射引擎已做除零防护，Vacuum 准直器的散射贡献正确归零，不产生 NaN。

## 已知问题

### EHE parallel-hole parameter validation

Run the EHE generator to create all three 218/440 response parameter sets and
validate the serialized binary files:

```matlab
generate_ehe_pb_nai_218_440_response_params
```

The generator automatically calls `validate_ehe_parallel_hole_params`. It
checks file lengths and record layouts, the 2312-crystal NaI detector, all 1250
holes, triangular-lattice nearest-neighbor distances, plate boundaries,
detector/collimator contact, and consistency among the 218, 440, and
440-to-218-window cases. The batch-safe numerical report is written to:

```text
output/EHE_validation/EHE_parallel_hole_validation.txt
```

Optional PNG/PDF diagnostics can be requested from an interactive MATLAB
session with `validate_ehe_parallel_hole_params([], true)`. They are disabled
by default because headless graphics backends can block batch generation.

In this model, `hole_diameter = 2.5 mm` and `septal_thickness = 3.4 mm` mean
an edge-to-edge lead septum of 3.4 mm, giving a hole-center pitch of 5.9 mm.
The Pb attenuation coefficients are stored in the collimator-layer header.
Each aperture record stores zero attenuation coefficients because the hole
interior is air/vacuum; assigning Pb coefficients to the aperture would make
the plate behave like solid lead and can reduce the 218-keV PE matrix to zero.
The CUDA kernels add `fov2collimator0` to local collimator and detector Y
coordinates. JSCC keeps its established local-Y origin at 170 mm, placing the
front face of its first 3-mm detector layer at `170 + 30 - 1.5 = 198.5 mm`.
For EHE, local Y=0 is the center of the 50.5-mm collimator, so the generator
uses `fov2collimator0 = 198.5 + 50.5/2 = 223.75 mm`. The EHE collimator then
occupies global Y = 198.5..249.0 mm and its front face exactly matches the JSCC
first-layer detector front face. With the current image grid ending at
Y = 153 mm, the resulting FOV-to-collimator clearance is 45.5 mm.

### Params_Collimator.dat 截断（2026-07-09 发现）

重新生成 218/440 keV 参数时，`write_dat_files.m` 输出的 `Params_Collimator.dat` 仅包含 **100 floats**（400 bytes），
而预期应为 4600 floats（含 500 个孔记录 × 9 字段 + header）。实际文件只含 header（前 4 字段），
孔数据（`col[10]`=numHoles 读到 0）完全缺失。

**影响**：当 `collimator_material = 'Vacuum'`（μ=0）时，CUDA 引擎的 `col[10]=0` 导致孔循环直接跳过，
等效于无准直器——光子自由穿过，与 Vacuum 物理一致，**不影响 PE/Scatter 矩阵值**。
但如果后续切换到 Pb/W 等真实准直器材料，此 bug 会导致所有孔信息丢失，矩阵完全错误。

**修复方向**：检查 `write_dat_files.m` 中准直器参数写入逻辑，确认 `col[10]`（numHoles）
是否被正确写入，以及孔记录的循环写入是否正常执行。

### Detector 缓冲区溢出（CUDA 引擎，已修复）

`PEGen_CircularHole.cpp` 和 `scatter.cu` 原先将探测器缓冲区硬编码为 `80000 floats`，
但实际需要 `138241 floats`（11520 × 12 字段 + 1 header）。
`80000 / 12 = 6666`，探测器 6667 及之后的所有行越界读显存垃圾。

**表现**：PE 矩阵第 4 层（y=120mm）+x 半区 4853 个晶体整行归零，
截断点精确落在 x = -24.15 mm（det index 6666 对应位置）。
前 3 层和层 4 左半区不受影响（idx < 6667，在界内）。

**修复**：`PEGen_CircularHole.cpp:29,40`、`PESysMatGen.cu:729,730`、
`scatter.cu:2217,2218,2250,2251`、`ScatterGen_CircularHole.cpp:41,52` 中
所有 `80000` → `200000`。修复后 PE 矩阵 0 零行（验证通过）。

## 与原代码的关系

本生成器整合并修复了两个原始 FileGenerater：

| 原始位置 | 本生成器对应 |
|---|---|
| `GenerateSysMat_2025_Cuda/FileGenerater_3D/`（JSCC，有越界 bug） | `geometry_type='JSCC'` 路径 |
| `GenerateSysMat_2025_Cuda/FileGenerater_3D_ConventionalSPECT/` | `geometry_type='ConventionalSPECT'` 路径 |

**修复点**：
1. `generateDet_3D.m:68` 的 `pos1 = zeros(unit_num_x, unit_num_x, ...)` 越界 bug（第二维应为 `unit_num_y`）
2. JSCC 准直器衰减系数缺失（原版 `collimator_params(16..18)=0`）→ 现从材料库填入
3. 探测器能量分辨率字段缺失（原版 `Detector[id*12+10]=0`）→ 现填入 `cfg.energy_resolution`（散射引擎必需）
4. 第 4 层晶体位置偏移 bug（原版 `generateDet_3D.m:93-95` 索引交叉错误，X/Z 未居中）→ 已修正居中
5. 晶体尺寸朝向错误（原版轴置换混乱，第 4 层 6mm 错放 X 方向）→ 按 CUDA 约定 `det[+4]=wDet(X)`, `det[+5]=tDet(Y朝FOV)`, `det[+6]=hDet(Z)` 正确赋值
