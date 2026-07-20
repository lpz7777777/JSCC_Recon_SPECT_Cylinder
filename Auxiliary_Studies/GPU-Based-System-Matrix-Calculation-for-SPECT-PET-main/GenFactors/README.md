# GenFactors

## Current production convention

All newly generated polar Factors use an activity-density basis by default:

```text
B = A * diag(DeltaV_mm3)
y = B * rho
rho unit = emitted photons / mm3
```

`gen_factors.m` computes radial/axial midpoint cell boundaries and equal
angular sectors per ring, writes `polar_cell_volume_mm3.csv/.float64`, and
multiplies every polar matrix column by its full cell volume before writing
`SysMat_polar`. Center-inclusive sampling is the default. Set
`apply_polar_volume_weighting=false` only for an explicit legacy study.

The long-term JSCC production entry point is:

```matlab
addpath("Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main/GenFactors")
results = run_gen_jscc_production_factors;
```

It reads the three `_pe_v4` V4-S runs, includes the center sample, applies the
validated combined center/Uniform-FOV layer calibration, omits the disposable
Cartesian `SysMat_tmp`, and atomically installs the no-suffix standard Factors:

```text
Factors/218keV_RotateNum20
Factors/440keV_RotateNum20
Factors/440keV_to218win_RotateNum20
```

The generic `run_gen_response_factors` also defaults to volume weighting and a
center point, but defaults to `calibration_profile='none'`. Calibration must be
selected explicitly so a historical profile cannot silently contaminate a new
transport-model comparison.

把 GPU 引擎生成的 `.sysmat`（笛卡尔系统矩阵）转换成重建代码使用的 `Factors/` 目录格式（极坐标系统矩阵 + 旋转矩阵 + 坐标表 + 探测器表）。

## 文件清单

| 文件 | 作用 |
|---|---|
| `gen_factors.m` | 核心转换函数：读 .sysmat → 闪烁体过滤 → 笛卡尔→极坐标 → 生成全部 Factors 文件 |
| `run_gen_factors.m` | 驱动脚本：为 218/440keV 的合并矩阵各跑一次 |
| `run_gen_response_factors.m` | 为 JSCC/EHE 批量生成 A218、A440、C440→218 六套 Factors，并写响应 manifest |

## 流程

```
runs/<E>keV/SysMat_withScatter_*.sysmat  (11520 晶体 × 52020 体素, 笛卡尔)
        │
        ▼  gen_factors.m
  1. reshape 成 [51,51,20,11520]
  2. 清洗残余 NaN/Inf → 0（保险）
  3. 过滤闪烁体（Params_Detector flag==1）→ 10496 晶体
  4. 存 SysMat_tmp（过滤后的笛卡尔矩阵）
  5. 笛卡尔→极坐标 interp2（1280 点/层 × 20 层）
  6. 生成 coor_polar / RotMat / RotMatInv / Detector.csv
        │
        ▼
Factors/<E>keV_RotateNum20/
```

## 极坐标配置

与现有 `511keV_RotateNum20` Factors 完全一致：
- 半径 `r = 6:6:150`（25 个）
- 角度数随半径分段：r∈[6,30]→20，[42,72]→40，[78,108]→60，[114,150]→80
- 每层 1280 个极坐标点，20 个 z 层 → **25600 个极坐标体素**
- 旋转数 20

## 用法

从主工程根目录批量生成 JSCC 与 EHE 的六套双能响应，并直接安装到主工程
`Factors/`：

```matlab
addpath("Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main/GenFactors")
results = run_gen_response_factors;
```

To export side-by-side PE v4 runs without replacing the standard run inputs:

```matlab
grid = struct( ...
    'include_center_point', true, ...
    'run_name_suffix', '_pe_v4', ...
    'calibration_profile', 'center_point_20260716');
results = run_gen_response_factors( ...
    ["JSCC/A218", "JSCC/A440", "JSCC/C440to218"], ...
    "CenterPoint_PEv4", grid);
```

If a selected run contains `PE_v4_manifest.json`, its model and quadrature
metadata are embedded in the generated `factor_manifest.json`. Use
`calibration_profile='none'` only for an explicitly uncalibrated study.

批处理在修改任何 Factors 前会预检所有输入 `.sysmat` 和
`Params_Detector.dat`。每套响应先写入 `.build_*` 临时目录，维度、字节数、
探测器数、极坐标点数和旋转映射通过验证后才原子替换目标目录。每个目录都包含
`factor_manifest.json`，明确记录 `A218`、`A440` 或 `C440to218`、源能量、
观测能窗、输入矩阵和“矩阵不含 225Ac gamma 产额”的归一化约定。

若只有某一套输入需要补生成，可用系统/响应选择器，避免重跑其他大矩阵：

```matlab
results = run_gen_response_factors("SPECTEHENaI/C440to218");
```

可用选择器为 `JSCC/A218`、`JSCC/A440`、`JSCC/C440to218`、
`SPECTEHENaI/A218`、`SPECTEHENaI/A440`、`SPECTEHENaI/C440to218`；也可直接
传目标 Factors 目录名。

其中交叉响应必须读取 forced 218-window run 的
`Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat`。不要使用该 run 的
`SysMat_withScatter`，因为直接 440 keV PE 响应不受该 forced 低能窗约束。

如果 EHE 交叉 run 只有 `ScatterGen.log/.pid`、没有 `.sysmat`，说明 ScatterGen
并未完成。Linux 服务器上可从引擎根目录用 `nohup` 补跑，避免 SSH 会话关闭时杀掉
长时间的准直器散射 kernel：

```bash
cd runs/EHE_PbNaI_440keV_to_218keVwin
nohup ../../ScatterGen_RayTracing_CircularHole/ScatterGen_CircularHole_optimized \
  -PE ../EHE_PbNaI_440keV/PE_SysMat_shift_0.000000_0.000000_0.000000_v3.sysmat \
  -cuda 0 > ScatterGen.log 2>&1 &
echo $! > ScatterGen.pid
```

完成标志是日志末尾出现 `Compton Scatter Sysmat written.`，并生成大小为
`481080960` bytes 的 `Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat`。
将该文件同步回本工程后，只需运行：

```matlab
run_gen_response_factors("SPECTEHENaI/C440to218")
```

历史的两能量直接响应入口仍可使用：

```matlab
>> cd GenFactors
>> run_gen_factors    % 自动处理 218 + 440 keV
```

或单独调用：
```matlab
>> gen_factors(218, 'runs/JSCC_218keV/SysMat_withScatter_shift_0.000000_0.000000_0.000000.sysmat', ...
                   'FileGenerater_3D_Unified/output/JSCC_218keV/Params_Detector.dat', ...
                   'Factors/218keV_RotateNum20', '')
```

## 输出文件

每个 `Factors/<E>keV_RotateNum20/` 目录含：

| 文件 | 内容 |
|---|---|
| `SysMat_tmp` | 闪烁体过滤后的笛卡尔矩阵 [51,51,20,10496] |
| `SysMat_polar` | 极坐标系统矩阵 [10496, 1280, 20] |
| `Detector.csv` | 闪烁体探测器表（index, x, y, z），10496 行 |
| `coor_polar.csv` / `.mat` | 单层极坐标采样点（1280×2） |
| `coor_polar_full.csv` / `.mat` | 全部极坐标采样点（25600×3，含 z） |
| `RotMat.csv` / `.mat` | 旋转映射（1280×20） |
| `RotMatInv.csv` / `.mat` | 逆旋转映射（1280×20） |
| `RotMat_full.csv` / `.mat` | 全层旋转映射（25600×20） |
| `RotMatInv_full.csv` / `.mat` | 全层逆旋转映射（25600×20） |

## 闪烁体过滤

过滤逻辑融合了原 `generateDet_3D.m` 的代码，但改用 **`Params_Detector` 的 flag 字段**（第 12 列）判定闪烁体，而非原始 CrystalMatrix 矩阵：
- `flag == 1`：闪烁体（NaI/GAGG）→ 保留
- `flag > 1`：高 Z 屏蔽体（W/Pb）→ 过滤掉

这样能正确对应 `.sysmat` 第 4 维（探测器）的实际排列顺序（含第 4 层细分晶体）。过滤后 11520 → 10496 晶体，与现有 511keV Factors 一致。

## NaN 处理

`gen_factors` 在 reshape 后会检查并清洗残余 NaN/Inf（置 0）。新版散射引擎（含 Vacuum 防护 + 几何退化修复）生成的矩阵已验证 **NaN = 0%**，此清洗仅作保险。

## 已修复问题记录（2026-07-09）

### 1. Scatter 矩阵 NaN 污染（99.9%）

**根因**：`scatter.cu:1830`，Vacuum 准直器（μ=0）时 `solid_angle / Collimator[15] * Collimator[17]`
计算为 `inf × 0 = NaN`，通过散射路径传播到几乎所有非零元素。

**修复**：在 `scatter.cu:1830` 加 μ_total 守卫，μ_total ≤ 0 时 early return（散射贡献归零）。

### 2. calculateConeAngle 零向量 NaN

**根因**：`calculateConeAngle` 函数在源-接收器重合或几何退化时产生零模向量，
`dot / 0` 生成 NaN cosTheta → NaN cone angle → NaN 散射贡献。

**修复**：在 `scatter.cu` 的 `calculateConeAngle` 中加模长守卫（< 1e-9f 时返回 0.0f），
并在两处 atomicAdd 前加 `isfinite(contrib) && contrib > 0.0f` 守卫。

### 3. PE 矩阵层 4 零行（4853 个晶体）

**根因**：`PEGen_CircularHole.cpp` 和 `scatter.cu` 将探测器缓冲区硬编码为 `80000 floats`，
但实际需要 `138241 floats`（11520 × 12 + 1）。`80000/12 = 6666`，
det 6667-11519 越界读显存垃圾 → `coeffTot` 为负 → kernel 早退归零。

**表现**：PE 矩阵第 4 层（y=120）+x 半区 4853 个晶体整行归零，
截断点精确在 x = -24.15 mm（det 6666 对应位置，`6666-3328=3338, /64=52, ×2.1-133.35=-24.15`）。

**修复**：全部 `80000` → `200000`（涉及 4 个文件共 10 处）。修复后 PE 0 零行。

### 4. Params_Collimator.dat 截断

**根因**：`write_dat_files.m` 重新生成 218/440 keV 参数时，准直器文件仅含 100 floats（无孔记录），
`col[10]`=numHoles 读到 0。Vacuum μ=0 时不影响矩阵值，但切换到 Pb/W 后会完全错误。

**状态**：待修复（`FileGenerater_3D_Unified` 的 `write_dat_files.m`）。

## 将 Factors 接入重建

`run_gen_response_factors` 已直接写入主工程 `Factors/`，无需再次复制。JSCC
串扰重建需要 `218keV_RotateNum20`、`440keV_RotateNum20` 与
`440keV_to218win_RotateNum20`；EHE 使用相同名字并追加 `_SPECTEHENaI`。
