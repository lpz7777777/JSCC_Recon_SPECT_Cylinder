# GenProj

> Polar-grid source convention: see `POLAR_SOURCE_MEASURE.md`. A constant
> value at every polar sample is not a uniform physical activity density.

Density-basis tools:

- `build_polar_volume_weighted_factors.py` creates `A*diag(DeltaV_mm3)`
  Factors while retaining the source Factors.
- `analyze_polar_source_measure.py` audits grid volumes and projection closure.
- `compare_polar_volume_reconstruction.py` compares old integrated-cell and
  new density-basis Geant4 reconstructions with common background scaling.

Routine Factors generation now belongs to the matrix project's MATLAB entry
`GenFactors/run_gen_jscc_production_factors.m`. The Python volume builder is
retained only to reproduce/audit the one-time migration from the old
integrated-cell Factors.

本目录保存基于系统矩阵的 MATLAB 前投影脚本，用于从已生成的 `Factors/` 生成单光子投影数据 `CntStat/`。这些脚本不运行 Geant4，也不生成康普顿 `List/`；它们适合做系统矩阵、旋转矩阵、噪声模型和单光子重建链路的快速验证。

## 脚本

| 文件 | 用途 | 主要输出 |
| --- | --- | --- |
| `GenProj_SPECT_PolarCoor.m` | 通用 contrast/hot-rod/point-array 单光子前投影脚本，保留了多个历史体模配置块 | `CntStat/CntStat.csv` |
| `GenProj_Hoffman_SPECT_PolarCoor.m` | Hoffman 压缩脑模体的单光子前投影 | `CntStat/<energy>keV_RotateNum<rotate>/CntStat_HoffmanCompressed_<count>.csv` |
| `GenProj_ContrastPhantom_DualEnergy_PolarCoor.m` | 与 `Geant4Sim/ContrastPhantom_DualEnergy_Rotate_3D.m` 对应的 225Ac 双能量 contrast phantom 前投影，显式加入 440→218 能窗串扰 | `CntStat/218keV_RotateNum20/...csv` 和 `CntStat/440keV_RotateNum20/...csv` |

## 运行方式

在 MATLAB 中从仓库根目录运行：

```matlab
run("GenProj/GenProj_ContrastPhantom_DualEnergy_PolarCoor.m")
```

或先从仓库根目录加入路径：

```matlab
addpath("GenProj")
GenProj_ContrastPhantom_DualEnergy_PolarCoor
```

脚本内部会根据自身位置向上一层定位仓库根目录，因此移动到 `GenProj/` 后仍会读写根目录下的 `Factors/`、`CntStat/`、`Geant4Sim/` 等路径。

## 数据约定

- 系统矩阵目录遵循 `Factors/<energy>keV_RotateNum<rotate>/`。
- `SysMat_polar` 为 float32 二进制文件，按当前重建代码约定读成 `detector_num x pixel_num`。
- 旋转矩阵使用 `RotMat.mat` 或 `RotMat_full.mat`，具体取决于脚本和系统矩阵版本。
- 输出的 `CntStat` 是 `rotate_num x detector_num` CSV，可直接供 `main_local_cntstat.py` 或分布式 SC-only 入口读取。

## 双能量 225Ac 脚本说明

`GenProj_ContrastPhantom_DualEnergy_PolarCoor.m` 默认使用：

- `cfg.energy_keV = [218, 440]`
- `cfg.yield = [0.114, 0.261]`
- `cfg.rod_energy_keV = [218, 440, 218, 440, 218, 440]`
- `cfg.rotate_num = 20`
- `cfg.noise_model = "poisson"`

脚本读取三套彼此独立、均按“每个对应能量的发射光子”归一化的响应：

```text
Factors/218keV_RotateNum20/                 A218
Factors/440keV_RotateNum20/                 A440
Factors/440keV_to218win_RotateNum20/        C440to218
```

前投影模型为：

```text
y218 = A218*x218 + C440to218*x440
y440 = A440*x440
```

`x218`、`x440` 的空间分布先分别归一化，再各自乘 `Y218=0.114`、
`Y440=0.261`；因此分支比只进入一次。响应矩阵和后续重建不能再次乘
`Y440/Y218`。218 直接项和 440→218 串扰项分别采样 Poisson 噪声后相加，
在统计上等价于对总均值采样，同时保留了可核验的贡献分解。

218 目录除标准观测文件外，还写出：

```text
CntStatDirect_<dataset>_<count>.csv
CntStatMeanDirect_<dataset>_<count>.csv
CntStatCrossTalk_<dataset>_<count>.csv
CntStatMeanCrossTalk_<dataset>_<count>.csv
```

并在 `CntStat/ProjectionManifest_<dataset>.json` 记录响应目录、gamma 产额、
计数汇总与 218 能窗串扰比例。对 EHE Pb/NaI Factors，设置
`cfg.factor_dir_suffix = "_SPECTEHENaI"`；输出 CntStat 目录会使用相同后缀。

非交互批处理无需编辑 MATLAB 文件，可以在启动 MATLAB 前设置环境变量：

```powershell
$env:JSCC_RECON_FACTOR_DIR_SUFFIX = '_SPECTEHENaI'
$env:JSCC_RECON_CNTSTAT_DIR_SUFFIX = '_SPECTEHENaI'
matlab -batch "run(fullfile(pwd,'GenProj','GenProj_ContrastPhantom_DualEnergy_PolarCoor.m'))"
```

环境变量未设置时保持原有 JSCC 默认值；CntStat 后缀未单独设置时自动跟随
Factors 后缀。

它优先读取主工程 `Factors/`，如果主工程没有 218/440 keV 因子，会自动回退到：

```text
Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main/Factors/
```

如果后续生成了 `RotateNum60` 的三套响应，只需要把脚本中的 `cfg.rotate_num`
改为 `60`，并保证直接响应和交叉响应目录都存在。

## 与 Geant4 数据的命名隔离

GenProj 默认输出仍位于无后缀的 `CntStat/218keV_RotateNum20` 和
`CntStat/440keV_RotateNum20`。Geant4 JSCC 实际模拟数据使用独立后缀：

```text
CntStat/218keV_RotateNum20_Geant4JSCC
CntStat/440keV_RotateNum20_Geant4JSCC
```

两类数据的 dataset stem 相同是为了让重建入口复用，但不能互相覆盖。读取 Geant4
数据时只设置 `--cntstat-dir-suffix _Geant4JSCC`，Factors 仍使用无后缀 JSCC
版本。Geant4 同时产生的 List 是没有 primary-energy 标签的 218/440 混合 List，
保存在 `List/218-440keV_RotateNum20_Geant4JSCC`，不属于 GenProj 输出，也不能
作为纯单能 List 使用。
