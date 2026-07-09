# GenProj

本目录保存基于系统矩阵的 MATLAB 前投影脚本，用于从已生成的 `Factors/` 生成单光子投影数据 `CntStat/`。这些脚本不运行 Geant4，也不生成康普顿 `List/`；它们适合做系统矩阵、旋转矩阵、噪声模型和单光子重建链路的快速验证。

## 脚本

| 文件 | 用途 | 主要输出 |
| --- | --- | --- |
| `GenProj_SPECT_PolarCoor.m` | 通用 contrast/hot-rod/point-array 单光子前投影脚本，保留了多个历史体模配置块 | `CntStat/CntStat.csv` |
| `GenProj_Hoffman_SPECT_PolarCoor.m` | Hoffman 压缩脑模体的单光子前投影 | `CntStat/<energy>keV_RotateNum<rotate>/CntStat_HoffmanCompressed_<count>.csv` |
| `GenProj_ContrastPhantom_DualEnergy_PolarCoor.m` | 与 `Geant4Sim/ContrastPhantom_DualEnergy_Rotate_3D.m` 对应的 225Ac 双能量 contrast phantom 前投影 | `CntStat/218keV_RotateNum20/...csv` 和 `CntStat/440keV_RotateNum20/...csv` |

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

它优先读取主工程 `Factors/`，如果主工程没有 218/440 keV 因子，会自动回退到：

```text
Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main/Factors/
```

如果后续生成了 `RotateNum60` 的 218/440 keV 因子，只需要把脚本中的 `cfg.rotate_num` 改为 `60`，并保证对应目录存在。
