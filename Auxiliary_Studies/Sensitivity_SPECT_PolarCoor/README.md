# Sensi_d 计算工具

## 标准可视化输出

每次完成 `Sensi_d` 计算后，应使用以下命令生成标准对照图：

```powershell
python Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\visualize_sensi_d_vs_single_photon.py
```

脚本会读取当前 Factors 中的 `SysMat_polar`、`RotMatInv_full.csv` 和计算出的
`Sensi_d`，并输出到本次 `Result` 目录。图中的空间图已经转换为连续的平面直角
坐标 `(x, y)`，不是极坐标采样点散点图：20 个 z 层先在相同 `(x,y)` 位置平均，
再插值到规则 Cartesian 网格。由于当前 Factors 是 `A * DeltaV` 的密度基准，
直角坐标显示会自动除以 `polar_cell_volume_mm3`，恢复单位发射光子的点响应，
避免不同极坐标环的单元体积造成一圈一圈的显示伪影。该除法仅用于显示，不修改
磁盘上的 `Sensi_d`、`Sensi_s` 或 Factors。

- `Sensi_d_vs_single_photon_cartesian_xy.png`：直角坐标平面图；
- `Sensi_d_vs_single_photon.png`：同一标准图文件名，内容也为直角坐标平面图；
- `Sensi_d_vs_single_photon_radial.csv`：径向中位数和平均值；
- `Sensi_d_vs_single_photon_summary.json`：范围、变异系数和归一化相关性。

标准图固定为从左到右四栏：`Sensi_s / DeltaV`、`Sensi_d / DeltaV`、未做中位数
归一化的 `Sensi_s / Sensi_d`，以及前两者未做中位数归一化的径向曲线。这里
`Sensi_s` 必须按 MATLAB 列主序读取 `SysMat_polar`，与本地重建代码相同；否则会
错误打乱 detector/pixel 维度并产生伪影。`Sensi_d` 与 `Sensi_s` 代表不同事件链路，
两者的比值用于比较两种探测效率，不应解释成已经标定好的重建修正因子。

## Cartesian 点响应实验链路

正式的密度基准矩阵满足 `B=A*diag(DeltaV)`。为了避免在事件核中混入柱坐标
单元体积，Compton 灵敏度的推荐计算顺序是：

```text
Cartesian SysMat_tmp point response
  -> accumulate normalized event kernels on equal-spacing Cartesian samples
  -> normalize to kept_events / emitted_photons
  -> interpolate point efficiency to the polar grid
  -> rotation average and absolute-efficiency closure
  -> Sensi_d[j] = epsilon_d[j] * DeltaV[j]
```

`prepare_cartesian_sensitivity_input.py` 从带 `SysMat_tmp` 的临时 Factors 中提取
`r<=153 mm` 的圆柱支持域，并转换成 Python 重建代码使用的磁盘布局。
`convert_cartesian_sensitivity_to_polar.py` 完成插值、旋转平均、整体效率闭合和最终
密度基准转换。当前 Geant4 `EventAction` 写入 List 前已经对能量进行展宽，因此运行
计算时必须添加 `--input-energies-already-smeared`，禁止 Python 再次展宽。

2026-07-21 的 0.2% List 验证表明：Cartesian A440 的读取和插值与正式 polar
Factors 的单光子点效率相关系数为 `0.999999876`，中位绝对相对误差为 `1.1e-5`；
转换后的全局 Compton 效率闭合误差为 `4.4e-16`。但 Compton 点效率仍随半径下降，
说明体积基准不是该趋势的根因。当前事件核仍以 A440 光电峰响应近似首次 Compton
作用似然，并将每个有限-FOV锥面归一化，因此不能保证恢复真实的逐位置 Compton
探测效率。正式使用前仍应通过记录每个有效 List 事件的真实 primary 发射位置来验证。

项目当前状态、当前需要生成的单能全 FOV List 数据和后续 Compton 重建顺序见
`../../docs/DEVELOPMENT_HANDOFF.md`。

> 当前正式 Factors 使用活度浓度基底 `B=A*diag(DeltaV_mm3)`。因此正式
> `Sensi_d` 必须来自覆盖完整极坐标单元边界的连续均匀体积源，并使用
> `Sensi_d = accumulator * Vsource / Nprimary` 归一化。当前网格对应
> `R=153 mm`、`z=-30..30 mm`、`V=4412492.545673008 mm3`。配套 Geant4
> macro 位于 `Geant4Sim/Macro/SensiD_UniformFullFOV/`。

底层点响应 `A_E(d,j)` 表示在位置 `j` 发射一个指定能量的光子后，在响应通道
`d` 被接受的概率/期望计数；它不包含 225Ac gamma 产额。正式矩阵
`B_E=A_E*diag(DeltaV)` 的列已经乘以 `mm3`，所以它不再是单个点光子的无量纲
概率矩阵，而是把“该能量的发射光子浓度”映射到期望数据。`Sensi_d` 必须使用
同一密度基底。

本目录用于从 Monte Carlo 生成的两次作用 Compton List 计算极坐标重建所需的
`Sensi_d`。它由原独立工程
`F:\lipeize\1_code\0_repository\Sensitivity_SPECT_PolarCoor` 整理而来，已经改成
直接使用当前仓库的 `Factors/<energy>keV_RotateNum<rotate_num>/`，不再依赖旧工程的
`SysMat/<system-name>/{SC,Coor,Detector,RotMat,Compton}` 目录。

本目录只保存代码、测试和说明。体积很大的 `SysMat_polar`、Compton List、运行结果
和检查点均不复制到这里，也不会提交到 Git。

## 1. 计算目标

对每个通过能量、运动学、跨层和稳定性筛选的 Compton 事件，程序计算一行体素
反投影核：

```text
t_i(x) = ComptonCone_i(x) * SysMat_polar[first_detector_i, x]
```

每个有效事件先按体素归一化：

```text
t_i_normalized(x) = t_i(x) / sum_x t_i(x)
```

随后累加并进行绝对效率标定：

```text
S_raw(x) = sum_i t_i_normalized(x)
sum_x(S_raw_scaled) / source_volume_mm3 =
    kept_event_count / represented_source_photons
```

最后使用 `RotMat_full.csv` 对 `S_raw_scaled` 做 `rotate_num` 个角度的平均，得到重建
实际读取的 `Sensi_d`。

## 2. 相比原独立工程的改动

- 直接读取当前 `Factors` 目录中的 `SysMat_polar`、`Detector.csv`、
  `coor_polar_full.csv` 和 `RotMat_full.csv`。
- 默认严格检查探测器数为 `10496`。这对应当前 Geant4/Factors 几何中去除钨块后
  的有效探测器数。
- Compton CSV 按批流式读取，不再一次性把数 GB List 装入内存。
- 每批的“事件数 x 体素数”矩阵在 CPU/GPU 上立即归约，只把一张体素累加图保留
  到下一批，显著减少 CPU 传输和峰值内存。
- 不再用含义不准确的 `num_workers` 表示分块数，改为直接设置 `--batch-size`。
- 固定随机种子，能量展宽结果可重复。
- 支持多个 List 文件或一个包含多个 CSV 的目录；文件按名称排序后连续处理。
- 支持定期检查点和 `--resume` 续算。
- 保存完整 `run_metadata.json`，包括输入文件、筛选计数、物理参数、归一化误差和
  输出路径。
- 默认使用当前 `process_list_plane_strict.py` 的位置不确定度约定：第一次作用晶体
  在“源点到第一次作用点”一段的尺寸展宽不重复计入，因为 `SysMat_polar` 已经包含
  该空间响应。仅在复现旧结果时才使用
  `--include-first-hit-source-leg-uncertainty`。

## 3. 环境

依赖：

- Python 3.9+
- NumPy
- PyTorch
- CUDA 可选，但正式全量计算建议使用 CUDA

本机推荐使用已有 `pytorch` conda 环境：

```powershell
conda run --no-capture-output -n pytorch python -u `
  .\Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\run_compton_sensitivity.py --help
```

## 4. 必要输入

以 `Factors/440keV_RotateNum20/` 为例，默认读取：

| 文件 | 作用 |
| --- | --- |
| `SysMat_polar` | 第一次作用探测器到极坐标体素的响应，float32 二进制 |
| `Detector.csv` | `[detector_id, x, y, z]`；ID 必须从 1 连续到 10496 |
| `coor_polar_full.csv` | 完整极坐标体素坐标 `[x, y, z]` |
| `RotMat_full.csv` | 每列必须是 `1..pixel_count` 的一个完整排列 |
| Compton List CSV | 至少前四列为 `[cpnum1, e1, cpnum2, e2]`，第五列会忽略 |

程序根据坐标行数和探测器行数严格核对 `SysMat_polar` 文件大小。当前
`10496 x 25620` 因子应为：

```text
10496 * 25620 * 4 = 1,075,630,080 bytes
```

### `--source-photons` 的含义

`--source-photons` 是全部输入 List 文件所代表的、该能量实际发射的光子数，不是
保留下来的 List 行数，也不一定是 225Ac 衰变总数。

- 如果独立模拟了 `N_440` 个 440 keV 光子，传入 `N_440`。
- 如果独立模拟了 `N_218` 个 218 keV 光子，传入 `N_218`。
- 如果 218+440 按 225Ac 分支比在同一个 run 中发射，应先按真实初始能量拆成两套
  List，再分别使用该 run 中 218 和 440 的实际发射数计算两张 `Sensi_d`。
- 不要把混合 List 直接作为单一能量输入；程序的 Compton 运动学和阈值只对应一个
  `energy_mev`。

## 5. 440 keV 示例

下面的阈值与当前工程的 440 keV List 设置一致：

```powershell
conda run --no-capture-output -n pytorch python -u `
  .\Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\run_compton_sensitivity.py `
  --factor-dir .\Factors\440keV_RotateNum20 `
  --compton-list D:\SPECT_Data\Compton\440keV `
  --source-photons 5e10 `
  --energy-mev 0.440 `
  --rotate-num 20 `
  --energy-threshold-sum-mev 0.40 `
  --device cuda `
  --batch-size 256 `
  --output-dir .\Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\Result\440keV_RotateNum20
```

## 6. 218 keV 示例

```powershell
conda run --no-capture-output -n pytorch python -u `
  .\Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\run_compton_sensitivity.py `
  --factor-dir .\Factors\218keV_RotateNum20 `
  --compton-list D:\SPECT_Data\Compton\218keV `
  --source-photons 5e10 `
  --energy-mev 0.218 `
  --rotate-num 20 `
  --energy-threshold-sum-mev 0.18 `
  --device cuda `
  --batch-size 256 `
  --output-dir .\Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\Result\218keV_RotateNum20
```

能量和旋转数可从标准 Factors 目录名自动推导。工程内置的总能阈值默认值为：

| 入射能量 | 默认 `e1 + e2` 下限 |
| ---: | ---: |
| 218 keV | 0.18 MeV |
| 440 keV | 0.40 MeV |
| 511 keV | 0.46 MeV |
| 662 keV | 0.60 MeV |

其他能量默认使用 `0.9 * energy_mev`。正式计算仍建议在命令中显式写出阈值，并与
后续重建 List 的筛选参数保持一致。

## 7. 输出

每次运行目录包含：

| 文件 | 说明 |
| --- | --- |
| `Sensi_d` | 旋转平均后的最终 float32 灵敏度图 |
| `Sensi_d_raw` | 旋转平均前的绝对标定结果；可用 `--no-save-raw` 关闭 |
| `run_metadata.json` | 输入、参数、筛选计数、归一化和运行信息 |
| `checkpoint.npz` | 未完成运行的断点；成功结束后默认删除 |

程序在写出前自动检查：

- 输出长度等于 `pixel_count`；
- 所有值有限且非负；
- 密度基 Factors 满足
  `sum(Sensi_d)/source_volume_mm3 = kept_events/represented_source_photons`，
  旧积分活度基 Factors 才使用
  `mean(Sensi_d) = kept_events/represented_source_photons`；归一化相对误差不超过
  `5e-5`；
- 每个旋转矩阵列都是完整的一对一排列，因此旋转平均不会改变总灵敏度。

## 8. 断点续算

默认每 100 个批次写一次检查点。中断后使用完全相同的输入和数值参数，并添加
`--resume`：

```powershell
conda run --no-capture-output -n pytorch python -u `
  .\Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\run_compton_sensitivity.py `
  <原来的全部参数> `
  --resume
```

检查点包含累加图、已处理 CSV 字节位置、筛选计数和随机数生成器状态。若输入文件
大小、修改时间或关键参数变化，程序会拒绝续算，以免拼接出不一致结果。

可调整：

```text
--checkpoint-every-batches 100
--progress-every-batches 10
--keep-checkpoint
```

## 9. 小比例验证

`--event-fraction` 只读取按文件名拼接后 List 的前缀，并按实际选中行数同比缩放
`source_photons`。例如：

```powershell
--event-fraction 0.000001
```

它适合检查路径、几何、内存和输出格式，不适合产生正式 `Sensi_d`，因为 List 前缀
未必是理想的随机样本。正式结果应使用默认 `--event-fraction 1.0`。

## 10. 安装到 Factors

建议先在 `Result/` 检查 `run_metadata.json` 和数值，再显式安装：

```powershell
<原命令> --install-to-factor-dir
```

如果 `<factor-dir>/Sensi_d` 已存在，程序默认拒绝覆盖。确认需要替换后使用：

```powershell
<原命令> --install-to-factor-dir --overwrite
```

`--overwrite` 也允许覆盖同一输出目录中的旧结果，因此使用前应确认目标路径。

## 11. 性能和显存

- `SysMat_polar` 会常驻所选设备；当前 10496x25620 float32 矩阵约占 1.00 GiB。
- 批次计算还会生成多个 `batch_size x pixel_count` 临时张量。
- 48 GB GPU 可从 `--batch-size 256` 开始；显存较小时依次尝试 128、64、32。
- CPU 模式可用于格式验证，但全量 List 通常很慢。
- 程序先快速统计 CSV 行数，因此正式计算开始前会有一次顺序磁盘读取。

## 12. 测试

运行内置合成测试：

```powershell
conda run --no-capture-output -n pytorch python `
  -m unittest discover `
  -s .\Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\tests `
  -v
```

测试覆盖：

- float32 系统矩阵布局；
- List 分批与事件比例；
- 绝对归一化和旋转平均；
- 10496 探测器数量保护；
- 218/440/511 keV 默认阈值；
- 检查点保存和恢复。

## 13. 当前边界

- 当前工具计算的是每个能量各自的 Compton `Sensi_d`，不处理 440 keV 进入 218
  能窗的串扰灵敏度。
- 它不负责推断两次作用的先后顺序，输入 List 的 `cpnum1/e1` 必须已经代表第一次
  作用。
- 四层晶体尺寸模型固定为前三层 `3x3x3 mm`、最后一层 `2x6x2 mm`，并要求
  `Detector.csv` 中能识别出四个绝对 y 层。
- 若改变探测器结构、晶体尺寸或 List 筛选逻辑，必须同步修改本工具和
  `process_list_plane_strict.py`，不能只修改其中一处。
