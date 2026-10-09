# EHE：系统矩阵前投影 + Poisson噪声，期望发射剂量5e9

新组已完成。完整输入10次验证、200次正式重建、独立严格验收、逐文件SHA取回、9张科学图和数值/视觉QA均已实际通过。定时任务compton-v5仍保持暂停。

## 数据生成和重建

用户2026-10-09要求建立独立实验`ehe_forward_poisson_5e9_200`。只读复用冻结74e129c4460163c5的完整A218、A440和C440→218系统矩阵，20视角、每视角2312个bin，完整120mm/132040圆点/78920活动单元。未重算Geant4、PE/Scatter或转换，未改变能窗、几何、MLEM更新式或正则化。

5e9为**期望发射gamma数**。产额0.114/0.259乘实际三维源积分后，218/440期望预算分别1469038993.409136/3530961006.590864；218份额0.29380779868182727，每视角使用各能量预算的1/20。真实源背景密度分别456.398946/1036.906378gamma/mm³。它不是5e9探测计数，也不是一次新的5e9输运。

生成使用本工程H60实际3mm真值的全部体素质量、原Cartesian双线性插值和20视角旋转。重建使用原完整体积加权Polar算子；两者离散基底不同。原真值、坐标、体积、旋转、三Factor清单和执行代码在生成前冻结，见[freeze.json](freeze.json)。

设三组前投影均值为A218×218源、A440×440源、C440→218×440源，每个view/bin/分量独立Poisson抽样。PCG64种子32100101、32100102、32100103，实际生成NumPy2.0.2。218观测逐bin等于直接与串窗样本相加，440观测等于A440样本；未匹配旧计数或拟合亮度。

| 生成分量 | 前投影期望计数 | 实际Poisson计数 |
|---|---:|---:|
| A218直接 | 13473.895759 | 13483 |
| A440直接 | 11951.142755 | 11921 |
| C440→218 | 7498.180746 | 7486 |

218窗总计20969=13483+7486，440窗11921。本组生成分量串窗占218窗35.70032%；这是模型生成分量比例。完整逐视角数据见[counts_by_view.csv](comparison/counts_by_view.csv)，均值、种子、分量和原投影保存在独立generated目录。

正式先重建440单光子200次，再重建218校正单光子200次。218固定加性Poisson背景来自**本组440最终200图**前投影，背景总计7461.205717；生成时保存的真实串窗分量仅用于核验，未作为重建背景。第三路逐帧相加同迭代两能gamma密度，非Ac225活度。两能均全1初值、原MLEM、无正则化；每10次保存，40个完整atomic/fsync检查点、三路各20帧，另在图集/指标中显式加入迭代0。

## 实际执行和验收

| 阶段 | 唯一作业 | 实际退出 | Slurm耗时 |
|---|---:|---|---:|
| 生成+完整输入validation10 | 1680214 | COMPLETED/0:0 | 05:32 |
| validation独立只读验收 | 1680238 | COMPLETED/0:0 | 08:36 |
| formal200/save10 | 1680255 | COMPLETED/0:0 | 17:11 |
| formal独立只读验收 | 1680300 | COMPLETED/0:0 | 02:54 |

执行冻结a73d3e0f89c133d8，独立验收发布e8ab82289ca738a5。10次原MLEM与checkpoint版本两能L2=0、历史逐值一致。全部2312行/20视角的独立CPU float64前投影均值全局L2分别4.6215e-8、4.9207e-8、4.9176e-8；原算子全视角前向/转置最大相对差2.2974e-7，矩阵自身S闭合。本组最终440200背景重算L2=2.8496e-7。

[formal_summary.json](formal_summary.json)严格验收通过：40个检查点、三路20帧、末帧/历史一致、两能和精确闭合、有限非负、活动域外零、固定背景身份、矩阵/代码/坐标/旋转/体积身份闭合。130份结果文件已按远端验收SHA严格取回。两能实际MLEM阶段耗时86.073/65.699秒；Slurm总耗时包含共享存储读取、全SHA核验、初始化和结果写出。

正式实际AllocTRES=94500MiB；Slurm MaxRSS=12791736K，占13.2190%。40个检查点记录的最大进程RSS占3.6442%，最大GPU reserved占16.0141%；实际资源均保留至少20%余量。CPU只读算子检查不是GPU成像资源证书。独立验收原始receipt和sacct分别保存在本目录，执行字节身份见[execution_identity_local.json](execution_identity_local.json)。4项本地合同测试通过，不替代实际完整输入验证。

## 图集和完整迭代曲线

新矩阵噪声EHE、原输运EHE各用0–200迭代；JSCC六类结果用0–10000迭代。总图包含12条路径，各自迭代标签和横轴完整保留，**同列不表示同收敛**。全部使用本工程H60真实3D球ROI，crop0、no smoothing、no fitted gain；显示按各自真实发射源背景密度归一到同一固定0–10尺度，超过显示上限的像素计数保存在[comparison_report.json](comparison/comparison_report.json)。指标保留完整未裁剪原值。

- [新EHE三路72mm MIP](comparison/synthetic_mip72.png)
- [新EHE轴面](comparison/synthetic_axial.png)、[冠面](comparison/synthetic_coronal.png)、[矢面](comparison/synthetic_sagittal.png)
- [新EHE/原输运EHE/JSCC十二路整体MIP对比](comparison/overall_mip72_12_routes.png)
- [所有球CNR完整曲线](comparison/cnr_full_trajectories.png)、[CRC完整曲线](comparison/crc_full_trajectories.png)
- [背景CV、密度峰值和高分位曲线](comparison/density_noise_curves.png)
- [总积分恢复、峰位置、轴向泄漏曲线](comparison/integral_position_curves.png)

新组指标包括63行原生120mm指标及252行3D球指标；原EHE/JSCC指标只读复用。逐帧数据见[synthetic_native_metrics.csv](comparison/synthetic_native_metrics.csv)、[synthetic_sphere_metrics.csv](comparison/synthetic_sphere_metrics.csv)。数值QA独立检查441项原生标量身份、189项float64背景标量及996项float64球ROI标量，9张最终图均直接视觉检查通过，见[scientific_figure_acceptance.json](scientific_figure_acceptance.json)及[visual_qa.json](visual_qa.json)。文件SHA图集清单见[artifact_manifest.json](comparison/artifact_manifest.json)。

## 本组结果

新组仍出现后期噪声放大：218的28mm球CNR在30次为6.769，到200次降为2.769；440的37mm球在20次为5.544，到200次降为1.773。背景CV、密度峰值和MIP尖峰随迭代增大。部分球的CNR峰值在较晚迭代，例如440的13mm球为180次，不能概括为统一的最佳迭代。

| 路径 | 球直径mm | 保存帧中CNR最高迭代 | 最高CNR | 200次CNR | 200次CRC |
|---|---:|---:|---:|---:|---:|
| 440单光子 | 13 | 180 | 3.170 | 3.165 | 1.126 |
| 440单光子 | 22 | 10 | 1.468 | 0.371 | 0.104 |
| 440单光子 | 37 | 20 | 5.544 | 1.773 | 0.491 |
| 218校正单光子 | 10 | 200 | -0.122 | -0.122 | -0.044 |
| 218校正单光子 | 17 | 30 | 2.630 | 1.505 | 0.317 |
| 218校正单光子 | 28 | 30 | 6.769 | 2.769 | 0.683 |
| 两能gamma密度和 | 10 | 200 | -0.451 | -0.451 | -0.411 |
| 两能gamma密度和 | 13 | 170 | 2.540 | 2.530 | 1.088 |
| 两能gamma密度和 | 17 | 70 | 0.896 | 0.668 | 0.500 |
| 两能gamma密度和 | 22 | 10 | 1.227 | 0.277 | 0.090 |
| 两能gamma密度和 | 28 | 10 | 2.848 | 0.689 | 0.609 |
| 两能gamma密度和 | 37 | 20 | 5.637 | 1.677 | 0.536 |

“最高迭代”只指本次每10次保存帧中的最大值，不是每步搜索或预先验证的停止规则。218的10mm球全部保存帧CNR为负，表中200次是最大负值，不能解释为已检出热球。

| 200次路径 | 背景均值gamma/mm³ | 背景CV | 总积分恢复 | 真源轴向域外泄漏（\|z\|>30mm） |
|---|---:|---:|---:|---:|
| 440单光子 | 893.211 | 2.852 | 1.1399 | 6.3599% |
| 218校正单光子 | 398.860 | 2.789 | 1.2071 | 5.4031% |
| 两能gamma密度和 | 1292.071 | 2.135 | 1.1596 | 6.0673% |

这是一组独立Poisson噪声实现，曲线和峰值为描述性结果；尚无多噪声重复的置信区间。矩阵自生数据不能独立验证物理响应，原输运证据保持。新组和原输运组均出现后期噪声，不能仅据此确定物理偏差根因。

EHE与JSCC保留各自材料、覆盖、截断和实际计数；固定218背景预算新EHE/原EHE为各自440200，JSCC为原44010000。此对比包含这些差异，不能全部归因于算法，也不代表真实设备性能。两能组合真值按真实gamma背景密度加权；实际球ROI为3D，未使用2D替代真值。

## 文件和复现

本组结果位于`experiments/ELLIPSE500x300_H120/generated/ehe_forward_poisson_5e9_200/results/formal`，均值/Poisson分量位于同实验`counts`。现有完成登记禁止重复提交已完成阶段。

```powershell
python -X utf8 experiments/ELLIPSE500x300_H120/ehe_forward_poisson_workflow.py status
```

推进入口具有本地PID和唯一作业登记保护；当前控制器已complete/exit0。本实验已完成，compton-v5保持用户要求的暂停状态。源码、报告、图集和小型验收证明进入安全Git交付；矩阵、响应块、历史大数据、构建二进制、压缩包和敏感文件不进入Git。
