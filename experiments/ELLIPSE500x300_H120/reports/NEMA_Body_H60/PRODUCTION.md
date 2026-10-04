# NEMA H60：1e9 Geant4 与椭圆 FOV 重建运行簿

2026-10-04响应修正续作见[完整单元与A场精度](compton_response_geometry_v3/WHOLE_CELL_VALIDATION.md)。复用首散射修正1e9及既有独立验证数据，没有新输运。稳定几何R1作业1661583的原q3保留50次回归两路L2=0、完整10次试跑通过；独立近侧A采样加密至0.375mm后84/84响应案例达到相邻1%标准。原插值场的具体单元误差可达38%，尚需接入实际R2积分场并生成/验收S2。本次未完成正式2000次R1/R2配对，旧正则化和自动任务保持停止。下列日期和指标属于原1e9六路基线。

状态日期：2026-10-01。**1e9 输运与10000次六路重建的文件完整性已通过；科学质量仍有明显限制。** 此页是运行记录与验收证据；不能仅凭 Slurm `COMPLETED` 推断模拟或重建成功。几何与活度设计见 [README.md](README.md)，椭圆实验基线见 [主 README](../../README.md)。

## 冻结输入

- 本地 3 mm 真值：`generated/NEMA_Body_H60/truth_3mm.npz`，SHA-256 `2612f0ed6839f9460722711e1017a10102e83adf77cf715d5c2553cfaec948af`；完整描述及源配置哈希见 `manifest.json`。此处修正了早期文字抄录遗漏，生成器与分析始终核对 manifest 中的完整64位哈希。
- 20 视角、200 worker 的 `Simulation_1e9/jobs.json`，SHA-256 `d6a45235e38d1217949887eb42c13ec6cb64be83350219ec15933c06ee7e9b2e`。每 worker 5e6，独立种子 30093001–30093200，总计 **1e9** 初级 γ。
- 3 mm 体素真值合并为 218 keV **1945** 个、440 keV **1992** 个 cuboid；体积与 0.114/0.259 光子产额加权后，预计初级 γ 组成 **29.3808% / 70.6192%**。两能量各自的热球/背景**活度浓度**比为 10:1，不能把光子产额差误当作浓度差。
- 20 份宏在 maty 逐份核对 SHA-256、`/xcat/angle`、`/run/beamOn` 和 cuboid 数。与已有矩阵相同的中心 `(0,-345,0) mm`、20 视角、13% FWHM@511 keV 的 List 展宽链，源无人体材料衰减。`/xcat/add` 对每个体素盒均匀采样，边界保留 3 mm 真值的部分体积权重；后续解析球面精度限制须在小球评估中指出。

## maty 作业与故障记录

工作区：`/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928`。所有 NEMA 工件在其 `experiments/ELLIPSE500x300_H120/generated/NEMA_Body_H60/`，旧体模结果保持原样。

| 作业 | 范围 | 状态与证据 |
|---|---|---|
| `15506070` | worker 0、1e4 初级 γ 短程 | 通过：218/440/其他为 2911/7089/0；worker 哈希及探测器计数形状通过 |
| `15506077` | 首次 0–199 正式阵列 | 批处理脚本在 `set -u` 下使用空 `args[@]` 报错；Geant4 **未启动**，无 `workers/` 输出，不计入生产 |
| `15506288` | worker 0、完整 5e6 | 通过：218/440/其他为 1470218/3529782/0；约 2 分 44 秒；输出哈希、初级比例及探测器计数核验通过 |
| `15506302` | worker 1–199，最多 32 个并行 | 全部完成；连同 worker 0，共 200/200 独立 worker 逐一通过 `--stage complete` 核验 |

完整发射数为 218/440/其他 **293821153/706178847/0**；20 视角、200 种子、全部输出哈希及计数形状通过。收集后 218/440 能窗的 CntStat 总计分别为 **2462927/1073047**，合并 List 共 **1093494** 行；List 行数不作为初级发射数。对应不可变证据在 [`transport_1e9.json`](transport_1e9.json)。收集 manifest SHA-256 为 `3a1aac57da8bc80e5323c8ccc6623942a3beeaf401bee878ce6c60f1e8659b62`。

在 maty 上核查：

```bash
cd /WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928
python3 experiments/ELLIPSE500x300_H120/check_simulation.py experiments/ELLIPSE500x300_H120/generated/NEMA_Body_H60/Simulation_1e9/jobs.json
python3 experiments/ELLIPSE500x300_H120/validate_nema_simulation.py experiments/ELLIPSE500x300_H120/generated/NEMA_Body_H60/Simulation_1e9/jobs.json --stage complete
```

第二条只有 200 个 worker 全部完成且实际初级光子数、能量比例、文件 SHA-256、晶体计数形状均通过才会返回成功；本次已通过。`experiments/FOV120/workflow.py collect` 已校验 20 视角和唯一种子，生成 `CntStat`、`List` 和 `collections/NEMA_Body_H60_1e9.json`。收集器拒绝 smoke 混入正式输入。

## scxi717 重建门槛

`package_imaging.py --datasets NEMA_Body_H60 --archive-stem nema_h60_imaging_1e9` 已生成逐文件 SHA-256 包，归档为 23 个文件、15,886,400 字节；`deploy_imaging.py --archive-stem nema_h60_imaging_1e9` 已安装到 scxi717 的**同一椭圆实验** `generated/` 并逐文件验哈希，禁止新建矩形支持域或复用 FOV120 Factors。

完整网格及完整事件的 10 次资源试跑 `1643079` 在 scxi717 用 4 节点×每节点 1 张 4090 完成（Slurm `COMPLETED 0:0`），`check_remote_pilot.py` 对六路 active/full/history、串窗预测、有限非负值及活动列检查通过。重建筛选后的 Compton 事件为 **97299**，4 个 rank 的显存最大预留 **11.734375 GiB**，最小设备容量约 **23.51648 GiB**，实际峰值约为容量的 50%；Slurm 聚合 step MaxRSS **26508596 KiB**，每节点分配 **60000 MB**。按 55 GiB/节点保守预算重新执行 `resource_budget.py`，GPU 和主存均满足 20% 余量。stderr 在进程退出阶段有 TCPStore `RendezvousConnectionError`，但退出码为 0 且全部试跑输出核验通过；正式作业仍需关注该日志是否在计算阶段复现。

10000 次、每 50 次保存的六路 JSCC 正式作业 **`1643142`** 已于 2026-09-30 在 `gpu_4090` 以相同 4 节点×1 GPU、6 CPU/GPU 启动，NCCL 固定 `bond0`，设置 24 小时上限。作业通过资源预检后仍须以实际结果校验，不能仅按 Slurm 状态宣布成像完成。重建继续使用已有新距离的三套 Factors、椭圆活动列和新 `Sensi_d`。

正式结果必须由 `verify_formal_result.py` 检查 1e9 初级 γ、20 视角、200 个独立 worker、六路最终图、各 200 帧历史、串窗预测、有限非负值、几何/灵敏度哈希及资源余量。科学评价再分别对 Ø10/17/28 的 218 通道与 Ø13/22/37 的 440 通道计算真值对照的 CRC/CNR、背景偏差/CV、积分恢复和迭代曲线；必须区分单能 10:1 真值与两能量直接相加后的 5:1 预览。整个实验不应因某个作业退出码为 0 就宣布有效 FOV 通过。

## 2026-10-01 正式验收结果

`1643142` 的 Slurm 起止时间为北京时间 2026-09-30 22:55:52 至2026-10-01 04:37:25，耗时 **5:41:33**，全部step退出码 **0:0**。本次stderr仅见设备id未显式传入的PyTorch提示，未复现试跑退出时的TCPStore异常。作业结束后计算节点SSH被`pam_slurm_adopt`拒绝、squeue报告invalid job id是分配释放后的正常现象，已用sacct和结果文件独立确认完成。

`verify_formal_result.py` 返回 `ELLIPSE_FORMAL_INTEGRITY_OK`。证据见 [完整性JSON](../NEMA_Body_H60_1e9_1643142_integrity.json)：

- 1e9实际初级γ、20视角、200独立worker；正式接受Compton事件 **97299**，四rank分别24268/24354/24314/24363。
- 六路132040列最终图、82040活动列图及每路200×82040历史；全图在活动列外为零，历史最后一帧与最终图逐元素相同；所有值有限且非负。
- 几何、`Sensi_d`、collection哈希与冻结输入一致；六路最终图/历史及串窗预测全部逐文件SHA-256取回。串窗预测计数和 **772454.77368927**，此值是模型预测，不能直接当成实测串窗事件数。
- GPU最大预留/设备容量 **51.024%**；Slurm step最大单task RSS为 **26668416 KiB≈25.43 GiB**，相对于保守55GiB/节点预算为 **46.24%**。两者均满足至少20%余量。详见 [资源与验收记录](NEMA_Body_H60_1e9_1643142/acceptance.json)。

### 图集、方法及复现

[全部图集与数据入口](NEMA_Body_H60_1e9_1643142/README.md)。轴位图包含z=+1.5、−28.5、+28.5、−58.5、+58.5mm，显示100/500/1000/2000/5000/10000次的六路结果；最终图包含轴位、冠状位、矢状位、轴向MIP。CSV包含全部200个实际保存迭代点的CRC/CNR/CV、背景偏差、真值拟合NRMSE和积分恢复率。

使用冻结的本实验3D双能球体真值，不使用可视化技能中另一个全热柱NEMA目录。极坐标图在匹配z层进行XY三角形重心线性插值到3mm画布；源与重建物理支持仍为椭圆柱。图像使用`gray_r`（白低黑高）、sigma=0、无边缘裁剪；主图固定0–10色标，每通道以**最终帧同一个背景均值**归一化所有显示迭代，避免逐帧亮度拉伸掩盖变化。超过10只在显示中饱和，指标采用完整原始数值。中央细节图仅改变显示窗口，不修改FOV或数组。

单能真值背景1、目标球10。实际重建相加的是γ密度，因此两复合通道真值采用 `(.114×activity218+.259×activity440)/.373`；其218球/背景约3.056、440球/背景约6.944。原几何页面的**未乘产额的活度预览**才是背景2、各球10（5:1），不能用来直接计算γ复合图CRC，更不能当母体225Ac活度。

球ROI为3mm球体体积占比分数加权均值；局部背景取XY球心半径`球半径+25mm`、z半宽`球半径+3mm`范围内纯背景体素；背景空间标准差使用ddof=1。CRC=`(hot/bg−1)/(expected_ratio−1)`，CNR=`(hot−bg)/std_bg`。全局背景CV取`|z|≤25.5mm`纯背景。源期望背景发射密度由实际初级计数除以真值体积积分得到。总γ恢复率在**原极坐标活动密度**上乘`ΔV×ellipse_fraction`积分，不在插值显示图上积分。NRMSE额外拟合一个非负整体标量，仅用于指标，不改变图集尺度。

复现（从仓库根目录执行）：

```powershell
python experiments/ELLIPSE500x300_H120/fetch_integrity_report.py NEMA_Body_H60_1e9_1643142
python experiments/ELLIPSE500x300_H120/fetch_nema_results.py NEMA_Body_H60_1e9_1643142
python experiments/ELLIPSE500x300_H120/plot_nema_iterations.py NEMA_Body_H60_1e9_1643142
python experiments/ELLIPSE500x300_H120/analyze_nema_result.py NEMA_Body_H60_1e9_1643142
```

### 科学质量：尚不能判定小球和全视野达标

| 通道与球径 | 10000次CRC | 10000次CNR | 同一体素真值的CRC参考 |
|---|---:|---:|---:|
| 218校正，10mm | −0.0246 | −0.153 | 0.662 |
| 218校正，17mm | 0.6913 | 4.437 | 0.830 |
| 218校正，28mm | 0.8805 | 5.803 | 0.897 |
| 440 JSCC，13mm | −0.0533 | −0.381 | 0.768 |
| 440 JSCC，22mm | 0.2076 | 1.703 | 0.856 |
| 440 JSCC，37mm | 0.4300 | 2.056 | 0.919 |

球体真值CRC参考低于1是3mm边界部分体积和分数ROI平均的结果，小球偏差不能全部归因于重建。218的17/28mm球形成对比度；10mm和440的13mm球在最终帧低于局部背景。440的22/37mm球仍明显欠恢复。10000次全局背景CV：440单光子1.964、Compton5.921、440JSCC1.773、218校正1.505、两复合图1.422/1.304。背景CV持续增长；440JSCC37mm球CNR在2000次约3.31，至10000次降至2.06，不能将迭代数越高视为成像越好。

六路总γ积分恢复率约98.38%–100.88%，但这不足以证明空间分布正确。218校正与440JSCC的纯背景密度偏差分别约−11.75%/−10.44%；重建积分中位于**源不存在的`|z|>30mm`**区域的比例分别约5.52%/13.86%，Compton单独约17.32%。多平面/MIP和端层图已直接呈现泄漏与尖峰；目前只能确认完整流程与输出核验通过，不能确认有效椭圆FOV或所有球的定量性能通过。原因还需独立研究，不由本轮自动任务启动1e10或更改响应模型。
