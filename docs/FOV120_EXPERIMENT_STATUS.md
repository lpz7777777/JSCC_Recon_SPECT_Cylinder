# ²²⁵Ac 218+440 keV：FOV120 实验目录与进度交接

核查日期：2026-09-25。本页是已核查快照，作业状态会变化。
命令详见 [实验操作手册](../experiments/FOV120/README.md)，远程访问详见
[安全连接说明](REMOTE_COMPUTE_ACCESS.md)。原有 60 mm 数据不移动、不覆盖。

## 完整点源核查与报告交付（2026-09-25 最新）

点源扫描 **15385868：162/162 全部 COMPLETED**。完整收集验证通过：162 个唯一种子、
实际 1.62e9 光子、无缺失，三通道各 81 个点、共 243 组响应对照。
文件 `generated/FullData/point_responses_all.npz/.json`、`point_comparison_all.json`；
旧 snapshot1 保留为历史快照，以下部分快照不再代表当前完整性。

完整 MC/模型总计数 min/median/max：218 **0.99109/0.99922/1.00872**；
440 **0.98694/0.99858/1.01443**；串窗 **0.95698/0.99676/1.02515**。
逐晶体去噪 L2 中位数仍为 6.30%/4.91%/0.94%；四层计数比全部点的范围分别为
218 0.95485–1.02348、440 0.91838–1.04421、串窗 0.89822–1.10282。
这些含统计误差和点响应插值影响，不能将全部偏差精确解释为物理模型误差。

14 图 HTML 报告已刷新为完整 162 点，所有图片/CSV/JSON 链接检查通过，代表性图像已目视检查。
入口 `generated/ClosedLoop_VisualReport/index.html`；便携包 `generated/FOV120_VisualReport.zip`。
图中无额外高斯平滑、无裁剪、无强度拟合；统计直接来自原始极坐标数据。

Uniform 1e9 作业 **15386226** 和 Contrast 1e9 作业 **15386227** 正在运行；
自动收集作业 **15386284** 依赖 afterok:15386226:15386227，只有两组全部成功才运行。
收集使用既有 workflow 的发射数、任务/种子、文件哈希与视角齐全检查，完成标记为
`FOV120_UNIFORM_CONTRAST_1E9_COLLECTED`。尚无这两组实际 Geant4 六路图像。

## 可视化、点源部分收集与 1e9 验证启动（2026-09-25）

点源短程作业 15385867 完成，0:0、8分46秒。完整扫描 15385868 最新快照 **160/162 完成**，
560/561 仍在运行（均为 440 keV、r=135 mm、z=+57 mm，方位 180/270 度）。
已按 manifest 校验 160 worker 的任务字段、实际发射数、纯能量、全输出哈希、晶体/可执行文件一致性和唯一种子，
共 1.6e9 光子；未完成点明确记录，未用零值填充。证据 `generated/FullData/point_responses_snapshot1.npz/.json`。
收集器 `collect_point_responses.py` 默认拒绝缺失，只有显式 --allow-partial 才导出部分快照。

在 65114 用校准 B 除单元体积，做层内重心插值和 z 线性插值，得到每发射光子的点源响应。
单视角模拟不除以 20；保留三通道及四层绝对计数对照，无总量拟合。
结果 `generated/FullData/point_comparison_snapshot1.json`，由 `compare_point_responses.py` 重现：

| 通道 | Geant4/矩阵总计数 最小 / 中位 / 最大 | 去 Poisson 噪声的逐晶体相对 L2 中位数 |
|---|---|---:|
| 218 | 0.99109 / 0.99922 / 1.00872 | 6.30% |
| 440 | 0.98694 / 0.99837 / 1.01443 | 4.91% |
| 440→218 | 0.96184 / 0.99676 / 1.02515 | 0.94% |

逐晶体差异估计为 sqrt(max(sum((MC−pred)^2−MC),0)/sum(pred^2))，包括网格插值误差及统计估计不确定性，
不是严格的模型误差置信区间。总计数一致不代表空间响应形状完全一致；少数分层比偏差仍达约 10%。
完整扫描和响应检验不能替代 20 视角点源定位/FWHM 测试。

详细可视化入口：`experiments/FOV120/generated/ClosedLoop_VisualReport/index.html`。
共 14 张 PNG：218/440 真值—无噪声—Poisson 三轴位与 MIP；全高冠状/矢状/MIP；
50/200/500/1000 次演化；绝对误差；全局收敛；CRC/CNR 曲线；点源总量比、效率及分层比。
全部 **crop=0、sigma=0、无强度拟合**，gray_r 白低黑高，共用各通道真值绝对色标。
指标在原始极坐标上按体积权重计算；仅显示采用层内线性插值。轴位实际 z=-46.5,-1.5,43.5 mm。
报告附 80 行全局指标、720 行热柱指标、真值副本、来源哈希与绘图脚本，最终指标与既有数值报告逐项一致。
这些是同矩阵闭环单光子图，不是 Geant4 六路图，也没有伪造 Compton 结果。

已继续提交独立规则体模 **1e9 验证**（不是 1e10 生产）：
- Uniform：**15386226**，200 workers×5e6，20 视角，最多 20 并发。
- Contrast：**15386227**，同上。两组目前各 20 RUNNING，余下排队。
- 原 162 个点源 worker 不重跑。新任务目录事先检查为空，沿用原宏、原 seed、原 manifest。
- maty 的 MaxArraySize=1001；首次 962–1161 直接索引提交失败，无任务启动。
  `maty_pilot.sh` 增加经过数字校验的 FOV120_JOB_INDEX_OFFSET，Slurm 数组均 0–199，
  Uniform 偏移 562，Contrast 偏移 962，保持物理任务编号不变。bash -n 通过。
- 提交凭证在 maty `generated/Uniform_1e9_submission.json`、`Contrast_1e9_submission.json`。

下一步：收齐最后两点并刷新响应图；收集两组 1e9 数据后在 scxi717 做实际事件显存/六路短程验证，
再进行完整迭代与同体模单光子/Compton/联合对照。1e10 与 XCAT 正式成像仍待验证评估。

## 部署完成与当前作业（2026-09-25）

正式 Factors 已上传到 scxi717 用户指定工程的 `experiments/FOV120_20260924/experiments/FOV120/generated/Factors`。
65114→本地→scxi717 的完整 6,512,916,480 bytes tar 包 SHA256 全部一致：
`b7da8211a7af3c4421f72d9fc2400faaf7de2ecb86521397451f941b1804df2a`。
远端出现 `FACTORS_DEPLOYED_HASH_VERIFIED` 和 `DEPLOYED_GEOMETRY_AND_SENSITIVITY_VALIDATED`。
三路均 51240 点 / 40 层 / 2,151,260,160 bytes，体积 8,824,985.091346018 mm³；
全矩阵有限非负检查、坐标/体积/旋转映射一致性和 Sensi_d 来源全哈希检查通过。
本地复检记录：`generated/FullData/deployed_factors_validation.json`。

首次部署校验暴露 Python 3.9 缺少 hashlib.file_digest；已将 fov_config.py 改为 8 MiB 分块流式 SHA256，
并同步到重建超算后重新通过。15 项既有 FOV120 测试通过；新增旧版 API 缺失及跨块尾部篡改测试通过。
原始矩阵和灵敏度没有因此改变。完整事件显存测试仍需相应 20 视角 Geant4 成像数据，尚未完成。

点源短程作业 **15385867** 最近核查 RUNNING（6分32秒），数组 **15385868** PENDING/Dependency；
仅在四次短程核查成功退出后启动。此后应检查逐 worker PrimaryCount/种子/哈希，按点收集，
比较探测器计数和分层响应；不能以作业完成替代科学验证。

以下为实施过程快照，已被本节明确更新的上传/连接状态不再作为当前状态。

## 闭环诊断补充（2026-09-25）

新增可复现入口 `experiments/FOV120/diagnose_closedloop.py`，已在 65114 对完整网格运行通过。
用独立 NumPy GenProj、真实 218/440 密度、真实串窗投影调用正式 Torch MLEM 单步更新：
440 / 218 的体积加权相对变化为 1.5112e−7 / 1.6045e−7，最大体素相对变化为
7.9012e−7 / 1.0097e−6，低于 2e−5 检查门槛。未发现明显的旋转、密度基底或计数归一化不一致；
该结果不证明实际探测器响应模型正确，也不保证逆问题易于求解。

诊断采用冻结真值的分数体积热柱 ROI；背景限定 r≤120 mm、相同热柱轴向区间并排除所有热柱触及体素。
CRC=(重建热柱/背景−1)/(真值热柱/背景−1)，均使用体积加权均值，不做强度拟合、裁剪或平滑。
无噪声第 1000 次 CRC 仅约 **1.59%–7.40%**，第 500→1000 次仍上升，中心和两端均恢复不足。
无噪声 440 相对图像误差 50/500/1000 次为 44.19% / 42.72% / 41.97%；218 为
39.92% / 37.72% / 36.62%。Poisson 对应 440 为 44.19% / 43.35% / 44.38%，218 为
39.91% / 38.54% / 40.22%，后期出现噪声相关误差增加。

不能仅凭小投影残差进入科学验收，也不能保证增加迭代即可解决。下一步独立点源 Geant4 响应扫描，
核对矩阵空间变化；后续六路中的 Compton / 联合结果应与单光子分开评价。
完整诊断结果：`generated/FullData/closedloop_diagnostics.json`；正式 1000 次配置保持不变。
新增 `maty_point_smoke.sh`，对 218/440 的中心及 r=135、z=57 mm 边缘先做 1e4 光子短程核查。
短程作业 **15385867** 已启动；完整响应扫描 **15385868** 以 afterok:15385867 为依赖提交，
索引 400–561、162 个 worker、每个 1e7 光子、最多 40 并发，总计 1.62e9。
这是单视角响应核查，不能用来宣称 20 视角定位或 FWHM 已通过；PointImaging 仍单独待运行。

## 最新结果：两组闭环完成、正式包上传中（2026-09-25）

65114 连接在大文件下载结束后恢复。Noiseless / Poisson 两组闭环均已完成 1000 次 MLEM，
耗时 468.90 / 468.43 秒；4 个输出各 51240 点，20 个历史帧完整且有限非负，末帧与最终图一致。
包括 440、未校正 218（诊断对照）、校正 218、校正后通道和；本次不是六路 Compton 重建。

| 检查项 | 无噪声 | Poisson |
|---|---:|---:|
| 218 投影相对 L2 残差 | 0.20729% | 17.30481% |
| 440 投影相对 L2 残差 | 0.23619% | 27.03734% |
| 440 体积积分恢复 | 99.73711% | 99.72927% |
| 校正 218 体积积分恢复 | 99.87439% | 99.94891% |
| 440 图像体积加权相对 L2 | 41.97325% | 44.37918% |
| 校正 218 图像体积加权相对 L2 | 36.61979% | 40.21951% |

加权误差定义为 sqrt(sum(ΔV*(recon−truth)^2)/sum(ΔV*truth^2))，在完整计算支持域计算。
无噪声 440 的中心/中间/边缘误差为 36.65% / 45.88% / 45.97%；218 为 31.73% / 40.34% / 40.43%。
**计数链和总量基本闭合，空间质量仍未通过验收**；小投影残差不能证明图像准确。
需进一步检查热柱恢复、轴向定位及迭代演化，不应把偏差全部归因于 Poisson 噪声或用额外平滑掩盖。

本地证据：`generated/FullData/closedloop_summary.json`，完整结果归档 `generated/ClosedLoop_results.tar.gz`。
远端原始结果位于 `generated/ClosedLoop/{Noiseless,Poisson}` 下的 `.../Polar/`。
Factors_production.tar 已完整下载，SHA256 与下方远端记录完全一致；正在通过本机加密凭据连接
上传至 scxi717 指定实验目录的 `generated/Factors_production.tar.part`。上传脚本完成后还会进行远端
SHA256 校验，再解包为 Factors；**在看到 FACTORS_DEPLOYED_HASH_VERIFIED 前不能认定部署完成**。
临时执行脚本为本地 `generated/deploy_factors_once.py`，凭据不写入脚本或集群。

## 接续更新：闭环启动、跨节点测试通过（2026-09-25）

- 已补齐 65114 缺失的 `distributed` 源码依赖，入口 `--help` 导入检查通过。初次启动在导入阶段失败，无重建结果；日志保留为 `closedloop_*.initial_missing_module.log`。
- Contrast 1e9 的 Noiseless/Poisson 闭环分别在 GPU 0/1 启动，PID 3282393/3282396；均为完整 51240 点、20 视角、1000 次 MLEM、每 50 次保存。调用 `main_local_multi_energy_cntstat.py`，先重建 440，固定串窗预测后校正 218；本步仅单光子闭环，不是六路 Geant4 成像。
- 启动记录位于 65114 的 `generated/closedloop_launch.json`，输出 `generated/ClosedLoop/{Noiseless,Poisson}`，日志 `generated/closedloop_{Noiseless,Poisson}.log`。
- 启动后 65114 SSH 多次超时；下载仍间歇增长，因此不能据此判定服务器关机或任务失败。**闭环状态为已启动、完成情况待核查**。重连后先读日志和进程，禁止盲目重提。
- 正式 Factors 已打包为 `generated/Factors_production.tar`，远端 SHA256 为 `b7da8211a7af3c4421f72d9fc2400faaf7de2ecb86521397451f941b1804df2a`。本地同名文件正在下载，未经完整哈希验证不得使用；尚未部署到 scxi717。
- scxi717 指定工程所在存储可用约 346 GB。新增 `paracloud_multinode_smoke.sh` 已通过 `bash -n` 并运行作业 **1625684**：**COMPLETED / 0:0 / 36 秒**，节点 `wqd10nah09g3` 与 `wqd10nah09g4`，每节点 1 GPU；单光子、Compton、联合 MLEM 与串行参考全部一致。
- 跨节点进程退出阶段 stderr 有 TCPStore / RendezvousConnectionError 警告（连接被关闭）；数值断言已通过，Slurm 各步骤均 0:0。保留警告，后续长作业需继续检查通信稳定性。
- 跨节点日志已下载为 `generated/nccl_2node.1625684.out/.err`。这是小问题跨节点验证，仍不能代替完整矩阵/实际事件的峰值显存与六路测试。

## 当前阶段：正式校准与新 Sensi_d 完成（2026-09-25）

三套原始响应和未校准极坐标 Factors 全部完成，每套 2,151,260,160 bytes，
51240 点、40 层、10496 晶体，三路几何、有限非负检查通过。矩阵已无剩余生产任务。
完整中心 20 层回归：218/440 直接响应逐字节一致；440→218 相对 L2 差异
3.4740176163e−8、最大绝对差异 9.0949470177e−13，通过 1e−5 门槛。
证据保存在 generated 的 center_*fingerprints.json 和 cross_center_regression.json。

Geant4 扩充数组 **15384175** 和收集作业 **15384216** 均完成（0:0）。
四组分别为 calibration_218、calibration_440、sensitivity_440、sensitivity_validation_440；
每组 100 worker、实际 1e9 光子，合计 400 worker / 400 个不同种子 / 4e9 光子。
收集标记为 FOV120_ALL_FOUR_GROUPS_COLLECTED，全部输入输出及 PrimaryCount 检查通过。
完整归档 full_collected.tar.gz 的 SHA256：
`893dc589ce42c1e0aa9277e4050c73ee17798f1fc04b06e93bd1b5e2daf126bb`。
本地保存在 `experiments/FOV120/generated/FullData/`，并已校验后传入 65114 的 generated。

65114 已生成正式 `generated/Factors`，保留 `FactorsRaw`。四层系数依次对应
200 / 230 / 260 / 290 mm；校准计数相对标准误差范围为 0.0420%–0.21694%。

| 通道 | 四层校准系数 |
|---|---|
| 218→218 | 0.87701797, 0.87713908, 0.87478797, 0.86182702 |
| 440→440 | 0.88979251, 0.88854249, 0.88748235, 0.86980905 |
| 440→218 | 1.14430131, 1.18520845, 1.22655555, 1.25391590 |

报告已下载到 `generated/FullData/calibration_report.json`。这些是拟合系数，不能代替独立空间验证。
Contrast 的 1e9 光子极坐标真值和无噪声/Poisson 投影已在 65114 生成：
`generated/Truth_Contrast_1e9.npz`、`generated/GenProj_Contrast_1e9/`；尚未完成闭环重建。
投影按 20 视角分配总发射密度，Poisson 种子为 260924。q=8 真值体积积分相对解析值的误差为
218: −0.0173805%，440: −0.0191067%，报告为 `generated/FullData/Truth_Contrast_1e9.json`。

NCCL 测试 **1624002 已通过**：COMPLETED/0:0、1分57秒、两张 RTX5090；
单光子/Compton/联合 MLEM 与串行参考一致。跨节点及完整事件显存测试仍待完成。
新 Sensi_d 已在 65114 GPU 0（RTX A6000）完成并安装进正式 Factors，整体退出码为 0。
独立环境为 PyTorch 2.8.0+cu128，SciPy 1.18.1、Matplotlib 3.11.2；计算期间观测显存约 3.6 GiB。
计算组输入 2,041,975 行，接受 281,816 个事件；绝对归一化为 2.818160e−4，与有效事件数/1e9 一致。
独立组输入 2,039,981 行，接受 282,395 个事件；闭合比体积加权均值 **1.00203298**，
CV **0.148924%**，最小/中位/最大 **0.985523 / 1.002128 / 1.010986**。
Sensi_d 为 51240 个正有限 float32、204960 bytes；Sensi_d_provenance.json 绑定网格、矩阵和物理参数哈希。
本地证据位于 `generated/FullData/sensitivity_run_metadata.json`、
`sensitivity_independent_closure.json`、`Sensi_d_provenance.json`。
这只验证均匀源灵敏度闭合，不代表所有空间分布的重建无偏。

下一步：完成无噪声/Poisson 闭环，将完整 Factors 包部署到 scxi717 指定工程子目录；
随后完整事件预检、跨节点测试、轴向质控和真实体模成像。正式体模 1e9/1e10 六路重建尚未完成。

以下为历史阶段记录；若与本节冲突，以本节和最新远端证据为准。

## 最新接续修正（2026-09-24）

- Geant4 试验 15377351 的 40 个 worker 已全部成功，四组各 1e8、合计 4e8 初级光子；尚待收集和统计评估。
- NCCL 作业 1623854 实际 FAILED，退出 127:0：`module: command not found`，未执行任何数值测试，不能记为验证通过。
- 已在启动脚本显式初始化 `/etc/profile.d/modules.sh`，shell 语法与模块加载已验证。
- 按用户指定，将旧独立重建部署整体迁入主工程的 `experiments/FOV120_20260924/`，包括失败日志；旧同级目录不再作为工作目录。
- 新双 GPU 测试作业 **1624002** 已在该目录提交。下面的 17:03 状态为历史快照，以此修正和实时日志为准。
- 该子目录是独立代码根目录，内部仍有 `experiments/FOV120/`；运行时 `JSCC_REPO_ROOT` 应指向独立代码根，不能误指向它内部的配置目录。

## 1. 已冻结的研究定义

- 硬件不变：四层、10496 有效晶体，面向 FOV 约 270×135 mm；法向位置 200/230/260/290 mm。
- 物理目标 Φ300×120 mm；计算单元支持半径 153 mm。体模边界半径为 150 mm，两者不可混写。
- Cartesian 网格 51×51×40；极坐标 1281 点/层、51240 点；z 中心 −58.5:3:58.5 mm，20 视角。
- 三路响应：218→218、440→440、440→218；PE-v4 与密度基底 B=A·diag(ΔV) 不变。
- 真空中的双 γ 源代理，无人体材料衰减；不是完整衰变链输运。新体模产额 0.114/0.259，旧 0.261 保留原标识。
- 单光子先重建 440，再计算 440→218 预测串窗，作为固定加性背景重建 218；共六路输出。
- 440 Compton 保留共享 K*B、13% FWHM@511 keV、能量和 ≥350 keV，List 已展宽，不重复展宽。
- 正式图像 1000 次 MLEM、每 50 次保存；不以加大平滑或缩小显示范围掩盖边缘偏差。

## 2. 代码导航

| 目录/文件 | 职责与关键检查 |
|---|---|
| `experiments/FOV120/config.json` | 唯一实验几何、源、XCAT 裁剪配置 |
| `fov_config.py` | 三路坐标、晶体、体积、逆旋转、矩阵尺寸与 Sensi_d 来源校验 |
| 矩阵工程 `FileGenerater_3D_Unified/generate_jscc_218_440_response_params.m` | 按配置生成独立 `_pe_v4_FOV120` 参数 |
| 矩阵工程 `GenFactors/gen_factors.m`、`run_gen_response_factors.m`、`run_gen_fov120_factors.m` | 动态网格、单精度转换、170 mm 平移、保存未校准 Factors |
| `experiments/FOV120/run_matrices.py`、`finish_matrices.py` | 两个 PE/三个散射响应、输入/二进制来源记录、依赖完成后转换 |
| `compare_center_matrices.py`、`matrix_fingerprint.py` | 新旧公共 z 区域数值/完整字节回归、流式有限性检查 |
| `workflow.py` | 生成任务、独立 seed/worker、真实 PrimaryCount、合并前完整性/哈希检查 |
| `calibrate.py`、`run_sensitivity.py` | 新均匀源四层校准、新 440 Sensi_d 与独立闭合检查 |
| `Geant4Sim/Geant4Code/` | 当前源发射、晶体计数、事件 List 与 PrimaryCount C++ 实现 |
| `Geant4Sim/generate_xcat_ac225_psma_abdomen.py`、`validate_xcat_ac225_psma_abdomen.py` | 60/120 mm 可配置裁剪、混合尺寸源、真值/器官/边界验证 |
| `distributed/dual_energy_compton_python/` | 分布式六路重建、相同输入根目录的预检、单/多进程一致性 |
| `experiments/FOV120/imaging.py` | 体积重叠真值、无噪声/Poisson GenProj、全高切片/MIP/迭代与分区指标 |
| `reconstruct_point.py` | 点源定位、轴向 FWHM 与边界截断标记 |
| `maty_build_smoke.sh`、`maty_pilot.sh` | maty 集群编译/短程与 1e8 试验 |
| `paracloud_gpu_smoke.sh`、`reconstruct.sh` | scxi717 双 GPU 算法检查与正式多节点入口 |
| `remote_status.py`、`cluster_status.py`、`reconstruction_ssh.py` | 三处资源状态/访问工具；不包含凭据 |
| `tests/test_fov120.py`、`tests/test_distributed_dual_energy_compton.py` | 网格、源、收集、图像/指标及分布式回归 |

表中未加前缀的 Python/Shell 文件位于 `experiments/FOV120/`；“矩阵工程”指
`Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main/`。

## 3. 本地数据与可复现实验产物

| 位置 | 已有内容 | Git 策略 |
|---|---|---|
| `Factors/`、`CntStat/`、`List/` | 原 60 mm 矩阵、投影、事件；是回归基线 | 本地保留，生成数据不跟踪 |
| `experiments/FOV120/generated/baseline_snapshot.json` | 基线矩阵 SHA256 与几何快照 | 本地证据 |
| `generated/XCAT/` | 80×200×200 原生裁剪、40×100×100 真值、器官掩膜、22 个宏、预览/校验/积分结果 | 本地生成 |
| `generated/Simulation/` | 1762 个任务清单、宏、六个本地 smoke 输出 | 本地生成；集群 worker 结果在远端 |
| `generated/PointImaging/` | 162 个单能源位置 ×20 视角，共 3240 任务 | 仅生成，未执行 |
| `generated/factor_grid_test/` | MATLAB 20/40 层解析转换夹具、小样本 PE | 测试数据，不是生产 Factors |
| `generated/pe218_full_regression.json` | 完整 218 PE 中心 20 层 SHA256 一致性证据 | 本地证据 |
| `generated/*status.json`、`reconstruction_deployment.json` | 三处资源快照、部署根目录/作业号/包哈希 | 有时间戳的本地快照 |
| `generated/*bundle*`、`wheels/` | 上传包和依赖缓存 | 不入 Git |

`generated/` 在本表中均指 `experiments/FOV120/generated/`。不要把 `smoke/`、
解析测试夹具或合成绘图输入当成真实 FOV120 成像结果。大文件不随 Git 分发，
新 checkout 需按实验手册生成/从授权数据位置取回并核对哈希。

## 4. 现有科学结果：60 mm 与 120 mm 分开

### 60 mm 高计数基线已经完成

位置：`Results/Reconstruction/Distributed_JSCC_ComptonValidation_Geant4_1e10_Iter1000_1node8gpu/`。
`run_manifest.json` 记录 1e10 初级光子、8 GPU、1000 次迭代、每 50 次保存、
接受 **1,852,124** 个 Compton 事件。六路最终 `Image_*`、基础通道历史帧、
`PredictedCntStat_218_From440.float32` 和可视化均在该目录。

径向背景仍明显不均匀。该目录的 `compton_radial_bias_summary.json` 在第 1000 次给出：

| 通道 | 外环中位数/内区中位数 |
|---|---:|
| 440 单光子 | 0.39972 |
| 440 Compton | 1.24271 |
| 440 联合 | 0.80715 |

这些指标来自旧 Φ240×30 mm 对比度体模的规定背景 ROI（内区 r≤30 mm、外环
r=90–108 mm、选取 |z中心|≤13.5 mm），不是全 FOV 均匀源验收。旧 JSON 中可能
保留 F: 盘绝对路径；本地现目录为当前仓库的 D: 盘路径，不应据此重复创建数据。
旧中心 39 mm 的图像仅供历史比较，不能代表新 120 mm 全高质量。

### 120 mm 已验证内容

- 三套 MATLAB 参数成功生成；Detector.csv 与原基线一致。
- MATLAB 20/40 层解析测试通过，包含体积、插值、170 mm 平移和旋转互逆。
- 218 PE 全矩阵 4,794,163,200 bytes，有限且非负；中间 20 层与旧矩阵逐字节一致。
- XCAT 裁剪 `[755,835)`；双肾标签保留完整 XCAT 的 **81.91%**，两端仍接触边界；病灶轴向完整。
- 20 个混合能量宏和两个单能宏的反算活度最大误差约 1.67e−7。
- 极坐标真值按内部 3 mm/边缘 1.5 mm 的实际混合源积分，q=16 时总量误差约 **0.0054%**。
- 六次本地 Geant4 11.1.1 和六次集群 Geant4 11.1.0 万光子核查通过。
- 31 项相关 Python 回归此前通过；新增可选 NCCL 分支后 CPU/GLOO 三类重建一致性再次通过。

**尚无真实 FOV120 六路重建图像或有效 FOV 达标结论。**

## 5. 历史远端状态快照（2026-09-24 17:03，已被上方进度替代）

| 资源 | 实验根目录 | 本次状态 |
|---|---|---|
| 65114 | `/home/lipeize/JSCC_FOV120_20260924` | GPU 0 矩阵生产 PID 2800483；转换依赖 PID 2802514 |
| maty | `/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/FOV120_20260924` | 编译/六次 smoke 作业 15376312 已完成；试验数组 15377351 运行 |
| scxi717 | `/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/FOV120_20260924` | 双 GPU NCCL 测试 1623854 因 Priority 排队 |

矩阵：218/440 PE 分别用时 367.94/371.96 秒；218 散射最近日志进入
`scatterStart=5728` 分块，此值不是整个矩阵流程的完成百分比。440 散射、
440→218 散射、极坐标转换尚未完成。应检查输出大小/完成标记，不能仅凭进程消失判断成功。

Geant4：数组 15377351 共 40 worker、每个 1e7 光子；四组各 1e8，分别为
calibration_218、calibration_440、sensitivity_440、sensitivity_validation_440。
20 并发，当前前 20 个运行、后 20 个因 JobArrayTaskLimit 等待；尚无完成记录。
失败的初次编译 15376310 是系统 GCC 4.8.5 选择错误，已显式指定 GCC 12.2 修复。
`cnmix` 不接受本次 `--mem=4G`，已沿用原工程资源申报；首次提交失败未启动模拟。

重建：已核查 `gpugpu MaxTRESPJ=node=8`、账号 `GrpTRES=gres/gpu=100`，
5090 分区每节点 8 GPU。原 4×8 配置未超过已查到的额度；不要沿用登录横幅的通用 16 卡限制。
当前测试仅单节点双 GPU，尚未验证跨节点通信、完整 FOV120 显存峰值或六路生产数据。
其他既有训练作业不属于本实验，不取消、不修改。

## 6. 接续顺序与验收门槛

1. 检查三套矩阵与 raw Factors 完成；三路几何一致、40 层、51240 点、10496 晶体、20 视角。
2. 收集四组试验，严格核对 seed、PrimaryCount、哈希和失败/缺失任务；评估层计数及吞吐。
3. 校准和灵敏度数据分别累计到至少 1e9，不能混用拟合与独立验证；保留未校准矩阵。
4. 新 Sensi_d 用完整网格、同一筛选和 K*B 计算，核查绝对闭合；旧 Sensi_d 不补零复用。
5. 完成无噪声/Poisson 闭环、旧体模新旧支持域回归、轴向点源/均匀/对比度独立验证。
6. NCCL 小问题通过后再做完整 Factors/事件短程预检；跨节点验证及显存保留至少 20%。
7. 规则体模与 XCAT 先 1e9 再 1e10，总光子数为所有视角合计，正式采样率不降。
8. 六路成像全高展示；分 |z|≤30、30<|z|≤45、45<|z|≤60 mm 报告灵敏度、偏差、CV、CNR、CRC、定位和体积积分恢复。

基础设施完成、流程跑通、科学验收分别记录。组合图仍为 γ 通道复合图，不能直接解释为母体 ²²⁵Ac 活度。

## 7. 本次版本整理与发布范围

- 纳入整个工作区现有源代码、测试、文档改动，包括 XCAT 源生成、分布式六路算法、FOV120 流程及诊断/绘图工具。
- 数据目录忽略规则改为根目录限定，避免误忽略 `tools/diagnostics/list/` 源码。
- 57 个原已跟踪的 Factors、MATLAB fig、训练模型和 IDE 文件只从 Git 索引移除，约 58 MB 本地数据全部保留。
- 生成宏、矩阵、真值数组、重建结果、安装包、日志和本机凭据不纳入提交；保留可复现生成器与结果说明。
- 提交前相关 31 项测试通过，Git 空白检查通过；暂存区和原有 3 个未推送提交的对象经大小/凭据模式检查，无命中，最大被检文件约 0.68 MiB。
- 不重写历史：已经存在于历史提交的大数据并不会因本次取消跟踪而从历史中消失；本次保证新版本与待推送新增内容排除这些产物。
- 安全扫描仅覆盖已定义模式与大小，并另查字面量密码赋值；不声称可证明所有未知形式的秘密都不存在。
