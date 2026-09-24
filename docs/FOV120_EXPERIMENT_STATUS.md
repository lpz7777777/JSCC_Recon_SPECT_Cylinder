# ²²⁵Ac 218+440 keV：FOV120 实验目录与进度交接

核查日期：2026-09-24（北京时间约 17:03）。本页是已核查快照，作业状态会变化。
命令详见 [实验操作手册](../experiments/FOV120/README.md)，远程访问详见
[安全连接说明](REMOTE_COMPUTE_ACCESS.md)。原有 60 mm 数据不移动、不覆盖。

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

## 5. 远端目录、作业与状态快照

| 资源 | 实验根目录 | 本次状态 |
|---|---|---|
| 65114 | `/home/lipeize/JSCC_FOV120_20260924` | GPU 0 矩阵生产 PID 2800483；转换依赖 PID 2802514 |
| maty | `/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/FOV120_20260924` | 编译/六次 smoke 作业 15376312 已完成；试验数组 15377351 运行 |
| scxi717 | `/data/run01/scxi717/lpz/FOV120_20260924` | 双 GPU NCCL 测试 1623854 因 Priority 排队 |

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
