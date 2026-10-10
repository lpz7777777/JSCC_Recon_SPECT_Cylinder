# 218与440 keV双能重建 近期结果总入口

2026-10-10最新规则：后续不再计算218与440的跨能量叠加图。球ROI采用中心距球面至少1.5 mm的体素；所有球统一使用Phantom内部完整背景体素，排除六球、肺插入物和边界。[新规则、重算指标及12条独立路线图表](reports/nema_interior_roi_20261010/README.md)为当前分析入口。

最新新增授权实验：[JSCC实际5e10、三路10000次](reports/NEMA_Body_H60/jscc_geant4_5e10_10000/README.md)。启动检查15705111已实际COMPLETED/0:0；用户改为独立CPU调度后，未启动的18节点15705120已取消，唯一新输运数组15708389为1000worker、每worker1节点1核，可共享节点、最高并行1000，完成后统一收集。GPU优先8节点×4卡。正式仅保留218 corrected、440 single、440 Compton三路；原模型和MLEM保持，完整输入validation10通过后继续formal10000/save50。该实验尚未交付；新 `jscc-5e10` 每15分钟推进，旧compton-v5保持暂停。

2026-10-10 22:28上海：该实验完整实际5e10输运和严格输入早已交付；新增8×2张5090独立事件筛选1686507成功退出，用时8分48秒、选中4849087事件，实际筛选资源保留20%余量。匹配16rank的独立只读全行验收1686546成功退出，完整1003成员/全选中缓存/原日志严格SHA取回通过；原主1685272和4090 1686107保持不动。当前8×2完整Compton响应保守内存预算超过80%门槛，不直接提交16GPU正式重建；详细证据与后续阶段以新实验README为准。

17:22上海最新里程碑：15708389全部1000个子作业与步骤已实际COMPLETED/0:0，唯一收集15709588正在核对完整worker/源/计数/SHA。实际5e10输入闭合与严格取回仍待完成，尚未提交GPU重建。

此前五组主要NEMA H60研究均已实际完成、独立验收和严格取回，旧推进器已退出，compton-v5保持暂停。新增JSCC5e10独立研究仍在推进。JSCC既有六路10000次作为历史实现回归基准；EHE实际输运与模型加噪声是四个独立研究组。

- [历史研究报告 PDF：旧ROI与叠加图定义](reports/dual_energy_review_20261010/dual_energy_nema_research_report_20261010.pdf)
- [整理目录及存储清理说明](reports/dual_energy_review_20261010/README.md)
- [当前球ROI末帧指标](reports/nema_interior_roi_20261010/sphere_endpoint_metrics.csv)与[当前热球CNR峰值及末帧](reports/nema_interior_roi_20261010/hot_sphere_cnr_peak_and_final.csv)
- [历史18类完整图集和曲线](reports/NEMA_Body_H60/ehe_forward_poisson_5e10_200/RESULTS.md)，其中叠加图只读保留，不作为后续比较路线。

## 已完成的主结果

| 实验 | 正式作业 | 末迭代 |
|---|---:|---:|
| [JSCC实际5e9，六路10000](reports/NEMA_Body_H60/compton_energy_probability_v5_5e9_full10000/ACCEPTANCE.md) | 1669255 | 10000 |
| [EHE实际Geant4 5e9，三路200](reports/NEMA_Body_H60/ehe_spect_5e9_200/RESULTS.md) | 1679415 | 200 |
| [EHE期望5e9，矩阵前投影加Poisson，三路200](reports/NEMA_Body_H60/ehe_forward_poisson_5e9_200/README.md) | 1680255 | 200 |
| [EHE实际Geant4 5e10，三路200](reports/NEMA_Body_H60/ehe_spect_5e10_200/RESULTS.md) | 1681346 | 200 |
| [EHE期望5e10，矩阵前投影加Poisson，三路200](reports/NEMA_Body_H60/ehe_forward_poisson_5e10_200/RESULTS.md) | 1683357 | 200 |

比较保留EHE 0–200、JSCC 0–10000各自范围；同迭代或同图列不表示同收敛。真值为工程现有3mm三维球源，完整重建域为120mm，当前球/背景ROI位于H60体模内部，MIP图示中央72mm；crop0、无平滑、固定发射源密度尺度，无单图亮度拟合。旧报告的分数球ROI和局部背景CNR/CRC不能与本次重算指标混用。

EHE 5e10全4π、一事件一光子；输运15684979复用首次13个成功worker后补齐987个，总实际5e10。旧15683333失败及恢复证据保留。[源发射角核验](reports/NEMA_Body_H60/ehe_spect_5e10_200/SOURCE_ANGLE.md)说明宏的/xcat/angle只旋转源位置，不限半球，剂量倍数1。

原EHE物理审计仍有响应偏差和逐bin统计不足，[HOLD报告](reports/NEMA_Body_H60/ehe_spect_5e9_200/PHYSICAL_HOLD.md)保留。用户随后明确要求继续原方法，正式重建已完成；不得把执行验收或模型自生数据解释为该物理偏差已消失。

## 方法历史与固定基准

- [JSCC完整流程历史基准](../../docs/DUAL_ENERGY_BASELINE.md)：原六路入口及冻结发布只读保留；新任务应按用户选择的路线另行冻结输出合同，不沿用六路入口提交。
- [截至10月7日的方法研究回顾](../../docs/DUAL_ENERGY_RESEARCH_REVIEW.md)，包括密度基底、首散射、稳定几何、连续核、绑定/Huber及未完成精细场。
- [5e9两核2000对照](reports/NEMA_Body_H60/compton_energy_probability_v5_5e9/ACCEPTANCE.md)；[ideal1e9两核2000](reports/NEMA_Body_H60/compton_energy_probability_v5/ACCEPTANCE.md)不是纯剂量对照。
- [体模定义与几何真值](reports/NEMA_Body_H60/README.md)。

## 数据和代码保留规则

正式响应、观测、完整图像历史、冻结发布、科学诊断与验收证据保留。一次性文档/标题修补器及过时控制器已清出当前源码，恢复提交和逐文件SHA见清理记录；通用工作流、求解器、科学绘图和验证器保留。本次没有新增模拟、响应或重建，也不恢复已经停止的旧研究。

[本次整理前README原字节](HISTORY_20261010.md)、[10月7日快照](HISTORY_20261007.md)及[更早记录](HISTORY.md)包含当时RUNNING/PENDING语句，应按记录时间阅读。

2026-10-10 17:33上海：独立JSCC5e10的1000×50M实际4π输运已全部完成、唯一collection15709588成功退出并严格SHA取回；窗口218/440=123180986/53642922、原List54631328行。整批模拟32分36秒，不含排队和4分钟收集。新输入GPU同步和8×4事件筛选准备继续，三路完整输入验证/10000正式重建/QA尚未交付；见reports/NEMA_Body_H60/jscc_geant4_5e10_10000/transport_measurement.json。

2026-10-10 17:46上海：GPU4090的40CPU/4GPU启动请求被每卡6核配额拒绝，未运行科学作业。独立启动控制修复ce90d678eb32757f保持8节点×4GPU，改gpu_5090每节点32CPU、自动预期504000MiB；本地13项/实际Linux4项及真实scheduler test-only通过，原科学kernel、5e10输入和原失败证据保留。下一步唯一事件筛选，实际资源与科学验证仍待完整运行。


2026-10-10 21:47上海用户新增授权：保留5090主筛选1685272及4090试跑1686107，另提交独立8节点×2张5090完整输入事件筛选 **1686450**。每节点16CPU、16rank、按配额自动252000MiB/节点，无显式mem。原20view/54631328原List/全部132040点/10496bin/稳定float64全圆q≤3和矩阵科学字节保持；独立试跑发布1516a532adb59f94仅适配16进程身份、资源验收和最终全行汇总。当前PENDING(Resources)，原两项PENDING(Priority)；上限4小时不是ETA。正式重建仍未提交，按实际筛选事件数及后续完整响应内存验证决定，不凭筛选RSS放行Compton重建。


2026-10-10 22:06 JSCC5e10最新8×2筛选试跑：恢复作业1686507实际RUNNING；旧8×2 1686450和4090 1686107在UUID监控处失败、证据保留，主5090 1685272仍排队且未修改。恢复只修正设备身份字符串，原科学计算与阈值保持；实际16GPU启动身份通过，完整事件筛选/内存结论待退出验收。详情见reports/NEMA_Body_H60/jscc_geant4_5e10_10000/README.md及最新selection_5090_8x2_monitor_repair_job.json。
