# 218与440 keV双能重建 近期结果总入口

2026-10-10最新规则：后续不再计算218与440的跨能量叠加图。球ROI采用中心距球面至少1.5 mm的体素；所有球统一使用Phantom内部完整背景体素，排除六球、肺插入物和边界。[新规则、重算指标及12条独立路线图表](reports/nema_interior_roi_20261010/README.md)为当前分析入口。

五组主要NEMA H60研究均已实际完成、独立验收和严格取回。当前没有待继续的本地推进器，compton-v5保持暂停。JSCC既有六路10000次作为历史实现回归基准；EHE实际输运与模型加噪声是四个独立研究组。

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

- [JSCC完整流程历史基准](../../docs/DUAL_ENERGY_BASELINE.md)：原六路入口及冻结发布只读保留；新任务应另行冻结四路输出合同，不沿用六路入口提交。
- [截至10月7日的方法研究回顾](../../docs/DUAL_ENERGY_RESEARCH_REVIEW.md)，包括密度基底、首散射、稳定几何、连续核、绑定/Huber及未完成精细场。
- [5e9两核2000对照](reports/NEMA_Body_H60/compton_energy_probability_v5_5e9/ACCEPTANCE.md)；[ideal1e9两核2000](reports/NEMA_Body_H60/compton_energy_probability_v5/ACCEPTANCE.md)不是纯剂量对照。
- [体模定义与几何真值](reports/NEMA_Body_H60/README.md)。

## 数据和代码保留规则

正式响应、观测、完整图像历史、冻结发布、科学诊断与验收证据保留。一次性文档/标题修补器及过时控制器已清出当前源码，恢复提交和逐文件SHA见清理记录；通用工作流、求解器、科学绘图和验证器保留。本次没有新增模拟、响应或重建，也不恢复已经停止的旧研究。

[本次整理前README原字节](HISTORY_20261010.md)、[10月7日快照](HISTORY_20261007.md)及[更早记录](HISTORY.md)包含当时RUNNING/PENDING语句，应按记录时间阅读。
