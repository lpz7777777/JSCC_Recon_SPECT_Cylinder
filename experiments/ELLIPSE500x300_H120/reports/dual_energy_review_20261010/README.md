# 218与440 keV双能重建研究报告与整理记录

2026-10-10。这里统一整理已实际完成的五组NEMA H60主研究、18条成像路线及截至10月7日的相关方法研究。正式数据继续保留在原实验目录；本目录为只读衍生报告、摘要图表和存储治理证据。

- [18页研究报告](dual_energy_nema_research_report_20261010.pdf)：源与几何、观测/串扰、方法演变、完整轨迹、各球CRC/CNR、原生域CV/积分/轴外泄漏、图集、实际计算代价和解释边界。
- [五组结果目录](result_catalog.json)：正式作业号、原结果入口及验收SHA。
- [18路线末帧指标](endpoint_native_metrics.csv)与[全部适用球CNR峰值/末帧](sphere_cnr_peak_and_final.csv)。峰值只描述这一次噪声实现的保存帧，不构成最佳停止策略。
- [原始CSV、18份完整历史及8幅摘要图SHA](scientific_sources.json)。原报告中的[完整18类图集及曲线](../NEMA_Body_H60/ehe_forward_poisson_5e10_200/RESULTS.md)继续保留。

## 科学比较规则

EHE使用0–200次；JSCC使用0–10000次，不把同迭代/同图列当同收敛。使用本工程3mm三维真值及现有球ROI；原生统计覆盖120mm，摘要MIP为中央72mm。crop0、无平滑、固定发射源背景密度尺度，不拟合单图亮度。两能和为gamma密度和，不是母核活度。

独立Geant4观测与同响应矩阵自生Poisson数据分别陈述。原EHE物理偏差和逐bin统计不足仍保留，用户随后授权继续原方法已经完成。执行及严格取回通过不表示物理偏差被消除。材料、灵敏度、覆盖和计数差异不能全部归因于算法。本轮没有新增输运、响应或重建；compton-v5继续暂停。

## 本地代码清理

[计划](local_cleanup_plan.json)和[实际记录](local_cleanup_acceptance.json)保存每个路径、大小、SHA和恢复依据。

- 删除8份已经完成的一次性标题/文档收尾程序及过时的7月控制器；原字节可由提交176f403eb123408f58a74b9f88fb1a6e271a56f8恢复。其余生产代码没有调用这些文件。
- 19份generated根目录的历史进度/会计/文档辅助脚本先逐字节归档，再移除；保留件为`../../generated/historical_process_scripts_20261010.tar.gz`。归档仅用于代码追溯，不授权重启旧实验。
- 移除491份未跟踪Python缓存。实际移除518个文件、6386573字节，保留归档22677字节。缓存可在以后正常运行时重新生成。
- 工作流、求解器、矩阵/散射生产程序、通用科学绘图、验收器和科学诊断源码保留。历史README原字节另存，不把当时RUNNING状态作为当前状态。

`tools/research`保留本轮盘点、重复归档核对、带证据清理和报告生成工具，供复查整理依据；有清理完成记录时清理器拒绝重复运行。它们不是新的计算入口。

## scxi717空间调查与保留规则

[远端只读盘点](remote_storage_inventory.json)、[重复传输包核对](redundant_transfer_acceptance.json)和[大目录保留/回收计划](storage_retention_plan.csv)记录原命令、返回码、SHA及各目录处理条件。本工程旧511 keV List约57.23GiB、历史Factors约36.85GiB；另一个lpz旧工程约313.97GiB、zxc/SCSPECT_EXP约224.35GiB，是进一步项目级回收候选，尚未删除。du分配块与df容量口径不同。

近期三套响应、原12个GPU块、输运观测、完整迭代历史、冻结发布和验收证明属于保护资产。当前本地14.6GB Factors传输归档尚无同成员集的已展开本地副本，继续作为备份保留；下载缓存不包含已安装的conda环境。

实际远端清理前后容量以`remote_cleanup_acceptance.json`为准，整合说明以`cleanup_acceptance.json`为准；不能把候选大小写成已回收空间。

## 本轮实际回收

本地已清理8份过时的一次性程序、归档后移除19份历史辅助脚本及491份Python缓存，共518个文件；移除6386573字节，保留22677字节代码归档及Git恢复依据。远端已删除SHA核对一致且本地备份保留的14.60 GB重复传输包和4.07 GB pip下载缓存，共17.39 GiB逻辑文件；df可用空间实际由47.31升至64.59 GiB，使用率由95%降至92%。正式矩阵、原始观测、完整图像历史、冻结发布和conda环境均保留。

[远端删除与容量原始记录](remote_cleanup_acceptance.json)和[整合验收](cleanup_acceptance.json)保存字节口径与前后df；记录已通过，清理器拒绝重复执行。
