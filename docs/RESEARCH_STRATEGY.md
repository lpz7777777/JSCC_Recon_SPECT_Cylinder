# JSCC 信息响应与系统优化的研究入口

研究选择记录日期：2026-10-12；导航核对日期：2026-10-19，Asia/Shanghai。

用户与导师讨论后，当前优先研究主线为：**异质事件的信息与响应 → 任务和不确定性驱动的系统优化**。当前已确定可以全部做模拟，不要求实物实验；主任务为整幅图像恢复，先研究 511 keV 平面四层 GAGG/W。相关光子及 MeV 拓展保留为后续方向。

## 跨工程研究文件

早期阶段记录：[固定 FOV/距离与多指标权重协议归档](../../20250307_JSCCGC_32x32x4_DiffEne/Analysis_Plots/research/design_optimization_20261012/archive/legacy_navigation/MULTIOBJECTIVE_PROTOCOL.md)和[平面参数化输运](../../20250307_JSCCGC_32x32x4_DiffEne/Analysis_Plots/research/design_optimization_20261012/studies/01_observations_tasks/fixed_task_transport_v1/REPORT.md)。后者记录当时160万历史与108项测试；这些历史技术检查不代表当前物理消费资格。最新方法、勘误和放行状态以[平面工程当前入口](../../20250307_JSCCGC_32x32x4_DiffEne/Analysis_Plots/research/design_optimization_20261012/README.md)为准，本三维工程没有因此启动新计算。

完整内容集中维护在同级平面 JSCC 工程，避免两份正文结论漂移。以下相对链接假设两个仓库仍位于同一父目录。

- [研究总入口](../../20250307_JSCCGC_32x32x4_DiffEne/Analysis_Plots/research/research_strategy_20261011/README.md)
- [约束设计与遗传算法优化计划](../../20250307_JSCCGC_32x32x4_DiffEne/Analysis_Plots/research/research_strategy_20261011/CONSTRAINED_GENETIC_DESIGN_PLAN_20261012.md)：用户已设为当前执行计划；固定参数、变量范围与偏好输入，快速任务评价和独立输运校正，先在平面结构族验证。
- [第一批实施与工程验证](../../20250307_JSCCGC_32x32x4_DiffEne/Analysis_Plots/research/design_optimization_20261012/README.md)：参数化几何、硬约束、任务指标接口和七候选 Geant4 核对已完成；尚无新成像排名，未运行遗传搜索。本三维工程未改代码或启动计算。
- [九步详细执行计划](../../20250307_JSCCGC_32x32x4_DiffEne/Analysis_Plots/research/research_strategy_20261011/DETAILED_EXECUTION_PLAN.md)
- [平面信息响应研究入口](../../20250307_JSCCGC_32x32x4_DiffEne/Analysis_Plots/research/heterogeneous_information_20261011/README.md)
- [最新五源混合恢复与误差分解](../../20250307_JSCCGC_32x32x4_DiffEne/Analysis_Plots/research/heterogeneous_information_20261011/mixture_recovery_v1/REPORT.md)：能量属性改善已知位置源强恢复，但持续响应偏差仍限制定量；不是二维整图或三维验收。
- [无序两击中响应首轮独立验证](../../20250307_JSCCGC_32x32x4_DiffEne/Analysis_Plots/research/heterogeneous_information_20261011/unordered_response_v2/REPORT.md)
- [响应修正与新一轮独立验证](../../20250307_JSCCGC_32x32x4_DiffEne/Analysis_Plots/research/heterogeneous_information_20261011/response_repair_v1/REPORT.md)：空间共享、中央预测、剩余校准/效率问题及 MC 信息解释。
- [优先主线与工作包](../../20250307_JSCCGC_32x32x4_DiffEne/Analysis_Plots/research/research_strategy_20261011/INFORMATION_RESPONSE_OPTIMIZATION_ROADMAP.md)
- [四方向完整调研](../../20250307_JSCCGC_32x32x4_DiffEne/Analysis_Plots/research/research_strategy_20261011/FOUR_DIRECTIONS_REVIEW.md)
- [项目证据与文献索引](../../20250307_JSCCGC_32x32x4_DiffEne/Analysis_Plots/research/research_strategy_20261011/EVIDENCE_AND_REFERENCES.md)

## 本工程的研究位置

本工程为主线提供三维、双能、极坐标密度、串窗及响应校准问题。近期最有价值的工作是明确观测语义与统计相关性，量化响应误差如何影响可用信息，并将结果连接到整幅图像恢复与相应局部诊断。

| 已有基础 | 后续问题 |
| --- | --- |
| B=A diag(ΔV) 密度基底 | 在一致单位和测度下计算任务信息与风险 |
| CntStat 与 List 来自同次输运 | 不重复计数的观测模型或明确相关性的估计 |
| 连续材料能量核 v5 的独立校准及配对图像 | 区分响应改善、噪声放大及不可辨识性 |
| 顺序推断、首次作用与材料能量尾诊断 | 可观测选择、软边缘化及虚拟标定约束 |
| 大规模流式重建与存储经验 | 在受控响应误差下实现可计算的信息分析及优化 |

已有结论参见[双能研究回顾](DUAL_ENERGY_RESEARCH_REVIEW.md)、[历史流程基准](DUAL_ENERGY_BASELINE.md)、[源测度](../GenProj/POLAR_SOURCE_MEASURE.md)、[事件顺序研究](../Auxiliary_Studies/EventOrderInference_Experiment/README.md)。

## 研究路线与执行状态

本文件记录研究选择与后续问题，不修改已有数据、科学核、发布、ROI 或输出合同。平面工程已完成首轮 6800 万历史及新增 4600 万历史的响应修正验证；空间共享和中央预测得到改善，能量校准及新位置效率仍未通过。结果不能直接移用于本三维几何；本三维工程没有因此启动模拟、训练、重建或自动任务。

具体执行仍以[实验最新约定](../experiments/ELLIPSE500x300_H120/AGENTS.md)及相应合同为准。历史 5e9 六路基准不代表当前生产输出；路线中的未来联合估计不恢复跨能量求和，也不变更当前 5e10 三路合同。218 与 440 的源场及单位分别保留，不将代理源当作真实级联或直接 Ac225 活度。

后续新增实验应建立独立目录，明确任务、事件选择、标定和评价条件，再记录实际结果。平面工程的信息距离和 C48 结果不能直接当作本三维几何的性能证明。
