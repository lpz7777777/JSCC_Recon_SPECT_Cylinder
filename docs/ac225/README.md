# 225Ac标签下的218/440 keV双能研究导航

本目录统一连接三维JSCC/EHE工程中的代码、体模、数据、验证、图像和历史证据。此处“225Ac”是研究标签：实际模拟的剂量语义是各合同定义的gamma初级数，未模拟完整核素衰变链或符合级联。

## 按任务查找

| 任务 | 入口 |
|---|---|
| 现在做到哪里 | [当前状态](CURRENT_STATUS.md) |
| 看所有主要实验与作业 | [实验表](EXPERIMENTS.md)及[catalog.json](catalog.json) |
| 看目录如何分层 | [目录图](DIRECTORY_MAP.md)及[directory_index.json](directory_index.json) |
| 找某个函数/工作流/验证器 | [源码索引](SOURCE_INDEX.md)及[source_index.json](source_index.json) |
| 分清218/440、SC、Compton、JSCC | [科学约定](SCIENCE_CONVENTIONS.md) |
| 找数据/恢复步骤 | [操作说明](OPERATIONS.md)、[数据规则](DATA_POLICY.md) |
| 看最新Derenzo结果 | [可视化目录](../../Analysis_Plots/README.md) |
| 理解方法发展和历史代价 | [研究回顾](../DUAL_ENERGY_RESEARCH_REVIEW.md) |

## 四条独立研究线

1. **NEMA双能JSCC**：既有5e9连续核v5完整10000次是历史数值基准；新增实际5e10采用单独三路合同，不能混作已交付结果。
2. **NEMA双能EHE**：实际Geant4 5e9/5e10与矩阵+Poisson 5e9/5e10分别登记。实际投影比较首选Geant4组；原响应物理限制保留。
3. **Derenzo独立单能**：218和440使用不同棒几何，各探头/能量各实际5e9；不是同一个双能源。218不注入另一个440体模的背景。
4. **物理与算法诊断**：连续核校准、首散射、稳定几何、尖峰消融、已停止精细响应场分别保留，不把未完成研究当成生产结果。

“联合”须看通道：440的SC+Compton是同能量联合观测；后续218+440跨能量求和图已停用。目录整理不恢复已暂停研究、不提交新科学计算。

[实验执行约定](../../experiments/ELLIPSE500x300_H120/AGENTS.md)控制操作；各研究自己的CONTRACT/验收控制科学身份。目录索引用于导航，不覆盖合同。
