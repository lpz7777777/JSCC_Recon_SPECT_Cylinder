# ELLIPSE500x300_H120成像工程

当前研究覆盖NEMA、独立单能Derenzo、JSCC单光子/同能量联合、EHE真实输运及模型合成对照。完整FOV500×300×120mm、132040全点/78920活动单元，真实输运主要20角；60角只属于独立EHEPoisson对照。

2026-11-17导航：柱坐标为上述基线；[Cartesian研究](studies/nema5e9_cartesian_he_20261115/README.md)与[HE](studies/opengate_he_derenzo_20261116/README.md)使用独立4mm网格。最终只读证据：5090 1822462和4090 r2 1822534均触发原80%资源守卫失败，原控制器已按失败策略退出；实际阶段以[新证据状态](../../docs/ac225/CURRENT_STATUS.md)为准，旧研究README中的排队/未提交描述保留历史。[环境、各阶段运行边界与输入输出](../../docs/ac225/WORKFLOW.md)。

| 查什么 | 入口 |
|---|---|
| 当前计算与待处理项 | [状态](../../docs/ac225/CURRENT_STATUS.md) |
| 图像与研究结论 | [结果](../../Analysis_Plots/results/REPORT.html) · [结论](../../docs/ac225/CONCLUSIONS.md) |
| 每个研究的条件/代码/数据 | [分类研究页](../../docs/ac225/studies/README.md) |
| 科学源码路线 | [CODE_GUIDE](../../docs/ac225/CODE_GUIDE.md) |
| 带水NEMA衰减/散射修正原型 | [模块说明](ac225_corrections/README.md) · [新正式任务与证据](reports/NEMA_Body_H60/ac225_water_corrections_20261103/README.md) |
| 小型合同与证据 | [reports入口](reports/README.md) |
| 命名、剂量、体积及3D ROI | [科学口径](../../docs/ac225/SCIENCE_CONVENTIONS.md) |
| 操作约束 | [AGENTS](AGENTS.md) |

顶层代码、generated运行目录和远端发布路径保持稳定。图表中的JSCC只指同能量SC+Compton；纯Compton单列，218/440不求和。所有发射预算按合同定义，不当作225Ac衰变数。

旧README和完整人工授权时间线在[本次整理前备份](../../docs/organization/20261103/before/experiments/ELLIPSE500x300_H120/README.md)及[历史目录](history)；旧排队描述只作日期证据。当前源和矩阵身份始终由研究合同/冻结决定。
