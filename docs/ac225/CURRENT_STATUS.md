# 当前状态与阻点

快照时间：2026-10-19T13:10:49.801628+08:00，Asia/Shanghai。来自本机已登记/验收证据，不是实时Slurm查询。

| 研究 | 当前阶段 | 主要作业 |
|---|---|---|
| `nema-jscc-5e10` | formal-held-time | validation=1704559, verification=1704566, formal=1704572 |
| `nema-jscc-5e9-v5` | delivered-historical | formal=1669255 |
| `nema-ehe-g4-5e9` | delivered | formal=1679415 |
| `nema-ehe-poisson-5e9` | reference-only | formal=1680255 |
| `nema-ehe-g4-5e10` | delivered | formal=1681346 |
| `nema-ehe-poisson-5e10` | reference-only | formal=1683357 |
| `derenzo-jscc-218` | delivered | validation=1705505, validation_verify=1705507, formal=1705508, formal_verify=1706056 |
| `derenzo-jscc-440` | formal-registered | selection=1705543, selection_verify=1705544, validation=1713939, validation_verify=1713940, formal=1713943 |
| `derenzo-ehe-218` | delivered | validation=1705509, validation_verify=1705510, formal=1705512, formal_verify=1705779 |
| `derenzo-ehe-440` | delivered | validation=1705578, validation_verify=1705579, formal=1705580, formal_verify=1705853 |

## NEMA JSCC5e10

完整32GPU validation10任务1704559和独立验收1704566已成功退出，结果严格取回并生成本次authority。正式1704572仍暂挂、没有10000次正式结果。

实测吞吐外推三路10000次共54.87小时（准备只计一次），1.5倍余量需4969分钟。现有启动脚本的48小时总/24小时单阶段门槛不足，恢复控制器按门槛退出；这不是超算平台最大时限的声明。不得绕过门槛或把本次10次验证图当作10000次正式图。

[原始时长证据](../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/jscc_geant4_5e10_10000/production_5090_8x4_queue_measured_time_hold_20261019.json)；后续需独立执行时限或分段方案修复，保留原请求与成功输出。

## Derenzo

JSCC218正式10000次、EHE218/440正式各200次已验收并严格取回。JSCC440筛选已验收取回，完整验证1713939、验收1713940和暂挂正式1713943已登记；尚无formal_acceptance证明其正式完成。

此前只读SSH通道超时和过期PID身份问题已用独立执行控制恢复处理，Derenzo一次性流程继续，定时自动任务未恢复。活跃PID需按实际进程身份核对，不把旧登记状态当作进程存活证明。

## 已完成历史组与分析边界

NEMA JSCC5e9 v5及EHE实际5e9/5e10已完成；EHE前投影+Poisson是独立模型参考。历史六路/三路求和只读，当前只分析获授权分能量通道。图像噪声、小球、EHE物理响应HOLD及原生网格限制均保留。

新Derenzo图形已通过输入SHA、形状、迭代帧、真值体积和独立视觉QA；EHE明显边缘伪影如实保留，不等同分辨率通过。

具体数据、图形和验收入口见[实验表](EXPERIMENTS.md)。机器快照及证据SHA见[status_snapshot.json](status_snapshot.json)。
