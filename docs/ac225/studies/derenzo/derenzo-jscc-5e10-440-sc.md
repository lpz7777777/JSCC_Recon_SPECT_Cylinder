# Derenzo实际5e10：440 SC成功输出保留

父任务1752461失败但成功SC10000/200保留并于11月12日严格取回；联合失败9950单独登记，不重算成功SC。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | actual-Geant4-monoenergy |
| 版本地位 | current |
| 计算 | SC10000次已保存；父任务失败 |
| 严格交付 | 已严格取回 |
| 可视化 | see-meeting-report |
| 视角 | 20 |
| 迭代 / 保存 | 10000 / 50 |
| 作业 | {"parent_formal": 1752461} |
| 状态依据日期 | 2026-11-12 |
| 发射量级 | 5e10 |

## 报告与原始说明

- [experiments/ELLIPSE500x300_H120/reports/Derenzo/derenzo_jscc_mono_5e10_20261030/README.md](../../../../experiments/ELLIPSE500x300_H120/reports/Derenzo/derenzo_jscc_mono_5e10_20261030/README.md)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/Derenzo/derenzo_jscc_mono_5e10_20261030/failure_1752461_root_cause_20261102.json](../../../../experiments/ELLIPSE500x300_H120/reports/Derenzo/derenzo_jscc_mono_5e10_20261030/failure_1752461_root_cause_20261102.json)
- [experiments/ELLIPSE500x300_H120/reports/Derenzo/derenzo_jscc_mono_5e10_20261030/JSCC_440/formal_job.json](../../../../experiments/ELLIPSE500x300_H120/reports/Derenzo/derenzo_jscc_mono_5e10_20261030/JSCC_440/formal_job.json)
- [experiments/ELLIPSE500x300_H120/generated/derenzo_sc440_5e10_1752461_20261112/strict_transfer.json](../../../../experiments/ELLIPSE500x300_H120/generated/derenzo_sc440_5e10_1752461_20261112/strict_transfer.json)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show derenzo-jscc-5e10-440-sc`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
