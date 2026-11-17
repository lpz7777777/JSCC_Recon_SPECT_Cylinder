# NEMA实际5e9：旧Compton核回退对照

近期NEMA实际投影配旧核/匹配S/OSEM8完成10000次，明显环纹和尖峰仍存在，回退未消除问题。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | actual-Geant4-reused-projection |
| 版本地位 | diagnostic |
| 计算 | 完成 |
| 严格交付 | 已严格取回 |
| 可视化 | 图形已复核 |
| 视角 | 20 |
| 迭代 / 保存 | 见原合同 / 见原合同 |
| 作业 | {"formal": 1739632} |
| 状态依据日期 | 2026-11-03T11:31:24.567092+08:00 |

## 解读边界

- 旧OSEM8每轮8次子集更新，不与MLEM同迭代等同收敛。
- 旧子集除完整S的约定使总量约新流程1/8；未额外乘8拟合。

## 报告与原始说明

- [experiments/ELLIPSE500x300_H120/README.md](../../../../experiments/ELLIPSE500x300_H120/README.md)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e9_prejune_compton_8x1_20261026/README.md](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e9_prejune_compton_8x1_20261026/README.md)
- [Analysis_Plots/research/nema_prejune_compton_1739632_20261026/data/metadata.json](../../../../Analysis_Plots/research/nema_prejune_compton_1739632_20261026/data/metadata.json)

## 源码入口

- [experiments/ELLIPSE500x300_H120/prejune_compton_workflow.py](../../../../experiments/ELLIPSE500x300_H120/prejune_compton_workflow.py)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show nema-prejune-compton`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
