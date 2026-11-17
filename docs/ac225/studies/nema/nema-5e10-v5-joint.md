# NEMA实际5e10：新v5流程440 SC+Compton联合重建

1740189同440keV联合10000次完成；背景BV约35.94%，低于SC约40.50%，但13mm热球CNR下降；不说明环伪影成因已修正。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | actual-Geant4-dual-energy-proxy |
| 版本地位 | current |
| 计算 | 完成 |
| 严格交付 | 完整历史已严格交付 |
| 可视化 | 图形已复核 |
| 视角 | 20 |
| 迭代 / 保存 | 10000 / 50 |
| 作业 | {"formal": 1740189} |
| 状态依据日期 | 2026-10-29T12:42:52.493684+08:00 |

## 报告与原始说明

- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e10_v5_joint_20261026/README.md](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e10_v5_joint_20261026/README.md)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e10_v5_joint_20261026/formal_job.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e10_v5_joint_20261026/formal_job.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e10_v5_joint_20261026/latest_status.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e10_v5_joint_20261026/latest_status.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e10_v5_joint_20261026/scheduler_script_check.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e10_v5_joint_20261026/scheduler_script_check.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e10_v5_joint_20261026/status_20261027_135215/status.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e10_v5_joint_20261026/status_20261027_135215/status.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e10_v5_joint_20261026/status_20261028_132132/status.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e10_v5_joint_20261026/status_20261028_132132/status.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e10_v5_joint_20261026/status_20261029_124522/status.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e10_v5_joint_20261026/status_20261029_124522/status.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e10_v5_joint_20261026/result_delivery.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema_5e10_v5_joint_20261026/result_delivery.json)
- [experiments/ELLIPSE500x300_H120/generated/nema_joint5e10_formal_1740189_20261029/strict_transfer.json](../../../../experiments/ELLIPSE500x300_H120/generated/nema_joint5e10_formal_1740189_20261029/strict_transfer.json)
- [experiments/ELLIPSE500x300_H120/generated/nema_joint5e10_formal_1740189_20261029/numerical_check.json](../../../../experiments/ELLIPSE500x300_H120/generated/nema_joint5e10_formal_1740189_20261029/numerical_check.json)
- [Analysis_Plots/research/nema_joint5e10_1740189_20261029/data/scientific_qa.json](../../../../Analysis_Plots/research/nema_joint5e10_1740189_20261029/data/scientific_qa.json)
- [Analysis_Plots/research/nema_joint5e10_1740189_20261029/data/visual_qa.json](../../../../Analysis_Plots/research/nema_joint5e10_1740189_20261029/data/visual_qa.json)
- [Analysis_Plots/research/nema_joint5e10_1740189_20261029/data/independent_numeric_qa.json](../../../../Analysis_Plots/research/nema_joint5e10_1740189_20261029/data/independent_numeric_qa.json)

## 源码入口

- [experiments/ELLIPSE500x300_H120/nema_5e10_joint_workflow_20261026.py](../../../../experiments/ELLIPSE500x300_H120/nema_5e10_joint_workflow_20261026.py)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/nema_joint5e10_formal_1740189_20261029](../../../../experiments/ELLIPSE500x300_H120/generated/nema_joint5e10_formal_1740189_20261029)

## 图表入口

- [Analysis_Plots/research/nema_joint5e10_1740189_20261029/REPORT.html](../../../../Analysis_Plots/research/nema_joint5e10_1740189_20261029/REPORT.html)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show nema-5e10-v5-joint`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
