# HE Derenzo：四路完成及PPT交付

4路800帧严格交付，数值/视觉与14页PowerPoint交付完成。复用HE NEMA响应，没有新响应发射；几何/网格/有限板近似仍未获物理认证。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | actual-Geant4 |
| 版本地位 | current |
| 计算 | 完成 |
| 严格交付 | 已严格取回 |
| 可视化 | 图形已复核 |
| 视角 | 20 |
| 迭代 / 保存 | 200 / 1 |
| 作业 | {"imaging": [16649504, 16649586, 16649668, 16650224]} |
| 状态依据日期 | 2026-11-17 |
| 矩阵 | HE reused 46d438089710a6ce; Cartesian4mm |
| 发射量级 | 5e9, 5e10 |

## 解读边界

- 数值或视觉QA不代表物理响应、分辨率或图像质量认证。

## 报告与原始说明

- [experiments/ELLIPSE500x300_H120/studies/opengate_he_derenzo_20261116/COMPLETED_20261117.md](../../../../experiments/ELLIPSE500x300_H120/studies/opengate_he_derenzo_20261116/COMPLETED_20261117.md)
- [experiments/ELLIPSE500x300_H120/studies/opengate_he_derenzo_20261116/README.md](../../../../experiments/ELLIPSE500x300_H120/studies/opengate_he_derenzo_20261116/README.md)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/Derenzo/opengate_he_derenzo_20261116/freeze.json](../../../../experiments/ELLIPSE500x300_H120/reports/Derenzo/opengate_he_derenzo_20261116/freeze.json)
- [experiments/ELLIPSE500x300_H120/reports/Derenzo/opengate_he_derenzo_20261116/controller_exit.json](../../../../experiments/ELLIPSE500x300_H120/reports/Derenzo/opengate_he_derenzo_20261116/controller_exit.json)
- [experiments/ELLIPSE500x300_H120/reports/Derenzo/opengate_he_derenzo_20261116/PPT_delivery_20261117.json](../../../../experiments/ELLIPSE500x300_H120/reports/Derenzo/opengate_he_derenzo_20261116/PPT_delivery_20261117.json)
- [experiments/ELLIPSE500x300_H120/reports/Derenzo/opengate_he_derenzo_20261116/5e9_218/strict_delivery.json](../../../../experiments/ELLIPSE500x300_H120/reports/Derenzo/opengate_he_derenzo_20261116/5e9_218/strict_delivery.json)
- [experiments/ELLIPSE500x300_H120/reports/Derenzo/opengate_he_derenzo_20261116/5e9_440/strict_delivery.json](../../../../experiments/ELLIPSE500x300_H120/reports/Derenzo/opengate_he_derenzo_20261116/5e9_440/strict_delivery.json)
- [experiments/ELLIPSE500x300_H120/reports/Derenzo/opengate_he_derenzo_20261116/5e10_218/strict_delivery.json](../../../../experiments/ELLIPSE500x300_H120/reports/Derenzo/opengate_he_derenzo_20261116/5e10_218/strict_delivery.json)
- [experiments/ELLIPSE500x300_H120/reports/Derenzo/opengate_he_derenzo_20261116/5e10_440/strict_delivery.json](../../../../experiments/ELLIPSE500x300_H120/reports/Derenzo/opengate_he_derenzo_20261116/5e10_440/strict_delivery.json)

## 源码入口

- [experiments/ELLIPSE500x300_H120/studies/opengate_he_derenzo_20261116/README.md](../../../../experiments/ELLIPSE500x300_H120/studies/opengate_he_derenzo_20261116/README.md)
- [experiments/ELLIPSE500x300_H120/studies/opengate_he_derenzo_20261116/workflow.py](../../../../experiments/ELLIPSE500x300_H120/studies/opengate_he_derenzo_20261116/workflow.py)
- [experiments/ELLIPSE500x300_H120/opengate_ehe_local_cpu_20261113.py](../../../../experiments/ELLIPSE500x300_H120/opengate_ehe_local_cpu_20261113.py)
- [experiments/ELLIPSE500x300_H120/studies/opengate_he_derenzo_20261116/visualize.py](../../../../experiments/ELLIPSE500x300_H120/studies/opengate_he_derenzo_20261116/visualize.py)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/opengate_he_derenzo_20261116](../../../../experiments/ELLIPSE500x300_H120/generated/opengate_he_derenzo_20261116)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`34a52489b6362ba0`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show opengate-he-derenzo`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
