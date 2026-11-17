# 带水NEMA：独立供体固定散射模型v2

8路1600帧严格交付；独立剂量供体仍含相同体模布局先验。侧窗和440→218串窗偏差单列；并非物理认证。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | actual-Geant4 |
| 版本地位 | current |
| 计算 | 完成 |
| 严格交付 | 已严格取回 |
| 可视化 | see-dated-overview |
| 视角 | 20 |
| 迭代 / 保存 | 10000 / 50 |
| 作业 | {"440": 1783495, "218": 1783499} |
| 状态依据日期 | 2026-11-17 |
| 矩阵 | polar; water attenuation; independent donor fixed scatter |
| 发射量级 | 5e9, 5e10 |

## 解读边界

- 数值或视觉QA不代表物理响应、分辨率或图像质量认证。

## 报告与原始说明

- [Analysis_Plots/research/ac225_water_correction_overview_20261108/REPORT.html](../../../../Analysis_Plots/research/ac225_water_correction_overview_20261108/REPORT.html)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ac225_water_scatter_v2_20261107/README.md](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ac225_water_scatter_v2_20261107/README.md)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ac225_water_scatter_v2_20261107/freeze.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ac225_water_scatter_v2_20261107/freeze.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ac225_water_scatter_v2_20261107/strict_delivery.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ac225_water_scatter_v2_20261107/strict_delivery.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ac225_water_scatter_v2_20261107/frozen_projection_QA.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ac225_water_scatter_v2_20261107/frozen_projection_QA.json)

## 源码入口

- [experiments/ELLIPSE500x300_H120/ac225_water_scatter_delivery_20261107.py](../../../../experiments/ELLIPSE500x300_H120/ac225_water_scatter_delivery_20261107.py)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/ac225_water_scatter_v2_20261107](../../../../experiments/ELLIPSE500x300_H120/generated/ac225_water_scatter_v2_20261107)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show water-scatter-v2`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
