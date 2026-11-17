# OpenGATE HE NEMA 5e9/5e10

HEGP孔规格+现有NaI，两剂量两能量800帧交付并完成数值/视觉QA。背景BV降低，部分小球恢复下降；不是完整商用相机或纯单因素比较。

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
| 作业 | {} |
| 状态依据日期 | 2026-11-17 |
| 矩阵 | HE MC-CDR; Cartesian4mm; depth/phase approximation |
| 发射量级 | 5e9, 5e10 |

## 解读边界

- 数值或视觉QA不代表物理响应、分辨率或图像质量认证。

## 报告与原始说明

- [Analysis_Plots/research/opengate_he_vs_ehe_nema_20261114/index.html](../../../../Analysis_Plots/research/opengate_he_vs_ehe_nema_20261114/index.html)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/opengate_he_nema_20261114/README.md](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/opengate_he_nema_20261114/README.md)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/opengate_he_nema_20261114/freeze.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/opengate_he_nema_20261114/freeze.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/opengate_he_nema_20261114/completed_execution_summary.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/opengate_he_nema_20261114/completed_execution_summary.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/opengate_he_nema_20261114/completed_visual_delivery.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/opengate_he_nema_20261114/completed_visual_delivery.json)

## 源码入口

- [experiments/ELLIPSE500x300_H120/opengate_he_nema_20261114.py](../../../../experiments/ELLIPSE500x300_H120/opengate_he_nema_20261114.py)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/opengate_he_nema_20261114](../../../../experiments/ELLIPSE500x300_H120/generated/opengate_he_nema_20261114)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`46d438089710a6ce`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show opengate-he-nema`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
