# 带水NEMA：四分支32路SC完整交付

严格交付32路6400帧及背景；原metadata的visual_review_pending=true，14路预览和后续专题不能代替全32路视觉验收。旧README的未交付描述被strict_delivery更新。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | actual-Geant4 |
| 版本地位 | current |
| 计算 | 完成 |
| 严格交付 | 已严格取回 |
| 可视化 | pending-full-32-review |
| 视角 | 20 |
| 迭代 / 保存 | 10000 / 50 |
| 作业 | {} |
| 状态依据日期 | 2026-11-17 |
| 矩阵 | polar; attenuation own S; none/AC/AC+rawTEW/AC+pool3 |
| 发射量级 | 5e9, 5e10 |

## 解读边界

- 数值或视觉QA不代表物理响应、分辨率或图像质量认证。

## 报告与原始说明

- [Analysis_Plots/research/ac225_water_corrections_20261103/REPORT.html](../../../../Analysis_Plots/research/ac225_water_corrections_20261103/REPORT.html)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ac225_water_corrections_20261103/README.md](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ac225_water_corrections_20261103/README.md)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ac225_water_corrections_20261103/freeze.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ac225_water_corrections_20261103/freeze.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ac225_water_corrections_20261103/formal_effective_routes_r5.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ac225_water_corrections_20261103/formal_effective_routes_r5.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ac225_water_corrections_20261103/strict_delivery.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ac225_water_corrections_20261103/strict_delivery.json)
- [Analysis_Plots/research/ac225_water_corrections_20261103/data/metadata.json](../../../../Analysis_Plots/research/ac225_water_corrections_20261103/data/metadata.json)

## 源码入口

- [experiments/ELLIPSE500x300_H120/ac225_corrections/README.md](../../../../experiments/ELLIPSE500x300_H120/ac225_corrections/README.md)
- [experiments/ELLIPSE500x300_H120/ac225_water_gpu_workflow_20261103.py](../../../../experiments/ELLIPSE500x300_H120/ac225_water_gpu_workflow_20261103.py)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/ac225_water_corrections_20261103/formal_delivery](../../../../experiments/ELLIPSE500x300_H120/generated/ac225_water_corrections_20261103/formal_delivery)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`79a78bf7c0a6a287`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show water-sc-first`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
