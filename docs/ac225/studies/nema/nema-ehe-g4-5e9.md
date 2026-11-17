# NEMA原EHE矩阵 5e9：实际Geant4

旧矩阵真实Geant4重建已交付；当前EHE实际比较优先使用11月dense65536独立重建。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | actual-Geant4-dual-energy-proxy |
| 版本地位 | historical-reference |
| 计算 | delivered |
| 严格交付 | 见原阶段证据 |
| 可视化 | 见原专题报告 |
| 视角 | 20 |
| 迭代 / 保存 | 200 / 见原合同 |
| 作业 | {"formal": 1679415} |
| 状态依据日期 | 2026-10-29T12:42:52.493684+08:00 |
| 矩阵 | original256 |

## 报告与原始说明

- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ehe_spect_5e9_200/RESULTS.md](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ehe_spect_5e9_200/RESULTS.md)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ehe_spect_5e9_200/formal_summary.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/ehe_spect_5e9_200/formal_summary.json)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/ehe_spect_5e9_200/results/formal](../../../../experiments/ELLIPSE500x300_H120/generated/ehe_spect_5e9_200/results/formal)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show nema-ehe-g4-5e9`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
